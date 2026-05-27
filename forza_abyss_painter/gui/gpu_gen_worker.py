"""QObject worker that spawns torch_runner as a subprocess and streams
its stderr line-by-line as Qt signals.

Separated from generate_dialog.py so unit tests can exercise the IPC
protocol parsing + cancel path + error routing without spinning up the
full GenerateLocallyDialog GUI on every assertion.

## Flow

  1. EXE-side GenerateLocallyDialog builds a config JSON, writes to disk
  2. Constructs GpuGenWorker(config_path) + QThread, moves worker to thread
  3. Connects signals (started / progress / done / error) to dialog slots
  4. starts the thread → worker spawns the subprocess, reads stderr loop
  5. Subprocess emits JSON-per-line events → worker parses + emits Qt signals
  6. On 'done' event, output_path lands in config; dialog reads via load_json
  7. On 'error' event, dialog surfaces a modal + leaves itself open

## Cancel path

GpuGenWorker.cancel() is a Qt slot — call from the GUI thread to flip
an atomic flag + SIGTERM the subprocess. The reader loop checks the flag
between every readline, exits cleanly, the thread.quit() chain fires
finished signals in order. SIGTERM lets torch_runner do its own cleanup
(no orphaned temp files if it was mid-save).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal, Slot


# Windows subprocess flags.
#   CREATE_NO_WINDOW (0x08000000) — hide the cmd window of the embedded
#       python.exe so testers don't close it thinking it's stuck.
#   CREATE_NEW_PROCESS_GROUP (0x00000200) — give the child its own
#       process group, which is REQUIRED to send Ctrl-Break events to
#       the subprocess (Windows analog of SIGINT). Without this, the
#       parent EXE can't signal the runner to stop gracefully; cancel
#       has to use TerminateProcess (forceful) which doesn't give the
#       runner a chance to free CUDA cache → GPU stuck → PC restart.
_CREATE_NO_WINDOW = 0x08000000
_CREATE_NEW_PROCESS_GROUP = 0x00000200


def _subprocess_creationflags() -> int:
    """Windows-only flag suite for subprocess.Popen. Returns 0 on
    non-Windows since creationflags has no useful values elsewhere."""
    if sys.platform == "win32":
        return _CREATE_NO_WINDOW | _CREATE_NEW_PROCESS_GROUP
    return 0


# Cancel-ladder timeouts. The ladder is:
#   1. send graceful-stop signal (CTRL_BREAK_EVENT on Windows, SIGINT
#      on Unix). Runner's signal handler should empty_cache + exit.
#   2. wait GRACE_TIMEOUT_S seconds.
#   3. if still alive: terminate() (SIGTERM / TerminateProcess).
#   4. wait FORCE_TIMEOUT_S more seconds.
#   5. if STILL alive: kill() (SIGKILL on Unix; TerminateProcess on
#      Windows — Popen.kill() is the same as terminate() on Windows
#      but we call it anyway in case the implementation changes).
#   6. if alive after all that, log + give up. The OS owns the zombie
#      now and the user's only recovery is to wait for the kernel to
#      release or restart the EXE. Should be ~impossible to reach.
_GRACEFUL_TIMEOUT_S = 8.0    # generous — torch.cuda.empty_cache() can be slow
_FORCE_TIMEOUT_S = 4.0


class GpuGenWorker(QObject):
    """Streams subprocess IPC → Qt signals.

    Signals reflect the subprocess's IPC contract (see torch_runner.py
    module docstring). The GUI layer subscribes to the ones it cares
    about; unused ones are no-cost.
    """

    started = Signal(dict)        # cfg_summary (echoed config)
    progress = Signal(int, int)   # shape_count, total — fine-grained
    checkpoint = Signal(int, int) # shape_count, total — periodic
    done = Signal(str, int)       # output_path, shape_count
    error = Signal(str, str)      # stage, message
    finished = Signal()           # always — clean termination signal

    def __init__(
        self,
        embedded_python_exe: Path,
        config_path: Path,
        parent: QObject | None = None,
        *,
        _popen_factory=None,
    ) -> None:
        """`embedded_python_exe` is the interpreter to invoke (production
        passes torch_installer.embedded_python_exe(); tests can pass
        sys.executable when the test stub also lives in the regular
        Python env).

        `_popen_factory` is a DI hook for tests — a callable matching
        subprocess.Popen's signature that returns a Popen-like object
        with .stderr.readline, .wait(), .terminate(), .poll() methods.
        Production callers omit it (uses real subprocess.Popen).
        """
        super().__init__(parent)
        self._py = Path(embedded_python_exe)
        self._config_path = Path(config_path)
        self._popen_factory = _popen_factory or subprocess.Popen
        self._proc = None        # set in run() once Popen succeeds
        self._cancelled = False  # gets set from cancel() on the GUI thread

    @Slot()
    def run(self) -> None:
        """Worker entry point. Connected to QThread.started so it runs
        on the worker thread, not the GUI thread. Drives the subprocess
        to completion (or cancellation) and emits finished() last."""
        from forza_abyss_painter.runtime.gpu_logger import get_gpu_logger
        logger = get_gpu_logger()
        logger.log("gen_worker_run_started",
                   embedded_python=str(self._py),
                   config_path=str(self._config_path))

        # Pre-flight: python.exe must exist or the user is calling Generate
        # before the runtime install completed. The error here is
        # actionable — the dialog should pop the install prompt.
        if not self._py.exists():
            logger.log("gen_worker_missing_python_exe",
                       expected_path=str(self._py))
            self.error.emit(
                "missing_runtime",
                f"Embedded Python not found at {self._py}. "
                f"Run Tools → Install GPU runtime first.",
            )
            self.finished.emit()
            return

        # Spawn the subprocess. Pipe stderr (the IPC channel) and stdout
        # (just for diagnostics; not parsed). Line buffering = events
        # arrive as fast as the subprocess flushes them.
        try:
            cmd = [str(self._py), "-m",
                   "forza_abyss_painter.runtime.torch_runner",
                   "--config", str(self._config_path)]
            # Inherit parent env + set PyTorch allocator config to reduce
            # fragmentation overhead. Without expandable_segments, the
            # CUDA caching allocator can reserve 2-3× the actually-needed
            # peak (Cursor's QUASAR smoke: 47.5 GiB reported allocated
            # against ~17 GiB of real intermediates). expandable_segments
            # makes the allocator track per-stream allocations more
            # tightly + release back to the driver between phases, so
            # the chunked-K math actually maps to the observed peak.
            env = dict(os.environ)
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            logger.log("gen_worker_subprocess_spawn", cmd=cmd,
                       pytorch_alloc_conf=env["PYTORCH_CUDA_ALLOC_CONF"])
            self._proc = self._popen_factory(
                cmd,
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=env,
                # CREATE_NO_WINDOW on Windows — same fix as the install
                # phase. Without this, testers see a leaked cmd window
                # during GPU generation and close it, killing the
                # subprocess + the run.
                creationflags=_subprocess_creationflags(),
            )
        except OSError as exc:
            logger.log_exception("gen_worker_subprocess_spawn_failed", exc)
            self.error.emit("spawn", f"failed to spawn subprocess: {exc}")
            self.finished.emit()
            return

        # Handle cancel-before-spawn race: if cancel() was called between
        # GUI thread dispatch and worker thread arriving here, the flag
        # is already set + proc was None at cancel time. Terminate now
        # so wait() below doesn't block on an indefinitely-running
        # subprocess.
        if self._cancelled:
            try:
                self._proc.terminate()
            except OSError:
                pass

        # Stream events. iter(readline, '') terminates when the subprocess
        # closes stderr (i.e., exits). Each line is one JSON event.
        emitted_error = False
        emitted_done = False
        try:
            for raw in iter(self._proc.stderr.readline, ""):
                if self._cancelled:
                    # Cancel-during-stream: terminate the subprocess so
                    # wait() doesn't block. The flag set by cancel() may
                    # or may not have already triggered terminate (it
                    # races with the proc-is-not-None check there);
                    # calling terminate again is safe + idempotent.
                    try:
                        self._proc.terminate()
                    except OSError:
                        pass
                    break
                line = raw.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    # Non-protocol line on stderr (e.g., python warning).
                    # The runner's own output is all JSON; warnings from
                    # underlying libs may interleave. Skip them — they're
                    # informational, not contract.
                    continue
                if self._dispatch(event):
                    emitted_error = emitted_error or event.get("kind") == "error"
                    emitted_done = emitted_done or event.get("kind") == "done"
        except Exception as exc:  # pragma: no cover — defensive
            self.error.emit("read_loop", f"stderr read failed: {exc}")
            emitted_error = True

        # Drain + reap the subprocess. wait() ensures returncode is set.
        try:
            rc = self._proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            rc = self._proc.wait()
            if not emitted_error:
                self.error.emit("hang", "subprocess hung after stderr close")
                emitted_error = True

        # Post-process: surface cancel + unknown-failure cases. The
        # subprocess IPC contract says non-zero exit MUST be preceded
        # by an 'error' event — a violation here is a runner bug.
        if self._cancelled:
            self.error.emit("cancelled", "Generation cancelled by user")
        elif rc != 0 and not emitted_error:
            self.error.emit(
                "unknown",
                f"Subprocess exited {rc} without an error event "
                f"(IPC contract violation).",
            )
        elif rc == 0 and not emitted_done:
            self.error.emit(
                "incomplete",
                "Subprocess exited 0 but never emitted a 'done' event.",
            )

        logger.log("gen_worker_finished",
                   returncode=rc,
                   cancelled=self._cancelled,
                   emitted_done=emitted_done,
                   emitted_error=emitted_error)
        self.finished.emit()

    def _dispatch(self, event: dict) -> bool:
        """Route one parsed IPC event to the right Qt signal. Returns True
        if the event matched a known kind, False otherwise (caller can
        log/ignore unknowns)."""
        kind = event.get("kind")
        if kind == "started":
            self.started.emit(event.get("cfg_summary", {}))
        elif kind == "progress":
            self.progress.emit(
                int(event.get("shape_count", 0)),
                int(event.get("total", 0)),
            )
        elif kind == "checkpoint":
            self.checkpoint.emit(
                int(event.get("shape_count", 0)),
                int(event.get("total", 0)),
            )
        elif kind == "done":
            self.done.emit(
                str(event.get("output_path", "")),
                int(event.get("shape_count", 0)),
            )
        elif kind == "error":
            self.error.emit(
                str(event.get("stage", "")),
                str(event.get("message", "")),
            )
        else:
            return False
        return True

    @Slot()
    def cancel(self) -> None:
        """Called from the GUI thread when the user clicks Cancel/Abort.

        Robust three-stage kill ladder so the run can NEVER end up with
        an unkillable subprocess holding GPU memory (the 'had to restart
        the PC' failure mode):

          1. Send a GRACEFUL stop signal (CTRL_BREAK_EVENT on Windows,
             SIGINT on Unix). The runner's signal handler frees the
             CUDA cache before exiting.
          2. Wait up to GRACEFUL_TIMEOUT_S.
          3. If still alive: SIGTERM (subprocess.terminate). On Unix
             this is catchable; on Windows it's TerminateProcess.
          4. Wait up to FORCE_TIMEOUT_S.
          5. If STILL alive: SIGKILL (subprocess.kill). Last resort.
          6. Log and give up.

        Safe to call multiple times (the ladder restart if the user
        clicks Cancel twice, but each call still respects the same
        timeouts).

        Runs on a background thread spawned inline so the GUI doesn't
        block while waiting for the kernel to release.
        """
        self._cancelled = True
        if self._proc is None or self._proc.poll() is not None:
            return
        # Spin up the kill ladder on its own daemon thread so the GUI
        # stays responsive during the up-to-12s timeout window.
        import threading
        threading.Thread(
            target=self._kill_ladder, name="GpuGenWorker.kill_ladder",
            daemon=True,
        ).start()

    def _kill_ladder(self) -> None:
        """The actual escalation logic. Runs on a daemon thread because
        each stage involves wait()ing on the subprocess. The reader
        loop in run() finishes whenever the subprocess exits (or the
        cancelled flag short-circuits the loop), so this thread's only
        job is to ensure the subprocess actually goes away."""
        import signal
        proc = self._proc
        if proc is None:
            return

        # Stage 1: graceful signal.
        try:
            if sys.platform == "win32":
                # Requires CREATE_NEW_PROCESS_GROUP at spawn time.
                proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                proc.send_signal(signal.SIGINT)
        except (OSError, ValueError):
            pass

        # Wait for graceful exit.
        try:
            proc.wait(timeout=_GRACEFUL_TIMEOUT_S)
            return   # exited cleanly
        except subprocess.TimeoutExpired:
            pass

        # Stage 2: SIGTERM / TerminateProcess.
        try:
            proc.terminate()
        except OSError:
            pass
        try:
            proc.wait(timeout=_FORCE_TIMEOUT_S)
            return
        except subprocess.TimeoutExpired:
            pass

        # Stage 3: SIGKILL — last resort. After this, if the process
        # is STILL alive, we're stuck in a CUDA kernel that the OS
        # can't preempt. Log + give up. The user's recovery is to
        # wait for the kernel to finish on its own (usually seconds)
        # or restart the EXE (NOT the PC).
        try:
            proc.kill()
            proc.wait(timeout=2.0)
        except (OSError, subprocess.TimeoutExpired):
            pass


def build_run_config(
    image_path: Path,
    output_json_path: Path,
    preset: dict,
    sticker_mode: bool = False,
    vram_budget_gib: float = 0.0,
) -> dict:
    """Map a Generate dialog preset entry → torch_runner RunConfig dict.

    Pulled out into a module-level function so it's testable without
    a Qt event loop + so the dialog stays focused on UI flow.

    `preset` is one entry from generate_dialog.LOCAL_PRESETS — must
    carry num_shapes, max_resolution, random_samples, label. Any future
    knobs (edge_strength, posterize, polish_steps) get added to both
    the preset table and this builder simultaneously.
    """
    return {
        "image_path": str(image_path),
        "output_json_path": str(output_json_path),
        "num_shapes": int(preset["num_shapes"]),
        "max_resolution": int(preset["max_resolution"]),
        "random_samples": int(preset["random_samples"]),
        "sticker_mode": bool(sticker_mode),
        # Checkpoint cadence: 20 progress events per run gives the user
        # a smooth-ish progress bar without burdening the IPC channel.
        # Tunable per preset later if needed.
        "checkpoint_every": max(1, int(preset["num_shapes"]) // 20),
        "lock_alpha": True,   # hard system constraint per CLAUDE.md §3
        "preset_label": str(preset.get("label", "")),
        # Joint polish after greedy fill: N gradient/hill-climb steps
        # over all placed shapes. CPU presets have always polished; GPU
        # presets now match via the calibrated values in LOCAL_PRESETS.
        # 0 = no polish (back-compat for presets / fixtures that omit it).
        "joint_polish_steps": int(preset.get("joint_polish_steps", 0)),
        # Chunked-K mode: engine splits the K candidate batch into
        # VRAM-safe sub-batches when budget > 0. 0 = no budget = run
        # the full K in one pass (original behavior). Trade wall time
        # linearly for peak VRAM.
        "vram_budget_gib": float(vram_budget_gib),
        # Force the bbox-local code path for the EXE. The full_canvas
        # branch materializes (K, H, W) masks in rasterize_hard BEFORE
        # scoring kicks in — at K=8192 + 720px that's 16 GiB monolithic
        # and OOMs even with chunked scoring. bbox-local uses
        # crop_score_ellipse_batch_chunked which never allocates the
        # full (K, H, W) tensor. Cursor's QUASAR step-trace confirmed
        # bbox-local peaks at ~3.66 GiB per chunk on the same workload
        # that OOMs full_canvas at 47.5 GiB. Until strategy-B chunked
        # rasterize lands (#129), this is the only safe path.
        "bbox_local": True,
    }


def build_polish_config(
    source_image_path: Path,
    input_shapes_path: Path,
    output_path: Path,
    steps: int,
    lock_alpha: bool = True,
    sticker_mode: bool = False,
) -> dict:
    """Map PolishDialog values + paths → torch_runner RunConfig dict
    with mode='polish_only'. Sibling to build_run_config for #86.

    num_shapes / max_resolution / random_samples are intentionally
    omitted — RunConfig.from_dict treats them as optional under
    polish_only and ignores them at the runner level (canvas dims come
    from the loaded JSON's image_size).
    """
    return {
        "image_path": str(source_image_path),
        "output_json_path": str(output_path),
        "mode": "polish_only",
        "input_shapes_path": str(input_shapes_path),
        "polish_steps_override": int(steps),
        "lock_alpha": bool(lock_alpha),
        "sticker_mode": bool(sticker_mode),
        # Hold to the same defensive defaults the fresh builder applies.
        "bbox_local": True,
        "vram_budget_gib": 0.0,
        "preset_label": "polish_only",
    }
