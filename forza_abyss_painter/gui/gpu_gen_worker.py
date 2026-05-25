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
import subprocess
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal, Slot


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
        # Pre-flight: python.exe must exist or the user is calling Generate
        # before the runtime install completed. The error here is
        # actionable — the dialog should pop the install prompt.
        if not self._py.exists():
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
            self._proc = self._popen_factory(
                [str(self._py), "-m",
                 "forza_abyss_painter.runtime.torch_runner",
                 "--config", str(self._config_path)],
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
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
        Sets the flag the reader loop checks, and SIGTERMs the subprocess
        so it can clean up. Safe to call multiple times."""
        self._cancelled = True
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.terminate()
            except OSError:
                # Process already gone, race with the reader loop ending.
                pass


def build_run_config(
    image_path: Path,
    output_json_path: Path,
    preset: dict,
    sticker_mode: bool = False,
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
    }
