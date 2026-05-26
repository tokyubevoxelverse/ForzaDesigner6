"""Tests for the subprocess kill ladder in gpu_gen_worker.

Validates the failure-mode fix the user hit on QUASAR: cancel-during-
GPU-run hung because terminate() didn't reap the CUDA-stuck process,
forcing a PC restart. The new ladder is graceful-signal → SIGTERM →
SIGKILL with timeouts at each stage so the user can ALWAYS reclaim
the subprocess from the EXE side, even if the runner is uncooperative.

Uses a FakeProc that simulates each level of stubbornness so we can
verify every escalation path without spawning real subprocesses.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

PySide6 = pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication   # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv)

from forza_abyss_painter.gui.gpu_gen_worker import (   # noqa: E402
    GpuGenWorker, _subprocess_creationflags,
    _CREATE_NEW_PROCESS_GROUP, _CREATE_NO_WINDOW,
)


# ===========================================================
# Spawn-flag verification: CREATE_NEW_PROCESS_GROUP must be set on
# Windows so we can send Ctrl-Break to the runner for graceful stop


def test_creationflags_include_new_process_group_on_windows(monkeypatch):
    """The kill ladder's graceful-stop stage sends CTRL_BREAK_EVENT,
    which only works if the child was created in a new process group.
    Without this flag, terminate() is the only kill option — which
    doesn't free CUDA cache, leading to the user's restart-the-PC
    failure mode."""
    monkeypatch.setattr(sys, "platform", "win32")
    flags = _subprocess_creationflags()
    assert flags & _CREATE_NEW_PROCESS_GROUP, (
        f"CREATE_NEW_PROCESS_GROUP missing from flags 0x{flags:08X} — "
        f"graceful-stop signal won't reach the runner subprocess"
    )
    assert flags & _CREATE_NO_WINDOW, (
        "CREATE_NO_WINDOW also required (don't leak cmd window)"
    )


def test_creationflags_zero_on_non_windows(monkeypatch):
    """Linux/macOS use start_new_session=True (handled elsewhere); the
    creationflags helper returns 0 because no Windows flag applies."""
    for plat in ("darwin", "linux"):
        monkeypatch.setattr(sys, "platform", plat)
        assert _subprocess_creationflags() == 0


# ===========================================================
# Kill-ladder behavior — uses a FakeProc to simulate each level


class FakeProc:
    """Mimics subprocess.Popen's surface area the kill ladder uses.

    Behavior controlled by `dies_after_stage`:
      'graceful' — exits cleanly on send_signal (CTRL_BREAK / SIGINT)
      'terminate' — survives the signal but exits on terminate()
      'kill' — survives signal + terminate, exits only on kill()
      'never' — survives everything (simulates a truly stuck process)
    """

    def __init__(self, dies_after_stage: str = "graceful") -> None:
        self.dies_after_stage = dies_after_stage
        self._alive = True
        self.signal_calls: list = []
        self.terminate_calls = 0
        self.kill_calls = 0
        self.stderr = None
        self.stdout = None

    def send_signal(self, sig: int) -> None:
        self.signal_calls.append(sig)
        if self.dies_after_stage == "graceful":
            self._alive = False

    def terminate(self) -> None:
        self.terminate_calls += 1
        if self.dies_after_stage == "terminate":
            self._alive = False

    def kill(self) -> None:
        self.kill_calls += 1
        if self.dies_after_stage in ("kill", "graceful", "terminate"):
            self._alive = False

    def poll(self) -> "int | None":
        return None if self._alive else 0

    def wait(self, timeout=None) -> int:
        if self._alive:
            raise subprocess.TimeoutExpired(cmd="fake", timeout=timeout or 0)
        return 0


def _build_worker_with_fake_proc(tmp_path, proc: FakeProc) -> GpuGenWorker:
    """Construct a worker pointed at an existing fake-python path so
    the missing_runtime guard doesn't fire, then inject the fake proc
    directly so we can drive the kill ladder."""
    py = tmp_path / "fake_python"
    py.write_text("#!/bin/sh\necho fake", encoding="utf-8")
    cfg = tmp_path / "cfg.json"
    cfg.write_text("{}", encoding="utf-8")
    worker = GpuGenWorker(py, cfg)
    worker._proc = proc
    return worker


def _wait_for_thread(timeout_s: float = 12.0) -> None:
    """Wait for any GpuGenWorker.kill_ladder thread to finish. The
    ladder uses real time.sleep / wait so tests need to drain it before
    asserting on end state."""
    import threading
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        ladder_threads = [t for t in threading.enumerate()
                          if t.name == "GpuGenWorker.kill_ladder"]
        if not ladder_threads:
            return
        time.sleep(0.05)
    raise RuntimeError("kill_ladder thread didn't finish in time")


def test_graceful_stop_exits_subprocess_without_terminate(tmp_path, monkeypatch):
    """Stage 1 wins: send_signal triggers a graceful exit; terminate()
    and kill() are never called. This is the happy path — the runner's
    SIGINT/CTRL_BREAK handler ran, freed the CUDA cache, exited."""
    # Speed up the test by patching the timeouts to short values.
    import forza_abyss_painter.gui.gpu_gen_worker as gw
    monkeypatch.setattr(gw, "_GRACEFUL_TIMEOUT_S", 0.5)
    monkeypatch.setattr(gw, "_FORCE_TIMEOUT_S", 0.5)

    proc = FakeProc(dies_after_stage="graceful")
    worker = _build_worker_with_fake_proc(tmp_path, proc)
    worker.cancel()
    _wait_for_thread()

    assert len(proc.signal_calls) == 1, "graceful signal not sent"
    assert proc.terminate_calls == 0, "terminate fired after graceful exit"
    assert proc.kill_calls == 0, "kill fired after graceful exit"


def test_escalates_to_terminate_when_signal_ignored(tmp_path, monkeypatch):
    """Stage 1 fails (proc ignores signal). Stage 2 reaps via
    terminate(). kill() never fires. This is the typical 'runner
    handler hung but TerminateProcess still works' case."""
    import forza_abyss_painter.gui.gpu_gen_worker as gw
    monkeypatch.setattr(gw, "_GRACEFUL_TIMEOUT_S", 0.3)
    monkeypatch.setattr(gw, "_FORCE_TIMEOUT_S", 0.5)

    proc = FakeProc(dies_after_stage="terminate")
    worker = _build_worker_with_fake_proc(tmp_path, proc)
    worker.cancel()
    _wait_for_thread()

    assert len(proc.signal_calls) == 1
    assert proc.terminate_calls == 1, "terminate not called when signal ignored"
    assert proc.kill_calls == 0, "kill fired even though terminate succeeded"


def test_escalates_to_kill_when_terminate_ignored(tmp_path, monkeypatch):
    """Stages 1 and 2 fail. Stage 3 reaps via kill(). This is the case
    that prevents the user's 'restart the PC' failure mode — even when
    a CUDA kernel hangs and ignores terminate, kill() reaches the
    OS-level reaper."""
    import forza_abyss_painter.gui.gpu_gen_worker as gw
    monkeypatch.setattr(gw, "_GRACEFUL_TIMEOUT_S", 0.3)
    monkeypatch.setattr(gw, "_FORCE_TIMEOUT_S", 0.3)

    proc = FakeProc(dies_after_stage="kill")
    worker = _build_worker_with_fake_proc(tmp_path, proc)
    worker.cancel()
    _wait_for_thread()

    assert len(proc.signal_calls) == 1
    assert proc.terminate_calls == 1
    assert proc.kill_calls == 1, "kill not called when terminate ignored"


def test_gives_up_after_kill_attempt_without_hanging(tmp_path, monkeypatch):
    """Worst case: process survives signal + terminate + kill (truly
    OS-level stuck in a kernel that the driver hasn't yielded yet).
    The ladder must NOT hang indefinitely — it gives up and returns
    so the GUI can move on. The user's recovery is to wait or restart
    the EXE; the PC stays usable."""
    import forza_abyss_painter.gui.gpu_gen_worker as gw
    monkeypatch.setattr(gw, "_GRACEFUL_TIMEOUT_S", 0.2)
    monkeypatch.setattr(gw, "_FORCE_TIMEOUT_S", 0.2)

    proc = FakeProc(dies_after_stage="never")
    worker = _build_worker_with_fake_proc(tmp_path, proc)
    start = time.monotonic()
    worker.cancel()
    _wait_for_thread(timeout_s=5.0)
    elapsed = time.monotonic() - start

    assert proc.kill_calls >= 1
    # Total elapsed should be roughly graceful + force + 2s final wait
    # = ~2.4s. If it's much longer, the ladder is hanging somewhere.
    assert elapsed < 5.0, f"kill ladder hung for {elapsed:.1f}s on stuck proc"


def test_cancel_is_idempotent(tmp_path, monkeypatch):
    """Clicking Cancel twice spawns two ladder threads. Both should
    complete cleanly without deadlocking each other on proc.wait."""
    import forza_abyss_painter.gui.gpu_gen_worker as gw
    monkeypatch.setattr(gw, "_GRACEFUL_TIMEOUT_S", 0.2)
    monkeypatch.setattr(gw, "_FORCE_TIMEOUT_S", 0.2)

    proc = FakeProc(dies_after_stage="terminate")
    worker = _build_worker_with_fake_proc(tmp_path, proc)
    worker.cancel()
    worker.cancel()   # second click while first ladder is still running
    _wait_for_thread()

    # First or second ladder reaps it; what we care about is that BOTH
    # threads finish without hanging the test.
    assert proc.terminate_calls >= 1


def test_cancel_when_proc_already_exited_is_noop(tmp_path):
    """If the process is already dead (poll() returns 0), cancel
    should be a no-op — don't spawn a ladder thread for nothing."""
    import threading
    proc = FakeProc(dies_after_stage="graceful")
    proc._alive = False   # already exited
    worker = _build_worker_with_fake_proc(tmp_path, proc)
    thread_count_before = len([t for t in threading.enumerate()
                               if t.name == "GpuGenWorker.kill_ladder"])
    worker.cancel()
    thread_count_after = len([t for t in threading.enumerate()
                              if t.name == "GpuGenWorker.kill_ladder"])
    assert thread_count_after == thread_count_before, (
        "cancel spawned a kill thread for an already-dead proc"
    )
    assert proc.signal_calls == [], "signal sent to already-dead proc"
