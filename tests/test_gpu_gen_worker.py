"""Tests for forza_abyss_painter.gui.gpu_gen_worker — the QObject worker
that bridges torch_runner subprocess IPC ↔ Qt signals.

Uses a fake Popen factory to inject a controllable subprocess so we
can verify:
  - signal routing for each event kind (started, progress, checkpoint,
    done, error)
  - cancel path SIGTERMs the subprocess + emits 'cancelled' error
  - non-zero exit without an error event raises 'unknown' error
  - zero exit without a done event raises 'incomplete' error
  - missing embedded_python_exe short-circuits with 'missing_runtime'
  - build_run_config() preserves preset values + sets lock_alpha=True

No real subprocess.Popen is spawned by any test in this file — every
process boundary is faked so the tests run anywhere (no torch, no
embedded Python required).
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

PySide6 = pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication, QEventLoop, QTimer   # noqa: E402
from PySide6.QtWidgets import QApplication   # noqa: E402

# Use QApplication (not QCoreApplication) so other tests in the file
# that touch QWidget paths don't crash. Singleton-safe.
_app = QApplication.instance() or QApplication(sys.argv)

from forza_abyss_painter.gui.gpu_gen_worker import (   # noqa: E402
    GpuGenWorker, build_run_config,
)


# ====================================================================
# Fake Popen — drives the worker's stderr loop deterministically


class FakeProc:
    """Mimics subprocess.Popen's surface area the worker uses."""

    def __init__(
        self,
        stderr_lines: list[str],
        returncode: int = 0,
        wait_raises_timeout: bool = False,
        terminate_callback=None,
    ) -> None:
        # iter(readline, '') treats empty string as EOF, so terminate
        # the iterable with "" naturally.
        self.stderr = io.StringIO("\n".join(stderr_lines) + "\n" if stderr_lines else "")
        # iter(readline, '') stops on empty string; StringIO.readline
        # returns "" at EOF — perfect.
        self.stdout = io.StringIO("")
        self._rc = returncode
        self._wait_raises_timeout = wait_raises_timeout
        self._terminated = False
        self._killed = False
        self._terminate_callback = terminate_callback

    def wait(self, timeout=None):
        if self._wait_raises_timeout and not self._terminated:
            import subprocess
            raise subprocess.TimeoutExpired(cmd="fake", timeout=timeout or 0)
        return self._rc

    def poll(self):
        return self._rc if (self._terminated or self._killed) else None

    def terminate(self):
        self._terminated = True
        if self._terminate_callback:
            self._terminate_callback()

    def kill(self):
        self._killed = True


def _popen_factory_from_proc(proc: FakeProc):
    """Wrap a FakeProc in a callable that matches subprocess.Popen's
    signature. The worker calls _popen_factory([cmd...], stderr=...,
    stdout=..., text=..., bufsize=...) and expects a Popen-like back."""
    def _factory(cmd, **kwargs):
        return proc
    return _factory


@pytest.fixture
def _existing_python_exe(tmp_path):
    """Worker checks embedded_python_exe.exists() before spawning. Drop
    a stub file at the path so the gate passes."""
    py = tmp_path / "embedded_python_stub"
    py.write_text("#!/bin/sh\necho fake", encoding="utf-8")
    return py


def _drive_worker_to_completion(worker: GpuGenWorker) -> dict[str, list]:
    """Run the worker's run() method on the main thread (it doesn't need
    a real QThread for the IPC tests — we're testing the signal-routing
    logic, not Qt's thread affinity). Capture every emitted signal
    into a dict for assertions."""
    captured = {
        "started": [], "progress": [], "checkpoint": [],
        "done": [], "error": [], "finished": [],
    }
    worker.started.connect(lambda s: captured["started"].append(s))
    worker.progress.connect(lambda n, t: captured["progress"].append((n, t)))
    worker.checkpoint.connect(lambda n, t: captured["checkpoint"].append((n, t)))
    worker.done.connect(lambda p, n: captured["done"].append((p, n)))
    worker.error.connect(lambda s, m: captured["error"].append((s, m)))
    worker.finished.connect(lambda: captured["finished"].append(True))
    worker.run()
    return captured


# ====================================================================
# Happy path — events dispatched in order


def test_started_progress_done_signals_routed_in_order(_existing_python_exe, tmp_path):
    """Happy path: subprocess emits started → progress → done → exits 0.
    Worker emits each signal in the same order; finished fires last."""
    cfg = tmp_path / "cfg.json"
    cfg.write_text("{}", encoding="utf-8")
    proc = FakeProc(stderr_lines=[
        json.dumps({"kind": "started", "cfg_summary": {"num_shapes": 100}}),
        json.dumps({"kind": "progress", "shape_count": 10, "total": 100}),
        json.dumps({"kind": "progress", "shape_count": 50, "total": 100}),
        json.dumps({"kind": "done", "output_path": str(tmp_path / "out.json"),
                    "shape_count": 100}),
    ], returncode=0)
    worker = GpuGenWorker(_existing_python_exe, cfg,
                          _popen_factory=_popen_factory_from_proc(proc))
    captured = _drive_worker_to_completion(worker)

    assert captured["started"] == [{"num_shapes": 100}]
    assert captured["progress"] == [(10, 100), (50, 100)]
    assert captured["done"] == [(str(tmp_path / "out.json"), 100)]
    assert captured["error"] == []
    assert captured["finished"] == [True]


def test_checkpoint_events_routed_to_checkpoint_signal(_existing_python_exe, tmp_path):
    """Checkpoint events go on the checkpoint signal — the dialog wires
    BOTH progress and checkpoint to its progress-bar slot, but downstream
    consumers (like a future preview-rendering hook) need the distinction."""
    cfg = tmp_path / "cfg.json"
    cfg.write_text("{}", encoding="utf-8")
    proc = FakeProc(stderr_lines=[
        json.dumps({"kind": "started", "cfg_summary": {}}),
        json.dumps({"kind": "checkpoint", "shape_count": 50, "total": 200}),
        json.dumps({"kind": "done", "output_path": "x.json", "shape_count": 200}),
    ], returncode=0)
    worker = GpuGenWorker(_existing_python_exe, cfg,
                          _popen_factory=_popen_factory_from_proc(proc))
    captured = _drive_worker_to_completion(worker)
    assert captured["checkpoint"] == [(50, 200)]
    assert captured["progress"] == []   # routed to its own signal


def test_error_event_routed_to_error_signal(_existing_python_exe, tmp_path):
    """Subprocess emits an 'error' event → worker emits error signal
    with (stage, message). Non-zero exit after this should NOT trigger
    the 'unknown' fallback — the runner already named the failure."""
    cfg = tmp_path / "cfg.json"
    cfg.write_text("{}", encoding="utf-8")
    proc = FakeProc(stderr_lines=[
        json.dumps({"kind": "started", "cfg_summary": {}}),
        json.dumps({"kind": "error", "stage": "engine_run",
                    "message": "CUDA OOM: drop RANDOM_SAMPLES to 6144"}),
    ], returncode=1)
    worker = GpuGenWorker(_existing_python_exe, cfg,
                          _popen_factory=_popen_factory_from_proc(proc))
    captured = _drive_worker_to_completion(worker)
    # ONE error event (from the protocol), not a duplicate 'unknown'.
    assert len(captured["error"]) == 1
    stage, msg = captured["error"][0]
    assert stage == "engine_run"
    assert "RANDOM_SAMPLES" in msg


# ====================================================================
# Error paths


def test_missing_python_exe_short_circuits_with_missing_runtime_error(
    tmp_path,
):
    """If embedded_python_exe doesn't exist on disk (user deleted runtime
    dir, partial install), worker emits error(missing_runtime, ...) +
    finished WITHOUT spawning a subprocess. The error message tells the
    user to install the runtime first."""
    cfg = tmp_path / "cfg.json"
    cfg.write_text("{}", encoding="utf-8")
    no_py = tmp_path / "does-not-exist"
    factory_called = [0]
    def _track(*a, **kw):
        factory_called[0] += 1
        return FakeProc([])
    worker = GpuGenWorker(no_py, cfg, _popen_factory=_track)
    captured = _drive_worker_to_completion(worker)
    assert factory_called[0] == 0, "Popen factory was called despite missing exe"
    assert len(captured["error"]) == 1
    stage, msg = captured["error"][0]
    assert stage == "missing_runtime"
    assert "Install GPU runtime" in msg
    assert captured["finished"] == [True]


def test_zero_exit_without_done_event_emits_incomplete_error(
    _existing_python_exe, tmp_path,
):
    """If the subprocess exits 0 but never emitted 'done', that's a
    runner-side bug — the IPC contract requires 'done' precede a
    successful exit. Worker surfaces this so it doesn't silently look
    like a success."""
    cfg = tmp_path / "cfg.json"
    cfg.write_text("{}", encoding="utf-8")
    proc = FakeProc(stderr_lines=[
        json.dumps({"kind": "started", "cfg_summary": {}}),
    ], returncode=0)
    worker = GpuGenWorker(_existing_python_exe, cfg,
                          _popen_factory=_popen_factory_from_proc(proc))
    captured = _drive_worker_to_completion(worker)
    assert captured["done"] == []
    assert any(stage == "incomplete" for stage, _ in captured["error"]), (
        f"expected 'incomplete' error; got {captured['error']}"
    )


def test_non_zero_exit_without_error_event_emits_unknown_error(
    _existing_python_exe, tmp_path,
):
    """Non-zero exit WITHOUT an error event = IPC contract violation by
    the runner. Worker emits a synthetic 'unknown' error so the GUI
    can still surface SOMETHING to the user."""
    cfg = tmp_path / "cfg.json"
    cfg.write_text("{}", encoding="utf-8")
    proc = FakeProc(stderr_lines=[
        json.dumps({"kind": "started", "cfg_summary": {}}),
    ], returncode=42)
    worker = GpuGenWorker(_existing_python_exe, cfg,
                          _popen_factory=_popen_factory_from_proc(proc))
    captured = _drive_worker_to_completion(worker)
    assert any(stage == "unknown" for stage, _ in captured["error"])
    # The exit code is in the message so a tester can grep logs.
    assert any("42" in msg for _, msg in captured["error"])


def test_non_json_stderr_lines_are_skipped_not_crashed(
    _existing_python_exe, tmp_path,
):
    """The runner's own output is all JSON, but underlying libs (torch,
    CUDA, Pillow) can spit warnings onto stderr that interleave with
    the IPC stream. Worker must skip non-JSON lines silently, not crash."""
    cfg = tmp_path / "cfg.json"
    cfg.write_text("{}", encoding="utf-8")
    proc = FakeProc(stderr_lines=[
        json.dumps({"kind": "started", "cfg_summary": {}}),
        "UserWarning: deprecated API",   # non-JSON
        "  at line 42 in torch/foo.py",  # non-JSON
        json.dumps({"kind": "progress", "shape_count": 5, "total": 10}),
        json.dumps({"kind": "done", "output_path": "x.json", "shape_count": 10}),
    ], returncode=0)
    worker = GpuGenWorker(_existing_python_exe, cfg,
                          _popen_factory=_popen_factory_from_proc(proc))
    captured = _drive_worker_to_completion(worker)
    # All real events still landed.
    assert captured["progress"] == [(5, 10)]
    assert len(captured["done"]) == 1
    assert captured["error"] == []


# ====================================================================
# Cancel


def test_cancel_terminates_subprocess_and_emits_cancelled_error(
    _existing_python_exe, tmp_path,
):
    """Pre-cancel the worker so it sees the flag set as soon as it
    enters the read loop. Subprocess gets terminate()'d; an 'error'
    event with stage='cancelled' is emitted."""
    cfg = tmp_path / "cfg.json"
    cfg.write_text("{}", encoding="utf-8")
    # FakeProc that has events queued, but the worker's _cancelled flag
    # will trip on the first readline so we never get past 'started'.
    proc = FakeProc(stderr_lines=[
        json.dumps({"kind": "started", "cfg_summary": {}}),
        json.dumps({"kind": "progress", "shape_count": 1, "total": 100}),
    ], returncode=0)
    worker = GpuGenWorker(_existing_python_exe, cfg,
                          _popen_factory=_popen_factory_from_proc(proc))
    # Cancel BEFORE run() — the next readline check will fire and exit.
    # In production, cancel() is called from the GUI thread while the
    # worker thread is mid-readline; we approximate that ordering by
    # cancelling before run.
    worker.cancel()
    captured = _drive_worker_to_completion(worker)
    assert any(stage == "cancelled" for stage, _ in captured["error"]), (
        f"expected cancelled error; got {captured['error']}"
    )
    assert proc._terminated, "subprocess was not terminated on cancel"


# ====================================================================
# build_run_config


def test_build_run_config_carries_preset_values(tmp_path):
    """The config dict the dialog writes must include every field the
    runner's RunConfig.from_dict requires. If a key drifts off, the
    runner exits 2 (config_load) and the user blames the GUI."""
    preset = {
        "label": "Headshot 700",
        "num_shapes": 700,
        "max_resolution": 600,
        "random_samples": 6144,
        "est_peak_vram_gib": 3.5,
        "description": "Portraits",
    }
    cfg = build_run_config(
        image_path=tmp_path / "src.png",
        output_json_path=tmp_path / "out.json",
        preset=preset,
        sticker_mode=False,
    )
    assert cfg["num_shapes"] == 700
    assert cfg["max_resolution"] == 600
    assert cfg["random_samples"] == 6144
    assert cfg["preset_label"] == "Headshot 700"
    assert cfg["image_path"] == str(tmp_path / "src.png")
    assert cfg["output_json_path"] == str(tmp_path / "out.json")


def test_build_run_config_forces_lock_alpha_true(tmp_path):
    """lock_alpha=True is a hard system constraint (CLAUDE.md §3 +
    GPUConfig.lock_alpha docstring). The dialog config builder MUST
    set it True regardless of preset state — even if a future preset
    accidentally carries lock_alpha=False, this overrides."""
    preset = {
        "label": "X", "num_shapes": 10, "max_resolution": 240,
        "random_samples": 64, "lock_alpha": False,   # decoy
    }
    cfg = build_run_config(tmp_path / "x.png", tmp_path / "x.json", preset)
    assert cfg["lock_alpha"] is True


def test_build_run_config_picks_reasonable_checkpoint_cadence(tmp_path):
    """20 checkpoint events per run gives a smooth progress bar without
    flooding the IPC channel. checkpoint_every = num_shapes // 20."""
    for num_shapes, expected_every in [(20, 1), (100, 5), (1000, 50), (3000, 150)]:
        preset = {"label": "X", "num_shapes": num_shapes,
                  "max_resolution": 240, "random_samples": 64}
        cfg = build_run_config(tmp_path / "x.png", tmp_path / "x.json", preset)
        assert cfg["checkpoint_every"] == expected_every, (
            f"num_shapes={num_shapes}: expected every={expected_every}, "
            f"got {cfg['checkpoint_every']}"
        )


def test_build_run_config_clamps_checkpoint_every_to_at_least_1(tmp_path):
    """num_shapes < 20 → integer division gives 0, which the runner
    treats as 'no checkpoint events ever'. Floor at 1 so even tiny
    test runs emit at least one progress point."""
    preset = {"label": "X", "num_shapes": 5, "max_resolution": 240,
              "random_samples": 64}
    cfg = build_run_config(tmp_path / "x.png", tmp_path / "x.json", preset)
    assert cfg["checkpoint_every"] >= 1
