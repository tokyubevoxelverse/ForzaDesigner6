"""Tests for forza_abyss_painter.gui.gpu_install_worker — the QObject
worker that wraps torch_installer.install_runtime() with Qt signal
emission.

Uses a fake install_fn (DI hook on the worker constructor) so tests can
drive the worker through every outcome — happy path, per-stage
InstallError, generic Exception — without actually downloading torch.
"""
from __future__ import annotations

import os
import sys

import pytest

PySide6 = pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication   # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv)

from forza_abyss_painter.gui.gpu_install_worker import GpuInstallWorker   # noqa: E402
from forza_abyss_painter.runtime.torch_installer import (   # noqa: E402
    InstallError, RuntimeInfo,
)


# ====================================================================
# Helpers


def _drive_worker(worker: GpuInstallWorker) -> dict[str, list]:
    """Run the worker on the main thread + capture every emitted signal.
    We don't need a real QThread for IPC-routing tests — that adds Qt
    affinity complexity without changing what we're verifying."""
    captured = {"progress": [], "done": [], "error": [], "finished": []}
    worker.progress.connect(lambda p, s: captured["progress"].append((p, s)))
    worker.done.connect(lambda info: captured["done"].append(info))
    worker.error.connect(lambda st, m: captured["error"].append((st, m)))
    worker.finished.connect(lambda: captured["finished"].append(True))
    worker.run()
    return captured


# ====================================================================
# Happy path


def test_happy_path_emits_progress_then_done_then_finished():
    """install_runtime succeeds → worker emits progress events for each
    phase callback, then done(RuntimeInfo dict), then finished. The
    GUI dialog uses this exact order to update the progress bar then
    auto-accept on done."""
    def _fake_install_fn(progress_cb=None, **kw):
        if progress_cb:
            progress_cb(0, "Starting")
            progress_cb(50, "Pip installing")
            progress_cb(95, "Writing marker")
        return RuntimeInfo(
            python_version="3.11.9",
            torch_version="2.4.1+cu121",
            cuda_available=True,
            cuda_device_name="NVIDIA RTX 4090",
            installed_at_utc="2026-05-25T12:00:00Z",
        )
    worker = GpuInstallWorker(_install_fn=_fake_install_fn)
    captured = _drive_worker(worker)

    assert captured["progress"] == [
        (0, "Starting"), (50, "Pip installing"), (95, "Writing marker"),
    ]
    assert len(captured["done"]) == 1
    info = captured["done"][0]
    assert info["torch_version"] == "2.4.1+cu121"
    assert info["cuda_available"] is True
    assert info["cuda_device_name"] == "NVIDIA RTX 4090"
    assert captured["error"] == []
    assert captured["finished"] == [True]


def test_progress_callback_is_passed_to_install_runtime():
    """The worker must hand its own progress callback into
    install_runtime so per-phase progress reaches the GUI. If this
    wiring breaks, the progress bar sits at 0 for 15 minutes."""
    captured_cb = []
    def _fake_install_fn(progress_cb=None, **kw):
        captured_cb.append(progress_cb)
        return RuntimeInfo(
            python_version="3.11.9", torch_version="2.4.1",
            cuda_available=True, cuda_device_name="X",
            installed_at_utc="2026-05-25T12:00:00Z",
        )
    worker = GpuInstallWorker(_install_fn=_fake_install_fn)
    _drive_worker(worker)
    assert len(captured_cb) == 1
    assert captured_cb[0] is not None, (
        "worker didn't pass a progress_cb — install_runtime can't report progress"
    )
    assert callable(captured_cb[0])


# ====================================================================
# Error paths


def test_install_error_routes_to_error_signal_with_stage():
    """InstallError carries a stage tag that the dialog uses to surface
    a stage-specific help message. Verify worker forwards both fields
    unchanged — no wrapping or summarization."""
    def _fake_install_fn(progress_cb=None, **kw):
        raise InstallError(
            stage="pip_install",
            message="ERROR: Could not find a version that satisfies torch==2.4.1",
        )
    worker = GpuInstallWorker(_install_fn=_fake_install_fn)
    captured = _drive_worker(worker)

    assert captured["done"] == [], "done emitted on error path"
    assert len(captured["error"]) == 1
    stage, msg = captured["error"][0]
    assert stage == "pip_install"
    assert "satisfies torch==2.4.1" in msg
    # Finished still fires so the thread teardown chain runs.
    assert captured["finished"] == [True]


@pytest.mark.parametrize("stage", [
    "download_python", "extract_python", "enable_site", "download_pip",
    "bootstrap_pip", "pip_install", "copy_package", "verify_cuda",
])
def test_every_install_error_stage_propagates(stage):
    """Each known InstallError stage propagates with its tag unchanged.
    Drift here would silently downgrade actionable errors to generic
    ones (the dialog routes on stage to pick the right help text)."""
    def _fake_install_fn(progress_cb=None, **kw):
        raise InstallError(stage=stage, message=f"failure at {stage}")
    worker = GpuInstallWorker(_install_fn=_fake_install_fn)
    captured = _drive_worker(worker)
    assert len(captured["error"]) == 1
    assert captured["error"][0][0] == stage


def test_generic_exception_routes_to_unknown_stage():
    """Anything that's NOT an InstallError (bug in the installer, OS
    error during file op, etc.) routes through stage='unknown' so the
    dialog still surfaces something to the user."""
    def _fake_install_fn(progress_cb=None, **kw):
        raise RuntimeError("unexpected internal failure")
    worker = GpuInstallWorker(_install_fn=_fake_install_fn)
    captured = _drive_worker(worker)
    assert len(captured["error"]) == 1
    stage, msg = captured["error"][0]
    assert stage == "unknown"
    assert "RuntimeError" in msg
    assert "unexpected internal failure" in msg


def test_finished_always_fires_regardless_of_outcome():
    """Even on InstallError or generic exception, finished() fires so
    the QThread.quit() → deleteLater chain runs. Skipping finished
    leaks the worker thread + the worker QObject indefinitely."""
    for raise_what in [
        InstallError("stage_x", "msg"),
        RuntimeError("generic"),
    ]:
        def _fake_install_fn(progress_cb=None, **kw):
            raise raise_what
        worker = GpuInstallWorker(_install_fn=_fake_install_fn)
        captured = _drive_worker(worker)
        assert captured["finished"] == [True], (
            f"finished didn't fire for {type(raise_what).__name__} — "
            f"thread leak"
        )


# ====================================================================
# Idempotency-when-already-installed


def test_returns_existing_runtime_info_when_install_short_circuits():
    """If torch_installer's idempotency kicks in (already installed),
    install_fn returns the existing RuntimeInfo without progress
    callbacks. Worker emits ONLY done — no progress events. The
    dialog handles this by auto-accepting immediately."""
    existing = RuntimeInfo(
        python_version="3.11.9", torch_version="2.4.1+cu121",
        cuda_available=True, cuda_device_name="EXISTING",
        installed_at_utc="2026-01-01T00:00:00Z",
    )
    def _fake_install_fn(progress_cb=None, **kw):
        # Don't fire progress_cb — represents the short-circuit case
        # where the install was already done.
        return existing
    worker = GpuInstallWorker(_install_fn=_fake_install_fn)
    captured = _drive_worker(worker)
    assert captured["progress"] == []
    assert len(captured["done"]) == 1
    assert captured["done"][0]["cuda_device_name"] == "EXISTING"
