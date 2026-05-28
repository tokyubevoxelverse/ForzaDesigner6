"""MainWindow snapshot signal + resume slot wiring (smoke test).

Per CLAUDE.md §8h: single MainWindow per process."""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from forza_abyss_painter.gui.main_window import MainWindow


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _snapshot(path: Path):
    path.write_text(json.dumps({
        "format": "fd6.shapes", "version": 1,
        "source_image": "x.png",
        "image_size": [32, 32], "shape_count": 1,
        "generated_at": "", "profile": "",
        "sticker_mode": False,
        "shapes": [
            {"type": "rotated_ellipse", "x": 16.0, "y": 16.0,
             "rx": 4.0, "ry": 4.0, "angle": 0.0,
             "color": [200, 80, 80, 255]},
        ],
        "_run_config": {
            "target_shape_count": 100,
            "random_samples": 1024,
            "max_resolution": 360,
            "edge_strength": 0.0,
            "posterize_levels": 0,
            "sticker_mode": False,
            "lock_alpha": True,
            "bbox_local": True,
            "joint_polish_steps": 0,
            "vram_budget_gib": 0.0,
            "preset_label": "test",
        },
    }), encoding="utf-8")


def test_main_window_has_resume_slot(qapp):
    win = MainWindow()
    try:
        assert hasattr(win, "_on_resume_requested")
        assert hasattr(win, "_on_gpu_snapshot")
        # upload_panel.resume_requested must exist on the wiring side.
        assert hasattr(win.upload, "resume_requested")
    finally:
        win.close()
        win.deleteLater()


def test_resume_slot_handles_snapshot(qapp, tmp_path, monkeypatch):
    """End-to-end smoke (no actual subprocess): click flow from
    upload_panel.resume_requested → ResumeDialog → would-spawn-worker.

    Patches runtime install prompt + ResumeDialog.exec to auto-accept,
    then asserts the dialog values dict was assembled.
    """
    snap = tmp_path / "x_50.json"
    _snapshot(snap)
    src = tmp_path / "x.png"
    src.write_bytes(b"\x89PNG\r\n\x1a\n")

    # Avoid the runtime install modal.
    from forza_abyss_painter.gui import runtime_install_dialog as rid
    monkeypatch.setattr(rid, "prompt_install_or_use_existing",
                         lambda parent: True)

    # Auto-accept the ResumeDialog.
    from forza_abyss_painter.gui import resume_dialog as rd_mod
    from PySide6.QtWidgets import QDialog

    captured_values: list[dict] = []
    original_values = rd_mod.ResumeDialog.values

    def patched_exec(self):
        captured_values.append(original_values(self))
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(rd_mod.ResumeDialog, "exec", patched_exec)

    # Patch embedded_python_exe to avoid the "runtime missing" modal.
    from forza_abyss_painter.runtime import torch_installer as ti
    monkeypatch.setattr(ti, "embedded_python_exe", lambda: src)

    # Avoid actually spawning the worker or a real QThread.
    from forza_abyss_painter.gui import gpu_gen_worker as ggw
    import PySide6.QtCore as _qtcore
    spawned: list = []

    class _StubWorker:
        def __init__(self, embedded_python_exe, config_path):
            spawned.append(Path(config_path))
        def moveToThread(self, *a, **k): pass
        def cancel(self): pass
        def stop(self): pass
        def run(self): pass

        class _Sig:
            def connect(self, *a, **k): pass
        started = progress = checkpoint = done = error = finished = snapshot = _Sig()

    class _StubThread:
        def __init__(self, parent=None): pass
        def start(self): pass
        def quit(self): pass
        def wait(self, ms=0): pass
        def isRunning(self): return False

        class _Sig:
            def connect(self, *a, **k): pass
        started = finished = _Sig()

    monkeypatch.setattr(ggw, "GpuGenWorker", _StubWorker)
    monkeypatch.setattr(_qtcore, "QThread", _StubThread)

    # Override the source-image resolver to avoid a real picker.
    from forza_abyss_painter.gui import main_window as mw_mod
    monkeypatch.setattr(mw_mod, "_resolve_source_image_path",
                         lambda json_path, name: src)

    # Bypass VRAM preflight gate (correction Task 9). Without this the
    # resume slot would invoke gpu_run_preflight, which queries nvidia-smi
    # — unavailable on the test box, blocking the run via modal.
    monkeypatch.setattr(mw_mod, "gpu_run_preflight",
                         lambda **kwargs: (True, dict(kwargs["preset"])))

    win = MainWindow()
    try:
        win._on_resume_requested(snap)
        # ResumeDialog was constructed + exec'd → values captured.
        assert len(captured_values) == 1
        v = captured_values[0]
        assert v["mode"] == "fresh"
        assert v["seed_shapes_path"] == str(snap)
        assert v["num_shapes"] == 100
    finally:
        win.close()
        win.deleteLater()


def test_resume_blocked_while_run_active(qapp, tmp_path, monkeypatch):
    """If a GPU run is in progress (self._thread.isRunning()), clicking
    Resume must NOT overwrite self._worker/_thread. The old worker's
    finished signal would later tear down the NEW thread."""
    snap = tmp_path / "x_50.json"
    _snapshot(snap)

    # Build a fake "running thread" — a QThread.isRunning() check is
    # the guard's source of truth. Use a real QThread that we keep
    # alive via a slow callable. Simpler: monkeypatch isRunning to
    # return True without actually starting a thread.
    from PySide6.QtCore import QThread

    win = MainWindow()
    try:
        # Stub up an "active" thread.
        fake_thread = QThread()
        monkeypatch.setattr(fake_thread, "isRunning", lambda: True)
        win._thread = fake_thread
        original_worker = getattr(win, "_worker", None)

        # Patch QMessageBox.information so the modal doesn't actually
        # block. Just record that it was called.
        called: list = []
        from PySide6.QtWidgets import QMessageBox
        monkeypatch.setattr(QMessageBox, "information",
                             lambda *args, **kwargs: called.append(args))

        win._on_resume_requested(snap)

        # The guard fired → modal shown, worker NOT overwritten.
        assert called, "QMessageBox.information should have been called"
        assert getattr(win, "_worker", None) is original_worker, (
            "self._worker was overwritten despite active run"
        )
    finally:
        # Don't call win._teardown_thread — the fake_thread isn't real.
        win._thread = None
        win.close()
        win.deleteLater()
