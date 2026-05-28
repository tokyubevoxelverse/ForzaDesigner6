"""_start_gpu must delegate VRAM gating to gpu_run_preflight and abort
the worker spawn when proceed=False.

Patches the helper at its module-level binding in main_window so we can
detect (a) that it was called and (b) whether GpuGenWorker was
constructed downstream. Avoids the runtime-install modal by patching
is_runtime_installed + embedded_python_exe.
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _make_image(tmp_path: Path) -> Path:
    from PIL import Image
    p = tmp_path / "smoke.png"
    Image.new("RGBA", (64, 64), (255, 0, 0, 255)).save(p)
    return p


def _patch_runtime(monkeypatch):
    """Patch the function-local imports inside _start_gpu so the runtime
    install modal doesn't fire."""
    from forza_abyss_painter.runtime import torch_installer as ti
    monkeypatch.setattr(ti, "is_runtime_installed", lambda: True)
    monkeypatch.setattr(ti, "embedded_python_exe", lambda: Path("/usr/bin/python3"))


def test_start_gpu_calls_preflight_and_aborts_on_block(qapp, tmp_path, monkeypatch):
    """When gpu_run_preflight returns (False, _), _start_gpu must NOT
    construct GpuGenWorker."""
    _patch_runtime(monkeypatch)
    from forza_abyss_painter.gui.main_window import MainWindow

    win = MainWindow()
    try:
        img = _make_image(tmp_path)
        profile = win.settings_panel.build_profile()

        with patch(
            "forza_abyss_painter.gui.main_window.gpu_run_preflight",
            return_value=(False, {
                "peak_gib": 12.0, "chunks_per_shape": 12, "free_gib": 17.0,
            }),
        ) as preflight, patch(
            "forza_abyss_painter.gui.gpu_gen_worker.GpuGenWorker"
        ) as Worker:
            win._start_gpu(img, profile, sticker_mode=False)
            preflight.assert_called_once()
            Worker.assert_not_called()
    finally:
        win.close()
        win.deleteLater()


def test_start_gpu_proceeds_when_preflight_ok(qapp, tmp_path, monkeypatch):
    """When gpu_run_preflight returns (True, effective), _start_gpu must
    construct GpuGenWorker with the effective preset."""
    _patch_runtime(monkeypatch)

    # Stub the worker + thread so we don't actually spawn a subprocess.
    from forza_abyss_painter.gui import gpu_gen_worker as ggw
    import PySide6.QtCore as _qtcore

    spawned: list = []

    class _Sig:
        def connect(self, *a, **k): pass

    class _StubWorker:
        def __init__(self, embedded_python_exe, config_path):
            spawned.append(Path(config_path))
        def moveToThread(self, *a, **k): pass
        def stop(self): pass
        def cancel(self): pass
        def run(self): pass
        progress = checkpoint = snapshot = done = error = finished = _Sig()

    class _StubThread:
        def __init__(self, parent=None): pass
        def start(self): pass
        def quit(self): pass
        def wait(self, ms=0): pass
        def isRunning(self): return False
        started = finished = _Sig()

    monkeypatch.setattr(ggw, "GpuGenWorker", _StubWorker)
    monkeypatch.setattr(_qtcore, "QThread", _StubThread)

    from forza_abyss_painter.gui.main_window import MainWindow

    win = MainWindow()
    try:
        img = _make_image(tmp_path)
        profile = win.settings_panel.build_profile()
        info = {"peak_gib": 12.0, "chunks_per_shape": 1, "free_gib": 17.0}
        with patch(
            "forza_abyss_painter.gui.main_window.gpu_run_preflight",
            return_value=(True, info),
        ) as preflight:
            win._start_gpu(img, profile, sticker_mode=False)
            preflight.assert_called_once()
            assert spawned, "GpuGenWorker should have been constructed on proceed=True"
    finally:
        win.close()
        win.deleteLater()
