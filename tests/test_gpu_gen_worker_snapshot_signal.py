"""GpuGenWorker has a new `snapshot = Signal(int, int, str)` that
fires when the runner emits a snapshot event."""
from __future__ import annotations

from pathlib import Path

import pytest

from forza_abyss_painter.gui.gpu_gen_worker import GpuGenWorker


def test_snapshot_signal_exists():
    """The signal must exist on the class (introspection check, no
    QApplication needed)."""
    assert hasattr(GpuGenWorker, "snapshot")


def test_dispatch_routes_snapshot_event(tmp_path):
    """_dispatch returns True for a 'snapshot' kind event and emits
    the right Signal payload."""
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])

    py = Path("/usr/bin/python3")   # not invoked; dispatch test only
    cfg = tmp_path / "cfg.json"
    cfg.write_text("{}")
    worker = GpuGenWorker(embedded_python_exe=py, config_path=cfg)

    received: list = []
    worker.snapshot.connect(lambda c, t, p: received.append((c, t, p)))

    handled = worker._dispatch({
        "kind": "snapshot",
        "shape_count": 100,
        "total": 300,
        "path": "/tmp/fixture_100.json",
    })
    assert handled is True
    assert received == [(100, 300, "/tmp/fixture_100.json")]


def test_dispatch_unknown_returns_false(tmp_path):
    """Unknown event kinds aren't snapshots and aren't routed."""
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])

    py = Path("/usr/bin/python3")
    cfg = tmp_path / "cfg.json"
    cfg.write_text("{}")
    worker = GpuGenWorker(embedded_python_exe=py, config_path=cfg)
    handled = worker._dispatch({"kind": "unrecognized_event"})
    assert handled is False
