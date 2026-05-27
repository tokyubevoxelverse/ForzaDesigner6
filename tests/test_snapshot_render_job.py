"""_RenderSnapshotJob reads a fixture snapshot JSON, renders it via
render_shapes, and pushes the numpy canvas into PreviewPanel.on_preview.

The job must run synchronously when invoked from the same thread so
we don't need a Qt event loop for the test — it's a QRunnable, and
calling .run() directly executes the body. Cross-thread marshaling
to the GUI thread is tested separately under MainWindow."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from forza_abyss_painter.gui.preview_panel import PreviewPanel
from forza_abyss_painter.gui.snapshot_render import _RenderSnapshotJob


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _snapshot(path: Path, w=32, h=32):
    path.write_text(json.dumps({
        "format": "fd6.shapes", "version": 1,
        "source_image": "x.png",
        "image_size": [w, h], "shape_count": 1,
        "generated_at": "", "profile": "",
        "sticker_mode": False,
        "shapes": [
            {"type": "rotated_ellipse", "x": 16.0, "y": 16.0,
             "rx": 8.0, "ry": 8.0, "angle": 0.0,
             "color": [200, 80, 80, 255]},
        ],
    }), encoding="utf-8")


def test_render_job_calls_on_preview(qapp, tmp_path):
    snap = tmp_path / "fixture_1.json"
    _snapshot(snap)

    panel = PreviewPanel()
    # Capture the canvas the preview panel receives.
    received = {}
    original = panel.on_preview
    def spy(arr):
        received["canvas"] = arr
        original(arr)
    panel.on_preview = spy   # type: ignore

    job = _RenderSnapshotJob(snap, panel)
    job.run()
    # Inline run dispatches QMetaObject.invokeMethod with QueuedConnection,
    # which posts an event. Drain the queue.
    qapp.processEvents()

    assert "canvas" in received, "preview.on_preview was never called"
    canvas = received["canvas"]
    assert canvas.shape == (32, 32, 3) or canvas.shape == (32, 32, 4)
    panel.deleteLater()


def test_render_job_swallows_corrupt_snapshot(qapp, tmp_path):
    """If the snapshot is mid-write or invalid, the job must return
    cleanly without raising — the next snapshot fires within seconds."""
    snap = tmp_path / "corrupt_1.json"
    snap.write_text("{not json")

    panel = PreviewPanel()
    received = {}
    original = panel.on_preview
    def spy(arr):
        received["canvas"] = arr
        original(arr)
    panel.on_preview = spy   # type: ignore

    job = _RenderSnapshotJob(snap, panel)
    job.run()   # must not raise
    qapp.processEvents()

    assert "canvas" not in received
    panel.deleteLater()


def test_render_job_handles_empty_shapes(qapp, tmp_path):
    """Edge: snapshot at count=0 (theoretically) has empty shapes
    list. Render should produce a clean canvas, not raise."""
    snap = tmp_path / "empty_0.json"
    snap.write_text(json.dumps({
        "format": "fd6.shapes", "version": 1,
        "source_image": "x.png",
        "image_size": [32, 32], "shape_count": 0,
        "generated_at": "", "profile": "",
        "sticker_mode": False,
        "shapes": [],
    }), encoding="utf-8")

    panel = PreviewPanel()
    received = {}
    original = panel.on_preview
    def spy(arr):
        received["canvas"] = arr
        original(arr)
    panel.on_preview = spy   # type: ignore

    job = _RenderSnapshotJob(snap, panel)
    job.run()
    qapp.processEvents()

    assert "canvas" in received
    panel.deleteLater()
