"""ResumeDialog detects snapshot.max_resolution > recommended and
offers to lower with re-rasterization. .values() returns the lowered
max_resolution + original snapshot canvas as seed_canvas_size for
the engine to scale seeded shape coords."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from unittest.mock import patch

from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _write_snapshot(
    path: Path, *, K=8192, max_res=1200, image_size=(1200, 800),
    count=500, target=1000,
):
    path.write_text(json.dumps({
        "format": "fd6.shapes", "version": 1,
        "source_image": "src.png",
        "image_size": list(image_size),
        "shape_count": count,
        "generated_at": "",
        "profile": "_default",
        "sticker_mode": False,
        "shapes": [
            {"type": "rotated_ellipse", "x": 100.0, "y": 100.0,
             "rx": 5.0, "ry": 5.0, "angle": 0.0,
             "color": [128, 128, 128, 255]},
        ] * count,
        "_run_config": {
            "target_shape_count": target,
            "random_samples": K,
            "max_resolution": max_res,
            "image_size": list(image_size),
            "edge_strength": 0.0,
            "posterize_levels": 0,
            "sticker_mode": False,
            "lock_alpha": True,
            "bbox_local": True,
            "joint_polish_steps": 250,
            "vram_budget_gib": 0.0,
            "preset_label": "_default",
        },
    }), encoding="utf-8")


def test_resume_dialog_offers_lower_when_mismatch(qapp, tmp_path):
    """When the recommended max_res is lower than snapshot's baked
    max_res, .values() should return the lowered value AND the
    original snapshot canvas as seed_canvas_size."""
    from forza_abyss_painter.gui.resume_dialog import ResumeDialog

    snap = tmp_path / "src_500.json"
    _write_snapshot(
        snap, K=8192, max_res=1200, image_size=(1200, 800),
        count=500, target=1000,
    )
    src = tmp_path / "src.png"
    src.write_bytes(b"\x89PNG\r\n\x1a\n")

    with patch(
        "forza_abyss_painter.gui.resume_dialog._probe_free_gib",
        return_value=22.0,
    ), patch(
        "forza_abyss_painter.gui.resume_dialog._recommend",
        return_value=480,
    ):
        dlg = ResumeDialog(
            parent=None,
            snapshot_path=snap,
            source_image_path=src,
            gpu_budget_gib=24.0,
        )
        try:
            vals = dlg.values()
            assert vals["max_resolution"] == 480
            assert vals["seed_canvas_size"] == (1200, 800)
        finally:
            dlg.deleteLater()


def test_resume_dialog_passthrough_when_fits(qapp, tmp_path):
    """When the recommended max_res >= baked max_res, .values()
    should return the original baked max_resolution unchanged, and
    seed_canvas_size still reflects the snapshot canvas."""
    from forza_abyss_painter.gui.resume_dialog import ResumeDialog

    snap = tmp_path / "src_500.json"
    _write_snapshot(
        snap, K=1000, max_res=720, image_size=(720, 480),
        count=500, target=1000,
    )
    src = tmp_path / "src.png"
    src.write_bytes(b"\x89PNG\r\n\x1a\n")

    with patch(
        "forza_abyss_painter.gui.resume_dialog._probe_free_gib",
        return_value=22.0,
    ), patch(
        "forza_abyss_painter.gui.resume_dialog._recommend",
        return_value=1200,
    ):
        dlg = ResumeDialog(
            parent=None,
            snapshot_path=snap,
            source_image_path=src,
            gpu_budget_gib=24.0,
        )
        try:
            vals = dlg.values()
            assert vals["max_resolution"] == 720
            assert vals["seed_canvas_size"] == (720, 480)
        finally:
            dlg.deleteLater()


def test_resume_dialog_summary_mentions_lowered_when_mismatch(qapp, tmp_path):
    """When lowered, the dialog body should surface the mismatch via
    a warning label (not a separate modal)."""
    from forza_abyss_painter.gui.resume_dialog import ResumeDialog

    snap = tmp_path / "src_500.json"
    _write_snapshot(
        snap, K=8192, max_res=1200, image_size=(1200, 800),
        count=500, target=1000,
    )
    src = tmp_path / "src.png"
    src.write_bytes(b"\x89PNG\r\n\x1a\n")

    with patch(
        "forza_abyss_painter.gui.resume_dialog._probe_free_gib",
        return_value=4.0,
    ), patch(
        "forza_abyss_painter.gui.resume_dialog._recommend",
        return_value=480,
    ):
        dlg = ResumeDialog(
            parent=None,
            snapshot_path=snap,
            source_image_path=src,
            gpu_budget_gib=24.0,
        )
        try:
            # The dialog should have a lower-warning label that
            # mentions both the baked and the lowered values.
            assert hasattr(dlg, "lower_warning_label")
            text = dlg.lower_warning_label.text()
            assert "1200" in text
            assert "480" in text
            assert dlg.lower_warning_label.isVisibleTo(dlg)
        finally:
            dlg.deleteLater()
