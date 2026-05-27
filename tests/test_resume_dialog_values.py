"""ResumeDialog reads _run_config from the snapshot, shows the
continue summary, and returns a RunConfig dict via .values()."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from forza_abyss_painter.gui.resume_dialog import ResumeDialog


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _snapshot_with_config(path: Path, count=2900, target=3000):
    path.write_text(json.dumps({
        "format": "fd6.shapes", "version": 1,
        "source_image": "ziz.png",
        "image_size": [1200, 981], "shape_count": count,
        "generated_at": "", "profile": "_default",
        "sticker_mode": False,
        "shapes": [
            {"type": "rotated_ellipse", "x": 100.0, "y": 100.0,
             "rx": 5.0, "ry": 5.0, "angle": 0.0,
             "color": [128, 128, 128, 255]},
        ] * count,
        "_run_config": {
            "target_shape_count": target,
            "random_samples": 1000,
            "max_resolution": 1200,
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


def _snapshot_without_config(path: Path, count=500):
    path.write_text(json.dumps({
        "format": "fd6.shapes", "version": 1,
        "source_image": "x.png",
        "image_size": [720, 720], "shape_count": count,
        "generated_at": "", "profile": "",
        "sticker_mode": False,
        "shapes": [
            {"type": "rotated_ellipse", "x": 100.0, "y": 100.0,
             "rx": 5.0, "ry": 5.0, "angle": 0.0,
             "color": [128, 128, 128, 255]},
        ] * count,
    }), encoding="utf-8")


def test_values_from_embedded_run_config(qapp, tmp_path):
    snap = tmp_path / "ziz_2900.json"
    _snapshot_with_config(snap, count=2900, target=3000)
    src = tmp_path / "ziz.png"
    src.write_bytes(b"\x89PNG\r\n\x1a\n")

    dlg = ResumeDialog(parent=None, snapshot_path=snap, source_image_path=src)
    values = dlg.values()
    assert values["mode"] == "fresh"
    assert values["seed_shapes_path"] == str(snap)
    assert values["image_path"] == str(src)
    assert values["num_shapes"] == 3000
    assert values["random_samples"] == 1000
    assert values["max_resolution"] == 1200
    assert values["joint_polish_steps"] == 250
    assert values["lock_alpha"] is True
    dlg.deleteLater()


def test_continue_summary_in_label(qapp, tmp_path):
    snap = tmp_path / "ziz_2900.json"
    _snapshot_with_config(snap, count=2900, target=3000)
    src = tmp_path / "ziz.png"
    src.write_bytes(b"\x89PNG\r\n\x1a\n")

    dlg = ResumeDialog(parent=None, snapshot_path=snap, source_image_path=src)
    text = dlg.summary_label.text()
    assert "2900" in text
    assert "3000" in text
    assert "ziz_2900.json" in text
    dlg.deleteLater()


def test_missing_run_config_shows_preset_picker(qapp, tmp_path):
    snap = tmp_path / "x_500.json"
    _snapshot_without_config(snap, count=500)
    src = tmp_path / "x.png"
    src.write_bytes(b"\x89PNG\r\n\x1a\n")

    dlg = ResumeDialog(parent=None, snapshot_path=snap, source_image_path=src)
    assert dlg.preset_combo is not None
    assert dlg.preset_combo.isVisibleTo(dlg)
    dlg.preset_combo.setCurrentIndex(0)
    values = dlg.values()
    assert values["num_shapes"] > 500
    assert values["seed_shapes_path"] == str(snap)
    dlg.deleteLater()


def test_target_must_exceed_current_count(qapp, tmp_path):
    """If the snapshot is already at target (2900 of 2900), Resume is
    not meaningful. The dialog should disable the Resume button."""
    snap = tmp_path / "ziz_2900.json"
    _snapshot_with_config(snap, count=2900, target=2900)
    src = tmp_path / "ziz.png"
    src.write_bytes(b"\x89PNG\r\n\x1a\n")

    dlg = ResumeDialog(parent=None, snapshot_path=snap, source_image_path=src)
    assert dlg.resume_btn.isEnabled() is False
    text = dlg.summary_label.text().lower()
    assert "already at target" in text or "nothing to resume" in text, (
        f"summary should explain why Resume is disabled; got {text!r}"
    )
    dlg.deleteLater()


def test_underscore_n_only_stem_falls_back(qapp, tmp_path):
    """Snapshot named `_100.json` (stem is just '_100') must not
    produce '.json' as the output path. Fall back to the original
    stem so the output stays a valid filename."""
    snap = tmp_path / "_100.json"
    _snapshot_with_config(snap, count=100, target=200)
    src = tmp_path / "ziz.png"
    src.write_bytes(b"\x89PNG\r\n\x1a\n")

    dlg = ResumeDialog(parent=None, snapshot_path=snap, source_image_path=src)
    values = dlg.values()
    # Output should NOT be ".json" — should fall back to the original stem.
    output = Path(values["output_json_path"])
    assert output.name != ".json", (
        f"output path is the literal hidden filename: {output}"
    )
    assert output.name == "_100.json" or output.name.endswith("_100.json")
    dlg.deleteLater()
