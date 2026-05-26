"""PolishDialog exposes (steps, lock_alpha, output_path) via .values()."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from forza_abyss_painter.gui.polish_dialog import PolishDialog  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def test_defaults(qapp, tmp_path):
    loaded = tmp_path / "shapes.json"
    loaded.write_text("{}")
    src = tmp_path / "nikke.png"
    src.write_bytes(b"\x89PNG\r\n\x1a\n")

    dlg = PolishDialog(parent=None, loaded_json_path=loaded, source_image_path=src)

    values = dlg.values()
    assert values["steps"] == 150
    assert values["lock_alpha"] is True
    # Default output: <loaded_stem>_polished.json next to the loaded JSON.
    assert values["output_path"] == loaded.parent / "shapes_polished.json"
    dlg.deleteLater()


def test_steps_range_clamped(qapp, tmp_path):
    loaded = tmp_path / "shapes.json"
    loaded.write_text("{}")
    src = tmp_path / "nikke.png"
    src.write_bytes(b"\x89PNG\r\n\x1a\n")

    dlg = PolishDialog(parent=None, loaded_json_path=loaded, source_image_path=src)

    # Spin box bounds: 50..500.
    assert dlg.steps_spinbox.minimum() == 50
    assert dlg.steps_spinbox.maximum() == 500
    dlg.steps_spinbox.setValue(75)
    assert dlg.values()["steps"] == 75
    dlg.deleteLater()


def test_lock_alpha_toggle(qapp, tmp_path):
    loaded = tmp_path / "shapes.json"
    loaded.write_text("{}")
    src = tmp_path / "nikke.png"
    src.write_bytes(b"\x89PNG\r\n\x1a\n")

    dlg = PolishDialog(parent=None, loaded_json_path=loaded, source_image_path=src)
    dlg.lock_alpha_cb.setChecked(False)
    assert dlg.values()["lock_alpha"] is False
    dlg.deleteLater()


def test_output_path_override(qapp, tmp_path):
    loaded = tmp_path / "shapes.json"
    loaded.write_text("{}")
    src = tmp_path / "nikke.png"
    src.write_bytes(b"\x89PNG\r\n\x1a\n")
    override = tmp_path / "custom_out.json"

    dlg = PolishDialog(parent=None, loaded_json_path=loaded, source_image_path=src)
    dlg.set_output_path(override)
    assert dlg.values()["output_path"] == override
    dlg.deleteLater()
