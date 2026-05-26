"""GenerateLocallyDialog accepts an `initial_source_path` kwarg so the
#85 Re-shape-gen flow can pre-fill the source image from a loaded JSON."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from forza_abyss_painter.gui.generate_dialog import GenerateLocallyDialog  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def test_no_initial_source_path_keeps_source_field_empty(qapp, tmp_path):
    dlg = GenerateLocallyDialog(parent=None)
    assert dlg.source_path is None
    assert dlg.source_field.text() == ""
    assert dlg.generate_btn.isEnabled() is False
    dlg.deleteLater()


def test_initial_source_path_prefills_source_field(qapp, tmp_path):
    img = tmp_path / "nikke.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")
    dlg = GenerateLocallyDialog(parent=None, initial_source_path=img)
    assert dlg.source_path == img
    assert dlg.source_field.text() == str(img)
    assert dlg.generate_btn.isEnabled() is True
    # Output placeholder should reflect the pre-filled source.
    placeholder = dlg.output_field.placeholderText()
    assert "nikke" in placeholder
    assert placeholder.endswith(".json")
    dlg.deleteLater()


def test_initial_source_path_missing_file_keeps_source_unset(qapp, tmp_path):
    img = tmp_path / "missing.png"   # not created
    dlg = GenerateLocallyDialog(parent=None, initial_source_path=img)
    assert dlg.source_path is None, (
        "constructor must reject a missing initial_source_path so "
        "the user re-picks rather than running on a stale path"
    )
    assert dlg.generate_btn.isEnabled() is False
    dlg.deleteLater()
