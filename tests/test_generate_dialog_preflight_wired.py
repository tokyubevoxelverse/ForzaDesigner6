"""GenerateLocallyDialog._on_generate_clicked must call gpu_run_preflight
and NOT accept (close with QDialog.Accepted) if proceed=False. When
proceed is True, .values() flows through unchanged -- chunked-K handles
VRAM fit inside the engine, so the dialog never overrides max_resolution.

Chunked-K Task 2 (preflight simplification): the dialog no longer
stashes an effective_max_resolution override. This test pins the new
"preset flows through unchanged" contract.
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from unittest.mock import patch  # noqa: E402

from PySide6.QtWidgets import QApplication, QDialog  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _make_dialog(tmp_path):
    """Build a dialog with a real source path so the generate-button
    gate doesn't short-circuit _on_generate_clicked."""
    from forza_abyss_painter.gui.generate_dialog import GenerateLocallyDialog
    img = tmp_path / "src.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")
    dlg = GenerateLocallyDialog(
        parent=None,
        initial_source_path=img,
        gpu_budget_gib=24.0,
    )
    return dlg


def test_generate_dialog_blocks_accept_on_preflight_block(qapp, tmp_path):
    dlg = _make_dialog(tmp_path)
    try:
        with patch(
            "forza_abyss_painter.gui.generate_dialog.gpu_run_preflight",
            return_value=(False, {
                "peak_gib": 99.0, "chunks_per_shape": 1, "free_gib": 4.0,
            }),
        ) as preflight:
            dlg._on_generate_clicked()
            preflight.assert_called_once()
            assert dlg.result() != QDialog.DialogCode.Accepted
            # Worker must NOT have been constructed.
            assert not hasattr(dlg, "_worker") or dlg._worker is None
    finally:
        dlg.deleteLater()


def test_generate_dialog_values_unchanged_when_preflight_ok(qapp, tmp_path):
    """When preflight returns proceed=True, .values() returns the user's
    chosen preset unchanged -- chunked-K handles fit inside the engine,
    so the dialog never overrides max_resolution.

    We stop short of letting _on_generate_clicked actually spawn the worker
    by patching embedded_python_exe to point at a non-existent path -- the
    real-path early-return surfaces a modal and the dialog stays open,
    which is fine; we only care that .values() reflects the unchanged preset.
    """
    dlg = _make_dialog(tmp_path)
    try:
        baseline = dict(dlg.values())

        from pathlib import Path
        with patch(
            "forza_abyss_painter.gui.generate_dialog.gpu_run_preflight",
            return_value=(True, {
                "peak_gib": 12.0, "chunks_per_shape": 12, "free_gib": 17.0,
            }),
        ), patch(
            "forza_abyss_painter.gui.generate_dialog.embedded_python_exe",
            return_value=Path("/nonexistent/python.exe"),
        ), patch(
            "forza_abyss_painter.gui.generate_dialog.QMessageBox.critical",
        ):
            dlg._on_generate_clicked()
            # .values() must reflect the user's original preset, NOT a
            # lowered effective value.
            assert dlg.values()["max_resolution"] == baseline["max_resolution"]
            assert dlg.values()["random_samples"] == baseline["random_samples"]
    finally:
        dlg.deleteLater()
