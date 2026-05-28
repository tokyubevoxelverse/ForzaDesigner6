"""_on_polish_requested calls gpu_run_preflight before spawning."""
import os
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from unittest.mock import patch
from pathlib import Path
from PySide6.QtWidgets import QApplication


def test_polish_runs_preflight_and_aborts_on_block(tmp_path):
    app = QApplication.instance() or QApplication([])
    from forza_abyss_painter.gui.main_window import MainWindow

    # Build a minimal fd6.shapes JSON the dialog can load.
    snap_json = tmp_path / "doc.json"
    snap_json.write_text(
        '{"format":"fd6.shapes","version":1,"image_size":[720,480],"shapes":[]}'
    )

    win = MainWindow()
    try:
        with patch(
            "forza_abyss_painter.gui.main_window.gpu_run_preflight",
            return_value=(False, {"peak_gib": 12.0, "chunks_per_shape": 1, "free_gib": 17.0}),
        ) as preflight, patch(
            "forza_abyss_painter.gui.gpu_gen_worker.GpuGenWorker"
        ) as Worker, patch(
            "forza_abyss_painter.gui.main_window.MainWindow._resolve_source_for_loaded_json",
            return_value=Path(str(tmp_path / "fake_source.png")),
        ), patch(
            "forza_abyss_painter.gui.runtime_install_dialog.prompt_install_or_use_existing",
            return_value=True,
        ), patch(
            "forza_abyss_painter.gui.polish_dialog.PolishDialog"
        ) as DlgCls:
            from PySide6.QtWidgets import QDialog
            dlg = DlgCls.return_value
            dlg.exec.return_value = QDialog.DialogCode.Accepted
            dlg.values.return_value = {
                "output_path": tmp_path / "out.json",
                "steps": 60,
                "lock_alpha": True,
            }
            win._on_polish_requested(snap_json)
            preflight.assert_called_once()
            Worker.assert_not_called()
    finally:
        win.close()
