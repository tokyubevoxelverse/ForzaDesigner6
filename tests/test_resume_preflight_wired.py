"""_on_resume_requested calls gpu_run_preflight before spawning, using
ResumeDialog's chosen max_resolution + seed_canvas_size."""
import os
import json
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from unittest.mock import patch, MagicMock
from pathlib import Path
from PySide6.QtWidgets import QApplication, QDialog


def _values_dict(tmp_path, snap_path):
    return {
        "random_samples": 1000,
        "max_resolution": 720,
        "seed_canvas_size": (720, 480),
        "image_path": str(tmp_path / "x.png"),
        "output_json_path": str(tmp_path / "x.json"),
        "mode": "fresh",
        "seed_shapes_path": str(snap_path),
        "num_shapes": 1000,
        "joint_polish_steps": 60,
        "sticker_mode": False,
        "lock_alpha": False,
        "bbox_local": True,
        "preset_label": "Resume",
        "checkpoint_every": 100,
        "device": "cuda",
    }


def test_resume_runs_preflight_and_aborts_on_block(tmp_path):
    app = QApplication.instance() or QApplication([])
    from forza_abyss_painter.gui.main_window import MainWindow

    snap = {
        "format": "fd6.shapes", "version": 1,
        "image_size": [720, 480],
        "shapes": [],
        "_run_config": {
            "random_samples": 1000, "max_resolution": 720,
            "image_size": [720, 480],
        },
    }
    snap_path = tmp_path / "snap.json"
    snap_path.write_text(json.dumps(snap))

    # Source image must exist next to snapshot for resolve_source_image
    # to find it without prompting.
    src_path = tmp_path / "x.png"
    src_path.write_bytes(b"")

    win = MainWindow()
    try:
        with patch(
            "forza_abyss_painter.gui.main_window.gpu_run_preflight",
            return_value=(False, {"peak_gib": 12.0, "chunks_per_shape": 1, "free_gib": 17.0}),
        ) as preflight, patch(
            "forza_abyss_painter.gui.resume_dialog.ResumeDialog"
        ) as DlgCls, patch(
            "forza_abyss_painter.gui.gpu_gen_worker.GpuGenWorker"
        ) as Worker, patch(
            "forza_abyss_painter.gui.main_window._resolve_source_image_path",
            return_value=src_path,
        ), patch(
            "forza_abyss_painter.gui.runtime_install_dialog.prompt_install_or_use_existing",
            return_value=True,
        ):
            dlg = DlgCls.return_value
            dlg.exec.return_value = QDialog.DialogCode.Accepted
            dlg.values.return_value = _values_dict(tmp_path, snap_path)
            win._on_resume_requested(snap_path)
            preflight.assert_called_once()
            Worker.assert_not_called()
    finally:
        win.close()


def test_resume_passes_seed_canvas_size_to_build_run_config(tmp_path):
    """After preflight passes, build_run_config receives seed_canvas_size
    from ResumeDialog.values()."""
    app = QApplication.instance() or QApplication([])
    from forza_abyss_painter.gui.main_window import MainWindow

    snap = {
        "format": "fd6.shapes", "version": 1,
        "image_size": [720, 480],
        "shapes": [],
        "_run_config": {
            "random_samples": 1000, "max_resolution": 720,
            "image_size": [720, 480],
        },
    }
    snap_path = tmp_path / "snap.json"
    snap_path.write_text(json.dumps(snap))
    src_path = tmp_path / "x.png"
    src_path.write_bytes(b"")

    win = MainWindow()
    try:
        with patch(
            "forza_abyss_painter.gui.main_window.gpu_run_preflight",
            return_value=(True, {"peak_gib": 12.0, "chunks_per_shape": 1, "free_gib": 17.0}),
        ), patch(
            "forza_abyss_painter.gui.resume_dialog.ResumeDialog"
        ) as DlgCls, patch(
            "forza_abyss_painter.gui.main_window.build_run_config"
        ) as build_cfg, patch(
            "forza_abyss_painter.gui.gpu_gen_worker.GpuGenWorker"
        ), patch(
            "forza_abyss_painter.gui.main_window._resolve_source_image_path",
            return_value=src_path,
        ), patch(
            "forza_abyss_painter.gui.runtime_install_dialog.prompt_install_or_use_existing",
            return_value=True,
        ), patch(
            "forza_abyss_painter.runtime.torch_installer.embedded_python_exe",
            return_value=Path(str(tmp_path / "fakepython.exe")),
        ):
            # Force embedded_python_exe.exists() True so we hit the worker spawn.
            from forza_abyss_painter.runtime import torch_installer
            fake_py = tmp_path / "fakepython.exe"
            fake_py.write_bytes(b"")

            dlg = DlgCls.return_value
            dlg.exec.return_value = QDialog.DialogCode.Accepted
            dlg.values.return_value = _values_dict(tmp_path, snap_path)
            build_cfg.return_value = {
                "image_path": "", "output_json_path": "",
                "num_shapes": 1000, "max_resolution": 720,
            }
            win._on_resume_requested(snap_path)
            build_cfg.assert_called_once()
            kwargs = build_cfg.call_args.kwargs
            assert kwargs.get("seed_canvas_size") == (720, 480)
    finally:
        win.close()
