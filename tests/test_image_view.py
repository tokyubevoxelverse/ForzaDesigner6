from PySide6.QtCore import Qt

from fd6.gui.widgets.image_view import _pick_transform_mode


def test_pick_transform_mode_prefers_fast_upscale_for_preview():
    mode = _pick_transform_mode(320, 180, 1280, 720, True)
    assert mode == Qt.FastTransformation


def test_pick_transform_mode_keeps_smooth_when_downscaling():
    mode = _pick_transform_mode(1280, 720, 320, 180, True)
    assert mode == Qt.SmoothTransformation


def test_pick_transform_mode_keeps_smooth_when_crisp_mode_disabled():
    mode = _pick_transform_mode(320, 180, 1280, 720, False)
    assert mode == Qt.SmoothTransformation
