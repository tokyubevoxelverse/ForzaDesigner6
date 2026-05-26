import numpy as np
from PIL import Image

from fd6.shapegen.profile import Profile
from fd6.shapegen.worker import (
    _posterize_rgb,
    _prepare_line_guide_source_image,
    _prepare_target_image,
    _preprocess_target,
)


def test_preprocess_target_boosts_step_edge_contrast():
    target = np.empty((8, 8, 3), dtype=np.uint8)
    target[:, :4] = (96, 96, 96)
    target[:, 4:] = (160, 160, 160)

    processed = _preprocess_target(target, 256)

    before_gap = int(target[4, 4, 0]) - int(target[4, 3, 0])
    after_gap = int(processed[4, 4, 0]) - int(processed[4, 3, 0])
    assert after_gap > before_gap
    assert int(processed[4, 3, 0]) < int(target[4, 3, 0])
    assert int(processed[4, 4, 0]) > int(target[4, 4, 0])


def test_posterize_rgb_respects_requested_level_count():
    gradient = np.linspace(0, 255, 16, dtype=np.uint8)
    target = np.stack([gradient, gradient, gradient], axis=1).reshape(1, 16, 3)

    posterized = _posterize_rgb(target, 4)

    assert np.unique(posterized[:, :, 0]).tolist() == [0, 85, 170, 255]


def test_prepare_target_image_keeps_sticker_alpha_through_resize():
    rgba = np.zeros((3, 6, 4), dtype=np.uint8)
    rgba[:, :, :3] = (40, 80, 120)
    rgba[:, :3, 3] = 255
    rgba[:, 3:, 3] = 0
    img = Image.fromarray(rgba, "RGBA")
    profile = Profile(max_resolution=4, posterize_levels=256)

    target, alpha_mask = _prepare_target_image(img, profile, True)

    assert target.shape == (2, 4, 3)
    assert alpha_mask is not None
    assert alpha_mask.shape == (2, 4)
    assert int(alpha_mask[:, :2].mean()) > int(alpha_mask[:, 2:].mean())


def test_prepare_line_guide_source_matches_non_sticker_target_geometry():
    rgb = np.full((4, 8, 3), 255, dtype=np.uint8)
    rgb[:, :2] = 0
    img = Image.fromarray(rgb, "RGB")
    profile = Profile(max_resolution=8, posterize_levels=256)

    target, _alpha = _prepare_target_image(img, profile, False)
    line_source = _prepare_line_guide_source_image(img, profile, False)

    assert line_source.size == (target.shape[1], target.shape[0])
    line_arr = np.asarray(line_source, dtype=np.uint8)
    assert int(line_arr[0].mean()) > 240
    assert int(line_arr[3, 0].mean()) < 80
