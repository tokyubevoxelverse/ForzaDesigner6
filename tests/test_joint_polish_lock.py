import numpy as np
import torch

from forza_abyss_painter.shapegen.gpu.device import DTYPE, get_device
from forza_abyss_painter.shapegen.gpu.joint_polish import joint_polish


def _tiny_input():
    device = get_device()
    h, w = 16, 16
    target = torch.full((h, w, 3), 200, dtype=torch.uint8, device=device)
    canvas_init = torch.full((h, w, 3), 40, dtype=torch.uint8, device=device)
    shapes = [
        {"type": "rotated_ellipse", "x": 5, "y": 5, "rx": 3, "ry": 3, "angle": 0.0,
         "color": [200, 0, 0, 128]},  # alpha=128 input
        {"type": "rotated_ellipse", "x": 10, "y": 10, "rx": 3, "ry": 3, "angle": 0.0,
         "color": [0, 200, 0, 96]},   # alpha=96 input
    ]
    return shapes, target, canvas_init, h, w


def test_lock_alpha_forces_output_alpha_to_255():
    shapes, target, canvas_init, h, w = _tiny_input()
    refined, _ = joint_polish(shapes, target, alpha_t=None, alpha_mask_f=None,
                              edge_weight=None, canvas_init=canvas_init,
                              h=h, w=w, steps=5, lock_alpha=True)
    assert len(refined) == 2
    assert all(s["color"][3] == 255 for s in refined)


def test_lock_alpha_off_preserves_old_behavior():
    """Without lock_alpha, alpha is optimized (output != input but in 16..255 range)."""
    shapes, target, canvas_init, h, w = _tiny_input()
    refined, _ = joint_polish(shapes, target, alpha_t=None, alpha_mask_f=None,
                              edge_weight=None, canvas_init=canvas_init,
                              h=h, w=w, steps=5, lock_alpha=False)
    # alpha is optimizable; at least one shape's alpha differs from its input value
    assert any(s["color"][3] != ipt["color"][3] for s, ipt in zip(refined, shapes))


def test_lock_alpha_geometry_still_optimized_in_legacy_mode():
    """Even with alpha locked, geometry still moves under the LEGACY joint-geom polish mode
    (freeze_geometry=False). Pins that geom is in the Adam optimizer when freeze is off.
    The production default flipped to freeze_geometry=True later, so to exercise this
    pre-flip contract we must explicitly opt INTO the legacy mode."""
    shapes, target, canvas_init, h, w = _tiny_input()
    refined, _ = joint_polish(shapes, target, alpha_t=None, alpha_mask_f=None,
                              edge_weight=None, canvas_init=canvas_init,
                              h=h, w=w, steps=20, lr=2.0, lock_alpha=True,
                              freeze_geometry=False)
    # at least one geometry field moved
    moved = any(
        abs(r["x"] - s["x"]) > 0.01 or abs(r["y"] - s["y"]) > 0.01
        or abs(r["rx"] - s["rx"]) > 0.01 or abs(r["ry"] - s["ry"]) > 0.01
        for r, s in zip(refined, shapes)
    )
    assert moved
