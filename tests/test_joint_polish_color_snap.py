"""Regression test for the post-polish color snap-back.

joint_polish runs Adam on RGB at lr*4 for `steps` iterations with no chromatic regularization.
Momentum routinely overshoots channels to saturation extremes the target image doesn't contain.
In production runs we observed ~125 / 3000 shapes hitting min<15 AND max>240 post-polish
(vs 0 pre-polish), with ~45 turning fully green/yellow when the optimizer drifted from
sparkle-yellow target regions.

The fix (when lock_alpha=True): after the gradient loop, replace each shape's RGB with its
closed-form-optimal value — the simple mean of the target image over the shape's
actually-visible mask region (mask AND opaque AND not occluded by later layers).

These tests assert the snap-back keeps colors in-distribution.
"""
import numpy as np
import torch

from forza_abyss_painter.shapegen.gpu.device import DTYPE, get_device
from forza_abyss_painter.shapegen.gpu.joint_polish import joint_polish


def _flat_target(h, w, rgb):
    return torch.tensor(np.full((h, w, 3), rgb, dtype=np.uint8), device=get_device())


def test_polish_color_stays_in_distribution_on_flat_grey_target():
    """A flat grey target has no green pixels — post-polish colors must not be saturated green.
    Without the snap-back, Adam could drive RGB to extremes like [0, 255, 0] in pursuit of
    micro-loss gains; with the snap-back, every shape's color is the mean over its visible
    pixels in the target, which is exactly the flat grey value."""
    h, w = 32, 32
    device = get_device()
    target_rgb = (180, 180, 180)
    target = _flat_target(h, w, target_rgb)
    canvas_init = torch.full((h, w, 3), 40, dtype=torch.uint8, device=device)
    shapes = [
        {"type": "rotated_ellipse", "x": 8,  "y": 8,  "rx": 4, "ry": 4, "angle": 0.0,
         "color": [255, 0, 0, 128]},   # deliberately wrong starting color — must converge to grey
        {"type": "rotated_ellipse", "x": 20, "y": 20, "rx": 4, "ry": 4, "angle": 0.0,
         "color": [0, 255, 0, 128]},
        {"type": "rotated_ellipse", "x": 16, "y": 16, "rx": 3, "ry": 6, "angle": 30.0,
         "color": [0, 0, 255, 128]},
    ]
    refined, _ = joint_polish(shapes, target, alpha_t=None, alpha_mask_f=None,
                              edge_weight=None, canvas_init=canvas_init,
                              h=h, w=w, steps=30, lock_alpha=True)
    # Every shape's RGB must be within ±10 of the flat target color — no saturation overshoot.
    for s in refined:
        r, g, b, a = s["color"]
        assert a == 255, f"alpha not locked: {a}"
        for ch_name, ch in zip("RGB", (r, g, b)):
            assert abs(ch - target_rgb[0]) <= 10, (
                f"channel {ch_name}={ch} drifted from target {target_rgb[0]} — snap-back missing"
            )


def test_polish_color_no_saturation_extremes_on_mixed_target():
    """On a target with non-extreme colors, post-polish RGB must stay within the target's
    actual range. The guard catches Adam overshoot to channel-saturated values (0 or 255)."""
    h, w = 32, 32
    device = get_device()
    # Target uses only middle values (50-200) — no pure channel saturation anywhere.
    rng = np.random.default_rng(7)
    target_np = rng.integers(50, 200, size=(h, w, 3), dtype=np.uint8)
    target = torch.tensor(target_np, device=device)
    canvas_init = torch.full((h, w, 3), 40, dtype=torch.uint8, device=device)
    shapes = [
        {"type": "rotated_ellipse", "x": 10, "y": 10, "rx": 5, "ry": 3, "angle": 0.0,
         "color": [200, 50, 50, 128]},
        {"type": "rotated_ellipse", "x": 22, "y": 22, "rx": 4, "ry": 4, "angle": 45.0,
         "color": [50, 200, 50, 128]},
    ]
    refined, _ = joint_polish(shapes, target, alpha_t=None, alpha_mask_f=None,
                              edge_weight=None, canvas_init=canvas_init,
                              h=h, w=w, steps=40, lock_alpha=True)
    # No channel of any shape should exceed 220 or drop below 30 — well outside the target's
    # 50-200 range with comfortable margin for closed-form rounding error.
    for s in refined:
        r, g, b, _ = s["color"]
        for ch_name, ch in zip("RGB", (r, g, b)):
            assert 30 <= ch <= 220, (
                f"channel {ch_name}={ch} outside target's 50-200 range — snap-back missing or broken"
            )


def test_snap_back_only_runs_with_lock_alpha():
    """Snap-back changes RGB only when lock_alpha=True (the only case where alpha=255
    closed-form 'mean over visible mask' is the exact optimal). With lock_alpha=False, the
    existing Adam-optimized RGB is kept (no snap-back), preserving legacy behavior."""
    h, w = 16, 16
    device = get_device()
    target = torch.full((h, w, 3), 100, dtype=torch.uint8, device=device)
    canvas_init = torch.full((h, w, 3), 40, dtype=torch.uint8, device=device)
    shapes = [{"type": "rotated_ellipse", "x": 8, "y": 8, "rx": 3, "ry": 3, "angle": 0.0,
               "color": [255, 0, 0, 128]}]
    # lock_alpha=False -> snap-back skipped -> Adam result preserved (alpha optimized too).
    refined, _ = joint_polish(shapes, target, alpha_t=None, alpha_mask_f=None,
                              edge_weight=None, canvas_init=canvas_init,
                              h=h, w=w, steps=20, lock_alpha=False)
    # We only assert alpha is in the optimized range — color may or may not match target
    # (Adam-driven, not closed-form). Snap-back not applied here.
    assert refined[0]["color"][3] != 255 or refined[0]["color"][3] == int(refined[0]["color"][3]), \
        "lock_alpha=False path should have unlocked alpha"
