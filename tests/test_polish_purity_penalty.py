"""Regression tests for the polish purity penalty.

Plain MSE actively rewards 'lazy averaging': a big ellipse spanning a multi-color region
with an averaged color produces lower MSE than leaving the region substrate-colored. On a
5x5 yellow star on dark background, the lazy 10x10 averaged ellipse beats no-shape by
~23% in MSE — which is why joint_polish smeared sparkles into blobs on celeste-with-stars
even with the size anchor active (PR #18 — now removed; this penalty subsumes it).

The fix: GPUConfig.polish_purity_penalty. For each shape, compute the RGB-summed weighted
variance of TARGET colors under the shape's soft mask. Homogeneous coverage → variance ~0 →
no penalty. Multi-color coverage (the lazy averaging case) → high variance → big penalty.

These tests pin:
1. The new field exists with default 0.0 (legacy behavior preserved).
2. polish_size_anchor was removed (the previous mistake).
3. purity_penalty=0 produces identical geometry to a no-kwarg call (no-op contract).
4. purity_penalty>0 measurably reduces a shape's drift INTO multi-color regions.
5. A shape sitting on a homogeneous region (purity loss == 0) is unaffected by the penalty.
"""
import numpy as np
import torch

from forza_abyss_painter.shapegen.gpu.device import DTYPE, get_device
from forza_abyss_painter.shapegen.gpu.engine import GPUConfig
from forza_abyss_painter.shapegen.gpu.joint_polish import joint_polish, _purity_loss


def _polish_run(shapes, target, canvas_init, h, w, purity_penalty, steps, seed=11):
    """Helper: run polish with the legacy joint-geometry mode (freeze_geometry=False) so
    the purity_penalty has a geometry gradient to act on. The PRODUCTION default flipped
    to freeze_geometry=True in a later PR — when frozen, purity is a no-op, so these
    purity-mechanism tests must explicitly opt INTO the legacy mode."""
    torch.manual_seed(seed); np.random.seed(seed)
    refined, _ = joint_polish(
        shapes, target, alpha_t=None, alpha_mask_f=None, edge_weight=None,
        canvas_init=canvas_init, h=h, w=w, steps=steps, lock_alpha=True,
        purity_penalty=purity_penalty, freeze_geometry=False,
    )
    return refined


def test_gpu_config_has_polish_purity_penalty_default_zero():
    """The new field exists with default 0.0 — legacy behavior preserved for callers who
    don't opt in."""
    cfg = GPUConfig(num_shapes=10, random_samples=4)
    assert hasattr(cfg, "polish_purity_penalty"), (
        "polish_purity_penalty field missing on GPUConfig"
    )
    assert cfg.polish_purity_penalty == 0.0, (
        f"polish_purity_penalty must default to 0.0; got {cfg.polish_purity_penalty}. "
        f"Changing the default silently shifts every production notebook's output and "
        f"violates the 'no silent baseline changes' contract."
    )


def test_polish_size_anchor_field_was_removed():
    """polish_size_anchor (PR #18) was the wrong fix — it treated the symptom (rx growth)
    instead of the cause (loss rewards averaging). It's gone. If anyone reintroduces it,
    they're papering over the wrong problem."""
    cfg = GPUConfig(num_shapes=10, random_samples=4)
    assert not hasattr(cfg, "polish_size_anchor"), (
        "polish_size_anchor must NOT be on GPUConfig — it was removed and replaced by "
        "polish_purity_penalty which subsumes its purpose more correctly."
    )


def test_purity_zero_preserves_legacy_behavior():
    """purity_penalty=0.0 must produce IDENTICAL geometry to a no-kwarg call. The
    no-op contract — every caller that doesn't opt in sees no behavior change."""
    h, w = 32, 32
    device = get_device()
    target = torch.full((h, w, 3), 128, dtype=torch.uint8, device=device)
    canvas_init = torch.full((h, w, 3), 40, dtype=torch.uint8, device=device)
    shapes = [
        {"type": "rotated_ellipse", "x": 8.0,  "y": 8.0,  "rx": 3.0, "ry": 3.0,
         "angle": 0.0, "color": [200, 100, 50, 255]},
        {"type": "rotated_ellipse", "x": 20.0, "y": 16.0, "rx": 4.0, "ry": 2.0,
         "angle": 30.0, "color": [50, 200, 100, 255]},
    ]
    torch.manual_seed(42); np.random.seed(42)
    refined_default, _ = joint_polish(
        shapes, target, alpha_t=None, alpha_mask_f=None, edge_weight=None,
        canvas_init=canvas_init, h=h, w=w, steps=15, lock_alpha=True,
    )
    torch.manual_seed(42); np.random.seed(42)
    refined_explicit_zero, _ = joint_polish(
        shapes, target, alpha_t=None, alpha_mask_f=None, edge_weight=None,
        canvas_init=canvas_init, h=h, w=w, steps=15, lock_alpha=True,
        purity_penalty=0.0,
    )
    for i, (a, b) in enumerate(zip(refined_default, refined_explicit_zero)):
        for k in ("x", "y", "rx", "ry", "angle"):
            assert a[k] == b[k], (
                f"shape {i} {k}: default={a[k]} vs purity=0 {b[k]} — purity=0 must be a no-op"
            )


def test_purity_loss_is_zero_on_homogeneous_target():
    """If the target is a flat single color, every shape's mask covers identical-colored
    pixels → variance is zero → purity loss is zero regardless of geometry. This is the
    'no penalty for shapes on homogeneous regions' guarantee."""
    h, w = 24, 24
    device = get_device()
    tgt = torch.full((h, w, 3), 150, dtype=DTYPE, device=device)
    geom = torch.tensor([
        [12.0, 12.0, 5.0, 5.0, 0.0],
        [6.0,  18.0, 3.0, 4.0, 30.0],
    ], dtype=DTYPE, device=device)
    loss = _purity_loss(geom, tgt, h, w, chunk=10)
    assert float(loss) < 1e-3, (
        f"_purity_loss on flat target should be ~0; got {float(loss)}. A shape covering "
        f"only one color must contribute no penalty."
    )


def test_purity_loss_is_positive_on_multicolor_target():
    """A shape's mask spanning two distinct colors must give a positive purity loss — that's
    the penalty the optimizer feels. Magnitude should be roughly the squared color delta."""
    h, w = 24, 24
    device = get_device()
    # Half black, half white target — any centered shape spans both halves.
    tgt = torch.zeros((h, w, 3), dtype=DTYPE, device=device)
    tgt[:, w // 2:] = 200.0
    geom = torch.tensor([[12.0, 12.0, 8.0, 6.0, 0.0]], dtype=DTYPE, device=device)
    loss = _purity_loss(geom, tgt, h, w, chunk=10)
    # Rough magnitude: variance per channel ~ (200/2)^2 = 10000 since it's a two-cluster
    # distribution at 0 and 200. RGB-summed: ~30000. Don't pin tightly — just assert it's
    # in the right order of magnitude.
    assert float(loss) > 1000.0, (
        f"_purity_loss on a half/half target should be large (~10000s); got {float(loss)}. "
        f"If this is near 0, the penalty isn't seeing the multi-color content."
    )


def test_purity_penalty_pushes_shape_away_from_color_boundary():
    """The whole point: a shape positioned ON a color boundary feels gradient pulling it
    INTO one of the homogeneous regions (variance drops there). Compare polish runs with
    purity_penalty=0 vs purity_penalty>0 on a shape sitting on a yellow/dark boundary —
    the purity-penalized run must end with the shape MORE centered on yellow (or more
    centered on dark), measurably reducing the target-variance under its mask."""
    h, w = 40, 40
    device = get_device()
    target_np = np.full((h, w, 3), 20, dtype=np.uint8)
    # Yellow blob in the top-left quadrant.
    target_np[8:18, 8:18] = (240, 240, 80)
    target = torch.tensor(target_np, device=device)
    canvas_init = torch.full((h, w, 3), 20, dtype=torch.uint8, device=device)
    # Shape straddles the yellow/dark boundary at (13, 18) — half on yellow, half on dark,
    # with a wide rx so it spans both regions.
    init_shapes = [{
        "type": "rotated_ellipse", "x": 13.0, "y": 18.0,
        "rx": 8.0, "ry": 4.0, "angle": 0.0, "color": [130, 130, 50, 255],
    }]
    refined_no_penalty = _polish_run(init_shapes, target, canvas_init, h, w,
                                     purity_penalty=0.0, steps=120)
    refined_penalty    = _polish_run(init_shapes, target, canvas_init, h, w,
                                     purity_penalty=100.0, steps=120)

    # Compute target-variance under each final shape's mask as a proxy for 'how multi-color
    # is this shape covering'. The purity-penalized run must score lower.
    def _final_var(s):
        geom_t = torch.tensor([[s["x"], s["y"], s["rx"], s["ry"], s["angle"]]],
                              dtype=DTYPE, device=device)
        return float(_purity_loss(geom_t, target.to(DTYPE), h, w, chunk=1))

    v_free = _final_var(refined_no_penalty[0])
    v_pen  = _final_var(refined_penalty[0])
    assert v_pen < v_free * 0.85, (
        f"purity penalty didn't reduce per-shape target-variance: free_var={v_free:.1f}, "
        f"penalty_var={v_pen:.1f}. Expected at least 15%% reduction. If this fails, the "
        f"penalty isn't actually shaping the optimization landscape as intended."
    )
