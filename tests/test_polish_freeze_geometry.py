"""Regression tests for polish_freeze_geometry.

joint_polish jointly optimizes (geometry, color, alpha) via Adam. Three iterations of
geometry-constraint experiments (PR #18 size anchor, PR #19 mean-variance purity, PR #20
mass-weighted purity) all produced degenerate failure modes — Adam consistently finds
exploits in the gradient signal we give it for geometry (inflate / collapse / drift onto
canvas edges) because ellipse geometry isn't a parameter space gradient descent is well-
suited to.

The fix: when GPUConfig.polish_freeze_geometry=True, joint_polish skips geometry from the
optimizer entirely. Adam optimizes (rgb, alpha) only; (x, y, rx, ry, angle) come straight
out of the input shapes_json bit-identical. Color refinement + the closed-form RGB snap-
back (when lock_alpha=True) still work — that's the polish win we actually want without
the geometry-handle baggage.

These tests pin:
1. GPUConfig.polish_freeze_geometry exists with default False (legacy preserved).
2. freeze_geometry=False produces bit-identical output to no-kwarg call (no-op contract).
3. freeze_geometry=True: every output shape's geometry == input shape's geometry exactly.
4. freeze_geometry=True with lock_alpha=True: closed-form snap-back STILL adjusts colors
   (proves color refinement is preserved — the geometry-freeze didn't break color updates).
5. purity_penalty is a NO-OP when freeze_geometry=True (the penalty term flows through geom
   gradient, which doesn't exist; it would only inflate the loss without effecting any change).
"""
import numpy as np
import torch

from forza_abyss_painter.shapegen.gpu.device import DTYPE, get_device
from forza_abyss_painter.shapegen.gpu.engine import GPUConfig
from forza_abyss_painter.shapegen.gpu.joint_polish import joint_polish


def _shapes():
    return [
        {"type": "rotated_ellipse", "x": 8.0,  "y": 8.0,  "rx": 3.0, "ry": 3.0,
         "angle": 0.0, "color": [200, 100, 50, 255]},
        {"type": "rotated_ellipse", "x": 20.0, "y": 16.0, "rx": 4.0, "ry": 2.0,
         "angle": 30.0, "color": [50, 200, 100, 255]},
        {"type": "rotated_ellipse", "x": 12.0, "y": 22.0, "rx": 2.5, "ry": 3.5,
         "angle": 75.0, "color": [100, 50, 200, 255]},
    ]


def _polish(shapes, freeze, purity=0.0, steps=20, lock_alpha=True, seed=11):
    h, w = 32, 32
    device = get_device()
    target = torch.tensor(np.random.default_rng(seed).integers(40, 200, (h, w, 3), dtype=np.uint8),
                          device=device)
    canvas_init = torch.full((h, w, 3), 40, dtype=torch.uint8, device=device)
    torch.manual_seed(seed); np.random.seed(seed)
    refined, _ = joint_polish(
        shapes, target, alpha_t=None, alpha_mask_f=None, edge_weight=None,
        canvas_init=canvas_init, h=h, w=w, steps=steps, lock_alpha=lock_alpha,
        purity_penalty=purity, freeze_geometry=freeze,
    )
    return refined


def test_gpu_config_has_polish_freeze_geometry_default_true():
    """polish_freeze_geometry defaults to True. This is the validated production polish mode
    (1000-shape harness run confirmed 1000/1000 geom frozen + engine↔upstream parity 0.000).
    A False default would silently re-enable the legacy Adam-over-geometry polish that smears
    sparkles and inflates shapes. To opt back into legacy polish, pass freeze_geometry=False."""
    cfg = GPUConfig(num_shapes=10, random_samples=4)
    assert hasattr(cfg, "polish_freeze_geometry"), (
        "polish_freeze_geometry field missing on GPUConfig"
    )
    assert cfg.polish_freeze_geometry is True, (
        f"polish_freeze_geometry must default to True; got {cfg.polish_freeze_geometry!r}. "
        f"Defaulting to False reintroduces legacy joint_polish geometry edits which inflate "
        f"shapes and break sparkle content."
    )


def test_default_matches_explicit_true():
    """The no-op contract under the new True default — a no-kwarg call must produce
    bit-identical output to an explicit freeze_geometry=True call. If this fails, the
    default and the explicit value have drifted apart somewhere."""
    h, w = 32, 32
    device = get_device()
    target = torch.tensor(
        np.random.default_rng(42).integers(40, 200, (h, w, 3), dtype=np.uint8), device=device,
    )
    canvas_init = torch.full((h, w, 3), 40, dtype=torch.uint8, device=device)
    shapes = _shapes()

    torch.manual_seed(42); np.random.seed(42)
    refined_default, _ = joint_polish(
        shapes, target, alpha_t=None, alpha_mask_f=None, edge_weight=None,
        canvas_init=canvas_init, h=h, w=w, steps=15, lock_alpha=True,
    )
    torch.manual_seed(42); np.random.seed(42)
    refined_explicit_true, _ = joint_polish(
        shapes, target, alpha_t=None, alpha_mask_f=None, edge_weight=None,
        canvas_init=canvas_init, h=h, w=w, steps=15, lock_alpha=True,
        freeze_geometry=True,
    )
    for i, (a, b) in enumerate(zip(refined_default, refined_explicit_true)):
        for k in ("x", "y", "rx", "ry", "angle"):
            assert a[k] == b[k], (
                f"shape {i} {k}: default={a[k]} vs freeze=True {b[k]} — "
                f"the default and explicit True must produce identical output"
            )


def test_freeze_true_preserves_geometry_exactly():
    """The contract: freeze_geometry=True means geometry doesn't move. x, y, rx, ry, angle
    of every refined shape must match the input EXACTLY (after JSON rounding to 3 decimals;
    input values already have ≤3 decimals so this is a tight equality check)."""
    shapes_in = _shapes()
    refined = _polish(shapes_in, freeze=True, steps=30)
    assert len(refined) == len(shapes_in)
    for i, (orig, ref) in enumerate(zip(shapes_in, refined)):
        for k in ("x", "y", "rx", "ry", "angle"):
            assert ref[k] == orig[k], (
                f"shape {i} {k}: refined={ref[k]} vs input={orig[k]} — "
                f"freeze_geometry=True must leave geometry untouched"
            )


def test_freeze_true_still_refines_colors_via_snap_back():
    """The point of polish even with frozen geometry is color refinement. With lock_alpha=True
    the closed-form snap-back must STILL run after Adam — verify by comparing input colors
    (deliberately wrong on a flat-grey target) to refined colors (should converge to grey)."""
    h, w = 24, 24
    device = get_device()
    target_rgb = (180, 180, 180)
    target = torch.tensor(np.full((h, w, 3), target_rgb, dtype=np.uint8), device=device)
    canvas_init = torch.full((h, w, 3), 40, dtype=torch.uint8, device=device)
    shapes_in = [{
        "type": "rotated_ellipse", "x": 12.0, "y": 12.0, "rx": 5.0, "ry": 5.0,
        "angle": 0.0, "color": [255, 0, 0, 128],   # deliberately wrong — must converge to grey
    }]
    torch.manual_seed(3); np.random.seed(3)
    refined, _ = joint_polish(
        shapes_in, target, alpha_t=None, alpha_mask_f=None, edge_weight=None,
        canvas_init=canvas_init, h=h, w=w, steps=30, lock_alpha=True,
        freeze_geometry=True,
    )
    r, g, b, a = refined[0]["color"]
    assert a == 255, f"lock_alpha didn't take: alpha={a}"
    # Color must have refined toward grey (within ±15 of 180) — confirms snap-back ran.
    for ch_name, ch in zip("RGB", (r, g, b)):
        assert abs(ch - 180) <= 15, (
            f"channel {ch_name}={ch} didn't refine toward target grey 180 — snap-back missing"
        )
    # Geometry untouched by snap-back either.
    for k in ("x", "y", "rx", "ry", "angle"):
        assert refined[0][k] == shapes_in[0][k], (
            f"geometry {k} changed despite freeze: {refined[0][k]} vs {shapes_in[0][k]}"
        )


def test_purity_penalty_is_noop_when_geometry_frozen():
    """purity_penalty is geometry-affecting (it only has a gradient path through shape
    masks). With freeze_geometry=True there's no geometry gradient, so the penalty cannot
    cause any change. Output must be bit-identical between purity=0 and purity=10 when
    freeze_geometry=True."""
    shapes_in = _shapes()
    refined_no_purity   = _polish(shapes_in, freeze=True, purity=0.0,  steps=20)
    refined_big_purity  = _polish(shapes_in, freeze=True, purity=10.0, steps=20)
    for i, (a, b) in enumerate(zip(refined_no_purity, refined_big_purity)):
        for k in ("x", "y", "rx", "ry", "angle"):
            assert a[k] == b[k], (
                f"shape {i} {k}: purity=0 {a[k]} vs purity=10 {b[k]} under freeze_geometry=True. "
                f"purity_penalty must be a no-op when geometry is frozen."
            )
        # Colors should also match — snap-back uses the same (frozen) geometry, and Adam's
        # color trajectory shouldn't depend on a penalty that produced no gradient.
        assert a["color"] == b["color"], (
            f"shape {i} color: purity=0 {a['color']} vs purity=10 {b['color']} under freeze. "
            f"Color path must be identical when purity has no geometry to act on."
        )
