"""Joint-polish pass: diffvg-style global refinement of all shapes at once.

Greedy generation (geometrize and ours alike) freezes each shape the moment it's placed, so
early shapes are wrong in hindsight and later shapes waste budget compensating. This pass
attacks that ceiling: after greedy placement, it makes the *whole* composite differentiable
and gradient-optimizes ALL shapes' geometry + color + alpha simultaneously against the full
target. One optimization run fixes errors that greedy would need many more shapes to paper
over — more quality per shape, without geometrize's brute-force sample counts.

Memory: N sequential soft-composites would store N intermediate canvases for backward. We
chunk the composite and gradient-checkpoint each chunk (recompute its forward during
backward), so peak memory is O(num_chunks + chunk_size) canvases, not O(N).

Ellipse-only (matches the shippable shape set).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.utils.checkpoint as cp

from forza_abyss_painter.shapegen.gpu.device import DTYPE
from forza_abyss_painter.shapegen.gpu.scoring import ALPHA_FIXED
from forza_abyss_painter.shapegen.gpu.shapes_gpu import _ellipse_soft
from forza_abyss_painter.shapegen.gpu.rasterize import rasterize_rotated_ellipses

_ALPHA_MIN = 16.0   # don't let opacities collapse to invisible during optimization


def _composite_chunk(canvas, geom_c, rgb_c, alpha_c, h, w):
    """Soft-composite a chunk of shapes (in order) onto `canvas`. Differentiable.
    geom_c (C,5), rgb_c (C,3) in 0..255, alpha_c (C,) in 0..255.

    No silhouette clipping: shapes paint their full mask. The in-game and CPU renderers have
    no source alpha mask to clip against, so we don't either — the optimizer trains against
    what the game will actually display. Spill penalty is encoded into the target instead
    (engine pre-fills out-of-silhouette target pixels with the canvas substrate color, so
    paint that lands there shows up as positive loss against an otherwise-zero baseline).
    """
    masks = _ellipse_soft(geom_c, h, w)          # (C, H, W), soft/differentiable
    C = geom_c.shape[0]
    for i in range(C):
        m = masks[i]
        a = (alpha_c[i] / 255.0).clamp(0.0, 1.0)
        src = rgb_c[i].clamp(0.0, 255.0).view(1, 1, 3)
        mm = m.unsqueeze(-1)
        canvas = mm * (a * src + (1.0 - a) * canvas) + (1.0 - mm) * canvas
    return canvas


def _purity_loss(geom, tgt, h, w, chunk):
    """Per-shape MSE-equivalent spillover loss, summed over shapes, normalized by canvas pixels.

    For each shape i with soft mask m_i (H, W) and target t (H, W, 3):
        mass_i        = sum(m_i)
        mean_i        = sum(m_i * t) / mass_i                        (3,)
        spillover_i   = sum( m_i * (t - mean_i).pow(2) ).sum()       scalar (RGB-summed)
        # spillover_i ≡ the MSE the canvas would carry over shape i's mask if i painted
        # with its optimal mean color — equal to mass_i × per-pixel-variance.

    Returns: sum_i(spillover_i) / (H * W * 3), in the same MSE-equivalent units as the
    main loss above. This is the load-bearing change vs the original PR #19 formulation
    (which returned per-shape AVERAGE variance — mass-independent — so a tiny lazy shape
    and a huge lazy shape paid the same penalty while their MSE gradients differed by
    orders of magnitude). The mass-weighted form makes big lazy shapes pay proportionally
    more, which is exactly what we want: the optimizer's incentive to grow a shape (more
    pixels of MSE benefit) is now balanced by a proportionally-growing penalty.

    Intuition for tuning purity_penalty:
      = 1.0   → lazy paint penalty equals its MSE saving; lazy and no-paint break even
      < 1.0   → painting still preferred to no-paint, but homogeneity strongly encouraged
      > 1.0   → no-paint strictly preferred to lazy paint over the same multi-color region

    Chunked so we never materialize all N masks at once at production shape counts.
    Fully differentiable wrt geom.
    """
    N = geom.shape[0]
    if N == 0:
        return torch.zeros((), dtype=DTYPE, device=geom.device)
    spillover_sum = torch.zeros((), dtype=DTYPE, device=geom.device)
    for lo in range(0, N, chunk):
        hi = min(lo + chunk, N)
        masks = _ellipse_soft(geom[lo:hi], h, w)              # (C, H, W)
        mass = masks.sum(dim=(1, 2)).clamp(min=1e-6)          # (C,)
        weighted = (masks.unsqueeze(-1) * tgt.unsqueeze(0)).sum(dim=(1, 2))   # (C, 3)
        mean_color = weighted / mass.unsqueeze(-1)            # (C, 3)
        diff = tgt.unsqueeze(0) - mean_color.view(-1, 1, 1, 3)               # (C, H, W, 3)
        # Mass-weighted (NOT mass-normalized) sum of squared deviations per shape.
        # This is `mass * variance_per_pixel` for each shape, RGB-summed.
        spillover_per_shape = (masks.unsqueeze(-1) * diff.pow(2)).sum(dim=(1, 2, 3))  # (C,)
        spillover_sum = spillover_sum + spillover_per_shape.sum()
    return spillover_sum / float(h * w * 3)


def _forward_composite(canvas0, geom, rgb, alpha, h, w, chunk, use_ckpt):
    canvas = canvas0
    N = geom.shape[0]
    for lo in range(0, N, chunk):
        hi = min(lo + chunk, N)
        g, c, a = geom[lo:hi], rgb[lo:hi], alpha[lo:hi]
        if use_ckpt and torch.is_grad_enabled():
            canvas = cp.checkpoint(_composite_chunk, canvas, g, c, a,
                                   h, w, use_reentrant=False)
        else:
            canvas = _composite_chunk(canvas, g, c, a, h, w)
    return canvas


def _hard_render(canvas0, geom, rgb, alpha, h, w):
    """Final crisp render with hard ellipse edges. UNCLIPPED — matches what the in-game and
    CPU exe renderers produce for the emitted JSON. (Was previously clipping by alpha_mask_f
    in sticker mode, which produced a clean preview that diverged from the actual game render
    and hid out-of-silhouette spillover. See _composite_chunk docstring.)"""
    canvas = canvas0.clone()
    N = geom.shape[0]
    for i in range(N):
        mask = rasterize_rotated_ellipses(geom[i:i + 1], h, w)[0]
        a = float(alpha[i].clamp(0.0, 255.0).item()) / 255.0
        src = rgb[i].clamp(0.0, 255.0).view(1, 1, 3)
        m = mask.unsqueeze(-1)
        canvas = m * (a * src + (1.0 - a) * canvas) + (1.0 - m) * canvas
    return canvas


def _resolve_rgb_closed_form(geom, target, alpha_mask_f, h, w):
    """Closed-form post-polish RGB resolution for lock_alpha=True.

    For each shape i (in commit order), its actually-visible pixels are:
        mask_i AND (opaque substrate, if sticker) AND (no later shape covers it)

    Since alpha=255 fully overwrites within mask_i, shape_i's contribution to the rendered
    image is exactly the pixels it owns at the top of the z-stack. The closed-form optimal
    color (minimizing per-pixel SSE against the target there) is the simple mean of the
    target's RGB over those visible pixels.

    Walks shapes in REVERSE order so we can accumulate the "claimed" union from front to
    back. Fully-occluded shapes (mask entirely covered by later layers) fall back to the
    mean over their full mask — preserves whatever color they had if they ever get
    un-occluded by future edits.
    """
    n = geom.shape[0]
    device = geom.device
    target_f = target.to(DTYPE)   # (H, W, 3)

    new_rgb = torch.zeros((n, 3), dtype=DTYPE, device=device)
    claimed = torch.zeros((h, w), dtype=torch.bool, device=device)
    opaque = (alpha_mask_f > 0) if alpha_mask_f is not None else None

    fallback_color = torch.tensor([128.0, 128.0, 128.0], dtype=DTYPE, device=device)
    for i in reversed(range(n)):
        mask = rasterize_rotated_ellipses(geom[i:i + 1], h, w)[0]   # (H, W) {0, 1}
        m_bool = mask > 0
        visible = m_bool & ~claimed
        if opaque is not None:
            visible = visible & opaque
        if visible.any():
            new_rgb[i] = target_f[visible].mean(dim=0)
        elif m_bool.any():
            # Fully occluded — keep a sane in-distribution color via the all-mask mean
            # (still beats Adam saturation extremes).
            new_rgb[i] = target_f[m_bool].mean(dim=0)
        else:
            new_rgb[i] = fallback_color
        claimed |= m_bool
    return new_rgb


def joint_polish(shapes_json, target, alpha_t, alpha_mask_f, edge_weight, canvas_init,
                 h, w, steps, lr=1.5, chunk=200, progress=False, lock_alpha=False,
                 purity_penalty=0.0, freeze_geometry=True):
    """Jointly refine all shapes. Returns (refined_shapes_json, final_canvas_u8_numpy).

    target/canvas_init: (H,W,3) uint8 device tensors (target already posterized + sticker-
    zeroed by the caller; canvas_init is the same init the greedy run used).
    alpha_t (H,W uint8) / alpha_mask_f (H,W float) for sticker; edge_weight (H,W) or None.

    purity_penalty (float, default 0.0): when > 0 AND freeze_geometry is False, adds a
    per-shape MSE-equivalent 'spillover' loss (mass × per-pixel variance, summed over
    shapes, normalized by canvas pixel count — same units as the main MSE). At
    purity_penalty=1.0, lazy paint cancels its MSE saving (break-even with omission).
    Empirically the penalty creates a degenerate exploit (Adam collapses shapes to rx=1
    to escape the penalty), so freeze_geometry=True is the recommended polish mode.

    freeze_geometry (bool, default False): when True, Adam optimizes (rgb, alpha) ONLY —
    geometry (x, y, rx, ry, angle) stays bit-identical to the input shapes_json. This is
    the recommended polish mode: it keeps the color refinement + snap-back win without
    giving Adam the geometry handles it consistently mis-uses (inflate, collapse, drift).
    purity_penalty is a no-op when freeze_geometry=True (it's a geometry-affecting term).
    """
    device = target.device
    if not shapes_json:
        return shapes_json, _hard_render(canvas_init.to(DTYPE), torch.zeros((0, 5), device=device),
                                         torch.zeros((0, 3), device=device),
                                         torch.zeros((0,), device=device),
                                         h, w).clamp(0, 255).round().to(torch.uint8).cpu().numpy()

    geom = torch.tensor([[s["x"], s["y"], s["rx"], s["ry"], s["angle"]] for s in shapes_json],
                        dtype=DTYPE, device=device)
    rgb = torch.tensor([s["color"][:3] for s in shapes_json], dtype=DTYPE, device=device)
    alpha = torch.tensor([s["color"][3] for s in shapes_json], dtype=DTYPE, device=device)

    if not freeze_geometry:
        geom.requires_grad_(True)
    rgb.requires_grad_(True)
    if lock_alpha:
        alpha.fill_(255.0)   # force the locked value even if input had alpha < 255
    else:
        alpha.requires_grad_(True)
    canvas0 = canvas_init.to(DTYPE)
    tgt = target.to(DTYPE)

    # Per-pixel loss weight: edge emphasis only. NO opaque (alpha_t) gate — that previously
    # zeroed out-of-silhouette pixels in the loss, which prevented the optimizer from feeling
    # spillover past the silhouette. The caller pre-fills the target with the canvas substrate
    # color outside the silhouette (see engine.py:341), so out-of-mask paint is now naturally
    # penalized by plain MSE without any gating. lw stays None unless edge_weight is provided.
    lw = edge_weight
    if lw is not None:
        lw3 = lw.unsqueeze(-1)
        denom = (lw.sum() * 3.0).clamp(min=1.0)

    opt_groups = [{"params": [rgb], "lr": lr * 4.0}]   # colors always optimized
    if not freeze_geometry:
        opt_groups.append({"params": [geom], "lr": lr})
    if not lock_alpha:
        opt_groups.append({"params": [alpha], "lr": lr * 4.0})
    opt = torch.optim.Adam(opt_groups)

    init_loss = None
    for step in range(steps):
        opt.zero_grad()
        canvas = _forward_composite(canvas0, geom, rgb, alpha, h, w,
                                    chunk, use_ckpt=True)
        diff = canvas - tgt
        if lw is not None:
            loss = (diff * diff * lw3).sum() / denom
        else:
            loss = (diff * diff).mean()
        # purity_penalty is a GEOMETRY-affecting penalty (the only way to reduce it is to
        # change shape masks, i.e. geometry). When freeze_geometry is on, geom has no grad,
        # so the penalty would only inflate the loss without producing any update — skip it.
        if purity_penalty > 0.0 and not freeze_geometry:
            loss = loss + purity_penalty * _purity_loss(geom, tgt, h, w, chunk)
        loss.backward()
        opt.step()
        with torch.no_grad():
            if not freeze_geometry:
                geom[:, 0].clamp_(0.0, w - 1)
                geom[:, 1].clamp_(0.0, h - 1)
                geom[:, 2].clamp_(1.0, float(w))
                geom[:, 3].clamp_(1.0, float(h))
            rgb.clamp_(0.0, 255.0)
            if not lock_alpha:
                alpha.clamp_(_ALPHA_MIN, 255.0)
        if init_loss is None:
            init_loss = float(loss.detach().cpu())
        if progress and (step + 1) % max(1, steps // 5) == 0:
            print(f"  joint-polish step {step+1}/{steps}  loss {float(loss.detach().cpu()):.2f}")

    geom_f = geom.detach(); rgb_f = rgb.detach(); alpha_f = alpha.detach()
    with torch.no_grad():
        # Color snap-back. Adam runs RGB at lr*4 across `steps` updates with no chromatic
        # regularization — momentum routinely pushes channels to saturation extremes that the
        # underlying target image doesn't contain. (Observed: ~125/3000 shapes hitting
        # min<15 AND max>240 in 150-step polish, with ~45 turning fully green/yellow when
        # the optimizer overshot from sparkle-yellow target regions.) For each polished
        # shape, recompute the closed-form-optimal opaque color as the MEAN of the target
        # image over the shape's actually-visible mask region (mask AND opaque substrate
        # AND not yet claimed by a later layer in z-order). This is the exact value Adam was
        # approximating and it stays in the target's color distribution by construction.
        if lock_alpha:
            rgb_f = _resolve_rgb_closed_form(geom_f, target.detach(), alpha_mask_f, h, w)
        final_canvas = _hard_render(canvas0, geom_f, rgb_f, alpha_f, h, w)
        final_canvas = final_canvas.clamp(0, 255).round().to(torch.uint8).cpu().numpy()

    g = geom_f.cpu().tolist(); c = rgb_f.round().cpu().tolist(); al = alpha_f.round().cpu().tolist()
    refined = [{
        "type": "rotated_ellipse",
        "x": round(float(g[i][0]), 3), "y": round(float(g[i][1]), 3),
        "rx": round(float(g[i][2]), 3), "ry": round(float(g[i][3]), 3),
        "angle": round(float(g[i][4]) % 180.0, 3),
        "color": [int(c[i][0]), int(c[i][1]), int(c[i][2]),
                  255 if lock_alpha else int(al[i])],
    } for i in range(len(shapes_json))]
    return refined, final_canvas
