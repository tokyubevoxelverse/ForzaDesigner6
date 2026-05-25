"""Post-greedy shape-budget reclamation.

THE PROBLEM: greedy commits N shapes one at a time, each one optimal at its commit
moment. Later commits can fully occlude earlier ones (especially in regions where the
target has fine detail that takes multiple shapes to capture). At production budgets
(1000-3000 shapes) we measure ~5% of commits ending up fully-occluded — pure dead
weight in the final JSON. The user asked for N shapes; the engine delivers N commits
but only ~0.95 N actually contribute pixels to the final canvas.

THE FIX (option B from the design discussion):
After greedy + polish + snap-back completes, run a focused cleanup-and-refill pass:

  1. Detect dead shapes (visible-pixel count below threshold under z-order)
  2. Drop them from the shape list
  3. Re-render the canvas from the surviving live shapes
  4. Mini-greedy: commit N_dead NEW shapes targeting the current residual (the parts
     of the canvas that diverge from the target). Same bbox-local + gradient-refine
     logic as the main greedy loop, just bounded to N_dead iterations and starting
     from the post-cleanup canvas state.
  5. Closed-form RGB snap-back over the full final shape set (existing + new), so
     each shape's color is the mean of target over its visible region in the new
     occlusion structure.
  6. Re-render final canvas via _hard_render.

User asks for 3000; gets ~3000 LIVE shapes that all contribute pixels to the final
render. The 5% waste that was happening invisibly in the JSON gets re-allocated to
useful work in high-residual regions.

This is essentially the polish_existing pipeline from earlier sessions, but cleaner
and integrated as a default-on stage in run_gpu rather than a separate notebook flow.
"""
from __future__ import annotations

import torch

from forza_abyss_painter.shapegen.gpu.device import DTYPE
from forza_abyss_painter.shapegen.gpu.rasterize import rasterize_rotated_ellipses
from forza_abyss_painter.shapegen.gpu.bbox_score import (
    crop_score_ellipse_batch, crop_score_ellipse_batch_chunked,
)
from forza_abyss_painter.shapegen.gpu.engine import _resolve_k_chunk_size
from forza_abyss_painter.shapegen.gpu.shapes_gpu import KINDS
from forza_abyss_painter.shapegen.gpu.scoring import ALPHA_FIXED
from forza_abyss_painter.shapegen.gpu.joint_polish import _hard_render, _resolve_rgb_closed_form


def _identify_dead_shape_indices(shapes_json: list[dict], h: int, w: int,
                                  device, min_visible_px: int) -> set[int]:
    """Walk shapes in REVERSE z-order (topmost first). For each, the visible pixel set
    is its mask AND NOT-yet-claimed. Accumulate `claimed` from top down. Return indices
    whose visible-pixel count is below threshold (= effectively invisible in the final
    composite under alpha=255 z-order)."""
    n = len(shapes_json)
    if n == 0:
        return set()
    geom = torch.tensor(
        [[s["x"], s["y"], s["rx"], s["ry"], s["angle"]] for s in shapes_json],
        dtype=DTYPE, device=device,
    )
    claimed = torch.zeros((h, w), dtype=torch.bool, device=device)
    dead: set[int] = set()
    for i in reversed(range(n)):
        mask = rasterize_rotated_ellipses(geom[i:i + 1], h, w)[0] > 0
        visible = mask & ~claimed
        if int(visible.sum().cpu()) < min_visible_px:
            dead.add(i)
        claimed |= mask
    return dead


def _render_live_canvas(shapes_json: list[dict], canvas_init: torch.Tensor,
                         h: int, w: int, device) -> torch.Tensor:
    """Hard-render the (live, in-order) shapes onto a fresh canvas. Returns DTYPE
    tensor of shape (H, W, 3) — same format the greedy loop maintains."""
    if not shapes_json:
        return canvas_init.to(DTYPE)
    geom = torch.tensor(
        [[s["x"], s["y"], s["rx"], s["ry"], s["angle"]] for s in shapes_json],
        dtype=DTYPE, device=device,
    )
    rgb = torch.tensor([s["color"][:3] for s in shapes_json], dtype=DTYPE, device=device)
    alpha = torch.tensor([s["color"][3] for s in shapes_json], dtype=DTYPE, device=device)
    return _hard_render(canvas_init.to(DTYPE), geom, rgb, alpha, h, w)


def _mini_greedy_refill(canvas: torch.Tensor, target: torch.Tensor,
                        alpha_t, alpha_mask_f, edge_weight,
                        n_to_commit: int, cfg, h: int, w: int,
                        generator) -> tuple[list[dict], torch.Tensor]:
    """Commit exactly n_to_commit new rotated_ellipse shapes targeting the current
    residual (canvas vs target). Self-contained mini-greedy using the same bbox-local
    + gradient-refine primitives as the main greedy loop in engine.py, but bounded to
    a fixed iteration count and starting from the post-cleanup canvas.

    NOTE: ellipse-only, gradient-refine-only path — matches the production preset
    config (bbox_local=True, refine_mode='gradient', shape_types=['rotated_ellipse']).
    If extending refill to multi-shape or hill-climb modes in the future, expand here.
    """
    # Lazy import to avoid circular dependency: engine.py imports from this module
    # (well, run_gpu calls clean_and_refill) and this needs _refine_gradient + _composite_one
    # from engine. Lazy import sidesteps that during module load.
    from forza_abyss_painter.shapegen.gpu.engine import _refine_gradient, _composite_one

    ekind = KINDS["rotated_ellipse"]
    new_shapes: list[dict] = []
    canvas_u8 = canvas.clamp(0, 255).round().to(torch.uint8)
    target_u8 = target.to(torch.uint8) if target.dtype != torch.uint8 else target
    device = canvas.device

    # Resolve chunk size once per refill pass (config doesn't change).
    _k_chunk = _resolve_k_chunk_size(
        K=cfg.random_samples, bbox_local=True,
        max_resolution=max(int(h), int(w)),
        vram_budget_gib=cfg.vram_budget_gib,
        k_chunk_override=cfg.k_chunk_size,
        bbox_crop_max=cfg.bbox_crop_max,
    )
    for _ in range(n_to_commit):
        # 1. Random search over K candidates targeting the residual canvas.
        params = ekind.init(cfg.random_samples, w, h, generator).to(device)
        scores, colors, alphas = crop_score_ellipse_batch_chunked(
            params, canvas_u8, target_u8,
            alpha_t=alpha_t, edge_weight=edge_weight,
            alpha_levels=cfg.alpha_levels, max_crop_radius=cfg.bbox_crop_max,
            chunk_size=_k_chunk,
        )
        if not torch.isfinite(scores).any():
            break   # all candidates rejected (eg sticker overlap too low); nothing useful to add
        best_idx = int(scores.argmin().cpu().item())
        best_score = float(scores[best_idx].cpu().item())
        if best_score == float("inf"):
            break

        # 2. Gradient-refine top-M candidates.
        M = min(cfg.grad_starts, params.shape[0])
        top_idx = torch.topk(scores, M, largest=False).indices
        init = params[top_idx].clone()
        best_params, best_color, best_alpha, refined_score = _refine_gradient(
            ekind, init, canvas_u8, target_u8, alpha_t, alpha_mask_f, h, w,
            cfg.grad_steps, cfg.grad_lr, alpha_levels=cfg.alpha_levels,
            edge_weight=edge_weight,
        )
        if refined_score == float("inf"):
            break

        # 3. Commit.
        canvas_u8 = _composite_one(ekind, canvas_u8, best_params, best_color, h, w,
                                    alpha_mask_f, best_alpha)
        c = best_color.cpu().tolist()
        color = [int(c[0]), int(c[1]), int(c[2]), int(best_alpha)]
        new_shapes.append(ekind.to_json(best_params.cpu().tolist(), color))

    return new_shapes, canvas_u8.to(DTYPE)


def clean_and_refill(
    shapes_json: list[dict],
    target_rgb: torch.Tensor,           # (H, W, 3) uint8 or DTYPE tensor
    canvas_init: torch.Tensor,           # the canvas-init the engine started from
    alpha_t, alpha_mask_f, edge_weight,
    cfg, h: int, w: int, generator,
) -> tuple[list[dict], torch.Tensor]:
    """Drop dead shapes from `shapes_json`, refill the freed slots with new mini-greedy
    commits targeting current residual, snap-back colors. Returns (new_shapes_list,
    final_canvas_uint8_numpy).

    No-op if cfg.refill_dead_shapes is False or no dead shapes are detected — in that
    case the input shapes pass through and the canvas is just re-rendered for return.

    Caller is expected to have already run greedy + joint_polish + snap-back; this is
    the final stage before serialization."""
    if not shapes_json:
        # Empty input — return empty canvas.
        empty = canvas_init.to(DTYPE).clamp(0, 255).round().to(torch.uint8).cpu().numpy()
        return [], empty

    # Caller (engine.run_gpu) is expected to pre-gate on ellipse-only — the dead-shape
    # detection here uses the rotated_ellipse rasterizer and the mini-greedy commits
    # rotated_ellipses. Defensive assert for clarity if someone wires this in elsewhere.
    assert all(s["type"] == "rotated_ellipse" for s in shapes_json), (
        "clean_and_refill is ellipse-only; caller must gate on shape type"
    )

    device = canvas_init.device

    # 1. Detect dead shapes.
    dead_indices = _identify_dead_shape_indices(
        shapes_json, h, w, device, cfg.refill_min_visible_px,
    )
    if not dead_indices:
        # No waste — render whatever's in shapes_json and return.
        canvas = _render_live_canvas(shapes_json, canvas_init, h, w, device)
        return shapes_json, canvas.clamp(0, 255).round().to(torch.uint8).cpu().numpy()

    # 2. Drop dead, rebuild live canvas.
    live_shapes = [s for i, s in enumerate(shapes_json) if i not in dead_indices]
    n_dead = len(dead_indices)
    canvas_after_cleanup = _render_live_canvas(live_shapes, canvas_init, h, w, device)

    # 3. Mini-greedy: refill the freed slots.
    new_shapes, canvas_after_refill = _mini_greedy_refill(
        canvas_after_cleanup, target_rgb,
        alpha_t, alpha_mask_f, edge_weight,
        n_dead, cfg, h, w, generator,
    )

    # 4. Combine + closed-form RGB snap-back over the full set (so existing shapes get
    #    their colors re-tuned for the new occlusion structure after dead-drop + refill,
    #    and new shapes get their snap-back color too).
    full_shapes = live_shapes + new_shapes
    if cfg.lock_alpha and full_shapes:
        geom_full = torch.tensor(
            [[s["x"], s["y"], s["rx"], s["ry"], s["angle"]] for s in full_shapes],
            dtype=DTYPE, device=device,
        )
        target_f = target_rgb.to(DTYPE)
        rgb_new = _resolve_rgb_closed_form(geom_full, target_f, alpha_mask_f, h, w)
        # Write the snapped colors back into the shape dicts (preserve alpha=255).
        rgb_list = rgb_new.round().clamp(0, 255).cpu().tolist()
        for i, s in enumerate(full_shapes):
            s["color"] = [int(rgb_list[i][0]), int(rgb_list[i][1]),
                          int(rgb_list[i][2]), s["color"][3]]
        # Final render uses the snapped colors.
        alpha_full = torch.tensor(
            [s["color"][3] for s in full_shapes], dtype=DTYPE, device=device,
        )
        final_canvas = _hard_render(canvas_init.to(DTYPE), geom_full, rgb_new,
                                     alpha_full, h, w)
    else:
        # No snap-back when lock_alpha is off (which shouldn't happen given v0.1.7's gate,
        # but defensive). Use the mini-greedy's committed canvas as-is.
        final_canvas = canvas_after_refill

    final_u8 = final_canvas.clamp(0, 255).round().to(torch.uint8).cpu().numpy()
    return full_shapes, final_u8
