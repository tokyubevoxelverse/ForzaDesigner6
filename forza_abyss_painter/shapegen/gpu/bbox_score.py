"""Bounding-box-local scoring for the random-search phase (ellipse-only).

The full-canvas scorer materializes (K, H, W, 3) intermediates, capping K at ~1-2k. But a
shape only touches its own bounding box; pixels outside are unchanged. So we can score each
candidate over a small per-candidate crop and compare candidates by the *incremental* SSE
delta they produce inside their box:

    delta_k = SSE_after_in_box(k) - SSE_before_in_box(k)

argmin over k of delta_k is the globally-best shape (the global SSE-before is identical for
all candidates, and each shape changes only its own box). This is geometrize's incremental
energy, batched on GPU. Crops are ~10x+ smaller than the full canvas, so the random phase
can run ~10x+ more candidates for the same VRAM — the geometrize "raise random samples"
lever, made affordable.

v1 uses a single crop size for the batch (= the batch's max ellipse radius), so the box
always contains every candidate's mask. Size-binning (smaller crops for small shapes → even
more samples) is a future refinement. Ellipse-only; gradient refinement stays full-canvas.
"""
from __future__ import annotations

import torch

from forza_abyss_painter.shapegen.gpu.device import DTYPE
from forza_abyss_painter.shapegen.gpu.scoring import ALPHA_FIXED, STICKER_OVERLAP_MIN, _BODY_THRESHOLD


def crop_score_ellipse_batch(
    params: torch.Tensor,            # (K, 5) ellipse params on device
    canvas_u8: torch.Tensor,         # (H, W, 3) uint8
    target_u8: torch.Tensor,         # (H, W, 3) uint8
    alpha_t: torch.Tensor | None = None,    # (H, W) uint8 sticker mask or None
    edge_weight: torch.Tensor | None = None,  # (H, W) per-pixel loss weight or None
    alpha_levels: list[int] | None = None,
    max_crop_radius: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (delta_scores (K,), colors_u8 (K,3), alphas_u8 (K,)).

    delta_scores: in-bbox SSE change if the candidate were committed (lower = better;
                  typically negative). +inf for sticker-rejected candidates. argmin over K
                  picks the globally best candidate.
    """
    K = params.shape[0]
    device = params.device
    H, W = canvas_u8.shape[:2]

    # Crop half-size E = batch's max radius (so every candidate's mask fits the box),
    # capped at max_crop_radius. B = 2E+1.
    rmax = float(params[:, 2:4].max().clamp(min=1.0).ceil().item())
    E = min(int(max_crop_radius), int(rmax) + 1)
    B = 2 * E + 1

    cur = canvas_u8.to(DTYPE)
    tgt = target_u8.to(DTYPE)

    cx_i = params[:, 0].round().long()
    cy_i = params[:, 1].round().long()
    ox = cx_i - E                                   # (K,) crop origin (top-left)
    oy = cy_i - E
    lin = torch.arange(B, device=device)
    gx = ox.view(K, 1) + lin.view(1, B)             # (K, B) global col idx
    gy = oy.view(K, 1) + lin.view(1, B)             # (K, B) global row idx
    vx = (gx >= 0) & (gx < W)                       # (K, B) col validity
    vy = (gy >= 0) & (gy < H)
    gxc = gx.clamp(0, W - 1)
    gyc = gy.clamp(0, H - 1)
    valid = (vy.unsqueeze(2) & vx.unsqueeze(1)).to(DTYPE)   # (K, B, B)

    row_idx = gyc.unsqueeze(2)                       # (K, B, 1)
    col_idx = gxc.unsqueeze(1)                       # (K, 1, B)
    cur_c = cur[row_idx, col_idx]                    # (K, B, B, 3)
    tgt_c = tgt[row_idx, col_idx]

    # Hard ellipse mask in crop coords (global pixel coords = origin + local index).
    cxp = params[:, 0].view(K, 1, 1)
    cyp = params[:, 1].view(K, 1, 1)
    rx = params[:, 2].clamp(min=1e-3).view(K, 1, 1)
    ry = params[:, 3].clamp(min=1e-3).view(K, 1, 1)
    ang = torch.deg2rad(params[:, 4]).view(K, 1, 1)
    ca, sa = torch.cos(ang), torch.sin(ang)
    lin_f = lin.to(DTYPE)
    gxf = ox.to(DTYPE).view(K, 1, 1) + lin_f.view(1, 1, B)   # (K, 1, B)
    gyf = oy.to(DTYPE).view(K, 1, 1) + lin_f.view(1, B, 1)   # (K, B, 1)
    dxc = gxf - cxp
    dyc = gyf - cyp
    xr = ca * dxc + sa * dyc
    yr = -sa * dxc + ca * dyc
    d2 = (xr / rx) ** 2 + (yr / ry) ** 2             # (K, B, B)
    mask = (d2 <= 1.0).to(DTYPE) * valid             # zero outside canvas

    # Per-pixel loss weight: opaque gate (sticker) × edge emphasis × validity, folded WITH
    # the shape mask into a single (K,B,B,1) combined weight `mw3` — this is multiplied into
    # every SSE term, so folding it once avoids holding mask3 + pw3 separately (each
    # (K,B,B,1) tensor is sizeable at high K). Frees intermediates aggressively because at
    # high K every (K,B,B,3) tensor is large; minimizing how many are live at once is what
    # keeps the bbox path within VRAM.
    if alpha_t is not None:
        alpha_c = alpha_t.to(DTYPE)[row_idx, col_idx]    # (K, B, B) 0..255
        pw = (alpha_c > 0).to(DTYPE)
    else:
        alpha_c = None
        pw = torch.ones((K, B, B), dtype=DTYPE, device=device)
    if edge_weight is not None:
        pw = pw * edge_weight[row_idx, col_idx]
    pw = pw * valid
    mw3 = (mask * pw).unsqueeze(-1)                       # (K,B,B,1) combined mask×weight
    del pw

    diff = cur_c - tgt_c
    sse_old = (mw3 * diff.square()).sum(dim=(1, 2, 3))    # (K,)
    del diff

    # Sticker overlap rejection (body must sit inside the opaque region).
    if alpha_t is not None:
        body = mask >= _BODY_THRESHOLD
        body_total = body.sum(dim=(1, 2)).to(DTYPE)
        opaque_body = (alpha_c >= 128) & (valid > 0)
        inside = (body & opaque_body).sum(dim=(1, 2)).to(DTYPE)
        ratio = torch.where(body_total > 0, inside / body_total.clamp(min=1.0),
                            torch.zeros_like(body_total))
        rejected = ratio < STICKER_OVERLAP_MIN
    else:
        rejected = torch.zeros(K, dtype=torch.bool, device=device)
    del valid

    best_delta = best_col = best_a = None
    for alpha in (alpha_levels or [ALPHA_FIXED]):
        a = float(alpha) / 255.0
        eff = mask * (alpha_c / 255.0) if alpha_t is not None else mask   # color weight
        eff3 = eff.unsqueeze(-1)
        wsum = eff.sum(dim=(1, 2)).clamp(min=1e-3)
        sum_mt = (eff3 * tgt_c).sum(dim=(1, 2))
        sum_mc = (eff3 * cur_c).sum(dim=(1, 2))
        del eff, eff3
        src = ((sum_mt - (1.0 - a) * sum_mc) / (a * wsum.unsqueeze(-1))).clamp(0.0, 255.0)
        diff_new = a * src.view(K, 1, 1, 3) + (1.0 - a) * cur_c - tgt_c
        sse_new = (mw3 * diff_new.square()).sum(dim=(1, 2, 3))
        del diff_new
        delta = sse_new - sse_old
        if best_delta is None:
            best_delta, best_col = delta, src
            best_a = torch.full((K,), float(alpha), dtype=DTYPE, device=device)
        else:
            better = delta < best_delta
            best_delta = torch.where(better, delta, best_delta)
            best_col = torch.where(better.unsqueeze(-1), src, best_col)
            best_a = torch.where(better, torch.full_like(best_a, float(alpha)), best_a)

    best_delta = torch.where(rejected, torch.full_like(best_delta, float("inf")), best_delta)
    return best_delta, best_col.round().to(torch.uint8), best_a.round().to(torch.uint8)
