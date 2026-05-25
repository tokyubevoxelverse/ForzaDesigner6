"""Batched scoring for GPU shape-gen prototype.

Memory note: `score_batch` materializes `(K, H, W, 3)` intermediates during the
optimal-color reduction and SSE decomposition. At K=256 and H=W=1200 the peak
unified-memory footprint is roughly 12 GB float32 (rasterizer + scorer combined).
On 8 GB Apple Silicon, reduce K (random_samples / mutations_per_round in GPUConfig)
or max-resolution. 16+ GB unified memory is comfortable for the default settings.
"""
from __future__ import annotations

import torch

from forza_abyss_painter.shapegen.gpu.device import DTYPE

# CPU engine uses alpha=128 for ellipses (see shapes/ellipse.py:72). Lock the same.
ALPHA_FIXED: int = 128

# Fraction of a candidate shape's body that must sit inside the opaque region.
# Re-exports the same constant as forza_abyss_painter/shapegen/scoring.py:99 so the GPU and CPU
# paths reject identically. Imported by the engine and tested for equality with CPU.
STICKER_OVERLAP_MIN: float = 0.995

# A pixel of the rasterized soft mask is part of the "body" if its coverage >= this.
# CPU code uses mask_local >= 128 on a 0..255 mask, which is the same threshold (0.5).
_BODY_THRESHOLD: float = 0.5


def _to_float(t: torch.Tensor) -> torch.Tensor:
    """uint8 HWC → float32 HWC (no normalization; range stays 0..255)."""
    return t.to(DTYPE)


def _score_core(
    masks: torch.Tensor,                       # (K, H, W) float in 0..1
    current_u8: torch.Tensor,                  # (H, W, 3) uint8
    target_u8: torch.Tensor,                   # (H, W, 3) uint8
    alpha_mask: torch.Tensor | None = None,    # (H, W) uint8 or None
    alpha: int = ALPHA_FIXED,
    edge_weight: torch.Tensor | None = None,   # (H, W) float per-pixel loss weight or None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score K candidate masks at a SINGLE alpha. Returns (scores, colors). See score_batch
    for the public entry point that searches over alpha levels.

    Non-sticker math: optimal src = (target - (1-a)*current) / a averaged over the mask;
                      SSE-after = SSE-before - SSE_inside_old + SSE_inside_new.
    Sticker math: same closed-form but weighted by min(mask, alpha) so transparent pixels
                  don't influence color, and RMS uses (alpha > 0) as the loss weight so
                  transparent pixels are ignored.

    `edge_weight` (H, W), if given, multiplies into the per-pixel loss so high-contrast
    detail (eyes, lineart) counts more — steering the search to spend shapes there. It
    combines multiplicatively with the sticker weight. Optimal color is left as the mask
    average (placement steering is the dominant effect).
    """
    if masks.ndim != 3:
        raise ValueError(f"masks must be (K, H, W); got {tuple(masks.shape)}")
    K, H, W = masks.shape
    device = masks.device

    cur = _to_float(current_u8)   # (H, W, 3)
    tgt = _to_float(target_u8)    # (H, W, 3)
    a = float(alpha) / 255.0
    ew = edge_weight.unsqueeze(-1) if edge_weight is not None else None  # (H, W, 1)

    if alpha_mask is None:
        # ---- Non-sticker (opaque) path ----
        m = masks.unsqueeze(-1)                                    # (K, H, W, 1)
        weight = masks.sum(dim=(1, 2)).clamp(min=1e-3)             # (K,)
        sum_mt = (m * tgt).sum(dim=(1, 2))                         # (K, 3)
        sum_mc = (m * cur).sum(dim=(1, 2))                         # (K, 3)
        src = ((sum_mt - (1.0 - a) * sum_mc) / (a * weight.unsqueeze(-1))).clamp(0.0, 255.0)

        diff_old = cur - tgt                                       # (H, W, 3)
        diff_old_sq = diff_old * diff_old
        src_expanded = src.view(K, 1, 1, 3)
        diff_new_inside = a * src_expanded + (1.0 - a) * cur - tgt
        diff_new_sq = diff_new_inside * diff_new_inside
        # Soft-edge approximation: at fringe pixels (0<m<1) the SSE term overestimates the
        # true post-blend error by a non-negative offset shared by every candidate, so argmin
        # ordering is preserved (CPU reference uses the exact post-blend SSE).
        if ew is None:
            sse_before = diff_old_sq.sum()
            sse_in_old = (m * diff_old_sq).sum(dim=(1, 2, 3))
            sse_in_new = (m * diff_new_sq).sum(dim=(1, 2, 3))
            n = float(H * W * 3)
        else:
            sse_before = (diff_old_sq * ew).sum()
            sse_in_old = (m * diff_old_sq * ew).sum(dim=(1, 2, 3))
            sse_in_new = (m * diff_new_sq * ew).sum(dim=(1, 2, 3))
            n = (ew.sum() * 3.0).clamp(min=1.0)

        sse_after = sse_before - sse_in_old + sse_in_new
        rms = torch.sqrt(sse_after.clamp(min=0.0) / n)
        return rms, src.round().to(torch.uint8)

    # ---- Sticker path ----
    alpha_f = alpha_mask.to(DTYPE) / 255.0                         # (H, W) in 0..1
    opaque_body = (alpha_mask >= 128)                              # (H, W) bool — for overlap check
    weight_full_2d = (alpha_mask > 0).to(DTYPE)                    # (H, W) in {0, 1}
    if edge_weight is not None:
        weight_full_2d = weight_full_2d * edge_weight              # fold in edge emphasis
    weight_full = weight_full_2d.unsqueeze(-1)                     # (H, W, 1) — per-pixel RMS weight

    # Effective mask for optimal-color reduction: AND mask with alpha (in 0..1).
    eff = masks * alpha_f.unsqueeze(0)                             # (K, H, W)
    eff_w = eff.unsqueeze(-1)                                      # (K, H, W, 1)
    eff_weight = eff.sum(dim=(1, 2)).clamp(min=1e-3)               # (K,)

    sum_mt = (eff_w * tgt).sum(dim=(1, 2))                         # (K, 3)
    sum_mc = (eff_w * cur).sum(dim=(1, 2))                         # (K, 3)
    src = ((sum_mt - (1.0 - a) * sum_mc) / (a * eff_weight.unsqueeze(-1))).clamp(0.0, 255.0)

    # Weighted RMS. The shape paints with `masks` (not `eff`); every SSE term is multiplied
    # by `weight_full` (alpha gate × edge emphasis) so transparent pixels never contribute
    # and high-contrast pixels count more.
    m_w = masks.unsqueeze(-1)                                       # (K, H, W, 1)
    diff_old = cur - tgt                                            # (H, W, 3)
    sse_before = ((diff_old * diff_old) * weight_full).sum()        # scalar
    sse_in_old = (m_w * (diff_old * diff_old) * weight_full).sum(dim=(1, 2, 3))
    src_expanded = src.view(K, 1, 1, 3)
    diff_new_inside = a * src_expanded + (1.0 - a) * cur - tgt      # (K, H, W, 3)
    # (same soft-edge approximation as above — argmin ordering is preserved)
    sse_in_new = (m_w * (diff_new_inside * diff_new_inside) * weight_full).sum(dim=(1, 2, 3))

    sse_after = sse_before - sse_in_old + sse_in_new
    n = weight_full_2d.sum() * 3.0
    rms = torch.sqrt(sse_after.clamp(min=0.0) / n.clamp(min=1.0))

    # Sticker overlap constraint: count body pixels and how many sit in opaque area.
    body = masks >= _BODY_THRESHOLD                                 # (K, H, W) bool
    body_total = body.sum(dim=(1, 2)).to(DTYPE)                     # (K,)
    inside = (body & opaque_body.unsqueeze(0)).sum(dim=(1, 2)).to(DTYPE)  # (K,)
    ratio = torch.where(body_total > 0, inside / body_total.clamp(min=1.0),
                        torch.zeros_like(body_total))
    rejected = ratio < STICKER_OVERLAP_MIN
    rms = torch.where(rejected, torch.full_like(rms, float("inf")), rms)

    return rms, src.round().to(torch.uint8)


def score_batch(
    masks: torch.Tensor,                       # (K, H, W) float in 0..1
    current_u8: torch.Tensor,                  # (H, W, 3) uint8
    target_u8: torch.Tensor,                   # (H, W, 3) uint8
    alpha_mask: torch.Tensor | None = None,    # (H, W) uint8 or None
    alpha: int = ALPHA_FIXED,
    alpha_levels: list[int] | None = None,
    edge_weight: torch.Tensor | None = None,   # (H, W) per-pixel loss weight or None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Score K candidate masks. Returns (scores, colors, alphas).

    scores:  (K,) float  — RMS-if-committed (lower better). +inf for sticker-rejected shapes.
    colors:  (K, 3) uint8 — closed-form optimal RGB for the chosen alpha.
    alphas:  (K,) uint8  — the per-candidate alpha that minimized RMS.

    If `alpha_levels` is None, every candidate uses the single fixed `alpha` (the original
    behavior). If a list is given, each candidate is scored at every level and the alpha
    with the lowest RMS is kept per candidate — this lets a high-contrast dark feature pick
    a near-opaque alpha (crisp black lines) while a smooth gradient picks a low alpha.
    The optimal RGB is recomputed per alpha (it depends on alpha), so color and alpha stay
    consistent.
    """
    levels = alpha_levels if alpha_levels else [alpha]
    best_rms = best_col = best_a = None
    for a in levels:
        rms, col = _score_core(masks, current_u8, target_u8, alpha_mask, a, edge_weight)
        if best_rms is None:
            best_rms, best_col = rms, col
            best_a = torch.full_like(rms, float(a))
        else:
            better = rms < best_rms
            best_rms = torch.where(better, rms, best_rms)
            best_col = torch.where(better.unsqueeze(-1), col, best_col)
            best_a = torch.where(better, torch.full_like(rms, float(a)), best_a)
    return best_rms, best_col, best_a.round().to(torch.uint8)


def score_batch_chunked(
    masks: "torch.Tensor",
    current_u8: "torch.Tensor",
    target_u8: "torch.Tensor",
    alpha_mask: "torch.Tensor | None" = None,
    alpha: int = ALPHA_FIXED,
    alpha_levels: "list[int] | None" = None,
    edge_weight: "torch.Tensor | None" = None,
    *,
    chunk_size: int = 0,
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
    """Chunked wrapper around `score_batch`.

    Splits the K-sized mask batch into chunks of `chunk_size` candidates,
    scores each chunk independently, concatenates the per-candidate
    outputs. Mathematically equivalent to a single full-K call (same
    candidates, same per-candidate (rms, color, alpha) tuples, same
    argmin downstream) at a fraction of the peak VRAM.

    Why this exists: the (K, H, W, 3) intermediate inside `score_batch`
    is the dominant VRAM cost. K=16384 at 1200px = ~40 GiB. Chunking
    K=16384 into 4 chunks of K=4096 caps the live intermediate at
    ~10 GiB, fits an RTX 3060 / 4070, takes ~4x wall time. That's the
    painter-fh6 architectural trade-off (sequential candidates = low
    VRAM, slow) generalized: the user picks where on the
    speed-vs-memory curve they sit by setting a VRAM budget.

    `chunk_size <= 0` or `chunk_size >= K` runs as a single pass
    (the original `score_batch` behavior) — no overhead when not
    chunking.
    """
    import torch as _t
    K = masks.shape[0]
    if chunk_size <= 0 or chunk_size >= K:
        return score_batch(masks, current_u8, target_u8,
                           alpha_mask=alpha_mask, alpha=alpha,
                           alpha_levels=alpha_levels, edge_weight=edge_weight)
    # Per-chunk scoring + concat. The output tensors are small relative
    # to the intermediates (just K-sized lists), so cat is cheap.
    score_parts = []
    color_parts = []
    alpha_parts = []
    for k_start in range(0, K, chunk_size):
        k_end = min(k_start + chunk_size, K)
        s, c, a = score_batch(
            masks[k_start:k_end], current_u8, target_u8,
            alpha_mask=alpha_mask, alpha=alpha,
            alpha_levels=alpha_levels, edge_weight=edge_weight,
        )
        score_parts.append(s)
        color_parts.append(c)
        alpha_parts.append(a)
    return (_t.cat(score_parts, dim=0),
            _t.cat(color_parts, dim=0),
            _t.cat(alpha_parts, dim=0))
