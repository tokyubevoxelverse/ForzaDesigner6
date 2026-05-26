from __future__ import annotations

import numpy as np

from fd6.shapegen.quality import QualityContext, gradient_score, sobel_xy, rgb_to_luma
from fd6.shapegen.shapes.base import Shape


def _rasterized_mask(shape: Shape, w: int, h: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    cache = getattr(shape, "_fd6_mask_cache", None)
    if isinstance(cache, tuple) and len(cache) == 4 and cache[0] == w and cache[1] == h:
        return cache[2], cache[3]
    mask_local, bbox = shape.rasterize_mask(w, h)
    setattr(shape, "_fd6_mask_cache", (w, h, mask_local, bbox))
    return mask_local, bbox


def rms_error(
    a: np.ndarray,
    b: np.ndarray,
    alpha_mask: np.ndarray | None = None,
    edge_weight: np.ndarray | None = None,
) -> float:
    diff = a.astype(np.int32) - b.astype(np.int32)
    sq = diff * diff
    if edge_weight is not None:
        weight = edge_weight[:, :, None]
        total = float((sq * weight).sum())
        n = float(edge_weight.sum() * 3)
        if n < 1:
            return 0.0
        return float(np.sqrt(total / n))
    if alpha_mask is None:
        return float(np.sqrt(sq.mean()))
    weight = (alpha_mask > 0)[:, :, None].astype(np.float32)
    total = float((sq * weight).sum())
    n = float(weight.sum() * 3)
    if n < 1:
        return 0.0
    return float(np.sqrt(total / n))


def compute_optimal_color(
    target: np.ndarray,
    current: np.ndarray,
    mask_local: np.ndarray,
    bbox: tuple[int, int, int, int],
    alpha: int,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    if x1 <= x0 or y1 <= y0 or mask_local.size == 0:
        return (0, 0, 0, alpha)
    tgt = target[y0:y1, x0:x1].astype(np.float32)
    cur = current[y0:y1, x0:x1].astype(np.float32)
    m = mask_local.astype(np.float32) / 255.0
    weight = m.sum()
    if weight < 0.5:
        return (0, 0, 0, alpha)
    a = alpha / 255.0
    if a < 1e-6:
        return (0, 0, 0, alpha)
    src = (tgt - (1.0 - a) * cur) / a
    src_masked = src * m[:, :, None]
    avg = src_masked.reshape(-1, 3).sum(axis=0) / weight
    avg = np.clip(avg, 0, 255).astype(np.int32)
    return (int(avg[0]), int(avg[1]), int(avg[2]), alpha)


def apply_shape_inplace(
    current: np.ndarray,
    shape: Shape,
    alpha_mask: np.ndarray | None = None,
) -> tuple[int, int, int, int]:
    h, w = current.shape[:2]
    mask_local, bbox = _rasterized_mask(shape, w, h)
    x0, y0, x1, y1 = bbox
    if x1 <= x0 or y1 <= y0 or mask_local.size == 0:
        return bbox
    if alpha_mask is not None:
        region_alpha = alpha_mask[y0:y1, x0:x1]
        effective_mask = np.minimum(mask_local, region_alpha)
    else:
        effective_mask = mask_local
    color = shape.color
    region = current[y0:y1, x0:x1]
    if alpha_mask is None:
        mask_bool = effective_mask >= 128
        if not mask_bool.any():
            return bbox
        alpha = int(color[3])
        if alpha <= 0:
            return bbox
        src = np.asarray(color[:3], dtype=np.uint32)
        if alpha >= 255:
            region[mask_bool] = src.astype(np.uint8)
            return bbox
        region_src = region[mask_bool].astype(np.uint32)
        blended = ((alpha * src + (255 - alpha) * region_src) // 255).astype(np.uint8)
        region[mask_bool] = blended
        return bbox
    a = color[3] / 255.0
    region_cur = region.astype(np.float32)
    src = np.array(color[:3], dtype=np.float32)
    m = (effective_mask.astype(np.float32) / 255.0)[:, :, None]
    blended = m * (a * src + (1.0 - a) * region_cur) + (1.0 - m) * region_cur
    region[:] = np.clip(blended, 0, 255).astype(np.uint8)
    return bbox


def composite(
    current: np.ndarray,
    shape: Shape,
    target: np.ndarray,
    alpha_mask: np.ndarray | None = None,
    edge_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    score, color = score_shape(shape, current, target, alpha_mask, edge_weight=edge_weight)
    if score == float("inf"):
        return current, rms_error(current, target, alpha_mask, edge_weight)
    new = current.copy()
    shape.color = color
    apply_shape_inplace(new, shape, alpha_mask)
    return new, score


STICKER_OVERLAP_MIN = 0.995


def precompute_canvas_error(
    current: np.ndarray,
    target: np.ndarray,
    alpha_mask: np.ndarray | None = None,
    edge_weight: np.ndarray | None = None,
) -> tuple[float, float]:
    if edge_weight is not None:
        weight_full = edge_weight[:, :, None]
        diff = (current.astype(np.float32) - target.astype(np.float32)) ** 2
        full_sq = float((diff * weight_full).sum())
        n = float(edge_weight.sum() * 3)
        return full_sq, n
    if alpha_mask is None:
        diff = current.astype(np.int32) - target.astype(np.int32)
        full_sq = float((diff * diff).sum())
        n = float(current.shape[0] * current.shape[1] * 3)
        return full_sq, n
    weight_full = (alpha_mask > 0)[:, :, None].astype(np.float32)
    diff = (current.astype(np.float32) - target.astype(np.float32)) ** 2
    full_sq = float((diff * weight_full).sum())
    n = float(weight_full.sum() * 3)
    return full_sq, n


def score_shape(
    shape: Shape,
    current: np.ndarray,
    target: np.ndarray,
    alpha_mask: np.ndarray | None = None,
    *,
    canvas_full_sq: float | None = None,
    canvas_norm: float | None = None,
    quality_context: QualityContext | None = None,
    weighted_full_sq: float | None = None,
    weighted_norm: float | None = None,
    gradient_full_error: float | None = None,
    gradient_norm: float | None = None,
    quality_edge_weight: np.ndarray | None = None,
    edge_weight: np.ndarray | None = None,
    fixed_color: tuple[int, int, int, int] | None = None,
    base_rms: float | None = None,
    base_canvas_full_sq: float | None = None,
) -> tuple[float, tuple[int, int, int, int]]:
    h, w = current.shape[:2]
    mask_local, bbox = _rasterized_mask(shape, w, h)
    x0, y0, x1, y1 = bbox
    if x1 <= x0 or y1 <= y0 or mask_local.size == 0:
        return float("inf"), shape.color
    if alpha_mask is None and quality_context is None and edge_weight is None:
        return _score_shape_plain_binary(
            shape,
            current,
            target,
            mask_local,
            bbox,
            canvas_full_sq,
            canvas_norm,
        )
    effective_mask = mask_local
    if alpha_mask is not None:
        region_alpha = alpha_mask[y0:y1, x0:x1]
        shape_body = mask_local >= 128
        body_total = float(shape_body.sum())
        if body_total < 1.0:
            return float("inf"), shape.color
        opaque_body = region_alpha >= 128
        if not opaque_body.any():
            return float("inf"), shape.color
        inside = float((shape_body & opaque_body).sum())
        if inside / body_total < STICKER_OVERLAP_MIN:
            return float("inf"), shape.color
        effective_mask = np.minimum(mask_local, region_alpha)
    if fixed_color is None:
        color = compute_optimal_color(target, current, effective_mask, bbox, shape.color[3])
    else:
        color = tuple(int(v) for v in fixed_color)
    a = color[3] / 255.0
    region_cur = current[y0:y1, x0:x1].astype(np.float32)
    region_tgt = target[y0:y1, x0:x1].astype(np.float32)
    src = np.array(color[:3], dtype=np.float32)
    m = (mask_local.astype(np.float32) / 255.0)[:, :, None]
    blended = m * (a * src + (1.0 - a) * region_cur) + (1.0 - m) * region_cur
    diff_in = blended - region_tgt
    # Edge-weighted path supersedes the boolean alpha gate when present.
    if edge_weight is not None:
        if canvas_full_sq is None or canvas_norm is None:
            full_sq, n = precompute_canvas_error(current, target, alpha_mask, edge_weight)
        else:
            full_sq, n = canvas_full_sq, canvas_norm
        weight_region = edge_weight[y0:y1, x0:x1][:, :, None]
        region_old_sq = float((((region_cur - region_tgt) ** 2) * weight_region).sum())
        region_new_sq = float(((diff_in ** 2) * weight_region).sum())
        total_sq = full_sq - region_old_sq + region_new_sq
        if n < 1:
            return 0.0, color
        return float(np.sqrt(max(0.0, total_sq) / n)), color
    if alpha_mask is None:
        if base_canvas_full_sq is not None and base_rms is not None:
            total_sq = float(base_canvas_full_sq)
            rms = float(base_rms)
        else:
            if canvas_full_sq is None or canvas_norm is None:
                full_sq, n_px = precompute_canvas_error(current, target, None)
            else:
                full_sq, n_px = canvas_full_sq, canvas_norm
            region_old_sq = float(((region_cur - region_tgt) ** 2).sum())
            region_new_sq = float((diff_in ** 2).sum())
            total_sq = full_sq - region_old_sq + region_new_sq
            rms = float(np.sqrt(max(0.0, total_sq) / n_px))
        objective = _quality_objective(
            rms,
            total_sq,
            shape,
            current,
            target,
            blended,
            bbox,
            quality_context,
            weighted_full_sq,
            weighted_norm,
            gradient_full_error,
            gradient_norm,
            quality_edge_weight,
        )
        return objective, color
    if canvas_full_sq is None or canvas_norm is None:
        full_sq, n = precompute_canvas_error(current, target, alpha_mask)
    else:
        full_sq, n = canvas_full_sq, canvas_norm
    weight_region = ((alpha_mask[y0:y1, x0:x1] > 0).astype(np.float32))[:, :, None]
    region_old_sq = float((((region_cur - region_tgt) ** 2) * weight_region).sum())
    region_new_sq = float(((diff_in ** 2) * weight_region).sum())
    total_sq = full_sq - region_old_sq + region_new_sq
    if n < 1:
        return 0.0, color
    rms = float(np.sqrt(max(0.0, total_sq) / n))
    objective = _quality_objective(
        rms,
        total_sq,
        shape,
        current,
        target,
        blended,
        bbox,
        quality_context,
        weighted_full_sq,
        weighted_norm,
        gradient_full_error,
        gradient_norm,
        quality_edge_weight,
    )
    return objective, color


def _score_shape_plain_binary(
    shape: Shape,
    current: np.ndarray,
    target: np.ndarray,
    mask_local: np.ndarray,
    bbox: tuple[int, int, int, int],
    canvas_full_sq: float | None,
    canvas_norm: float | None,
) -> tuple[float, tuple[int, int, int, int]]:
    x0, y0, x1, y1 = bbox
    mask_bool = mask_local >= 128
    if not mask_bool.any():
        return float("inf"), shape.color
    alpha = int(shape.color[3])
    a = alpha / 255.0
    if a < 1e-6:
        return float("inf"), shape.color
    region_cur_u8 = current[y0:y1, x0:x1]
    region_tgt_u8 = target[y0:y1, x0:x1]
    region_cur = region_cur_u8[mask_bool].astype(np.float32)
    region_tgt = region_tgt_u8[mask_bool].astype(np.float32)
    weight = float(region_cur.shape[0])
    if weight < 0.5:
        return float("inf"), shape.color
    avg = ((region_tgt - (1.0 - a) * region_cur) / a).sum(axis=0) / weight
    avg = np.clip(avg, 0, 255).astype(np.int32)
    color = (int(avg[0]), int(avg[1]), int(avg[2]), alpha)
    src = np.array(color[:3], dtype=np.float32)
    blended = a * src + (1.0 - a) * region_cur
    if canvas_full_sq is None or canvas_norm is None:
        full_sq, n_px = precompute_canvas_error(current, target, None)
    else:
        full_sq, n_px = canvas_full_sq, canvas_norm
    old_sq = float(((region_cur - region_tgt) ** 2).sum())
    new_sq = float(((blended - region_tgt) ** 2).sum())
    total_sq = max(0.0, float(full_sq) - old_sq + new_sq)
    rms = float(np.sqrt(total_sq / float(n_px)))
    setattr(shape, "_fd6_rms_score", rms)
    setattr(shape, "_fd6_canvas_full_sq", total_sq)
    return rms, color


def _quality_objective(
    rms: float,
    total_sq: float,
    shape: Shape,
    current: np.ndarray,
    target: np.ndarray,
    blended: np.ndarray,
    bbox: tuple[int, int, int, int],
    context: QualityContext | None,
    weighted_full_sq: float | None,
    weighted_norm: float | None,
    gradient_full_error: float | None,
    gradient_norm: float | None,
    quality_edge_weight: np.ndarray | None = None,
) -> float:
    setattr(shape, "_fd6_rms_score", float(rms))
    setattr(shape, "_fd6_canvas_full_sq", float(max(0.0, total_sq)))
    if context is None or not context.enabled:
        return rms
    x0, y0, x1, y1 = bbox
    edge_weight = context.edge_weight if quality_edge_weight is None else quality_edge_weight
    if weighted_full_sq is None or weighted_norm is None:
        diff = current.astype(np.float32) - target.astype(np.float32)
        weights = edge_weight[:, :, None]
        weighted_full_sq = float(((diff * diff) * weights).sum())
        weighted_norm = float(edge_weight.sum() * 3.0)
    region_cur = current[y0:y1, x0:x1].astype(np.float32)
    region_tgt = target[y0:y1, x0:x1].astype(np.float32)
    region_weight = edge_weight[y0:y1, x0:x1, None]
    old_weighted = float((((region_cur - region_tgt) ** 2) * region_weight).sum())
    new_weighted = float((((blended - region_tgt) ** 2) * region_weight).sum())
    weighted_total = max(0.0, float(weighted_full_sq) - old_weighted + new_weighted)
    weighted_rms = rms if float(weighted_norm) < 1.0 else float(np.sqrt(weighted_total / float(weighted_norm)))
    setattr(shape, "_fd6_weighted_full_sq", weighted_total)
    if context.gradient_weight <= 0.0:
        setattr(shape, "_fd6_gradient_full_error", 0.0 if gradient_full_error is None else float(gradient_full_error))
        return weighted_rms
    grad_total = gradient_full_error
    grad_norm = gradient_norm
    if grad_total is None or grad_norm is None:
        gx_full, gy_full = sobel_xy(rgb_to_luma(current))
        grad_total = float((((gx_full - context.target_gx) ** 2 + (gy_full - context.target_gy) ** 2) * edge_weight).sum())
        grad_norm = float(edge_weight.sum() * 2.0)
    xg0 = max(0, x0 - 1)
    yg0 = max(0, y0 - 1)
    xg1 = min(current.shape[1], x1 + 1)
    yg1 = min(current.shape[0], y1 + 1)
    old_patch = current[yg0:yg1, xg0:xg1]
    new_patch = old_patch.copy()
    new_patch[y0 - yg0:y1 - yg0, x0 - xg0:x1 - xg0] = np.clip(blended, 0, 255).astype(np.uint8)
    old_gx, old_gy = sobel_xy(rgb_to_luma(old_patch))
    new_gx, new_gy = sobel_xy(rgb_to_luma(new_patch))
    target_gx = context.target_gx[yg0:yg1, xg0:xg1]
    target_gy = context.target_gy[yg0:yg1, xg0:xg1]
    weight = edge_weight[yg0:yg1, xg0:xg1]
    old_grad = float((((old_gx - target_gx) ** 2 + (old_gy - target_gy) ** 2) * weight).sum())
    new_grad = float((((new_gx - target_gx) ** 2 + (new_gy - target_gy) ** 2) * weight).sum())
    gradient_total = max(0.0, float(grad_total) - old_grad + new_grad)
    setattr(shape, "_fd6_gradient_full_error", gradient_total)
    return weighted_rms + context.gradient_weight * gradient_score(gradient_total, float(grad_norm))
