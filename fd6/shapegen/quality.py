from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class QualityContext:
    edge_weight: np.ndarray
    base_edge_weight: np.ndarray
    line_weight: np.ndarray
    target_gx: np.ndarray
    target_gy: np.ndarray
    edge_y: np.ndarray
    edge_x: np.ndarray
    edge_angle: np.ndarray
    edge_sample_cdf: np.ndarray
    edge_weight_strength: float
    gradient_weight: float
    edge_alpha: int
    line_guide_strength: float = 0.0
    line_guide_agreement: float = 0.0

    @property
    def enabled(self) -> bool:
        return (
            self.edge_weight_strength > 0.0
            or self.gradient_weight > 0.0
            or self.line_guide_strength > 0.0
        )

    @property
    def has_edge_points(self) -> bool:
        return self.edge_x.size > 0

    def weight_for_line_factor(self, line_factor: float = 1.0) -> np.ndarray:
        factor = max(0.0, min(1.0, float(line_factor)))
        if self.line_weight.size == 0 or self.line_guide_strength <= 0.0:
            return self.edge_weight
        if factor >= 0.999:
            return self.edge_weight
        return (self.base_edge_weight + self.line_weight * factor).astype(np.float32)

    def norm_for_line_factor(self, line_factor: float = 1.0) -> float:
        return float(self.weight_for_line_factor(line_factor).sum() * 3.0)


def rgb_to_luma(rgb: np.ndarray) -> np.ndarray:
    work = rgb.astype(np.float32)
    return work[:, :, 0] * 0.299 + work[:, :, 1] * 0.587 + work[:, :, 2] * 0.114


def sobel_xy(luma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    src = luma.astype(np.float32, copy=False)
    padded = np.pad(src, ((1, 1), (1, 1)), mode="edge")
    gx = (
        -padded[:-2, :-2]
        - 2.0 * padded[1:-1, :-2]
        - padded[2:, :-2]
        + padded[:-2, 2:]
        + 2.0 * padded[1:-1, 2:]
        + padded[2:, 2:]
    )
    gy = (
        -padded[:-2, :-2]
        - 2.0 * padded[:-2, 1:-1]
        - padded[:-2, 2:]
        + padded[2:, :-2]
        + 2.0 * padded[2:, 1:-1]
        + padded[2:, 2:]
    )
    return gx.astype(np.float32), gy.astype(np.float32)


def rgb_edge_confidence(rgb: np.ndarray, valid: np.ndarray) -> np.ndarray:
    work = rgb.astype(np.float32)
    total = np.zeros(work.shape[:2], dtype=np.float32)
    for channel in range(3):
        gx, gy = sobel_xy(work[:, :, channel])
        total += gx * gx + gy * gy
    mag = np.sqrt(total)
    valid_mag = mag[valid]
    if valid_mag.size == 0:
        return np.zeros_like(mag, dtype=np.float32)
    high = float(np.percentile(valid_mag, 95.0))
    if high < 1e-6:
        return np.zeros_like(mag, dtype=np.float32)
    return np.clip(mag / high, 0.0, 1.0).astype(np.float32)


def _dilate3(values: np.ndarray) -> np.ndarray:
    padded = np.pad(values, ((1, 1), (1, 1)), mode="edge")
    out = padded[1:-1, 1:-1].copy()
    for yy in range(3):
        for xx in range(3):
            out = np.maximum(out, padded[yy:yy + values.shape[0], xx:xx + values.shape[1]])
    return out


def build_quality_context(
    target: np.ndarray,
    alpha_mask: np.ndarray | None,
    *,
    edge_weight_strength: float,
    gradient_weight: float,
    edge_alpha: int,
    line_guide: np.ndarray | None = None,
    line_guide_strength: float = 0.0,
    line_guide_agreement: float = 0.0,
) -> QualityContext | None:
    edge_weight_strength = max(0.0, float(edge_weight_strength))
    gradient_weight = max(0.0, float(gradient_weight))
    line_guide_strength = max(0.0, float(line_guide_strength))
    line_guide_agreement = max(0.0, min(1.0, float(line_guide_agreement)))
    if edge_weight_strength <= 0.0 and gradient_weight <= 0.0 and line_guide_strength <= 0.0:
        return None
    luma = rgb_to_luma(target)
    gx, gy = sobel_xy(luma)
    mag = np.sqrt(gx * gx + gy * gy)
    valid = np.ones(mag.shape, dtype=bool) if alpha_mask is None else alpha_mask > 0
    valid_mag = mag[valid]
    if valid_mag.size == 0:
        return None
    high = float(np.percentile(valid_mag, 95))
    if high < 1e-6:
        edge_norm = np.zeros_like(mag, dtype=np.float32)
    else:
        edge_norm = np.clip(mag / high, 0.0, 1.0).astype(np.float32)
    threshold = max(12.0, float(np.percentile(valid_mag, 75)))
    edge_mask = (mag >= threshold) & valid
    edge_norm = _dilate3(edge_norm)
    if alpha_mask is not None:
        edge_norm = edge_norm * valid.astype(np.float32)
    color_norm = rgb_edge_confidence(target, valid)
    source_confidence = np.maximum(edge_norm, color_norm)
    if alpha_mask is not None:
        source_confidence = source_confidence * valid.astype(np.float32)
    guide_norm = np.zeros_like(edge_norm, dtype=np.float32)
    guide_mask = np.zeros_like(edge_mask, dtype=bool)
    guide_angle = np.zeros_like(edge_norm, dtype=np.float32)
    guide_weight = np.zeros_like(edge_norm, dtype=np.float32)
    adjusted_line_strength = line_guide_strength
    if line_guide_strength > 0.0 and line_guide is not None:
        guide_src = np.asarray(line_guide, dtype=np.float32)
        if guide_src.shape == edge_norm.shape:
            guide_norm = np.nan_to_num(
                np.clip(guide_src, 0.0, 1.0),
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            ).astype(np.float32)
            if alpha_mask is not None:
                guide_norm = guide_norm * valid.astype(np.float32)
            valid_guide = guide_norm[valid]
            if valid_guide.size:
                density = float((valid_guide > 0.15).sum()) / float(valid_guide.size)
                if density > 0.35:
                    adjusted_line_strength *= max(0.25, 1.0 - min(0.75, (density - 0.35) * 1.8))
                positives = valid_guide[valid_guide > 0.03]
                if positives.size:
                    guide_threshold = max(0.08, float(np.percentile(positives, 65.0)))
                    agreement = (1.0 - line_guide_agreement) + line_guide_agreement * source_confidence
                    guide_weight = np.clip(guide_norm * agreement, 0.0, 1.0).astype(np.float32)
                    guide_mask = (guide_norm >= guide_threshold) & valid & (guide_weight > 0.0)
                    guide_gx, guide_gy = sobel_xy(guide_norm)
                    guide_angle = (np.degrees(np.arctan2(guide_gy, guide_gx) + (math.pi * 0.5)) % 180.0).astype(np.float32)
    base_edge_weight = (1.0 + edge_weight_strength * edge_norm).astype(np.float32)
    line_weight = (adjusted_line_strength * guide_weight).astype(np.float32)
    edge_weight = (base_edge_weight + line_weight).astype(np.float32)
    if alpha_mask is not None:
        base_edge_weight = base_edge_weight * valid.astype(np.float32)
        line_weight = line_weight * valid.astype(np.float32)
        edge_weight = edge_weight * valid.astype(np.float32)
    target_angle = (np.degrees(np.arctan2(gy, gx) + (math.pi * 0.5)) % 180.0).astype(np.float32)
    angle_map = np.where(guide_weight > edge_norm, guide_angle, target_angle).astype(np.float32)
    combined_mask = (edge_mask | guide_mask) & valid
    edge_y, edge_x = np.nonzero(combined_mask)
    if edge_y.size:
        edge_angle = angle_map[edge_y, edge_x].astype(np.float32)
        sample_weight = (
            0.05
            + edge_norm[edge_y, edge_x]
            + source_confidence[edge_y, edge_x]
            + (1.0 + adjusted_line_strength) * guide_weight[edge_y, edge_x]
        ).astype(np.float64)
        sample_weight = np.nan_to_num(sample_weight, nan=0.05, posinf=1.0, neginf=0.05)
        sample_weight = np.maximum(sample_weight, 0.001)
        edge_sample_cdf = np.cumsum(sample_weight, dtype=np.float64)
    else:
        edge_angle = np.zeros(0, dtype=np.float32)
        edge_sample_cdf = np.zeros(0, dtype=np.float64)
    return QualityContext(
        edge_weight=edge_weight,
        base_edge_weight=base_edge_weight,
        line_weight=line_weight,
        target_gx=gx,
        target_gy=gy,
        edge_y=edge_y.astype(np.int32),
        edge_x=edge_x.astype(np.int32),
        edge_angle=edge_angle,
        edge_sample_cdf=edge_sample_cdf,
        edge_weight_strength=edge_weight_strength,
        gradient_weight=gradient_weight,
        edge_alpha=max(1, min(255, int(edge_alpha))),
        line_guide_strength=adjusted_line_strength,
        line_guide_agreement=line_guide_agreement,
    )


def precompute_weighted_rgb_error(
    current: np.ndarray,
    target: np.ndarray,
    context: QualityContext | None,
    line_guide_factor: float = 1.0,
) -> tuple[float, float]:
    if context is None:
        return 0.0, 0.0
    diff = current.astype(np.float32) - target.astype(np.float32)
    edge_weight = context.weight_for_line_factor(line_guide_factor)
    weights = edge_weight[:, :, None]
    total = float(((diff * diff) * weights).sum())
    norm = float(edge_weight.sum() * 3.0)
    return total, norm


def gradient_error(rgb: np.ndarray, context: QualityContext | None, line_guide_factor: float = 1.0) -> float:
    if context is None:
        return 0.0
    gx, gy = sobel_xy(rgb_to_luma(rgb))
    weight = context.weight_for_line_factor(line_guide_factor)
    diff = ((gx - context.target_gx) ** 2 + (gy - context.target_gy) ** 2) * weight
    norm = float(weight.sum() * 2.0)
    if norm < 1.0:
        return 0.0
    return float(np.sqrt(float(diff.sum()) / norm) / 4.0)


def precompute_gradient_error(
    current: np.ndarray,
    context: QualityContext | None,
    line_guide_factor: float = 1.0,
) -> tuple[float, float]:
    if context is None:
        return 0.0, 0.0
    gx, gy = sobel_xy(rgb_to_luma(current))
    weight = context.weight_for_line_factor(line_guide_factor)
    total = float((((gx - context.target_gx) ** 2 + (gy - context.target_gy) ** 2) * weight).sum())
    norm = float(weight.sum() * 2.0)
    return total, norm


def gradient_score(total: float, norm: float) -> float:
    if norm < 1.0:
        return 0.0
    return float(np.sqrt(max(0.0, total) / norm) / 4.0)


def edge_mask(rgb: np.ndarray) -> np.ndarray:
    gx, gy = sobel_xy(rgb_to_luma(rgb))
    mag = np.sqrt(gx * gx + gy * gy)
    if mag.size == 0:
        return np.zeros(mag.shape, dtype=bool)
    threshold = max(12.0, float(np.percentile(mag, 75)))
    return mag >= threshold


def edge_f1(candidate: np.ndarray, target: np.ndarray) -> float:
    cand = edge_mask(candidate)
    tgt = edge_mask(target)
    if not cand.any() and not tgt.any():
        return 1.0
    cand_d = _dilate3(cand.astype(np.float32)) > 0.0
    tgt_d = _dilate3(tgt.astype(np.float32)) > 0.0
    precision_denom = int(cand.sum())
    recall_denom = int(tgt.sum())
    if precision_denom == 0 or recall_denom == 0:
        return 0.0
    precision = float((cand & tgt_d).sum()) / float(precision_denom)
    recall = float((tgt & cand_d).sum()) / float(recall_denom)
    if precision + recall <= 0.0:
        return 0.0
    return float((2.0 * precision * recall) / (precision + recall))


def _dilate_radius(values: np.ndarray, radius: int) -> np.ndarray:
    radius = max(0, int(radius))
    if radius <= 0:
        return values.astype(bool)
    out = values.astype(bool)
    work = values.astype(np.float32)
    for _ in range(radius):
        work = _dilate3(work)
    return work > 0.0


def edge_precision_recall_f1(candidate: np.ndarray, target: np.ndarray, tolerance: int = 1) -> tuple[float, float, float]:
    cand = edge_mask(candidate)
    tgt = edge_mask(target)
    if not cand.any() and not tgt.any():
        return 1.0, 1.0, 1.0
    precision_denom = int(cand.sum())
    recall_denom = int(tgt.sum())
    if precision_denom == 0 or recall_denom == 0:
        return 0.0, 0.0, 0.0
    cand_d = _dilate_radius(cand, tolerance)
    tgt_d = _dilate_radius(tgt, tolerance)
    precision = float((cand & tgt_d).sum()) / float(precision_denom)
    recall = float((tgt & cand_d).sum()) / float(recall_denom)
    f1 = 0.0 if precision + recall <= 0.0 else float((2.0 * precision * recall) / (precision + recall))
    return precision, recall, f1


def ssim_index(a: np.ndarray, b: np.ndarray) -> float:
    ya = rgb_to_luma(a).astype(np.float64)
    yb = rgb_to_luma(b).astype(np.float64)
    if ya.size == 0:
        return 1.0
    c1 = 6.5025
    c2 = 58.5225
    mux = float(ya.mean())
    muy = float(yb.mean())
    vx = float(((ya - mux) ** 2).mean())
    vy = float(((yb - muy) ** 2).mean())
    cov = float(((ya - mux) * (yb - muy)).mean())
    denom = (mux * mux + muy * muy + c1) * (vx + vy + c2)
    if denom == 0.0:
        return 1.0
    value = ((2.0 * mux * muy + c1) * (2.0 * cov + c2)) / denom
    return float(max(-1.0, min(1.0, value)))
