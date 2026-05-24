"""Batched GPU shape kinds: hard + differentiable-soft rasterizers, random init, gradient
clamp, and JSON serialization for each supported shape type.

All rasterizers take a (K, P) param tensor (P depends on type) already on the device and
return a (K, H, W) mask in 0..1. The HARD rasterizer is used at commit time (crisp edges);
the SOFT rasterizer is differentiable w.r.t. params and used inside gradient refinement.

JSON output keys match the CPU shapes in forza_abyss_painter/shapegen/shapes/{ellipse,rectangle,triangle}.py
so the Windows FH6 injector loads them unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from forza_abyss_painter.shapegen.gpu.device import DTYPE
from forza_abyss_painter.shapegen.gpu.rasterize import rasterize_rotated_ellipses


# ----------------------------------------------------------------------------------------
# rotated_ellipse — params (x, y, rx, ry, angle)
# ----------------------------------------------------------------------------------------

def _ellipse_init(K, w, h, gen):
    out = torch.empty((K, 5), dtype=DTYPE)
    out[:, 0] = torch.rand(K, generator=gen) * w
    out[:, 1] = torch.rand(K, generator=gen) * h
    out[:, 2] = 1.0 + torch.rand(K, generator=gen) * (w / 8)
    out[:, 3] = 1.0 + torch.rand(K, generator=gen) * (h / 8)
    out[:, 4] = torch.rand(K, generator=gen) * 180.0
    return out


def _ellipse_soft(params, h, w):
    M = params.shape[0]
    dev = params.device
    x = params[:, 0].view(M, 1, 1); y = params[:, 1].view(M, 1, 1)
    rx = params[:, 2].clamp(min=1e-3).view(M, 1, 1); ry = params[:, 3].clamp(min=1e-3).view(M, 1, 1)
    ang = torch.deg2rad(params[:, 4]).view(M, 1, 1)
    ca, sa = torch.cos(ang), torch.sin(ang)
    ys = torch.arange(h, dtype=DTYPE, device=dev).view(1, h, 1)
    xs = torch.arange(w, dtype=DTYPE, device=dev).view(1, 1, w)
    dx = xs - x; dy = ys - y
    xr = ca * dx + sa * dy; yr = -sa * dx + ca * dy
    d2 = (xr / rx) ** 2 + (yr / ry) ** 2
    ramp = (2.0 / torch.minimum(rx, ry)).clamp(min=1e-3, max=0.5)
    return ((1.0 - d2 + ramp / 2.0) / ramp).clamp(0.0, 1.0)


def _ellipse_clamp_(p, w, h):
    p[:, 0].clamp_(0.0, w - 1); p[:, 1].clamp_(0.0, h - 1)
    p[:, 2].clamp_(1.0, float(w)); p[:, 3].clamp_(1.0, float(h))
    p[:, 4].remainder_(180.0)


def _ellipse_json(row, color):
    return {
        "type": "rotated_ellipse",
        "x": round(float(row[0]), 3), "y": round(float(row[1]), 3),
        "rx": round(float(row[2]), 3), "ry": round(float(row[3]), 3),
        "angle": round(float(row[4]), 3), "color": color,
    }


# ----------------------------------------------------------------------------------------
# rotated_rectangle — params (x, y, hw, hh, angle)
# ----------------------------------------------------------------------------------------

def _rect_init(K, w, h, gen):
    out = torch.empty((K, 5), dtype=DTYPE)
    out[:, 0] = torch.rand(K, generator=gen) * w
    out[:, 1] = torch.rand(K, generator=gen) * h
    out[:, 2] = 1.0 + torch.rand(K, generator=gen) * (w / 8)
    out[:, 3] = 1.0 + torch.rand(K, generator=gen) * (h / 8)
    out[:, 4] = torch.rand(K, generator=gen) * 180.0
    return out


def _rect_dist(params, h, w):
    """Chebyshev box distance d = max(|xr|/hw, |yr|/hh); inside iff d <= 1. (K, H, W)."""
    M = params.shape[0]
    dev = params.device
    x = params[:, 0].view(M, 1, 1); y = params[:, 1].view(M, 1, 1)
    hw = params[:, 2].clamp(min=1e-3).view(M, 1, 1); hh = params[:, 3].clamp(min=1e-3).view(M, 1, 1)
    ang = torch.deg2rad(params[:, 4]).view(M, 1, 1)
    ca, sa = torch.cos(ang), torch.sin(ang)
    ys = torch.arange(h, dtype=DTYPE, device=dev).view(1, h, 1)
    xs = torch.arange(w, dtype=DTYPE, device=dev).view(1, 1, w)
    dx = xs - x; dy = ys - y
    xr = ca * dx + sa * dy; yr = -sa * dx + ca * dy
    d = torch.maximum(xr.abs() / hw, yr.abs() / hh)
    ramp = (1.0 / torch.minimum(hw, hh)).clamp(min=1e-3, max=0.5)
    return d, ramp


def _rect_hard(params, h, w):
    d, _ = _rect_dist(params, h, w)
    return (d <= 1.0).to(DTYPE)


def _rect_soft(params, h, w):
    d, ramp = _rect_dist(params, h, w)
    # Linear edge centered at d=1, ~1 normalized-ramp wide.
    return ((1.0 - d) / ramp + 0.5).clamp(0.0, 1.0)


def _rect_json(row, color):
    return {
        "type": "rotated_rectangle",
        "x": round(float(row[0]), 3), "y": round(float(row[1]), 3),
        "hw": round(float(row[2]), 3), "hh": round(float(row[3]), 3),
        "angle": round(float(row[4]), 3), "color": color,
    }


# ----------------------------------------------------------------------------------------
# triangle — params (x1, y1, x2, y2, x3, y3)
# ----------------------------------------------------------------------------------------

def _tri_init(K, w, h, gen):
    spread = max(4.0, min(w, h) / 8.0)
    cx = (torch.rand(K, generator=gen) * w).view(K, 1)
    cy = (torch.rand(K, generator=gen) * h).view(K, 1)
    out = torch.empty((K, 6), dtype=DTYPE)
    for v in range(3):
        out[:, 2 * v] = (cx[:, 0] + torch.randn(K, generator=gen) * spread)
        out[:, 2 * v + 1] = (cy[:, 0] + torch.randn(K, generator=gen) * spread)
    out[:, 0::2].clamp_(0.0, w - 1)
    out[:, 1::2].clamp_(0.0, h - 1)
    return out


def _tri_signed(params, h, w):
    """Per-pixel inside-amount: min over the 3 edges of the winding-oriented perpendicular
    distance (pixels). >=0 inside. Differentiable w.r.t. vertices. Returns (K, H, W)."""
    M = params.shape[0]
    dev = params.device
    ax = params[:, 0].view(M, 1, 1); ay = params[:, 1].view(M, 1, 1)
    bx = params[:, 2].view(M, 1, 1); by = params[:, 3].view(M, 1, 1)
    cx = params[:, 4].view(M, 1, 1); cy = params[:, 5].view(M, 1, 1)
    ys = torch.arange(h, dtype=DTYPE, device=dev).view(1, h, 1)
    xs = torch.arange(w, dtype=DTYPE, device=dev).view(1, 1, w)

    def edge(p0x, p0y, p1x, p1y):
        # signed area*2 of (p0, p1, pixel); also = perpendicular dist * edge length
        e = (p1x - p0x) * (ys - p0y) - (p1y - p0y) * (xs - p0x)
        length = torch.sqrt((p1x - p0x) ** 2 + (p1y - p0y) ** 2).clamp(min=1e-3)
        return e / length   # (M, H, W) perpendicular distance (signed)

    d_ab = edge(ax, ay, bx, by)
    d_bc = edge(bx, by, cx, cy)
    d_ca = edge(cx, cy, ax, ay)
    # winding sign from the triangle's own area (orient so inside is positive)
    area2 = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)   # (M,1,1)
    s = torch.where(area2 >= 0, torch.ones_like(area2), -torch.ones_like(area2))
    inside = torch.minimum(torch.minimum(s * d_ab, s * d_bc), s * d_ca)
    return inside


def _tri_hard(params, h, w):
    return (_tri_signed(params, h, w) >= 0.0).to(DTYPE)


def _tri_soft(params, h, w):
    inside = _tri_signed(params, h, w)
    return (inside + 0.5).clamp(0.0, 1.0)   # ~1px AA centered at the boundary


def _tri_clamp_(p, w, h):
    p[:, 0::2].clamp_(0.0, w - 1)   # x coords
    p[:, 1::2].clamp_(0.0, h - 1)   # y coords


def _tri_json(row, color):
    return {
        "type": "triangle",
        "x1": round(float(row[0]), 3), "y1": round(float(row[1]), 3),
        "x2": round(float(row[2]), 3), "y2": round(float(row[3]), 3),
        "x3": round(float(row[4]), 3), "y3": round(float(row[5]), 3),
        "color": color,
    }


# ----------------------------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ShapeKind:
    name: str
    param_count: int
    init: Callable          # (K, w, h, gen) -> (K, P) cpu tensor
    rasterize_hard: Callable  # (params, h, w) -> (K, H, W)
    rasterize_soft: Callable  # (params, h, w) -> (K, H, W) differentiable
    clamp_: Callable        # (params, w, h) -> in-place
    to_json: Callable       # (row_list, color_list) -> dict


KINDS: dict[str, ShapeKind] = {
    "rotated_ellipse": ShapeKind(
        "rotated_ellipse", 5, _ellipse_init,
        rasterize_rotated_ellipses, _ellipse_soft, _ellipse_clamp_, _ellipse_json),
    "rotated_rectangle": ShapeKind(
        "rotated_rectangle", 5, _rect_init,
        _rect_hard, _rect_soft, _ellipse_clamp_, _rect_json),
    "triangle": ShapeKind(
        "triangle", 6, _tri_init,
        _tri_hard, _tri_soft, _tri_clamp_, _tri_json),
}
