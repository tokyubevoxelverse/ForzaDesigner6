from __future__ import annotations

import torch

from forza_abyss_painter.shapegen.gpu.device import DTYPE


def rasterize_rotated_ellipses(params: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Batched rasterization of K rotated ellipses to a (K, H, W) HARD mask in {0, 1}.

    `params` is (K, 5): columns are (x, y, rx, ry, angle_degrees). Tensor must already live
    on the target device (call `.to(get_device())` before passing in).

    Hard edge (d2 <= 1) at every size — this is the COMMIT/SCORING rasterizer and matches
    both the CPU engine and how FH6 renders vinyl ellipses (crisp vector edges). A soft
    anti-aliased variant (`_ellipse_soft` in shapes_gpu) is used only inside gradient
    refinement, where differentiability is needed. Committing soft edges instead would
    accumulate ~1px fringes across thousands of overlapping translucent shapes into visible
    haze/bleed, and would also overstate smoothness vs. the actual in-game render.
    """
    if params.ndim != 2 or params.shape[1] != 5:
        raise ValueError(f"params must be (K, 5); got {tuple(params.shape)}")
    device = params.device
    K = params.shape[0]
    if K == 0:
        return torch.zeros((0, h, w), dtype=DTYPE, device=device)

    x = params[:, 0].view(K, 1, 1)
    y = params[:, 1].view(K, 1, 1)
    rx = params[:, 2].clamp(min=1e-3).view(K, 1, 1)
    ry = params[:, 3].clamp(min=1e-3).view(K, 1, 1)
    angle_rad = torch.deg2rad(params[:, 4]).view(K, 1, 1)
    cos_a = torch.cos(angle_rad)
    sin_a = torch.sin(angle_rad)

    ys = torch.arange(h, dtype=DTYPE, device=device).view(1, h, 1)
    xs = torch.arange(w, dtype=DTYPE, device=device).view(1, 1, w)

    dx = xs - x   # (K, 1, W)
    dy = ys - y   # (K, H, 1)
    xr = cos_a * dx + sin_a * dy
    yr = -sin_a * dx + cos_a * dy
    nx = xr / rx
    ny = yr / ry
    d2 = nx * nx + ny * ny   # (K, H, W)
    return (d2 <= 1.0).to(DTYPE)
