import math

import numpy as np
import torch

from forza_abyss_painter.shapegen.gpu.device import get_device, DTYPE
from forza_abyss_painter.shapegen.gpu.rasterize import rasterize_rotated_ellipses
from forza_abyss_painter.shapegen.shapes.ellipse import RotatedEllipse


def _cpu_mask(e: RotatedEllipse, h: int, w: int) -> np.ndarray:
    local, (x0, y0, x1, y1) = e.rasterize_mask(w, h)
    full = np.zeros((h, w), dtype=np.float32)
    if local.size:
        full[y0:y1, x0:x1] = local.astype(np.float32) / 255.0
    return full


def test_single_axis_aligned_ellipse_matches_cpu():
    h, w = 64, 64
    e = RotatedEllipse(x=32.0, y=32.0, rx=20.0, ry=10.0, angle=0.0)
    params = torch.tensor([[32.0, 32.0, 20.0, 10.0, 0.0]], dtype=DTYPE, device=get_device())
    gpu = rasterize_rotated_ellipses(params, h, w).cpu().numpy()
    cpu = _cpu_mask(e, h, w)
    # GPU uses smooth edge (anti-aliased) vs CPU hard threshold. Allow ~2% disagreement.
    disagreement = float(np.mean(np.abs(gpu[0] - cpu)))
    assert disagreement < 0.02, f"per-pixel disagreement too high: {disagreement}"


def test_rotated_ellipse_has_correct_area():
    h, w = 128, 128
    rx, ry = 30.0, 10.0
    for angle in [0.0, 30.0, 45.0, 90.0, 135.0]:
        params = torch.tensor(
            [[64.0, 64.0, rx, ry, angle]], dtype=DTYPE, device=get_device()
        )
        mask = rasterize_rotated_ellipses(params, h, w).cpu().numpy()[0]
        area = mask.sum()
        expected = math.pi * rx * ry
        # Hard (binary) rasterization is area-accurate only up to boundary discretization
        # (~1 pixel ring), so allow 3% rather than the soft-edge era's sub-pixel tolerance.
        assert abs(area - expected) / expected < 0.03, (
            f"angle={angle}: area {area} vs expected {expected}"
        )


def test_batch_of_distinct_ellipses_produces_distinct_masks():
    h, w = 32, 32
    params = torch.tensor(
        [
            [8.0, 8.0, 4.0, 4.0, 0.0],
            [24.0, 24.0, 4.0, 4.0, 0.0],
            [16.0, 16.0, 10.0, 2.0, 45.0],
        ],
        dtype=DTYPE,
        device=get_device(),
    )
    masks = rasterize_rotated_ellipses(params, h, w).cpu().numpy()
    assert masks.shape == (3, h, w)
    assert masks[0].sum() > 0 and masks[1].sum() > 0 and masks[2].sum() > 0
    assert not np.allclose(masks[0], masks[1])
    assert not np.allclose(masks[0], masks[2])


def test_offscreen_ellipse_produces_empty_mask():
    h, w = 32, 32
    params = torch.tensor([[-1000.0, -1000.0, 4.0, 4.0, 0.0]], dtype=DTYPE, device=get_device())
    mask = rasterize_rotated_ellipses(params, h, w).cpu().numpy()[0]
    assert mask.sum() == 0.0


def test_commit_rasterizer_is_hard_at_all_sizes():
    """The commit/scoring rasterizer is hard (binary) at every size — crisp edges that match
    how FH6 renders ellipses and avoid accumulated fringe-bleed from soft edges."""
    h, w = 32, 32
    for r in (2.0, 8.0):
        params = torch.tensor([[16.0, 16.0, r, r, 0.0]], dtype=DTYPE, device=get_device())
        mask = rasterize_rotated_ellipses(params, h, w).cpu().numpy()[0]
        distinct = set(np.unique(mask).tolist())
        assert distinct.issubset({0.0, 1.0}), f"r={r}: commit mask not binary: {sorted(distinct)}"
        assert mask.sum() > 0


def test_soft_rasterizer_is_differentiable_with_fractional_edge():
    """The gradient-only soft rasterizer keeps fractional AA values so gradients can flow."""
    from forza_abyss_painter.shapegen.gpu.shapes_gpu import KINDS
    h, w = 32, 32
    p = torch.tensor([[16.0, 16.0, 8.0, 8.0, 0.0]], dtype=DTYPE,
                     device=get_device(), requires_grad=True)
    soft = KINDS["rotated_ellipse"].rasterize_soft(p, h, w)
    arr = soft.detach().cpu().numpy()[0]
    assert np.any((arr > 0.0) & (arr < 1.0)), "soft rasterizer should have fractional AA edge"
    soft.sum().backward()
    assert p.grad is not None and p.grad.abs().sum() > 0
