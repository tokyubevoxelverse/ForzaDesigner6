import numpy as np
import torch

from forza_abyss_painter.shapegen.gpu.device import get_device, DTYPE
from forza_abyss_painter.shapegen.gpu.rasterize import rasterize_rotated_ellipses
from forza_abyss_painter.shapegen.gpu.scoring import score_batch
from forza_abyss_painter.shapegen.gpu.bbox_score import crop_score_ellipse_batch


def _rand_params(K, w, h, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    p = torch.empty((K, 5), dtype=DTYPE)
    p[:, 0] = torch.rand(K, generator=g) * w
    p[:, 1] = torch.rand(K, generator=g) * h
    p[:, 2] = 1.0 + torch.rand(K, generator=g) * (w / 8)
    p[:, 3] = 1.0 + torch.rand(K, generator=g) * (h / 8)
    p[:, 4] = torch.rand(K, generator=g) * 180.0
    return p.to(get_device())


def _target(size=64):
    rng = np.random.default_rng(3)
    a = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    return a


def test_bbox_argmin_matches_full_canvas_opaque():
    """The bbox-local delta argmin must select the same best candidate as the full-canvas
    RMS argmin — they rank candidates identically because pixels outside each shape's box
    are unchanged."""
    h = w = 64
    target = _target(w)
    mean = target.reshape(-1, 3).mean(0).astype(np.uint8)
    canvas = np.tile(mean, (h, w, 1)).astype(np.uint8)
    params = _rand_params(64, w, h, seed=1)
    cu = torch.from_numpy(canvas).to(get_device())
    tu = torch.from_numpy(target).to(get_device())

    masks = rasterize_rotated_ellipses(params, h, w)
    full_scores, full_colors, full_alphas = score_batch(masks, cu, tu)
    delta, colors, alphas = crop_score_ellipse_batch(params, cu, tu, max_crop_radius=256)

    assert int(full_scores.argmin()) == int(delta.argmin()), "bbox argmin != full-canvas argmin"
    # winning candidate's optimal color should match closely (same closed form)
    bi = int(delta.argmin())
    assert np.allclose(full_colors[bi].cpu().numpy(), colors[bi].cpu().numpy(), atol=2)


def test_bbox_argmin_matches_full_canvas_sticker():
    """Same equivalence under sticker mode (opaque-gated weighting + overlap rejection)."""
    h = w = 64
    yy, xx = np.indices((h, w))
    inside = (xx - 32) ** 2 + (yy - 24) ** 2 <= 16 * 16
    alpha = np.where(inside, 255, 0).astype(np.uint8)
    target = np.zeros((h, w, 3), dtype=np.uint8)
    target[inside] = (200, 60, 40)
    canvas = np.full((h, w, 3), 40, dtype=np.uint8)
    params = _rand_params(80, w, h, seed=2)

    cu = torch.from_numpy(canvas).to(get_device())
    tu = torch.from_numpy(target).to(get_device())
    au = torch.from_numpy(alpha).to(get_device())

    masks = rasterize_rotated_ellipses(params, h, w)
    full_scores, _, _ = score_batch(masks, cu, tu, alpha_mask=au)
    delta, _, _ = crop_score_ellipse_batch(params, cu, tu, alpha_t=au, max_crop_radius=256)

    # Both must reject the same candidates (+inf) and agree on the best finite candidate.
    full_rej = ~torch.isfinite(full_scores)
    bbox_rej = ~torch.isfinite(delta)
    assert torch.equal(full_rej, bbox_rej), "sticker rejection sets differ"
    if torch.isfinite(delta).any():
        assert int(full_scores.argmin()) == int(delta.argmin())


def test_bbox_alpha_search_returns_per_candidate_alpha():
    h = w = 48
    target = _target(w)
    canvas = np.full((h, w, 3), 128, dtype=np.uint8)
    params = _rand_params(32, w, h, seed=5)
    cu = torch.from_numpy(canvas).to(get_device())
    tu = torch.from_numpy(target).to(get_device())
    delta, colors, alphas = crop_score_ellipse_batch(
        params, cu, tu, alpha_levels=[96, 160, 255], max_crop_radius=256)
    assert delta.shape == (32,)
    assert colors.shape == (32, 3)
    assert set(alphas.cpu().numpy().tolist()).issubset({96, 160, 255})
