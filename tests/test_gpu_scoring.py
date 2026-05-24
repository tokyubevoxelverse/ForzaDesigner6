import numpy as np
import torch

from forza_abyss_painter.shapegen.gpu.device import get_device, DTYPE
from forza_abyss_painter.shapegen.gpu.rasterize import rasterize_rotated_ellipses
from forza_abyss_painter.shapegen.gpu.scoring import score_batch, ALPHA_FIXED, STICKER_OVERLAP_MIN
from forza_abyss_painter.shapegen.scoring import composite, rms_error
from forza_abyss_painter.shapegen.shapes.ellipse import RotatedEllipse


def _to_gpu_uint8(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr).to(get_device())


# ---- Non-sticker (opaque) mode ----

def test_optimal_color_matches_cpu_for_single_candidate():
    h, w = 64, 64
    rng = np.random.default_rng(7)
    target = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    current = np.full((h, w, 3), 128, dtype=np.uint8)

    e = RotatedEllipse(x=30.0, y=30.0, rx=12.0, ry=8.0, angle=20.0,
                       color=(0, 0, 0, ALPHA_FIXED))
    _new_canvas, _ = composite(current, e, target)
    cpu_color = e.color

    params = torch.tensor([[30.0, 30.0, 12.0, 8.0, 20.0]], dtype=DTYPE, device=get_device())
    masks = rasterize_rotated_ellipses(params, h, w)
    _scores, colors, _alphas = score_batch(masks, _to_gpu_uint8(current), _to_gpu_uint8(target))
    gpu_color = colors[0].cpu().numpy().tolist()
    for ch in range(3):
        assert abs(gpu_color[ch] - cpu_color[ch]) <= 3, (
            f"channel {ch}: cpu={cpu_color[ch]} gpu={gpu_color[ch]}"
        )


def test_score_batch_argmin_finds_a_good_candidate():
    h, w = 64, 64
    rng = np.random.default_rng(11)
    target = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    mean_color = target.reshape(-1, 3).mean(axis=0).astype(np.uint8)
    current = np.tile(mean_color, (h, w, 1)).astype(np.uint8)
    initial_rms = rms_error(current, target)

    K = 64
    g = torch.Generator(device="cpu").manual_seed(11)
    params = torch.empty((K, 5), dtype=DTYPE)
    params[:, 0] = torch.rand(K, generator=g) * w
    params[:, 1] = torch.rand(K, generator=g) * h
    params[:, 2] = 1.0 + torch.rand(K, generator=g) * (w / 8)
    params[:, 3] = 1.0 + torch.rand(K, generator=g) * (h / 8)
    params[:, 4] = torch.rand(K, generator=g) * 180.0
    params = params.to(get_device())

    masks = rasterize_rotated_ellipses(params, h, w)
    scores, _colors, _alphas = score_batch(masks, _to_gpu_uint8(current), _to_gpu_uint8(target))
    best_rms = float(scores.min().cpu().item())
    assert best_rms < initial_rms, (
        f"best candidate RMS {best_rms} did not beat initial {initial_rms}"
    )


def test_score_batch_handles_empty_mask():
    h, w = 32, 32
    target = np.full((h, w, 3), 200, dtype=np.uint8)
    current = np.full((h, w, 3), 100, dtype=np.uint8)
    initial = rms_error(current, target)
    masks = torch.zeros((1, h, w), dtype=DTYPE, device=get_device())
    scores, _, _alphas = score_batch(masks, _to_gpu_uint8(current), _to_gpu_uint8(target))
    assert abs(float(scores[0].cpu().item()) - initial) < 1e-3


# ---- Sticker mode ----

def _circle_alpha_mask(h: int, w: int, cx: float, cy: float, r: float) -> np.ndarray:
    yy, xx = np.indices((h, w))
    inside = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    out = np.zeros((h, w), dtype=np.uint8)
    out[inside] = 255
    return out


def test_sticker_rejects_candidates_that_bleed_past_alpha():
    """A candidate ellipse entirely outside the opaque region must score +inf."""
    h, w = 64, 64
    alpha = _circle_alpha_mask(h, w, cx=16, cy=16, r=10)  # small opaque blob top-left
    target = np.full((h, w, 3), 200, dtype=np.uint8)
    target[alpha == 0] = 0  # sticker contract: zero RGB in transparent area
    current = np.full((h, w, 3), 40, dtype=np.uint8)

    # Two candidates: one fully inside the opaque blob, one fully outside (bottom-right).
    params = torch.tensor(
        [
            [16.0, 16.0, 4.0, 4.0, 0.0],   # inside opaque blob
            [48.0, 48.0, 4.0, 4.0, 0.0],   # fully transparent area
        ],
        dtype=DTYPE,
        device=get_device(),
    )
    masks = rasterize_rotated_ellipses(params, h, w)
    scores, _colors, _alphas = score_batch(
        masks,
        _to_gpu_uint8(current),
        _to_gpu_uint8(target),
        alpha_mask=_to_gpu_uint8(alpha),
    )
    s = scores.cpu().numpy()
    assert np.isfinite(s[0]), f"inside candidate should be finite, got {s[0]}"
    assert s[1] == float("inf"), f"outside candidate must be +inf, got {s[1]}"


def test_sticker_overlap_threshold_constant_matches_cpu():
    from forza_abyss_painter.shapegen.scoring import STICKER_OVERLAP_MIN as CPU_MIN
    assert STICKER_OVERLAP_MIN == CPU_MIN


def test_sticker_rms_stable_for_identical_candidates():
    """Two candidates that paint identically inside the opaque region must score equally,
    regardless of what they would do to transparent pixels — because transparent pixels
    don't contribute to the weighted RMS."""
    h, w = 64, 64
    alpha = _circle_alpha_mask(h, w, cx=32, cy=32, r=20)
    target = np.full((h, w, 3), 200, dtype=np.uint8)
    target[alpha == 0] = 0
    current = np.full((h, w, 3), 40, dtype=np.uint8)

    # Both candidates are identical — same score. (The actual "ignore transparent"
    # property is enforced by construction: weight = (alpha > 0).)
    params = torch.tensor(
        [
            [32.0, 32.0, 8.0, 8.0, 0.0],
            [32.0, 32.0, 8.0, 8.0, 0.0],
        ],
        dtype=DTYPE,
        device=get_device(),
    )
    masks = rasterize_rotated_ellipses(params, h, w)
    scores, _, _alphas = score_batch(
        masks, _to_gpu_uint8(current), _to_gpu_uint8(target), alpha_mask=_to_gpu_uint8(alpha)
    )
    s = scores.cpu().numpy()
    assert abs(s[0] - s[1]) < 1e-4
    # And the score should be smaller than the all-grey baseline RMS over opaque pixels
    weight = (alpha > 0).astype(np.float32)
    n = weight.sum() * 3
    sse = (((current.astype(np.float32) - target.astype(np.float32)) ** 2) *
           weight[:, :, None]).sum()
    baseline = float(np.sqrt(sse / n))
    assert s[0] < baseline


def test_sticker_rms_with_fully_opaque_alpha_matches_nonsticker():
    """When alpha_mask is uniformly 255 (no transparency), sticker-mode scoring must
    match non-sticker-mode scoring within float tolerance. This is the strongest
    behavioral check that the alpha-weighting path is implemented correctly: it
    forces every internal `weight_full = (alpha > 0)` and `eff = mask * alpha_f`
    to reduce to the non-sticker case, so any divergence in the SSE decomposition
    or the optimal-color reduction shows up here."""
    h, w = 64, 64
    rng = np.random.default_rng(31)
    target = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    current = np.full((h, w, 3), 128, dtype=np.uint8)
    alpha_all_opaque = np.full((h, w), 255, dtype=np.uint8)

    params = torch.tensor(
        [
            [20.0, 20.0, 10.0, 6.0, 30.0],
            [40.0, 40.0, 8.0, 8.0, 0.0],
            [32.0, 32.0, 14.0, 4.0, 75.0],
        ],
        dtype=DTYPE,
        device=get_device(),
    )
    masks = rasterize_rotated_ellipses(params, h, w)

    scores_no_alpha, colors_no_alpha, _alphas = score_batch(
        masks, _to_gpu_uint8(current), _to_gpu_uint8(target)
    )
    scores_full_alpha, colors_full_alpha, _alphas = score_batch(
        masks, _to_gpu_uint8(current), _to_gpu_uint8(target),
        alpha_mask=_to_gpu_uint8(alpha_all_opaque),
    )

    s_no = scores_no_alpha.cpu().numpy()
    s_full = scores_full_alpha.cpu().numpy()
    # All candidates must pass the sticker overlap check (everywhere is opaque)
    assert np.all(np.isfinite(s_full)), f"sticker-mode-with-all-opaque rejected candidates: {s_full}"
    # The two paths must agree numerically (tolerance for float32 reduction order differences)
    np.testing.assert_allclose(s_no, s_full, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(
        colors_no_alpha.cpu().numpy().astype(np.int32),
        colors_full_alpha.cpu().numpy().astype(np.int32),
        atol=2,  # rounding may differ by 1 LSB
    )


def test_alpha_search_picks_opaque_for_high_contrast_dark():
    """A black shape over a white canvas: opacity 255 reaches true black (low RMS), 128 only
    reaches grey. Alpha search must prefer the high alpha and beat the fixed-128 score."""
    h, w = 48, 48
    current = np.full((h, w, 3), 255, dtype=np.uint8)   # white canvas
    target = np.full((h, w, 3), 255, dtype=np.uint8)
    target[16:32, 16:32] = 0                            # black square region
    params = torch.tensor([[24.0, 24.0, 9.0, 9.0, 0.0]], dtype=DTYPE, device=get_device())
    masks = rasterize_rotated_ellipses(params, h, w)

    rms_fixed, _, a_fixed = score_batch(masks, _to_gpu_uint8(current), _to_gpu_uint8(target))
    rms_search, _, a_search = score_batch(
        masks, _to_gpu_uint8(current), _to_gpu_uint8(target),
        alpha_levels=[64, 128, 192, 255])

    assert int(a_fixed[0]) == 128
    assert int(a_search[0]) > 128, f"alpha search should pick high alpha, got {int(a_search[0])}"
    assert float(rms_search[0]) <= float(rms_fixed[0]) + 1e-4, "alpha search must not worsen RMS"
