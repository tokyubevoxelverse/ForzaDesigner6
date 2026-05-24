import numpy as np

from forza_abyss_painter.shapegen.gpu.engine import GPUConfig, run_gpu
from forza_abyss_painter.shapegen.scoring import rms_error


def _target_with_blue_square(size: int = 64) -> np.ndarray:
    arr = np.full((size, size, 3), 200, dtype=np.uint8)
    arr[16:48, 16:48] = (20, 30, 240)
    return arr


def test_gpu_engine_reduces_rms_and_returns_shapes_and_canvas():
    target = _target_with_blue_square(64)
    cfg = GPUConfig(
        num_shapes=20, random_samples=128, mutation_rounds=4,
        mutations_per_round=64, seed=42,
    )
    shapes, final_canvas = run_gpu(target, cfg)

    assert len(shapes) == cfg.num_shapes
    assert final_canvas.shape == target.shape
    assert final_canvas.dtype == np.uint8

    s0 = shapes[0]
    assert s0["type"] == "rotated_ellipse"
    for key in ("x", "y", "rx", "ry", "angle", "color"):
        assert key in s0
    assert len(s0["color"]) == 4

    mean_color = target.reshape(-1, 3).mean(axis=0).astype(np.uint8)
    baseline = np.tile(mean_color, (target.shape[0], target.shape[1], 1)).astype(np.uint8)
    initial = rms_error(baseline, target)
    final = rms_error(final_canvas, target)
    assert final < initial, f"GPU engine did not reduce RMS (initial={initial}, final={final})"


def test_gpu_engine_is_deterministic_with_same_seed():
    target = _target_with_blue_square(32)
    cfg = GPUConfig(num_shapes=5, random_samples=32, mutation_rounds=2,
                    mutations_per_round=16, seed=7)
    s1, _ = run_gpu(target, cfg)
    s2, _ = run_gpu(target, cfg)
    # Boundary masks are appended deterministically from (w, h) so they don't carry
    # ellipse-specific keys (rx/ry). Compare only the drawable ellipses for those fields.
    d1 = s1
    d2 = s2
    for key in ("x", "y", "rx", "ry", "angle"):
        assert [d[key] for d in d1] == [d[key] for d in d2], f"seed leak on {key}"
    assert [d["color"] for d in d1] == [d["color"] for d in d2], "seed leak on color"


def _circle_target_and_alpha(size: int, cx: int, cy: int, r: int,
                             color: tuple[int, int, int]):
    """Build a sticker-like target: opaque blob with `color` over zero-RGB transparent area."""
    yy, xx = np.indices((size, size))
    inside = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    alpha = np.where(inside, 255, 0).astype(np.uint8)
    target = np.zeros((size, size, 3), dtype=np.uint8)
    target[inside] = color
    return target, alpha


def test_gpu_engine_sticker_mode_paints_only_inside_opaque_region():
    """In sticker mode, the final canvas must be unchanged (grey 40) in transparent areas."""
    target, alpha = _circle_target_and_alpha(64, cx=32, cy=32, r=14, color=(220, 30, 30))
    cfg = GPUConfig(num_shapes=15, random_samples=128, mutation_rounds=3,
                    mutations_per_round=64, seed=99)
    shapes, final_canvas = run_gpu(target, cfg, alpha_mask=alpha)

    # Transparent pixels (alpha==0) must remain the sticker canvas init color (40).
    transparent = alpha == 0
    grey_region = final_canvas[transparent]
    assert grey_region.size > 0
    # Tolerate a tiny ramp from soft-edge bleed at the silhouette: 1-px ring max
    mean_diff = float(np.mean(np.abs(grey_region.astype(np.int32) - 40)))
    assert mean_diff < 5.0, f"transparent area was painted (mean delta from grey={mean_diff})"

    # Opaque pixels should be closer to red than to grey
    opaque = alpha > 0
    mean_red = float(final_canvas[opaque][:, 0].mean())
    mean_blue = float(final_canvas[opaque][:, 2].mean())
    assert mean_red > mean_blue + 30, (
        f"opaque area not red-dominant: R={mean_red} B={mean_blue}"
    )


def test_gpu_engine_sticker_mode_handles_no_opaque_region_gracefully():
    """If alpha mask is entirely transparent, engine must stop early without crashing."""
    h = w = 32
    target = np.zeros((h, w, 3), dtype=np.uint8)
    alpha = np.zeros((h, w), dtype=np.uint8)
    cfg = GPUConfig(num_shapes=50, random_samples=16, mutation_rounds=1,
                    mutations_per_round=8, seed=1)
    shapes, final_canvas = run_gpu(target, cfg, alpha_mask=alpha)
    # Engine should bail well before num_shapes (every candidate gets rejected).
    assert len(shapes) < cfg.num_shapes
    assert final_canvas.shape == target.shape


def test_gpu_engine_gradient_mode_reduces_rms():
    """Gradient refine_mode commits shapes and reduces RMS below the mean-color baseline."""
    target = _target_with_blue_square(64)
    cfg = GPUConfig(num_shapes=15, random_samples=64, seed=42,
                    refine_mode="gradient", grad_starts=8, grad_steps=30, grad_lr=2.0)
    shapes, final_canvas = run_gpu(target, cfg)
    assert len(shapes) == cfg.num_shapes
    mean_color = target.reshape(-1, 3).mean(axis=0).astype(np.uint8)
    baseline = np.tile(mean_color, (64, 64, 1)).astype(np.uint8)
    assert rms_error(final_canvas, target) < rms_error(baseline, target)


def test_gpu_engine_gradient_beats_or_matches_hillclimb():
    """At matched shape count + seed, gradient should reach <= hillclimb RMS — the whole
    reason gradient is the default refine mode."""
    target = _target_with_blue_square(64)
    common = dict(num_shapes=15, random_samples=64, seed=7)
    g_shapes, g_canvas = run_gpu(
        target, GPUConfig(refine_mode="gradient", grad_starts=8, grad_steps=40, **common))
    h_shapes, h_canvas = run_gpu(
        target, GPUConfig(refine_mode="hillclimb", mutation_rounds=8,
                          mutations_per_round=32, **common))
    g_rms = rms_error(g_canvas, target)
    h_rms = rms_error(h_canvas, target)
    # Allow a tiny slack for float noise; gradient must not be meaningfully worse.
    assert g_rms <= h_rms + 1.0, f"gradient {g_rms:.2f} should beat/match hillclimb {h_rms:.2f}"


def _angular_target(size: int = 64) -> np.ndarray:
    """A target dominated by straight edges + a sharp corner — the regime where rectangles
    and triangles should beat ellipses. A rotated solid bar on a flat background."""
    arr = np.full((size, size, 3), 210, dtype=np.uint8)
    yy, xx = np.indices((size, size))
    # diagonal bar (rectangle-ish) + a triangular wedge
    bar = (np.abs((xx - yy)) < 6)
    wedge = (yy > xx) & (xx > 10) & (yy < 50)
    arr[bar] = (20, 20, 200)
    arr[wedge] = (200, 30, 30)
    return arr


def test_gpu_engine_multitype_commits_valid_shapes():
    """Engine with all 3 kinds enabled commits shapes whose JSON matches each type's schema."""
    target = _angular_target(64)
    cfg = GPUConfig(num_shapes=20, random_samples=120, seed=3,
                    refine_mode="gradient", grad_starts=6, grad_steps=25,
                    shape_types=["rotated_ellipse", "triangle", "rotated_rectangle"])
    shapes, _ = run_gpu(target, cfg)
    assert len(shapes) == 20
    seen = set()
    schema = {
        "rotated_ellipse": {"x", "y", "rx", "ry", "angle", "color", "type"},
        "rotated_rectangle": {"x", "y", "hw", "hh", "angle", "color", "type"},
        "triangle": {"x1", "y1", "x2", "y2", "x3", "y3", "color", "type"},
    }
    drawable_shapes = shapes
    for s in drawable_shapes:
        assert s["type"] in schema, f"unexpected type {s['type']}"
        assert set(s.keys()) == schema[s["type"]], f"{s['type']} keys {set(s.keys())}"
        assert len(s["color"]) == 4
        seen.add(s["type"])
    # On an angular target we expect the search to pick at least one non-ellipse type.
    assert seen - {"rotated_ellipse"}, f"no rect/triangle chosen; types={seen}"


def test_gpu_engine_multitype_beats_ellipse_only_on_angular_target():
    """All-types should reach <= ellipse-only RMS on a straight-edge / sharp-corner target."""
    target = _angular_target(64)
    common = dict(num_shapes=25, random_samples=120, seed=11,
                  refine_mode="gradient", grad_starts=6, grad_steps=30)
    _, ell_canvas = run_gpu(target, GPUConfig(shape_types=["rotated_ellipse"], **common))
    _, multi_canvas = run_gpu(
        target,
        GPUConfig(shape_types=["rotated_ellipse", "triangle", "rotated_rectangle"], **common))
    ell_rms = rms_error(ell_canvas, target)
    multi_rms = rms_error(multi_canvas, target)
    assert multi_rms <= ell_rms + 0.5, (
        f"multi-type {multi_rms:.2f} should beat/match ellipse-only {ell_rms:.2f}")


def test_gpu_engine_gradient_sticker_paints_only_inside_opaque():
    """Gradient + sticker: transparent area stays grey (40), opaque area gets painted."""
    size = 64
    yy, xx = np.indices((size, size))
    inside = (xx - 32) ** 2 + (yy - 32) ** 2 <= 16 * 16
    alpha = np.where(inside, 255, 0).astype(np.uint8)
    target = np.zeros((size, size, 3), dtype=np.uint8)
    target[inside] = (220, 40, 40)
    cfg = GPUConfig(num_shapes=12, random_samples=128, seed=99,
                    refine_mode="gradient", grad_starts=8, grad_steps=40)
    shapes, final_canvas = run_gpu(target, cfg, alpha_mask=alpha)
    assert len(shapes) > 0, "gradient sticker committed no shapes"
    grey_delta = float(np.mean(np.abs(final_canvas[alpha == 0].astype(np.int32) - 40)))
    assert grey_delta < 5.0, f"paint bled into transparent area (delta {grey_delta})"
    red_dom = float(final_canvas[inside][:, 0].mean()) - float(final_canvas[inside][:, 2].mean())
    assert red_dom > 30, f"opaque region not red-dominant ({red_dom})"


def test_gpu_engine_edge_weight_prioritizes_high_contrast_detail():
    """With a tight shape budget split between a large SMOOTH gradient (lots of total error,
    low per-pixel contrast) and tiny SHARP 'eyes' (little area, high contrast), uniform loss
    spends shapes smoothing the gradient and neglects the eyes. Edge weighting boosts the
    high-contrast eye boundaries so the search spends shapes there → lower eye-region error."""
    size = 96
    ramp = np.linspace(150, 230, size).astype(np.uint8)
    target = np.repeat(ramp[None, :], size, axis=0)[:, :, None].repeat(3, axis=2).copy()
    target[40:48, 30:40] = (10, 10, 10)
    target[40:48, 56:66] = (10, 10, 10)
    eye = np.zeros((size, size), dtype=bool)
    eye[40:48, 30:40] = True
    eye[40:48, 56:66] = True

    common = dict(num_shapes=12, random_samples=120, seed=4, refine_mode="gradient",
                  grad_starts=8, grad_steps=30, shape_types=["rotated_ellipse"])
    _, flat_canvas = run_gpu(target, GPUConfig(edge_strength=0.0, **common))
    _, edge_canvas = run_gpu(target, GPUConfig(edge_strength=4.0, **common))

    def eye_err(canvas):
        d = canvas[eye].astype(np.int32) - target[eye].astype(np.int32)
        return float(np.sqrt((d * d).mean()))

    assert eye_err(edge_canvas) < eye_err(flat_canvas), (
        f"edge weighting should lower eye-region error: "
        f"edge={eye_err(edge_canvas):.1f} vs flat={eye_err(flat_canvas):.1f}")


def test_gpu_engine_posterize_reduces_target_color_count():
    """posterize_levels quantizes the target so the rendered output uses few distinct levels
    per channel — confirms the quantization is actually applied before fitting."""
    from forza_abyss_painter.shapegen.gpu.engine import _posterize
    rng = np.random.default_rng(0)
    target = rng.integers(0, 256, size=(40, 40, 3), dtype=np.uint8)
    q = _posterize(target, 4)
    # 4 levels per channel → at most 4 distinct values per channel
    for ch in range(3):
        assert len(np.unique(q[:, :, ch])) <= 4, f"channel {ch} not posterized to 4 levels"
    # off cases pass through unchanged
    assert np.array_equal(_posterize(target, 0), target)
    assert np.array_equal(_posterize(target, 256), target)


def test_gpu_engine_runs_with_posterize():
    """Engine runs end-to-end with posterization on and still reduces RMS."""
    target = _target_with_blue_square(64)
    cfg = GPUConfig(num_shapes=12, random_samples=64, seed=1, refine_mode="gradient",
                    grad_starts=6, grad_steps=25, posterize_levels=8)
    shapes, canvas = run_gpu(target, cfg)
    assert len(shapes) == 12
    mean = np.tile(target.reshape(-1, 3).mean(0).astype(np.uint8), (64, 64, 1))
    assert rms_error(canvas, target) < rms_error(mean, target)


def test_gpu_engine_bbox_local_runs_and_reduces_rms():
    """bbox_local random scoring (ellipse, gradient) commits shapes and reduces RMS."""
    target = _target_with_blue_square(64)
    cfg = GPUConfig(num_shapes=15, random_samples=256, seed=42, refine_mode="gradient",
                    grad_starts=8, grad_steps=25, shape_types=["rotated_ellipse"],
                    bbox_local=True, bbox_crop_max=64)
    shapes, canvas = run_gpu(target, cfg)
    assert len(shapes) == 15
    assert all(s["type"] == "rotated_ellipse" for s in shapes)
    mean = np.tile(target.reshape(-1, 3).mean(0).astype(np.uint8), (64, 64, 1))
    assert rms_error(canvas, target) < rms_error(mean, target)


def test_gpu_engine_bbox_local_sticker_paints_inside():
    """bbox_local in sticker mode keeps paint inside the opaque region."""
    size = 64
    yy, xx = np.indices((size, size))
    inside = (xx - 32) ** 2 + (yy - 32) ** 2 <= 16 * 16
    alpha = np.where(inside, 255, 0).astype(np.uint8)
    target = np.zeros((size, size, 3), dtype=np.uint8)
    target[inside] = (220, 40, 40)
    cfg = GPUConfig(num_shapes=12, random_samples=256, seed=7, refine_mode="gradient",
                    grad_starts=8, grad_steps=25, shape_types=["rotated_ellipse"],
                    bbox_local=True)
    shapes, canvas = run_gpu(target, cfg, alpha_mask=alpha)
    assert len(shapes) > 0
    grey_delta = float(np.mean(np.abs(canvas[alpha == 0].astype(np.int32) - 40)))
    assert grey_delta < 5.0, f"bbox sticker bled into transparent area ({grey_delta})"


def test_gpu_engine_joint_polish_runs_and_keeps_valid_output():
    """Joint-polish after greedy returns the same count of valid ellipse shapes and a canvas
    that's at least as good as (usually better than) the pre-polish greedy result."""
    target = _target_with_blue_square(64)
    base = dict(num_shapes=20, random_samples=128, seed=42, refine_mode="gradient",
                grad_starts=8, grad_steps=25, shape_types=["rotated_ellipse"])
    _, greedy_canvas = run_gpu(target, GPUConfig(**base))
    polished_shapes, polished_canvas = run_gpu(
        target, GPUConfig(joint_polish_steps=40, joint_polish_lr=1.5, **base))
    assert len(polished_shapes) == 20
    for s in polished_shapes:
        assert s["type"] == "rotated_ellipse"
        assert len(s["color"]) == 4 and 16 <= s["color"][3] <= 255
    # Joint-polish optimizes the full-image fit; hard-rendered result should not be worse
    # than greedy by more than a small soft-vs-hard slack.
    assert rms_error(polished_canvas, target) <= rms_error(greedy_canvas, target) + 2.0


def test_gpu_engine_joint_polish_sticker_paints_inside():
    size = 64
    yy, xx = np.indices((size, size))
    inside = (xx - 32) ** 2 + (yy - 32) ** 2 <= 16 * 16
    alpha = np.where(inside, 255, 0).astype(np.uint8)
    target = np.zeros((size, size, 3), dtype=np.uint8)
    target[inside] = (220, 40, 40)
    cfg = GPUConfig(num_shapes=12, random_samples=128, seed=7, refine_mode="gradient",
                    grad_starts=8, grad_steps=20, shape_types=["rotated_ellipse"],
                    joint_polish_steps=30)
    shapes, canvas = run_gpu(target, cfg, alpha_mask=alpha)
    assert len(shapes) > 0
    grey_delta = float(np.mean(np.abs(canvas[alpha == 0].astype(np.int32) - 40)))
    assert grey_delta < 6.0, f"joint-polish sticker bled into transparent area ({grey_delta})"


def test_edge_weight_map_is_chroma_aware():
    """A boundary between two regions of EQUAL luminance but different color (a hue shift)
    has ~zero grayscale gradient but must register in the chroma-aware edge map — this is
    what lets edge weighting prioritize faint colored detail like tattoos on skin."""
    from forza_abyss_painter.shapegen.gpu.engine import _edge_weight_map
    import torch
    from forza_abyss_painter.shapegen.gpu.device import get_device
    size = 32
    # left half (200,100,100), right half (100,200,100): channel means 133.3 both → equal
    # luminance, so a grayscale-mean gradient at the seam is ~0.
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    arr[:, :16] = (200, 100, 100)
    arr[:, 16:] = (100, 200, 100)
    t = torch.from_numpy(arr).to(get_device())
    ew = _edge_weight_map(t, strength=4.0).cpu().numpy()
    # The seam column (around x=15..16) must be weighted above the flat-region baseline of 1.0
    seam = ew[:, 14:18].max()
    flat = ew[:, :10].max()
    assert seam > 1.5, f"chroma edge not detected at seam (seam={seam:.2f})"
    assert flat < 1.05, f"flat region should stay ~1.0 (got {flat:.2f})"

    # Sanity: a pure grayscale-mean gradient at this seam would be ~0 (means are equal).
    gray = arr.astype(np.float32).mean(axis=2)
    gray_grad_seam = float(np.abs(gray[:, 16] - gray[:, 14]).max())
    assert gray_grad_seam < 1.0, "test setup invalid: luminance should be ~equal across seam"


def test_gpu_engine_alpha_threshold_drops_soft_fringe():
    """alpha_threshold binarizes the sticker mask, excluding the anti-aliased fringe (the
    white-halo source). With threshold on, semi-transparent edge pixels (alpha < threshold)
    must be treated as fully transparent → no paint there."""
    size = 48
    yy, xx = np.indices((size, size))
    r = np.sqrt((xx - 24.0) ** 2 + (yy - 24.0) ** 2)
    # opaque core (r<14, alpha 255), soft AA ring 14<=r<18 with LOW alpha (~80), else 0
    alpha = np.zeros((size, size), dtype=np.uint8)
    alpha[r < 18] = 80          # fringe ring
    alpha[r < 14] = 255         # solid core
    target = np.zeros((size, size, 3), dtype=np.uint8)
    target[r < 18] = (240, 240, 240)   # light RGB everywhere inside (incl. fringe) — halo source

    cfg = GPUConfig(num_shapes=10, random_samples=128, seed=1, refine_mode="gradient",
                    grad_starts=6, grad_steps=20, shape_types=["rotated_ellipse"],
                    alpha_threshold=128)
    shapes, canvas = run_gpu(target, cfg, alpha_mask=alpha)
    # The fringe ring (alpha 80 < 128) must stay the grey substrate (40), not painted light.
    fringe = (r >= 14) & (r < 18)
    fringe_val = float(canvas[fringe].mean())
    assert fringe_val < 80, f"fringe should stay near grey 40 (got {fringe_val}) — halo not removed"
