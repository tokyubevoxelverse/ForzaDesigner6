from __future__ import annotations

from dataclasses import dataclass, field, replace
import time

import numpy as np
import torch

from forza_abyss_painter.shapegen.gpu.device import DTYPE, get_device
from forza_abyss_painter.shapegen.gpu.rasterize import rasterize_rotated_ellipses
from forza_abyss_painter.shapegen.gpu.scoring import ALPHA_FIXED, score_batch
from forza_abyss_painter.shapegen.gpu.shapes_gpu import KINDS, ShapeKind
from forza_abyss_painter.shapegen.gpu.bbox_score import crop_score_ellipse_batch
from forza_abyss_painter.shapegen.gpu.joint_polish import joint_polish


# Matches MAX_CONSECUTIVE_SKIPS in forza_abyss_painter/shapegen/engine.py:134
_MAX_CONSECUTIVE_SKIPS = 80

# Mutation sigmas (image-pixel units / degrees). From RotatedEllipse.mutate
# in forza_abyss_painter/shapegen/shapes/ellipse.py:112 — coarse single-axis + fine xy/angle modes.
_MUT_SIGMA_XY_COARSE = 16.0
_MUT_SIGMA_R_COARSE = 16.0
_MUT_SIGMA_ANGLE = 25.0
_MUT_SIGMA_XY_FINE = 8.0
_MUT_SIGMA_ANGLE_FINE = 15.0

# Soft penalty weight for paint bleeding outside the opaque region during gradient
# refinement. Bleed term is in pixel units; the MSE term is a per-pixel mean, so this
# needs to be O(1) to actually bite. Larger = shapes pushed harder to stay inside.
STICKER_GRAD_PENALTY = 1.0


@dataclass
class GPUConfig:
    """One-shot GPU engine config.

    refine_mode:
      "gradient"  — differentiable rasterizer + Adam (sharper, more sample-efficient)
      "hillclimb" — parallel mutation rounds (older path, kept for A/B comparison)

    Memory: score_batch materializes (K, H, W, 3) intermediates; peak ~ K*H*W*12 bytes,
    where K is random_samples (the seed batch). Gradient refinement only touches
    grad_starts candidates, so it's cheap on memory.
    """
    num_shapes: int = 500
    random_samples: int = 256
    seed: int = 0   # 0 → time-based
    refine_mode: str = "gradient"
    # Candidate shape types (gradient mode tries all and picks the best per shape).
    # hillclimb mode is rotated_ellipse-only (no per-kind mutation defined for others).
    shape_types: list[str] = field(default_factory=lambda: ["rotated_ellipse"])
    # Per-shape alpha search. None → every shape uses ALPHA_FIXED (128). A list lets each
    # shape pick the opacity that fits best — high alpha for crisp black lines / dark eye
    # detail, low alpha for smooth gradients. The chosen alpha is written to the JSON color.
    alpha_levels: list[int] | None = None
    # Edge-weighted loss. 0 → uniform loss. >0 weights error by target edge magnitude so the
    # search prioritizes high-contrast detail (eyes, lineart, boundaries) over flat regions.
    # Typical useful range 1-4; weight map is 1 + edge_strength * normalized_edge.
    edge_strength: float = 0.0
    # Posterize the target to this many color levels per channel before fitting (0 or >=256
    # = off). Flattens smooth gradients into flat bands → shapes fit cleaner, edges/lineart
    # come in crisper. This is what the geometrize-based tools do (posterizeLevels 20-100).
    posterize_levels: int = 0
    # Bounding-box-local random scoring (ellipse-only, gradient mode). Scores each random
    # candidate over its own crop instead of the full canvas → ~10x+ cheaper per sample, so
    # random_samples can go far higher (geometrize-style volume) for the same VRAM. Ignored
    # for non-ellipse kinds or hillclimb mode.
    bbox_local: bool = False
    bbox_crop_max: int = 256   # cap on crop half-size (px); bounds memory for huge shapes
    # Joint-polish: after greedy, gradient-optimize ALL shapes' geometry+color+alpha against
    # the full target for this many steps (0 = off). Breaks greedy myopia — early shapes get
    # un-stuck. Ellipse-only. Cost is one optimization run, not a per-shape search.
    joint_polish_steps: int = 0
    joint_polish_lr: float = 1.5
    # polish_purity_penalty: when > 0, joint_polish adds a per-shape MSE-equivalent
    # 'spillover' loss = mass_i × variance_under_mask_i, summed over shapes, normalized
    # by canvas pixel count. This is exactly the MSE contribution shape i would make if
    # it painted with its optimal mean color, so the penalty has the SAME UNITS as the
    # main MSE loss above.
    #
    # Tuning intuition:
    #   = 1.0  → lazy multi-color paint penalty equals its MSE saving (break-even with omission)
    #   < 1.0  → painting preferred, but homogeneous coverage strongly encouraged
    #   > 1.0  → no-paint strictly preferred to lazy paint over the same multi-color region
    #
    # Without this, plain MSE actively REWARDS painting a big ellipse across multi-color
    # content with an averaged color (a 5x5 yellow star on dark loses ~23% MSE to a lazy
    # 10x10 averaged ellipse vs no-shape). That's why polish smeared sparkles into blobs
    # on celeste-with-stars. The mass-weighted form fixes the original PR #19 formulation,
    # which used per-shape AVERAGE variance — mass-independent — so huge lazy ellipses paid
    # the same penalty as tiny ones while their MSE gradient was 1000x bigger. The penalty
    # now scales with what the shape actually paints.
    #
    # Default 0.0 = legacy MSE-only polish. Production starting point: 1.0.
    polish_purity_penalty: float = 0.0
    # polish_freeze_geometry: when True (DEFAULT), joint_polish runs Adam over (rgb, alpha)
    # ONLY — x, y, rx, ry, angle from greedy are preserved bit-identically. Closed-form RGB
    # snap-back (when lock_alpha=True) still applies. This is the production polish mode:
    # Adam-over-geometry is gradient-greedy on a parameter space ellipses don't naturally
    # live in and consistently finds degenerate exploits (size_anchor: inflated;
    # purity_penalty: collapse to rx=1). Freezing geometry eliminates the entire failure
    # mode while keeping polish's actual win — fixing Adam-overshot colors via snap-back.
    # Verified at 1000 shapes (1000/1000 geom frozen, engine↔upstream parity 0.000).
    # Set False for the legacy joint-geom-and-color polish (preserved as an escape hatch
    # for future investigations into structured shape refinement). polish_purity_penalty
    # is a no-op when freeze_geometry=True (it's a geometry-affecting term).
    polish_freeze_geometry: bool = True
    # Injector-safe modes — see docs/superpowers/specs/2026-05-22-alpha-injector-fix-design.md.
    # lock_alpha: when True, every shape's alpha is forced to 255 throughout greedy + polish,
    # matching the FH injector's binary-alpha constraint. ALPHA_LEVELS becomes [255]; the
    # joint_polish alpha tensor isn't optimized.
    lock_alpha: bool = False
    # prune_threshold: drop a shape if removing it changes full-canvas RMS by less than this
    # many uint8 units. 0.0 = no pruning. Currently unused by run_gpu; kept for forward-compat
    # with reprocess flows.
    prune_threshold: float = 0.5
    # Binarize the sticker alpha mask at this level (0 = off, keep soft alpha). Kills the white
    # halo from anti-aliased edges on stickers exported over a light background. 128 is the
    # midpoint; raise toward 160-200 if a halo persists.
    alpha_threshold: int = 0
    # hill-climb knobs (refine_mode == "hillclimb")
    mutation_rounds: int = 4
    mutations_per_round: int = 128
    # gradient knobs (refine_mode == "gradient")
    grad_starts: int = 16
    grad_steps: int = 50
    grad_lr: float = 2.0


def _random_params(K: int, w: int, h: int, gen: torch.Generator) -> torch.Tensor:
    """Sample K random ellipses on the CPU generator (deterministic), return CPU tensor."""
    out = torch.empty((K, 5), dtype=DTYPE)
    out[:, 0] = torch.rand(K, generator=gen) * w
    out[:, 1] = torch.rand(K, generator=gen) * h
    out[:, 2] = 1.0 + torch.rand(K, generator=gen) * (w / 8)
    out[:, 3] = 1.0 + torch.rand(K, generator=gen) * (h / 8)
    out[:, 4] = torch.rand(K, generator=gen) * 180.0
    return out


def _mutate(best: torch.Tensor, K: int, w: int, h: int, gen: torch.Generator) -> torch.Tensor:
    """K children of `best`, with the 4 CPU mutation modes evenly distributed.

    Splits K into 4 groups, one per single-axis (or fine-axis) mode, so a batch round
    explores each axis independently — this is what lets the search reach extreme aspect
    ratios (rx >> ry) that all-axis-coarse mutation can't find.
      0: translate (x, y)        2: rotate (angle)
      1: resize (rx, ry)         3: fine xy + angle
    """
    base = best.view(1, 5).expand(K, 5).contiguous()
    noise = torch.zeros((K, 5), dtype=DTYPE)
    per = max(1, K // 4)

    s = slice(0, per); n = s.stop - s.start
    noise[s, 0] = torch.randn(n, generator=gen) * _MUT_SIGMA_XY_COARSE
    noise[s, 1] = torch.randn(n, generator=gen) * _MUT_SIGMA_XY_COARSE

    s = slice(per, 2 * per); n = s.stop - s.start
    noise[s, 2] = torch.randn(n, generator=gen) * _MUT_SIGMA_R_COARSE
    noise[s, 3] = torch.randn(n, generator=gen) * _MUT_SIGMA_R_COARSE

    s = slice(2 * per, 3 * per); n = s.stop - s.start
    noise[s, 4] = torch.randn(n, generator=gen) * _MUT_SIGMA_ANGLE

    s = slice(3 * per, K); n = s.stop - s.start
    if n > 0:
        noise[s, 0] = torch.randn(n, generator=gen) * _MUT_SIGMA_XY_FINE
        noise[s, 1] = torch.randn(n, generator=gen) * _MUT_SIGMA_XY_FINE
        noise[s, 4] = torch.randn(n, generator=gen) * _MUT_SIGMA_ANGLE_FINE

    out = base + noise
    out[:, 0].clamp_(0.0, w - 1)
    out[:, 1].clamp_(0.0, h - 1)
    out[:, 2].clamp_(1.0, float(w))
    out[:, 3].clamp_(1.0, float(h))
    out[:, 4] = out[:, 4] % 180.0
    return out


def _composite_one(kind: ShapeKind, canvas_u8: torch.Tensor, params: torch.Tensor,
                   color_u8: torch.Tensor, h: int, w: int,
                   alpha_mask_f: torch.Tensor | None, alpha: int = ALPHA_FIXED) -> torch.Tensor:
    """Composite a single shape (of the given kind) with RGB color + `alpha` onto canvas.
    If alpha_mask_f is provided, the shape's effective mask is `mask * alpha_mask_f` so
    paint never lands in transparent areas (the grey canvas init stays visible there).
    """
    mask = kind.rasterize_hard(params.view(1, -1), h, w)[0]   # (H, W)
    if alpha_mask_f is not None:
        mask = mask * alpha_mask_f
    a = float(alpha) / 255.0
    cur = canvas_u8.to(DTYPE)
    src = color_u8.to(DTYPE).view(1, 1, 3)
    m = mask.unsqueeze(-1)
    blended = m * (a * src + (1.0 - a) * cur) + (1.0 - m) * cur
    return blended.clamp(0.0, 255.0).round().to(torch.uint8)


# --- Gradient-based refinement (differentiable rasterizer + Adam) ---

def _posterize(target_rgb: np.ndarray, levels: int) -> np.ndarray:
    """Quantize an (H, W, 3) uint8 image to `levels` evenly-spaced values per channel.
    levels<2 or >=256 returns the image unchanged. Flattens gradients into bands so the
    greedy shape fit produces cleaner flat regions and crisper edges (geometrize does this).
    """
    if levels < 2 or levels >= 256:
        return target_rgb
    f = target_rgb.astype(np.float32) / 255.0
    q = np.round(f * (levels - 1)) / (levels - 1) * 255.0
    return q.round().clip(0, 255).astype(np.uint8)


def _edge_weight_map(target_u8: torch.Tensor, strength: float) -> torch.Tensor:
    """(H, W, 3) uint8 target → (H, W) per-pixel loss weight = 1 + strength * norm_edge.

    CHROMA-AWARE: the gradient is computed per RGB channel and combined as the L2 over both
    spatial directions AND channels — `sqrt(sum_c(gx_c^2 + gy_c^2))`. This catches *color*
    edges that a luminance-only (grayscale-mean) gradient misses: a feature that shifts hue
    but not brightness (e.g. a faint tattoo on skin) cancels in the channel mean but shows
    up strongly per channel. It still fully captures luminance edges (those appear in all
    channels). Normalized by the 99th percentile; high near eyes/lineart/colored detail.
    """
    t = target_u8.to(DTYPE)                                      # (H, W, 3)
    gx = torch.zeros_like(t); gy = torch.zeros_like(t)
    gx[:, 1:-1, :] = (t[:, 2:, :] - t[:, :-2, :]) * 0.5
    gy[1:-1, :, :] = (t[2:, :, :] - t[:-2, :, :]) * 0.5
    edge = torch.sqrt((gx * gx + gy * gy).sum(dim=2))            # (H, W) over channels
    flat = edge.flatten()
    # 99th percentile via kthvalue (quantile can be unsupported / slow on some backends)
    k = max(1, int(0.99 * flat.numel()))
    norm = torch.kthvalue(flat, k).values.clamp(min=1e-3)
    edge_norm = (edge / norm).clamp(0.0, 1.0)
    return 1.0 + float(strength) * edge_norm


def _optimal_color_batch(masks, cur_f, tgt_f, alpha_mask_f, a):
    """Closed-form optimal RGB per candidate (M, 3), float 0..255. Caller detaches."""
    if alpha_mask_f is not None:
        eff = masks * alpha_mask_f.unsqueeze(0)
    else:
        eff = masks
    m = eff.unsqueeze(-1)
    weight = eff.sum(dim=(1, 2)).clamp(min=1e-3)
    sum_mt = (m * tgt_f).sum(dim=(1, 2))
    sum_mc = (m * cur_f).sum(dim=(1, 2))
    return ((sum_mt - (1.0 - a) * sum_mc) / (a * weight.unsqueeze(-1))).clamp(0.0, 255.0)


def _refine_gradient(kind: ShapeKind, init_params, canvas_u8, target_u8, alpha_t,
                     alpha_mask_f, h, w, steps, lr, alpha_levels=None, edge_weight=None):
    """Multi-start Adam refinement for one shape kind. init_params is (M, P). Returns the
    single best (params (P,), color_u8 (3,), alpha int, score float).

    Geometry is refined at ALPHA_FIXED (alpha mostly affects intensity, not where a shape
    should sit). Color is recomputed closed-form each step and detached (alternating
    optimization). The final selection pools init + refined and picks the global best via
    the real hard-mask, sticker-aware scorer WITH alpha search — so the committed shape
    gets the opacity that fits best. Gradient descent can never do worse than the random
    seed (the seed candidates are in the final pool).
    """
    a = ALPHA_FIXED / 255.0
    cur_f = canvas_u8.to(DTYPE)
    tgt_f = target_u8.to(DTYPE)
    M = init_params.shape[0]

    # Per-pixel loss weight = (alpha gate, if sticker) × (edge emphasis, if enabled).
    lw_2d = None
    if alpha_t is not None:
        lw_2d = (alpha_t > 0).to(DTYPE)
    if edge_weight is not None:
        lw_2d = edge_weight if lw_2d is None else lw_2d * edge_weight
    if lw_2d is not None:
        lw4 = lw_2d.view(1, h, w, 1)
        denom = (lw_2d.sum() * 3.0).clamp(min=1.0)

    p = init_params.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([p], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        masks = kind.rasterize_soft(p, h, w)
        color = _optimal_color_batch(masks, cur_f, tgt_f, alpha_mask_f, a).detach()
        src = color.view(M, 1, 1, 3)
        if alpha_mask_f is not None:
            m_paint = (masks * alpha_mask_f.unsqueeze(0)).unsqueeze(-1)
        else:
            m_paint = masks.unsqueeze(-1)
        blended = m_paint * (a * src + (1.0 - a) * cur_f) + (1.0 - m_paint) * cur_f
        diff = blended - tgt_f
        if lw_2d is not None:
            per = (diff * diff * lw4).sum(dim=(1, 2, 3)) / denom
        else:
            per = (diff * diff).mean(dim=(1, 2, 3))
        if alpha_mask_f is not None:
            bleed = (masks * (1.0 - alpha_mask_f.unsqueeze(0))).sum(dim=(1, 2))
            per = per + STICKER_GRAD_PENALTY * bleed
        per.sum().backward()
        opt.step()
        with torch.no_grad():
            kind.clamp_(p, w, h)

    refined = p.detach()
    candidates = torch.cat([init_params, refined], dim=0)
    masks = kind.rasterize_hard(candidates, h, w)
    scores, colors, alphas = score_batch(masks, canvas_u8, target_u8, alpha_mask=alpha_t,
                                         alpha_levels=alpha_levels, edge_weight=edge_weight)
    best = int(scores.argmin().cpu().item())
    return (candidates[best], colors[best], int(alphas[best].cpu().item()),
            float(scores[best].cpu().item()))


def run_gpu(
    target_rgb: np.ndarray,
    cfg: GPUConfig,
    alpha_mask: np.ndarray | None = None,
    progress_every: int = 0,
    checkpoint_cb=None,
    checkpoint_every: int = 0,
) -> tuple[list[dict], np.ndarray]:
    """Run the GPU shape-gen loop. Returns (shapes_as_json_dicts, final_canvas_u8_numpy).

    If `checkpoint_cb` and `checkpoint_every > 0`, the greedy phase calls
    `checkpoint_cb(shape_idx, list_of_shapes_so_far)` every `checkpoint_every` committed
    shapes — so a caller can persist recoverable progress (e.g. to Google Drive) and survive
    a session reset mid-run. The callback gets a cheap copy of the shapes list (no canvas).

    If `alpha_mask` is provided (H×W uint8, 0=transparent, 255=opaque), sticker mode is
    active: target RGB is zeroed in transparent areas, canvas inits to grey (40), the
    scorer rejects candidates that bleed past the silhouette, and paint is clipped to
    the opaque region. Caller must match the alpha_mask's H/W to target_rgb's H/W.

    If `progress_every > 0`, prints a progress line every that many committed shapes with
    rate, ETA, and current best RMS — useful for long Colab runs.
    """
    if target_rgb.ndim != 3 or target_rgb.shape[2] != 3 or target_rgb.dtype != np.uint8:
        raise ValueError("target_rgb must be (H, W, 3) uint8")
    if alpha_mask is not None:
        if alpha_mask.ndim != 2 or alpha_mask.dtype != np.uint8:
            raise ValueError("alpha_mask must be (H, W) uint8 or None")
        if alpha_mask.shape != target_rgb.shape[:2]:
            raise ValueError(
                f"alpha_mask shape {alpha_mask.shape} != target {target_rgb.shape[:2]}"
            )
        # Alpha threshold: binarize the mask at this level (0 = off, keep soft alpha). Stickers
        # exported over a light background have an anti-aliased edge whose semi-transparent
        # pixels carry that light RGB; counting them as opaque paints a bright HALO around the
        # silhouette on the grey substrate. Thresholding drops the soft fringe → clean hard
        # silhouette (which is what FH6 renders anyway). Raise toward 160-200 if a halo persists.
        if cfg.alpha_threshold > 0:
            alpha_mask = np.where(alpha_mask >= cfg.alpha_threshold, 255, 0).astype(np.uint8)

    if cfg.lock_alpha:
        # Lock every shape to alpha=255: the FH injector forces binary alpha at inject time,
        # so optimizing soft alpha produces JSONs whose engine PNG diverges from the game render.
        cfg = replace(cfg, alpha_levels=[255])

    device = get_device()
    h, w = target_rgb.shape[:2]

    # Posterize the target (color quantization) before fitting — cleaner flat bands, crisper
    # edges. Done on the raw target so both the scored target and the canvas mean see it.
    if cfg.posterize_levels:
        target_rgb = _posterize(target_rgb, cfg.posterize_levels)

    # Resolve candidate shape kinds. Gradient mode tries all; hillclimb is ellipse-only
    # (no per-kind mutation is defined for triangle/rectangle).
    kinds = [KINDS[t] for t in cfg.shape_types]
    if cfg.refine_mode == "hillclimb":
        kinds = [KINDS["rotated_ellipse"]]

    # bbox-local random scoring applies only to the single-ellipse gradient path.
    use_bbox = (cfg.bbox_local and cfg.refine_mode == "gradient"
                and len(kinds) == 1 and kinds[0].name == "rotated_ellipse")

    if alpha_mask is not None:
        # Fill out-of-silhouette target pixels with the canvas substrate color (grey 40).
        # Rationale: in-game and CPU renderers paint every shape's full ellipse unclipped —
        # they have no source alpha mask to clip against. To make the optimizer's view of the
        # world match the game's, we DON'T clip shapes in the renderer. But then we need a
        # different mechanism to discourage paint from spilling past the silhouette. The trick:
        # if target == canvas_init outside the silhouette, then "no paint there" produces zero
        # loss, and "paint that spills there" produces positive loss — naturally penalizing
        # spill via plain MSE. Previously target was zeroed (RGB=0) outside the silhouette and
        # the loss was lw-masked to ignore those pixels; the optimizer never felt spill and
        # joint_polish would silently let shapes drift across the boundary, which the engine
        # then hid by clipping at render time. That clipped preview diverged from the unclipped
        # game render. Aligning target with substrate fixes the loss signal so JSON quality
        # matches what the game will actually display.
        opaque_mask3 = (alpha_mask > 0)[:, :, None].astype(np.uint8)
        substrate = np.full_like(target_rgb, 40)
        target_clean = np.where(opaque_mask3 > 0, target_rgb, substrate)
    else:
        target_clean = target_rgb

    target = torch.from_numpy(target_clean).to(device)
    alpha_t: torch.Tensor | None = None
    alpha_mask_f: torch.Tensor | None = None
    if alpha_mask is not None:
        alpha_t = torch.from_numpy(alpha_mask).to(device)
        alpha_mask_f = alpha_t.to(DTYPE) / 255.0

    # Edge-weight map (constant across the run): emphasizes eyes/lineart/boundaries.
    edge_weight = _edge_weight_map(target, cfg.edge_strength) if cfg.edge_strength > 0 else None

    if alpha_mask is not None:
        canvas = torch.full((h, w, 3), 40, dtype=torch.uint8, device=device)
    else:
        mean = target.to(DTYPE).reshape(-1, 3).mean(dim=0).round().clamp(0, 255).to(torch.uint8)
        canvas = mean.view(1, 1, 3).expand(h, w, 3).contiguous().clone()

    seed = cfg.seed or (int(time.time() * 1000) & 0x7FFFFFFF)
    gen = torch.Generator(device="cpu").manual_seed(seed)

    shapes: list[dict] = []
    consecutive_skips = 0
    shape_idx = 0
    t_start = time.perf_counter()
    per_kind = max(1, cfg.random_samples // len(kinds))
    while shape_idx < cfg.num_shapes:
        # 1) Random search across all candidate kinds; keep each kind's best seed.
        #    score_batch is type-agnostic (operates on the rasterized mask), so all kinds
        #    are scored the same way and compared directly.
        best_kind = None
        best_kind_params = None   # (per_kind, P) device tensor of the winning kind
        best_kind_scores = None   # per-candidate scores (RMS, or bbox delta) for top-M
        best_score = float("inf")
        best_color = None
        best_alpha = ALPHA_FIXED
        if use_bbox:
            # bbox-local: score each random ellipse over its own crop (cheap → high volume).
            # `scores` here are incremental SSE deltas (lower=better); argmin/top-M still
            # select the globally-best candidate. The committed RMS is recomputed full-canvas
            # by the gradient refiner below.
            ekind = kinds[0]
            params = ekind.init(cfg.random_samples, w, h, gen).to(device)
            scores, colors, alphas = crop_score_ellipse_batch(
                params, canvas, target, alpha_t=alpha_t, edge_weight=edge_weight,
                alpha_levels=cfg.alpha_levels, max_crop_radius=cfg.bbox_crop_max)
            idx_t = scores.argmin()
            best_score = float(scores[idx_t].cpu().item())
            if best_score != float("inf"):
                best_kind = ekind
                best_kind_params = params
                best_kind_scores = scores
                bi = int(idx_t.cpu().item())
                best_color = colors[bi].clone()
                best_alpha = int(alphas[bi].cpu().item())
                best_params = params[bi].clone()
        else:
            for kind in kinds:
                params = kind.init(per_kind, w, h, gen).to(device)
                masks = kind.rasterize_hard(params, h, w)
                scores, colors, alphas = score_batch(masks, canvas, target, alpha_mask=alpha_t,
                                                     alpha_levels=cfg.alpha_levels,
                                                     edge_weight=edge_weight)
                idx_t = scores.argmin()
                val = float(scores[idx_t].cpu().item())
                if val < best_score:
                    best_score = val
                    best_kind = kind
                    best_kind_params = params
                    best_kind_scores = scores
                    bi = int(idx_t.cpu().item())
                    best_color = colors[bi].clone()
                    best_alpha = int(alphas[bi].cpu().item())
                    best_params = params[bi].clone()
                del masks
                if kind is not best_kind:
                    del params, scores, colors, alphas

        if best_score == float("inf"):
            consecutive_skips += 1
            if consecutive_skips >= _MAX_CONSECUTIVE_SKIPS:
                break
            continue

        if cfg.refine_mode == "gradient":
            # Multi-start within the winning kind: top-M of its candidates.
            M = min(cfg.grad_starts, best_kind_params.shape[0])
            top_idx = torch.topk(best_kind_scores, M, largest=False).indices
            init = best_kind_params[top_idx].clone()
            best_params, best_color, best_alpha, best_score = _refine_gradient(
                best_kind, init, canvas, target, alpha_t, alpha_mask_f, h, w,
                cfg.grad_steps, cfg.grad_lr, alpha_levels=cfg.alpha_levels,
                edge_weight=edge_weight,
            )
            if best_score == float("inf"):
                consecutive_skips += 1
                if consecutive_skips >= _MAX_CONSECUTIVE_SKIPS:
                    break
                continue
        else:
            # Hill-climb (rotated_ellipse only — kinds was forced to ellipse above).
            for _r in range(cfg.mutation_rounds):
                mut_cpu = _mutate(best_params.cpu(), cfg.mutations_per_round, w, h, gen)
                mut = mut_cpu.to(device)
                mut_masks = rasterize_rotated_ellipses(mut, h, w)
                mut_scores, mut_colors, mut_alphas = score_batch(
                    mut_masks, canvas, target, alpha_mask=alpha_t, alpha_levels=cfg.alpha_levels,
                    edge_weight=edge_weight)
                local_best_t = mut_scores.argmin()
                local_best_score = float(mut_scores[local_best_t].cpu().item())
                if local_best_score < best_score:
                    best_score = local_best_score
                    local_best = int(local_best_t.cpu().item())
                    best_params = mut[local_best].clone()
                    best_color = mut_colors[local_best].clone()
                    best_alpha = int(mut_alphas[local_best].cpu().item())
                del mut_masks, mut_scores, mut_colors, mut_alphas, mut, mut_cpu

        # Commit
        canvas = _composite_one(best_kind, canvas, best_params, best_color, h, w,
                                alpha_mask_f, best_alpha)
        c = best_color.cpu().tolist()
        color = [int(c[0]), int(c[1]), int(c[2]), int(best_alpha)]
        shapes.append(best_kind.to_json(best_params.cpu().tolist(), color))
        consecutive_skips = 0
        shape_idx += 1

        if checkpoint_cb and checkpoint_every and shape_idx % checkpoint_every == 0:
            checkpoint_cb(shape_idx, list(shapes))

        if progress_every and shape_idx % progress_every == 0:
            elapsed = time.perf_counter() - t_start
            rate = shape_idx / elapsed if elapsed > 0 else 0.0
            eta = (cfg.num_shapes - shape_idx) / rate if rate > 0 else 0.0
            print(f"  {shape_idx}/{cfg.num_shapes} shapes  "
                  f"({rate:.1f}/s, ETA {eta:.0f}s, RMS {best_score:.2f})")

    # Joint-polish: globally refine all committed shapes (ellipse-only) against the target.
    if cfg.joint_polish_steps > 0 and shapes and all(s["type"] == "rotated_ellipse" for s in shapes):
        if progress_every:
            print(f"  joint-polishing {len(shapes)} shapes for {cfg.joint_polish_steps} steps...")
        shapes, canvas_np = joint_polish(
            shapes, target, alpha_t, alpha_mask_f, edge_weight, canvas,
            h, w, cfg.joint_polish_steps, lr=cfg.joint_polish_lr,
            progress=bool(progress_every), lock_alpha=cfg.lock_alpha,
            purity_penalty=cfg.polish_purity_penalty,
            freeze_geometry=cfg.polish_freeze_geometry,
        )
        return shapes, canvas_np

    return shapes, canvas.cpu().numpy()
