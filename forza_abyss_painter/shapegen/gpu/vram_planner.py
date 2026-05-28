"""VRAM planning math — pure Python, no torch dependency.

Lives outside the torch-importing engine module so tests + the EXE
settings panel can use the same formulas without pulling CUDA. The
GPU engine imports `_resolve_k_chunk_size` from here at scoring time.

## What this module computes

Two questions, same formula reversed:

  estimate_peak_vram_gib(K, footprint, safety) -> GiB
      "If I run K candidates at this footprint, how much VRAM peaks?"
      Used by the SettingsPanel to show the user 'this run wants X GiB'.

  resolve_k_chunk_size(K, budget_gib, footprint, safety) -> int
      "Given my budget, how many candidates fit per chunk?"
      Used by the engine to split a K-batch into VRAM-safe chunks.
      Returns 0 if the full K fits (no chunking needed) or budget==0.

Both share the same memory model — they MUST stay in sync with what
the scorers actually materialize (the (K, footprint, 3) intermediate
in score_batch / crop_score_ellipse_batch). If a scorer's peak shape
changes, update this module's `_peak_bytes_per_candidate` too.
"""
from __future__ import annotations

import math

# Calibrated safety multipliers for each scoring path. RE-CALIBRATED
# 2026-05-25 against Cursor's QUASAR smoke evidence:
#   K=8192 + max_resolution=720 + bbox_local=True
#   predicted with old 5.5 mult: ~17.7 GiB
#   actually allocated by PyTorch: 47.5 GiB (cu128 + torch 2.7)
#   → real multiplier ≈ 5.5 × (47.5/17.7) ≈ 14.7
#
# Bumping to 15.0 with a small safety margin for fragmentation
# variability. The PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# env var (set in torch_runner) reduces but doesn't eliminate the
# fragmentation overhead — calibration still has to account for it.
#
# If a future torch / driver bump changes the allocator behavior,
# Cursor's smoke (run_gpu_smoke.ps1 stress tier) is the canary —
# any new under-prediction shows up immediately as OOM on a budgeted
# run.
BBOX_LOCAL_SAFETY = 15.0    # was 5.5; under-predicted cu128 + torch 2.7
FULL_CANVAS_SAFETY = 8.0    # was 3.5; same calibration ratio applied

# Chunk-size floor — Python overhead dominates below this; budgets
# too tight to fit 8 candidates are user errors (lower the resolution).
MIN_CHUNK_SIZE = 8

# 3 RGB channels × 4 bytes/float32. The full per-pixel cost is
# _BYTES_PER_PIXEL_RAW * BBOX_LOCAL_SAFETY (or _FULL_CANVAS_SAFETY).
# Update this if the scorers ever switch to float16, add alpha, etc.
_BYTES_PER_PIXEL_RAW = 12.0


def _peak_bytes_per_candidate(
    bbox_local: bool,
    max_resolution: int,
    bbox_crop_max: int = 256,
) -> float:
    """Predicted peak bytes per candidate in the K-dim, including the
    calibrated safety margin. Multiply by K to get total peak bytes.

    bbox-local: footprint = min((2*crop_e+1)², res²) where crop_e =
                min(bbox_crop_max, res/8) — bounded crop around each
                ellipse, much smaller than the full canvas.
    full-canvas: footprint = res² — every candidate's mask covers the
                 whole canvas.
    """
    res = max(1, int(max_resolution))
    if bbox_local:
        crop_e = min(bbox_crop_max, max(1, res // 8))
        footprint = min((2 * crop_e + 1) ** 2, res * res)
        safety = BBOX_LOCAL_SAFETY
    else:
        footprint = res * res
        safety = FULL_CANVAS_SAFETY
    # 3 channels × 4 bytes/float32 = 12 bytes/pixel × safety
    return footprint * _BYTES_PER_PIXEL_RAW * safety


def estimate_peak_vram_gib(
    K: int,
    bbox_local: bool,
    max_resolution: int,
    bbox_crop_max: int = 256,
) -> float:
    """Predicted peak VRAM in GiB for the given K + canvas. The
    SettingsPanel uses this to show the user 'this run wants ~X GiB'
    before they hit Start."""
    if K <= 0:
        return 0.0
    bytes_per_cand = _peak_bytes_per_candidate(
        bbox_local, max_resolution, bbox_crop_max,
    )
    return (K * bytes_per_cand) / 1e9


def resolve_k_chunk_size(
    K: int,
    bbox_local: bool,
    max_resolution: int,
    vram_budget_gib: float,
    k_chunk_override: int = 0,
    bbox_crop_max: int = 256,
) -> int:
    """Given a K-batch + VRAM budget, return the chunk size that fits.

    Returns 0 when chunking is unnecessary (budget allows the full K
    in one pass, or budget is 0 = unlimited). Returns >0 chunk size
    when K must be split.

    `k_chunk_override` is a power-user knob: if > 0, that value wins
    regardless of budget. Use it to pin chunk size for reproducibility
    or to work around an under/over-estimate in the formula for an
    unusual canvas geometry.
    """
    if k_chunk_override > 0:
        return k_chunk_override
    if vram_budget_gib <= 0:
        return 0
    bytes_per_cand = _peak_bytes_per_candidate(
        bbox_local, max_resolution, bbox_crop_max,
    )
    if bytes_per_cand <= 0:
        return 0
    k_max = int((vram_budget_gib * 1e9) / bytes_per_cand)
    k_max = max(MIN_CHUNK_SIZE, k_max)
    if k_max >= K:
        return 0   # full K fits in one pass
    return k_max


def recommend_max_resolution(
    free_gib: float,
    K: int,
    *,
    bbox_local: bool = True,
    bbox_crop_max: int = 256,
    safety_floor_px: int = 720,
    safety_margin: float = 0.85,
    absolute_ceiling_px: int = 4096,
) -> int:
    """Return the largest max_resolution that fits in
    `safety_margin * free_gib` of VRAM at the given K, in the bbox_local
    (or full_canvas) scoring regime.

    Always returns at least `safety_floor_px` (default 720). Probe-
    unavailable callers (free_gib <= 0) get the floor. When the math
    indicates a value above `absolute_ceiling_px` (i.e., Regime B is
    fully satisfied), the ceiling is returned — callers separately
    clamp by the source image's long side.

    The math is documented in
    `docs/superpowers/specs/2026-05-26-131-vram-backprop-planner-design.md`
    §4. In short, the K-batch footprint formula has two regimes:

      Regime A — crop_e = max_resolution // 8 (small canvas):
          footprint = (2 * crop_e + 1)²       quadratic in max_res
      Regime B — crop_e = bbox_crop_max       (big canvas, capped):
          footprint = (2 * bbox_crop_max + 1)²   constant

    Regime B is the "we're past the bbox cap; canvas size no longer
    affects K-peak" regime — if it fits, ceiling out.
    """
    if free_gib <= 0 or K < 1:
        return safety_floor_px

    target_bytes = free_gib * safety_margin * 1e9
    if bbox_local:
        safety = BBOX_LOCAL_SAFETY
    else:
        safety = FULL_CANVAS_SAFETY
    bytes_per_pixel = _BYTES_PER_PIXEL_RAW * safety   # 3 channels × 4 bytes × safety

    # Regime B check: at crop_e = bbox_crop_max, footprint is constant.
    # If THAT fits, then any max_res >= bbox_crop_max * 8 is fine.
    if bbox_local:
        regime_b_footprint = (2 * bbox_crop_max + 1) ** 2
        regime_b_total = K * regime_b_footprint * bytes_per_pixel
        if regime_b_total <= target_bytes:
            return int(absolute_ceiling_px)

    # Regime A: footprint = (2 * (max_res // 8) + 1)². Solve for max_res.
    # Solve: K * (2 * (m // 8) + 1)² * bytes_per_pixel <= target_bytes
    # i.e.: (2 * (m // 8) + 1) <= sqrt(target_bytes / (K * bytes_per_pixel))
    # i.e.: m <= 8 * ((sqrt(...) - 1) / 2)
    if K * bytes_per_pixel <= 0:
        return safety_floor_px
    inner = target_bytes / (K * bytes_per_pixel)
    if inner <= 1:
        return safety_floor_px
    sqrt_inner = math.sqrt(inner)
    max_res = int(8 * (sqrt_inner - 1) / 2)

    # full_canvas path uses the canvas² footprint (no bbox cap), so the
    # solve is just: max_res² <= target_bytes / (K * bytes_per_pixel) = inner.
    if not bbox_local:
        max_res = int(sqrt_inner)

    # Floor + ceiling clamps.
    max_res = max(safety_floor_px, max_res)
    max_res = min(absolute_ceiling_px, max_res)
    return max_res
