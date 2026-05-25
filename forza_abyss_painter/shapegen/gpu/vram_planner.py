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


# Calibrated safety multipliers for each scoring path. Don't tune
# without re-measuring against the engine's actual peak — these are
# what the colab CELL_RESOLUTION_PLANNER converged on after
# bench-testing real runs.
BBOX_LOCAL_SAFETY = 5.5    # crop-local scoring; smaller footprint, more intermediates
FULL_CANVAS_SAFETY = 3.5   # full-canvas scoring; bigger footprint, fewer intermediates

# Chunk-size floor — Python overhead dominates below this; budgets
# too tight to fit 8 candidates are user errors (lower the resolution).
MIN_CHUNK_SIZE = 8


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
    return footprint * 12.0 * safety


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
