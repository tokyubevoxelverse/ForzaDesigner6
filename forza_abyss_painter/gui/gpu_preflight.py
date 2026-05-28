"""GPU-spawn preflight gate. Chunk-aware peak estimator + free-VRAM probe.

Called by every path that spawns a GPU shape-gen run:
  - main_window._start_gpu
  - GenerateLocallyDialog._on_generate_clicked
  - main_window._on_polish_requested
  - main_window._on_resume_requested

Flow:
  1. Probe free VRAM (clamped to configured budget -- user's intent).
  2. Ask estimate_effective_peak_gib for the actual runtime peak
     accounting for chunked-K.
  3. Block if peak > free. Warn if peak > 0.85 * free. Otherwise proceed.

NO back-prop. NO auto-lower modal. NO silent raise. The engine handles
fit via chunked-K; this helper only catches catastrophic mismatch
(e.g. budget set high but probed free is far lower than expected).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PySide6.QtWidgets import QMessageBox, QWidget

from forza_abyss_painter.shapegen.gpu.vram_planner import (
    estimate_effective_peak_gib,
)

WARN_FRAC = 0.85


@dataclass
class PreflightOutcome:
    verdict: str       # "ok" | "warn" | "block"
    proceed: bool
    summary: str


def _decide(*, peak_gib: float, free_gib: float, budget_gib: float) -> PreflightOutcome:
    if peak_gib > free_gib:
        return PreflightOutcome(
            verdict="block",
            proceed=False,
            summary=f"Won't fit: peak {peak_gib:.1f} GiB > free {free_gib:.1f} GiB",
        )
    if peak_gib > WARN_FRAC * free_gib:
        return PreflightOutcome(
            verdict="warn",
            proceed=True,
            summary=f"Tight: peak {peak_gib:.1f} GiB / free {free_gib:.1f} GiB",
        )
    return PreflightOutcome(
        verdict="ok",
        proceed=True,
        summary=f"Fits: peak {peak_gib:.1f} GiB / free {free_gib:.1f} GiB",
    )


def _probe_free_gib(budget_gib: float) -> float:
    """nvidia-smi free-VRAM probe. Returns budget_gib if probe unavailable.

    Wraps `runtime.nvidia_smi.probe_free_vram(force=True)` and unwraps the
    `ProbeResult` to a float. The cache is bypassed (force=True) because
    a stale value from before the user closed FH6 would defeat the entire
    point of the preflight.
    """
    from forza_abyss_painter.runtime.nvidia_smi import probe_free_vram
    probe = probe_free_vram(force=True)
    if probe.available and probe.free_gib is not None:
        return float(probe.free_gib)
    return float(budget_gib)


def _show_block_modal(
    parent: QWidget | None, *, context: str, summary: str, free_gib: float,
) -> None:
    QMessageBox.critical(
        parent,
        f"{context} - won't fit",
        f"{summary}\n\nClose FH6 or free more VRAM.\n"
        f"Free VRAM: {free_gib:.1f} GiB",
    )


def _show_warn_modal(
    parent: QWidget | None, *, context: str, summary: str,
) -> bool:
    """Return True to proceed anyway, False to cancel."""
    ret = QMessageBox.warning(
        parent,
        f"{context} - tight on VRAM",
        f"{summary}\n\nProceed anyway?",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.No,
    )
    return ret == QMessageBox.StandardButton.Yes


def gpu_run_preflight(
    *,
    parent: QWidget | None,
    preset: dict[str, Any],
    budget_gib: float,
    context: str,
) -> tuple[bool, dict[str, Any]]:
    """Return (proceed_ok, info) where info has 'peak_gib', 'chunks_per_shape',
    'free_gib' keys. Caller uses info for status-bar messaging.

    Caller must NOT spawn worker if proceed_ok is False. The preset is
    NOT modified -- chunking is handled inside the engine, not here.

    Args:
      parent: QWidget for modal parenting (None ok in headless tests).
      preset: must contain 'random_samples' and 'max_resolution'.
      budget_gib: card's total VRAM budget (typically settings.gpu_budget_gib).
      context: human label for modals ("Generate locally", "Generate from
        drop", "Polish loaded JSON", "Resume from snapshot").
    """
    K = int(preset["random_samples"])
    max_resolution = int(preset["max_resolution"])

    free_gib = _probe_free_gib(budget_gib)
    peak_gib, chunks_per_shape = estimate_effective_peak_gib(
        K=K, max_resolution=max_resolution, budget_gib=budget_gib,
    )

    info = {
        "peak_gib": peak_gib,
        "chunks_per_shape": chunks_per_shape,
        "free_gib": free_gib,
    }

    outcome = _decide(peak_gib=peak_gib, free_gib=free_gib, budget_gib=budget_gib)

    if outcome.verdict == "block":
        _show_block_modal(parent, context=context, summary=outcome.summary, free_gib=free_gib)
        return False, info
    if outcome.verdict == "warn":
        ok = _show_warn_modal(parent, context=context, summary=outcome.summary)
        return ok, info
    return True, info
