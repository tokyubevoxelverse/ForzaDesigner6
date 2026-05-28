"""Single source of truth for the GPU-spawn preflight gate.

Called by every path that spawns a GPU shape-gen run:
  - main_window._start_gpu (auto-queue from drop)
  - GenerateLocallyDialog._on_generate_clicked
  - main_window._on_polish_requested
  - main_window._on_resume_requested

Three-tier verdict + back-prop with optional auto-lower:
  1. Probe free VRAM via nvidia-smi (cached per call).
  2. Compute recommended max_res via recommend_max_resolution.
  3. If recommended < baked AND recommended >= safety_floor: show
     "Lower to Y / Cancel" modal; on Lower, swap max_res; on Cancel,
     return (False, preset).
  4. If recommended < safety_floor: show block modal, return (False, preset).
  5. Compute peak_gib via estimate_peak_vram_gib with effective max_res.
  6. Verdict: peak > free -> block; peak > 0.85*free -> warn; else ok.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PySide6.QtWidgets import QMessageBox, QWidget

from forza_abyss_painter.shapegen.gpu.vram_planner import (
    estimate_peak_vram_gib,
    recommend_max_resolution,
)

SAFETY_FLOOR_PX = 256
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


def _recommend(K: int, free_gib: float) -> int:
    return recommend_max_resolution(K=K, free_gib=free_gib, bbox_local=True)


def _show_lower_modal(
    parent: QWidget | None, *, context: str, baked: int, lowered: int,
    free_gib: float, peak_lowered: float,
) -> bool:
    """Return True if user accepted the lower. False = cancel."""
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Warning)
    box.setWindowTitle(f"{context} - VRAM tight")
    box.setText(
        f"Your preset wants max_resolution={baked} px, which needs more "
        f"VRAM than you have free ({free_gib:.1f} GiB)."
    )
    box.setInformativeText(
        f"Lower to {lowered} px (peak ~{peak_lowered:.1f} GiB)?\n\n"
        f"Or cancel and close FH6 / free more VRAM."
    )
    lower_btn = box.addButton(f"Lower to {lowered} px", QMessageBox.ButtonRole.AcceptRole)
    box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
    box.exec()
    return box.clickedButton() is lower_btn


def _show_block_modal(
    parent: QWidget | None, *, context: str, summary: str, free_gib: float,
) -> None:
    QMessageBox.critical(
        parent,
        f"{context} - won't fit",
        f"{summary}\n\nClose FH6 or pick a smaller preset.\n"
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
    """See module docstring for flow.

    Args:
      parent: QWidget for modal parenting (None ok in headless tests).
      preset: must contain 'random_samples' and 'max_resolution'.
      budget_gib: card's total VRAM budget (typically settings.gpu_budget_gib).
      context: human label for modals ("Generate locally", "Re-shape-gen",
        "Polish loaded JSON", "Resume from snapshot").

    Returns: (proceed_ok, effective_preset). Caller must NOT spawn worker
    if proceed_ok is False. effective_preset may have a lowered max_resolution.
    """
    K = int(preset["random_samples"])
    baked = int(preset["max_resolution"])
    effective = dict(preset)

    free_gib = _probe_free_gib(budget_gib)
    recommended = _recommend(K=K, free_gib=free_gib)

    if recommended < baked:
        if recommended < SAFETY_FLOOR_PX:
            summary = (
                f"Free VRAM {free_gib:.1f} GiB can only fit ~{recommended} px, "
                f"below the {SAFETY_FLOOR_PX} px floor."
            )
            _show_block_modal(parent, context=context, summary=summary, free_gib=free_gib)
            return False, effective

        peak_lowered = estimate_peak_vram_gib(
            K=K, bbox_local=True, max_resolution=recommended,
        )
        ok = _show_lower_modal(
            parent, context=context, baked=baked, lowered=recommended,
            free_gib=free_gib, peak_lowered=peak_lowered,
        )
        if not ok:
            return False, effective
        effective["max_resolution"] = recommended
        # Trust the lower-modal acceptance: recommend_max_resolution
        # already picked a value that fits in free VRAM by its own
        # contract. No second verdict needed.
        return True, effective

    elif recommended > baked:
        # Raise silently on big-VRAM cards; UI shows the autotune message via status bar.
        effective["max_resolution"] = recommended

    peak_gib = estimate_peak_vram_gib(
        K=K, bbox_local=True, max_resolution=int(effective["max_resolution"]),
    )
    outcome = _decide(peak_gib=peak_gib, free_gib=free_gib, budget_gib=budget_gib)

    if outcome.verdict == "block":
        _show_block_modal(parent, context=context, summary=outcome.summary, free_gib=free_gib)
        return False, effective
    if outcome.verdict == "warn":
        ok = _show_warn_modal(parent, context=context, summary=outcome.summary)
        return ok, effective
    return True, effective
