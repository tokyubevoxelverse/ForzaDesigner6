"""Preflight (_vram_preflight_verdict) must catch the Run-4-class
OOM scenario when given the full-pipeline estimate.

This test doesn't construct MainWindow — it directly exercises
estimate_full_pipeline_gib + _vram_preflight_verdict to pin the
math. The GUI wiring is verified in the next task's smoke."""
from __future__ import annotations

from forza_abyss_painter.shapegen.gpu.vram_planner import (
    estimate_full_pipeline_gib,
)
from forza_abyss_painter.gui.main_window import _vram_preflight_verdict


def test_run_4_scenario_blocks_at_32_gib_card():
    """Run 4: K=8192, max_res=720, ~22 GiB free on 32G card.
    Full-pipeline estimate must trigger 'block' verdict."""
    peak = estimate_full_pipeline_gib(K=8192, bbox_local=True, max_resolution=720)
    severity, _msg = _vram_preflight_verdict(
        peak_gib=peak, free_gib=22.0, budget_gib=8.0,
    )
    assert severity == "block", (
        f"Run-4 scenario (peak {peak:.1f} GiB vs 22 free) should block; "
        f"got {severity}"
    )


def test_hi_res_3000_scenario_blocks():
    """QUASAR 2026-05-27: K=12288, max_res=1000, FH6 not open
    (~27 GiB free on 32G). Should block — measured OOM at 53.7 GiB."""
    peak = estimate_full_pipeline_gib(K=12288, bbox_local=True, max_resolution=1000)
    severity, _msg = _vram_preflight_verdict(
        peak_gib=peak, free_gib=27.0, budget_gib=8.0,
    )
    assert severity == "block"


def test_small_run_on_big_card_passes():
    """K=1024, max_res=480 (Lineart 400 preset) on a 95 GiB workstation
    card: full estimate ~37 GiB, plenty of headroom. Should NOT block."""
    peak = estimate_full_pipeline_gib(K=1024, bbox_local=True, max_resolution=480)
    severity, _msg = _vram_preflight_verdict(
        peak_gib=peak, free_gib=95.0, budget_gib=32.0,
    )
    assert severity == "ok"


def test_medium_on_32g_with_fh6_closed_passes_or_warns():
    """Medium 1000 (K=8192, max_res=720) on RTX 5090 with FH6 closed
    (~27 GiB free): full estimate ~47 GiB. 27 < 47 → should block."""
    peak = estimate_full_pipeline_gib(K=8192, bbox_local=True, max_resolution=720)
    severity, _msg = _vram_preflight_verdict(
        peak_gib=peak, free_gib=27.0, budget_gib=8.0,
    )
    # Run 4 evidence: this scenario DID OOM. Block is the right call.
    assert severity == "block"


def test_settings_panel_estimate_returns_full_pipeline():
    """SettingsPanel.estimate_peak_vram_gib(profile) must return the
    full-pipeline number, not the K-only one (UI consumers don't have
    a reason to see K-only)."""
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from forza_abyss_painter.gui.settings_panel import SettingsPanel

    panel = SettingsPanel()
    # Build a fake profile object with the attributes the method
    # reads. SettingsPanel's method just calls vram_planner — we just
    # need a Profile-like duck.
    class _Profile:
        random_samples = 8192
        max_resolution = 720
    profile = _Profile()
    est = panel.estimate_peak_vram_gib(profile)
    # Same scenario as Run 4. Full-pipeline estimate ≥ 40 GiB.
    assert est >= 40.0
    panel.deleteLater()
