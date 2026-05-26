"""Pre-Start VRAM verdict helper — pure logic, no Qt."""
from __future__ import annotations

from forza_abyss_painter.gui.main_window import _vram_preflight_verdict


def test_block_when_peak_exceeds_free():
    severity, msg = _vram_preflight_verdict(peak_gib=48.0, free_gib=27.0, budget_gib=8.0)
    assert severity == "block"
    assert "48.0" in msg and "27.0" in msg


def test_warn_when_peak_in_top_15_percent_of_free():
    severity, _ = _vram_preflight_verdict(peak_gib=24.0, free_gib=27.0, budget_gib=8.0)
    assert severity == "warn"


def test_warn_when_free_less_than_90_percent_of_budget():
    # peak fits comfortably, but free is well below the user's budget —
    # signals contention from other apps.
    severity, _ = _vram_preflight_verdict(peak_gib=4.0, free_gib=10.0, budget_gib=16.0)
    assert severity == "warn"


def test_ok_when_peak_comfortably_fits():
    severity, _ = _vram_preflight_verdict(peak_gib=4.0, free_gib=27.0, budget_gib=8.0)
    assert severity == "ok"


def test_ok_when_probe_unavailable():
    severity, _ = _vram_preflight_verdict(peak_gib=48.0, free_gib=None, budget_gib=8.0)
    assert severity == "ok"


def test_ok_when_free_is_zero():
    severity, _ = _vram_preflight_verdict(peak_gib=48.0, free_gib=0.0, budget_gib=8.0)
    assert severity == "ok"


def test_block_takes_precedence_over_warn():
    severity, _ = _vram_preflight_verdict(peak_gib=50.0, free_gib=27.0, budget_gib=8.0)
    assert severity == "block"


def test_run_4_real_scenario_blocks():
    # Run 4 evidence: K=8192, max_res=720, bbox_local=True →
    # estimate ~48 GiB. RTX 5090 free at ~22 GiB after FH6.
    severity, msg = _vram_preflight_verdict(peak_gib=48.0, free_gib=22.0, budget_gib=8.0)
    assert severity == "block", (
        "Run 4's exact OOM scenario must be hard-blocked, not warn-only"
    )
