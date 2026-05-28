"""K-only formula is the calibrated truth. Pin predictions against
QUASAR-measured peaks so future engine refactors that move the peak
trigger this test and force a recalibration of BBOX_LOCAL_SAFETY.

Evidence anchors (see docs/superpowers/specs/2026-05-27-vram-honesty-correction-design.md §1):
- Run 4 (K=8192, max_res=720): formula predicts 48.3 GiB, measured 47.5 GiB.
- ziz_dance (K=1000, max_res=1200): formula predicts 16.3 GiB, fit in <=17 GiB free.
- Hi-Res preset (K=12288, max_res=1000): formula predicts 139 GiB, OOMed on 32 GiB card.
"""
from forza_abyss_painter.shapegen.gpu.vram_planner import (
    BBOX_LOCAL_SAFETY,
    estimate_peak_vram_gib,
)


def test_safety_multiplier_pinned():
    """The K-only formula's safety multiplier is the calibration knob."""
    assert BBOX_LOCAL_SAFETY == 15.0


def test_run4_calibration():
    """QUASAR Run 4: K=8192, max_res=720 -> 48.3 GiB predicted, 47.5 GiB measured."""
    peak = estimate_peak_vram_gib(K=8192, bbox_local=True, max_resolution=720)
    assert 47.0 < peak < 50.0


def test_ziz_dance_calibration():
    """ziz_dance: K=1000, max_res=1200 -> ~16 GiB predicted, fit in <=17 free."""
    peak = estimate_peak_vram_gib(K=1000, bbox_local=True, max_resolution=1200)
    assert 14.0 < peak < 19.0


def test_hi_res_preset_calibration():
    """Hi-Res 3000: K=12288, max_res=1000 -> ~130 GiB predicted (blocks 32G)."""
    peak = estimate_peak_vram_gib(K=12288, bbox_local=True, max_resolution=1000)
    assert 120.0 < peak < 145.0


def test_pipeline_overhead_symbol_removed():
    """PR A's PIPELINE_OVERHEAD_GIB and estimate_full_pipeline_gib must be gone."""
    import forza_abyss_painter.shapegen.gpu.vram_planner as vp
    assert not hasattr(vp, "PIPELINE_OVERHEAD_GIB")
    assert not hasattr(vp, "estimate_full_pipeline_gib")
