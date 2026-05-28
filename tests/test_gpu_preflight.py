"""gpu_run_preflight is chunk-aware. No back-prop, no lower modal.
Probes free VRAM, asks vram_planner.estimate_effective_peak_gib for
the actual runtime peak, blocks if it exceeds free, warns if tight,
proceeds otherwise.
"""
from unittest.mock import MagicMock
import pytest

from forza_abyss_painter.gui.gpu_preflight import PreflightOutcome, _decide


def test_ok_when_peak_under_free():
    outcome = _decide(peak_gib=12.0, free_gib=24.0, budget_gib=24.0)
    assert outcome.verdict == "ok"
    assert outcome.proceed is True


def test_warn_when_peak_above_85pct_of_free():
    outcome = _decide(peak_gib=22.0, free_gib=24.0, budget_gib=24.0)
    assert outcome.verdict == "warn"


def test_block_when_peak_exceeds_free():
    outcome = _decide(peak_gib=130.0, free_gib=24.0, budget_gib=24.0)
    assert outcome.verdict == "block"
    assert outcome.proceed is False


def test_preflight_uses_chunked_estimator(monkeypatch):
    """gpu_run_preflight calls estimate_effective_peak_gib, not the
    naive full-K estimator."""
    from forza_abyss_painter.gui import gpu_preflight as pre

    monkeypatch.setattr(pre, "_probe_free_gib", lambda budget_gib: 22.0)

    called_with = {}
    def fake_effective(*, K, max_resolution, budget_gib):
        called_with["K"] = K
        called_with["max_resolution"] = max_resolution
        called_with["budget_gib"] = budget_gib
        return (12.0, 12)
    monkeypatch.setattr(pre, "estimate_effective_peak_gib", fake_effective)

    monkeypatch.setattr(pre, "_show_warn_modal", MagicMock(return_value=True))
    monkeypatch.setattr(pre, "_show_block_modal", MagicMock())

    proceed, info = pre.gpu_run_preflight(
        parent=None,
        preset={"random_samples": 12288, "max_resolution": 1000},
        budget_gib=17.0,
        context="Generate locally",
    )

    assert proceed is True
    assert called_with["K"] == 12288
    assert called_with["max_resolution"] == 1000
    assert called_with["budget_gib"] == 17.0
    assert info["chunks_per_shape"] == 12
    assert info["peak_gib"] == 12.0


def test_block_modal_fires_when_peak_exceeds_free(monkeypatch):
    from forza_abyss_painter.gui import gpu_preflight as pre

    monkeypatch.setattr(pre, "_probe_free_gib", lambda budget_gib: 1.0)
    monkeypatch.setattr(
        pre, "estimate_effective_peak_gib",
        lambda **kw: (10.0, 1),
    )

    block = MagicMock()
    monkeypatch.setattr(pre, "_show_block_modal", block)
    monkeypatch.setattr(pre, "_show_warn_modal", MagicMock())

    proceed, info = pre.gpu_run_preflight(
        parent=None,
        preset={"random_samples": 12288, "max_resolution": 1000},
        budget_gib=24.0,
        context="Generate locally",
    )

    assert proceed is False
    block.assert_called_once()


def test_no_lower_modal_anymore():
    """_show_lower_modal must not exist."""
    from forza_abyss_painter.gui import gpu_preflight as pre
    assert not hasattr(pre, "_show_lower_modal")
