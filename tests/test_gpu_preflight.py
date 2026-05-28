"""gpu_run_preflight is the single source of truth for the GPU-spawn
gate. Three-tier verdict (ok / warn / block) + back-prop with optional
auto-lower modal. Logic tested without Qt by injecting fake probe +
modal callables.
"""
from unittest.mock import MagicMock

import pytest

from forza_abyss_painter.gui.gpu_preflight import (
    PreflightOutcome,
    _decide,
)


def test_ok_when_peak_well_under_free():
    """K=1000, max_res=1200 needs ~16 GiB; 30 GiB free -> ok."""
    outcome = _decide(peak_gib=16.3, free_gib=30.0, budget_gib=24.0)
    assert outcome.verdict == "ok"
    assert outcome.proceed is True


def test_warn_when_peak_above_85pct_of_free():
    """Inside the danger zone but not over -> warn (user must confirm)."""
    outcome = _decide(peak_gib=22.0, free_gib=24.0, budget_gib=24.0)
    assert outcome.verdict == "warn"


def test_block_when_peak_exceeds_free():
    """Hard block."""
    outcome = _decide(peak_gib=130.0, free_gib=24.0, budget_gib=24.0)
    assert outcome.verdict == "block"
    assert outcome.proceed is False


def test_backprop_lowers_when_recommended_below_baked(monkeypatch):
    """When recommended < baked, helper proposes lower max_res."""
    from forza_abyss_painter.gui import gpu_preflight as pre

    monkeypatch.setattr(pre, "_probe_free_gib", lambda budget_gib: 22.0)
    monkeypatch.setattr(pre, "_recommend", lambda K, free_gib: 480)

    lower_modal = MagicMock(return_value=True)
    monkeypatch.setattr(pre, "_show_lower_modal", lower_modal)
    monkeypatch.setattr(pre, "_show_block_modal", MagicMock())
    monkeypatch.setattr(pre, "_show_warn_modal", MagicMock(return_value=True))

    proceed, effective = pre.gpu_run_preflight(
        parent=None,
        preset={"random_samples": 12288, "max_resolution": 1000},
        budget_gib=24.0,
        context="Generate locally",
    )

    assert proceed is True
    assert effective["max_resolution"] == 480
    lower_modal.assert_called_once()


def test_lower_modal_cancel_aborts(monkeypatch):
    """User cancels the lower modal -> proceed=False, no spawn."""
    from forza_abyss_painter.gui import gpu_preflight as pre

    monkeypatch.setattr(pre, "_probe_free_gib", lambda budget_gib: 22.0)
    monkeypatch.setattr(pre, "_recommend", lambda K, free_gib: 480)
    monkeypatch.setattr(pre, "_show_lower_modal", MagicMock(return_value=False))
    monkeypatch.setattr(pre, "_show_block_modal", MagicMock())
    monkeypatch.setattr(pre, "_show_warn_modal", MagicMock())

    proceed, effective = pre.gpu_run_preflight(
        parent=None,
        preset={"random_samples": 12288, "max_resolution": 1000},
        budget_gib=24.0,
        context="Generate locally",
    )

    assert proceed is False
    assert effective["max_resolution"] == 1000


def test_block_modal_fires_when_no_safe_max_res(monkeypatch):
    """Recommended below safety_floor -> block modal, no lower offer."""
    from forza_abyss_painter.gui import gpu_preflight as pre

    monkeypatch.setattr(pre, "_probe_free_gib", lambda budget_gib: 4.0)
    monkeypatch.setattr(pre, "_recommend", lambda K, free_gib: 200)
    block_modal = MagicMock()
    monkeypatch.setattr(pre, "_show_block_modal", block_modal)
    monkeypatch.setattr(pre, "_show_lower_modal", MagicMock())
    monkeypatch.setattr(pre, "_show_warn_modal", MagicMock())

    proceed, effective = pre.gpu_run_preflight(
        parent=None,
        preset={"random_samples": 12288, "max_resolution": 1000},
        budget_gib=24.0,
        context="Generate locally",
    )

    assert proceed is False
    block_modal.assert_called_once()


def test_backprop_raises_silently_on_big_card(monkeypatch):
    """When recommended > baked, helper raises max_res without any modal."""
    from forza_abyss_painter.gui import gpu_preflight as pre

    monkeypatch.setattr(pre, "_probe_free_gib", lambda budget_gib: 80.0)
    monkeypatch.setattr(pre, "_recommend", lambda K, free_gib: 1200)

    lower_modal = MagicMock()
    warn_modal = MagicMock()
    block_modal = MagicMock()
    monkeypatch.setattr(pre, "_show_lower_modal", lower_modal)
    monkeypatch.setattr(pre, "_show_warn_modal", warn_modal)
    monkeypatch.setattr(pre, "_show_block_modal", block_modal)

    proceed, effective = pre.gpu_run_preflight(
        parent=None,
        preset={"random_samples": 1000, "max_resolution": 700},
        budget_gib=80.0,
        context="Generate from drop",
    )

    assert proceed is True
    assert effective["max_resolution"] == 1200
    lower_modal.assert_not_called()
    warn_modal.assert_not_called()
    block_modal.assert_not_called()
