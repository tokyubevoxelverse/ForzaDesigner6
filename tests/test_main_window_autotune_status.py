"""When _start_gpu builds its preset dict, it must apply the back-prop
recommendation and write an auto-tune line to the status bar (#131).

Cannot test the actual GPU spawn (no torch on test host), so we exercise
the preset-dict construction via a module-level helper that the slot
calls.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from forza_abyss_painter.gui import main_window as main_window_mod  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def big_card_probe(monkeypatch):
    from forza_abyss_painter.runtime import nvidia_smi as ns_mod

    def fake_probe(*, force=False):
        return ns_mod.ProbeResult(
            available=True,
            free_mib=95 * 1024, total_mib=98 * 1024,
            name="NVIDIA RTX PRO 6000 Blackwell",
            driver_version="555.85",
            probed_at=0.0,
        )

    monkeypatch.setattr(
        "forza_abyss_painter.gui.main_window.probe_free_vram",
        fake_probe,
    )


def test_apply_autotune_to_preset_returns_bumped_dict(big_card_probe):
    """Pure helper: given a preset dict + probe, returns the dict with
    bumped max_resolution and an auto-tune message."""
    preset = {
        "label": "Lineart — 400 shapes",
        "num_shapes": 400,
        "max_resolution": 480,
        "random_samples": 4096,
    }
    bumped, message = main_window_mod._apply_autotune_to_preset(preset)
    assert bumped["max_resolution"] >= 480, "must never lower baked value"
    assert "auto-tuned" in message.lower()
    assert "GiB free" in message
    assert "95" in message   # the free-GiB number is present


def test_apply_autotune_preserves_baked_value_on_probe_failure(monkeypatch):
    from forza_abyss_painter.runtime import nvidia_smi as ns_mod

    def fake_probe(*, force=False):
        return ns_mod.ProbeResult(
            available=False, reason="nvidia-smi missing", probed_at=0.0,
        )

    monkeypatch.setattr(
        "forza_abyss_painter.gui.main_window.probe_free_vram",
        fake_probe,
    )
    preset = {
        "label": "Hi-Res — 3000 shapes",
        "num_shapes": 3000,
        "max_resolution": 1000,
        "random_samples": 12288,
    }
    bumped, message = main_window_mod._apply_autotune_to_preset(preset)
    assert bumped["max_resolution"] == 1000   # unchanged
    assert "probe unavailable" in message.lower() or \
           "safety floor" in message.lower()


def test_apply_autotune_does_not_mutate_input(big_card_probe):
    preset = {
        "label": "Test",
        "num_shapes": 1000,
        "max_resolution": 720,
        "random_samples": 8192,
    }
    bumped, _ = main_window_mod._apply_autotune_to_preset(preset)
    # Input dict must be untouched (caller may want to retry).
    assert preset["max_resolution"] == 720
    # Output dict may have the same max_resolution if back-prop returned
    # the floor — that's fine.
    assert "max_resolution" in bumped
