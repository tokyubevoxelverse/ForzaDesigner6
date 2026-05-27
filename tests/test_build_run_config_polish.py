"""build_run_config forwards joint_polish_steps from the preset so
fresh GPU runs polish at end (matching CPU calibration)."""
from __future__ import annotations

from pathlib import Path

import pytest

from forza_abyss_painter.gui.gpu_gen_worker import build_run_config
from forza_abyss_painter.gui.generate_dialog import LOCAL_PRESETS


def _img(tmp_path) -> Path:
    p = tmp_path / "src.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\n")
    return p


def test_preset_polish_steps_forwarded(tmp_path):
    """When the preset dict carries joint_polish_steps, build_run_config
    forwards it to the runner config dict."""
    preset = {
        "label": "Medium — 1000 shapes",
        "num_shapes": 1000,
        "max_resolution": 720,
        "random_samples": 8192,
        "joint_polish_steps": 150,
    }
    cfg = build_run_config(_img(tmp_path), tmp_path / "out.json", preset)
    assert cfg["joint_polish_steps"] == 150


def test_polish_steps_defaults_to_zero_when_missing(tmp_path):
    """Back-compat: presets without joint_polish_steps (custom user
    configs, older test fixtures) get 0 (no polish)."""
    preset = {
        "label": "Custom",
        "num_shapes": 100,
        "max_resolution": 360,
        "random_samples": 1024,
    }
    cfg = build_run_config(_img(tmp_path), tmp_path / "out.json", preset)
    assert cfg["joint_polish_steps"] == 0


def test_local_presets_all_have_polish_steps():
    """The shipped LOCAL_PRESETS must all carry joint_polish_steps per
    the spec calibration table."""
    expected = {
        "Lineart — 400 shapes": 100,
        "Headshot — 700 shapes": 150,
        "Medium — 1000 shapes": 150,
        "Hi-Res — 3000 shapes (FH6 closed only)": 250,
    }
    for p in LOCAL_PRESETS:
        label = p["label"]
        assert "joint_polish_steps" in p, (
            f"preset {label!r} missing joint_polish_steps"
        )
        assert p["joint_polish_steps"] == expected[label], (
            f"preset {label!r} polish_steps={p['joint_polish_steps']} "
            f"expected {expected[label]} per spec §3.4"
        )
