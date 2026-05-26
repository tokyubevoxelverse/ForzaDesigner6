"""RunConfig accepts mode='polish_only' with conditional-required fields."""
from __future__ import annotations

from pathlib import Path

import pytest

from forza_abyss_painter.runtime.torch_runner import RunConfig


def _polish_dict(tmp_path) -> dict:
    image = tmp_path / "img.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    shapes = tmp_path / "shapes.json"
    shapes.write_text("{}")
    return {
        "image_path": str(image),
        "output_json_path": str(tmp_path / "out.json"),
        "mode": "polish_only",
        "input_shapes_path": str(shapes),
        "polish_steps_override": 100,
        # Fresh-mode required fields stay absent in polish_only.
    }


def _fresh_dict(tmp_path) -> dict:
    image = tmp_path / "img.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    return {
        "image_path": str(image),
        "output_json_path": str(tmp_path / "out.json"),
        "num_shapes": 100,
        "max_resolution": 480,
        "random_samples": 1024,
    }


def test_default_mode_is_fresh(tmp_path):
    cfg = RunConfig.from_dict(_fresh_dict(tmp_path))
    assert cfg.mode == "fresh"
    assert cfg.input_shapes_path is None
    assert cfg.polish_steps_override is None


def test_polish_only_mode_parses(tmp_path):
    cfg = RunConfig.from_dict(_polish_dict(tmp_path))
    assert cfg.mode == "polish_only"
    assert cfg.input_shapes_path == Path(_polish_dict(tmp_path)["input_shapes_path"])
    assert cfg.polish_steps_override == 100


def test_polish_only_does_not_require_num_shapes(tmp_path):
    """num_shapes / max_resolution / random_samples are required for fresh
    but not for polish_only. Their absence in polish_only is OK."""
    d = _polish_dict(tmp_path)
    cfg = RunConfig.from_dict(d)
    assert cfg.num_shapes == 0   # placeholder default in polish_only
    assert cfg.max_resolution == 0
    assert cfg.random_samples == 0


def test_polish_only_requires_input_shapes_path(tmp_path):
    d = _polish_dict(tmp_path)
    del d["input_shapes_path"]
    with pytest.raises(ValueError, match="input_shapes_path"):
        RunConfig.from_dict(d)


def test_polish_only_requires_existing_input_shapes_path(tmp_path):
    d = _polish_dict(tmp_path)
    d["input_shapes_path"] = str(tmp_path / "does_not_exist.json")
    with pytest.raises(ValueError, match="not found"):
        RunConfig.from_dict(d)


def test_unknown_mode_raises(tmp_path):
    d = _polish_dict(tmp_path)
    d["mode"] = "frobnicate"
    with pytest.raises(ValueError, match="mode"):
        RunConfig.from_dict(d)


def test_fresh_mode_still_requires_num_shapes(tmp_path):
    d = _fresh_dict(tmp_path)
    del d["num_shapes"]
    with pytest.raises(ValueError, match="num_shapes"):
        RunConfig.from_dict(d)


def test_polish_steps_override_optional(tmp_path):
    d = _polish_dict(tmp_path)
    del d["polish_steps_override"]
    cfg = RunConfig.from_dict(d)
    assert cfg.polish_steps_override is None


def test_polish_steps_override_must_be_positive(tmp_path):
    d = _polish_dict(tmp_path)
    d["polish_steps_override"] = 0
    with pytest.raises(ValueError, match="polish_steps_override"):
        RunConfig.from_dict(d)
