"""RunConfig.seed_canvas_size + build_run_config plumbing.

Task 7 of the VRAM honesty correction plan. engine.run_gpu gained a
seed_canvas_size kwarg in Task 6 (commit 7cfaa8e) so that seeded shape
coords get rescaled when a snapshot's canvas dims don't match the
current run's max_resolution-derived canvas. This test pins the IPC
plumbing that carries the optional value from the GUI builder
through the JSON config boundary into the RunConfig dataclass that
torch_runner consumes.

Default None means existing fresh/polish/resume call sites keep their
exact current behavior — only ResumeDialog (Task 8) and _on_resume
(Task 9) will populate it.

JSON round-trip safety: dicts the builder writes get serialized to a
JSON config file the subprocess reads, where tuples become lists. The
RunConfig parser must coerce list -> tuple.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from forza_abyss_painter.runtime.torch_runner import RunConfig


def _fresh_dict(tmp_path) -> dict:
    image = tmp_path / "img.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    return {
        "image_path": str(image),
        "output_json_path": str(tmp_path / "out.json"),
        "num_shapes": 100,
        "max_resolution": 360,
        "random_samples": 1024,
    }


# ----------------------------------------------------------- RunConfig


def test_default_seed_canvas_size_is_none(tmp_path):
    """Omitting the key keeps the field None — existing call sites
    unaffected."""
    cfg = RunConfig.from_dict(_fresh_dict(tmp_path))
    assert hasattr(cfg, "seed_canvas_size")
    assert cfg.seed_canvas_size is None


def test_seed_canvas_size_parses_tuple_from_list(tmp_path):
    """JSON serializes tuples as lists; from_dict must coerce back to
    tuple so engine.run_gpu (which type-checks with isinstance) gets the
    canonical type."""
    d = _fresh_dict(tmp_path)
    d["seed_canvas_size"] = [720, 540]
    cfg = RunConfig.from_dict(d)
    assert cfg.seed_canvas_size == (720, 540)
    assert isinstance(cfg.seed_canvas_size, tuple)


def test_seed_canvas_size_accepts_tuple(tmp_path):
    """Direct tuple (pre-JSON-roundtrip) flows through unchanged."""
    d = _fresh_dict(tmp_path)
    d["seed_canvas_size"] = (480, 360)
    cfg = RunConfig.from_dict(d)
    assert cfg.seed_canvas_size == (480, 360)
    assert isinstance(cfg.seed_canvas_size, tuple)


def test_seed_canvas_size_survives_json_roundtrip(tmp_path):
    """Full IPC simulation: builder dict -> json.dumps -> json.loads ->
    RunConfig.from_dict -> tuple. This is the actual path the runner
    subprocess takes (Path(args.config).read_text -> json.loads)."""
    d = _fresh_dict(tmp_path)
    d["seed_canvas_size"] = (720, 540)
    serialized = json.dumps(d, default=str)
    reloaded = json.loads(serialized)
    cfg = RunConfig.from_dict(reloaded)
    assert cfg.seed_canvas_size == (720, 540)
    assert isinstance(cfg.seed_canvas_size, tuple)


# ----------------------------------------------------------- build_run_config


def test_build_run_config_default_seed_canvas_size_omitted(tmp_path):
    """Default kwarg is None; the builder either omits the key OR
    emits None — both are valid for from_dict's optional handling."""
    from forza_abyss_painter.gui.gpu_gen_worker import build_run_config

    preset = {
        "label": "X", "num_shapes": 100, "max_resolution": 360,
        "random_samples": 1024,
    }
    cfg = build_run_config(tmp_path / "x.png", tmp_path / "x.json", preset)
    # Whether the key is absent or present-as-None, from_dict resolves
    # both to seed_canvas_size = None.
    parsed = RunConfig.from_dict({
        **cfg,
        "image_path": str(tmp_path / "x.png"),
        "output_json_path": str(tmp_path / "x.json"),
    })
    # Need image_path to exist for from_dict (it doesn't existence-check
    # image, only seed_shapes_path) — actually image_path only Path-wraps,
    # never stat. But re-write to be safe across future tighten-ups:
    (tmp_path / "x.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    parsed2 = RunConfig.from_dict(cfg)
    assert parsed2.seed_canvas_size is None


def test_build_run_config_forwards_seed_canvas_size(tmp_path):
    """The kwarg makes it into the dict the worker hands to the runner.
    Task 9 will wire ResumeDialog to call this; Task 8 will populate
    the value from the snapshot's image_size header."""
    from forza_abyss_painter.gui.gpu_gen_worker import build_run_config

    (tmp_path / "x.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    preset = {
        "label": "X", "num_shapes": 100, "max_resolution": 360,
        "random_samples": 1024,
    }
    cfg = build_run_config(
        tmp_path / "x.png", tmp_path / "x.json", preset,
        seed_canvas_size=(720, 540),
    )
    assert cfg["seed_canvas_size"] == (720, 540)
    # Round-trip through from_dict — the full IPC contract.
    parsed = RunConfig.from_dict(cfg)
    assert parsed.seed_canvas_size == (720, 540)
    assert isinstance(parsed.seed_canvas_size, tuple)
