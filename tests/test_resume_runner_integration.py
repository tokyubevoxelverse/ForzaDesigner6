"""Subprocess: fresh run with seed_shapes_path → output contains
seeded shapes + newly-generated. Tests the full resume runner flow."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
PIL = pytest.importorskip("PIL")


def _image(path: Path, h=32, w=32):
    from PIL import Image
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, :w // 2] = (200, 80, 80)
    arr[:, w // 2:] = (80, 80, 200)
    Image.fromarray(arr, "RGB").save(path)


def _snapshot(path: Path, w=32, h=32) -> dict:
    shapes = [
        {"type": "rotated_ellipse", "x": 8.0, "y": 16.0, "rx": 4.0, "ry": 4.0,
         "angle": 0.0, "color": [128, 128, 128, 255]},
        {"type": "rotated_ellipse", "x": 16.0, "y": 16.0, "rx": 4.0, "ry": 4.0,
         "angle": 30.0, "color": [128, 128, 128, 255]},
        {"type": "rotated_ellipse", "x": 24.0, "y": 16.0, "rx": 4.0, "ry": 4.0,
         "angle": 60.0, "color": [128, 128, 128, 255]},
    ]
    doc = {
        "format": "fd6.shapes", "version": 1,
        "source_image": "img.png",
        "image_size": [w, h], "shape_count": len(shapes),
        "generated_at": "", "profile": "test",
        "sticker_mode": False, "shapes": shapes,
    }
    path.write_text(json.dumps(doc), encoding="utf-8")
    return doc


def test_resume_runner_appends_to_seed(tmp_path):
    image = tmp_path / "img.png"
    _image(image)
    snap = tmp_path / "out" / "fixture_3.json"
    snap.parent.mkdir(parents=True, exist_ok=True)
    in_doc = _snapshot(snap, w=32, h=32)

    out = tmp_path / "out" / "fixture.json"
    cfg = {
        "image_path": str(image),
        "output_json_path": str(out),
        "num_shapes": 6,
        "max_resolution": 32,
        "random_samples": 16,
        "seed_shapes_path": str(snap),
        "checkpoint_every": 0,
        "device": "cpu",
        "lock_alpha": True,
    }
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "-m", "forza_abyss_painter.runtime.torch_runner",
         "--config", str(cfg_path)],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, (
        f"runner exited {result.returncode}\nstderr:\n{result.stderr}"
    )
    out_doc = json.loads(out.read_text(encoding="utf-8"))
    assert out_doc["shape_count"] == 6
    for i in range(3):
        for key in ("x", "y", "rx", "ry", "angle"):
            assert abs(in_doc["shapes"][i][key] - out_doc["shapes"][i][key]) < 1e-6


def test_resume_runner_rejects_non_ellipse_seed(tmp_path):
    """Snapshot with a rectangle shape → runner emits clean error event
    before invoking run_gpu."""
    image = tmp_path / "img.png"
    _image(image)
    snap = tmp_path / "out" / "bad_2.json"
    snap.parent.mkdir(parents=True, exist_ok=True)
    snap.write_text(json.dumps({
        "format": "fd6.shapes", "version": 1,
        "source_image": "img.png",
        "image_size": [32, 32], "shape_count": 1,
        "generated_at": "", "profile": "test",
        "sticker_mode": False,
        "shapes": [
            {"type": "rectangle", "x": 16.0, "y": 16.0,
             "hw": 8.0, "hh": 8.0,
             "color": [100, 100, 100, 255]},
        ],
    }), encoding="utf-8")

    out = tmp_path / "out" / "fixture.json"
    cfg = {
        "image_path": str(image),
        "output_json_path": str(out),
        "num_shapes": 4,
        "max_resolution": 32,
        "random_samples": 16,
        "seed_shapes_path": str(snap),
        "checkpoint_every": 0,
        "device": "cpu",
        "lock_alpha": True,
    }
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "-m", "forza_abyss_painter.runtime.torch_runner",
         "--config", str(cfg_path)],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode != 0
    assert "resume_unsupported_shape" in result.stderr or \
           "rotated_ellipse" in result.stderr
    assert not out.exists(), "output written despite resume error"
