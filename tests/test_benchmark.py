import json

import numpy as np
from PIL import Image

from fd6.shapegen.benchmark import (
    benchmark_line_guide,
    load_benchmark_inputs,
    synthetic_line_guide,
    synthetic_target,
    write_benchmark_report,
)


def test_synthetic_benchmark_inputs_have_matching_shapes():
    target, guide, source, prepare_seconds, prepare_vram = load_benchmark_inputs(None, None, size=24)

    assert target.shape == (24, 24, 3)
    assert guide.shape == (24, 24)
    assert target.dtype == np.uint8
    assert guide.dtype == np.float32
    assert float(guide.max()) > 0.0
    assert source == "synthetic"
    assert prepare_seconds == 0.0
    assert prepare_vram is None


def test_benchmark_inputs_report_external_image_source(tmp_path):
    image = np.full((8, 8, 3), 180, dtype=np.uint8)
    image[2:6, 3:5] = 0
    image_path = tmp_path / "img.png"
    Image.fromarray(image, "RGB").save(image_path)
    guide_image = np.full((8, 8), 255, dtype=np.uint8)
    guide_image[:, 4] = 0
    guide_path = tmp_path / "guide.png"
    Image.fromarray(guide_image, "L").save(guide_path)

    target, guide, source, prepare_seconds, prepare_vram = load_benchmark_inputs(
        str(image_path),
        str(guide_path),
        size=24,
    )

    assert target.shape == (8, 8, 3)
    assert guide.shape == (8, 8)
    assert source == "image:guide.png"
    assert prepare_seconds == 0.0
    assert prepare_vram is None


def test_benchmark_line_guide_returns_recordable_metrics():
    target = synthetic_target(24)
    guide = synthetic_line_guide(target)

    report = benchmark_line_guide(
        target,
        guide,
        seed=11,
        stop_at=4,
        random_samples=6,
        mutated_samples=2,
        compute_backend="cpu",
    )

    assert report["parameters"]["seed"] == 11
    assert report["baseline"]["shape_count"] == 4
    assert report["line_guide"]["shape_count"] == 4
    assert report["baseline"]["elapsed_seconds"] >= 0.0
    assert report["line_guide"]["elapsed_seconds"] >= 0.0
    assert "environment" in report
    assert report["guide"]["source"] == "synthetic"
    assert report["guide"]["prepare_seconds"] == 0.0
    assert "prepare_vram_peak_mb" in report["guide"]
    assert "edge_f1" in report["delta"]
    assert "edge_f1_t1" in report["delta"]


def test_write_benchmark_report_creates_json(tmp_path):
    report = {"parameters": {"seed": 1}, "baseline": {}, "line_guide": {}, "delta": {}}
    out = tmp_path / "nested" / "report.json"

    write_benchmark_report(report, out)

    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded == report
