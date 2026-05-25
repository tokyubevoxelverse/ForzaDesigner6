"""Tests for forza_abyss_painter.cli.clean — the fap-clean CLI."""
import json
from pathlib import Path

import pytest

from forza_abyss_painter.cli.clean import (
    clean_doc,
    main,
    _identify_padding_whites,
    _compute_visible_pixel_counts,
    _padding_margin_px,
    _is_white,
    _WHITE_THRESHOLD,
)


def _shape(x=50.0, y=50.0, rx=10.0, ry=10.0, angle=0.0, color=(128, 128, 128, 255),
           type_="rotated_ellipse"):
    return {"type": type_, "x": x, "y": y, "rx": rx, "ry": ry, "angle": angle,
            "color": list(color)}


def _doc(shapes, image_size=(200, 200)):
    return {
        "format": "fd6.shapes",
        "version": 1,
        "image_size": list(image_size),
        "shape_count": len(shapes),
        "sticker_mode": False,
        "shapes": list(shapes),
    }


def test_padding_margin_matches_notebook_formula():
    """The CLI's padding-margin formula MUST match what the notebook used to compute
    pad_px in _load_image_bytes(). If these drift apart we'd misidentify which shapes
    are in the padding region."""
    # Notebook formula: pad_px = max(8, round(max(w,h) * 0.08))
    assert _padding_margin_px((100, 100)) == 8     # max(8, 8) = 8
    assert _padding_margin_px((1000, 500)) == 80   # max(8, 80) = 80
    assert _padding_margin_px((1263, 765)) == 101  # max(8, round(1263*0.08)) = 101
    assert _padding_margin_px((50, 50)) == 8       # floor-clamped at 8


def test_is_white_threshold():
    assert _is_white([255, 255, 255, 255])
    assert _is_white([230, 230, 230, 255])
    assert not _is_white([229, 255, 255, 255])  # one channel below
    assert not _is_white([255, 255, 100, 255])  # blue too low
    assert not _is_white([0, 0, 0, 255])


def test_identify_padding_whites_drops_only_white_in_margin():
    """A white shape inside the source region must STAY (legit white feature).
    A non-white shape in the padding region must STAY (Option A is white-only).
    A white shape in the padding region must DROP."""
    # 200x200 canvas → pad = max(8, round(200*0.08)) = 16
    shapes = [
        _shape(x=5, y=5, color=(255, 255, 255, 255)),       # 0: white in padding   → DROP
        _shape(x=100, y=100, color=(255, 255, 255, 255)),   # 1: white in source    → keep
        _shape(x=5, y=5, color=(0, 0, 0, 255)),             # 2: black in padding   → keep
        _shape(x=190, y=190, color=(255, 255, 255, 255)),   # 3: white in padding   → DROP
        _shape(x=100, y=20, color=(255, 255, 255, 255)),    # 4: white at edge but y=20 is inside (>=16) → keep
        _shape(x=100, y=10, color=(255, 255, 255, 255)),    # 5: white at y=10 in padding → DROP
    ]
    drop = _identify_padding_whites(shapes, image_size=(200, 200))
    assert drop == {0, 3, 5}


def test_compute_visible_pixel_counts_fully_occluded_is_zero():
    """A shape COMPLETELY covered by a later shape has visible count == 0."""
    # Shape A is small + centered. Shape B is large + same center → fully covers A.
    shapes = [
        _shape(x=100, y=100, rx=5, ry=5),    # 0: small, will be occluded by 1
        _shape(x=100, y=100, rx=50, ry=50),  # 1: large, occludes 0
    ]
    counts = _compute_visible_pixel_counts(shapes, image_size=(200, 200))
    assert counts[0] == 0, "shape 0 (smaller, lower in z) should be fully occluded by shape 1"
    assert counts[1] > 100, "shape 1 (larger, top of z) should have many visible pixels"


def test_compute_visible_pixel_counts_no_overlap_keeps_both():
    """Non-overlapping shapes both have full visibility."""
    shapes = [
        _shape(x=50, y=50, rx=10, ry=10),
        _shape(x=150, y=150, rx=10, ry=10),
    ]
    counts = _compute_visible_pixel_counts(shapes, image_size=(200, 200))
    assert counts[0] > 100
    assert counts[1] > 100
    # Roughly equal — same size, neither occludes the other
    assert abs(counts[0] - counts[1]) < 5


def test_clean_doc_default_drops_padding_whites_and_dead():
    """Both passes engaged by default; both drop sets are reported separately."""
    shapes = [
        _shape(x=5, y=5, color=(255, 255, 255, 255)),       # padding-white → DROP
        _shape(x=100, y=100, rx=5, ry=5),                   # will be occluded by next
        _shape(x=100, y=100, rx=50, ry=50),                 # occludes prev → keep
        _shape(x=190, y=190, color=(255, 255, 255, 255)),   # padding-white (190 > 200-16=184) → DROP
        _shape(x=50, y=50, rx=20, ry=20),                   # standalone → keep
    ]
    doc = _doc(shapes)
    cleaned, report = clean_doc(doc)
    assert report["input_count"] == 5
    assert report["dropped_total"] == 3  # 2 padding-whites + 1 dead
    assert report["dropped_padding_whites_only"] == 2
    assert report["dropped_dead_shapes_only"] == 1
    assert report["dropped_both_conditions"] == 0
    assert report["output_count"] == 2
    assert cleaned["shape_count"] == 2


def test_clean_doc_disable_padding_whites():
    """--no-padding-whites means white-in-padding shapes survive."""
    shapes = [
        _shape(x=5, y=5, color=(255, 255, 255, 255)),     # padding-white but kept
        _shape(x=100, y=100, rx=5, ry=5),                 # occluded
        _shape(x=100, y=100, rx=50, ry=50),               # occludes prev
    ]
    cleaned, report = clean_doc(_doc(shapes), drop_padding_whites=False)
    # Only the dead shape gets dropped.
    assert report["dropped_total"] == 1
    assert report["dropped_padding_whites_only"] == 0
    assert report["output_count"] == 2


def test_clean_doc_disable_dead_shapes():
    """--no-dead-shapes means occluded shapes survive."""
    shapes = [
        _shape(x=5, y=5, color=(255, 255, 255, 255)),     # padding-white → DROP
        _shape(x=100, y=100, rx=5, ry=5),                 # occluded but kept
        _shape(x=100, y=100, rx=50, ry=50),               # keep
    ]
    cleaned, report = clean_doc(_doc(shapes), drop_dead_shapes=False)
    assert report["dropped_total"] == 1
    assert report["dropped_dead_shapes_only"] == 0
    assert report["output_count"] == 2


def test_clean_doc_does_not_mutate_input():
    """The input doc dict must NOT be modified by clean_doc."""
    shapes = [_shape(x=5, y=5, color=(255, 255, 255, 255))]
    doc = _doc(shapes)
    original_shape_count = doc["shape_count"]
    original_shapes_len = len(doc["shapes"])
    clean_doc(doc)
    assert doc["shape_count"] == original_shape_count
    assert len(doc["shapes"]) == original_shapes_len


def test_clean_doc_requires_image_size():
    """Without image_size we can't compute padding margin or visibility — must raise."""
    with pytest.raises(ValueError, match="image_size"):
        clean_doc({"shapes": [_shape()]})


def test_cli_main_writes_cleaned_file(tmp_path: Path):
    """End-to-end: CLI reads input, writes <input>_cleaned.json by default."""
    shapes = [
        _shape(x=5, y=5, color=(255, 255, 255, 255)),       # → DROP
        _shape(x=100, y=100, rx=20, ry=20),                 # keep
    ]
    src = tmp_path / "input.json"
    src.write_text(json.dumps(_doc(shapes)))
    rc = main([str(src)])
    assert rc == 0
    out = tmp_path / "input_cleaned.json"
    assert out.exists()
    out_doc = json.loads(out.read_text(encoding="utf-8"))
    assert out_doc["shape_count"] == 1


def test_cli_main_in_place_overwrites(tmp_path: Path):
    shapes = [
        _shape(x=5, y=5, color=(255, 255, 255, 255)),       # → DROP
        _shape(x=100, y=100, rx=20, ry=20),                 # keep
    ]
    src = tmp_path / "input.json"
    src.write_text(json.dumps(_doc(shapes)))
    rc = main([str(src), "--in-place"])
    assert rc == 0
    out_doc = json.loads(src.read_text(encoding="utf-8"))
    assert out_doc["shape_count"] == 1


def test_cli_main_report_mode_does_not_write(tmp_path: Path):
    """--report: stats printed, no output file created."""
    shapes = [_shape(x=5, y=5, color=(255, 255, 255, 255)), _shape(x=100, y=100, rx=20, ry=20)]
    src = tmp_path / "input.json"
    src.write_text(json.dumps(_doc(shapes)))
    rc = main([str(src), "--report"])
    assert rc == 0
    # No _cleaned.json should exist
    assert not (tmp_path / "input_cleaned.json").exists()
    # Source unchanged
    assert json.loads(src.read_text(encoding="utf-8"))["shape_count"] == len(shapes)


def test_cli_main_inplace_and_output_are_mutually_exclusive(tmp_path: Path):
    src = tmp_path / "input.json"
    src.write_text(json.dumps(_doc([])))
    rc = main([str(src), "--in-place", "-o", str(tmp_path / "out.json")])
    assert rc == 2   # error exit


def test_cli_main_missing_input_returns_2(tmp_path: Path):
    rc = main([str(tmp_path / "does_not_exist.json")])
    assert rc == 2
