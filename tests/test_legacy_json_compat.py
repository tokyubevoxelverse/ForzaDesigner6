"""Compat tests for the legacy geometrize / forza-painter JSON format.

We accept two top-level shapes for `Upload JSON` in the EXE:
  1. Our native `fd6.shapes` format
  2. The legacy painter format (no `format` key, integer shape types,
     `data` arrays for geometry)

The legacy format is what `forza-painter`, geometrize-lib, and earlier
upstream-ForzaDesigner6 outputs produce. Without compat support a user
with a pre-existing painter JSON (eg shared by a friend or downloaded
from a community pack) gets a hard "Unsupported document format" error
when they Upload JSON in our EXE — which is exactly the regression
that drove this work (erebus's chibi nihilister.1000.json).
"""
from __future__ import annotations

import json
import pytest

from forza_abyss_painter.io.json_schema import FD6Document, FD6_FORMAT


# --- Helpers ----------------------------------------------------------------

def _painter_json(
    image_size: tuple[int, int] = (256, 256),
    bg_alpha: int = 0,
    drawables: list[dict] | None = None,
) -> dict:
    """Build a minimal painter-format JSON dict. Mirrors the convention used
    by forza-painter-fh6/src/geometry_json.py and geometrize-lib."""
    w, h = image_size
    bg = {
        "type": 1,                            # RECTANGLE
        "data": [0, 0, w, h],                 # x, y, w, h (top-left + full size)
        "color": [128, 128, 128, bg_alpha],
        "score": 0.0,
    }
    return {"shapes": [bg, *(drawables or [])]}


# --- Detection --------------------------------------------------------------

def test_legacy_detected_when_no_format_key_and_integer_type_with_data_array():
    """Trigger condition: no `format` key, first shape has integer type AND data array."""
    legacy = _painter_json(drawables=[
        {"type": 16, "data": [50, 50, 30, 20, 45],
         "color": [255, 0, 0, 255], "score": 0.5},
    ])
    doc = FD6Document.from_dict(legacy)
    assert doc.format == FD6_FORMAT, "legacy converts to our native format"
    assert doc.profile == "legacy-geometrize-import"


def test_native_format_still_loads_through_the_same_entry_point():
    """The compat path must NOT break existing native JSONs. Native takes the
    fast path (no detection / conversion overhead)."""
    native = {
        "format": "fd6.shapes",
        "version": 1,
        "image_size": [100, 100],
        "shape_count": 1,
        "shapes": [{"type": "rotated_ellipse", "x": 50, "y": 50,
                    "rx": 10, "ry": 5, "angle": 0,
                    "color": [255, 0, 0, 255]}],
    }
    doc = FD6Document.from_dict(native)
    assert doc.format == FD6_FORMAT
    assert doc.profile == ""   # not the legacy import marker


def test_unknown_format_still_errors():
    """A `format` key with an UNKNOWN value (eg from a future version or a
    different tool) still raises — we don't silently treat unknown formats
    as legacy."""
    with pytest.raises(ValueError, match="Unsupported document format"):
        FD6Document.from_dict({"format": "geometrize.v999", "shapes": []})


# --- Conversion semantics ---------------------------------------------------

def test_legacy_extracts_image_size_from_bg_rect_data():
    """The painter convention: shapes[0] is the BG rectangle. data[2:4] is
    the canvas size. Our compat path must pull this out as image_size."""
    legacy = _painter_json(image_size=(900, 1200), drawables=[
        {"type": 16, "data": [100, 200, 30, 40, 0],
         "color": [0, 255, 0, 255], "score": 0.1},
    ])
    doc = FD6Document.from_dict(legacy)
    assert doc.image_size == (900, 1200)


def test_legacy_transparent_bg_is_skipped():
    """painter's load_geometry only emits the BG if its alpha > 0 (main.py:222).
    We mirror that — alpha=0 means the source had a transparent backdrop and
    we shouldn't synthesize a fake one."""
    legacy = _painter_json(bg_alpha=0, drawables=[
        {"type": 16, "data": [50, 50, 10, 10, 0],
         "color": [255, 0, 0, 255], "score": 0.5},
    ])
    doc = FD6Document.from_dict(legacy)
    # Only the drawable, not the transparent BG.
    assert doc.shape_count == 1
    assert doc.shapes[0]["type"] == "rotated_ellipse"


def test_legacy_visible_bg_emitted_as_centered_rectangle():
    """Visible BG (alpha > 0) → emit as a centered rectangle covering the
    canvas. Matches painter main.py:222 which re-creates it as
    `Shape(1, image_w//2, image_h//2, image_w, image_h, 0, ...)`."""
    legacy = _painter_json(image_size=(400, 300), bg_alpha=200, drawables=[
        {"type": 16, "data": [100, 100, 20, 20, 0],
         "color": [0, 255, 0, 255], "score": 0.5},
    ])
    doc = FD6Document.from_dict(legacy)
    assert doc.shape_count == 2   # BG + drawable
    bg = doc.shapes[0]
    assert bg["type"] == "rectangle"
    assert bg["x"] == 200.0 and bg["y"] == 150.0   # canvas center
    assert bg["hw"] == 200.0 and bg["hh"] == 150.0  # full canvas half-extents
    assert bg["color"][3] == 200


def test_legacy_rotated_ellipse_data_w_h_are_already_radii():
    """Geometrize's quirk: for ROTATED_ELLIPSE, data[2] and data[3] are
    ALREADY radii (half-extents), NOT full extents. Painter passes them
    directly to cv2.ellipse's `axes` parameter (which takes radii) and
    writes `w / 63` to the game expecting a radius value. So our rx/ry
    must come THROUGH unhalved.

    This was a wrong-by-2x bug that rendered every ellipse at quarter-
    area: erebus's chibi nihilister.1000.json loaded but the preview
    canvas was dominated by tiny dots because every ellipse was at 1/4
    its intended radius."""
    legacy = _painter_json(drawables=[
        {"type": 16, "data": [128, 128, 100, 60, 90],
         "color": [255, 128, 64, 255], "score": 0.3},
    ])
    doc = FD6Document.from_dict(legacy)
    s = doc.shapes[0]
    assert s["type"] == "rotated_ellipse"
    assert s["x"] == 128.0 and s["y"] == 128.0   # CENTER preserved
    assert s["rx"] == 100.0 and s["ry"] == 60.0  # w/h passed through as radii
    assert s["angle"] == 90.0


def test_legacy_rectangle_data_is_center_with_full_extents():
    """RECTANGLE (type=1, drawable): data=[x,y,w,h] same convention as
    rotated_ellipse minus rotation. Convert to our `rectangle` with hw/hh
    = w/2, h/2."""
    legacy = _painter_json(drawables=[
        {"type": 1, "data": [100, 100, 40, 20],
         "color": [50, 100, 150, 255], "score": 0.2},
    ])
    doc = FD6Document.from_dict(legacy)
    # shapes[0] is the drawable rect (bg was alpha=0 so skipped)
    s = doc.shapes[0]
    assert s["type"] == "rectangle"
    assert s["x"] == 100.0 and s["y"] == 100.0
    assert s["hw"] == 20.0 and s["hh"] == 10.0


def test_legacy_transparent_drawable_shapes_skipped():
    """Painter's load_geometry skips drawables with alpha <= 0 (main.py:227).
    Mirror this — an alpha=0 shape can't render, no point keeping it."""
    legacy = _painter_json(drawables=[
        {"type": 16, "data": [50, 50, 10, 10, 0],
         "color": [255, 0, 0, 0],     # alpha 0 — skip
         "score": 0.5},
        {"type": 16, "data": [60, 60, 10, 10, 0],
         "color": [0, 255, 0, 255],   # alpha 255 — keep
         "score": 0.5},
    ])
    doc = FD6Document.from_dict(legacy)
    assert doc.shape_count == 1
    assert doc.shapes[0]["color"][1] == 255   # the green one


def test_legacy_unsupported_shape_types_dropped():
    """Geometrize has more shape kinds (triangle=3, circle=4, ellipse=8, ...)
    but painter only handles RECTANGLE=1 and ROTATED_ELLIPSE=16. We do the
    same — anything else is silently dropped (returns None from converter)."""
    legacy = _painter_json(drawables=[
        {"type": 3, "data": [10, 10, 20, 20, 30, 30],   # triangle
         "color": [255, 0, 0, 255], "score": 0.1},
        {"type": 16, "data": [50, 50, 10, 10, 0],       # rotated_ellipse OK
         "color": [0, 255, 0, 255], "score": 0.5},
    ])
    doc = FD6Document.from_dict(legacy)
    assert doc.shape_count == 1
    assert doc.shapes[0]["type"] == "rotated_ellipse"


def test_legacy_drawables_materialize_to_rotated_ellipse_instances():
    """End-to-end: load → materialize → assert the Shape instances have the
    right fields. This is the path the injector takes from JSON to inject."""
    from forza_abyss_painter.shapegen.shapes.ellipse import RotatedEllipse
    legacy = _painter_json(drawables=[
        {"type": 16, "data": [100, 100, 50, 30, 45],
         "color": [255, 128, 64, 255], "score": 0.5},
    ])
    doc = FD6Document.from_dict(legacy)
    shapes = doc.materialize_shapes()
    assert len(shapes) == 1
    s = shapes[0]
    assert isinstance(s, RotatedEllipse)
    assert s.x == 100.0 and s.y == 100.0
    assert s.rx == 50.0 and s.ry == 30.0   # w/h passed through as radii (not halved)
    assert s.angle == 45.0
    assert s.color == (255, 128, 64, 255)


def test_legacy_empty_shapes_array_raises():
    """No shapes at all = caller passed garbage. Surface a clear error."""
    with pytest.raises(ValueError, match="empty shapes"):
        FD6Document.from_dict({"shapes": []})


def test_legacy_bg_with_short_data_raises():
    """BG rect must have at least [bg_x, bg_y, image_w, image_h]. Anything
    shorter means we can't extract image dimensions — fail clearly."""
    bad = {"shapes": [
        {"type": 1, "data": [0, 0],   # missing w, h
         "color": [0, 0, 0, 0], "score": 0},
    ]}
    with pytest.raises(ValueError, match="BG shape data too short"):
        FD6Document.from_dict(bad)


# --- Real-world fixture (the erebus repro) ----------------------------------

# If the JSON that drove this bug is present in the test env (eg on the
# original dev box), use it as a final sanity check. Skipped on CI where
# the file isn't checked in (it's a third-party design we don't ship).
import os
_EREBUS_PATH = "/Users/kusanagi/Downloads/chibi nihilister.1000.json"


@pytest.mark.skipif(not os.path.exists(_EREBUS_PATH),
                    reason="erebus nihilister fixture not present in this env")
def test_legacy_erebus_nihilister_loads_end_to_end():
    """The original repro: chibi nihilister.1000.json from upstream / painter
    pipeline. 900x900, 1000 rotated_ellipse shapes after BG skip."""
    with open(_EREBUS_PATH) as f:
        data = json.load(f)
    doc = FD6Document.from_dict(data)
    assert doc.image_size == (900, 900)
    # BG is alpha=0 → skipped; 1000 drawables come through.
    assert doc.shape_count == 1000
    shapes = doc.materialize_shapes()
    assert len(shapes) == 1000
    # All drawables are rotated_ellipse in this file.
    from forza_abyss_painter.shapegen.shapes.ellipse import RotatedEllipse
    assert all(isinstance(s, RotatedEllipse) for s in shapes)
