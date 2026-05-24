import random

import numpy as np
import pytest

from forza_abyss_painter.shapegen.shapes import (
    Ellipse, RotatedEllipse, Circle, Rectangle, RotatedRectangle, Triangle,
    SHAPE_REGISTRY, random_shape, shape_from_json,
)


ALL_TYPES = list(SHAPE_REGISTRY.keys())


@pytest.mark.parametrize("type_name", ALL_TYPES)
def test_random_then_rasterize_then_roundtrip(type_name):
    rng = random.Random(42)
    cls = SHAPE_REGISTRY[type_name]
    shape = cls.random(rng, 64, 48)
    mask, bbox = shape.rasterize_mask(64, 48)
    x0, y0, x1, y1 = bbox
    if mask.size:
        assert mask.shape == (y1 - y0, x1 - x0)
        assert mask.dtype == np.uint8

    payload = shape.to_json()
    assert payload["type"] == type_name
    restored = shape_from_json(payload)
    assert type(restored) is type(shape)
    assert restored.to_json() == payload


@pytest.mark.parametrize("type_name", ALL_TYPES)
def test_mutate_stays_in_bounds(type_name):
    rng = random.Random(7)
    cls = SHAPE_REGISTRY[type_name]
    shape = cls.random(rng, 100, 100)
    for _ in range(20):
        shape = shape.mutate(rng, 100, 100)
    # After mutation, bbox should still be clipped within image bounds.
    x0, y0, x1, y1 = shape.bbox(100, 100)
    assert 0 <= x0 <= x1 <= 100
    assert 0 <= y0 <= y1 <= 100


def test_random_shape_uses_allowed_types_only():
    rng = random.Random(0)
    allowed = ["ellipse", "circle"]
    for _ in range(20):
        s = random_shape(rng, 50, 50, allowed)
        assert s.type_name in allowed


def test_unknown_shape_type_rejected():
    with pytest.raises(ValueError):
        shape_from_json({"type": "nonsense", "x": 0, "y": 0})
