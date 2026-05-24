import json
from pathlib import Path

import pytest

from forza_abyss_painter.io.json_schema import FD6Document, FD6_FORMAT, FD6_VERSION
from forza_abyss_painter.io.exporter import save_json, load_json
from forza_abyss_painter.shapegen.shapes import RotatedEllipse


def test_document_roundtrip(tmp_path: Path):
    shapes = [
        RotatedEllipse(color=(255, 0, 0, 128), x=10.0, y=20.0, rx=5.0, ry=3.0, angle=45.0),
        RotatedEllipse(color=(0, 255, 0, 200), x=30.0, y=40.0, rx=8.0, ry=2.0, angle=0.0),
    ]
    doc = FD6Document.from_engine(
        source_image="test.png", image_size=(100, 100), shapes=shapes, profile_name="unit-test"
    )
    out = save_json(doc, tmp_path / "out.json")
    assert out.exists()

    loaded = load_json(out)
    assert loaded.format == FD6_FORMAT
    assert loaded.version == FD6_VERSION
    assert loaded.shape_count == 2
    assert loaded.image_size == (100, 100)
    materialized = loaded.materialize_shapes()
    assert len(materialized) == 2
    assert materialized[0].to_json() == shapes[0].to_json()


def test_unsupported_format_rejected(tmp_path: Path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"format": "forza.painter", "version": 1, "shapes": []}))
    with pytest.raises(ValueError):
        load_json(bad)


def test_unsupported_version_rejected(tmp_path: Path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"format": FD6_FORMAT, "version": 999, "shapes": []}))
    with pytest.raises(ValueError):
        load_json(bad)
