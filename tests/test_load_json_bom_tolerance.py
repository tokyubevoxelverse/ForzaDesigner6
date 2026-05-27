"""load_json must accept UTF-8 JSONs with or without a BOM.

Cursor Run 6 R6.3eA: a test JSON saved by Notepad on Windows has a
UTF-8 BOM (EF BB BF prefix) which json.load() rejects with `Expecting
value: line 1 column 1 (char 0)` when read via encoding='utf-8'. Real
users hand-editing JSON in Notepad would hit this too. Tolerating BOM
is a one-line encoding change.
"""
from __future__ import annotations

import json

import pytest

from forza_abyss_painter.io.exporter import load_json


def _doc_dict() -> dict:
    return {
        "format": "fd6.shapes",
        "version": 1,
        "source_image": "x.png",
        "image_size": [64, 64],
        "shape_count": 1,
        "generated_at": "",
        "profile": "",
        "sticker_mode": False,
        "shapes": [
            {"type": "rotated_ellipse", "x": 16.0, "y": 32.0,
             "rx": 8.0, "ry": 8.0, "angle": 0.0,
             "color": [128, 128, 128, 255]},
        ],
    }


def test_load_json_accepts_no_bom(tmp_path):
    """Baseline: plain UTF-8 (no BOM) must keep working."""
    path = tmp_path / "no_bom.json"
    path.write_bytes(json.dumps(_doc_dict()).encode("utf-8"))
    doc = load_json(path)
    assert doc.format == "fd6.shapes"
    assert doc.shape_count == 1


def test_load_json_accepts_utf8_bom(tmp_path):
    """UTF-8 BOM (EF BB BF) prefix must be tolerated, not rejected."""
    path = tmp_path / "with_bom.json"
    payload = "﻿" + json.dumps(_doc_dict())  # add BOM
    path.write_bytes(payload.encode("utf-8"))
    # Sanity: confirm the file actually starts with the BOM bytes.
    assert path.read_bytes()[:3] == b"\xef\xbb\xbf", \
        "test fixture must actually carry a BOM"

    doc = load_json(path)
    assert doc.format == "fd6.shapes"
    assert doc.shape_count == 1


def test_load_json_rejects_invalid_json_after_bom(tmp_path):
    """A BOM followed by malformed JSON must still raise (not crash on
    BOM, but also not silently accept garbage)."""
    path = tmp_path / "bom_garbage.json"
    path.write_bytes(b"\xef\xbb\xbf{not json")
    with pytest.raises(Exception):
        load_json(path)


def test_load_json_str_path_accepts_bom(tmp_path):
    """Path-or-str API: string paths must work the same as Path."""
    path = tmp_path / "str_path_bom.json"
    path.write_bytes(b"\xef\xbb\xbf" + json.dumps(_doc_dict()).encode("utf-8"))
    doc = load_json(str(path))
    assert doc.format == "fd6.shapes"
