import json
from pathlib import Path

from forza_abyss_painter.inject.patterns_io import (
    FieldSpec, PatternEntry, PatternsFile, ShapeStruct,
    load_patterns, save_patterns, has_usable_patterns,
)


def test_default_load_returns_empty_when_file_missing(tmp_path: Path):
    pf = load_patterns(tmp_path / "nope.json")
    assert pf.patterns == []
    assert pf.shape_struct.stride_bytes == 0
    assert not has_usable_patterns(pf)


def test_roundtrip_save_and_load(tmp_path: Path):
    pf = PatternsFile(
        game_executable="forzahorizon6.exe",
        patterns=[
            PatternEntry(name="shape_array_ref", pattern="48 8B 05 ?? ?? ?? ??", offset_after_match=3),
        ],
        shape_struct=ShapeStruct(
            stride_bytes=64,
            fields=[
                FieldSpec(name="x", offset=0, type="f32"),
                FieldSpec(name="y", offset=4, type="f32"),
                FieldSpec(name="color_rgba8", offset=32, type="rgba8"),
            ],
        ),
        vinyl_group={"layer_count_offset": 16, "shape_array_ptr_offset": 24},
    )
    path = save_patterns(pf, tmp_path / "patterns.json")
    assert path.exists()
    loaded = load_patterns(path)
    assert len(loaded.patterns) == 1
    assert loaded.patterns[0].pattern == "48 8B 05 ?? ?? ?? ??"
    assert loaded.shape_struct.stride_bytes == 64
    assert loaded.shape_struct.field_by_name("x").type == "f32"
    assert loaded.vinyl_group["layer_count_offset"] == 16
    assert has_usable_patterns(loaded)


def test_default_empty_file_is_not_usable():
    # The shipped fh6_patterns.json should NOT report as ready until populated.
    real = Path(__file__).resolve().parent.parent / "fd6" / "inject" / "patterns" / "fh6_patterns.json"
    if real.exists():
        loaded = load_patterns(real)
        assert not has_usable_patterns(loaded)
