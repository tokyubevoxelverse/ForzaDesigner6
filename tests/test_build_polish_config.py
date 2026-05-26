"""build_polish_config converts PolishDialog values + paths into a
RunConfig-compatible dict that mode='polish_only' accepts."""
from __future__ import annotations

from pathlib import Path

from forza_abyss_painter.gui.gpu_gen_worker import build_polish_config
from forza_abyss_painter.runtime.torch_runner import RunConfig


def test_build_polish_config_round_trips_through_RunConfig(tmp_path):
    image = tmp_path / "img.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    shapes = tmp_path / "shapes.json"
    shapes.write_text("{}")
    out = tmp_path / "out.json"

    cfg_dict = build_polish_config(
        source_image_path=image,
        input_shapes_path=shapes,
        output_path=out,
        steps=200,
        lock_alpha=True,
        sticker_mode=False,
    )
    assert cfg_dict["mode"] == "polish_only"
    assert cfg_dict["image_path"] == str(image)
    assert cfg_dict["input_shapes_path"] == str(shapes)
    assert cfg_dict["output_json_path"] == str(out)
    assert cfg_dict["polish_steps_override"] == 200
    assert cfg_dict["lock_alpha"] is True

    # Round-trip through the parser proves the schema is valid.
    parsed = RunConfig.from_dict(cfg_dict)
    assert parsed.mode == "polish_only"
    assert parsed.input_shapes_path == shapes
    assert parsed.polish_steps_override == 200


def test_build_polish_config_passes_sticker_mode(tmp_path):
    image = tmp_path / "img.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    shapes = tmp_path / "shapes.json"
    shapes.write_text("{}")
    out = tmp_path / "out.json"

    cfg_dict = build_polish_config(
        source_image_path=image,
        input_shapes_path=shapes,
        output_path=out,
        steps=100,
        lock_alpha=True,
        sticker_mode=True,
    )
    assert cfg_dict["sticker_mode"] is True
