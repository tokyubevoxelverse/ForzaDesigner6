"""Tests for the FH6 injector's pure functions (process-independent parts).

The injector now uses the LiveryGroup + layer_table strategy (no static struct
layout). We test the helpers that don't require an attached process:
  - `_pack_color`: RGBA packing with forced alpha=255
  - `_is_user_ptr`: pointer-range validator
  - layer offset constants (smoke test — these are what the inject loop writes)
"""

import struct

from forza_abyss_painter.inject.fh6_injector import (
    _is_user_ptr,
    _pack_color,
    patterns_are_populated,
    COUNT_OFF, TABLE_OFF,
    LAYER_POS_OFF, LAYER_SCALE_OFF, LAYER_ROT_OFF,
    LAYER_COLOR_OFF, LAYER_MASK_OFF, LAYER_SHAPE_ID_OFF,
    SCALE_DIVISOR_ELLIPSE, SCALE_DIVISOR_OTHER,
    SHAPE_ID_ELLIPSE, SHAPE_ID_OTHER,
)


def test_patterns_are_populated_always_true():
    """LiveryGroup approach finds shapes dynamically; static patterns no longer required."""
    assert patterns_are_populated() is True


def test_pack_color_forces_alpha_255():
    shape = {"color": [255, 128, 64, 50]}  # alpha 50 should be overridden to 255
    out = _pack_color(shape)
    assert out == bytes([255, 128, 64, 255])


def test_pack_color_three_channel_input():
    shape = {"color": [10, 20, 30]}
    out = _pack_color(shape)
    assert out == bytes([10, 20, 30, 255])


def test_pack_color_missing_defaults_to_white():
    shape = {"x": 1, "y": 2}
    out = _pack_color(shape)
    assert out == bytes([255, 255, 255, 255])


def test_pack_color_clamps_to_byte_range():
    shape = {"color": [300, -5, 256, 0]}
    out = _pack_color(shape)
    # 300 & 0xFF = 44, -5 & 0xFF = 251, 256 & 0xFF = 0
    assert out == bytes([44, 251, 0, 255])


def test_is_user_ptr_validates_range():
    assert _is_user_ptr(0x1A2B3C4D5E6F) is True   # plausible heap addr
    assert _is_user_ptr(0) is False
    assert _is_user_ptr(0x1000) is False           # too low
    assert _is_user_ptr(0xFFFFFFFFFFFFFFFF) is False  # too high


def test_layer_offsets_distinct_and_sane():
    """Smoke test: the offsets the inject loop writes must not overlap unexpectedly."""
    offs = {
        "pos": LAYER_POS_OFF,
        "scale": LAYER_SCALE_OFF,
        "rot": LAYER_ROT_OFF,
        "color": LAYER_COLOR_OFF,
        "mask": LAYER_MASK_OFF,
        "shape": LAYER_SHAPE_ID_OFF,
    }
    # All non-negative, all under reasonable struct size
    for name, off in offs.items():
        assert 0 <= off < 0x200, f"{name} offset {off:#x} out of plausible range"
    # POS (8 bytes) must not overlap SCALE (8 bytes)
    assert LAYER_POS_OFF + 8 <= LAYER_SCALE_OFF
    # SCALE (8 bytes) must not overlap ROT (4 bytes)
    assert LAYER_SCALE_OFF + 8 <= LAYER_ROT_OFF
    # ROT (4 bytes) must not overlap COLOR (4 bytes)
    assert LAYER_ROT_OFF + 4 <= LAYER_COLOR_OFF


def test_group_offsets_distinct():
    # COUNT (u16, 2 bytes) must not overlap TABLE (u64, 8 bytes)
    assert COUNT_OFF + 2 <= TABLE_OFF


def test_scale_divisor_ellipse_is_63():
    """bvzrays-confirmed divisor; matches what works in-game for current FH6 build."""
    assert SCALE_DIVISOR_ELLIPSE == 63.0
    assert SCALE_DIVISOR_OTHER == 127.0


def test_shape_ids():
    assert SHAPE_ID_ELLIPSE == 102
    assert SHAPE_ID_OTHER == 101


def test_pack_color_returns_4_bytes():
    """Every input shape must produce exactly 4 bytes for write to LAYER_COLOR_OFF."""
    for shape in [{}, {"color": []}, {"color": [1, 2]}, {"color": [1, 2, 3]}, {"color": [1, 2, 3, 4]}]:
        assert len(_pack_color(shape)) == 4
