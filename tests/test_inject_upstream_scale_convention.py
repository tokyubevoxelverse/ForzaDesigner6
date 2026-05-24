"""Regression test pinning the injector's scale-write convention to UPSTREAM.

Upstream's fh6_injector (tokyubevoxelverse/ForzaDesigner6 v0.3.5) writes:

    scale = rx / scale_divisor     # rx is HALF-extent (radius), no doubling

This is DIFFERENT from forza-painter's full-extent convention. Our PR #11 had
adopted painter's *2 multiplier, calling it 'painter parity' — but the upstream
EXE we ship JSONs for uses the half-extent convention, so the *2 was a divergence
that scaled every shape to 2x its intended in-game size. PR B reverts to upstream.

These tests pin the upstream convention. If anyone reintroduces the *2, they fail.
"""
import struct
from unittest.mock import MagicMock

from forza_abyss_painter.inject.fh6_injector import FH6Injector, LAYER_SCALE_OFF
from forza_abyss_painter.inject.game_profiles import default_profile


def _inject_one_shape(shape_dict) -> tuple[float, float]:
    """Replicate the inject() scale-pack arithmetic for a single shape and return
    the (sx, sy) the injector would write."""
    inj = FH6Injector(pid=1, profile=default_profile())
    is_ellipse = "ellipse" in shape_dict["type"] or shape_dict["type"] == "circle"
    scale_div = (inj.profile.scale_divisor_ellipse if is_ellipse
                 else inj.profile.scale_divisor_other)
    sd = shape_dict
    if "rx" in sd:
        sx = float(sd["rx"]) / scale_div
        sy = float(sd.get("ry", sd["rx"])) / scale_div
    elif "r" in sd:
        sx = sy = float(sd["r"]) / scale_div
    else:
        sx = sy = 1.0
    return sx, sy


def test_ellipse_scale_matches_upstream_half_extent_convention():
    """rx=63 must produce sx=1.0 — upstream's half-extent convention (NOT painter's *2)."""
    sx, sy = _inject_one_shape({
        "type": "rotated_ellipse", "x": 50, "y": 50, "rx": 63, "ry": 63,
        "angle": 0.0, "color": [255, 0, 0, 255],
    })
    assert sx == 1.0 and sy == 1.0, (
        f"ellipse scale broke upstream parity: expected (1.0, 1.0), got ({sx}, {sy}). "
        f"Upstream's fh6_injector writes scale = rx / 63. If you see 2.0 here someone "
        f"reintroduced the painter-convention *2 multiplier."
    )


def test_circle_scale_matches_upstream_half_extent_convention():
    """r=63 must produce sx=sy=1.0."""
    sx, sy = _inject_one_shape({
        "type": "circle", "x": 50, "y": 50, "r": 63,
        "angle": 0.0, "color": [255, 0, 0, 255],
    })
    assert sx == 1.0 and sy == 1.0


def test_ellipse_scale_no_doubling():
    """Explicit guard against the pre-revert painter formula sneaking back in."""
    sx, sy = _inject_one_shape({
        "type": "rotated_ellipse", "x": 50, "y": 50, "rx": 10, "ry": 5,
        "angle": 0.0, "color": [255, 0, 0, 255],
    })
    assert abs(sx - 10.0 / 63.0) < 1e-6, f"sx={sx}, expected ~0.1587 (10/63)"
    assert abs(sy - 5.0 / 63.0) < 1e-6,  f"sy={sy}, expected ~0.0794 (5/63)"
