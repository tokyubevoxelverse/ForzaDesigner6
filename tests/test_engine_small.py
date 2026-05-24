import numpy as np
import pytest

from forza_abyss_painter.shapegen.engine import Engine, EngineConfig
from forza_abyss_painter.shapegen.profile import Profile
from forza_abyss_painter.shapegen.scoring import rms_error


def _make_target(size: int = 32) -> np.ndarray:
    """Make a target image with a clear high-contrast region so shape fitting reduces RMS."""
    arr = np.full((size, size, 3), 200, dtype=np.uint8)
    arr[8:24, 8:24] = (20, 30, 240)  # blue square center
    return arr


def test_engine_reduces_rms_over_first_shapes():
    target = _make_target(32)
    profile = Profile(
        name="tiny",
        stop_at=10,
        random_samples=40,
        mutated_samples=10,
        preview_every=5,
        save_at=[],
        save_every=0,
        max_resolution=64,
        max_threads=1,
        shape_types=["rotated_ellipse"],
    )
    engine = Engine(target, EngineConfig(profile=profile, seed=12345))
    initial = engine.rms
    final_rms = None
    for ev in engine.run():
        if ev.kind == "done":
            final_rms = ev.rms
    assert final_rms is not None
    assert final_rms <= initial, f"RMS did not decrease: initial={initial}, final={final_rms}"
    assert len(engine.shapes) == profile.stop_at


def test_engine_stops_when_requested():
    target = _make_target(32)
    profile = Profile(
        name="tiny",
        stop_at=100,
        random_samples=20,
        mutated_samples=5,
        preview_every=1,
        save_at=[],
        save_every=0,
        max_threads=1,
        shape_types=["circle"],
    )
    engine = Engine(target, EngineConfig(profile=profile, seed=1))
    events = []
    for i, ev in enumerate(engine.run()):
        events.append(ev)
        if i == 3:
            engine.request_stop()
    # After stop, engine should produce a `done` event and shapes < stop_at
    assert any(e.kind == "done" for e in events)
    assert len(engine.shapes) < profile.stop_at
