"""Local smoke (CLAUDE.md §1b + §8g) — construct real MainWindow under
offscreen Qt, load a real JSON, verify Re-shape-gen + Polish buttons
become visible. Then call _run_polish_only directly with real inputs
and assert the output JSON validates clean.

Per CLAUDE.md §8h: ONE MainWindow per process. Both assertions share
the single construction.

Skipped when torch is not importable.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
PIL = pytest.importorskip("PIL")

from PySide6.QtWidgets import QApplication  # noqa: E402

from forza_abyss_painter.gui import feature_flags  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(autouse=True)
def both_flags_on(monkeypatch):
    monkeypatch.setattr(feature_flags, "RESHAPE_GEN_AVAILABLE", True)
    monkeypatch.setattr(feature_flags, "POLISH_LOADED_AVAILABLE", True)


def _write_image(path: Path, h=64, w=64):
    from PIL import Image
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, :w // 2] = (200, 80, 80)
    arr[:, w // 2:] = (80, 80, 200)
    Image.fromarray(arr, "RGB").save(path)


def _write_shapes_json(path: Path, w=64, h=64):
    doc = {
        "format": "fd6.shapes",
        "version": 1,
        "source_image": "img.png",
        "image_size": [w, h],
        "shape_count": 3,
        "generated_at": "",
        "profile": "smoke",
        "sticker_mode": False,
        "shapes": [
            {"type": "rotated_ellipse", "x": 16.0, "y": 32.0, "rx": 8.0,
             "ry": 8.0, "angle": 0.0, "color": [128, 128, 128, 255]},
            {"type": "rotated_ellipse", "x": 32.0, "y": 32.0, "rx": 8.0,
             "ry": 8.0, "angle": 30.0, "color": [128, 128, 128, 255]},
            {"type": "rotated_ellipse", "x": 48.0, "y": 32.0, "rx": 8.0,
             "ry": 8.0, "angle": 60.0, "color": [128, 128, 128, 255]},
        ],
    }
    path.write_text(json.dumps(doc), encoding="utf-8")


def test_tier_b_smoke(qapp, tmp_path):
    """Single test, single MainWindow:
      Part A — construct MainWindow, simulate JSON load via the same slot
               the upload panel signals, assert both buttons become visible.
      Part B — directly invoke _run_polish_only with the same inputs,
               assert the output JSON validates clean.

    Part B doesn't go through GpuGenWorker (which would need the embedded
    python that isn't installed in test envs). The subprocess hop is
    covered by tests/test_polish_runner_integration.py."""
    img = tmp_path / "img.png"
    _write_image(img)
    shapes = tmp_path / "shapes.json"
    _write_shapes_json(shapes)

    # ---- Part A: real MainWindow + button visibility ----
    from forza_abyss_painter.gui.main_window import MainWindow
    win = MainWindow()
    try:
        win._on_json_loaded_for_preview(shapes)
        assert win._loaded_json_path == shapes
        # Qt requires the ancestor chain to be shown before isVisible()
        # reflects child state — show the window briefly, then hide.
        win.show()
        try:
            assert win.upload.reshape_btn.isVisible(), (
                "Re-shape-gen button did not become visible after JSON load — "
                "set_json_loaded wiring is missing"
            )
            assert win.upload.polish_btn.isVisible(), (
                "Polish button did not become visible after JSON load — "
                "set_json_loaded wiring is missing"
            )
        finally:
            win.hide()
    finally:
        # Explicit cleanup so any second test construction wouldn't trip
        # the deleteLater hazard (CLAUDE.md §8h).
        win.close()
        win.deleteLater()

    # ---- Part B: polish runner end-to-end ----
    from forza_abyss_painter.runtime.torch_runner import RunConfig, _run_polish_only
    from forza_abyss_painter.io.exporter import load_json
    from forza_abyss_painter.io.validator import Severity, validate_document

    out_path = tmp_path / "shapes_polished.json"
    cfg = RunConfig.from_dict({
        "image_path": str(img),
        "output_json_path": str(out_path),
        "mode": "polish_only",
        "input_shapes_path": str(shapes),
        "polish_steps_override": 50,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "lock_alpha": True,
    })

    class _StubLogger:
        def log(self, *a, **kw): pass
        def log_exception(self, *a, **kw): pass
        def start_phase(self, *a, **kw):
            from contextlib import nullcontext
            return nullcontext()

    rc = _run_polish_only(cfg, sys.stderr, _StubLogger())
    assert rc == 0, f"polish_only exited {rc}"
    assert out_path.is_file()

    polished = load_json(str(out_path))
    issues = validate_document(polished.to_dict())
    errors = [i for i in issues if i.severity is Severity.ERROR]
    assert not errors, f"polished JSON failed validation: {errors}"
    assert polished.shape_count == 3
