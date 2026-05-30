"""Prove the array-module-agnostic GPU scorer matches the CPU `score_shape`.

We run EllipseBatchSearcher with xp=numpy (so it executes on CPU here, no CUDA
needed) and compare its per-candidate score to scoring.score_shape for the same
ellipse + canvas + edge weight. The same code runs on CuPy on a real GPU, so
parity here is strong evidence the GPU path scores correctly too.
"""

import math

import numpy as np

from fd6.shapegen.gpu import EllipseBatchSearcher, resolve_backend, gpu_detected_without_install
from fd6.shapegen.shapes.ellipse import RotatedEllipse
from fd6.shapegen.scoring import compute_edge_weight, precompute_canvas_error, score_shape


def _setup(size=80):
    rs = np.random.RandomState(0)
    target = rs.randint(0, 255, (size, size, 3)).astype(np.uint8)
    canvas = np.full((size, size, 3), 128, np.uint8)
    alpha = np.full((size, size), 255, np.uint8)
    alpha[:6, :] = 0  # an edge-buffer-like ring so the sticker path is exercised
    alpha[-6:, :] = 0
    edge = compute_edge_weight(target, alpha).astype(np.float32)
    return target, canvas, alpha, edge


def test_batched_score_matches_score_shape():
    target, canvas, alpha, edge = _setup()
    searcher = EllipseBatchSearcher(target, alpha, edge, xp=np)

    full_sq, n = precompute_canvas_error(canvas, target, alpha, edge)
    cur = np.asarray(canvas, np.float32)

    # A spread of ellipses comfortably inside the opaque interior.
    ellipses = [
        RotatedEllipse(color=(0, 0, 0, 128), x=40, y=40, rx=15, ry=10, angle=0),
        RotatedEllipse(color=(0, 0, 0, 128), x=30, y=45, rx=12, ry=18, angle=37),
        RotatedEllipse(color=(0, 0, 0, 128), x=50, y=38, rx=9, ry=9, angle=120),
    ]
    params = np.array([[e.x, e.y, e.rx, e.ry, e.angle] for e in ellipses], np.float32)

    # Tile must cover the biggest ellipse — use the searcher's own sizing.
    scores, _colors = searcher._score_batch(params, cur, float(full_sq))

    for i, e in enumerate(ellipses):
        cpu_score, _color = score_shape(
            e, canvas, target, alpha,
            canvas_full_sq=full_sq, canvas_norm=n, edge_weight=edge,
        )
        gpu_score = float(scores[i])
        assert math.isfinite(cpu_score) and math.isfinite(gpu_score)
        # Allow a hair of slack for float32 reduction order + color trunc.
        assert abs(cpu_score - gpu_score) < 0.05, (
            f"ellipse {i}: cpu={cpu_score} gpu={gpu_score}"
        )


def test_sticker_rejection_matches():
    target, canvas, alpha, edge = _setup()
    searcher = EllipseBatchSearcher(target, alpha, edge, xp=np)
    full_sq, n = precompute_canvas_error(canvas, target, alpha, edge)
    cur = np.asarray(canvas, np.float32)

    # An ellipse straddling the transparent ring must be rejected (+inf) by both.
    bad = RotatedEllipse(color=(0, 0, 0, 128), x=40, y=2, rx=12, ry=12, angle=0)
    params = np.array([[bad.x, bad.y, bad.rx, bad.ry, bad.angle]], np.float32)
    gpu_score = float(searcher._score_batch(params, cur, float(full_sq))[0][0])
    cpu_score, _ = score_shape(bad, canvas, target, alpha,
                               canvas_full_sq=full_sq, canvas_norm=n, edge_weight=edge)
    assert math.isinf(cpu_score) and math.isinf(gpu_score)


def test_resolve_backend_is_safe():
    # Must never raise. 'cpu' is always CPU; 'auto' resolves without ever
    # triggering a network install (only uses the GPU if a runtime is already
    # present). We avoid calling the install path here so tests stay offline.
    assert resolve_backend("cpu") == "cpu"
    assert resolve_backend("auto") in ("cpu", "gpu")
    assert isinstance(gpu_detected_without_install(), bool)
