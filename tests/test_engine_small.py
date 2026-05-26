import random

import numpy as np
import pytest
import math

import fd6.shapegen.engine as engine_module
from fd6.shapegen.engine import Engine, EngineConfig
from fd6.shapegen.profile import Profile
from fd6.shapegen.quality import build_quality_context, precompute_gradient_error, precompute_weighted_rgb_error
from fd6.shapegen.scoring import precompute_canvas_error, score_shape
from fd6.shapegen.shapes import SHAPE_REGISTRY
from fd6.shapegen.shapes.circle import Circle
from fd6.shapegen.shapes.ellipse import Ellipse, RotatedEllipse
from fd6.shapegen.shapes.rectangle import Rectangle, RotatedRectangle
from fd6.shapegen.shapes.triangle import Triangle
from fd6.shapegen.torch_backend import ComputeBackendInfo, TorchSearchRuntime


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
        compute_backend="cpu",
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


def test_engine_post_refine_revisits_seeded_shape():
    target = _make_target(32)
    profile = Profile(
        name="post-refine",
        stop_at=1,
        random_samples=20,
        mutated_samples=96,
        refine_passes=1,
        preview_every=0,
        save_at=[],
        save_every=0,
        max_resolution=64,
        max_threads=1,
        compute_backend="cpu",
        shape_types=["rotated_ellipse"],
    )
    engine = Engine(target, EngineConfig(profile=profile, seed=123))
    engine._refine_mutation_budget = lambda: 48
    engine.seed_shapes([
        RotatedEllipse(color=(0, 0, 0, 128), x=10.0, y=10.0, rx=5.0, ry=5.0, angle=0.0),
    ])
    initial = engine.rms
    final_rms = None
    for event in engine.run():
        if event.kind == "done":
            final_rms = event.rms
    assert final_rms is not None
    assert final_rms < initial
    assert len(engine.shapes) == 1


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
        compute_backend="cpu",
        shape_types=["circle"],
        edge_weight_strength=0.0,
        gradient_weight=0.0,
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


def test_gpu_rotated_ellipse_preview_syncs_dirty_canvas(monkeypatch):
    class FakeRuntime:
        def __init__(self, target, alpha_mask):
            self.canvas = np.zeros_like(target)
            self.copy_count = 0
            self.return_region_flags = []
            self.closed = False

        def sync_full_canvas(self, canvas):
            self.canvas[:] = canvas

        def copy_canvas_to(self, canvas):
            self.copy_count += 1
            canvas[:] = self.canvas

        def copy_canvas_region_to(self, canvas, bbox):
            self.copy_count += 1
            x0, y0, x1, y1 = bbox
            if x1 > x0 and y1 > y0:
                canvas[y0:y1, x0:x1] = self.canvas[y0:y1, x0:x1]

        def apply_rotated_ellipse(self, shape, return_region=True):
            self.return_region_flags.append(return_region)
            x0, y0, x1, y1 = shape.bbox(self.canvas.shape[1], self.canvas.shape[0])
            if x1 > x0 and y1 > y0:
                self.canvas[y0:y1, x0:x1] = np.array(shape.color[:3], dtype=np.uint8)
            return (x0, y0, x1, y1), np.zeros((0, 0, 3), dtype=np.uint8)

        def sync_region(self, canvas, bbox):
            x0, y0, x1, y1 = bbox
            if x1 > x0 and y1 > y0:
                self.canvas[y0:y1, x0:x1] = canvas[y0:y1, x0:x1]

        def can_use_rotated_ellipse_cupy(self):
            return False

        def close(self):
            self.closed = True
            self.canvas[:] = 0
            return None

    monkeypatch.setattr(
        engine_module,
        "resolve_compute_backend",
        lambda requested: ComputeBackendInfo(requested=str(requested), resolved="gpu", label="GPU (Test)"),
    )
    monkeypatch.setattr(engine_module, "TorchSearchRuntime", FakeRuntime)

    target = _make_target(16)
    profile = Profile(
        name="gpu-test",
        stop_at=1,
        random_samples=4,
        mutated_samples=2,
        preview_every=1,
        save_at=[],
        save_every=0,
        max_threads=1,
        compute_backend="gpu",
        shape_types=["rotated_ellipse"],
        edge_weight_strength=0.0,
        gradient_weight=0.0,
    )
    engine = Engine(target, EngineConfig(profile=profile, seed=1))
    shape = RotatedEllipse(color=(7, 8, 9, 255), x=8.0, y=8.0, rx=3.0, ry=2.0, angle=0.0)
    engine._parallel_search = lambda types, n_random, n_mutate: (0.0, shape)

    runtime = engine._gpu_runtime
    assert runtime is not None

    events = iter(engine.run())
    first = next(events)
    assert first.kind == "shape_committed"
    assert engine._gpu_canvas_dirty is True
    assert runtime.copy_count == 0
    assert runtime.return_region_flags == [False]

    second = next(events)
    assert second.kind == "preview"
    assert second.canvas is not None
    assert engine._gpu_canvas_dirty is False
    assert runtime.copy_count == 1
    assert np.array_equal(second.canvas[:, :, :3], engine.canvas)

    third = next(events)
    assert third.kind == "done"
    assert third.canvas is not None
    assert runtime.copy_count == 1
    done_canvas = third.canvas.copy()
    with pytest.raises(StopIteration):
        next(events)
    assert runtime.closed is True
    assert np.array_equal(third.canvas, done_canvas)


def test_torch_runtime_close_releases_gpu_memory_pools():
    class FakeCuda:
        def __init__(self):
            self.synchronized = False
            self.emptied = False
            self.ipc_collected = False

        def is_available(self):
            return True

        def synchronize(self, device=None):
            self.synchronized = True

        def empty_cache(self):
            self.emptied = True

        def ipc_collect(self):
            self.ipc_collected = True

    class FakeTorch:
        def __init__(self):
            self.cuda = FakeCuda()

    class FakePool:
        def __init__(self):
            self.freed = False

        def free_all_blocks(self):
            self.freed = True

    class FakeDevice:
        def __init__(self, cupy):
            self._cupy = cupy

        def synchronize(self):
            self._cupy.synchronized = True

    class FakeCupyCuda:
        def __init__(self, cupy):
            self._cupy = cupy

        def Device(self):
            return FakeDevice(self._cupy)

    class FakeCupy:
        def __init__(self):
            self.synchronized = False
            self.pool = FakePool()
            self.pinned_pool = FakePool()
            self.cuda = FakeCupyCuda(self)

        def get_default_memory_pool(self):
            return self.pool

        def get_default_pinned_memory_pool(self):
            return self.pinned_pool

    runtime = object.__new__(TorchSearchRuntime)
    runtime._torch = FakeTorch()
    runtime._cupy = FakeCupy()
    runtime.device = object()
    runtime._long_cache = {1: object()}
    runtime._float_cache = {(1, 1): object()}
    for name in (
        "_rotated_graph_default",
        "_rotated_graph_medium",
        "_cupy_score_fixed_kernel",
        "_cupy_apply_fixed_kernel",
        "_cupy_target_minus_half_canvas_flat",
        "_cupy_target_minus_half_canvas_sqsum_flat",
        "_cupy_canvas_minus_target_sqsum_flat",
        "_cupy_canvas_full_sq_scalar",
        "_cupy_canvas_flat",
        "_cupy_target_flat",
        "_cupy_canvas_minus_target_flat",
        "target",
        "target_flat",
        "canvas",
        "canvas_flat",
        "_target_minus_half_canvas",
        "_target_minus_half_canvas_flat",
        "_target_minus_half_canvas_sqsum",
        "_target_minus_half_canvas_sqsum_flat",
        "_canvas_minus_target",
        "_canvas_minus_target_flat",
        "_canvas_minus_target_sqsum",
        "_canvas_minus_target_sqsum_flat",
        "_quality_edge_weight",
        "_quality_edge_weight_flat",
        "_quality_target_gx",
        "_quality_target_gy",
        "_quality_target_gx_flat",
        "_quality_target_gy_flat",
        "_quality_weighted_full_sq_scalar",
        "_quality_gradient_full_error_scalar",
        "alpha",
        "alpha_scale",
        "alpha_nonzero",
        "alpha_inside",
        "alpha_scale_flat",
        "alpha_nonzero_flat",
        "alpha_inside_flat",
        "_long_base",
        "_float_base",
        "_canvas_full_sq_scalar",
        "_half_alpha_scale",
        "_one_minus_half_alpha",
        "_neighbor_offsets_x",
        "_neighbor_offsets_y",
        "_neighbor_offsets_rx",
        "_neighbor_offsets_ry",
        "_neighbor_offsets_angle",
        "_neighbor_keep_no_diag",
        "_neighbor_keep_core",
    ):
        setattr(runtime, name, object())

    runtime.close()

    assert runtime._long_cache == {}
    assert runtime._float_cache == {}
    assert runtime.canvas is None
    assert runtime.target is None
    assert runtime._torch.cuda.synchronized is True
    assert runtime._torch.cuda.emptied is True
    assert runtime._torch.cuda.ipc_collected is True
    assert runtime._cupy.synchronized is True
    assert runtime._cupy.pool.freed is True
    assert runtime._cupy.pinned_pool.freed is True


def test_gpu_search_scores_all_shape_types(monkeypatch):
    class FakeRuntime:
        def __init__(self, target, alpha_mask):
            self.scored_batches = []

        def sync_full_canvas(self, canvas):
            return None

        def sync_region(self, canvas, bbox):
            return None

        def score_shapes(self, shapes, canvas_full_sq, canvas_norm):
            self.scored_batches.append([shape.type_name for shape in shapes])
            scores = np.linspace(1.0, float(len(shapes)), len(shapes), dtype=np.float32)
            colors = np.tile(np.array([[1, 2, 3, 128]], dtype=np.int32), (len(shapes), 1))
            return scores, colors

        def can_use_rotated_ellipse_cupy(self):
            return False

        def close(self):
            return None

    monkeypatch.setattr(
        engine_module,
        "resolve_compute_backend",
        lambda requested: ComputeBackendInfo(requested=str(requested), resolved="gpu", label="GPU (Test)"),
    )
    monkeypatch.setattr(engine_module, "TorchSearchRuntime", FakeRuntime)

    ordered_types = ["ellipse", "rotated_ellipse", "circle", "rectangle", "rotated_rectangle", "triangle"]
    emitted = iter(ordered_types)

    def fake_random_shape(rng, w, h, allowed_types):
        type_name = next(emitted)
        assert type_name in allowed_types
        return SHAPE_REGISTRY[type_name].random(rng, w, h)

    monkeypatch.setattr(engine_module, "random_shape", fake_random_shape)

    target = _make_target(32)
    profile = Profile(
        name="gpu-all-types",
        stop_at=1,
        random_samples=len(ordered_types),
        mutated_samples=1,
        preview_every=0,
        save_at=[],
        save_every=0,
        max_threads=1,
        compute_backend="gpu",
        shape_types=ordered_types,
    )
    engine = Engine(target, EngineConfig(profile=profile, seed=1))
    try:
        score, shape = engine._gpu_search(profile.shape_types, profile.random_samples, profile.mutated_samples)
        runtime = engine._gpu_runtime
        assert runtime is not None
        assert np.isfinite(score)
        assert shape is not None
        assert runtime.scored_batches
        assert set(runtime.scored_batches[0]) == set(ordered_types)
    finally:
        engine._shutdown()


def test_gpu_search_batches_generic_mutations(monkeypatch):
    class FakeRuntime:
        def __init__(self, target, alpha_mask):
            self.batch_sizes = []

        def sync_full_canvas(self, canvas):
            return None

        def sync_region(self, canvas, bbox):
            return None

        def score_shapes(self, shapes, canvas_full_sq, canvas_norm):
            self.batch_sizes.append(len(shapes))
            scores = np.linspace(1.0, float(len(shapes)), len(shapes), dtype=np.float32)
            colors = np.tile(np.array([[1, 2, 3, 128]], dtype=np.int32), (len(shapes), 1))
            return scores, colors

        def can_use_rotated_ellipse_cupy(self):
            return False

        def close(self):
            return None

    monkeypatch.setattr(
        engine_module,
        "resolve_compute_backend",
        lambda requested: ComputeBackendInfo(requested=str(requested), resolved="gpu", label="GPU (Test)"),
    )
    monkeypatch.setattr(engine_module, "TorchSearchRuntime", FakeRuntime)

    target = _make_target(32)
    profile = Profile(
        name="gpu-grouped-mutate",
        stop_at=1,
        random_samples=2,
        mutated_samples=7,
        preview_every=0,
        save_at=[],
        save_every=0,
        max_threads=1,
        compute_backend="gpu",
        shape_types=["circle", "rectangle"],
    )
    engine = Engine(target, EngineConfig(profile=profile, seed=1))
    engine._gpu_mutation_group_size = lambda stall_limit: 3
    try:
        score, shape = engine._gpu_search(profile.shape_types, profile.random_samples, profile.mutated_samples)
        runtime = engine._gpu_runtime
        assert runtime is not None
        assert np.isfinite(score)
        assert shape is not None
        assert runtime.batch_sizes == [2, 3, 3, 1]
    finally:
        engine._shutdown()


def test_gpu_pick_group_winners_batches_equal_groups():
    torch = pytest.importorskip("torch")

    class FakeRuntime:
        _torch = torch
        device = torch.device("cpu")

        def close(self):
            return None

    target = _make_target(16)
    profile = Profile(
        name="gpu-group-winners",
        stop_at=1,
        random_samples=1,
        mutated_samples=1,
        preview_every=0,
        save_at=[],
        save_every=0,
        max_threads=1,
        compute_backend="cpu",
        shape_types=["circle"],
        edge_weight_strength=0.0,
        gradient_weight=0.0,
    )
    engine = Engine(target, EngineConfig(profile=profile, seed=1))
    engine._gpu_runtime = FakeRuntime()
    try:
        scores = torch.tensor([4.0, 1.0, 3.0, 9.0, 2.0, 5.0], dtype=torch.float32)
        colors = torch.tensor(
            [
                [4, 4, 4, 128],
                [1, 1, 1, 128],
                [3, 3, 3, 128],
                [9, 9, 9, 128],
                [2, 2, 2, 128],
                [5, 5, 5, 128],
            ],
            dtype=torch.int32,
        )
        winners, best_scores, best_colors = engine._gpu_pick_group_winners(scores, colors, [3, 3])
    finally:
        engine._shutdown()

    winners_seq = winners.cpu().tolist() if hasattr(winners, "cpu") else list(winners)
    assert winners_seq == [1, 4]
    best_scores_np = best_scores.cpu().numpy() if hasattr(best_scores, "cpu") else np.asarray(best_scores)
    best_colors_np = best_colors.cpu().numpy() if hasattr(best_colors, "cpu") else np.asarray(best_colors)
    assert np.allclose(best_scores_np, np.array([1.0, 2.0], dtype=np.float32))
    assert np.array_equal(
        best_colors_np,
        np.array([[1, 1, 1, 128], [2, 2, 2, 128]], dtype=np.int32),
    )


def test_gpu_pick_group_winners_batches_mixed_groups():
    torch = pytest.importorskip("torch")

    class FakeRuntime:
        _torch = torch
        device = torch.device("cpu")

        def close(self):
            return None

    target = _make_target(16)
    profile = Profile(
        name="gpu-group-winners-mixed",
        stop_at=1,
        random_samples=1,
        mutated_samples=1,
        preview_every=0,
        save_at=[],
        save_every=0,
        max_threads=1,
        compute_backend="cpu",
        shape_types=["circle"],
        edge_weight_strength=0.0,
        gradient_weight=0.0,
    )
    engine = Engine(target, EngineConfig(profile=profile, seed=1))
    engine._gpu_runtime = FakeRuntime()
    try:
        scores = torch.tensor([4.0, 1.0, 9.0, 2.0, 5.0, 7.0], dtype=torch.float32)
        colors = torch.tensor(
            [
                [4, 4, 4, 128],
                [1, 1, 1, 128],
                [9, 9, 9, 128],
                [2, 2, 2, 128],
                [5, 5, 5, 128],
                [7, 7, 7, 128],
            ],
            dtype=torch.int32,
        )
        winners, best_scores, best_colors = engine._gpu_pick_group_winners(scores, colors, [2, 3, 1])
    finally:
        engine._shutdown()

    winners_seq = winners.cpu().tolist() if hasattr(winners, "cpu") else list(winners)
    assert winners_seq == [1, 3, 5]
    best_scores_np = best_scores.cpu().numpy() if hasattr(best_scores, "cpu") else np.asarray(best_scores)
    best_colors_np = best_colors.cpu().numpy() if hasattr(best_colors, "cpu") else np.asarray(best_colors)
    assert np.allclose(best_scores_np, np.array([1.0, 2.0, 7.0], dtype=np.float32))
    assert np.array_equal(
        best_colors_np,
        np.array([[1, 1, 1, 128], [2, 2, 2, 128], [7, 7, 7, 128]], dtype=np.int32),
    )


def test_gpu_apply_scores_counts_group_misses():
    torch = pytest.importorskip("torch")
    target = _make_target(16)
    profile = Profile(
        name="gpu-apply-scores",
        stop_at=1,
        random_samples=1,
        mutated_samples=1,
        preview_every=0,
        save_at=[],
        save_every=0,
        max_threads=1,
        compute_backend="cpu",
        shape_types=["circle"],
    )
    engine = Engine(target, EngineConfig(profile=profile, seed=1))
    chains = [engine_module._GpuChainState(rng=random.Random(0), best_score=1.0, best_shape=None, no_improve=0)]
    candidate = Circle(color=(0, 0, 0, 128), x=8.0, y=8.0, r=3.0)
    try:
        engine._gpu_apply_scores(
            chains,
            [0],
            [candidate],
            torch.tensor([2.0], dtype=torch.float32),
            torch.tensor([[1, 2, 3, 128]], dtype=torch.int32),
            attempt_counts=[5],
        )
    finally:
        engine._shutdown()

    assert chains[0].no_improve == 5
    assert chains[0].best_shape is None


def test_find_best_candidate_uses_gpu_best_only_path(monkeypatch):
    torch = pytest.importorskip("torch")

    class FakeRuntime:
        _torch = torch
        device = torch.device("cpu")

        def score_shapes_torch(self, shapes, canvas_full_sq, canvas_norm):
            scores = torch.tensor([5.0, 1.5, 3.0], dtype=torch.float32)
            colors = torch.tensor(
                [
                    [5, 5, 5, 128],
                    [7, 8, 9, 128],
                    [3, 3, 3, 128],
                ],
                dtype=torch.int32,
            )
            return scores, colors

        def close(self):
            return None

    target = _make_target(16)
    profile = Profile(
        name="gpu-best-candidate",
        stop_at=1,
        random_samples=1,
        mutated_samples=1,
        preview_every=0,
        save_at=[],
        save_every=0,
        max_threads=1,
        compute_backend="cpu",
        shape_types=["circle"],
        edge_weight_strength=0.0,
        gradient_weight=0.0,
    )
    engine = Engine(target, EngineConfig(profile=profile, seed=1))
    engine._gpu_runtime = FakeRuntime()
    monkeypatch.setattr(engine, "_score_candidate_shapes", lambda candidates: (_ for _ in ()).throw(AssertionError("fallback should not be used")))
    candidates = [
        Circle(color=(0, 0, 0, 128), x=4.0, y=4.0, r=2.0),
        Circle(color=(0, 0, 0, 128), x=8.0, y=8.0, r=3.0),
        Circle(color=(0, 0, 0, 128), x=12.0, y=12.0, r=2.5),
    ]
    try:
        best_score, best_shape = engine._find_best_candidate(candidates)
    finally:
        engine._shutdown()

    assert best_shape is candidates[1]
    assert best_score == pytest.approx(1.5)
    assert best_shape.color == (7, 8, 9, 128)


def test_gpu_quality_reranks_top_k_per_group(monkeypatch):
    class FakeRuntime:
        def __init__(self):
            self.calls = 0

        def score_shapes(self, shapes, canvas_full_sq, canvas_norm):
            self.calls += 1
            scores = np.array([5, 4, 3, 2, 1, 1, 2, 3, 4, 5], dtype=np.float32)
            colors = np.tile(np.array([[1, 2, 3, 128]], dtype=np.int32), (len(shapes), 1))
            return scores[:len(shapes)], colors

        def close(self):
            return None

    target = _make_target(16)
    profile = Profile(
        name="gpu-quality-rerank",
        stop_at=1,
        random_samples=1,
        mutated_samples=1,
        preview_every=0,
        save_at=[],
        save_every=0,
        max_threads=1,
        compute_backend="cpu",
        shape_types=["circle"],
        edge_weight_strength=1.0,
        gradient_weight=0.1,
        edge_rerank_top_k=2,
    )
    engine = Engine(target, EngineConfig(profile=profile, seed=1))
    engine._gpu_runtime = FakeRuntime()
    candidates = [
        Circle(color=(0, 0, 0, 128), x=float(idx + 2), y=8.0, r=2.0)
        for idx in range(10)
    ]
    calls = []

    def fake_score_shape(shape, **kwargs):
        idx = candidates.index(shape)
        calls.append(idx)
        return float(idx), (idx, idx, idx, 128)

    monkeypatch.setattr(engine, "_score_shape", fake_score_shape)
    try:
        scores, colors = engine._gpu_score_grouped_shapes(candidates, [5, 5])
    finally:
        engine._shutdown()

    assert calls == [4, 3, 5, 6]
    assert np.isfinite(scores).sum() == 4
    assert np.array_equal(np.flatnonzero(np.isfinite(scores)), np.array([3, 4, 5, 6]))
    assert colors[4].tolist() == [4, 4, 4, 128]


def test_gpu_quality_uses_runtime_quality_scoring(monkeypatch):
    class FakeRuntime:
        def __init__(self):
            self.quality_shapes = []
            self.quality_colors = None

        def score_shapes(self, shapes, canvas_full_sq, canvas_norm):
            scores = np.array([5, 4, 3, 2, 1, 1, 2, 3, 4, 5], dtype=np.float32)
            colors = np.tile(np.array([[1, 2, 3, 128]], dtype=np.int32), (len(shapes), 1))
            return scores[:len(shapes)], colors

        def score_shapes_quality(
            self,
            shapes,
            base_colors,
            weighted_full_sq,
            weighted_norm,
            gradient_full_error,
            gradient_norm,
        ):
            self.quality_shapes = list(shapes)
            self.quality_colors = np.asarray(base_colors, dtype=np.int32)
            scores = np.asarray([float(idx) for idx, _shape in enumerate(shapes)], dtype=np.float32)
            colors = np.tile(np.array([[9, 8, 7, 128]], dtype=np.int32), (len(shapes), 1))
            return scores, colors

        def close(self):
            return None

    target = _make_target(16)
    profile = Profile(
        name="gpu-quality-runtime",
        stop_at=1,
        random_samples=1,
        mutated_samples=1,
        preview_every=0,
        save_at=[],
        save_every=0,
        max_threads=1,
        compute_backend="cpu",
        shape_types=["circle"],
        edge_weight_strength=1.0,
        gradient_weight=0.1,
        edge_rerank_top_k=2,
    )
    engine = Engine(target, EngineConfig(profile=profile, seed=1))
    runtime = FakeRuntime()
    engine._gpu_runtime = runtime
    candidates = [
        Circle(color=(0, 0, 0, 128), x=float(idx + 2), y=8.0, r=2.0)
        for idx in range(10)
    ]
    monkeypatch.setattr(engine, "_score_shape", lambda _shape, **_kwargs: (_ for _ in ()).throw(AssertionError("CPU quality fallback should not be used")))
    try:
        scores, colors = engine._gpu_score_grouped_shapes(candidates, [5, 5])
    finally:
        engine._shutdown()

    assert runtime.quality_shapes == [candidates[4], candidates[3], candidates[5], candidates[6]]
    assert runtime.quality_colors.shape == (4, 4)
    assert np.array_equal(np.flatnonzero(np.isfinite(scores)), np.array([3, 4, 5, 6]))
    assert colors[4].tolist() == [9, 8, 7, 128]


def test_gpu_quality_uses_runtime_quality_scoring_with_tensor_base_scores(monkeypatch):
    torch = pytest.importorskip("torch")

    class FakeRuntime:
        _torch = torch
        device = torch.device("cpu")

        def __init__(self):
            self.quality_shapes = []
            self.quality_colors = None

        def score_shapes_torch(self, shapes, canvas_full_sq, canvas_norm):
            scores = torch.tensor([5, 4, 3, 2, 1, 1, 2, 3, 4, 5], dtype=torch.float32)
            colors = torch.tensor([[1, 2, 3, 128]] * len(shapes), dtype=torch.int32)
            return scores[:len(shapes)], colors

        def score_shapes_quality(
            self,
            shapes,
            base_colors,
            weighted_full_sq,
            weighted_norm,
            gradient_full_error,
            gradient_norm,
        ):
            assert isinstance(base_colors, np.ndarray)
            self.quality_shapes = list(shapes)
            self.quality_colors = base_colors.copy()
            scores = np.arange(len(shapes), dtype=np.float32)
            colors = np.tile(np.array([[9, 8, 7, 128]], dtype=np.int32), (len(shapes), 1))
            return scores, colors

        def close(self):
            return None

    target = _make_target(16)
    profile = Profile(
        name="gpu-quality-torch-runtime",
        stop_at=1,
        random_samples=1,
        mutated_samples=1,
        preview_every=0,
        save_at=[],
        save_every=0,
        max_threads=1,
        compute_backend="cpu",
        shape_types=["circle"],
        edge_weight_strength=1.0,
        gradient_weight=0.1,
        edge_rerank_top_k=2,
    )
    engine = Engine(target, EngineConfig(profile=profile, seed=1))
    runtime = FakeRuntime()
    engine._gpu_runtime = runtime
    candidates = [
        Circle(color=(0, 0, 0, 128), x=float(idx + 2), y=8.0, r=2.0)
        for idx in range(10)
    ]
    monkeypatch.setattr(engine, "_score_shape", lambda _shape, **_kwargs: (_ for _ in ()).throw(AssertionError("CPU quality fallback should not be used")))
    try:
        scores, colors = engine._gpu_score_grouped_shapes(candidates, [5, 5])
    finally:
        engine._shutdown()

    assert runtime.quality_shapes == [candidates[4], candidates[3], candidates[5], candidates[6]]
    assert runtime.quality_colors.shape == (4, 4)
    assert np.array_equal(np.flatnonzero(np.isfinite(scores)), np.array([3, 4, 5, 6]))
    assert colors[4].tolist() == [9, 8, 7, 128]


def test_gpu_runtime_receives_quality_batch_pixel_limit(monkeypatch):
    class FakeRuntime:
        def __init__(self, target, alpha_mask):
            self.quality_batch_limit = None
            self.quality_context = None

        def set_quality_batch_pixel_limit(self, value):
            self.quality_batch_limit = value

        def set_quality_context(self, edge_weight, target_gx, target_gy, gradient_weight):
            self.quality_context = (edge_weight, target_gx, target_gy, gradient_weight)

        def sync_full_canvas(self, canvas):
            return None

        def close(self):
            return None

    monkeypatch.setattr(
        engine_module,
        "resolve_compute_backend",
        lambda requested: ComputeBackendInfo(requested=str(requested), resolved="gpu", label="GPU (Test)"),
    )
    monkeypatch.setattr(engine_module, "TorchSearchRuntime", FakeRuntime)
    target = _make_target(16)
    profile = Profile(
        name="gpu-quality-batch-limit",
        stop_at=1,
        random_samples=4,
        mutated_samples=1,
        preview_every=0,
        save_at=[],
        save_every=0,
        max_threads=1,
        compute_backend="gpu",
        shape_types=["circle"],
        edge_weight_strength=1.0,
        gradient_weight=0.1,
        quality_batch_pixels=128000,
    )
    engine = Engine(target, EngineConfig(profile=profile, seed=1))
    try:
        runtime = engine._gpu_runtime
        assert runtime is not None
        assert runtime.quality_batch_limit == 128000
        assert runtime.quality_context is not None
    finally:
        engine._shutdown()


def test_cpu_quality_random_search_reranks_top_k(monkeypatch):
    target = _make_target(16)
    canvas = np.full_like(target, 127)
    profile = Profile(
        name="cpu-quality-rerank",
        stop_at=1,
        random_samples=10,
        mutated_samples=1,
        preview_every=0,
        save_at=[],
        save_every=0,
        max_threads=1,
        compute_backend="cpu",
        shape_types=["circle"],
        edge_weight_strength=1.0,
        gradient_weight=0.1,
        edge_candidate_ratio=0.0,
        edge_rerank_top_k=3,
    )
    engine = Engine(target, EngineConfig(profile=profile, seed=1))
    quality = engine.quality_context
    assert quality is not None
    calls = {"plain": 0, "quality": 0}
    real_score_shape = engine_module.score_shape

    def counted_score_shape(*args, **kwargs):
        if kwargs.get("quality_context") is None:
            calls["plain"] += 1
        else:
            calls["quality"] += 1
        return real_score_shape(*args, **kwargs)

    monkeypatch.setattr(engine_module, "score_shape", counted_score_shape)
    try:
        score, color, shape = engine_module._independent_search(
            canvas,
            target,
            None,
            quality,
            (profile.shape_types, profile.random_samples, 0, 16, 16, 99, 0.0, profile.edge_rerank_top_k),
        )
    finally:
        engine._shutdown()

    assert np.isfinite(score)
    assert color is not None
    assert shape is not None
    assert calls["plain"] == profile.random_samples
    assert calls["quality"] == profile.edge_rerank_top_k + 1


def test_refine_existing_shape_uses_gpu_batch_cap(monkeypatch):
    class FakeRuntime:
        def close(self):
            return None

    target = _make_target(16)
    profile = Profile(
        name="gpu-refine-batch",
        stop_at=1,
        random_samples=1,
        mutated_samples=96,
        preview_every=0,
        save_at=[],
        save_every=0,
        max_threads=1,
        compute_backend="cpu",
        shape_types=["circle"],
    )
    engine = Engine(target, EngineConfig(profile=profile, seed=1))
    engine._gpu_runtime = FakeRuntime()
    base = Circle(color=(0, 0, 0, 128), x=8.0, y=8.0, r=3.0)
    seen_counts = []

    def fake_find_best(candidates):
        seen_counts.append(len(candidates))
        shape = candidates[0]
        return 1.0, shape

    monkeypatch.setattr(engine, "_find_best_candidate", fake_find_best)
    try:
        engine._refine_existing_shape(base, random.Random(0), 16)
    finally:
        engine._shutdown()

    assert seen_counts == [17]


def test_shape_mutation_drops_bbox_cache_but_with_color_keeps_it():
    shape = Circle(color=(0, 0, 0, 128), x=8.0, y=8.0, r=3.0)
    shape._fd6_bbox_cache = (32, 32, (5, 5, 12, 12), 49)
    shape._fd6_circle_geom_cache = (3.0, 9.0)

    recolored = shape.with_color((1, 2, 3, 128))
    mutated = shape.mutate(random.Random(0), 32, 32)

    assert getattr(recolored, "_fd6_bbox_cache", None) == (32, 32, (5, 5, 12, 12), 49)
    assert getattr(recolored, "_fd6_circle_geom_cache", None) == (3.0, 9.0)
    assert not hasattr(mutated, "_fd6_bbox_cache")
    assert not hasattr(mutated, "_fd6_circle_geom_cache")


def test_cache_shape_bbox_extends_legacy_cache():
    target = _make_target(16)
    profile = Profile(
        name="bbox-cache-upgrade",
        stop_at=1,
        random_samples=1,
        mutated_samples=1,
        preview_every=0,
        save_at=[],
        save_every=0,
        max_threads=1,
        compute_backend="cpu",
        shape_types=["circle"],
    )
    engine = Engine(target, EngineConfig(profile=profile, seed=1))
    shape = Circle(color=(0, 0, 0, 128), x=8.0, y=8.0, r=3.0)
    shape._fd6_bbox_cache = (16, 16, (5, 5, 12, 12), 49)
    try:
        engine._cache_shape_bbox(shape)
    finally:
        engine._shutdown()

    assert getattr(shape, "_fd6_bbox_cache", None) == (16, 16, (5, 5, 12, 12), 49, 7, 7)


def test_gpu_rotated_ellipse_search_passes_refine_stage_count():
    class FakeRuntime:
        def __init__(self):
            self.calls = []

        def search_rotated_ellipse_device(self, chain_count, random_count, canvas_full_sq, canvas_norm, seed, refine_stage_count=5):
            self.calls.append((chain_count, random_count, refine_stage_count))
            return 1.0, np.array([1.0, 8.0, 8.0, 3.0, 4.0, 15.0, 1.0, 2.0, 3.0, 128.0], dtype=np.float32)

        def close(self):
            return None

    target = _make_target(32)
    profile = Profile(
        name="gpu-rot-ellipse-refine-count",
        stop_at=1,
        random_samples=1000,
        mutated_samples=80,
        preview_every=0,
        save_at=[],
        save_every=0,
        max_threads=2,
        compute_backend="cpu",
        shape_types=["rotated_ellipse"],
    )
    engine = Engine(target, EngineConfig(profile=profile, seed=1))
    runtime = FakeRuntime()
    engine._gpu_runtime = runtime
    engine._n_workers = 2
    try:
        score, shape = engine._gpu_rotated_ellipse_search(1000, 80)
    finally:
        engine._shutdown()

    assert score == pytest.approx(1.0)
    assert isinstance(shape, engine_module._GpuRotatedEllipsePack)
    assert runtime.calls == [(2, 192, 2)]


def test_torch_runtime_score_shapes_splits_fixed_half_alpha_group(monkeypatch):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    target = _make_target(32)
    runtime = TorchSearchRuntime(target)
    runtime.max_batch_pixels = 1_000_000
    shapes = [
        Circle(color=(10, 10, 10, 128), x=8.0, y=8.0, r=4.0),
        Circle(color=(20, 20, 20, 255), x=20.0, y=20.0, r=5.0),
    ]
    seen_fixed_flags = []

    def fake_score_batch(type_name, packed, start, end, canvas_norm):
        seen_fixed_flags.append(bool(packed.get("fixed_half_alpha", False)))
        count = end - start
        scores = runtime._torch.full((count,), 1.0, device=runtime.device, dtype=runtime._torch.float32)
        colors = runtime._torch.zeros((count, 4), device=runtime.device, dtype=runtime._torch.int32)
        return scores, colors

    monkeypatch.setattr(runtime, "_score_batch", fake_score_batch)
    try:
        runtime.score_shapes(shapes, 0.0, 1.0)
    finally:
        runtime.close()

    assert seen_fixed_flags == [True, False]


def test_torch_runtime_score_shapes_splits_fixed_half_alpha_rotated_ellipse(monkeypatch):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    target = _make_target(32)
    runtime = TorchSearchRuntime(target)
    shapes = [
        RotatedEllipse(color=(10, 10, 10, 128), x=10.0, y=10.0, rx=4.0, ry=6.0, angle=15.0),
        RotatedEllipse(color=(20, 20, 20, 255), x=22.0, y=22.0, rx=5.0, ry=3.0, angle=35.0),
    ]
    seen_groups = []

    def fake_score_group(items, scores_out, colors_out, canvas_norm):
        seen_groups.append([item[1].color[3] for item in items])

    monkeypatch.setattr(runtime, "_score_rotated_ellipse_group", fake_score_group)
    try:
        runtime.score_shapes(shapes, 0.0, 1.0)
    finally:
        runtime.close()

    assert seen_groups == [[128], [255]]


def test_torch_runtime_rotated_exact_uses_rotated_extents(monkeypatch):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    target = _make_target(32)
    runtime = TorchSearchRuntime(target)
    captured = {}

    def fake_exact_batch(bx0, by0, bw, bh, max_w, max_h, bx, by, brx, bry, ba, canvas_norm, sample_stride):
        captured["width"] = bw.to(device="cpu", dtype=torch.int32).tolist()
        captured["height"] = bh.to(device="cpu", dtype=torch.int32).tolist()
        count = int(bx.shape[0])
        zeros = torch.zeros((count,), device=runtime.device, dtype=torch.float32)
        colors = torch.zeros((count, 4), device=runtime.device, dtype=torch.int32)
        return zeros, zeros, zeros, zeros, zeros, zeros, colors

    monkeypatch.setattr(runtime, "_ensure_rotated_ellipse_cupy", lambda: False)
    monkeypatch.setattr(runtime, "_score_rotated_ellipse_candidate_grid_exact_batch", fake_exact_batch)
    try:
        x_candidates = torch.tensor([[16.0]], device=runtime.device, dtype=torch.float32)
        y_candidates = torch.tensor([[16.0]], device=runtime.device, dtype=torch.float32)
        rx_candidates = torch.tensor([[8.0]], device=runtime.device, dtype=torch.float32)
        ry_candidates = torch.tensor([[2.0]], device=runtime.device, dtype=torch.float32)
        angle_candidates = torch.tensor([[45.0]], device=runtime.device, dtype=torch.float32)
        runtime._score_rotated_ellipse_candidate_grid_exact(
            x_candidates,
            y_candidates,
            rx_candidates,
            ry_candidates,
            angle_candidates,
            0.0,
            1.0,
            sample_stride=1,
        )
    finally:
        runtime.close()

    ext_x = math.sqrt((8.0 * math.cos(math.pi / 4.0)) ** 2 + (2.0 * math.sin(math.pi / 4.0)) ** 2)
    ext_y = math.sqrt((8.0 * math.sin(math.pi / 4.0)) ** 2 + (2.0 * math.cos(math.pi / 4.0)) ** 2)
    expected_width = int(math.ceil(16.0 + ext_x + 1.0)) - int(math.floor(16.0 - ext_x))
    expected_height = int(math.ceil(16.0 + ext_y + 1.0)) - int(math.floor(16.0 - ext_y))
    assert captured["width"] == [expected_width]
    assert captured["height"] == [expected_height]


def test_torch_runtime_score_shapes_matches_cpu_for_mixed_types():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    target = _make_target(48)
    canvas = np.zeros_like(target)
    shapes = [
        Circle(color=(10, 20, 30, 128), x=12.0, y=12.0, r=5.0),
        Ellipse(color=(30, 40, 50, 128), x=32.0, y=14.0, rx=7.0, ry=4.0),
        RotatedEllipse(color=(50, 60, 70, 128), x=18.0, y=32.0, rx=6.0, ry=9.0, angle=25.0),
        Rectangle(color=(70, 80, 90, 128), x=36.0, y=34.0, hw=6.0, hh=5.0),
        RotatedRectangle(color=(90, 100, 110, 128), x=14.0, y=36.0, hw=5.0, hh=8.0, angle=40.0),
        Triangle(color=(110, 120, 130, 128), x1=26.0, y1=24.0, x2=42.0, y2=26.0, x3=34.0, y3=42.0),
    ]
    canvas_full_sq, canvas_norm = precompute_canvas_error(canvas, target, None)
    runtime = TorchSearchRuntime(target)
    try:
        runtime.sync_full_canvas(canvas)
        gpu_scores, gpu_colors = runtime.score_shapes(shapes, canvas_full_sq, canvas_norm)
    finally:
        runtime.close()

    cpu_scores = []
    cpu_colors = []
    for shape in shapes:
        score, color = score_shape(
            shape,
            canvas,
            target,
            canvas_full_sq=canvas_full_sq,
            canvas_norm=canvas_norm,
        )
        cpu_scores.append(score)
        cpu_colors.append(color)

    assert np.allclose(gpu_scores, np.asarray(cpu_scores, dtype=np.float32), atol=1e-4, rtol=1e-4)
    assert np.array_equal(gpu_colors[:, 3], np.asarray([color[3] for color in cpu_colors], dtype=np.int32))
    assert np.all(np.abs(gpu_colors[:, :3] - np.asarray([color[:3] for color in cpu_colors], dtype=np.int32)) <= 1)


def test_torch_runtime_quality_scores_match_cpu_for_mixed_types():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    target = _make_target(32)
    canvas = np.full_like(target, 180)
    quality = build_quality_context(
        target,
        None,
        edge_weight_strength=0.75,
        gradient_weight=0.12,
        edge_alpha=224,
    )
    assert quality is not None
    canvas_full_sq, canvas_norm = precompute_canvas_error(canvas, target, None)
    weighted_full_sq, weighted_norm = precompute_weighted_rgb_error(canvas, target, quality)
    gradient_full_error, gradient_norm = precompute_gradient_error(canvas, quality)
    shapes = [
        Circle(color=(0, 0, 0, 128), x=8.0, y=8.0, r=4.0),
        Ellipse(color=(0, 0, 0, 128), x=20.0, y=11.0, rx=5.0, ry=3.0),
        RotatedEllipse(color=(0, 0, 0, 128), x=18.0, y=22.0, rx=6.0, ry=3.0, angle=25.0),
        Rectangle(color=(0, 0, 0, 128), x=10.0, y=22.0, hw=4.0, hh=3.0),
        RotatedRectangle(color=(0, 0, 0, 128), x=22.0, y=20.0, hw=5.0, hh=2.0, angle=35.0),
        Triangle(color=(0, 0, 0, 128), x1=4.0, y1=4.0, x2=16.0, y2=8.0, x3=7.0, y3=21.0),
    ]
    base_colors = []
    cpu_scores = []
    for shape in shapes:
        base_score, color = score_shape(
            shape,
            canvas,
            target,
            canvas_full_sq=canvas_full_sq,
            canvas_norm=canvas_norm,
        )
        score, _color = score_shape(
            shape,
            canvas,
            target,
            canvas_full_sq=canvas_full_sq,
            canvas_norm=canvas_norm,
            quality_context=quality,
            weighted_full_sq=weighted_full_sq,
            weighted_norm=weighted_norm,
            gradient_full_error=gradient_full_error,
            gradient_norm=gradient_norm,
            fixed_color=color,
            base_rms=base_score,
            base_canvas_full_sq=getattr(shape, "_fd6_canvas_full_sq", None),
        )
        base_colors.append(color)
        cpu_scores.append(score)
    runtime = TorchSearchRuntime(target)
    try:
        runtime.sync_full_canvas(canvas)
        runtime.set_quality_context(
            quality.edge_weight,
            quality.target_gx,
            quality.target_gy,
            quality.gradient_weight,
        )
        gpu_scores, gpu_colors = runtime.score_shapes_quality(
            shapes,
            np.asarray(base_colors, dtype=np.int32),
            weighted_full_sq,
            weighted_norm,
            gradient_full_error,
            gradient_norm,
        )
    finally:
        runtime.close()

    assert np.allclose(gpu_scores, np.asarray(cpu_scores, dtype=np.float32), atol=0.01, rtol=0.0)
    assert np.array_equal(gpu_colors, np.asarray(base_colors, dtype=np.int32))


def test_torch_runtime_score_shapes_keeps_order_for_single_generic_batch(monkeypatch):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    target = _make_target(32)
    shapes = [
        Ellipse(color=(10, 10, 10, 128), x=6.0, y=6.0, rx=2.0, ry=2.0),
        Ellipse(color=(20, 20, 20, 128), x=16.0, y=16.0, rx=7.0, ry=6.0),
    ]
    runtime = TorchSearchRuntime(target)
    runtime.max_batch_pixels = 1_000_000
    seen_batches: list[list[int]] = []

    def fake_score_batch(type_name, packed, start, end, canvas_norm):
        seen_batches.append(packed["indices"][start:end].detach().cpu().tolist())
        count = end - start
        scores = runtime._torch.full((count,), 1.0, device=runtime.device, dtype=runtime._torch.float32)
        colors = runtime._torch.zeros((count, 4), device=runtime.device, dtype=runtime._torch.int32)
        return scores, colors

    monkeypatch.setattr(runtime, "_score_batch", fake_score_batch)
    try:
        runtime.score_shapes(shapes, 0.0, 1.0)
    finally:
        runtime.close()

    assert seen_batches == [[0, 1]]


def test_gpu_rotated_ellipse_graph_matches_fallback():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    target = _make_target(540)
    profile = Profile(
        name="gpu-graph",
        stop_at=72,
        random_samples=1000,
        mutated_samples=200,
        preview_every=0,
        save_at=[],
        save_every=0,
        max_resolution=600,
        max_threads=0,
        compute_backend="gpu",
        shape_types=["rotated_ellipse"],
    )
    seed = 123456

    engine_graph = Engine(target, EngineConfig(profile=profile, seed=1))
    engine_fallback = Engine(target, EngineConfig(profile=profile, seed=1))
    try:
        runtime_graph = engine_graph._gpu_runtime
        runtime_fallback = engine_fallback._gpu_runtime
        assert runtime_graph is not None
        assert runtime_fallback is not None
        runtime_graph.prefer_rotated_ellipse_cupy = False
        runtime_fallback.prefer_rotated_ellipse_cupy = False
        runtime_fallback.enable_rotated_ellipse_cupy = False
        graph_result = runtime_graph.search_rotated_ellipse(2, 320, engine_graph.canvas_full_sq, engine_graph.canvas_norm, seed)
        runtime_fallback.enable_rotated_graph_default = False
        fallback_result = runtime_fallback.search_rotated_ellipse(2, 320, engine_fallback.canvas_full_sq, engine_fallback.canvas_norm, seed)
        assert np.isfinite(graph_result[0])
        assert graph_result[0] <= fallback_result[0] + 0.01
    finally:
        engine_graph._shutdown()
        engine_fallback._shutdown()


def test_gpu_rotated_ellipse_medium_graph_matches_fallback():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    target = _make_target(256)
    profile = Profile(
        name="gpu-graph-medium",
        stop_at=24,
        random_samples=1000,
        mutated_samples=200,
        preview_every=0,
        save_at=[],
        save_every=0,
        max_resolution=300,
        max_threads=0,
        compute_backend="gpu",
        shape_types=["rotated_ellipse"],
    )
    seed = 654321

    engine_graph = Engine(target, EngineConfig(profile=profile, seed=1))
    engine_fallback = Engine(target, EngineConfig(profile=profile, seed=1))
    try:
        runtime_graph = engine_graph._gpu_runtime
        runtime_fallback = engine_fallback._gpu_runtime
        assert runtime_graph is not None
        assert runtime_fallback is not None
        runtime_graph.prefer_rotated_ellipse_cupy = False
        runtime_fallback.prefer_rotated_ellipse_cupy = False
        runtime_fallback.enable_rotated_ellipse_cupy = False
        graph_result = runtime_graph.search_rotated_ellipse(2, 192, engine_graph.canvas_full_sq, engine_graph.canvas_norm, seed)
        runtime_fallback.enable_rotated_graph_default = False
        runtime_fallback.enable_rotated_graph_medium = False
        fallback_result = runtime_fallback.search_rotated_ellipse(2, 192, engine_fallback.canvas_full_sq, engine_fallback.canvas_norm, seed)
        assert np.isfinite(graph_result[0])
        assert abs(graph_result[0] - fallback_result[0]) <= 1e-4
    finally:
        engine_graph._shutdown()
        engine_fallback._shutdown()


def test_gpu_rotated_ellipse_cupy_matches_torch_fallback():
    torch = pytest.importorskip("torch")
    pytest.importorskip("cupy")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    target = _make_target(384)
    profile = Profile(
        name="gpu-cupy",
        stop_at=24,
        random_samples=1000,
        mutated_samples=200,
        preview_every=0,
        save_at=[],
        save_every=0,
        max_resolution=384,
        max_threads=0,
        compute_backend="gpu",
        shape_types=["rotated_ellipse"],
    )
    seed = 246810

    engine_cupy = Engine(target, EngineConfig(profile=profile, seed=1))
    engine_torch = Engine(target, EngineConfig(profile=profile, seed=1))
    try:
        runtime_cupy = engine_cupy._gpu_runtime
        runtime_torch = engine_torch._gpu_runtime
        assert runtime_cupy is not None
        assert runtime_torch is not None
        runtime_cupy.prefer_rotated_ellipse_cupy = True
        runtime_cupy.enable_rotated_graph_default = False
        runtime_cupy.enable_rotated_graph_medium = False
        runtime_torch.enable_rotated_ellipse_cupy = False
        runtime_torch.prefer_rotated_ellipse_cupy = False
        runtime_torch.enable_rotated_graph_default = False
        runtime_torch.enable_rotated_graph_medium = False
        cupy_result = runtime_cupy.search_rotated_ellipse(2, 256, engine_cupy.canvas_full_sq, engine_cupy.canvas_norm, seed)
        torch_result = runtime_torch.search_rotated_ellipse(2, 256, engine_torch.canvas_full_sq, engine_torch.canvas_norm, seed)
        assert np.isfinite(cupy_result[0])
        assert np.isfinite(torch_result[0])
        assert abs(cupy_result[0] - torch_result[0]) <= 1e-4
        assert cupy_result[2] == torch_result[2]
    finally:
        engine_cupy._shutdown()
        engine_torch._shutdown()


def test_gpu_rotated_ellipse_cupy_apply_pack_matches_torch():
    torch = pytest.importorskip("torch")
    pytest.importorskip("cupy")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    target = _make_target(384)
    profile = Profile(
        name="gpu-cupy-apply",
        stop_at=24,
        random_samples=1000,
        mutated_samples=200,
        preview_every=0,
        save_at=[],
        save_every=0,
        max_resolution=384,
        max_threads=0,
        compute_backend="gpu",
        shape_types=["rotated_ellipse"],
    )
    seed = 97531

    engine_cupy = Engine(target, EngineConfig(profile=profile, seed=1))
    engine_torch = Engine(target, EngineConfig(profile=profile, seed=1))
    try:
        runtime_cupy = engine_cupy._gpu_runtime
        runtime_torch = engine_torch._gpu_runtime
        assert runtime_cupy is not None
        assert runtime_torch is not None
        runtime_cupy.enable_rotated_graph_default = False
        runtime_cupy.enable_rotated_graph_medium = False
        runtime_torch.enable_rotated_graph_default = False
        runtime_torch.enable_rotated_graph_medium = False
        runtime_torch.enable_rotated_ellipse_cupy = False
        best_score, best_pack = runtime_cupy.search_rotated_ellipse_device(
            chain_count=2,
            random_count=256,
            canvas_full_sq=engine_cupy.canvas_full_sq,
            canvas_norm=engine_cupy.canvas_norm,
            seed=seed,
        )
        assert np.isfinite(best_score)
        assert best_pack is not None
        runtime_cupy.apply_rotated_ellipse_pack(best_pack, return_region=False)
        runtime_torch.apply_rotated_ellipse_pack(best_pack, return_region=False)
        torch.cuda.synchronize()
        assert torch.equal(runtime_cupy.canvas, runtime_torch.canvas)
        assert torch.equal(runtime_cupy._canvas_minus_target, runtime_torch._canvas_minus_target)
        assert torch.equal(runtime_cupy._canvas_minus_target_sqsum, runtime_torch._canvas_minus_target_sqsum)
        assert torch.allclose(runtime_cupy._target_minus_half_canvas, runtime_torch._target_minus_half_canvas, atol=1e-5, rtol=0.0)
        assert torch.allclose(
            runtime_cupy._target_minus_half_canvas_sqsum,
            runtime_torch._target_minus_half_canvas_sqsum,
            atol=1e-5,
            rtol=0.0,
        )
    finally:
        engine_cupy._shutdown()
        engine_torch._shutdown()
