import numpy as np
from PIL import Image
import random
import sys
import types

import fd6.shapegen.line_guide as line_guide_module
from fd6.shapegen.engine import Engine, EngineConfig, _edge_candidate_shape
from fd6.shapegen.line_guide import load_line_guide_image, load_line_guide_onnx, normalise_line_strength, resolve_line_guide
from fd6.shapegen.profile import Profile
from fd6.shapegen.quality import build_quality_context


def test_normalise_line_strength_detects_dark_lines():
    image = np.ones((12, 12), dtype=np.float32)
    image[:, 5:7] = 0.0

    guide = normalise_line_strength(image)

    assert guide[:, 5:7].mean() > 0.9
    assert guide[:, :3].mean() < 0.05


def test_load_line_guide_image_resizes_and_keeps_line_strength(tmp_path):
    image = np.full((10, 20), 255, dtype=np.uint8)
    image[:, 8:12] = 0
    path = tmp_path / "guide.png"
    Image.fromarray(image, "L").save(path)

    guide = load_line_guide_image(path, (8, 8), max_resolution=12)

    assert guide.shape == (8, 8)
    assert float(guide[:, 3:5].mean()) > float(guide[:, :2].mean()) + 0.25


def test_load_line_guide_image_can_pad_to_source_geometry(tmp_path):
    image = np.full((4, 8), 255, dtype=np.uint8)
    image[:, :2] = 0
    path = tmp_path / "wide-guide.png"
    Image.fromarray(image, "L").save(path)

    guide = load_line_guide_image(
        path,
        (8, 8),
        source_size=(8, 4),
        pad_to_square=True,
    )

    assert guide.shape == (8, 8)
    assert float(guide[:2, :].mean()) < 0.05
    assert float(guide[2:6, :2].mean()) > 0.8


def test_resolve_line_guide_falls_back_to_external_image(tmp_path):
    src = Image.new("RGB", (12, 12), (128, 128, 128))
    guide_img = np.full((12, 12), 255, dtype=np.uint8)
    guide_img[:, 6] = 0
    guide_path = tmp_path / "line.png"
    Image.fromarray(guide_img, "L").save(guide_path)
    profile = Profile(
        line_guide_enabled=True,
        line_guide_model_path="missing.onnx",
        line_guide_image_path=str(guide_path),
    )

    result = resolve_line_guide(src, (12, 12), profile, tmp_path / "source.png")

    assert result.guide is not None
    assert result.source == "image"
    assert "ONNX model not found" in result.message


def test_resolve_line_guide_falls_back_when_onnx_execution_fails(tmp_path, monkeypatch):
    src = Image.new("RGB", (12, 12), (128, 128, 128))
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"not-a-real-model")
    guide_img = np.full((12, 12), 255, dtype=np.uint8)
    guide_img[:, 5:7] = 0
    guide_path = tmp_path / "line.png"
    Image.fromarray(guide_img, "L").save(guide_path)
    profile = Profile(
        line_guide_enabled=True,
        line_guide_model_path=str(model_path),
        line_guide_image_path=str(guide_path),
    )

    def fail_onnx(*_args, **_kwargs):
        raise RuntimeError("load failed")

    monkeypatch.setattr(line_guide_module, "load_line_guide_onnx", fail_onnx)

    result = resolve_line_guide(src, (12, 12), profile, tmp_path / "source.png")

    assert result.guide is not None
    assert result.source == "image"
    assert "ONNX failed" in result.message


def test_resolve_line_guide_continues_when_onnxruntime_missing_without_image(tmp_path, monkeypatch):
    src = Image.new("RGB", (12, 12), (128, 128, 128))
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"fake")
    profile = Profile(
        line_guide_enabled=True,
        line_guide_model_path=str(model_path),
    )

    def missing_runtime(*_args, **_kwargs):
        raise ModuleNotFoundError("onnxruntime")

    monkeypatch.setattr(line_guide_module, "load_line_guide_onnx", missing_runtime)

    result = resolve_line_guide(src, (12, 12), profile, tmp_path / "source.png")

    assert result.guide is None
    assert result.source == "failed"
    assert "ONNX failed" in result.message
    assert "unavailable" in result.message
    assert "generation continues" in result.message


def test_resolve_line_guide_honors_cpu_backend_for_onnx(tmp_path, monkeypatch):
    src = Image.new("RGB", (12, 12), (128, 128, 128))
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"fake")
    profile = Profile(
        compute_backend="cpu",
        line_guide_enabled=True,
        line_guide_model_path=str(model_path),
    )
    captured = {}

    def fake_onnx(*_args, **kwargs):
        captured["prefer_gpu"] = kwargs["prefer_gpu"]
        return np.ones((12, 12), dtype=np.float32), ["CPUExecutionProvider"]

    monkeypatch.setattr(line_guide_module, "load_line_guide_onnx", fake_onnx)

    result = resolve_line_guide(src, (12, 12), profile, tmp_path / "source.png")

    assert result.source == "onnx"
    assert captured["prefer_gpu"] is False


def test_output_to_map_handles_two_channel_logits():
    logits = np.zeros((2, 8, 8), dtype=np.float32)
    logits[1, :, 3:5] = 8.0

    guide = line_guide_module._output_to_map(logits)

    assert guide.shape == (8, 8)
    assert float(guide[:, 3:5].mean()) > 0.8
    assert float(guide[:, :2].mean()) < 0.05


def test_load_line_guide_onnx_retries_cpu_after_gpu_session_failure(tmp_path, monkeypatch):
    calls = []
    fake_module = types.ModuleType("onnxruntime")

    class FakeInput:
        name = "input"
        shape = [1, 3, 8, 8]
        type = "tensor(float)"

    class FakeSession:
        def __init__(self, _path, providers):
            calls.append(list(providers))
            if "CUDAExecutionProvider" in providers:
                raise RuntimeError("gpu init failed")

        def get_inputs(self):
            return [FakeInput()]

        def run(self, _names, _feed):
            out = np.zeros((1, 1, 8, 8), dtype=np.float32)
            out[:, :, :, 3:5] = 1.0
            return [out]

    fake_module.get_available_providers = lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]
    fake_module.InferenceSession = FakeSession
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_module)
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"fake")

    guide, providers = load_line_guide_onnx(
        Image.new("RGB", (8, 8), (128, 128, 128)),
        model_path,
        (8, 8),
        prefer_gpu=True,
    )

    assert calls == [["CUDAExecutionProvider", "CPUExecutionProvider"], ["CPUExecutionProvider"]]
    assert providers == ["CPUExecutionProvider"]
    assert float(guide[:, 3:5].mean()) > 0.8


def test_load_line_guide_onnx_retries_cpu_after_gpu_run_failure(tmp_path, monkeypatch):
    calls = []
    fake_module = types.ModuleType("onnxruntime")

    class FakeInput:
        name = "input"
        shape = [1, 3, 8, 8]
        type = "tensor(float)"

    class FakeSession:
        def __init__(self, _path, providers):
            self.providers = list(providers)
            calls.append(list(providers))

        def get_inputs(self):
            return [FakeInput()]

        def run(self, _names, _feed):
            if self.providers and self.providers[0] == "CUDAExecutionProvider":
                raise RuntimeError("gpu run failed")
            out = np.zeros((1, 1, 8, 8), dtype=np.float32)
            out[:, :, 2:6, 4:5] = 1.0
            return [out]

    fake_module.get_available_providers = lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]
    fake_module.InferenceSession = FakeSession
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_module)
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"fake")

    guide, providers = load_line_guide_onnx(
        Image.new("RGB", (8, 8), (128, 128, 128)),
        model_path,
        (8, 8),
        prefer_gpu=True,
    )

    assert calls == [["CUDAExecutionProvider", "CPUExecutionProvider"], ["CPUExecutionProvider"]]
    assert providers == ["CPUExecutionProvider"]
    assert float(guide[2:6, 4:5].mean()) > 0.8


def test_load_line_guide_onnx_uses_model_config_for_input_and_output(tmp_path, monkeypatch):
    captured = {}
    fake_module = types.ModuleType("onnxruntime")

    class FakeInput:
        name = "input"
        shape = [1, 3, 4, 4]
        type = "tensor(float)"

    class FakeSession:
        def __init__(self, _path, providers):
            self.providers = providers

        def get_inputs(self):
            return [FakeInput()]

        def run(self, _names, feed):
            captured["feed"] = feed["input"]
            out = np.zeros((1, 2, 4, 4), dtype=np.float32)
            out[:, 0, :, 1:3] = 8.0
            return [out]

    fake_module.get_available_providers = lambda: ["CPUExecutionProvider"]
    fake_module.InferenceSession = FakeSession
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_module)
    model_path = tmp_path / "line_guide.onnx"
    model_path.write_bytes(b"fake")
    (tmp_path / "line_guide.json").write_text(
        """{
  "input": {"colorOrder": "bgr", "valueRange": "minus1_1"},
  "output": {"channel": 0, "activation": "softmax"}
}""",
        encoding="utf-8",
    )

    guide, providers = load_line_guide_onnx(
        Image.new("RGB", (4, 4), (10, 20, 30)),
        model_path,
        (4, 4),
        prefer_gpu=False,
    )

    feed = captured["feed"]
    assert providers == ["CPUExecutionProvider"]
    assert feed.shape == (1, 3, 4, 4)
    assert float(feed[0, 0, 0, 0]) > float(feed[0, 2, 0, 0])
    assert float(guide[:, 1:3].mean()) > 0.8
    assert float(guide[:, :1].mean()) < 0.05


def test_default_model_path_uses_models_env_before_other_roots(tmp_path, monkeypatch):
    env_models = tmp_path / "env_models"
    work = tmp_path / "work"
    cwd_models = work / "models"
    env_models.mkdir()
    cwd_models.mkdir(parents=True)
    env_model = env_models / "line_guide.onnx"
    cwd_model = cwd_models / "line_guide.onnx"
    env_model.write_bytes(b"env")
    cwd_model.write_bytes(b"cwd")
    monkeypatch.setenv("FD6_MODELS_DIR", str(env_models))
    monkeypatch.chdir(work)

    assert line_guide_module._default_model_path() == env_model


def test_default_model_path_uses_pyinstaller_meipass_models(tmp_path, monkeypatch):
    dist = tmp_path / "dist"
    internal_models = dist / "_internal" / "models"
    work = tmp_path / "work"
    internal_models.mkdir(parents=True)
    work.mkdir()
    internal_model = internal_models / "line_guide.onnx"
    internal_model.write_bytes(b"bundle")
    monkeypatch.delenv("FD6_MODELS_DIR", raising=False)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(dist / "FD64FH6354221_onnx.exe"))
    monkeypatch.setattr(sys, "_MEIPASS", str(dist / "_internal"), raising=False)
    monkeypatch.chdir(work)

    assert line_guide_module._default_model_path() == internal_model


def test_quality_context_uses_line_guide_weight_and_points():
    target = np.full((16, 16, 3), 128, dtype=np.uint8)
    guide = np.zeros((16, 16), dtype=np.float32)
    guide[:, 7:9] = 1.0

    context = build_quality_context(
        target,
        None,
        edge_weight_strength=0.0,
        gradient_weight=0.0,
        edge_alpha=224,
        line_guide=guide,
        line_guide_strength=2.0,
        line_guide_agreement=0.0,
    )

    assert context is not None
    assert context.enabled
    assert context.has_edge_points
    assert context.edge_weight[8, 8] > context.edge_weight[8, 2]
    assert context.line_guide_strength > 0.0


def test_line_guide_sampling_weights_prefer_stronger_lines():
    target = np.full((16, 16, 3), 128, dtype=np.uint8)
    target[:, 4:] = 220
    guide = np.zeros((16, 16), dtype=np.float32)
    guide[:, 3:4] = 1.0
    guide[:, 10:13] = 0.25

    context = build_quality_context(
        target,
        None,
        edge_weight_strength=0.0,
        gradient_weight=0.0,
        edge_alpha=224,
        line_guide=guide,
        line_guide_strength=1.0,
        line_guide_agreement=0.0,
    )

    assert context is not None
    increments = np.diff(np.concatenate(([0.0], context.edge_sample_cdf)))
    strong = increments[context.edge_x == 3].mean()
    weak = increments[np.isin(context.edge_x, [10, 11, 12])].mean()
    assert strong > weak


def test_edge_candidate_shape_follows_line_angle():
    target = np.zeros((24, 24, 3), dtype=np.uint8)
    target[11:13, :] = 255
    context = build_quality_context(
        target,
        None,
        edge_weight_strength=1.0,
        gradient_weight=0.0,
        edge_alpha=224,
    )

    assert context is not None
    shape = _edge_candidate_shape(
        random.Random(3),
        24,
        24,
        ["rotated_rectangle"],
        context,
    )

    assert shape is not None
    assert shape.type_name == "rotated_rectangle"
    assert min(abs(float(shape.angle)), abs(float(shape.angle) - 180.0)) < 35.0
    assert float(shape.hw) > float(shape.hh)


def test_engine_line_guide_boosts_and_decays_candidate_ratio():
    target = np.full((16, 16, 3), 128, dtype=np.uint8)
    guide = np.zeros((16, 16), dtype=np.float32)
    guide[:, 8] = 1.0
    profile = Profile(
        stop_at=10,
        random_samples=4,
        mutated_samples=1,
        preview_every=0,
        max_threads=1,
        compute_backend="cpu",
        shape_types=["rotated_ellipse"],
        edge_weight_strength=0.0,
        gradient_weight=0.0,
        edge_candidate_ratio=0.0,
        line_guide_enabled=True,
        line_guide_strength=1.0,
        line_guide_candidate_ratio=0.5,
        line_guide_decay=0.5,
    )
    engine = Engine(target, EngineConfig(profile=profile, seed=1), line_guide=guide)
    try:
        assert engine.quality_context is not None
        assert engine.edge_candidate_ratio == 0.5
        initial_weight = float(engine._quality_edge_weight[8, 8])
        engine._shape_count = 5
        engine._refresh_line_guide_factor(force=True)
        assert engine._effective_edge_candidate_ratio() < 0.5
        assert engine._line_guide_score_factor < 1.0
        assert float(engine._quality_edge_weight[8, 8]) < initial_weight
    finally:
        engine._shutdown()


def test_engine_ignores_line_guide_when_disabled():
    target = np.full((16, 16, 3), 128, dtype=np.uint8)
    guide = np.ones((16, 16), dtype=np.float32)
    profile = Profile(
        stop_at=10,
        random_samples=4,
        mutated_samples=1,
        preview_every=0,
        max_threads=1,
        compute_backend="cpu",
        shape_types=["circle"],
        edge_weight_strength=0.0,
        gradient_weight=0.0,
        edge_candidate_ratio=0.1,
        line_guide_enabled=False,
        line_guide_strength=2.0,
        line_guide_candidate_ratio=0.8,
    )
    engine = Engine(target, EngineConfig(profile=profile, seed=1), line_guide=guide)
    try:
        assert engine.quality_context is None
        assert engine.edge_candidate_ratio == 0.1
    finally:
        engine._shutdown()


def test_disabled_line_guide_keeps_seeded_generation_identical():
    target = np.full((16, 16, 3), 128, dtype=np.uint8)
    target[4:12, 5:11] = (20, 80, 220)
    guide = np.ones((16, 16), dtype=np.float32)
    profile = Profile(
        stop_at=3,
        random_samples=8,
        mutated_samples=2,
        preview_every=0,
        save_at=[],
        save_every=0,
        max_threads=1,
        compute_backend="cpu",
        shape_types=["circle", "rectangle"],
        edge_weight_strength=0.0,
        gradient_weight=0.0,
        edge_candidate_ratio=0.0,
        line_guide_enabled=False,
        line_guide_strength=2.0,
        line_guide_candidate_ratio=0.8,
    )
    plain = Engine(target, EngineConfig(profile=profile, seed=77))
    disabled = Engine(target, EngineConfig(profile=profile, seed=77), line_guide=guide)
    try:
        for event in plain.run():
            assert event.kind != "error"
        for event in disabled.run():
            assert event.kind != "error"
        assert plain.rms == disabled.rms
        assert [shape.to_json() for shape in plain.shapes] == [shape.to_json() for shape in disabled.shapes]
    finally:
        plain._shutdown()
        disabled._shutdown()
