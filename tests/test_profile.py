from fd6.shapegen.profile import Profile, load_profile
from fd6.shapegen.torch_backend import resolve_compute_backend


def test_profile_roundtrip_preserves_compute_backend():
    profile = Profile(
        name="gpu-test",
        compute_backend="gpu",
        max_threads=7,
        random_samples=321,
        refine_passes=2,
        edge_weight_strength=1.25,
        gradient_weight=0.3,
        edge_candidate_ratio=0.4,
        edge_candidate_alpha=240,
        edge_rerank_top_k=32,
        quality_batch_pixels=128000,
        line_guide_enabled=True,
        line_guide_image_path="guide.png",
        line_guide_model_path="models/line_guide.onnx",
        line_guide_strength=1.4,
        line_guide_decay=0.4,
        line_guide_agreement=0.7,
        line_guide_candidate_ratio=0.35,
        line_guide_max_resolution=768,
    )
    loaded = load_profile("gpu-test", profile.to_ini())
    assert loaded.compute_backend == "gpu"
    assert loaded.max_threads == 7
    assert loaded.random_samples == 321
    assert loaded.refine_passes == 2
    assert loaded.edge_weight_strength == 1.25
    assert loaded.gradient_weight == 0.3
    assert loaded.edge_candidate_ratio == 0.4
    assert loaded.edge_candidate_alpha == 240
    assert loaded.edge_rerank_top_k == 32
    assert loaded.quality_batch_pixels == 128000
    assert loaded.line_guide_enabled is True
    assert loaded.line_guide_image_path == "guide.png"
    assert loaded.line_guide_model_path == "models/line_guide.onnx"
    assert loaded.line_guide_strength == 1.4
    assert loaded.line_guide_decay == 0.4
    assert loaded.line_guide_agreement == 0.7
    assert loaded.line_guide_candidate_ratio == 0.35
    assert loaded.line_guide_max_resolution == 768


def test_resolve_compute_backend_cpu():
    info = resolve_compute_backend("cpu")
    assert info.requested == "cpu"
    assert info.resolved == "cpu"
    assert info.label == "CPU"


def test_resolve_compute_backend_auto():
    info = resolve_compute_backend("auto")
    assert info.requested == "auto"
    assert info.resolved in {"cpu", "gpu"}
