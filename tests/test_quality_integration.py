import time

import numpy as np

from fd6.shapegen.engine import Engine, EngineConfig
from fd6.shapegen.profile import Profile
from fd6.shapegen.quality import build_quality_context, edge_f1, gradient_error, ssim_index
from fd6.shapegen.scoring import rms_error


def _photo_like():
    y = np.linspace(0, 255, 32, dtype=np.uint8)[:, None]
    x = np.linspace(0, 255, 32, dtype=np.uint8)[None, :]
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[:, :, 0] = x
    img[:, :, 1] = y
    img[:, :, 2] = ((x.astype(np.uint16) + y.astype(np.uint16)) // 2).astype(np.uint8)
    img[9:23, 11:21] = (40, 70, 210)
    return img


def _anime_like():
    img = np.full((32, 32, 3), (238, 198, 166), dtype=np.uint8)
    img[:8, :] = (38, 42, 54)
    img[10:15, 8:13] = (20, 20, 30)
    img[10:15, 20:25] = (20, 20, 30)
    img[21:24, 12:21] = (160, 50, 70)
    return img


def _logo_like():
    img = np.full((32, 32, 3), 255, dtype=np.uint8)
    yy, xx = np.ogrid[:32, :32]
    circle = (xx - 16) ** 2 + (yy - 16) ** 2 <= 10 ** 2
    img[circle] = (20, 90, 210)
    img[14:18, 7:25] = (245, 245, 245)
    return img


def _text_like():
    img = np.full((32, 32, 3), 255, dtype=np.uint8)
    img[6:26, 5:9] = 0
    img[6:10, 5:17] = 0
    img[15:19, 5:16] = 0
    img[6:26, 21:25] = 0
    img[6:10, 17:29] = 0
    img[15:19, 17:28] = 0
    return img


def _profile(edge_enabled):
    return Profile(
        name="quality-integration",
        stop_at=6,
        random_samples=24,
        mutated_samples=8,
        preview_every=0,
        save_at=[],
        save_every=0,
        max_threads=1,
        compute_backend="cpu",
        shape_types=["rotated_ellipse", "ellipse", "circle", "triangle", "rectangle", "rotated_rectangle"],
        edge_weight_strength=0.9 if edge_enabled else 0.0,
        gradient_weight=0.14 if edge_enabled else 0.0,
        edge_candidate_ratio=0.5 if edge_enabled else 0.0,
        edge_candidate_alpha=240,
        edge_rerank_top_k=12,
    )


def _run(target, edge_enabled):
    started = time.perf_counter()
    engine = Engine(target, EngineConfig(profile=_profile(edge_enabled), seed=2468))
    done = None
    for event in engine.run():
        if event.kind == "done":
            done = event.canvas[:, :, :3]
    elapsed = time.perf_counter() - started
    assert done is not None
    context = build_quality_context(
        target,
        None,
        edge_weight_strength=1.0,
        gradient_weight=0.1,
        edge_alpha=224,
    )
    assert context is not None
    return {
        "rms": rms_error(done, target),
        "edge_f1": edge_f1(done, target),
        "gradient": gradient_error(done, context),
        "ssim": ssim_index(done, target),
        "elapsed": elapsed,
    }


def test_quality_profile_compares_image_categories_with_same_seed_and_shape_count():
    targets = [_photo_like(), _anime_like(), _logo_like(), _text_like()]

    for target in targets:
        baseline = _run(target, False)
        quality = _run(target, True)

        assert np.isfinite(baseline["rms"])
        assert np.isfinite(quality["rms"])
        assert 0.0 <= baseline["edge_f1"] <= 1.0
        assert 0.0 <= quality["edge_f1"] <= 1.0
        assert -1.0 <= baseline["ssim"] <= 1.0
        assert -1.0 <= quality["ssim"] <= 1.0
        assert quality["gradient"] <= baseline["gradient"] + 20.0
        assert quality["edge_f1"] >= baseline["edge_f1"] - 0.35
        assert quality["elapsed"] <= baseline["elapsed"] * 25.0 + 0.5
