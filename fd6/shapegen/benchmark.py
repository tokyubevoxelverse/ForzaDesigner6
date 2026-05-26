from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib
import json
from pathlib import Path
import subprocess
import threading
import time
from typing import Any

import numpy as np
from PIL import Image

from fd6.shapegen.engine import Engine, EngineConfig
from fd6.shapegen.line_guide import load_line_guide_image, load_line_guide_onnx
from fd6.shapegen.profile import Profile
from fd6.shapegen.quality import build_quality_context, edge_f1, edge_precision_recall_f1, gradient_error, ssim_index
from fd6.shapegen.scoring import rms_error


@dataclass(frozen=True)
class BenchmarkMetrics:
    elapsed_seconds: float
    rms: float
    edge_f1: float
    edge_precision_t1: float
    edge_recall_t1: float
    edge_f1_t1: float
    gradient_error: float
    ssim: float
    shape_count: int
    vram_peak_mb: float | None


def _nvidia_smi_memory_mb() -> float | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return None
    values: list[float] = []
    for line in result.stdout.splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            values.append(float(text))
        except ValueError:
            continue
    if not values:
        return None
    return max(values)


def _runtime_versions() -> dict[str, Any]:
    versions: dict[str, Any] = {}
    for name in ("torch", "onnxruntime"):
        try:
            module = importlib.import_module(name)
        except Exception:
            continue
        versions[name] = str(getattr(module, "__version__", "unknown"))
    try:
        ort = importlib.import_module("onnxruntime")
        versions["onnxruntime_providers"] = list(ort.get_available_providers())
    except Exception:
        pass
    try:
        torch = importlib.import_module("torch")
        if torch.cuda.is_available():
            versions["cuda_device"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return versions


class _GpuMemorySampler:
    def __init__(self, interval_seconds: float = 0.1) -> None:
        self._interval_seconds = max(0.05, float(interval_seconds))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._baseline_mb = _nvidia_smi_memory_mb()
        self.peak_mb = self._baseline_mb or 0.0

    def __enter__(self) -> "_GpuMemorySampler":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    def _run(self) -> None:
        while not self._stop.wait(self._interval_seconds):
            sample = _nvidia_smi_memory_mb()
            if sample is not None and sample > self.peak_mb:
                self.peak_mb = sample

    @property
    def peak_delta_mb(self) -> float | None:
        if self._baseline_mb is None:
            return None
        return max(0.0, float(self.peak_mb - self._baseline_mb))


def synthetic_target(size: int = 64) -> np.ndarray:
    img = np.full((size, size, 3), (238, 216, 184), dtype=np.uint8)
    margin = max(4, size // 8)
    img[margin:size - margin, margin:size - margin] = (48, 84, 212)
    img[size // 3:size // 3 + max(2, size // 16), margin:size - margin] = (245, 245, 245)
    img[margin:size - margin, size // 2:size // 2 + max(2, size // 18)] = (22, 24, 32)
    return img


def synthetic_line_guide(target: np.ndarray) -> np.ndarray:
    context = build_quality_context(
        target,
        None,
        edge_weight_strength=1.0,
        gradient_weight=0.0,
        edge_alpha=224,
    )
    if context is None:
        return np.zeros(target.shape[:2], dtype=np.float32)
    guide = np.clip(context.edge_weight - 1.0, 0.0, 1.0).astype(np.float32)
    return guide


def _cuda_peak_mb() -> float | None:
    try:
        torch = importlib.import_module("torch")
        if not torch.cuda.is_available():
            return None
        torch.cuda.synchronize()
        return float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))
    except Exception:
        return None


def _reset_cuda_peak() -> None:
    try:
        torch = importlib.import_module("torch")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
    except Exception:
        return


def _measure_engine(target: np.ndarray, profile: Profile, seed: int, line_guide: np.ndarray | None) -> BenchmarkMetrics:
    _reset_cuda_peak()
    started = time.perf_counter()
    engine = Engine(target, EngineConfig(profile=profile, seed=seed), line_guide=line_guide)
    done_canvas: np.ndarray | None = None
    with _GpuMemorySampler() as sampler:
        try:
            for event in engine.run():
                if event.kind == "error":
                    raise RuntimeError(str(event.message))
                if event.kind == "done" and event.canvas is not None:
                    done_canvas = event.canvas[:, :, :3]
            if done_canvas is None:
                done_canvas = engine.canvas.copy()
            elapsed = time.perf_counter() - started
            context = build_quality_context(
                target,
                None,
                edge_weight_strength=1.0,
                gradient_weight=0.1,
                edge_alpha=224,
            )
            precision_t1, recall_t1, f1_t1 = edge_precision_recall_f1(done_canvas, target, tolerance=1)
            vram_peak = sampler.peak_delta_mb
            torch_peak = _cuda_peak_mb()
            if vram_peak is None:
                vram_peak = torch_peak
            elif torch_peak is not None:
                vram_peak = max(vram_peak, torch_peak)
            return BenchmarkMetrics(
                elapsed_seconds=float(elapsed),
                rms=float(rms_error(done_canvas, target)),
                edge_f1=float(edge_f1(done_canvas, target)),
                edge_precision_t1=float(precision_t1),
                edge_recall_t1=float(recall_t1),
                edge_f1_t1=float(f1_t1),
                gradient_error=float(gradient_error(done_canvas, context)),
                ssim=float(ssim_index(done_canvas, target)),
                shape_count=len(engine.shapes),
                vram_peak_mb=vram_peak,
            )
        finally:
            engine._shutdown()


def benchmark_line_guide(
    target: np.ndarray,
    guide: np.ndarray,
    *,
    seed: int = 8,
    stop_at: int = 16,
    random_samples: int = 32,
    mutated_samples: int = 8,
    compute_backend: str = "cpu",
    guide_source: str = "synthetic",
    guide_prepare_seconds: float = 0.0,
    guide_prepare_vram_peak_mb: float | None = None,
    command: str = "",
) -> dict[str, Any]:
    base_profile = Profile(
        name="line-guide-benchmark-baseline",
        stop_at=stop_at,
        random_samples=random_samples,
        mutated_samples=mutated_samples,
        preview_every=0,
        save_at=[],
        save_every=0,
        max_threads=1,
        compute_backend=compute_backend,
        shape_types=["rotated_ellipse", "ellipse", "circle", "rectangle", "rotated_rectangle"],
        edge_weight_strength=0.75,
        gradient_weight=0.12,
        edge_candidate_ratio=0.18,
        edge_rerank_top_k=16,
        line_guide_enabled=False,
    )
    line_profile = Profile(**asdict(base_profile))
    line_profile.name = "line-guide-benchmark-enabled"
    line_profile.line_guide_enabled = True
    line_profile.line_guide_strength = 0.9
    line_profile.line_guide_decay = 0.55
    line_profile.line_guide_agreement = 0.7
    line_profile.line_guide_candidate_ratio = 0.25
    baseline = _measure_engine(target, base_profile, seed, None)
    line_guided = _measure_engine(target, line_profile, seed, guide)
    return {
        "parameters": {
            "seed": seed,
            "stop_at": stop_at,
            "random_samples": random_samples,
            "mutated_samples": mutated_samples,
            "compute_backend": compute_backend,
            "target_size": [int(target.shape[1]), int(target.shape[0])],
            "guide_source": guide_source,
            "command": command,
        },
        "environment": _runtime_versions(),
        "guide": {
            "source": guide_source,
            "prepare_seconds": float(guide_prepare_seconds),
            "prepare_vram_peak_mb": guide_prepare_vram_peak_mb,
        },
        "baseline": asdict(baseline),
        "line_guide": asdict(line_guided),
        "delta": {
            "elapsed_seconds": line_guided.elapsed_seconds - baseline.elapsed_seconds,
            "elapsed_ratio": (
                line_guided.elapsed_seconds / baseline.elapsed_seconds
                if baseline.elapsed_seconds > 0.0
                else None
            ),
            "rms": line_guided.rms - baseline.rms,
            "edge_f1": line_guided.edge_f1 - baseline.edge_f1,
            "edge_precision_t1": line_guided.edge_precision_t1 - baseline.edge_precision_t1,
            "edge_recall_t1": line_guided.edge_recall_t1 - baseline.edge_recall_t1,
            "edge_f1_t1": line_guided.edge_f1_t1 - baseline.edge_f1_t1,
            "gradient_error": line_guided.gradient_error - baseline.gradient_error,
            "ssim": line_guided.ssim - baseline.ssim,
        },
    }


def load_benchmark_inputs(
    image_path: str | None,
    guide_path: str | None,
    *,
    size: int = 40,
    model_path: str | None = None,
    prefer_gpu: bool = True,
) -> tuple[np.ndarray, np.ndarray, str, float, float | None]:
    guide_prepare_seconds = 0.0
    guide_prepare_vram_peak_mb: float | None = None
    if image_path:
        with Image.open(image_path) as img:
            source = img.convert("RGB")
            target = np.asarray(source, dtype=np.uint8)
    else:
        target = synthetic_target(size)
        source = Image.fromarray(target, "RGB")
    if model_path:
        started = time.perf_counter()
        with _GpuMemorySampler() as sampler:
            guide, providers = load_line_guide_onnx(
                source,
                Path(model_path),
                (target.shape[1], target.shape[0]),
                prefer_gpu=prefer_gpu,
            )
        guide_prepare_vram_peak_mb = sampler.peak_delta_mb
        guide_prepare_seconds = time.perf_counter() - started
        return (
            target,
            guide,
            f"onnx:{Path(model_path).name}:{','.join(providers)}",
            guide_prepare_seconds,
            guide_prepare_vram_peak_mb,
        )
    if guide_path:
        guide = load_line_guide_image(Path(guide_path), (target.shape[1], target.shape[0]))
        return target, guide, f"image:{Path(guide_path).name}", guide_prepare_seconds, guide_prepare_vram_peak_mb
    else:
        guide = synthetic_line_guide(target)
        return target, guide, "synthetic", guide_prepare_seconds, guide_prepare_vram_peak_mb


def write_benchmark_report(report: dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
