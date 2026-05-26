from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import ctypes
import math
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory

import numpy as np

from fd6.shapegen.profile import Profile
from fd6.shapegen.quality import (
    QualityContext,
    build_quality_context,
    precompute_gradient_error,
    precompute_weighted_rgb_error,
)
from fd6.shapegen.scoring import apply_shape_inplace, precompute_canvas_error, score_shape
from fd6.shapegen.shapes import Shape, random_shape
from fd6.shapegen.shapes.base import cached_bbox_metrics
from fd6.shapegen.shapes.circle import Circle
from fd6.shapegen.shapes.ellipse import Ellipse, RotatedEllipse
from fd6.shapegen.shapes.rectangle import Rectangle, RotatedRectangle
from fd6.shapegen.shapes.triangle import Triangle
from fd6.shapegen.torch_backend import TorchSearchRuntime, resolve_compute_backend


def _available_ram_mb() -> int:
    if sys.platform == "win32":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        try:
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return int(stat.ullAvailPhys // (1024 * 1024))
        except Exception:
            return 4096
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except Exception:
        pass
    return 4096


def _safe_worker_count(user_requested: int, random_samples: int) -> int:
    cpu = os.cpu_count() or 1
    headroom = 1 if cpu <= 4 else 2
    cpu_cap = max(1, cpu - headroom)
    free_mb = _available_ram_mb()
    ram_budget_mb = max(0, free_mb - 2048)
    ram_cap = max(1, ram_budget_mb // 250)
    work_cap = max(1, random_samples // 64)
    requested = user_requested if user_requested > 0 else cpu_cap
    return max(1, min(requested, cpu_cap, ram_cap, work_cap))


def _gpu_restart_count(user_requested: int, random_samples: int) -> int:
    requested = user_requested if user_requested > 0 else 2
    work_cap = max(1, random_samples // 128)
    return max(1, min(requested, 8, work_cap))


def _merge_bbox(
    left: tuple[int, int, int, int] | None,
    right: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    if left is None:
        return right
    return (
        min(left[0], right[0]),
        min(left[1], right[1]),
        max(left[2], right[2]),
        max(left[3], right[3]),
    )


@dataclass
class EngineConfig:
    profile: Profile
    seed: int = 0


@dataclass
class EngineEvent:
    kind: str
    shape_count: int = 0
    rms: float = 0.0
    canvas: np.ndarray | None = None
    message: str = ""


@dataclass
class _GpuChainState:
    rng: random.Random
    best_score: float = float("inf")
    best_shape: Shape | None = None
    no_improve: int = 0


@dataclass
class _GpuRotatedEllipsePack:
    pack: object


_W_TARGET: np.ndarray | None = None
_W_ALPHA: np.ndarray | None = None
_W_QUALITY: QualityContext | None = None
_W_CANVAS_SHM: shared_memory.SharedMemory | None = None
_W_CANVAS: np.ndarray | None = None


def _init_worker(
    target_bytes: bytes, target_shape: tuple,
    canvas_shm_name: str, canvas_shape: tuple,
    alpha_bytes: bytes | None, alpha_shape: tuple | None,
    line_guide_bytes: bytes | None, line_guide_shape: tuple | None,
    quality_params: tuple[float, float, int, float, float] | None,
) -> None:
    global _W_TARGET, _W_ALPHA, _W_QUALITY, _W_CANVAS_SHM, _W_CANVAS
    _W_TARGET = np.frombuffer(target_bytes, dtype=np.uint8).reshape(target_shape).copy()
    if alpha_bytes is not None and alpha_shape is not None:
        _W_ALPHA = np.frombuffer(alpha_bytes, dtype=np.uint8).reshape(alpha_shape).copy()
    else:
        _W_ALPHA = None
    if quality_params is None:
        _W_QUALITY = None
    else:
        edge_strength, gradient_weight, edge_alpha, line_guide_strength, line_guide_agreement = quality_params
        line_guide = None
        if line_guide_bytes is not None and line_guide_shape is not None:
            line_guide = np.frombuffer(line_guide_bytes, dtype=np.float32).reshape(line_guide_shape).copy()
        _W_QUALITY = build_quality_context(
            _W_TARGET,
            _W_ALPHA,
            edge_weight_strength=edge_strength,
            gradient_weight=gradient_weight,
            edge_alpha=edge_alpha,
            line_guide=line_guide,
            line_guide_strength=line_guide_strength,
            line_guide_agreement=line_guide_agreement,
        )
    _W_CANVAS_SHM = shared_memory.SharedMemory(name=canvas_shm_name)
    _W_CANVAS = np.ndarray(canvas_shape, dtype=np.uint8, buffer=_W_CANVAS_SHM.buf)


def _profile_quality_params(profile: Profile, has_line_guide: bool = False) -> tuple[float, float, int, float, float] | None:
    edge_strength = max(0.0, float(getattr(profile, "edge_weight_strength", 0.0)))
    gradient_weight = max(0.0, float(getattr(profile, "gradient_weight", 0.0)))
    line_guide_strength = 0.0
    if has_line_guide and bool(getattr(profile, "line_guide_enabled", False)):
        line_guide_strength = max(0.0, float(getattr(profile, "line_guide_strength", 0.0)))
    if edge_strength <= 0.0 and gradient_weight <= 0.0 and line_guide_strength <= 0.0:
        return None
    line_guide_agreement = max(0.0, min(1.0, float(getattr(profile, "line_guide_agreement", 0.0))))
    return edge_strength, gradient_weight, int(getattr(profile, "edge_candidate_alpha", 224)), line_guide_strength, line_guide_agreement


def _clamp_float(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _edge_candidate_shape(
    rng: random.Random,
    w: int,
    h: int,
    allowed_types: list[str],
    quality: QualityContext | None,
) -> Shape | None:
    if quality is None or not quality.has_edge_points:
        return None
    usable = [t for t in allowed_types if t in {
        "rotated_ellipse", "ellipse", "circle", "triangle", "rectangle", "rotated_rectangle",
    }]
    if not usable:
        return None
    if quality.edge_sample_cdf.size == quality.edge_x.size and quality.edge_sample_cdf.size > 0:
        pick = rng.random() * max(float(quality.edge_sample_cdf[-1]), 1e-6)
        idx = int(np.searchsorted(quality.edge_sample_cdf, pick, side="left"))
        idx = max(0, min(idx, int(quality.edge_x.size) - 1))
    else:
        idx = rng.randrange(int(quality.edge_x.size))
    x = _clamp_float(float(quality.edge_x[idx]) + rng.gauss(0.0, 1.5), 0.0, float(w - 1))
    y = _clamp_float(float(quality.edge_y[idx]) + rng.gauss(0.0, 1.5), 0.0, float(h - 1))
    angle = (float(quality.edge_angle[idx]) + rng.gauss(0.0, 12.0)) % 180.0
    short_side = max(1.0, float(min(w, h)))
    length = rng.uniform(3.0, max(5.0, short_side / 10.0))
    thickness = rng.uniform(1.0, max(2.0, short_side / 96.0))
    alpha = int(quality.edge_alpha)
    color = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255), alpha)
    type_name = rng.choice(usable)
    if type_name == "rotated_ellipse":
        candidate = RotatedEllipse(color=color, x=x, y=y, rx=length, ry=thickness, angle=angle)
        setattr(candidate, "_fd6_edge_candidate", True)
        return candidate
    if type_name == "rotated_rectangle":
        candidate = RotatedRectangle(color=color, x=x, y=y, hw=length, hh=thickness, angle=angle)
        setattr(candidate, "_fd6_edge_candidate", True)
        return candidate
    if type_name == "ellipse":
        rad = math.radians(angle)
        horizontal = abs(math.cos(rad)) >= abs(math.sin(rad))
        candidate = Ellipse(
            color=color,
            x=x,
            y=y,
            rx=length if horizontal else thickness,
            ry=thickness if horizontal else length,
        )
        setattr(candidate, "_fd6_edge_candidate", True)
        return candidate
    if type_name == "rectangle":
        rad = math.radians(angle)
        horizontal = abs(math.cos(rad)) >= abs(math.sin(rad))
        candidate = Rectangle(
            color=color,
            x=x,
            y=y,
            hw=length if horizontal else thickness,
            hh=thickness if horizontal else length,
        )
        setattr(candidate, "_fd6_edge_candidate", True)
        return candidate
    if type_name == "circle":
        candidate = Circle(color=color, x=x, y=y, r=rng.uniform(1.0, max(2.0, thickness * 2.0)))
        setattr(candidate, "_fd6_edge_candidate", True)
        return candidate
    rad = math.radians(angle)
    tx, ty = math.cos(rad), math.sin(rad)
    nx, ny = -ty, tx
    half = length * 0.5
    bend = rng.choice((-1.0, 1.0)) * thickness * rng.uniform(1.0, 2.4)
    candidate = Triangle(
        color=color,
        x1=_clamp_float(x - tx * half, 0.0, float(w - 1)),
        y1=_clamp_float(y - ty * half, 0.0, float(h - 1)),
        x2=_clamp_float(x + tx * half, 0.0, float(w - 1)),
        y2=_clamp_float(y + ty * half, 0.0, float(h - 1)),
        x3=_clamp_float(x + nx * bend + rng.uniform(-1.0, 1.0), 0.0, float(w - 1)),
        y3=_clamp_float(y + ny * bend + rng.uniform(-1.0, 1.0), 0.0, float(h - 1)),
    )
    setattr(candidate, "_fd6_edge_candidate", True)
    return candidate


def _search_shape(
    rng: random.Random,
    w: int,
    h: int,
    allowed_types: list[str],
    quality: QualityContext | None,
    edge_candidate_ratio: float,
) -> Shape:
    if quality is not None and edge_candidate_ratio > 0.0 and rng.random() < edge_candidate_ratio:
        candidate = _edge_candidate_shape(rng, w, h, allowed_types, quality)
        if candidate is not None:
            return candidate
    return random_shape(rng, w, h, allowed_types)


def _independent_search(
    canvas: np.ndarray,
    target: np.ndarray,
    alpha: np.ndarray | None,
    quality: QualityContext | None,
    args: tuple,
) -> tuple[float, tuple[int, int, int, int] | None, Shape | None]:
    types, n_random, n_mutate, w, h, seed, edge_candidate_ratio, edge_rerank_top_k = args[:8]
    rng = random.Random(seed)
    line_guide_factor = 1.0
    if len(args) >= 14:
        canvas_full_sq = float(args[8])
        canvas_norm = float(args[9])
        weighted_full_sq = float(args[10])
        weighted_norm = float(args[11])
        gradient_full_error = float(args[12])
        gradient_norm = float(args[13])
        if len(args) >= 15:
            line_guide_factor = float(args[14])
    else:
        canvas_full_sq, canvas_norm = precompute_canvas_error(canvas, target, alpha)
        weighted_full_sq, weighted_norm = precompute_weighted_rgb_error(canvas, target, quality, line_guide_factor)
        gradient_full_error, gradient_norm = precompute_gradient_error(canvas, quality, line_guide_factor)
    best_score = float("inf")
    best_color = None
    best_shape = None
    random_candidates = [
        _search_shape(rng, w, h, types, quality, edge_candidate_ratio)
        for _ in range(max(1, n_random))
    ]
    random_scores, random_colors = _score_search_candidates(
        random_candidates,
        canvas,
        target,
        alpha,
        quality,
        canvas_full_sq,
        canvas_norm,
        weighted_full_sq,
        weighted_norm,
        gradient_full_error,
        gradient_norm,
        edge_rerank_top_k,
        line_guide_factor,
    )
    for idx, s in enumerate(random_candidates):
        score = float(random_scores[idx])
        color = random_colors[idx]
        if score < best_score:
            best_score, best_color, best_shape = score, color, s
    if best_shape is None:
        return (float("inf"), None, None)
    best_shape.color = best_color
    no_improve = 0
    cap = max(1, n_mutate)
    top_k = int(edge_rerank_top_k)
    if quality is None or top_k <= 0 or n_mutate <= 0:
        for _ in range(cap):
            cand = best_shape.mutate(rng, w, h)
            score, color = score_shape(
                cand,
                canvas,
                target,
                alpha,
                canvas_full_sq=canvas_full_sq,
                canvas_norm=canvas_norm,
                quality_context=quality,
                weighted_full_sq=weighted_full_sq,
                weighted_norm=weighted_norm,
                gradient_full_error=gradient_full_error,
                gradient_norm=gradient_norm,
                quality_edge_weight=quality.weight_for_line_factor(line_guide_factor) if quality is not None else None,
            )
            if score < best_score:
                best_score, best_color, best_shape = score, color, cand
                best_shape.color = best_color
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= max(20, cap // 4):
                    break
    else:
        remaining = cap
        stall_limit = max(20, cap // 4)
        group_size = max(1, min(128, max(4, top_k * 4), remaining))
        while remaining > 0:
            count = min(group_size, remaining)
            mutation_candidates = [best_shape.mutate(rng, w, h) for _ in range(count)]
            mutation_scores, mutation_colors = _score_search_candidates(
                mutation_candidates,
                canvas,
                target,
                alpha,
                quality,
                canvas_full_sq,
                canvas_norm,
                weighted_full_sq,
                weighted_norm,
                gradient_full_error,
                gradient_norm,
                top_k,
                line_guide_factor,
            )
            best_idx = int(np.argmin(mutation_scores))
            score = float(mutation_scores[best_idx])
            color = mutation_colors[best_idx]
            cand = mutation_candidates[best_idx]
            if np.isfinite(score) and score < best_score and color is not None:
                best_score, best_color, best_shape = score, color, cand
                best_shape.color = best_color
                no_improve = 0
            else:
                no_improve += count
                if no_improve >= stall_limit:
                    break
            remaining -= count
    if best_color is not None and best_shape is not None:
        best_shape.color = best_color
    return (best_score, best_color, best_shape)


def _score_search_candidates(
    candidates: list[Shape],
    canvas: np.ndarray,
    target: np.ndarray,
    alpha: np.ndarray | None,
    quality: QualityContext | None,
    canvas_full_sq: float,
    canvas_norm: float,
    weighted_full_sq: float,
    weighted_norm: float,
    gradient_full_error: float,
    gradient_norm: float,
    edge_rerank_top_k: int,
    line_guide_factor: float = 1.0,
) -> tuple[np.ndarray, list[tuple[int, int, int, int] | None]]:
    scores = np.full(len(candidates), float("inf"), dtype=np.float32)
    colors: list[tuple[int, int, int, int] | None] = [None] * len(candidates)
    if not candidates:
        return scores, colors
    if quality is None:
        for idx, candidate in enumerate(candidates):
            score, color = score_shape(
                candidate,
                canvas,
                target,
                alpha,
                canvas_full_sq=canvas_full_sq,
                canvas_norm=canvas_norm,
            )
            scores[idx] = float(score)
            colors[idx] = color
        return scores, colors
    base_scores = np.full(len(candidates), float("inf"), dtype=np.float32)
    base_colors: list[tuple[int, int, int, int] | None] = [None] * len(candidates)
    for idx, candidate in enumerate(candidates):
        score, color = score_shape(
            candidate,
            canvas,
            target,
            alpha,
            canvas_full_sq=canvas_full_sq,
            canvas_norm=canvas_norm,
        )
        base_scores[idx] = float(score)
        base_colors[idx] = color
    finite = np.flatnonzero(np.isfinite(base_scores))
    if finite.size == 0:
        return scores, colors
    top_k = int(edge_rerank_top_k)
    if top_k <= 0 or top_k >= finite.size:
        chosen = finite[np.argsort(base_scores[finite])]
    else:
        chosen = finite[np.argsort(base_scores[finite])[:top_k]]
        edge_chosen = np.asarray(
            [idx for idx in finite if getattr(candidates[int(idx)], "_fd6_edge_candidate", False)],
            dtype=np.int64,
        )
        if edge_chosen.size:
            edge_order = np.argsort(base_scores[edge_chosen])[:min(top_k, edge_chosen.size)]
            edge_chosen = edge_chosen[edge_order]
            chosen = np.unique(np.concatenate((chosen, edge_chosen)))
    quality_edge_weight = quality.weight_for_line_factor(line_guide_factor)
    for idx in chosen:
        candidate = candidates[int(idx)]
        fixed_color = base_colors[int(idx)]
        base_rms = getattr(candidate, "_fd6_rms_score", None)
        base_total_sq = getattr(candidate, "_fd6_canvas_full_sq", None)
        score, color = score_shape(
            candidate,
            canvas,
            target,
            alpha,
            canvas_full_sq=canvas_full_sq,
            canvas_norm=canvas_norm,
            quality_context=quality,
            weighted_full_sq=weighted_full_sq,
            weighted_norm=weighted_norm,
            gradient_full_error=gradient_full_error,
            gradient_norm=gradient_norm,
            quality_edge_weight=quality_edge_weight,
            fixed_color=fixed_color,
            base_rms=base_rms,
            base_canvas_full_sq=base_total_sq,
        )
        scores[int(idx)] = float(score)
        colors[int(idx)] = color
    return scores, colors


def _worker_independent_search(args: tuple) -> tuple[float, tuple[int, int, int, int] | None, Shape | None]:
    return _independent_search(_W_CANVAS, _W_TARGET, _W_ALPHA, _W_QUALITY, args)


class Engine:
    def __init__(
        self,
        target_rgb: np.ndarray,
        config: EngineConfig,
        alpha_mask: np.ndarray | None = None,
        line_guide: np.ndarray | None = None,
        line_guide_status: str = "",
    ) -> None:
        if target_rgb.ndim != 3 or target_rgb.shape[2] != 3:
            raise ValueError("target_rgb must be HxWx3 RGB uint8")
        self.target = target_rgb.astype(np.uint8)
        self.config = config
        self.profile = config.profile
        self.h, self.w = self.target.shape[:2]
        self.line_guide_status = line_guide_status
        self.line_guide_map: np.ndarray | None = None
        if line_guide is not None and bool(getattr(self.profile, "line_guide_enabled", False)):
            guide = np.asarray(line_guide, dtype=np.float32)
            if guide.shape == (self.h, self.w):
                self.line_guide_map = np.nan_to_num(
                    np.clip(guide, 0.0, 1.0),
                    nan=0.0,
                    posinf=1.0,
                    neginf=0.0,
                ).astype(np.float32)
        self.alpha_mask = alpha_mask if alpha_mask is not None else None
        if self.alpha_mask is not None:
            mask3 = (self.alpha_mask > 0)[:, :, None]
            self.target = self.target * mask3.astype(np.uint8)
            initial_canvas = np.full((self.h, self.w, 3), 40, dtype=np.uint8)
        else:
            avg = self.target.reshape(-1, 3).mean(axis=0).astype(np.uint8)
            initial_canvas = np.tile(avg, (self.h, self.w, 1)).astype(np.uint8)
        self._initial_canvas = initial_canvas.copy()
        self._canvas_shm: shared_memory.SharedMemory | None = shared_memory.SharedMemory(
            create=True, size=initial_canvas.nbytes,
        )
        self.canvas = np.ndarray(initial_canvas.shape, dtype=np.uint8, buffer=self._canvas_shm.buf)
        self.canvas[:] = initial_canvas
        self.shapes: list[Shape] = []
        self._pending_gpu_shape_packs: list[object] = []
        self._shape_count = 0
        quality_params = _profile_quality_params(self.profile, self.line_guide_map is not None)
        self.quality_context = None if quality_params is None else build_quality_context(
            self.target,
            self.alpha_mask,
            edge_weight_strength=quality_params[0],
            gradient_weight=quality_params[1],
            edge_alpha=quality_params[2],
            line_guide=self.line_guide_map,
            line_guide_strength=quality_params[3],
            line_guide_agreement=quality_params[4],
        )
        self._line_guide_score_factor = self._line_guide_candidate_factor()
        self._quality_edge_weight = (
            None
            if self.quality_context is None
            else self.quality_context.weight_for_line_factor(self._line_guide_score_factor)
        )
        self._base_edge_candidate_ratio = max(0.0, min(1.0, float(getattr(self.profile, "edge_candidate_ratio", 0.0))))
        self._line_guide_candidate_ratio = max(0.0, min(1.0, float(getattr(self.profile, "line_guide_candidate_ratio", 0.0))))
        self.edge_candidate_ratio = self._effective_edge_candidate_ratio()
        self.weighted_full_sq = 0.0
        self.weighted_norm = 0.0
        self.gradient_full_error = 0.0
        self.gradient_norm = 0.0
        self._refresh_score_state()
        self.rms = 0.0 if self.canvas_norm < 1 else float(np.sqrt(self.canvas_full_sq / self.canvas_norm))
        self.start_rms = self.rms
        self._stop = False
        self._pause = False
        seed = config.seed or int(time.time() * 1000) & 0xFFFFFFFF
        self.rng = random.Random(seed)
        backend_info = resolve_compute_backend(self.profile.compute_backend)
        self.compute_backend = backend_info.resolved
        self.compute_label = backend_info.label
        self._executor: ProcessPoolExecutor | None = None
        self._gpu_runtime: TorchSearchRuntime | None = None
        self._gpu_canvas_dirty = False
        self._gpu_dirty_bbox: tuple[int, int, int, int] | None = None
        if self.compute_backend == "gpu":
            self._n_workers = _gpu_restart_count(
                user_requested=self.profile.max_threads,
                random_samples=self.profile.random_samples,
            )
            self._gpu_runtime = TorchSearchRuntime(self.target, self.alpha_mask)
            quality_batch_pixels = int(getattr(self.profile, "quality_batch_pixels", 0))
            if quality_batch_pixels > 0:
                set_quality_batch_pixel_limit = getattr(self._gpu_runtime, "set_quality_batch_pixel_limit", None)
                if set_quality_batch_pixel_limit is not None:
                    set_quality_batch_pixel_limit(quality_batch_pixels)
            set_quality_context = getattr(self._gpu_runtime, "set_quality_context", None)
            if self.quality_context is not None and set_quality_context is not None:
                set_quality_context(
                    self._quality_edge_weight if self._quality_edge_weight is not None else self.quality_context.edge_weight,
                    self.quality_context.target_gx,
                    self.quality_context.target_gy,
                    self.quality_context.gradient_weight,
                )
            rotated_ellipse_enabled = "rotated_ellipse" in {t for t in self.profile.shape_types if t}
            graph_large_ready = self._n_workers == 2 and self.profile.random_samples >= 320
            graph_medium_ready = self._n_workers == 2 and self.profile.random_samples >= 192
            self._gpu_runtime.enable_rotated_graph_default = graph_large_ready
            self._gpu_runtime.enable_rotated_graph_medium = graph_medium_ready
            self._gpu_runtime.rotated_ellipse_reduce_stage4_after = 80 if self.profile.stop_at >= 120 else None
            self._gpu_runtime.rotated_ellipse_reduce_stage12_to9_after = 96 if self.profile.stop_at >= 120 else None
            if self.profile.stop_at < 96:
                self._gpu_runtime.rotated_graph_default_candidate_count = 3072
            elif self.profile.stop_at < 200:
                self._gpu_runtime.rotated_graph_default_candidate_count = 1536
            else:
                self._gpu_runtime.rotated_graph_default_candidate_count = 2048
            self._gpu_runtime.rotated_graph_default_topk = 2
            self._gpu_runtime.sync_full_canvas(self.canvas)
            if rotated_ellipse_enabled and self.alpha_mask is None:
                self._gpu_runtime.can_use_rotated_ellipse_cupy()
            self._gpu_runtime.prefer_rotated_ellipse_cupy = False
            if rotated_ellipse_enabled and self.alpha_mask is None:
                area = self.w * self.h
                if graph_large_ready and area >= 280_000:
                    self._gpu_runtime._init_rotated_ellipse_graph_default(self.canvas_norm)
                elif graph_medium_ready and area < 280_000 and self.profile.stop_at < 48:
                    self._gpu_runtime._init_rotated_ellipse_graph_medium(self.canvas_norm)
        else:
            self._n_workers = _safe_worker_count(
                user_requested=self.profile.max_threads,
                random_samples=self.profile.random_samples,
            )
            if self._n_workers > 1:
                target_bytes = self.target.tobytes()
                alpha_bytes = self.alpha_mask.tobytes() if self.alpha_mask is not None else None
                alpha_shape = self.alpha_mask.shape if self.alpha_mask is not None else None
                line_guide_bytes = self.line_guide_map.tobytes() if self.line_guide_map is not None else None
                line_guide_shape = self.line_guide_map.shape if self.line_guide_map is not None else None
                self._executor = ProcessPoolExecutor(
                    max_workers=self._n_workers,
                    initializer=_init_worker,
                    initargs=(
                        target_bytes, self.target.shape,
                        self._canvas_shm.name, self.canvas.shape,
                        alpha_bytes, alpha_shape,
                        line_guide_bytes, line_guide_shape,
                        quality_params,
                    ),
                )

    def request_stop(self) -> None:
        self._stop = True

    def set_pause(self, paused: bool) -> None:
        self._pause = paused

    def _line_guide_candidate_factor(self) -> float:
        if self.line_guide_map is None:
            return 0.0
        if not bool(getattr(self.profile, "line_guide_enabled", False)):
            return 0.0
        decay = max(0.05, float(getattr(self.profile, "line_guide_decay", 0.55)))
        progress = 0.0 if self.profile.stop_at <= 0 else min(1.0, float(self._shape_count) / float(self.profile.stop_at))
        return float(math.exp(-progress / decay))

    def _refresh_line_guide_factor(self, force: bool = False) -> bool:
        if self.quality_context is None or self.line_guide_map is None:
            return False
        factor = self._line_guide_candidate_factor()
        if not force and abs(factor - self._line_guide_score_factor) < 0.01:
            return False
        self._line_guide_score_factor = factor
        self._quality_edge_weight = self.quality_context.weight_for_line_factor(factor)
        if self._gpu_runtime is not None:
            set_quality_context = getattr(self._gpu_runtime, "set_quality_context", None)
            if set_quality_context is not None:
                set_quality_context(
                    self._quality_edge_weight,
                    self.quality_context.target_gx,
                    self.quality_context.target_gy,
                    self.quality_context.gradient_weight,
                )
        self._refresh_score_state()
        return True

    def _effective_edge_candidate_ratio(self) -> float:
        return max(
            0.0,
            min(
                1.0,
                self._base_edge_candidate_ratio
                + self._line_guide_candidate_ratio * self._line_guide_candidate_factor(),
            ),
        )

    def _sync_cpu_canvas_from_gpu(self) -> None:
        if self._gpu_runtime is None or not self._gpu_canvas_dirty:
            return
        if self._gpu_dirty_bbox is None:
            self._gpu_runtime.copy_canvas_to(self.canvas)
        else:
            self._gpu_runtime.copy_canvas_region_to(self.canvas, self._gpu_dirty_bbox)
        self._gpu_canvas_dirty = False
        self._gpu_dirty_bbox = None

    def _preview_canvas(self) -> np.ndarray:
        self._sync_cpu_canvas_from_gpu()
        if self.alpha_mask is not None:
            return np.dstack([self.canvas, self.alpha_mask]).copy()
        return self.canvas.copy()

    def _flush_pending_gpu_shapes(self) -> None:
        if not self._pending_gpu_shape_packs:
            return
        if self._gpu_runtime is None:
            self._pending_gpu_shape_packs.clear()
            return
        torch = self._gpu_runtime._torch
        with torch.inference_mode():
            rows = torch.stack(self._pending_gpu_shape_packs).to(device="cpu", dtype=torch.float32).tolist()
        for row in rows:
            self.shapes.append(
                RotatedEllipse(
                    color=(int(row[6]), int(row[7]), int(row[8]), int(row[9])),
                    x=float(row[1]),
                    y=float(row[2]),
                    rx=float(row[3]),
                    ry=float(row[4]),
                    angle=float(row[5]),
                )
            )
        self._pending_gpu_shape_packs.clear()

    def _refresh_score_state(self) -> None:
        self.canvas_full_sq, self.canvas_norm = precompute_canvas_error(self.canvas, self.target, self.alpha_mask)
        if self.quality_context is not None:
            self.weighted_full_sq, self.weighted_norm = precompute_weighted_rgb_error(
                self.canvas,
                self.target,
                self.quality_context,
                self._line_guide_score_factor,
            )
            if self.quality_context.gradient_weight > 0.0:
                self.gradient_full_error, self.gradient_norm = precompute_gradient_error(
                    self.canvas,
                    self.quality_context,
                    self._line_guide_score_factor,
                )
            else:
                self.gradient_full_error = 0.0
                self.gradient_norm = 0.0
        else:
            self.weighted_full_sq = 0.0
            self.weighted_norm = 0.0
            self.gradient_full_error = 0.0
            self.gradient_norm = 0.0
        self.rms = 0.0 if self.canvas_norm < 1 else float(np.sqrt(self.canvas_full_sq / self.canvas_norm))

    def _score_shape(
        self,
        shape: Shape,
        fixed_color: tuple[int, int, int, int] | None = None,
        base_rms: float | None = None,
        base_canvas_full_sq: float | None = None,
    ) -> tuple[float, tuple[int, int, int, int]]:
        return score_shape(
            shape,
            self.canvas,
            self.target,
            self.alpha_mask,
            canvas_full_sq=self.canvas_full_sq,
            canvas_norm=self.canvas_norm,
            quality_context=self.quality_context,
            weighted_full_sq=self.weighted_full_sq,
            weighted_norm=self.weighted_norm,
            gradient_full_error=self.gradient_full_error,
            gradient_norm=self.gradient_norm,
            quality_edge_weight=self._quality_edge_weight,
            fixed_color=fixed_color,
            base_rms=base_rms,
            base_canvas_full_sq=base_canvas_full_sq,
        )

    def _update_score_state_from_shape(self, shape: Shape, fallback_score: float) -> None:
        canvas_full_sq = getattr(shape, "_fd6_canvas_full_sq", None)
        if canvas_full_sq is None:
            self.canvas_full_sq = max(0.0, fallback_score * fallback_score * self.canvas_norm)
            self.rms = fallback_score
        else:
            self.canvas_full_sq = max(0.0, float(canvas_full_sq))
            self.rms = float(getattr(shape, "_fd6_rms_score", fallback_score))
        if self.quality_context is not None:
            weighted_full_sq = getattr(shape, "_fd6_weighted_full_sq", None)
            gradient_full_error = getattr(shape, "_fd6_gradient_full_error", None)
            if weighted_full_sq is None or gradient_full_error is None:
                self.weighted_full_sq, self.weighted_norm = precompute_weighted_rgb_error(
                    self.canvas,
                    self.target,
                    self.quality_context,
                    self._line_guide_score_factor,
                )
                if self.quality_context.gradient_weight > 0.0:
                    self.gradient_full_error, self.gradient_norm = precompute_gradient_error(
                        self.canvas,
                        self.quality_context,
                        self._line_guide_score_factor,
                    )
                else:
                    self.gradient_full_error = 0.0
                    self.gradient_norm = 0.0
            else:
                self.weighted_full_sq = max(0.0, float(weighted_full_sq))
                self.gradient_full_error = max(0.0, float(gradient_full_error))

    def seed_shapes(self, shapes: list[Shape]) -> None:
        self._sync_cpu_canvas_from_gpu()
        for shape in shapes:
            apply_shape_inplace(self.canvas, shape, self.alpha_mask)
            self.shapes.append(shape)
        self._shape_count = len(self.shapes)
        self._refresh_score_state()
        if self._gpu_runtime is not None:
            self._gpu_runtime.sync_full_canvas(self.canvas)
            self._gpu_canvas_dirty = False
            self._gpu_dirty_bbox = None

    def _reset_canvas_to_initial(self) -> None:
        self._sync_cpu_canvas_from_gpu()
        self.canvas[:] = self._initial_canvas
        self._refresh_score_state()
        if self._gpu_runtime is not None:
            self._gpu_runtime.sync_full_canvas(self.canvas)
            self._gpu_canvas_dirty = False
            self._gpu_dirty_bbox = None

    def _score_candidate_shapes(
        self,
        candidates: list[Shape],
    ) -> tuple[np.ndarray, np.ndarray]:
        if not candidates:
            return np.zeros(0, dtype=np.float32), np.zeros((0, 4), dtype=np.int32)
        if self._gpu_runtime is not None and self.quality_context is not None:
            return self._gpu_score_grouped_shapes(candidates, [len(candidates)])
        if self._gpu_runtime is not None and self.quality_context is None:
            try:
                return self._gpu_runtime.score_shapes(candidates, self.canvas_full_sq, self.canvas_norm)
            except (RuntimeError, ValueError):
                pass
        scores = np.empty(len(candidates), dtype=np.float32)
        colors = np.zeros((len(candidates), 4), dtype=np.int32)
        for idx, candidate in enumerate(candidates):
            score, color = self._score_shape(candidate)
            scores[idx] = float(score)
            colors[idx] = np.asarray(color, dtype=np.int32)
        return scores, colors

    def _cache_shape_bbox(self, shape: Shape) -> Shape:
        cached_bbox_metrics(shape, self.w, self.h)
        return shape

    def _find_best_candidate(self, candidates: list[Shape]) -> tuple[float, Shape | None]:
        if self._gpu_runtime is not None and self.quality_context is None:
            score_torch = getattr(self._gpu_runtime, "score_shapes_torch", None)
            if score_torch is not None and candidates:
                try:
                    scores, colors = score_torch(candidates, self.canvas_full_sq, self.canvas_norm)
                    torch = self._gpu_runtime._torch
                    best_score_t, best_idx_t = torch.min(scores, dim=0)
                    best_color_t = colors.index_select(0, best_idx_t.view(1)).squeeze(0).to(dtype=torch.float32)
                    best_meta = torch.cat(
                        (
                            best_score_t.view(1),
                            best_idx_t.to(dtype=torch.float32).view(1),
                            best_color_t,
                        ),
                    ).to(device="cpu", dtype=torch.float32).tolist()
                    best_score = float(best_meta[0])
                    if np.isfinite(best_score):
                        best_idx = int(best_meta[1])
                        best_color = best_meta[2:6]
                        best_shape = candidates[best_idx]
                        best_shape.color = tuple(int(v) for v in best_color)
                        return best_score, best_shape
                    return float("inf"), None
                except (RuntimeError, ValueError):
                    pass
        scores, colors = self._score_candidate_shapes(candidates)
        if scores.size == 0:
            return float("inf"), None
        best_idx = int(np.argmin(scores))
        best_score = float(scores[best_idx])
        if not np.isfinite(best_score):
            return float("inf"), None
        best_shape = candidates[best_idx]
        best_shape.color = tuple(int(v) for v in colors[best_idx])
        return best_score, best_shape

    def _commit_shape(self, shape: Shape, score: float) -> None:
        bbox = apply_shape_inplace(self.canvas, shape, self.alpha_mask)
        self._update_score_state_from_shape(shape, score)
        if self._gpu_runtime is not None:
            self._gpu_runtime.sync_region(self.canvas, bbox)
            self._gpu_canvas_dirty = False
            self._gpu_dirty_bbox = None

    def _refine_mutation_budget(self) -> int:
        return max(4, min(16, max(1, self.profile.mutated_samples // 12)))

    def _refine_existing_shape(
        self,
        shape: Shape,
        rng: random.Random,
        mutation_budget: int,
    ) -> tuple[float, Shape | None]:
        base_shape = self._cache_shape_bbox(shape.with_color(shape.color))
        remaining = max(0, mutation_budget)
        batch_cap = 16 if self._gpu_runtime is not None else 8
        batch_size = max(4, min(batch_cap, remaining if remaining > 0 else 4))
        if remaining > 0:
            count = min(batch_size, remaining)
            initial_candidates = [base_shape]
            initial_candidates.extend(
                self._cache_shape_bbox(base_shape.mutate(rng, self.w, self.h))
                for _ in range(count)
            )
            best_score, best_shape = self._find_best_candidate(initial_candidates)
            remaining -= count
        else:
            best_score, best_shape = self._find_best_candidate([base_shape])
        if best_shape is None:
            return float("inf"), None
        no_improve = 0
        stall_limit = max(batch_size, remaining // 2) if remaining > 0 else batch_size
        while remaining > 0:
            count = min(batch_size, remaining)
            candidates = [self._cache_shape_bbox(best_shape.mutate(rng, self.w, self.h)) for _ in range(count)]
            candidate_score, candidate_shape = self._find_best_candidate(candidates)
            if candidate_shape is not None and candidate_score < best_score:
                best_score = candidate_score
                best_shape = candidate_shape
                no_improve = 0
            else:
                no_improve += count
                if no_improve >= stall_limit:
                    break
            remaining -= count
        return best_score, best_shape

    def _refine_existing_shapes(self) -> Iterable[EngineEvent]:
        passes = max(0, int(self.profile.refine_passes))
        if passes <= 0:
            return
        self._flush_pending_gpu_shapes()
        if not self.shapes:
            return
        mutation_budget = self._refine_mutation_budget()
        for _ in range(passes):
            self._sync_cpu_canvas_from_gpu()
            previous_canvas = self.canvas.copy()
            previous_shapes = [shape.with_color(shape.color) for shape in self.shapes]
            previous_rms = self.rms
            previous_canvas_full_sq = self.canvas_full_sq
            self._reset_canvas_to_initial()
            refined_shapes: list[Shape] = []
            local_rng = random.Random(self.rng.randint(0, 2**31 - 1))
            for idx, shape in enumerate(previous_shapes):
                budget = 0 if self._stop else mutation_budget
                refined_score, refined_shape = self._refine_existing_shape(shape, local_rng, budget)
                if refined_shape is None or not np.isfinite(refined_score):
                    self.canvas[:] = previous_canvas
                    self.shapes = previous_shapes
                    self._refresh_score_state()
                    self.canvas_full_sq = previous_canvas_full_sq
                    self.rms = previous_rms
                    if self._gpu_runtime is not None:
                        self._gpu_runtime.sync_full_canvas(self.canvas)
                        self._gpu_canvas_dirty = False
                        self._gpu_dirty_bbox = None
                    return
                self._commit_shape(refined_shape, refined_score)
                refined_shapes.append(refined_shape)
                if self._stop:
                    tail = previous_shapes[idx + 1:]
                    for tail_shape in tail:
                        tail_score, tail_refined = self._refine_existing_shape(tail_shape, local_rng, 0)
                        if tail_refined is None or not np.isfinite(tail_score):
                            break
                        self._commit_shape(tail_refined, tail_score)
                        refined_shapes.append(tail_refined)
                    break
            if len(refined_shapes) != len(previous_shapes) or self.rms >= previous_rms - 1e-6:
                self.canvas[:] = previous_canvas
                self.shapes = previous_shapes
                self._refresh_score_state()
                self.canvas_full_sq = previous_canvas_full_sq
                self.rms = previous_rms
                if self._gpu_runtime is not None:
                    self._gpu_runtime.sync_full_canvas(self.canvas)
                    self._gpu_canvas_dirty = False
                    self._gpu_dirty_bbox = None
                break
            self.shapes = refined_shapes
            self._shape_count = len(self.shapes)
            if self.profile.preview_every:
                yield EngineEvent(
                    kind="preview",
                    shape_count=self._shape_count,
                    rms=self.rms,
                    canvas=self._preview_canvas(),
                )

    def _cpu_search(self, types: list[str], n_random: int, n_mutate: int) -> tuple[float, Shape | None]:
        if self._executor is not None and self.quality_context is not None:
            self._refresh_score_state()
        args_list = [
            (
                types,
                n_random,
                n_mutate,
                self.w,
                self.h,
                self.rng.randint(0, 2**31 - 1),
                self.edge_candidate_ratio,
                int(getattr(self.profile, "edge_rerank_top_k", 24)),
                self.canvas_full_sq,
                self.canvas_norm,
                self.weighted_full_sq,
                self.weighted_norm,
                self.gradient_full_error,
                self.gradient_norm,
                self._line_guide_score_factor,
            )
            for _ in range(self._n_workers)
        ]
        best_score = float("inf")
        best_shape: Shape | None = None
        if self._executor is None:
            for args in args_list:
                score, color, shape = _independent_search(
                    self.canvas,
                    self.target,
                    self.alpha_mask,
                    self.quality_context,
                    args,
                )
                if shape is not None and score < best_score:
                    shape.color = color
                    best_score, best_shape = score, shape
            return best_score, best_shape
        for score, color, shape in self._executor.map(_worker_independent_search, args_list):
            if shape is not None and score < best_score:
                shape.color = color
                best_score, best_shape = score, shape
        return best_score, best_shape

    def _gpu_rotated_ellipse_search(self, n_random: int, n_mutate: int) -> tuple[float, Shape | None]:
        if self._gpu_runtime is None:
            return float("inf"), None
        refine_stage_count = max(1, min(5, (max(1, n_mutate) + 39) // 40))
        long_side = max(self.w, self.h)
        if long_side >= 900:
            chain_count = min(max(1, self._n_workers), 5)
            random_count = max(1, min(n_random, 320))
        else:
            chain_count = self._n_workers
            random_count = max(1, min(n_random, 192))
        seed = self.rng.randint(0, 2**31 - 1)
        if self.alpha_mask is None:
            best_score, best_pack = self._gpu_runtime.search_rotated_ellipse_device(
                chain_count=chain_count,
                random_count=random_count,
                canvas_full_sq=self.canvas_full_sq,
                canvas_norm=self.canvas_norm,
                seed=seed,
                refine_stage_count=refine_stage_count,
            )
            if best_pack is None or not np.isfinite(best_score):
                return float("inf"), None
            return best_score, _GpuRotatedEllipsePack(best_pack)
        best_score, best_params, best_color = self._gpu_runtime.search_rotated_ellipse(
            chain_count=chain_count,
            random_count=random_count,
            canvas_full_sq=self.canvas_full_sq,
            canvas_norm=self.canvas_norm,
            seed=seed,
            refine_stage_count=refine_stage_count,
        )
        if best_params is None or best_color is None or not np.isfinite(best_score):
            return float("inf"), None
        best_x, best_y, best_rx, best_ry, best_angle = best_params
        return best_score, RotatedEllipse(
            color=best_color,
            x=best_x,
            y=best_y,
            rx=best_rx,
            ry=best_ry,
            angle=best_angle,
        )

    def _gpu_apply_scores(
        self,
        chains: list[_GpuChainState],
        owners: list[int],
        candidates: list[Shape],
        scores: object,
        colors: object,
        winner_indices: object | None = None,
        attempt_counts: list[int] | None = None,
    ) -> None:
        scores_seq = scores.detach().to(device="cpu").tolist() if hasattr(scores, "detach") else scores
        colors_seq = colors.detach().to(device="cpu").tolist() if hasattr(colors, "detach") else colors
        if winner_indices is None:
            winner_seq = None
        elif hasattr(winner_indices, "detach"):
            winner_seq = winner_indices.to(device="cpu").tolist()
        elif hasattr(winner_indices, "tolist"):
            winner_seq = winner_indices.tolist()
        else:
            winner_seq = winner_indices
        for pos, chain_idx in enumerate(owners):
            score = float(scores_seq[pos])
            chain = chains[chain_idx]
            if np.isfinite(score) and score < chain.best_score:
                candidate_pos = pos if winner_seq is None else int(winner_seq[pos])
                candidate = candidates[candidate_pos]
                candidate.color = tuple(int(v) for v in colors_seq[pos])
                chain.best_score = score
                chain.best_shape = candidate
                chain.no_improve = 0
            else:
                misses = 1 if attempt_counts is None else max(1, int(attempt_counts[pos]))
                chain.no_improve += misses

    def _gpu_mutation_group_size(self, stall_limit: int) -> int:
        long_side = max(1, max(self.h, self.w))
        work = max(1, self._n_workers)
        group = int(round(102_400 / (long_side * work)))
        return max(4, min(64, stall_limit, group))

    def _gpu_base_score_shapes(self, candidates: list[Shape]) -> tuple[object, object]:
        if self._gpu_runtime is None:
            return np.zeros(0, dtype=np.float32), np.zeros((0, 4), dtype=np.int32)
        score_torch = getattr(self._gpu_runtime, "score_shapes_torch", None)
        if score_torch is None:
            return self._gpu_runtime.score_shapes(candidates, self.canvas_full_sq, self.canvas_norm)
        return score_torch(candidates, self.canvas_full_sq, self.canvas_norm)

    def _gpu_score_shapes(self, candidates: list[Shape]) -> tuple[object, object]:
        if self.quality_context is not None:
            return self._gpu_score_grouped_shapes(candidates, [len(candidates)])
        return self._gpu_base_score_shapes(candidates)

    def _gpu_score_grouped_shapes(
        self,
        candidates: list[Shape],
        group_sizes: list[int],
    ) -> tuple[object, object]:
        if self._gpu_runtime is None or not candidates:
            return np.zeros(0, dtype=np.float32), np.zeros((0, 4), dtype=np.int32)
        if self.quality_context is None:
            return self._gpu_base_score_shapes(candidates)
        try:
            base_scores, base_colors = self._gpu_base_score_shapes(candidates)
        except (RuntimeError, ValueError):
            base_scores = np.full(len(candidates), float("inf"), dtype=np.float32)
            base_colors = np.zeros((len(candidates), 4), dtype=np.int32)
            for idx, candidate in enumerate(candidates):
                score, color = score_shape(
                    candidate,
                    self.canvas,
                    self.target,
                    self.alpha_mask,
                    canvas_full_sq=self.canvas_full_sq,
                    canvas_norm=self.canvas_norm,
                )
                base_scores[idx] = float(score)
                base_colors[idx] = np.asarray(color, dtype=np.int32)
        if (
            hasattr(base_scores, "detach")
            and hasattr(base_colors, "to")
            and getattr(self._gpu_runtime, "score_shapes_quality_torch", None) is not None
            and self.alpha_mask is None
        ):
            try:
                return self._gpu_score_grouped_shapes_torch(candidates, group_sizes, base_scores, base_colors)
            except (RuntimeError, ValueError, TypeError, AttributeError):
                pass
        if hasattr(base_scores, "detach"):
            base_scores_np = base_scores.detach().to(device="cpu").numpy()
        else:
            base_scores_np = np.asarray(base_scores, dtype=np.float32)
        if hasattr(base_colors, "detach"):
            base_colors_np = base_colors.detach().to(device="cpu").numpy().astype(np.int32)
        else:
            base_colors_np = np.asarray(base_colors, dtype=np.int32)
        scores = np.full(len(candidates), float("inf"), dtype=np.float32)
        colors = np.zeros((len(candidates), 4), dtype=np.int32)
        if base_colors_np.shape == colors.shape:
            colors[:] = base_colors_np
        top_k = max(1, int(getattr(self.profile, "edge_rerank_top_k", 24)))
        chosen_indices: list[int] = []
        chosen_seen: set[int] = set()
        offset = 0
        for size in group_sizes:
            if size <= 0:
                offset += size
                continue
            end = min(len(candidates), offset + size)
            segment = base_scores_np[offset:end]
            finite = np.flatnonzero(np.isfinite(segment))
            if finite.size == 0:
                offset += size
                continue
            chosen = finite[np.argsort(segment[finite])[:min(top_k, finite.size)]]
            edge_chosen = np.asarray(
                [
                    local_idx
                    for local_idx in finite
                    if getattr(candidates[offset + int(local_idx)], "_fd6_edge_candidate", False)
                ],
                dtype=np.int64,
            )
            if edge_chosen.size:
                edge_order = np.argsort(segment[edge_chosen])[:min(top_k, edge_chosen.size)]
                edge_chosen = edge_chosen[edge_order]
                chosen = np.unique(np.concatenate((chosen, edge_chosen)))
            for local_idx in chosen:
                idx = offset + int(local_idx)
                if idx not in chosen_seen:
                    chosen_indices.append(idx)
                    chosen_seen.add(idx)
            offset += size
        if chosen_indices:
            quality_score = getattr(self._gpu_runtime, "score_shapes_quality", None)
            if quality_score is not None and self.alpha_mask is None:
                try:
                    selected = [candidates[idx] for idx in chosen_indices]
                    selected_colors = base_colors_np[np.asarray(chosen_indices, dtype=np.int64)]
                    quality_scores, quality_colors = quality_score(
                        selected,
                        selected_colors,
                        self.weighted_full_sq,
                        self.weighted_norm,
                        self.gradient_full_error,
                        self.gradient_norm,
                    )
                    quality_scores_np = np.asarray(quality_scores, dtype=np.float32)
                    quality_colors_np = np.asarray(quality_colors, dtype=np.int32)
                    for out_pos, idx in enumerate(chosen_indices):
                        scores[idx] = float(quality_scores_np[out_pos])
                        colors[idx] = quality_colors_np[out_pos]
                    return scores, colors
                except (RuntimeError, ValueError, TypeError, AttributeError):
                    pass
            for idx in chosen_indices:
                fixed_color = tuple(int(v) for v in base_colors_np[idx])
                score, color = self._score_shape(
                    candidates[idx],
                    fixed_color=fixed_color,
                    base_rms=float(base_scores_np[idx]),
                    base_canvas_full_sq=getattr(candidates[idx], "_fd6_canvas_full_sq", None),
                )
                scores[idx] = float(score)
                colors[idx] = np.asarray(color, dtype=np.int32)
        return scores, colors

    def _gpu_score_grouped_shapes_torch(
        self,
        candidates: list[Shape],
        group_sizes: list[int],
        base_scores: object,
        base_colors: object,
    ) -> tuple[object, object]:
        if self._gpu_runtime is None:
            return np.zeros(0, dtype=np.float32), np.zeros((0, 4), dtype=np.int32)
        torch = self._gpu_runtime._torch
        device = self._gpu_runtime.device
        count = len(candidates)
        scores = torch.full((count,), float("inf"), device=device, dtype=torch.float32)
        colors = torch.zeros((count, 4), device=device, dtype=torch.int32)
        if tuple(base_colors.shape) == (count, 4):
            colors.copy_(base_colors.to(device=device, dtype=torch.int32))
        top_k = max(1, int(getattr(self.profile, "edge_rerank_top_k", 24)))
        edge_flags_py = [bool(getattr(candidate, "_fd6_edge_candidate", False)) for candidate in candidates]
        edge_flags = (
            torch.as_tensor(edge_flags_py, device=device, dtype=torch.bool)
            if any(edge_flags_py)
            else None
        )
        chosen_chunks = []
        offset = 0
        for size in group_sizes:
            if size <= 0:
                offset += size
                continue
            end = min(count, offset + size)
            segment = base_scores[offset:end]
            finite = torch.nonzero(torch.isfinite(segment), as_tuple=False).flatten()
            finite_count = int(finite.numel())
            if finite_count == 0:
                offset += size
                continue
            finite_scores = segment.index_select(0, finite)
            try:
                order = torch.argsort(finite_scores, stable=True)
            except TypeError:
                order = torch.argsort(finite_scores)
            selected_count = min(top_k, finite_count)
            selected = finite.index_select(0, order[:selected_count])
            if edge_flags is not None:
                group_edge_flags = edge_flags[offset:end]
                edge_selected = finite.masked_select(group_edge_flags.index_select(0, finite))
                if int(edge_selected.numel()) > 0:
                    edge_scores = segment.index_select(0, edge_selected)
                    edge_count = min(top_k, int(edge_selected.numel()))
                    try:
                        edge_order = torch.argsort(edge_scores, stable=True)[:edge_count]
                    except TypeError:
                        edge_order = torch.argsort(edge_scores)[:edge_count]
                    edge_selected = edge_selected.index_select(0, edge_order)
                    selected = torch.unique(torch.cat((selected, edge_selected)), sorted=True)
            chosen_chunks.append(selected + offset)
            offset += size
        if not chosen_chunks:
            return scores, colors
        selected_indices = torch.cat(chosen_chunks).to(device=device, dtype=torch.long)
        chosen_indices = selected_indices.to(device="cpu", dtype=torch.long).tolist()
        selected = [candidates[idx] for idx in chosen_indices]
        selected_colors = base_colors.index_select(0, selected_indices).to(device=device, dtype=torch.int32)
        quality_score = getattr(self._gpu_runtime, "score_shapes_quality_torch")
        quality_scores, quality_colors = quality_score(
            selected,
            selected_colors,
            self.weighted_full_sq,
            self.weighted_norm,
            self.gradient_full_error,
            self.gradient_norm,
        )
        scores.index_copy_(0, selected_indices, quality_scores.to(device=device, dtype=torch.float32))
        colors.index_copy_(0, selected_indices, quality_colors.to(device=device, dtype=torch.int32))
        return scores, colors

    def _gpu_pick_group_winners(
        self,
        scores: object,
        colors: object,
        group_sizes: list[int],
    ) -> tuple[object, object, object]:
        if self._gpu_runtime is None or not group_sizes:
            return [], np.zeros(0, dtype=np.float32), np.zeros((0, 4), dtype=np.int32)
        if hasattr(scores, "index_select"):
            torch = self._gpu_runtime._torch
            positive_sizes = [size for size in group_sizes if size > 0]
            if positive_sizes and len(positive_sizes) == len(group_sizes) and len(set(positive_sizes)) == 1:
                size = positive_sizes[0]
                grouped_scores = scores.view(len(group_sizes), size)
                best_local = torch.argmin(grouped_scores, dim=1)
                base = torch.arange(len(group_sizes), device=self._gpu_runtime.device, dtype=torch.long) * size
                winners_t = base + best_local
                best_scores = scores.index_select(0, winners_t)
                best_colors = colors.index_select(0, winners_t)
                return winners_t, best_scores, best_colors
            grouped_rows: dict[int, list[tuple[int, int]]] = {}
            offset = 0
            result_pos = 0
            for size in group_sizes:
                if size <= 0:
                    offset += size
                    continue
                grouped_rows.setdefault(size, []).append((result_pos, offset))
                offset += size
                result_pos += 1
            if not grouped_rows:
                return [], np.zeros(0, dtype=np.float32), np.zeros((0, 4), dtype=np.int32)
            winners_t = torch.empty(result_pos, device=self._gpu_runtime.device, dtype=torch.long)
            for size, rows in grouped_rows.items():
                positions_t = torch.as_tensor([row[0] for row in rows], device=self._gpu_runtime.device, dtype=torch.long)
                base_t = torch.as_tensor([row[1] for row in rows], device=self._gpu_runtime.device, dtype=torch.long)
                if size == 1:
                    winners_t.index_copy_(0, positions_t, base_t)
                    continue
                local_idx = base_t.unsqueeze(1) + torch.arange(size, device=self._gpu_runtime.device, dtype=torch.long).view(1, size)
                grouped_scores = scores.index_select(0, local_idx.reshape(-1)).view(len(rows), size)
                best_local = torch.argmin(grouped_scores, dim=1)
                winners_t.index_copy_(0, positions_t, base_t + best_local)
            best_scores = scores.index_select(0, winners_t)
            best_colors = colors.index_select(0, winners_t)
            return winners_t, best_scores, best_colors
        scores_np = np.asarray(scores)
        colors_np = np.asarray(colors)
        winners = []
        best_scores = []
        best_colors = []
        offset = 0
        for size in group_sizes:
            if size <= 0:
                offset += size
                continue
            local = scores_np[offset:offset + size]
            best_local = int(np.argmin(local))
            best_pos = offset + best_local
            winners.append(best_pos)
            best_scores.append(float(scores_np[best_pos]))
            best_colors.append(colors_np[best_pos])
            offset += size
        if not winners:
            return [], np.zeros(0, dtype=np.float32), np.zeros((0, 4), dtype=np.int32)
        return np.asarray(winners, dtype=np.int64), np.asarray(best_scores, dtype=np.float32), np.asarray(best_colors, dtype=np.int32)

    def _gpu_search_generic(self, types: list[str], n_random: int, n_mutate: int) -> tuple[float, Shape | None]:
        if self._gpu_runtime is None:
            return float("inf"), None
        if len(types) == 1 and types[0] == "rotated_ellipse" and self.quality_context is None:
            return self._gpu_rotated_ellipse_search(n_random, n_mutate)
        chains = [
            _GpuChainState(rng=random.Random(self.rng.randint(0, 2**31 - 1)))
            for _ in range(self._n_workers)
        ]
        random_count = max(1, n_random)
        random_candidates: list[Shape] = []
        for chain in chains:
            for _ in range(random_count):
                random_candidates.append(self._cache_shape_bbox(
                    _search_shape(chain.rng, self.w, self.h, types, self.quality_context, self.edge_candidate_ratio),
                ))
        random_group_sizes = [random_count for _ in chains]
        random_scores, random_colors = self._gpu_score_grouped_shapes(random_candidates, random_group_sizes)
        winner_indices, best_scores, best_colors = self._gpu_pick_group_winners(
            random_scores, random_colors, random_group_sizes,
        )
        self._gpu_apply_scores(
            chains,
            list(range(len(chains))),
            random_candidates,
            best_scores,
            best_colors,
            winner_indices=winner_indices,
        )
        cap = max(1, n_mutate)
        stall_limit = max(20, cap // 4)
        group_size = self._gpu_mutation_group_size(stall_limit)
        remaining = [cap for _ in chains]
        for chain in chains:
            chain.no_improve = 0
        while True:
            grouped_chain_indices: dict[int, list[int]] = {}
            for chain_idx, chain in enumerate(chains):
                if chain.best_shape is None or chain.no_improve >= stall_limit or remaining[chain_idx] <= 0:
                    continue
                count = min(group_size, remaining[chain_idx], max(0, stall_limit - chain.no_improve))
                if count <= 0:
                    continue
                grouped_chain_indices.setdefault(count, []).append(chain_idx)
                remaining[chain_idx] -= count
            if not grouped_chain_indices:
                break
            for count in sorted(grouped_chain_indices, reverse=True):
                chain_indices = grouped_chain_indices[count]
                candidates: list[Shape] = []
                active_chain_indices: list[int] = []
                for chain_idx in chain_indices:
                    base_shape = chains[chain_idx].best_shape
                    if base_shape is None:
                        continue
                    active_chain_indices.append(chain_idx)
                    for _ in range(count):
                        candidates.append(self._cache_shape_bbox(base_shape.mutate(chains[chain_idx].rng, self.w, self.h)))
                if not candidates:
                    continue
                group_sizes = [count for _ in active_chain_indices]
                scores, colors = self._gpu_score_grouped_shapes(candidates, group_sizes)
                winner_indices, best_scores, best_colors = self._gpu_pick_group_winners(scores, colors, group_sizes)
                self._gpu_apply_scores(
                    chains,
                    active_chain_indices,
                    candidates,
                    best_scores,
                    best_colors,
                    winner_indices=winner_indices,
                    attempt_counts=group_sizes,
                )
        best_score = float("inf")
        best_shape: Shape | None = None
        for chain in chains:
            if chain.best_shape is not None and chain.best_score < best_score:
                best_score = chain.best_score
                best_shape = chain.best_shape
        return best_score, best_shape

    def _gpu_search(self, types: list[str], n_random: int, n_mutate: int) -> tuple[float, Shape | None]:
        if self._gpu_runtime is None:
            return float("inf"), None
        if len(types) == 1 and types[0] == "rotated_ellipse" and self.quality_context is None:
            return self._gpu_rotated_ellipse_search(n_random, n_mutate)
        return self._gpu_search_generic(types, n_random, n_mutate)

    def _parallel_search(self, types: list[str], n_random: int, n_mutate: int) -> tuple[float, Shape | None]:
        n_random = max(1, n_random)
        n_mutate = max(1, n_mutate)
        if self.compute_backend == "gpu":
            return self._gpu_search(types, n_random, n_mutate)
        return self._cpu_search(types, n_random, n_mutate)

    def run(self) -> Iterable[EngineEvent]:
        p = self.profile
        types = [t for t in p.shape_types if t]
        if not types:
            types = ["rotated_ellipse"]
        save_at = set(p.save_at)
        try:
            consecutive_skips = 0
            max_consecutive_skips = 80
            while self._shape_count < p.stop_at and not self._stop:
                while self._pause and not self._stop:
                    time.sleep(0.05)
                self._refresh_line_guide_factor()
                if self._gpu_runtime is not None:
                    self._gpu_runtime.rotated_ellipse_shape_index = self._shape_count
                self.edge_candidate_ratio = self._effective_edge_candidate_ratio()
                refined_score, refined = self._parallel_search(
                    types, max(1, p.random_samples), max(1, p.mutated_samples),
                )
                if self.alpha_mask is not None:
                    sticker_attempts = 0
                    while sticker_attempts < 5:
                        if refined is not None and refined_score != float("inf"):
                            break
                        refined_score, refined = self._parallel_search(
                            types, max(1, p.random_samples), max(1, p.mutated_samples),
                        )
                        sticker_attempts += 1
                    else:
                        consecutive_skips += 1
                        if consecutive_skips >= max_consecutive_skips:
                            self._flush_pending_gpu_shapes()
                            yield EngineEvent(
                                kind="done",
                                shape_count=self._shape_count,
                                rms=self.rms,
                                canvas=self._preview_canvas(),
                                message=(
                                    f"Stopped early at {self._shape_count} shapes - couldn't "
                                    f"fit any more inside the opaque region after {max_consecutive_skips} "
                                    "consecutive attempts. Try increasing 'Random samples' or "
                                    "enabling smaller shape types."
                                ),
                            )
                            return
                        continue
                    consecutive_skips = 0
                if refined is None or refined_score == float("inf"):
                    continue
                if (
                    self._gpu_runtime is not None
                    and self.quality_context is None
                    and self.alpha_mask is None
                    and isinstance(refined, (RotatedEllipse, _GpuRotatedEllipsePack))
                ):
                    if isinstance(refined, _GpuRotatedEllipsePack):
                        bbox, _ = self._gpu_runtime.apply_rotated_ellipse_pack(refined.pack, return_region=False)
                        self._pending_gpu_shape_packs.append(refined.pack)
                    else:
                        bbox, _ = self._gpu_runtime.apply_rotated_ellipse(refined, return_region=False)
                    self._gpu_canvas_dirty = True
                    self._gpu_dirty_bbox = _merge_bbox(self._gpu_dirty_bbox, bbox)
                else:
                    self._sync_cpu_canvas_from_gpu()
                    bbox = apply_shape_inplace(self.canvas, refined, self.alpha_mask)
                self._update_score_state_from_shape(refined, refined_score)
                if self._gpu_runtime is not None and not (
                    self.quality_context is None
                    and self.alpha_mask is None
                    and isinstance(refined, (RotatedEllipse, _GpuRotatedEllipsePack))
                ):
                    self._gpu_runtime.sync_region(self.canvas, bbox)
                    self._gpu_canvas_dirty = False
                if not isinstance(refined, _GpuRotatedEllipsePack):
                    self.shapes.append(refined)
                self._shape_count += 1
                count = self._shape_count
                yield EngineEvent(kind="shape_committed", shape_count=count, rms=self.rms)
                if p.preview_every and (count % p.preview_every == 0):
                    yield EngineEvent(kind="preview", shape_count=count, rms=self.rms, canvas=self._preview_canvas())
                if count in save_at or (p.save_every and count % p.save_every == 0):
                    self._flush_pending_gpu_shapes()
                    yield EngineEvent(kind="checkpoint", shape_count=count, rms=self.rms)
            yield from self._refine_existing_shapes()
            self._flush_pending_gpu_shapes()
            yield EngineEvent(kind="done", shape_count=self._shape_count, rms=self.rms, canvas=self._preview_canvas())
        except Exception as exc:
            yield EngineEvent(kind="error", message=f"{type(exc).__name__}: {exc}")
        finally:
            self._shutdown()

    def _shutdown(self) -> None:
        try:
            if self._executor is not None:
                self._executor.shutdown(wait=False, cancel_futures=True)
                self._executor = None
        except Exception:
            pass
        try:
            if self._gpu_runtime is not None:
                self._gpu_runtime.close()
                self._gpu_runtime = None
        except Exception:
            pass
        try:
            if self._canvas_shm is not None:
                self._canvas_shm.close()
                self._canvas_shm.unlink()
                self._canvas_shm = None
        except Exception:
            pass

    def __del__(self) -> None:
        try:
            self._shutdown()
        except Exception:
            pass
