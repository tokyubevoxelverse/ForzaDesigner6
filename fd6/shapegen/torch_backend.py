from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import gc
import importlib
import math
import os
from pathlib import Path
import site as site_module
import sys

import numpy as np

from fd6.shapegen.scoring import STICKER_OVERLAP_MIN
from fd6.shapegen.shapes import Shape
from fd6.shapegen.shapes.base import cached_bbox_metrics


_UNSET = object()
_TORCH = _UNSET
_CUPY = _UNSET
_CUPY_SCORE_FIXED_KERNEL = _UNSET
_CUPY_APPLY_FIXED_KERNEL = _UNSET
_CUPY_SCORE_FIXED_BLOCK_SIZE = 1024
_EXTERNAL_PACKAGE_PATHS_READY = False
_EMPTY_RGB_REGION = np.zeros((0, 0, 3), dtype=np.uint8)
_GPU_BATCH_MIN_PIXELS = 384_000
_GPU_BATCH_MAX_PIXELS = 96_000_000
_GPU_BATCH_FALLBACK_PIXELS = 8_000_000
_GPU_BATCH_BYTES_PER_PIXEL = 160
_GPU_BATCH_MEMORY_FRACTION = 0.45
_GPU_BATCH_RESERVED_BYTES = 768 * 1024 * 1024
_GPU_QUALITY_BATCH_DIVISOR = 6


def _safe_max_batch_pixels(torch_module) -> int:
    try:
        free_mem, total_mem = torch_module.cuda.mem_get_info()
        reserve = max(_GPU_BATCH_RESERVED_BYTES, int(total_mem * 0.15))
        usable = int(free_mem * _GPU_BATCH_MEMORY_FRACTION)
        if free_mem > reserve:
            usable = min(usable, int(free_mem - reserve))
        pixels = usable // _GPU_BATCH_BYTES_PER_PIXEL
        return int(max(_GPU_BATCH_MIN_PIXELS, min(_GPU_BATCH_MAX_PIXELS, pixels)))
    except Exception:
        return _GPU_BATCH_FALLBACK_PIXELS


def _prepare_external_package_paths() -> None:
    global _EXTERNAL_PACKAGE_PATHS_READY
    if _EXTERNAL_PACKAGE_PATHS_READY:
        return
    _EXTERNAL_PACKAGE_PATHS_READY = True
    candidate_paths: list[str] = []
    try:
        candidate_paths.extend(site_module.getsitepackages())
    except Exception:
        pass
    try:
        user_site = site_module.getusersitepackages()
    except Exception:
        user_site = None
    if user_site:
        candidate_paths.append(user_site)
    for raw_path in candidate_paths:
        if not raw_path:
            continue
        package_path = str(Path(raw_path))
        if package_path in sys.path:
            continue
        if not Path(package_path).exists():
            continue
        sys.path.append(package_path)


def _load_torch():
    global _TORCH
    if _TORCH is _UNSET:
        _prepare_external_package_paths()
        try:
            torch_module = importlib.import_module("torch")
        except Exception:
            torch_module = None
        _TORCH = torch_module
    return _TORCH


def _rotated_ellipse_extent_xy(x_radius: float, y_radius: float, angle_degrees: float) -> tuple[float, float]:
    angle_radians = math.radians(angle_degrees)
    cos_a = abs(math.cos(angle_radians))
    sin_a = abs(math.sin(angle_radians))
    ext_x = math.sqrt((x_radius * cos_a) ** 2 + (y_radius * sin_a) ** 2)
    ext_y = math.sqrt((x_radius * sin_a) ** 2 + (y_radius * cos_a) ** 2)
    return ext_x, ext_y


def _get_circle_radius_sq(shape: Shape) -> float:
    cache = getattr(shape, "_fd6_circle_geom_cache", None)
    radius = float(shape.r)
    if isinstance(cache, tuple) and len(cache) == 2 and cache[0] == radius:
        return cache[1]
    radius_sq = float(max(radius, 1e-6) ** 2)
    setattr(shape, "_fd6_circle_geom_cache", (radius, radius_sq))
    return radius_sq


def _get_ellipse_inverse_radii(shape: Shape) -> tuple[float, float]:
    cache = getattr(shape, "_fd6_ellipse_geom_cache", None)
    radius_x = float(shape.rx)
    radius_y = float(shape.ry)
    if isinstance(cache, tuple) and len(cache) == 4 and cache[0] == radius_x and cache[1] == radius_y:
        return cache[2], cache[3]
    radius_x = float(max(radius_x, 1e-6))
    radius_y = float(max(radius_y, 1e-6))
    inv_radius_x_sq = float(1.0 / (radius_x * radius_x))
    inv_radius_y_sq = float(1.0 / (radius_y * radius_y))
    setattr(shape, "_fd6_ellipse_geom_cache", (float(shape.rx), float(shape.ry), inv_radius_x_sq, inv_radius_y_sq))
    return inv_radius_x_sq, inv_radius_y_sq


def _get_rotated_ellipse_metrics(shape: Shape) -> tuple[float, float, float, float]:
    cache = getattr(shape, "_fd6_rotated_ellipse_geom_cache", None)
    radius_x = float(shape.rx)
    radius_y = float(shape.ry)
    angle = float(shape.angle)
    if (
        isinstance(cache, tuple)
        and len(cache) == 7
        and cache[0] == radius_x
        and cache[1] == radius_y
        and cache[2] == angle
    ):
        return cache[3], cache[4], cache[5], cache[6]
    radius_x = float(max(radius_x, 1e-6))
    radius_y = float(max(radius_y, 1e-6))
    angle_radians = float(math.radians(angle))
    inv_radius_x_sq = float(1.0 / (radius_x * radius_x))
    inv_radius_y_sq = float(1.0 / (radius_y * radius_y))
    cos_angle = float(math.cos(angle_radians))
    sin_angle = float(math.sin(angle_radians))
    setattr(
        shape,
        "_fd6_rotated_ellipse_geom_cache",
        (float(shape.rx), float(shape.ry), float(shape.angle), inv_radius_x_sq, inv_radius_y_sq, cos_angle, sin_angle),
    )
    return inv_radius_x_sq, inv_radius_y_sq, cos_angle, sin_angle


def _get_rotated_rectangle_angle_metrics(shape: Shape) -> tuple[float, float]:
    cache = getattr(shape, "_fd6_rotated_rectangle_geom_cache", None)
    angle = float(shape.angle)
    if isinstance(cache, tuple) and len(cache) == 3 and cache[0] == angle:
        return cache[1], cache[2]
    angle_radians = float(math.radians(angle))
    cos_angle = float(math.cos(angle_radians))
    sin_angle = float(math.sin(angle_radians))
    setattr(shape, "_fd6_rotated_rectangle_geom_cache", (float(shape.angle), cos_angle, sin_angle))
    return cos_angle, sin_angle


def _prepare_cupy_cuda_path() -> None:
    if os.name != "nt":
        return
    cuda_path = os.environ.get("CUDA_PATH")
    cuda_root = Path(cuda_path) if cuda_path else None
    if cuda_root is None or not cuda_root.exists():
        search_roots: list[Path] = []
        try:
            for value in site_module.getsitepackages():
                search_roots.append(Path(value))
        except Exception:
            pass
        try:
            user_site = site_module.getusersitepackages()
        except Exception:
            user_site = None
        if user_site:
            search_roots.append(Path(user_site))
        for base in search_roots:
            for rel in ("nvidia/cu13", "nvidia/cu12", "nvidia/cu11"):
                candidate = base / rel
                if candidate.exists():
                    cuda_root = candidate
                    os.environ["CUDA_PATH"] = str(candidate)
                    break
            if cuda_root is not None and cuda_root.exists():
                break
    if cuda_root is None or not cuda_root.exists():
        return
    existing_path = os.environ.get("PATH", "")
    entries = existing_path.split(os.pathsep) if existing_path else []
    for candidate in (cuda_root / "bin" / "x86_64", cuda_root / "bin"):
        candidate_str = str(candidate)
        if candidate.exists() and candidate_str not in entries:
            os.environ["PATH"] = candidate_str + (os.pathsep + existing_path if existing_path else "")
            existing_path = os.environ["PATH"]
            entries.insert(0, candidate_str)


def _load_cupy():
    global _CUPY
    if _CUPY is _UNSET:
        _prepare_external_package_paths()
        _prepare_cupy_cuda_path()
        try:
            cupy_module = importlib.import_module("cupy")
        except Exception:
            cupy_module = None
        _CUPY = cupy_module
    return _CUPY


def _load_cupy_score_fixed_kernel():
    global _CUPY_SCORE_FIXED_KERNEL
    if _CUPY_SCORE_FIXED_KERNEL is _UNSET:
        cupy = _load_cupy()
        if cupy is None:
            _CUPY_SCORE_FIXED_KERNEL = None
        else:
            source = r"""
__device__ __forceinline__ float warp_reduce_sum(float value)
{
    value += __shfl_down_sync(0xffffffffu, value, 16);
    value += __shfl_down_sync(0xffffffffu, value, 8);
    value += __shfl_down_sync(0xffffffffu, value, 4);
    value += __shfl_down_sync(0xffffffffu, value, 2);
    value += __shfl_down_sync(0xffffffffu, value, 1);
    return value;
}

extern "C" __global__ void score_rotated_ellipse_fixed_half(
    const long long* x0_arr,
    const long long* y0_arr,
    const long long* width_arr,
    const long long* height_arr,
    const float* shape_x_arr,
    const float* shape_y_arr,
    const float* rx_arr,
    const float* ry_arr,
    const float* angle_arr,
    const float* target_half_flat,
    const float* target_half_sq_flat,
    const float* canvas_old_sq_flat,
    int image_w,
    const float* canvas_full_sq_ptr,
    float sample_scale,
    float denom,
    int stride,
    float* scores_out,
    int* colors_out)
{
    const int candidate = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    const int warp_count = (blockDim.x + 31) >> 5;
    __shared__ float shared_sum0[32];
    __shared__ float shared_sum1[32];
    __shared__ float shared_sum2[32];
    __shared__ float shared_new_sq[32];
    __shared__ float shared_old_sq[32];
    __shared__ float shared_weight[32];
    __shared__ long long shared_width;
    __shared__ long long shared_height;
    __shared__ long long shared_base_x;
    __shared__ long long shared_base_y;
    __shared__ long long shared_sample_w;
    __shared__ long long shared_sample_count;
    __shared__ float shared_cx;
    __shared__ float shared_cy;
    __shared__ float shared_cos_a;
    __shared__ float shared_sin_a;
    __shared__ float shared_inv_rx_sq;
    __shared__ float shared_inv_ry_sq;

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float new_sq = 0.0f;
    float old_sq = 0.0f;
    float weight = 0.0f;

    if (tid == 0) {
        const long long width = width_arr[candidate];
        const long long height = height_arr[candidate];
        shared_width = width;
        shared_height = height;
        if (width > 0 && height > 0) {
            const float cx = shape_x_arr[candidate];
            const float cy = shape_y_arr[candidate];
            const float rx = rx_arr[candidate] > 1.0e-6f ? rx_arr[candidate] : 1.0e-6f;
            const float ry = ry_arr[candidate] > 1.0e-6f ? ry_arr[candidate] : 1.0e-6f;
            const float angle = angle_arr[candidate];
            shared_base_x = x0_arr[candidate];
            shared_base_y = y0_arr[candidate];
            shared_cx = cx;
            shared_cy = cy;
            shared_cos_a = cosf(angle);
            shared_sin_a = sinf(angle);
            shared_inv_rx_sq = 1.0f / (rx * rx);
            shared_inv_ry_sq = 1.0f / (ry * ry);
            shared_sample_w = (width + stride - 1) / stride;
            shared_sample_count = shared_sample_w * ((height + stride - 1) / stride);
        } else {
            shared_base_x = 0;
            shared_base_y = 0;
            shared_cx = 0.0f;
            shared_cy = 0.0f;
            shared_cos_a = 1.0f;
            shared_sin_a = 0.0f;
            shared_inv_rx_sq = 0.0f;
            shared_inv_ry_sq = 0.0f;
            shared_sample_w = 0;
            shared_sample_count = 0;
        }
    }
    __syncthreads();

    const long long width = shared_width;
    const long long height = shared_height;
    const long long base_x = shared_base_x;
    const long long base_y = shared_base_y;
    const long long sample_w = shared_sample_w;
    const long long sample_count = shared_sample_count;
    const float cx = shared_cx;
    const float cy = shared_cy;
    const float cos_a = shared_cos_a;
    const float sin_a = shared_sin_a;
    const float inv_rx_sq = shared_inv_rx_sq;
    const float inv_ry_sq = shared_inv_ry_sq;

    if (width > 0 && height > 0) {
        for (long long index = tid; index < sample_count; index += blockDim.x) {
            const long long sample_x = (index % sample_w) * stride;
            const long long sample_y = (index / sample_w) * stride;
            const long long px = base_x + sample_x;
            const long long py = base_y + sample_y;
            const float x_rel = (float)px - cx;
            const float y_rel = (float)py - cy;
            const float xr = cos_a * x_rel + sin_a * y_rel;
            const float yr = -sin_a * x_rel + cos_a * y_rel;
            const float ellipse = (xr * xr) * inv_rx_sq + (yr * yr) * inv_ry_sq;
            if (ellipse <= 1.0f) {
                const long long flat_idx = py * (long long)image_w + px;
                const long long flat3 = flat_idx * 3LL;
                sum0 += target_half_flat[flat3 + 0];
                sum1 += target_half_flat[flat3 + 1];
                sum2 += target_half_flat[flat3 + 2];
                new_sq += target_half_sq_flat[flat_idx];
                old_sq += canvas_old_sq_flat[flat_idx];
                weight += 1.0f;
            }
        }
    }

    sum0 = warp_reduce_sum(sum0);
    sum1 = warp_reduce_sum(sum1);
    sum2 = warp_reduce_sum(sum2);
    new_sq = warp_reduce_sum(new_sq);
    old_sq = warp_reduce_sum(old_sq);
    weight = warp_reduce_sum(weight);

    if (lane == 0) {
        shared_sum0[warp_id] = sum0;
        shared_sum1[warp_id] = sum1;
        shared_sum2[warp_id] = sum2;
        shared_new_sq[warp_id] = new_sq;
        shared_old_sq[warp_id] = old_sq;
        shared_weight[warp_id] = weight;
    }
    __syncthreads();

    if (warp_id == 0) {
        sum0 = lane < warp_count ? shared_sum0[lane] : 0.0f;
        sum1 = lane < warp_count ? shared_sum1[lane] : 0.0f;
        sum2 = lane < warp_count ? shared_sum2[lane] : 0.0f;
        new_sq = lane < warp_count ? shared_new_sq[lane] : 0.0f;
        old_sq = lane < warp_count ? shared_old_sq[lane] : 0.0f;
        weight = lane < warp_count ? shared_weight[lane] : 0.0f;
        sum0 = warp_reduce_sum(sum0);
        sum1 = warp_reduce_sum(sum1);
        sum2 = warp_reduce_sum(sum2);
        new_sq = warp_reduce_sum(new_sq);
        old_sq = warp_reduce_sum(old_sq);
        weight = warp_reduce_sum(weight);
    }

    if (tid == 0) {
        const float alpha_scale = 128.0f / 255.0f;
        const float weight_total = weight;
        const float safe_weight = weight_total > 1.0e-6f ? weight_total : 1.0e-6f;
        const float inv_weight = 1.0f / (alpha_scale * safe_weight);
        const float rgb0_f = floorf(fminf(fmaxf(sum0 * inv_weight, 0.0f), 255.0f));
        const float rgb1_f = floorf(fminf(fmaxf(sum1 * inv_weight, 0.0f), 255.0f));
        const float rgb2_f = floorf(fminf(fmaxf(sum2 * inv_weight, 0.0f), 255.0f));
        const float applied0 = alpha_scale * rgb0_f;
        const float applied1 = alpha_scale * rgb1_f;
        const float applied2 = alpha_scale * rgb2_f;
        const float inside_new_sq =
            new_sq
            - 2.0f * (
                applied0 * sum0
                + applied1 * sum1
                + applied2 * sum2
            )
            + weight_total * (
                applied0 * applied0
                + applied1 * applied1
                + applied2 * applied2
            );
        const float canvas_base_sq = canvas_full_sq_ptr[0] / sample_scale;
        const float total_sq = canvas_base_sq - old_sq + inside_new_sq;
        const int color_base = candidate * 4;
        colors_out[color_base + 0] = (int)rgb0_f;
        colors_out[color_base + 1] = (int)rgb1_f;
        colors_out[color_base + 2] = (int)rgb2_f;
        colors_out[color_base + 3] = 128;
        if (width <= 0 || height <= 0 || weight_total < 0.5f) {
            scores_out[candidate] = __int_as_float(0x7f800000);
        } else {
            scores_out[candidate] = sqrtf(fmaxf(total_sq, 0.0f) / denom);
        }
    }
}
"""
            _CUPY_SCORE_FIXED_KERNEL = cupy.RawKernel(
                source,
                "score_rotated_ellipse_fixed_half",
                options=("--std=c++11",),
            )
    return _CUPY_SCORE_FIXED_KERNEL


def _load_cupy_apply_fixed_kernel():
    global _CUPY_APPLY_FIXED_KERNEL
    if _CUPY_APPLY_FIXED_KERNEL is _UNSET:
        cupy = _load_cupy()
        if cupy is None:
            _CUPY_APPLY_FIXED_KERNEL = None
        else:
            source = r"""
extern "C" __global__ void apply_rotated_ellipse_fixed_half(
    float* canvas_flat,
    const float* target_flat,
    float* canvas_minus_target_flat,
    float* target_minus_half_canvas_flat,
    float* canvas_minus_target_sqsum_flat,
    float* target_minus_half_sqsum_flat,
    int image_w,
    int x0,
    int y0,
    int width,
    int height,
    float cx,
    float cy,
    float rx,
    float ry,
    float angle,
    int src0,
    int src1,
    int src2)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int total = width * height;
    if (idx >= total) {
        return;
    }
    const int lx = idx % width;
    const int ly = idx / width;
    const int px = x0 + lx;
    const int py = y0 + ly;
    const float x_rel = (float)px - cx;
    const float y_rel = (float)py - cy;
    const float cos_a = cosf(angle);
    const float sin_a = sinf(angle);
    const float xr = cos_a * x_rel + sin_a * y_rel;
    const float yr = -sin_a * x_rel + cos_a * y_rel;
    const float inv_rx_sq = 1.0f / (rx * rx);
    const float inv_ry_sq = 1.0f / (ry * ry);
    const float ellipse = (xr * xr) * inv_rx_sq + (yr * yr) * inv_ry_sq;
    if (ellipse > 1.0f) {
        return;
    }
    const long long flat_idx = (long long)py * (long long)image_w + (long long)px;
    const long long flat3 = flat_idx * 3LL;
    const int old0 = (int)canvas_flat[flat3 + 0];
    const int old1 = (int)canvas_flat[flat3 + 1];
    const int old2 = (int)canvas_flat[flat3 + 2];
    const float new0 = (float)((128 * src0 + 127 * old0) / 255);
    const float new1 = (float)((128 * src1 + 127 * old1) / 255);
    const float new2 = (float)((128 * src2 + 127 * old2) / 255);
    canvas_flat[flat3 + 0] = new0;
    canvas_flat[flat3 + 1] = new1;
    canvas_flat[flat3 + 2] = new2;
    const float tgt0 = target_flat[flat3 + 0];
    const float tgt1 = target_flat[flat3 + 1];
    const float tgt2 = target_flat[flat3 + 2];
    const float d0 = new0 - tgt0;
    const float d1 = new1 - tgt1;
    const float d2 = new2 - tgt2;
    canvas_minus_target_flat[flat3 + 0] = d0;
    canvas_minus_target_flat[flat3 + 1] = d1;
    canvas_minus_target_flat[flat3 + 2] = d2;
    const float h0 = tgt0 - (127.0f / 255.0f) * new0;
    const float h1 = tgt1 - (127.0f / 255.0f) * new1;
    const float h2 = tgt2 - (127.0f / 255.0f) * new2;
    target_minus_half_canvas_flat[flat3 + 0] = h0;
    target_minus_half_canvas_flat[flat3 + 1] = h1;
    target_minus_half_canvas_flat[flat3 + 2] = h2;
    canvas_minus_target_sqsum_flat[flat_idx] = d0 * d0 + d1 * d1 + d2 * d2;
    target_minus_half_sqsum_flat[flat_idx] = h0 * h0 + h1 * h1 + h2 * h2;
}
"""
            _CUPY_APPLY_FIXED_KERNEL = cupy.RawKernel(
                source,
                "apply_rotated_ellipse_fixed_half",
                options=("--std=c++11",),
            )
    return _CUPY_APPLY_FIXED_KERNEL


@dataclass(frozen=True)
class ComputeBackendInfo:
    requested: str
    resolved: str
    label: str


def resolve_compute_backend(requested: str) -> ComputeBackendInfo:
    requested_value = (requested or "auto").strip().lower() or "auto"
    if requested_value == "cpu":
        return ComputeBackendInfo(requested="cpu", resolved="cpu", label="CPU")
    torch = _load_torch()
    if requested_value == "gpu":
        if torch is None:
            raise RuntimeError("GPU backend requires torch.")
        if not torch.cuda.is_available():
            raise RuntimeError("GPU backend requires CUDA.")
        name = torch.cuda.get_device_name(0)
        return ComputeBackendInfo(requested="gpu", resolved="gpu", label=f"GPU ({name})")
    if requested_value == "auto":
        if torch is not None and torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            return ComputeBackendInfo(requested="auto", resolved="gpu", label=f"GPU ({name})")
        return ComputeBackendInfo(requested="auto", resolved="cpu", label="CPU")
    raise ValueError(f"Unknown compute backend: {requested!r}")


class TorchSearchRuntime:
    def __init__(self, target: np.ndarray, alpha_mask: np.ndarray | None = None) -> None:
        torch = _load_torch()
        if torch is None or not torch.cuda.is_available():
            raise RuntimeError("CUDA backend is not available.")
        self._torch = torch
        self.device = torch.device("cuda")
        self.target = torch.as_tensor(target, device=self.device, dtype=torch.float32)
        self.target_flat = self.target.view(-1, 3)
        self.canvas = torch.empty_like(self.target)
        self.canvas_flat = self.canvas.view(-1, 3)
        self._target_minus_half_canvas = torch.empty_like(self.target)
        self._target_minus_half_canvas_flat = self._target_minus_half_canvas.view(-1, 3)
        self._target_minus_half_canvas_sqsum = torch.empty((target.shape[0], target.shape[1]), device=self.device, dtype=torch.float32)
        self._target_minus_half_canvas_sqsum_flat = self._target_minus_half_canvas_sqsum.view(-1)
        self._canvas_minus_target = torch.empty_like(self.target)
        self._canvas_minus_target_flat = self._canvas_minus_target.view(-1, 3)
        self._canvas_minus_target_sqsum = torch.empty((target.shape[0], target.shape[1]), device=self.device, dtype=torch.float32)
        self._canvas_minus_target_sqsum_flat = self._canvas_minus_target_sqsum.view(-1)
        self._quality_edge_weight = None
        self._quality_edge_weight_flat = None
        self._quality_target_gx = None
        self._quality_target_gy = None
        self._quality_target_gx_flat = None
        self._quality_target_gy_flat = None
        self._quality_gradient_weight = 0.0
        self._quality_weighted_full_sq_scalar = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        self._quality_gradient_full_error_scalar = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        if alpha_mask is None:
            self.alpha = None
            self.alpha_scale = None
            self.alpha_nonzero = None
            self.alpha_inside = None
            self.alpha_scale_flat = None
            self.alpha_nonzero_flat = None
            self.alpha_inside_flat = None
        else:
            self.alpha = torch.as_tensor(alpha_mask, device=self.device, dtype=torch.uint8)
            self.alpha_scale = self.alpha.to(torch.float32) / 255.0
            self.alpha_nonzero = (self.alpha > 0).to(torch.float32)
            self.alpha_inside = self.alpha >= 128
            self.alpha_scale_flat = self.alpha_scale.view(-1)
            self.alpha_nonzero_flat = self.alpha_nonzero.view(-1)
            self.alpha_inside_flat = self.alpha_inside.view(-1)
        self.h, self.w = target.shape[:2]
        max_side = max(self.h, self.w)
        self._long_base = torch.arange(max_side, device=self.device, dtype=torch.long)
        self._float_base = torch.arange(max_side, device=self.device, dtype=torch.float32)
        self._long_cache: dict[int, object] = {}
        self._float_cache: dict[tuple[int, int], object] = {}
        self._canvas_full_sq_scalar = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        self._half_alpha_scale = torch.tensor(128.0 / 255.0, device=self.device, dtype=torch.float32)
        self._one_minus_half_alpha = (1.0 - self._half_alpha_scale).to(torch.float32)
        self._neighbor_offsets_x = torch.tensor([0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], device=self.device, dtype=torch.float32)
        self._neighbor_offsets_y = torch.tensor([0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0], device=self.device, dtype=torch.float32)
        self._neighbor_offsets_rx = torch.tensor([0, 0, 0, 0, 0, 1, -1, 0, 0, 1, -1, 0, 0], device=self.device, dtype=torch.float32)
        self._neighbor_offsets_ry = torch.tensor([0, 0, 0, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0], device=self.device, dtype=torch.float32)
        self._neighbor_offsets_angle = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1], device=self.device, dtype=torch.float32)
        self._neighbor_keep_no_diag = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12], device=self.device, dtype=torch.long)
        self._neighbor_keep_core = torch.tensor([0, 1, 2, 3, 4, 7, 8, 11, 12], device=self.device, dtype=torch.long)
        self._batch_pixels_oom_cap: int | None = None
        self._user_max_batch_pixels: int | None = None
        self._user_max_quality_batch_pixels: int | None = None
        self._batch_limit_refresh_counter = 0
        self._batch_limit_refresh_interval = 16
        self.max_batch_pixels = _safe_max_batch_pixels(torch)
        self.max_quality_batch_pixels = max(
            64_000,
            self.max_batch_pixels // _GPU_QUALITY_BATCH_DIVISOR,
        )
        self._rotated_graph_default = None
        self._rotated_graph_default_failed = False
        self._rotated_graph_medium = None
        self._rotated_graph_medium_failed = False
        self.enable_rotated_graph_default = True
        self.enable_rotated_graph_medium = True
        self.enable_rotated_ellipse_cupy = True
        self.prefer_rotated_ellipse_cupy = False
        self._rotated_graph_default_autocast_dtype = None
        try:
            if torch.cuda.is_bf16_supported():
                self._rotated_graph_default_autocast_dtype = torch.bfloat16
        except Exception:
            self._rotated_graph_default_autocast_dtype = None
        self._cupy = _load_cupy()
        self._cupy_score_fixed_kernel = None
        self._cupy_apply_fixed_kernel = None
        self._cupy_fixed_failed = self._cupy is None
        self._cupy_target_minus_half_canvas_flat = None
        self._cupy_target_minus_half_canvas_sqsum_flat = None
        self._cupy_canvas_minus_target_sqsum_flat = None
        self._cupy_canvas_full_sq_scalar = None
        self._cupy_canvas_flat = None
        self._cupy_target_flat = None
        self._cupy_canvas_minus_target_flat = None
        self.rotated_ellipse_shape_index = 0
        self.rotated_ellipse_reduce_stage4_after: int | None = None
        self.rotated_ellipse_reduce_stage12_to9_after: int | None = None
        self.rotated_graph_default_candidate_count = 2048
        self.rotated_graph_default_topk = 2
        self.rotated_graph_default_initial_stride = 3
        self.rotated_graph_default_shortlist_stride = 3

    def set_quality_context(
        self,
        edge_weight: np.ndarray,
        target_gx: np.ndarray,
        target_gy: np.ndarray,
        gradient_weight: float,
    ) -> None:
        torch = self._torch
        with torch.inference_mode():
            self._quality_edge_weight = torch.as_tensor(edge_weight, device=self.device, dtype=torch.float32)
            self._quality_edge_weight_flat = self._quality_edge_weight.view(-1)
            self._quality_target_gx = torch.as_tensor(target_gx, device=self.device, dtype=torch.float32)
            self._quality_target_gy = torch.as_tensor(target_gy, device=self.device, dtype=torch.float32)
            self._quality_target_gx_flat = self._quality_target_gx.view(-1)
            self._quality_target_gy_flat = self._quality_target_gy.view(-1)
            self._quality_gradient_weight = max(0.0, float(gradient_weight))
            self._refresh_batch_limits(force=True)

    def _refresh_batch_limits(self, *, force: bool = False) -> None:
        if not force:
            self._batch_limit_refresh_counter += 1
            if self._batch_limit_refresh_counter % self._batch_limit_refresh_interval != 1:
                return
        safe_pixels = _safe_max_batch_pixels(self._torch)
        if self._batch_pixels_oom_cap is not None:
            safe_pixels = min(safe_pixels, self._batch_pixels_oom_cap)
        if self._user_max_batch_pixels is not None:
            safe_pixels = min(safe_pixels, self._user_max_batch_pixels)
        self.max_batch_pixels = max(_GPU_BATCH_MIN_PIXELS, safe_pixels)
        quality_pixels = self.max_batch_pixels // _GPU_QUALITY_BATCH_DIVISOR
        if self._user_max_quality_batch_pixels is not None:
            quality_pixels = min(quality_pixels, self._user_max_quality_batch_pixels)
        self.max_quality_batch_pixels = max(64_000, quality_pixels)

    def set_quality_batch_pixel_limit(self, quality_batch_pixels: int) -> None:
        value = int(quality_batch_pixels)
        self._user_max_quality_batch_pixels = value if value > 0 else None
        self._refresh_batch_limits(force=True)

    def _record_batch_oom(self) -> None:
        reduced = max(_GPU_BATCH_MIN_PIXELS, self.max_batch_pixels // 2)
        if self._batch_pixels_oom_cap is None:
            self._batch_pixels_oom_cap = reduced
        else:
            self._batch_pixels_oom_cap = min(self._batch_pixels_oom_cap, reduced)
        self._refresh_batch_limits(force=True)

    def sync_full_canvas(self, canvas: np.ndarray) -> None:
        torch = self._torch
        with torch.inference_mode():
            self.canvas.copy_(torch.as_tensor(canvas, device=self.device, dtype=torch.float32))
            self._canvas_minus_target.copy_(self.canvas - self.target)
            self._target_minus_half_canvas.copy_(self.target - self._one_minus_half_alpha * self.canvas)
            self._canvas_minus_target_sqsum.copy_((self._canvas_minus_target * self._canvas_minus_target).sum(dim=2))
            self._target_minus_half_canvas_sqsum.copy_((self._target_minus_half_canvas * self._target_minus_half_canvas).sum(dim=2))
            self._canvas_full_sq_scalar.copy_(self._canvas_minus_target_sqsum.sum())

    def sync_region(self, canvas: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
        x0, y0, x1, y1 = bbox
        if x1 <= x0 or y1 <= y0:
            return
        torch = self._torch
        with torch.inference_mode():
            region = torch.as_tensor(canvas[y0:y1, x0:x1], device=self.device, dtype=torch.float32)
            self.canvas[y0:y1, x0:x1].copy_(region)
            self._canvas_minus_target[y0:y1, x0:x1].copy_(region - self.target[y0:y1, x0:x1])
            self._target_minus_half_canvas[y0:y1, x0:x1].copy_(self.target[y0:y1, x0:x1] - self._one_minus_half_alpha * region)
            self._canvas_minus_target_sqsum[y0:y1, x0:x1].copy_(
                (self._canvas_minus_target[y0:y1, x0:x1] * self._canvas_minus_target[y0:y1, x0:x1]).sum(dim=2)
            )
            self._target_minus_half_canvas_sqsum[y0:y1, x0:x1].copy_(
                (self._target_minus_half_canvas[y0:y1, x0:x1] * self._target_minus_half_canvas[y0:y1, x0:x1]).sum(dim=2)
            )

    def close(self) -> None:
        torch = getattr(self, "_torch", None)
        cupy = getattr(self, "_cupy", None)
        device = getattr(self, "device", None)
        self._long_cache.clear()
        self._float_cache.clear()
        self._rotated_graph_default = None
        self._rotated_graph_medium = None
        self._cupy_score_fixed_kernel = None
        self._cupy_apply_fixed_kernel = None
        self._cupy_target_minus_half_canvas_flat = None
        self._cupy_target_minus_half_canvas_sqsum_flat = None
        self._cupy_canvas_minus_target_sqsum_flat = None
        self._cupy_canvas_full_sq_scalar = None
        self._cupy_canvas_flat = None
        self._cupy_target_flat = None
        self._cupy_canvas_minus_target_flat = None
        self.target = None
        self.target_flat = None
        self.canvas = None
        self.canvas_flat = None
        self._target_minus_half_canvas = None
        self._target_minus_half_canvas_flat = None
        self._target_minus_half_canvas_sqsum = None
        self._target_minus_half_canvas_sqsum_flat = None
        self._canvas_minus_target = None
        self._canvas_minus_target_flat = None
        self._canvas_minus_target_sqsum = None
        self._canvas_minus_target_sqsum_flat = None
        self._quality_edge_weight = None
        self._quality_edge_weight_flat = None
        self._quality_target_gx = None
        self._quality_target_gy = None
        self._quality_target_gx_flat = None
        self._quality_target_gy_flat = None
        self._quality_weighted_full_sq_scalar = None
        self._quality_gradient_full_error_scalar = None
        self.alpha = None
        self.alpha_scale = None
        self.alpha_nonzero = None
        self.alpha_inside = None
        self.alpha_scale_flat = None
        self.alpha_nonzero_flat = None
        self.alpha_inside_flat = None
        self._long_base = None
        self._float_base = None
        self._canvas_full_sq_scalar = None
        self._half_alpha_scale = None
        self._one_minus_half_alpha = None
        self._neighbor_offsets_x = None
        self._neighbor_offsets_y = None
        self._neighbor_offsets_rx = None
        self._neighbor_offsets_ry = None
        self._neighbor_offsets_angle = None
        self._neighbor_keep_no_diag = None
        self._neighbor_keep_core = None
        try:
            if cupy is not None:
                try:
                    cupy.cuda.Device().synchronize()
                except Exception:
                    pass
                try:
                    cupy.get_default_memory_pool().free_all_blocks()
                except Exception:
                    pass
                try:
                    cupy.get_default_pinned_memory_pool().free_all_blocks()
                except Exception:
                    pass
        except Exception:
            pass
        try:
            if torch is not None and getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
                try:
                    torch.cuda.synchronize(device=device)
                except Exception:
                    pass
                gc.collect()
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
        except Exception:
            pass

    def can_use_rotated_ellipse_cupy(self) -> bool:
        return self._ensure_rotated_ellipse_cupy()

    def _is_current_stream_capturing(self) -> bool:
        try:
            return bool(self._torch.cuda.is_current_stream_capturing())
        except Exception:
            return False

    def _ensure_rotated_ellipse_cupy(self) -> bool:
        if not self.enable_rotated_ellipse_cupy or self.alpha is not None or self._cupy_fixed_failed:
            return False
        if self._cupy is None:
            self._cupy_fixed_failed = True
            return False
        if self._cupy_score_fixed_kernel is None:
            try:
                self._cupy_score_fixed_kernel = _load_cupy_score_fixed_kernel()
            except Exception:
                self._cupy_fixed_failed = True
                self._cupy_score_fixed_kernel = None
                return False
        if self._cupy_apply_fixed_kernel is None:
            try:
                self._cupy_apply_fixed_kernel = _load_cupy_apply_fixed_kernel()
            except Exception:
                self._cupy_fixed_failed = True
                self._cupy_apply_fixed_kernel = None
                return False
        if self._cupy_score_fixed_kernel is None:
            self._cupy_fixed_failed = True
            return False
        if self._cupy_apply_fixed_kernel is None:
            self._cupy_fixed_failed = True
            return False
        try:
            if self._cupy_canvas_flat is None:
                self._cupy_canvas_flat = self._cupy.asarray(self.canvas_flat)
            if self._cupy_target_flat is None:
                self._cupy_target_flat = self._cupy.asarray(self.target_flat)
            if self._cupy_canvas_minus_target_flat is None:
                self._cupy_canvas_minus_target_flat = self._cupy.asarray(self._canvas_minus_target_flat)
            if self._cupy_target_minus_half_canvas_flat is None:
                self._cupy_target_minus_half_canvas_flat = self._cupy.asarray(self._target_minus_half_canvas_flat)
            if self._cupy_target_minus_half_canvas_sqsum_flat is None:
                self._cupy_target_minus_half_canvas_sqsum_flat = self._cupy.asarray(self._target_minus_half_canvas_sqsum_flat)
            if self._cupy_canvas_minus_target_sqsum_flat is None:
                self._cupy_canvas_minus_target_sqsum_flat = self._cupy.asarray(self._canvas_minus_target_sqsum_flat)
            if self._cupy_canvas_full_sq_scalar is None:
                self._cupy_canvas_full_sq_scalar = self._cupy.asarray(self._canvas_full_sq_scalar)
        except Exception:
            self._cupy_fixed_failed = True
            self._cupy_canvas_flat = None
            self._cupy_target_flat = None
            self._cupy_canvas_minus_target_flat = None
            self._cupy_target_minus_half_canvas_flat = None
            self._cupy_target_minus_half_canvas_sqsum_flat = None
            self._cupy_canvas_minus_target_sqsum_flat = None
            self._cupy_canvas_full_sq_scalar = None
            return False
        return True

    def copy_canvas_to(self, canvas: np.ndarray) -> None:
        torch = self._torch
        with torch.inference_mode():
            canvas[:] = self.canvas.to(torch.uint8).cpu().numpy()

    def copy_canvas_region_to(self, canvas: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
        x0, y0, x1, y1 = bbox
        if x1 <= x0 or y1 <= y0:
            return
        torch = self._torch
        with torch.inference_mode():
            canvas[y0:y1, x0:x1] = self.canvas[y0:y1, x0:x1].to(torch.uint8).cpu().numpy()

    def apply_rotated_ellipse(self, shape: object, return_region: bool = True) -> tuple[tuple[int, int, int, int], np.ndarray]:
        x0, y0, x1, y1 = shape.bbox(self.w, self.h)
        bbox = (x0, y0, x1, y1)
        if x1 <= x0 or y1 <= y0:
            return bbox, _EMPTY_RGB_REGION
        torch = self._torch
        width = x1 - x0
        height = y1 - y0
        alpha = int(shape.color[3])
        with torch.inference_mode():
            xs = self._float_arange(width, 1).view(1, width) + float(x0) - float(shape.x)
            ys = self._float_arange(height, 1).view(height, 1) + float(y0) - float(shape.y)
            angle_rad = math.radians(float(shape.angle))
            cos_a = math.cos(angle_rad)
            sin_a = math.sin(angle_rad)
            xr = cos_a * xs + sin_a * ys
            yr = -sin_a * xs + cos_a * ys
            rx = max(float(shape.rx), 1e-6)
            ry = max(float(shape.ry), 1e-6)
            dx = xr / rx
            dy = yr / ry
            mask = (dx * dx + dy * dy) <= 1.0
            region = self.canvas[y0:y1, x0:x1]
            if alpha >= 255:
                src = torch.tensor(shape.color[:3], device=self.device, dtype=torch.float32).view(1, 1, 3)
                region_new = torch.where(mask.unsqueeze(-1), src, region)
            elif alpha <= 0:
                region_new = region
            else:
                src = torch.tensor(shape.color[:3], device=self.device, dtype=torch.int32).view(1, 1, 3)
                region_int = region.to(torch.int32)
                blended = ((alpha * src + (255 - alpha) * region_int) // 255).to(torch.float32)
                region_new = torch.where(mask.unsqueeze(-1), blended, region)
            self.canvas[y0:y1, x0:x1].copy_(region_new)
            self._canvas_minus_target[y0:y1, x0:x1].copy_(region_new - self.target[y0:y1, x0:x1])
            self._target_minus_half_canvas[y0:y1, x0:x1].copy_(
                self.target[y0:y1, x0:x1] - self._one_minus_half_alpha * region_new
            )
            self._canvas_minus_target_sqsum[y0:y1, x0:x1].copy_(
                (self._canvas_minus_target[y0:y1, x0:x1] * self._canvas_minus_target[y0:y1, x0:x1]).sum(dim=2)
            )
            self._target_minus_half_canvas_sqsum[y0:y1, x0:x1].copy_(
                (self._target_minus_half_canvas[y0:y1, x0:x1] * self._target_minus_half_canvas[y0:y1, x0:x1]).sum(dim=2)
            )
            if return_region:
                region_cpu = region_new.to(torch.uint8).cpu().numpy()
            else:
                region_cpu = _EMPTY_RGB_REGION
        return bbox, region_cpu

    def apply_rotated_ellipse_pack(self, shape_pack: object, return_region: bool = True) -> tuple[tuple[int, int, int, int], np.ndarray]:
        torch = self._torch
        if self.alpha is None and not return_region and self._ensure_rotated_ellipse_cupy():
            with torch.inference_mode():
                if torch.is_tensor(shape_pack) and shape_pack.device == self.device and shape_pack.dtype == torch.float32:
                    pack = shape_pack
                else:
                    pack = shape_pack.to(device=self.device, dtype=torch.float32)
                x, y, rx_raw, ry_raw, angle_deg, src0_raw, src1_raw, src2_raw = (
                    pack[1:9].to(device="cpu", dtype=torch.float32).tolist()
                )
                x = float(x)
                y = float(y)
                rx = max(float(rx_raw), 1e-6)
                ry = max(float(ry_raw), 1e-6)
                ext_x, ext_y = _rotated_ellipse_extent_xy(rx, ry, float(angle_deg))
                x0 = max(0, int(math.floor(x - ext_x)))
                y0 = max(0, int(math.floor(y - ext_y)))
                x1 = min(self.w, int(math.ceil(x + ext_x + 1.0)))
                y1 = min(self.h, int(math.ceil(y + ext_y + 1.0)))
                bbox = (x0, y0, x1, y1)
                if x1 <= x0 or y1 <= y0:
                    return bbox, _EMPTY_RGB_REGION
                angle = math.radians(float(angle_deg))
                src0 = int(src0_raw)
                src1 = int(src1_raw)
                src2 = int(src2_raw)
                total = (x1 - x0) * (y1 - y0)
                blocks = ((total + 255) // 256,)
                stream = self._cupy.cuda.Stream.from_external(torch.cuda.current_stream(device=self.device))
                with stream:
                    self._cupy_apply_fixed_kernel(
                        blocks,
                        (256,),
                        (
                            self._cupy_canvas_flat,
                            self._cupy_target_flat,
                            self._cupy_canvas_minus_target_flat,
                            self._cupy_target_minus_half_canvas_flat,
                            self._cupy_canvas_minus_target_sqsum_flat,
                            self._cupy_target_minus_half_canvas_sqsum_flat,
                            np.int32(self.w),
                            np.int32(x0),
                            np.int32(y0),
                            np.int32(x1 - x0),
                            np.int32(y1 - y0),
                            np.float32(x),
                            np.float32(y),
                            np.float32(rx),
                            np.float32(ry),
                            np.float32(angle),
                            np.int32(src0),
                            np.int32(src1),
                            np.int32(src2),
                        ),
                    )
                return bbox, _EMPTY_RGB_REGION
        with torch.inference_mode():
            if torch.is_tensor(shape_pack) and shape_pack.device == self.device and shape_pack.dtype == torch.float32:
                pack = shape_pack
            else:
                pack = shape_pack.to(device=self.device, dtype=torch.float32)
            x = pack[1]
            y = pack[2]
            rx = torch.clamp(pack[3], min=1e-6)
            ry = torch.clamp(pack[4], min=1e-6)
            angle_deg = pack[5]
            color = pack[6:9].view(1, 1, 3)
            alpha = 128
            angle_rad = angle_deg * (math.pi / 180.0)
            abs_cos = torch.abs(torch.cos(angle_rad))
            abs_sin = torch.abs(torch.sin(angle_rad))
            ext_x = torch.sqrt((rx * abs_cos) * (rx * abs_cos) + (ry * abs_sin) * (ry * abs_sin))
            ext_y = torch.sqrt((rx * abs_sin) * (rx * abs_sin) + (ry * abs_cos) * (ry * abs_cos))
            x_cpu, y_cpu, ext_x_cpu, ext_y_cpu = torch.cat(
                (
                    x.view(1),
                    y.view(1),
                    ext_x.view(1),
                    ext_y.view(1),
                ),
            ).to(device="cpu", dtype=torch.float32).tolist()
            x0 = max(0, int(math.floor(x_cpu - ext_x_cpu)))
            y0 = max(0, int(math.floor(y_cpu - ext_y_cpu)))
            x1 = min(self.w, int(math.ceil(x_cpu + ext_x_cpu + 1.0)))
            y1 = min(self.h, int(math.ceil(y_cpu + ext_y_cpu + 1.0)))
            bbox = (x0, y0, x1, y1)
            if x1 <= x0 or y1 <= y0:
                return bbox, _EMPTY_RGB_REGION
            width = x1 - x0
            height = y1 - y0
            xs = self._float_arange(width, 1).view(1, width) + float(x0) - x
            ys = self._float_arange(height, 1).view(height, 1) + float(y0) - y
            cos_a = torch.cos(angle_rad)
            sin_a = torch.sin(angle_rad)
            xr = cos_a * xs + sin_a * ys
            yr = -sin_a * xs + cos_a * ys
            mask = ((xr / rx) * (xr / rx) + (yr / ry) * (yr / ry)) <= 1.0
            region = self.canvas[y0:y1, x0:x1]
            if alpha >= 255:
                region_new = torch.where(mask.unsqueeze(-1), color, region)
            elif alpha <= 0:
                region_new = region
            else:
                region_int = region.to(torch.int32)
                src = color.to(torch.int32)
                blended = ((alpha * src + (255 - alpha) * region_int) // 255).to(torch.float32)
                region_new = torch.where(mask.unsqueeze(-1), blended, region)
            self.canvas[y0:y1, x0:x1].copy_(region_new)
            self._canvas_minus_target[y0:y1, x0:x1].copy_(region_new - self.target[y0:y1, x0:x1])
            self._target_minus_half_canvas[y0:y1, x0:x1].copy_(
                self.target[y0:y1, x0:x1] - self._one_minus_half_alpha * region_new
            )
            self._canvas_minus_target_sqsum[y0:y1, x0:x1].copy_(
                (self._canvas_minus_target[y0:y1, x0:x1] * self._canvas_minus_target[y0:y1, x0:x1]).sum(dim=2)
            )
            self._target_minus_half_canvas_sqsum[y0:y1, x0:x1].copy_(
                (self._target_minus_half_canvas[y0:y1, x0:x1] * self._target_minus_half_canvas[y0:y1, x0:x1]).sum(dim=2)
            )
            if return_region:
                region_cpu = region_new.to(torch.uint8).cpu().numpy()
            else:
                region_cpu = _EMPTY_RGB_REGION
        return bbox, region_cpu

    def _score_shapes_device(
        self,
        shapes: list[Shape],
        canvas_full_sq: float,
        canvas_norm: float,
    ) -> tuple[object, object]:
        torch = self._torch
        if not shapes:
            scores_out = torch.full((0,), float("inf"), device=self.device, dtype=torch.float32)
            colors_out = torch.zeros((0, 4), device=self.device, dtype=torch.int32)
            return scores_out, colors_out
        self._refresh_batch_limits()
        scores_out = torch.full((len(shapes),), float("inf"), device=self.device, dtype=torch.float32)
        colors_out = torch.zeros((len(shapes), 4), device=self.device, dtype=torch.int32)
        groups: dict[tuple[str, bool | None], list[tuple[int, Shape, tuple[int, int, int, int], int, int, int]]] = {}
        for idx, shape in enumerate(shapes):
            bbox, area, width, height = cached_bbox_metrics(shape, self.w, self.h)
            fixed_half_key = shape.color[3] == 128
            groups.setdefault((shape.type_name, fixed_half_key), []).append((idx, shape, bbox, area, width, height))
        with torch.inference_mode():
            self._canvas_full_sq_scalar.fill_(float(canvas_full_sq))
            for (type_name, fixed_half_key), items in groups.items():
                if not items:
                    continue
                if type_name == "rotated_ellipse":
                    self._score_rotated_ellipse_group(items, scores_out, colors_out, canvas_norm)
                    continue
                areas = [max(1, item[3]) for item in items]
                widths = [max(1, item[4]) for item in items]
                heights = [max(1, item[5]) for item in items]
                footprints = [widths[pos] * heights[pos] for pos in range(len(items))]
                pixel_limit = self.max_batch_pixels
                fixed_half_alpha = bool(fixed_half_key)
                if self._group_needs_sort(areas, pixel_limit, widths, heights):
                    order = sorted(range(len(items)), key=footprints.__getitem__, reverse=True)
                    items = [items[pos] for pos in order]
                    areas = [areas[pos] for pos in order]
                    widths = [widths[pos] for pos in order]
                    heights = [heights[pos] for pos in order]
                packed = self._pack_shape_group(type_name, items)
                for start, end in self._split_batch_ranges(areas, pixel_limit, widths, heights):
                    scores, colors = self._score_batch_resilient(type_name, packed, start, end, canvas_norm)
                    batch_indices = packed["indices"][start:end]
                    scores_out.index_copy_(0, batch_indices, scores)
                    colors_out.index_copy_(0, batch_indices, colors)
        return scores_out, colors_out

    def score_shapes_torch(
        self,
        shapes: list[Shape],
        canvas_full_sq: float,
        canvas_norm: float,
    ) -> tuple[object, object]:
        return self._score_shapes_device(shapes, canvas_full_sq, canvas_norm)

    def score_shapes(
        self,
        shapes: list[Shape],
        canvas_full_sq: float,
        canvas_norm: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        scores_out, colors_out = self._score_shapes_device(shapes, canvas_full_sq, canvas_norm)
        return scores_out.cpu().numpy(), colors_out.cpu().numpy()

    def score_shapes_quality(
        self,
        shapes: list[Shape],
        base_colors: np.ndarray,
        weighted_full_sq: float,
        weighted_norm: float,
        gradient_full_error: float,
        gradient_norm: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        scores_out, colors_out = self.score_shapes_quality_torch(
            shapes,
            base_colors,
            weighted_full_sq,
            weighted_norm,
            gradient_full_error,
            gradient_norm,
        )
        return scores_out.cpu().numpy(), colors_out.cpu().numpy()

    def score_shapes_quality_torch(
        self,
        shapes: list[Shape],
        base_colors: object,
        weighted_full_sq: float,
        weighted_norm: float,
        gradient_full_error: float,
        gradient_norm: float,
    ) -> tuple[object, object]:
        if self.alpha is not None:
            raise ValueError("GPU quality scoring does not support alpha masks.")
        if (
            self._quality_edge_weight_flat is None
            or self._quality_target_gx_flat is None
            or self._quality_target_gy_flat is None
        ):
            raise ValueError("Quality context is not configured.")
        torch = self._torch
        if not shapes:
            return (
                torch.zeros(0, device=self.device, dtype=torch.float32),
                torch.zeros((0, 4), device=self.device, dtype=torch.int32),
            )
        if torch.is_tensor(base_colors):
            base_colors_t = base_colors.to(device=self.device, dtype=torch.int32)
        else:
            base_colors_t = torch.as_tensor(base_colors, device=self.device, dtype=torch.int32)
        self._refresh_batch_limits()
        scores_out = torch.full((len(shapes),), float("inf"), device=self.device, dtype=torch.float32)
        colors_out = torch.zeros((len(shapes), 4), device=self.device, dtype=torch.int32)
        groups: dict[tuple[str, bool | None], list[tuple[int, Shape, tuple[int, int, int, int], int, int, int]]] = {}
        for idx, shape in enumerate(shapes):
            bbox, area, width, height = cached_bbox_metrics(shape, self.w, self.h)
            fixed_half_key = int(shape.color[3]) == 128
            groups.setdefault((shape.type_name, fixed_half_key), []).append((idx, shape, bbox, area, width, height))
        with torch.inference_mode():
            self._quality_weighted_full_sq_scalar.fill_(float(weighted_full_sq))
            self._quality_gradient_full_error_scalar.fill_(float(gradient_full_error))
            for (type_name, _fixed_half_key), items in groups.items():
                if not items:
                    continue
                areas = [max(1, (item[4] + 2) * (item[5] + 2)) for item in items]
                widths = [max(1, item[4] + 2) for item in items]
                heights = [max(1, item[5] + 2) for item in items]
                footprints = [widths[pos] * heights[pos] for pos in range(len(items))]
                pixel_limit = max(1, self.max_quality_batch_pixels)
                if self._group_needs_sort(areas, pixel_limit, widths, heights):
                    order = sorted(range(len(items)), key=footprints.__getitem__, reverse=True)
                    items = [items[pos] for pos in order]
                    areas = [areas[pos] for pos in order]
                    widths = [widths[pos] for pos in order]
                    heights = [heights[pos] for pos in order]
                packed = self._pack_shape_group(type_name, items)
                for start, end in self._split_batch_ranges(areas, pixel_limit, widths, heights):
                    scores, colors = self._score_quality_batch_resilient(
                        type_name,
                        packed,
                        base_colors_t,
                        start,
                        end,
                        weighted_full_sq,
                        weighted_norm,
                        gradient_full_error,
                        gradient_norm,
                    )
                    batch_indices = packed["indices"][start:end]
                    scores_out.index_copy_(0, batch_indices, scores)
                    colors_out.index_copy_(0, batch_indices, colors)
        return scores_out, colors_out

    def _score_quality_batch_resilient(
        self,
        type_name: str,
        packed: dict[str, object],
        base_colors: object,
        start: int,
        end: int,
        weighted_full_sq: float,
        weighted_norm: float,
        gradient_full_error: float,
        gradient_norm: float,
    ) -> tuple[object, object]:
        torch = self._torch
        try:
            return self._score_quality_batch(
                type_name,
                packed,
                base_colors,
                start,
                end,
                weighted_full_sq,
                weighted_norm,
                gradient_full_error,
                gradient_norm,
            )
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or end - start <= 1:
                raise
            self._record_batch_oom()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            mid = start + max(1, (end - start) // 2)
            left_scores, left_colors = self._score_quality_batch_resilient(
                type_name,
                packed,
                base_colors,
                start,
                mid,
                weighted_full_sq,
                weighted_norm,
                gradient_full_error,
                gradient_norm,
            )
            right_scores, right_colors = self._score_quality_batch_resilient(
                type_name,
                packed,
                base_colors,
                mid,
                end,
                weighted_full_sq,
                weighted_norm,
                gradient_full_error,
                gradient_norm,
            )
            return torch.cat((left_scores, right_scores), dim=0), torch.cat((left_colors, right_colors), dim=0)

    def find_best(
        self,
        shapes: list[Shape],
        canvas_full_sq: float,
        canvas_norm: float,
    ) -> tuple[float, Shape | None]:
        if not shapes:
            return float("inf"), None
        scores, colors = self.score_shapes(shapes, canvas_full_sq, canvas_norm)
        best_idx = int(np.argmin(scores))
        if not np.isfinite(scores[best_idx]):
            return float("inf"), None
        best_shape = shapes[best_idx]
        best_shape.color = tuple(int(v) for v in colors[best_idx])
        return float(scores[best_idx]), best_shape

    def score_rotated_ellipse_params(
        self,
        x: np.ndarray,
        y: np.ndarray,
        rx: np.ndarray,
        ry: np.ndarray,
        angle: np.ndarray,
        alpha_values: np.ndarray,
        canvas_full_sq: float,
        canvas_norm: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        scores_out, colors_out = self._score_rotated_ellipse_params_gpu(
            x,
            y,
            rx,
            ry,
            angle,
            alpha_values,
            canvas_full_sq,
            canvas_norm,
        )
        return scores_out.cpu().numpy(), colors_out.cpu().numpy()

    def score_grouped_rotated_ellipse_params(
        self,
        x: np.ndarray,
        y: np.ndarray,
        rx: np.ndarray,
        ry: np.ndarray,
        angle: np.ndarray,
        alpha_values: np.ndarray,
        canvas_full_sq: float,
        canvas_norm: float,
        group_size: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        count = int(x.shape[0])
        if group_size <= 0:
            raise ValueError("group_size must be positive")
        if count == 0:
            return (
                np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.float32),
                np.zeros((0, 4), dtype=np.int32),
            )
        if count % group_size != 0:
            raise ValueError("count must be divisible by group_size")
        torch = self._torch
        scores_out, colors_out = self._score_rotated_ellipse_params_gpu(
            x,
            y,
            rx,
            ry,
            angle,
            alpha_values,
            canvas_full_sq,
            canvas_norm,
        )
        group_count = count // group_size
        group_scores = scores_out.view(group_count, group_size)
        best_offsets = torch.argmin(group_scores, dim=1)
        flat_idx = best_offsets + torch.arange(group_count, device=self.device, dtype=torch.long) * group_size
        best_scores = scores_out.index_select(0, flat_idx)
        best_colors = colors_out.index_select(0, flat_idx)
        return flat_idx.cpu().numpy(), best_scores.cpu().numpy(), best_colors.cpu().numpy()

    def score_grouped_rotated_ellipse_params_torch(
        self,
        x: np.ndarray,
        y: np.ndarray,
        rx: np.ndarray,
        ry: np.ndarray,
        angle: np.ndarray,
        alpha_values: np.ndarray,
        canvas_full_sq: float,
        canvas_norm: float,
        group_size: int,
    ) -> tuple[object, object, object, object, object, object, object]:
        count = int(x.shape[0])
        if group_size <= 0:
            raise ValueError("group_size must be positive")
        if count == 0:
            torch = self._torch
            empty_float = torch.zeros(0, device=self.device, dtype=torch.float32)
            empty_int = torch.zeros((0, 4), device=self.device, dtype=torch.int32)
            return empty_float, empty_float, empty_float, empty_float, empty_float, empty_float, empty_int
        if count % group_size != 0:
            raise ValueError("count must be divisible by group_size")
        return self._score_grouped_rotated_ellipse_params_device(
            self._to_device_float_tensor(x),
            self._to_device_float_tensor(y),
            self._to_device_float_tensor(rx),
            self._to_device_float_tensor(ry),
            self._to_device_float_tensor(angle),
            self._to_device_float_tensor(alpha_values),
            canvas_full_sq,
            canvas_norm,
            group_size,
            fixed_half_alpha=False,
        )

    def search_rotated_ellipse_device(
        self,
        chain_count: int,
        random_count: int,
        canvas_full_sq: float,
        canvas_norm: float,
        seed: int,
        refine_stage_count: int = 5,
    ) -> tuple[float, object | None]:
        if chain_count <= 0 or random_count <= 0:
            return float("inf"), None
        refine_stage_count = max(1, min(5, int(refine_stage_count)))
        use_graph = (not self.prefer_rotated_ellipse_cupy) and self._should_use_rotated_ellipse_graph_default(
            chain_count,
            random_count,
        ) and refine_stage_count >= 5
        if use_graph:
            graph_result = self._search_rotated_ellipse_graph_default(
                canvas_full_sq,
                canvas_norm,
                seed,
                chain_count,
                random_count,
            )
            if graph_result is not None:
                return graph_result
        torch = self._torch
        with torch.inference_mode():
            coarse_stride = 2 if self.w * self.h >= 280_000 else 1
            gen = torch.Generator(device=self.device)
            gen.manual_seed(int(seed) & 0xFFFFFFFF)
            total = chain_count * random_count
            x = torch.rand(total, device=self.device, dtype=torch.float32, generator=gen) * float(max(0, self.w - 1))
            y = torch.rand(total, device=self.device, dtype=torch.float32, generator=gen) * float(max(0, self.h - 1))
            max_rx = float(max(2.0, self.w / 8.0))
            max_ry = float(max(2.0, self.h / 8.0))
            rx = 1.0 + torch.rand(total, device=self.device, dtype=torch.float32, generator=gen) * max(0.0, max_rx - 1.0)
            ry = 1.0 + torch.rand(total, device=self.device, dtype=torch.float32, generator=gen) * max(0.0, max_ry - 1.0)
            angle = torch.rand(total, device=self.device, dtype=torch.float32, generator=gen) * 180.0
            best_scores, best_x, best_y, best_rx, best_ry, best_angle, best_colors = (
                self._score_grouped_rotated_ellipse_params_device(
                    x,
                    y,
                    rx,
                    ry,
                    angle,
                    None,
                    canvas_full_sq,
                    canvas_norm,
                    random_count,
                    sample_stride=coarse_stride,
                    fixed_half_alpha=True,
                )
            )
            long_side = float(max(self.w, self.h))
            step_sets = (
                (
                    max(24.0, long_side / 8.0),
                    max(24.0, long_side / 8.0),
                    max(32.0, long_side / 4.0),
                    max(32.0, long_side / 4.0),
                    30.0,
                    1,
                ),
                (
                    max(12.0, long_side / 16.0),
                    max(12.0, long_side / 16.0),
                    max(16.0, long_side / 8.0),
                    max(16.0, long_side / 8.0),
                    16.0,
                    1,
                ),
                (12.0, 12.0, 12.0, 12.0, 8.0, 1),
                (4.0, 4.0, 4.0, 4.0, 4.0, 1),
                (2.0, 2.0, 2.0, 2.0, 2.0, 1),
            )
            step_sets = step_sets[:refine_stage_count]
            if self.alpha is None:
                for idx, (sx, sy, srx, sry, sa, sample_stride) in enumerate(step_sets):
                    if self.w * self.h >= 280_000 and idx < 2:
                        trial_scores, trial_x, trial_y, trial_rx, trial_ry, trial_angle, trial_colors = (
                            self._refine_rotated_ellipse_step_shortlist(
                                best_x,
                                best_y,
                                best_rx,
                                best_ry,
                                best_angle,
                                canvas_full_sq,
                                canvas_norm,
                                sx,
                                sy,
                                srx,
                                sry,
                                sa,
                                2,
                                2,
                            )
                        )
                    elif self.w * self.h >= 280_000:
                        trial_scores, trial_x, trial_y, trial_rx, trial_ry, trial_angle, trial_colors = (
                            self._refine_rotated_ellipse_step(
                                best_x,
                                best_y,
                                best_rx,
                                best_ry,
                                best_angle,
                                canvas_full_sq,
                                canvas_norm,
                                sx,
                                sy,
                                srx,
                                sry,
                                sa,
                                sample_stride,
                            )
                        )
                    else:
                        trial_scores, trial_x, trial_y, trial_rx, trial_ry, trial_angle, trial_colors = (
                            self._refine_rotated_ellipse_step_grouped(
                                best_x,
                                best_y,
                                best_rx,
                                best_ry,
                                best_angle,
                                canvas_full_sq,
                                canvas_norm,
                                sx,
                                sy,
                                srx,
                                sry,
                                sa,
                                sample_stride,
                            )
                        )
                    improved = torch.isfinite(trial_scores) & (trial_scores < best_scores)
                    best_scores = torch.where(improved, trial_scores, best_scores)
                    best_x = torch.where(improved, trial_x, best_x)
                    best_y = torch.where(improved, trial_y, best_y)
                    best_rx = torch.where(improved, trial_rx, best_rx)
                    best_ry = torch.where(improved, trial_ry, best_ry)
                    best_angle = torch.where(improved, trial_angle, best_angle)
                    best_colors = torch.where(improved.view(-1, 1), trial_colors, best_colors)
                masked_scores = best_scores
            else:
                valid = torch.isfinite(best_scores)
                for idx, (sx, sy, srx, sry, sa, sample_stride) in enumerate(step_sets):
                    active_idx = torch.nonzero(valid, as_tuple=False).flatten()
                    if int(active_idx.numel()) == 0:
                        break
                    ax = best_x.index_select(0, active_idx)
                    ay = best_y.index_select(0, active_idx)
                    arx = best_rx.index_select(0, active_idx)
                    ary = best_ry.index_select(0, active_idx)
                    aa = best_angle.index_select(0, active_idx)
                    if self.w * self.h >= 280_000 and idx < 2:
                        trial_scores, trial_x, trial_y, trial_rx, trial_ry, trial_angle, trial_colors = (
                            self._refine_rotated_ellipse_step_shortlist(
                                ax,
                                ay,
                                arx,
                                ary,
                                aa,
                                canvas_full_sq,
                                canvas_norm,
                                sx,
                                sy,
                                srx,
                                sry,
                                sa,
                                2,
                                2,
                            )
                        )
                    elif self.w * self.h >= 280_000:
                        trial_scores, trial_x, trial_y, trial_rx, trial_ry, trial_angle, trial_colors = (
                            self._refine_rotated_ellipse_step(
                                ax,
                                ay,
                                arx,
                                ary,
                                aa,
                                canvas_full_sq,
                                canvas_norm,
                                sx,
                                sy,
                                srx,
                                sry,
                                sa,
                                sample_stride,
                            )
                        )
                    else:
                        trial_scores, trial_x, trial_y, trial_rx, trial_ry, trial_angle, trial_colors = (
                            self._refine_rotated_ellipse_step_grouped(
                                ax,
                                ay,
                                arx,
                                ary,
                                aa,
                                canvas_full_sq,
                                canvas_norm,
                                sx,
                                sy,
                                srx,
                                sry,
                                sa,
                                sample_stride,
                            )
                        )
                    current_scores = best_scores.index_select(0, active_idx)
                    improved = torch.isfinite(trial_scores) & (trial_scores < current_scores)
                    improved_idx = active_idx.index_select(0, torch.nonzero(improved, as_tuple=False).flatten())
                    best_scores.index_copy_(0, improved_idx, trial_scores[improved])
                    best_x.index_copy_(0, improved_idx, trial_x[improved])
                    best_y.index_copy_(0, improved_idx, trial_y[improved])
                    best_rx.index_copy_(0, improved_idx, trial_rx[improved])
                    best_ry.index_copy_(0, improved_idx, trial_ry[improved])
                    best_angle.index_copy_(0, improved_idx, trial_angle[improved])
                    best_colors.index_copy_(0, improved_idx, trial_colors[improved])
                    valid[improved_idx] = True
                masked_scores = torch.where(valid, best_scores, torch.full_like(best_scores, float("inf")))
            best_pos = torch.argmin(masked_scores)
            best_idx = best_pos.view(1)
            best_pack = torch.cat(
                (
                    masked_scores.index_select(0, best_idx),
                    best_x.index_select(0, best_idx),
                    best_y.index_select(0, best_idx),
                    best_rx.index_select(0, best_idx),
                    best_ry.index_select(0, best_idx),
                    best_angle.index_select(0, best_idx),
                    best_colors.index_select(0, best_idx).to(dtype=torch.float32).reshape(-1),
                ),
                dim=0,
            )
            best_score = float(best_pack[0].item())
            if not math.isfinite(best_score):
                return float("inf"), None
            return best_score, best_pack.detach().clone()

    def search_rotated_ellipse(
        self,
        chain_count: int,
        random_count: int,
        canvas_full_sq: float,
        canvas_norm: float,
        seed: int,
        refine_stage_count: int = 5,
    ) -> tuple[float, tuple[float, float, float, float, float] | None, tuple[int, int, int, int] | None]:
        best_score, best_pack = self.search_rotated_ellipse_device(
            chain_count,
            random_count,
            canvas_full_sq,
            canvas_norm,
            seed,
            refine_stage_count,
        )
        if best_pack is None or not math.isfinite(best_score):
            return float("inf"), None, None
        best_values = best_pack.to(device="cpu", dtype=self._torch.float32).tolist()
        params = (
            float(best_values[1]),
            float(best_values[2]),
            float(best_values[3]),
            float(best_values[4]),
            float(best_values[5]),
        )
        color = tuple(int(v) for v in best_values[6:10])
        return best_score, params, color

    def _should_use_rotated_ellipse_graph_default(self, chain_count: int, random_count: int) -> bool:
        if not (self.alpha is None and chain_count == 2):
            return False
        if random_count == 320 and self.w * self.h >= 280_000:
            return self.enable_rotated_graph_default and not self._rotated_graph_default_failed
        if random_count == 192 and self.w * self.h < 280_000:
            return self.enable_rotated_graph_medium and not self._rotated_graph_medium_failed
        return False

    def _score_rotated_ellipse_params_device_fixed(
        self,
        x: object,
        y: object,
        rx: object,
        ry: object,
        angle: object,
        canvas_norm: float,
        max_w: int,
        max_h: int,
        sample_stride: int = 1,
    ) -> tuple[object, object]:
        torch = self._torch
        rx_t = torch.clamp(rx, min=1e-6)
        ry_t = torch.clamp(ry, min=1e-6)
        angle_rad = angle * (math.pi / 180.0)
        abs_cos = torch.abs(torch.cos(angle_rad))
        abs_sin = torch.abs(torch.sin(angle_rad))
        ext_x = torch.sqrt((rx_t * abs_cos) * (rx_t * abs_cos) + (ry_t * abs_sin) * (ry_t * abs_sin))
        ext_y = torch.sqrt((rx_t * abs_sin) * (rx_t * abs_sin) + (ry_t * abs_cos) * (ry_t * abs_cos))
        x0 = torch.clamp(torch.floor(x - ext_x).to(torch.long), min=0)
        y0 = torch.clamp(torch.floor(y - ext_y).to(torch.long), min=0)
        x1 = torch.clamp(torch.ceil(x + ext_x + 1.0).to(torch.long), max=self.w)
        y1 = torch.clamp(torch.ceil(y + ext_y + 1.0).to(torch.long), max=self.h)
        width = torch.clamp(x1 - x0, min=0)
        height = torch.clamp(y1 - y0, min=0)
        if self.alpha is None and self._ensure_rotated_ellipse_cupy():
            return self._score_rotated_ellipse_cupy_fixed_batch(
                x0,
                y0,
                width,
                height,
                x,
                y,
                rx_t,
                ry_t,
                angle_rad,
                canvas_norm,
                sample_stride,
            )
        return self._score_rotated_ellipse_batch(
            x0,
            y0,
            width,
            height,
            x,
            y,
            rx_t,
            ry_t,
            angle_rad,
            None,
            canvas_norm,
            max_w,
            max_h,
            sample_stride,
            fixed_half_alpha=True,
        )

    def _score_grouped_rotated_ellipse_params_device_fixed(
        self,
        x: object,
        y: object,
        rx: object,
        ry: object,
        angle: object,
        canvas_norm: float,
        group_size: int,
        max_w: int,
        max_h: int,
        sample_stride: int = 1,
    ) -> tuple[object, object, object, object, object, object, object]:
        torch = self._torch
        count = int(x.shape[0])
        scores_out, colors_out = self._score_rotated_ellipse_params_device_fixed(
            x,
            y,
            rx,
            ry,
            angle,
            canvas_norm,
            max_w,
            max_h,
            sample_stride=sample_stride,
        )
        group_count = count // group_size
        group_scores = scores_out.view(group_count, group_size)
        best_offsets = torch.argmin(group_scores, dim=1)
        flat_idx = best_offsets + self._long_arange(group_count) * group_size
        best_scores = scores_out.index_select(0, flat_idx)
        best_x = x.index_select(0, flat_idx)
        best_y = y.index_select(0, flat_idx)
        best_rx = rx.index_select(0, flat_idx)
        best_ry = ry.index_select(0, flat_idx)
        best_angle = angle.index_select(0, flat_idx)
        best_colors = colors_out.index_select(0, flat_idx)
        return best_scores, best_x, best_y, best_rx, best_ry, best_angle, best_colors

    def _score_rotated_ellipse_candidate_grid_exact_fixed(
        self,
        x_candidates: object,
        y_candidates: object,
        rx_candidates: object,
        ry_candidates: object,
        angle_candidates: object,
        canvas_norm: float,
        max_w: int,
        max_h: int,
        sample_stride: int = 1,
    ) -> tuple[object, object, object, object, object, object, object]:
        torch = self._torch
        count = int(x_candidates.shape[0])
        candidate_count = int(x_candidates.shape[1])
        if self.alpha is None and self._ensure_rotated_ellipse_cupy():
            total = count * candidate_count
            return self._score_grouped_rotated_ellipse_params_device_fixed(
                x_candidates.reshape(total),
                y_candidates.reshape(total),
                rx_candidates.reshape(total),
                ry_candidates.reshape(total),
                angle_candidates.reshape(total),
                canvas_norm,
                candidate_count,
                max_w,
                max_h,
                sample_stride=sample_stride,
            )
        angle_radians = angle_candidates * (math.pi / 180.0)
        abs_cos = torch.abs(torch.cos(angle_radians))
        abs_sin = torch.abs(torch.sin(angle_radians))
        ext_x_candidates = torch.sqrt((rx_candidates * abs_cos) * (rx_candidates * abs_cos) + (ry_candidates * abs_sin) * (ry_candidates * abs_sin))
        ext_y_candidates = torch.sqrt((rx_candidates * abs_sin) * (rx_candidates * abs_sin) + (ry_candidates * abs_cos) * (ry_candidates * abs_cos))
        x0_candidates = torch.clamp(torch.floor(x_candidates - ext_x_candidates).to(torch.long), min=0)
        y0_candidates = torch.clamp(torch.floor(y_candidates - ext_y_candidates).to(torch.long), min=0)
        x1_candidates = torch.clamp(torch.ceil(x_candidates + ext_x_candidates + 1.0).to(torch.long), max=self.w)
        y1_candidates = torch.clamp(torch.ceil(y_candidates + ext_y_candidates + 1.0).to(torch.long), max=self.h)
        x0 = x0_candidates.min(dim=1).values
        y0 = y0_candidates.min(dim=1).values
        x1 = x1_candidates.max(dim=1).values
        y1 = y1_candidates.max(dim=1).values
        width = torch.clamp(x1 - x0, min=0)
        height = torch.clamp(y1 - y0, min=0)
        return self._score_rotated_ellipse_candidate_grid_exact_batch(
            x0,
            y0,
            width,
            height,
            max_w,
            max_h,
            x_candidates,
            y_candidates,
            rx_candidates,
            ry_candidates,
            angle_candidates,
            canvas_norm,
            sample_stride,
        )

    def _refine_rotated_ellipse_step_shortlist_fixed(
        self,
        x: object,
        y: object,
        rx: object,
        ry: object,
        angle: object,
        canvas_norm: float,
        sx: float,
        sy: float,
        srx: float,
        sry: float,
        sa: float,
        coarse_stride: int,
        shortlist_size: int,
        coarse_max_w: int,
        coarse_max_h: int,
        exact_max_w: int,
        exact_max_h: int,
    ) -> tuple[object, object, object, object, object, object, object]:
        torch = self._torch
        count = int(x.shape[0])
        x_candidates, y_candidates, rx_candidates, ry_candidates, angle_candidates = self._build_rotated_ellipse_neighbor_candidates(
            x,
            y,
            rx,
            ry,
            angle,
            sx,
            sy,
            srx,
            sry,
            sa,
        )
        neighbor_count = self._rotated_ellipse_neighbor_count()
        total = count * neighbor_count
        coarse_scores, _ = self._score_rotated_ellipse_params_device_fixed(
            x_candidates.reshape(total),
            y_candidates.reshape(total),
            rx_candidates.reshape(total),
            ry_candidates.reshape(total),
            angle_candidates.reshape(total),
            canvas_norm,
            coarse_max_w,
            coarse_max_h,
            sample_stride=coarse_stride,
        )
        shortlist = torch.topk(
            coarse_scores.view(count, neighbor_count),
            k=min(max(1, shortlist_size), neighbor_count),
            dim=1,
            largest=False,
        ).indices
        rows = self._long_arange(count).unsqueeze(1)
        x_short = x_candidates[rows, shortlist]
        y_short = y_candidates[rows, shortlist]
        rx_short = rx_candidates[rows, shortlist]
        ry_short = ry_candidates[rows, shortlist]
        angle_short = angle_candidates[rows, shortlist]
        shortlist_total = int(x_short.shape[0]) * int(x_short.shape[1])
        return self._score_grouped_rotated_ellipse_params_device_fixed(
            x_short.reshape(shortlist_total),
            y_short.reshape(shortlist_total),
            rx_short.reshape(shortlist_total),
            ry_short.reshape(shortlist_total),
            angle_short.reshape(shortlist_total),
            canvas_norm,
            int(x_short.shape[1]),
            exact_max_w,
            exact_max_h,
            sample_stride=1,
        )

    def _refine_rotated_ellipse_step_fixed(
        self,
        x: object,
        y: object,
        rx: object,
        ry: object,
        angle: object,
        canvas_norm: float,
        sx: float,
        sy: float,
        srx: float,
        sry: float,
        sa: float,
        max_w: int,
        max_h: int,
        sample_stride: int = 1,
    ) -> tuple[object, object, object, object, object, object, object]:
        x_candidates, y_candidates, rx_candidates, ry_candidates, angle_candidates = self._build_rotated_ellipse_neighbor_candidates(
            x,
            y,
            rx,
            ry,
            angle,
            sx,
            sy,
            srx,
            sry,
            sa,
        )
        return self._score_rotated_ellipse_candidate_grid_exact_fixed(
            x_candidates,
            y_candidates,
            rx_candidates,
            ry_candidates,
            angle_candidates,
            canvas_norm,
            max_w,
            max_h,
            sample_stride=sample_stride,
        )

    def _refine_rotated_ellipse_step_grouped_fixed(
        self,
        x: object,
        y: object,
        rx: object,
        ry: object,
        angle: object,
        canvas_norm: float,
        sx: float,
        sy: float,
        srx: float,
        sry: float,
        sa: float,
        max_w: int,
        max_h: int,
        sample_stride: int = 1,
    ) -> tuple[object, object, object, object, object, object, object]:
        x_candidates, y_candidates, rx_candidates, ry_candidates, angle_candidates = self._build_rotated_ellipse_neighbor_candidates(
            x,
            y,
            rx,
            ry,
            angle,
            sx,
            sy,
            srx,
            sry,
            sa,
        )
        total = int(x_candidates.shape[0]) * int(x_candidates.shape[1])
        return self._score_grouped_rotated_ellipse_params_device_fixed(
            x_candidates.reshape(total),
            y_candidates.reshape(total),
            rx_candidates.reshape(total),
            ry_candidates.reshape(total),
            angle_candidates.reshape(total),
            canvas_norm,
            int(x_candidates.shape[1]),
            max_w,
            max_h,
            sample_stride=sample_stride,
        )

    def _rotated_ellipse_graph_stage_caps(self) -> tuple[tuple[int, int], ...]:
        long_side = float(max(self.w, self.h))
        max_r = float(max(max(2.0, self.w / 8.0), max(2.0, self.h / 8.0)))
        stage_steps = (
            max(max(24.0, long_side / 8.0), max(32.0, long_side / 4.0)),
            max(max(12.0, long_side / 16.0), max(16.0, long_side / 8.0)),
            12.0,
            4.0,
            2.0,
        )
        stage_caps: list[tuple[int, int]] = []
        current_r = max_r
        cap = int(math.ceil(2.0 * current_r + 1.0))
        stage_caps.append((min(self.w, cap), min(self.h, cap)))
        for step in stage_steps:
            current_r += step
            cap = int(math.ceil(2.0 * current_r + 1.0))
            stage_caps.append((min(self.w, cap), min(self.h, cap)))
        return tuple(stage_caps)

    def _run_rotated_ellipse_graph_default_body(
        self,
        x: object,
        y: object,
        rx: object,
        ry: object,
        angle: object,
        canvas_norm: float,
        stage_caps: tuple[tuple[int, int], ...],
    ) -> tuple[object, object, object, object, object, object, object]:
        torch = self._torch
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=self._rotated_graph_default_autocast_dtype)
            if self._rotated_graph_default_autocast_dtype is not None
            else nullcontext()
        )
        with autocast_ctx:
            scores_out, colors_out = self._score_rotated_ellipse_params_device_fixed(
                x,
                y,
                rx,
                ry,
                angle,
                canvas_norm,
                stage_caps[0][0],
                stage_caps[0][1],
                sample_stride=self.rotated_graph_default_initial_stride,
            )
            best_idx = torch.topk(
                scores_out,
                k=min(max(1, int(self.rotated_graph_default_topk)), int(scores_out.shape[0])),
                dim=0,
                largest=False,
            ).indices
            best_scores = scores_out.index_select(0, best_idx)
            best_x = x.index_select(0, best_idx)
            best_y = y.index_select(0, best_idx)
            best_rx = rx.index_select(0, best_idx)
            best_ry = ry.index_select(0, best_idx)
            best_angle = angle.index_select(0, best_idx)
            best_colors = colors_out.index_select(0, best_idx)
            trial_scores, trial_x, trial_y, trial_rx, trial_ry, trial_angle, trial_colors = self._refine_rotated_ellipse_step_shortlist_fixed(
                best_x,
                best_y,
                best_rx,
                best_ry,
                best_angle,
                canvas_norm,
                max(24.0, float(max(self.w, self.h)) / 8.0),
                max(24.0, float(max(self.w, self.h)) / 8.0),
                max(32.0, float(max(self.w, self.h)) / 4.0),
                max(32.0, float(max(self.w, self.h)) / 4.0),
                30.0,
                self.rotated_graph_default_shortlist_stride,
                2,
                stage_caps[1][0],
                stage_caps[1][1],
                stage_caps[1][0],
                stage_caps[1][1],
            )
            improved = torch.isfinite(trial_scores) & (trial_scores < best_scores)
            best_scores = torch.where(improved, trial_scores, best_scores)
            best_x = torch.where(improved, trial_x, best_x)
            best_y = torch.where(improved, trial_y, best_y)
            best_rx = torch.where(improved, trial_rx, best_rx)
            best_ry = torch.where(improved, trial_ry, best_ry)
            best_angle = torch.where(improved, trial_angle, best_angle)
            best_colors = torch.where(improved.view(-1, 1), trial_colors, best_colors)
            trial_scores, trial_x, trial_y, trial_rx, trial_ry, trial_angle, trial_colors = self._refine_rotated_ellipse_step_shortlist_fixed(
                best_x,
                best_y,
                best_rx,
                best_ry,
                best_angle,
                canvas_norm,
                max(12.0, float(max(self.w, self.h)) / 16.0),
                max(12.0, float(max(self.w, self.h)) / 16.0),
                max(16.0, float(max(self.w, self.h)) / 8.0),
                max(16.0, float(max(self.w, self.h)) / 8.0),
                16.0,
                self.rotated_graph_default_shortlist_stride,
                2,
                stage_caps[2][0],
                stage_caps[2][1],
                stage_caps[2][0],
                stage_caps[2][1],
            )
            improved = torch.isfinite(trial_scores) & (trial_scores < best_scores)
            best_scores = torch.where(improved, trial_scores, best_scores)
            best_x = torch.where(improved, trial_x, best_x)
            best_y = torch.where(improved, trial_y, best_y)
            best_rx = torch.where(improved, trial_rx, best_rx)
            best_ry = torch.where(improved, trial_ry, best_ry)
            best_angle = torch.where(improved, trial_angle, best_angle)
            best_colors = torch.where(improved.view(-1, 1), trial_colors, best_colors)
            trial_scores, trial_x, trial_y, trial_rx, trial_ry, trial_angle, trial_colors = self._refine_rotated_ellipse_step_fixed(
                best_x,
                best_y,
                best_rx,
                best_ry,
                best_angle,
                canvas_norm,
                12.0,
                12.0,
                12.0,
                12.0,
                8.0,
                stage_caps[3][0],
                stage_caps[3][1],
                sample_stride=1,
            )
            improved = torch.isfinite(trial_scores) & (trial_scores < best_scores)
            best_scores = torch.where(improved, trial_scores, best_scores)
            best_x = torch.where(improved, trial_x, best_x)
            best_y = torch.where(improved, trial_y, best_y)
            best_rx = torch.where(improved, trial_rx, best_rx)
            best_ry = torch.where(improved, trial_ry, best_ry)
            best_angle = torch.where(improved, trial_angle, best_angle)
            best_colors = torch.where(improved.view(-1, 1), trial_colors, best_colors)
            trial_scores, trial_x, trial_y, trial_rx, trial_ry, trial_angle, trial_colors = self._refine_rotated_ellipse_step_fixed(
                best_x,
                best_y,
                best_rx,
                best_ry,
                best_angle,
                canvas_norm,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                stage_caps[4][0],
                stage_caps[4][1],
                sample_stride=1,
            )
            improved = torch.isfinite(trial_scores) & (trial_scores < best_scores)
            best_scores = torch.where(improved, trial_scores, best_scores)
            best_x = torch.where(improved, trial_x, best_x)
            best_y = torch.where(improved, trial_y, best_y)
            best_rx = torch.where(improved, trial_rx, best_rx)
            best_ry = torch.where(improved, trial_ry, best_ry)
            best_angle = torch.where(improved, trial_angle, best_angle)
            best_colors = torch.where(improved.view(-1, 1), trial_colors, best_colors)
            trial_scores, trial_x, trial_y, trial_rx, trial_ry, trial_angle, trial_colors = self._refine_rotated_ellipse_step_fixed(
                best_x,
                best_y,
                best_rx,
                best_ry,
                best_angle,
                canvas_norm,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                stage_caps[5][0],
                stage_caps[5][1],
                sample_stride=1,
            )
            improved = torch.isfinite(trial_scores) & (trial_scores < best_scores)
            best_scores = torch.where(improved, trial_scores, best_scores)
            best_x = torch.where(improved, trial_x, best_x)
            best_y = torch.where(improved, trial_y, best_y)
            best_rx = torch.where(improved, trial_rx, best_rx)
            best_ry = torch.where(improved, trial_ry, best_ry)
            best_angle = torch.where(improved, trial_angle, best_angle)
            best_colors = torch.where(improved.view(-1, 1), trial_colors, best_colors)
            best_pos = torch.argmin(best_scores).view(1)
            return torch.cat(
                (
                    best_scores.index_select(0, best_pos),
                    best_x.index_select(0, best_pos),
                    best_y.index_select(0, best_pos),
                    best_rx.index_select(0, best_pos),
                    best_ry.index_select(0, best_pos),
                    best_angle.index_select(0, best_pos),
                    best_colors.index_select(0, best_pos).to(dtype=torch.float32).reshape(-1),
                ),
                dim=0,
            )

    def _run_rotated_ellipse_graph_medium_body(
        self,
        x: object,
        y: object,
        rx: object,
        ry: object,
        angle: object,
        canvas_norm: float,
        stage_caps: tuple[tuple[int, int], ...],
    ) -> object:
        torch = self._torch
        long_side = float(max(self.w, self.h))
        step_sets = (
            (
                max(24.0, long_side / 8.0),
                max(24.0, long_side / 8.0),
                max(32.0, long_side / 4.0),
                max(32.0, long_side / 4.0),
                30.0,
                1,
            ),
            (
                max(12.0, long_side / 16.0),
                max(12.0, long_side / 16.0),
                max(16.0, long_side / 8.0),
                max(16.0, long_side / 8.0),
                16.0,
                1,
            ),
            (12.0, 12.0, 12.0, 12.0, 8.0, 1),
            (4.0, 4.0, 4.0, 4.0, 4.0, 1),
            (2.0, 2.0, 2.0, 2.0, 2.0, 1),
        )
        best_scores, best_x, best_y, best_rx, best_ry, best_angle, best_colors = self._score_grouped_rotated_ellipse_params_device_fixed(
            x,
            y,
            rx,
            ry,
            angle,
            canvas_norm,
            192,
            stage_caps[0][0],
            stage_caps[0][1],
            sample_stride=1,
        )
        for idx, (sx, sy, srx, sry, sa, sample_stride) in enumerate(step_sets):
            trial_scores, trial_x, trial_y, trial_rx, trial_ry, trial_angle, trial_colors = (
                self._refine_rotated_ellipse_step_grouped_fixed(
                    best_x,
                    best_y,
                    best_rx,
                    best_ry,
                    best_angle,
                    canvas_norm,
                    sx,
                    sy,
                    srx,
                    sry,
                    sa,
                    stage_caps[idx + 1][0],
                    stage_caps[idx + 1][1],
                    sample_stride=sample_stride,
                )
            )
            improved = torch.isfinite(trial_scores) & (trial_scores < best_scores)
            best_scores = torch.where(improved, trial_scores, best_scores)
            best_x = torch.where(improved, trial_x, best_x)
            best_y = torch.where(improved, trial_y, best_y)
            best_rx = torch.where(improved, trial_rx, best_rx)
            best_ry = torch.where(improved, trial_ry, best_ry)
            best_angle = torch.where(improved, trial_angle, best_angle)
            best_colors = torch.where(improved.view(-1, 1), trial_colors, best_colors)
        best_pos = torch.argmin(best_scores).view(1)
        return torch.cat(
            (
                best_scores.index_select(0, best_pos),
                best_x.index_select(0, best_pos),
                best_y.index_select(0, best_pos),
                best_rx.index_select(0, best_pos),
                best_ry.index_select(0, best_pos),
                best_angle.index_select(0, best_pos),
                best_colors.index_select(0, best_pos).to(dtype=torch.float32).reshape(-1),
            ),
            dim=0,
        )

    def _init_rotated_ellipse_graph_default(self, canvas_norm: float) -> None:
        if self._rotated_graph_default is not None or self._rotated_graph_default_failed:
            return
        torch = self._torch
        stage_caps = self._rotated_ellipse_graph_stage_caps()
        count = int(max(2, self.rotated_graph_default_candidate_count))
        x = torch.empty(count, device=self.device, dtype=torch.float32)
        y = torch.empty(count, device=self.device, dtype=torch.float32)
        rx = torch.empty(count, device=self.device, dtype=torch.float32)
        ry = torch.empty(count, device=self.device, dtype=torch.float32)
        angle = torch.empty(count, device=self.device, dtype=torch.float32)
        x.zero_()
        y.zero_()
        rx.fill_(1.0)
        ry.fill_(1.0)
        angle.zero_()
        try:
            stream = torch.cuda.Stream(device=self.device)
            stream.wait_stream(torch.cuda.current_stream(device=self.device))
            with torch.cuda.stream(stream):
                self._canvas_full_sq_scalar.fill_(1.0)
                self._run_rotated_ellipse_graph_default_body(x, y, rx, ry, angle, canvas_norm, tuple(stage_caps))
            torch.cuda.current_stream(device=self.device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            self._canvas_full_sq_scalar.fill_(1.0)
            with torch.cuda.graph(graph):
                best_pack = self._run_rotated_ellipse_graph_default_body(
                    x, y, rx, ry, angle, canvas_norm, tuple(stage_caps),
                )
            self._rotated_graph_default = {
                "graph": graph,
                "x": x,
                "y": y,
                "rx": rx,
                "ry": ry,
                "angle": angle,
                "best_pack": best_pack,
            }
        except Exception:
            self._rotated_graph_default_failed = True
            self._rotated_graph_default = None

    def _init_rotated_ellipse_graph_medium(self, canvas_norm: float) -> None:
        if self._rotated_graph_medium is not None or self._rotated_graph_medium_failed:
            return
        torch = self._torch
        stage_caps = self._rotated_ellipse_graph_stage_caps()
        x = torch.empty(384, device=self.device, dtype=torch.float32)
        y = torch.empty(384, device=self.device, dtype=torch.float32)
        rx = torch.empty(384, device=self.device, dtype=torch.float32)
        ry = torch.empty(384, device=self.device, dtype=torch.float32)
        angle = torch.empty(384, device=self.device, dtype=torch.float32)
        x.zero_()
        y.zero_()
        rx.fill_(1.0)
        ry.fill_(1.0)
        angle.zero_()
        try:
            stream = torch.cuda.Stream(device=self.device)
            stream.wait_stream(torch.cuda.current_stream(device=self.device))
            with torch.cuda.stream(stream):
                self._canvas_full_sq_scalar.fill_(1.0)
                self._run_rotated_ellipse_graph_medium_body(x, y, rx, ry, angle, canvas_norm, stage_caps)
            torch.cuda.current_stream(device=self.device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            self._canvas_full_sq_scalar.fill_(1.0)
            with torch.cuda.graph(graph):
                best_pack = self._run_rotated_ellipse_graph_medium_body(
                    x, y, rx, ry, angle, canvas_norm, stage_caps,
                )
            self._rotated_graph_medium = {
                "graph": graph,
                "x": x,
                "y": y,
                "rx": rx,
                "ry": ry,
                "angle": angle,
                "best_pack": best_pack,
            }
        except Exception:
            self._rotated_graph_medium_failed = True
            self._rotated_graph_medium = None

    def _search_rotated_ellipse_graph_default(
        self,
        canvas_full_sq: float,
        canvas_norm: float,
        seed: int,
        chain_count: int,
        random_count: int,
    ) -> tuple[float, object | None] | None:
        if chain_count != 2:
            return None
        if random_count == 320 and self.w * self.h >= 280_000:
            self._init_rotated_ellipse_graph_default(canvas_norm)
            state = self._rotated_graph_default
        elif random_count == 192 and self.w * self.h < 280_000:
            self._init_rotated_ellipse_graph_medium(canvas_norm)
            state = self._rotated_graph_medium
        else:
            return None
        if state is None:
            return None
        gen = self._torch.Generator(device=self.device)
        gen.manual_seed(int(seed) & 0xFFFFFFFF)
        with self._torch.inference_mode():
            state["x"].uniform_(0.0, float(max(0, self.w - 1)), generator=gen)
            state["y"].uniform_(0.0, float(max(0, self.h - 1)), generator=gen)
            state["rx"].uniform_(1.0, float(max(2.0, self.w / 8.0)), generator=gen)
            state["ry"].uniform_(1.0, float(max(2.0, self.h / 8.0)), generator=gen)
            state["angle"].uniform_(0.0, 180.0, generator=gen)
            self._canvas_full_sq_scalar.fill_(float(canvas_full_sq))
            state["graph"].replay()
            best_pack = state["best_pack"].detach().clone()
            best_score = float(best_pack[0].item())
            if not math.isfinite(best_score):
                return float("inf"), None
            return best_score, best_pack

    def _score_rotated_ellipse_cupy_fixed_batch(
        self,
        x0: object,
        y0: object,
        width: object,
        height: object,
        x: object,
        y: object,
        rx: object,
        ry: object,
        angle_rad: object,
        canvas_norm: float,
        sample_stride: int,
    ) -> tuple[object, object]:
        if not self._ensure_rotated_ellipse_cupy():
            raise RuntimeError("CuPy fixed-half-alpha scorer is not available.")
        torch = self._torch
        cupy = self._cupy
        if cupy is None or self._cupy_score_fixed_kernel is None:
            raise RuntimeError("CuPy fixed-half-alpha scorer is not available.")
        count = int(x0.shape[0])
        scores_out = torch.full((count,), float("inf"), device=self.device, dtype=torch.float32)
        colors_out = torch.zeros((count, 4), device=self.device, dtype=torch.int32)
        if count == 0:
            return scores_out, colors_out
        stride = max(1, int(sample_stride))
        sample_scale = float(stride * stride)
        denom = float(max(canvas_norm / sample_scale, 1.0))
        stream = cupy.cuda.Stream.from_external(torch.cuda.current_stream(device=self.device))
        with stream:
            self._cupy_score_fixed_kernel(
                (count,),
                (_CUPY_SCORE_FIXED_BLOCK_SIZE,),
                (
                    cupy.asarray(x0),
                    cupy.asarray(y0),
                    cupy.asarray(width),
                    cupy.asarray(height),
                    cupy.asarray(x),
                    cupy.asarray(y),
                    cupy.asarray(rx),
                    cupy.asarray(ry),
                    cupy.asarray(angle_rad),
                    self._cupy_target_minus_half_canvas_flat,
                    self._cupy_target_minus_half_canvas_sqsum_flat,
                    self._cupy_canvas_minus_target_sqsum_flat,
                    np.int32(self.w),
                    self._cupy_canvas_full_sq_scalar,
                    np.float32(sample_scale),
                    np.float32(denom),
                    np.int32(stride),
                    cupy.asarray(scores_out),
                    cupy.asarray(colors_out),
                ),
            )
        return scores_out, colors_out

    def _score_rotated_ellipse_params_device_cupy_fixed(
        self,
        x: object,
        y: object,
        rx: object,
        ry: object,
        angle: object,
        canvas_full_sq: float,
        canvas_norm: float,
        sample_stride: int = 1,
    ) -> tuple[object, object]:
        torch = self._torch
        count = int(x.shape[0])
        if count == 0:
            return (
                torch.full((0,), float("inf"), device=self.device, dtype=torch.float32),
                torch.zeros((0, 4), device=self.device, dtype=torch.int32),
            )
        x_t = self._to_device_float_tensor(x)
        y_t = self._to_device_float_tensor(y)
        rx_t = torch.clamp(self._to_device_float_tensor(rx), min=1e-6)
        ry_t = torch.clamp(self._to_device_float_tensor(ry), min=1e-6)
        angle_t = self._to_device_float_tensor(angle)
        angle_rad = angle_t * (math.pi / 180.0)
        abs_cos = torch.abs(torch.cos(angle_rad))
        abs_sin = torch.abs(torch.sin(angle_rad))
        ext_x = torch.sqrt((rx_t * abs_cos) * (rx_t * abs_cos) + (ry_t * abs_sin) * (ry_t * abs_sin))
        ext_y = torch.sqrt((rx_t * abs_sin) * (rx_t * abs_sin) + (ry_t * abs_cos) * (ry_t * abs_cos))
        x0 = torch.clamp(torch.floor(x_t - ext_x).to(torch.long), min=0)
        y0 = torch.clamp(torch.floor(y_t - ext_y).to(torch.long), min=0)
        x1 = torch.clamp(torch.ceil(x_t + ext_x + 1.0).to(torch.long), max=self.w)
        y1 = torch.clamp(torch.ceil(y_t + ext_y + 1.0).to(torch.long), max=self.h)
        width = torch.clamp(x1 - x0, min=0)
        height = torch.clamp(y1 - y0, min=0)
        widths_cpu, heights_cpu, area_cpu, max_w, max_h = self._width_height_area_cpu(width, height)
        stride = max(1, int(sample_stride))
        pixel_budget = max(1, self.max_batch_pixels * stride * stride)
        if int(area_cpu.sum()) <= pixel_budget:
            self._canvas_full_sq_scalar.fill_(float(canvas_full_sq))
            return self._score_rotated_ellipse_cupy_fixed_batch(
                x0,
                y0,
                width,
                height,
                x_t,
                y_t,
                rx_t,
                ry_t,
                angle_rad,
                canvas_norm,
                sample_stride,
            )
        scores_out = torch.full((count,), float("inf"), device=self.device, dtype=torch.float32)
        colors_out = torch.zeros((count, 4), device=self.device, dtype=torch.int32)
        order_cpu = np.argsort(-area_cpu, kind="stable")
        order_t = torch.as_tensor(order_cpu, device=self.device, dtype=torch.long)
        start = 0
        while start < count:
            total_area = 0
            end = start
            while end < count and (end == start or total_area + int(area_cpu[order_cpu[end]]) <= pixel_budget):
                total_area += int(area_cpu[order_cpu[end]])
                end += 1
            batch_idx = order_t[start:end]
            self._canvas_full_sq_scalar.fill_(float(canvas_full_sq))
            scores, colors = self._score_rotated_ellipse_cupy_fixed_batch(
                x0.index_select(0, batch_idx),
                y0.index_select(0, batch_idx),
                width.index_select(0, batch_idx),
                height.index_select(0, batch_idx),
                x_t.index_select(0, batch_idx),
                y_t.index_select(0, batch_idx),
                rx_t.index_select(0, batch_idx),
                ry_t.index_select(0, batch_idx),
                angle_rad.index_select(0, batch_idx),
                canvas_norm,
                sample_stride,
            )
            scores_out.index_copy_(0, batch_idx, scores)
            colors_out.index_copy_(0, batch_idx, colors)
            start = end
        return scores_out, colors_out

    def _to_device_float_tensor(self, values: np.ndarray | object) -> object:
        torch = self._torch
        if torch.is_tensor(values):
            if values.device == self.device and values.dtype == torch.float32:
                return values
            return values.to(device=self.device, dtype=torch.float32)
        return torch.as_tensor(values, device=self.device, dtype=torch.float32)

    def _score_rotated_ellipse_candidate_grid_exact(
        self,
        x_candidates: object,
        y_candidates: object,
        rx_candidates: object,
        ry_candidates: object,
        angle_candidates: object,
        canvas_full_sq: float,
        canvas_norm: float,
        sample_stride: int = 1,
    ) -> tuple[object, object, object, object, object, object, object]:
        torch = self._torch
        count = int(x_candidates.shape[0])
        empty_float = torch.zeros(0, device=self.device, dtype=torch.float32)
        empty_int = torch.zeros((0, 4), device=self.device, dtype=torch.int32)
        if count == 0:
            return empty_float, empty_float, empty_float, empty_float, empty_float, empty_float, empty_int
        candidate_count = int(x_candidates.shape[1])
        if self.alpha is None and not self._is_current_stream_capturing() and self._ensure_rotated_ellipse_cupy():
            total = count * candidate_count
            return self._score_grouped_rotated_ellipse_params_device(
                x_candidates.reshape(total),
                y_candidates.reshape(total),
                rx_candidates.reshape(total),
                ry_candidates.reshape(total),
                angle_candidates.reshape(total),
                None,
                canvas_full_sq,
                canvas_norm,
                candidate_count,
                sample_stride=sample_stride,
                fixed_half_alpha=True,
            )
        angle_radians = angle_candidates * (math.pi / 180.0)
        abs_cos = torch.abs(torch.cos(angle_radians))
        abs_sin = torch.abs(torch.sin(angle_radians))
        ext_x_candidates = torch.sqrt(
            (rx_candidates * abs_cos) * (rx_candidates * abs_cos)
            + (ry_candidates * abs_sin) * (ry_candidates * abs_sin),
        )
        ext_y_candidates = torch.sqrt(
            (rx_candidates * abs_sin) * (rx_candidates * abs_sin)
            + (ry_candidates * abs_cos) * (ry_candidates * abs_cos),
        )
        x0_candidates = torch.clamp(torch.floor(x_candidates - ext_x_candidates).to(torch.long), min=0)
        y0_candidates = torch.clamp(torch.floor(y_candidates - ext_y_candidates).to(torch.long), min=0)
        x1_candidates = torch.clamp(torch.ceil(x_candidates + ext_x_candidates + 1.0).to(torch.long), max=self.w)
        y1_candidates = torch.clamp(torch.ceil(y_candidates + ext_y_candidates + 1.0).to(torch.long), max=self.h)
        x0 = x0_candidates.min(dim=1).values
        y0 = y0_candidates.min(dim=1).values
        x1 = x1_candidates.max(dim=1).values
        y1 = y1_candidates.max(dim=1).values
        width = torch.clamp(x1 - x0, min=0)
        height = torch.clamp(y1 - y0, min=0)
        self._canvas_full_sq_scalar.fill_(float(canvas_full_sq))
        stride = max(1, int(sample_stride))
        pixel_budget = max(1, self.max_batch_pixels * stride * stride)
        widths_cpu, heights_cpu, area_cpu, max_w, max_h = self._width_height_area_cpu(width, height)
        if max_w * max_h * count * candidate_count <= pixel_budget:
            return self._score_rotated_ellipse_candidate_grid_exact_batch(
                x0,
                y0,
                width,
                height,
                max_w,
                max_h,
                x_candidates,
                y_candidates,
                rx_candidates,
                ry_candidates,
                angle_candidates,
                canvas_norm,
                sample_stride,
            )
        score_out = torch.full((count,), float("inf"), device=self.device, dtype=torch.float32)
        x_out = torch.zeros((count,), device=self.device, dtype=torch.float32)
        y_out = torch.zeros((count,), device=self.device, dtype=torch.float32)
        rx_out = torch.zeros((count,), device=self.device, dtype=torch.float32)
        ry_out = torch.zeros((count,), device=self.device, dtype=torch.float32)
        angle_out = torch.zeros((count,), device=self.device, dtype=torch.float32)
        color_out = torch.zeros((count, 4), device=self.device, dtype=torch.int32)
        order_cpu = np.argsort(-area_cpu, kind="stable")
        area_sorted_cpu = area_cpu[order_cpu]
        start = 0
        order_t = torch.as_tensor(order_cpu, device=self.device, dtype=torch.long)
        while start < count:
            batch_area = max(1, int(area_sorted_cpu[start]))
            batch_len = max(1, pixel_budget // max(1, batch_area * candidate_count))
            end = min(count, start + batch_len)
            batch_idx = order_t[start:end]
            bx0 = x0.index_select(0, batch_idx)
            by0 = y0.index_select(0, batch_idx)
            bw = width.index_select(0, batch_idx)
            bh = height.index_select(0, batch_idx)
            bx = x_candidates.index_select(0, batch_idx)
            by = y_candidates.index_select(0, batch_idx)
            brx = rx_candidates.index_select(0, batch_idx)
            bry = ry_candidates.index_select(0, batch_idx)
            ba = angle_candidates.index_select(0, batch_idx)
            batch_scores, batch_x, batch_y, batch_rx, batch_ry, batch_angle, batch_colors = (
                self._score_rotated_ellipse_candidate_grid_exact_batch(
                    bx0,
                    by0,
                    bw,
                    bh,
                    int(widths_cpu[order_cpu[start:end]].max()),
                    int(heights_cpu[order_cpu[start:end]].max()),
                    bx,
                    by,
                    brx,
                    bry,
                    ba,
                    canvas_norm,
                    sample_stride,
                )
            )
            score_out.index_copy_(0, batch_idx, batch_scores)
            x_out.index_copy_(0, batch_idx, batch_x)
            y_out.index_copy_(0, batch_idx, batch_y)
            rx_out.index_copy_(0, batch_idx, batch_rx)
            ry_out.index_copy_(0, batch_idx, batch_ry)
            angle_out.index_copy_(0, batch_idx, batch_angle)
            color_out.index_copy_(0, batch_idx, batch_colors)
            start = end
        return score_out, x_out, y_out, rx_out, ry_out, angle_out, color_out

    def _score_rotated_ellipse_candidate_grid_exact_batch(
        self,
        bx0: object,
        by0: object,
        bw: object,
        bh: object,
        max_w: int,
        max_h: int,
        bx: object,
        by: object,
        brx: object,
        bry: object,
        ba: object,
        canvas_norm: float,
        sample_stride: int,
    ) -> tuple[object, object, object, object, object, object, object]:
        torch = self._torch
        batch_count = int(bx.shape[0])
        candidate_count = int(bx.shape[1])
        stride = max(1, int(sample_stride))
        x_len = max(1, (max_w + stride - 1) // stride)
        y_len = max(1, (max_h + stride - 1) // stride)
        x_base_i = (self._long_arange(x_len) * stride).view(1, 1, x_len)
        y_base_i = (self._long_arange(y_len) * stride).view(1, y_len, 1)
        x_base_f = self._float_arange(x_len, stride).view(1, 1, x_len)
        y_base_f = self._float_arange(y_len, stride).view(1, y_len, 1)
        valid = (x_base_i < bw.view(-1, 1, 1)) & (y_base_i < bh.view(-1, 1, 1))
        x_idx = (bx0.view(-1, 1, 1) + x_base_i).clamp(max=self.w - 1)
        y_idx = (by0.view(-1, 1, 1) + y_base_i).clamp(max=self.h - 1)
        flat_idx = (y_idx * self.w + x_idx).reshape(-1)
        grid_shape = (x_idx.shape[0], y_idx.shape[1], x_idx.shape[2])
        with nullcontext():
            region_d_base = self._target_minus_half_canvas_flat.index_select(0, flat_idx).view(grid_shape[0], grid_shape[1], grid_shape[2], 3)
            region_d_sq_base = self._target_minus_half_canvas_sqsum_flat.index_select(0, flat_idx).view(grid_shape[0], grid_shape[1], grid_shape[2])
            region_old_sq_base = self._canvas_minus_target_sqsum_flat.index_select(0, flat_idx).view(grid_shape[0], grid_shape[1], grid_shape[2])
            x_abs = (bx0.view(-1, 1, 1) + x_base_f).unsqueeze(1)
            y_abs = (by0.view(-1, 1, 1) + y_base_f).unsqueeze(1)
            bxv = bx.view(-1, candidate_count, 1, 1)
            byv = by.view(-1, candidate_count, 1, 1)
            brxv = torch.clamp(brx.view(-1, candidate_count, 1, 1), min=1e-6)
            bryv = torch.clamp(bry.view(-1, candidate_count, 1, 1), min=1e-6)
            inv_brx_sq = torch.reciprocal(brxv * brxv)
            inv_bry_sq = torch.reciprocal(bryv * bryv)
            bav = (ba.view(-1, candidate_count, 1, 1) * (math.pi / 180.0))
            cos_a = torch.cos(bav)
            sin_a = torch.sin(bav)
            x_rel = x_abs - bxv
            y_rel = y_abs - byv
            xr = cos_a * x_rel + sin_a * y_rel
            yr = -sin_a * x_rel + cos_a * y_rel
            valid_mask = valid.unsqueeze(1)
            mask = ((((xr * xr) * inv_brx_sq) + ((yr * yr) * inv_bry_sq)) <= 1.0) & valid_mask
            valid_scores = ((bw > 0) & (bh > 0)).unsqueeze(1).expand(-1, candidate_count)
            if self.alpha is None:
                weight = mask.sum(dim=(2, 3), dtype=torch.float32)
                valid_scores = valid_scores & (weight >= 0.5)
                safe_weight = weight.clamp_min(1e-6).unsqueeze(-1)
                alpha_scale = self._half_alpha_scale
                sample_scale = float(stride * stride)
                d = region_d_base.unsqueeze(1)
                mask_float = mask.unsqueeze(-1)
                sum_d = (d * mask_float).sum(dim=(2, 3))
                sum_d_sq_total = (region_d_sq_base.unsqueeze(1) * mask).sum(dim=(2, 3), dtype=torch.float32)
                rgb_float = torch.floor(torch.clamp(sum_d / (alpha_scale * safe_weight), 0.0, 255.0))
                rgb = rgb_float.to(torch.int32)
                applied = alpha_scale * rgb_float
                inside_new_sq = sum_d_sq_total - 2.0 * (applied * sum_d).sum(dim=2) + weight * (applied * applied).sum(dim=2)
                inside_old_sq = (region_old_sq_base.unsqueeze(1) * mask).sum(dim=(2, 3), dtype=torch.float32)
            else:
                color_mask = mask.to(torch.float32)
                region_alpha_inside = self.alpha_inside_flat.index_select(0, flat_idx).view(grid_shape)
                region_alpha_scale = self.alpha_scale_flat.index_select(0, flat_idx).view(grid_shape)
                region_alpha_nonzero = self.alpha_nonzero_flat.index_select(0, flat_idx).view(grid_shape)
                body_total = mask.sum(dim=(2, 3), dtype=torch.float32)
                inside = (mask & region_alpha_inside.unsqueeze(1)).sum(dim=(2, 3), dtype=torch.float32)
                overlap_ok = (body_total > 0) & (inside >= body_total * STICKER_OVERLAP_MIN)
                effective_mask = color_mask * region_alpha_scale.unsqueeze(1)
                valid_scores = valid_scores & (body_total > 0) & overlap_ok
                weight = effective_mask.sum(dim=(2, 3))
                valid_scores = valid_scores & (weight >= 0.5)
                safe_weight = weight.clamp_min(1e-6).unsqueeze(-1)
                alpha_scale = self._half_alpha_scale
                sample_scale = float(stride * stride)
                d = region_d_base.unsqueeze(1)
                mask_float = color_mask.unsqueeze(-1)
                weight_float = region_alpha_scale.unsqueeze(1)
                sum_d = (d * weight_float.unsqueeze(-1) * mask_float).sum(dim=(2, 3))
                sum_d_sq = ((d * d) * mask_float).sum(dim=(2, 3))
                rgb_float = torch.floor(torch.clamp((sum_d / safe_weight) / alpha_scale, 0.0, 255.0))
                rgb = rgb_float.to(torch.int32)
                applied = alpha_scale * rgb_float
                allowed_mask = region_alpha_nonzero.unsqueeze(1) * color_mask
                allowed_mask_f = allowed_mask.to(torch.float32)
                sum_d_allowed = (d * allowed_mask_f.unsqueeze(-1)).sum(dim=(2, 3))
                sum_d_sq_allowed = ((d * d) * allowed_mask_f.unsqueeze(-1)).sum(dim=(2, 3))
                allowed_count = allowed_mask_f.sum(dim=(2, 3)).clamp_min(1e-6)
                inside_new_sq = sum_d_sq_allowed.sum(dim=2) - 2.0 * (applied * sum_d_allowed).sum(dim=2) + allowed_count * (applied * applied).sum(dim=2)
                inside_old_sq = (region_old_sq_base.unsqueeze(1) * allowed_mask_f).sum(dim=(2, 3))
            total_sq = (self._canvas_full_sq_scalar / sample_scale) - inside_old_sq + inside_new_sq
            denom = max(canvas_norm / sample_scale, 1.0)
            scores = torch.sqrt(torch.clamp(total_sq, min=0.0) / denom)
            scores = scores.masked_fill(~valid_scores, float("inf"))
        best_offsets = torch.argmin(scores, dim=1)
        gather_idx = best_offsets.unsqueeze(1)
        best_scores = scores.gather(1, gather_idx).squeeze(1)
        best_x = bx.gather(1, gather_idx).squeeze(1)
        best_y = by.gather(1, gather_idx).squeeze(1)
        best_rx = brx.gather(1, gather_idx).squeeze(1)
        best_ry = bry.gather(1, gather_idx).squeeze(1)
        best_angle = ba.gather(1, gather_idx).squeeze(1)
        best_rgb = rgb.gather(1, gather_idx.unsqueeze(-1).expand(-1, 1, 3)).squeeze(1)
        best_alpha = torch.full((best_rgb.shape[0], 1), 128, device=self.device, dtype=torch.int32)
        best_colors = torch.cat([best_rgb, best_alpha], dim=1)
        return best_scores, best_x, best_y, best_rx, best_ry, best_angle, best_colors

    def _build_rotated_ellipse_neighbor_candidates(
        self,
        x: object,
        y: object,
        rx: object,
        ry: object,
        angle: object,
        sx: float,
        sy: float,
        srx: float,
        sry: float,
        sa: float,
    ) -> tuple[object, object, object, object, object]:
        torch = self._torch
        x_candidates = torch.clamp(
            x.unsqueeze(1) + float(sx) * self._neighbor_offsets_x,
            min=0.0,
            max=float(self.w - 1),
        )
        y_candidates = torch.clamp(
            y.unsqueeze(1) + float(sy) * self._neighbor_offsets_y,
            min=0.0,
            max=float(self.h - 1),
        )
        rx_candidates = torch.clamp(
            rx.unsqueeze(1) + float(srx) * self._neighbor_offsets_rx,
            min=1.0,
            max=float(self.w),
        )
        ry_candidates = torch.clamp(
            ry.unsqueeze(1) + float(sry) * self._neighbor_offsets_ry,
            min=1.0,
            max=float(self.h),
        )
        angle_candidates = torch.remainder(
            angle.unsqueeze(1) + float(sa) * self._neighbor_offsets_angle,
            180.0,
        )
        return x_candidates, y_candidates, rx_candidates, ry_candidates, angle_candidates

    def _rotated_ellipse_neighbor_count(self) -> int:
        return int(self._neighbor_offsets_x.numel())

    def _reduced_rotated_ellipse_neighbor_keep(
        self,
        sx: float,
        sy: float,
        srx: float,
        sry: float,
        sa: float,
    ) -> object | None:
        if not (self.enable_rotated_graph_default and self.alpha is None and self.w * self.h >= 280_000):
            return None
        if (
            self.rotated_ellipse_reduce_stage12_to9_after is not None
            and self.rotated_ellipse_shape_index >= self.rotated_ellipse_reduce_stage12_to9_after
            and abs(float(sa) - 8.0) < 1e-6
            and abs(float(sx) - 12.0) < 1e-6
            and abs(float(sy) - 12.0) < 1e-6
            and abs(float(srx) - 12.0) < 1e-6
            and abs(float(sry) - 12.0) < 1e-6
        ):
            return self._neighbor_keep_core
        if (
            abs(float(sa) - 16.0) < 1e-6
            and abs(float(sx) - max(12.0, float(max(self.w, self.h)) / 16.0)) < 1e-6
            and abs(float(sy) - max(12.0, float(max(self.w, self.h)) / 16.0)) < 1e-6
            and abs(float(srx) - max(16.0, float(max(self.w, self.h)) / 8.0)) < 1e-6
            and abs(float(sry) - max(16.0, float(max(self.w, self.h)) / 8.0)) < 1e-6
        ):
            return self._neighbor_keep_no_diag
        if (
            abs(float(sa) - 8.0) < 1e-6
            and abs(float(sx) - 12.0) < 1e-6
            and abs(float(sy) - 12.0) < 1e-6
            and abs(float(srx) - 12.0) < 1e-6
            and abs(float(sry) - 12.0) < 1e-6
        ):
            return self._neighbor_keep_no_diag
        if (
            self.rotated_ellipse_reduce_stage4_after is not None
            and self.rotated_ellipse_shape_index >= self.rotated_ellipse_reduce_stage4_after
            and abs(float(sa) - 4.0) < 1e-6
            and abs(float(sx) - 4.0) < 1e-6
            and abs(float(sy) - 4.0) < 1e-6
            and abs(float(srx) - 4.0) < 1e-6
            and abs(float(sry) - 4.0) < 1e-6
        ):
            return self._neighbor_keep_no_diag
        return None

    def _refine_rotated_ellipse_step_shortlist(
        self,
        x: object,
        y: object,
        rx: object,
        ry: object,
        angle: object,
        canvas_full_sq: float,
        canvas_norm: float,
        sx: float,
        sy: float,
        srx: float,
        sry: float,
        sa: float,
        coarse_stride: int,
        shortlist_size: int,
    ) -> tuple[object, object, object, object, object, object, object]:
        torch = self._torch
        count = int(x.shape[0])
        empty_float = torch.zeros(0, device=self.device, dtype=torch.float32)
        empty_int = torch.zeros((0, 4), device=self.device, dtype=torch.int32)
        if count == 0:
            return empty_float, empty_float, empty_float, empty_float, empty_float, empty_float, empty_int
        x = self._to_device_float_tensor(x)
        y = self._to_device_float_tensor(y)
        rx = self._to_device_float_tensor(rx)
        ry = self._to_device_float_tensor(ry)
        angle = self._to_device_float_tensor(angle)
        x_candidates, y_candidates, rx_candidates, ry_candidates, angle_candidates = (
            self._build_rotated_ellipse_neighbor_candidates(
                x,
                y,
                rx,
                ry,
                angle,
                sx,
                sy,
                srx,
                sry,
                sa,
            )
        )
        keep = self._reduced_rotated_ellipse_neighbor_keep(sx, sy, srx, sry, sa)
        if keep is not None:
            x_candidates = x_candidates.index_select(1, keep)
            y_candidates = y_candidates.index_select(1, keep)
            rx_candidates = rx_candidates.index_select(1, keep)
            ry_candidates = ry_candidates.index_select(1, keep)
            angle_candidates = angle_candidates.index_select(1, keep)
        neighbor_count = int(x_candidates.shape[1])
        total = count * neighbor_count
        coarse_scores, _ = self._score_rotated_ellipse_params_device(
            x_candidates.reshape(total),
            y_candidates.reshape(total),
            rx_candidates.reshape(total),
            ry_candidates.reshape(total),
            angle_candidates.reshape(total),
            None,
            canvas_full_sq,
            canvas_norm,
            sample_stride=coarse_stride,
            fixed_half_alpha=True,
        )
        shortlist = torch.topk(
            coarse_scores.view(count, neighbor_count),
            k=min(max(1, shortlist_size), neighbor_count),
            dim=1,
            largest=False,
        ).indices
        rows = torch.arange(count, device=self.device).unsqueeze(1)
        x_short = x_candidates[rows, shortlist]
        y_short = y_candidates[rows, shortlist]
        rx_short = rx_candidates[rows, shortlist]
        ry_short = ry_candidates[rows, shortlist]
        angle_short = angle_candidates[rows, shortlist]
        shortlist_total = int(x_short.shape[0]) * int(x_short.shape[1])
        return self._score_grouped_rotated_ellipse_params_device(
            x_short.reshape(shortlist_total),
            y_short.reshape(shortlist_total),
            rx_short.reshape(shortlist_total),
            ry_short.reshape(shortlist_total),
            angle_short.reshape(shortlist_total),
            None,
            canvas_full_sq,
            canvas_norm,
            int(x_short.shape[1]),
            sample_stride=1,
            fixed_half_alpha=True,
        )

    def _refine_rotated_ellipse_step_grouped(
        self,
        x: object,
        y: object,
        rx: object,
        ry: object,
        angle: object,
        canvas_full_sq: float,
        canvas_norm: float,
        sx: float,
        sy: float,
        srx: float,
        sry: float,
        sa: float,
        sample_stride: int = 1,
    ) -> tuple[object, object, object, object, object, object, object]:
        torch = self._torch
        count = int(x.shape[0])
        empty_float = torch.zeros(0, device=self.device, dtype=torch.float32)
        empty_int = torch.zeros((0, 4), device=self.device, dtype=torch.int32)
        if count == 0:
            return empty_float, empty_float, empty_float, empty_float, empty_float, empty_float, empty_int
        x = self._to_device_float_tensor(x)
        y = self._to_device_float_tensor(y)
        rx = self._to_device_float_tensor(rx)
        ry = self._to_device_float_tensor(ry)
        angle = self._to_device_float_tensor(angle)
        x_candidates, y_candidates, rx_candidates, ry_candidates, angle_candidates = (
            self._build_rotated_ellipse_neighbor_candidates(
                x,
                y,
                rx,
                ry,
                angle,
                sx,
                sy,
                srx,
                sry,
                sa,
            )
        )
        total = count * self._rotated_ellipse_neighbor_count()
        return self._score_grouped_rotated_ellipse_params_device(
            x_candidates.reshape(total),
            y_candidates.reshape(total),
            rx_candidates.reshape(total),
            ry_candidates.reshape(total),
            angle_candidates.reshape(total),
            None,
            canvas_full_sq,
            canvas_norm,
            self._rotated_ellipse_neighbor_count(),
            sample_stride=sample_stride,
            fixed_half_alpha=True,
        )

    def _refine_rotated_ellipse_step(
        self,
        x: object,
        y: object,
        rx: object,
        ry: object,
        angle: object,
        canvas_full_sq: float,
        canvas_norm: float,
        sx: float,
        sy: float,
        srx: float,
        sry: float,
        sa: float,
        sample_stride: int = 1,
    ) -> tuple[object, object, object, object, object, object, object]:
        torch = self._torch
        count = int(x.shape[0])
        empty_float = torch.zeros(0, device=self.device, dtype=torch.float32)
        empty_int = torch.zeros((0, 4), device=self.device, dtype=torch.int32)
        if count == 0:
            return empty_float, empty_float, empty_float, empty_float, empty_float, empty_float, empty_int
        x = self._to_device_float_tensor(x)
        y = self._to_device_float_tensor(y)
        rx = self._to_device_float_tensor(rx)
        ry = self._to_device_float_tensor(ry)
        angle = self._to_device_float_tensor(angle)
        x_candidates, y_candidates, rx_candidates, ry_candidates, angle_candidates = (
            self._build_rotated_ellipse_neighbor_candidates(
                x,
                y,
                rx,
                ry,
                angle,
                sx,
                sy,
                srx,
                sry,
                sa,
            )
        )
        keep = self._reduced_rotated_ellipse_neighbor_keep(sx, sy, srx, sry, sa)
        if keep is not None:
            x_candidates = x_candidates.index_select(1, keep)
            y_candidates = y_candidates.index_select(1, keep)
            rx_candidates = rx_candidates.index_select(1, keep)
            ry_candidates = ry_candidates.index_select(1, keep)
            angle_candidates = angle_candidates.index_select(1, keep)
        return self._score_rotated_ellipse_candidate_grid_exact(
            x_candidates,
            y_candidates,
            rx_candidates,
            ry_candidates,
            angle_candidates,
            canvas_full_sq,
            canvas_norm,
            sample_stride=sample_stride,
        )

    def _score_grouped_rotated_ellipse_params_device(
        self,
        x: object,
        y: object,
        rx: object,
        ry: object,
        angle: object,
        alpha_values: object | None,
        canvas_full_sq: float,
        canvas_norm: float,
        group_size: int,
        sample_stride: int = 1,
        fixed_half_alpha: bool = False,
    ) -> tuple[object, object, object, object, object, object, object]:
        count = int(x.shape[0])
        if group_size <= 0:
            raise ValueError("group_size must be positive")
        if count == 0:
            torch = self._torch
            empty_float = torch.zeros(0, device=self.device, dtype=torch.float32)
            empty_int = torch.zeros((0, 4), device=self.device, dtype=torch.int32)
            return empty_float, empty_float, empty_float, empty_float, empty_float, empty_float, empty_int
        if count % group_size != 0:
            raise ValueError("count must be divisible by group_size")
        torch = self._torch
        x_t = self._to_device_float_tensor(x)
        y_t = self._to_device_float_tensor(y)
        rx_t = self._to_device_float_tensor(rx)
        ry_t = self._to_device_float_tensor(ry)
        angle_t = self._to_device_float_tensor(angle)
        alpha_t = None if fixed_half_alpha and alpha_values is None else self._to_device_float_tensor(alpha_values)
        scores_out, colors_out = self._score_rotated_ellipse_params_device(
            x_t,
            y_t,
            rx_t,
            ry_t,
            angle_t,
            alpha_t,
            canvas_full_sq,
            canvas_norm,
            sample_stride=sample_stride,
            fixed_half_alpha=fixed_half_alpha,
        )
        group_count = count // group_size
        group_scores = scores_out.view(group_count, group_size)
        best_offsets = torch.argmin(group_scores, dim=1)
        flat_idx = best_offsets + torch.arange(group_count, device=self.device, dtype=torch.long) * group_size
        best_scores = scores_out.index_select(0, flat_idx)
        best_x = x_t.index_select(0, flat_idx)
        best_y = y_t.index_select(0, flat_idx)
        best_rx = rx_t.index_select(0, flat_idx)
        best_ry = ry_t.index_select(0, flat_idx)
        best_angle = angle_t.index_select(0, flat_idx)
        best_colors = colors_out.index_select(0, flat_idx)
        return (
            best_scores,
            best_x,
            best_y,
            best_rx,
            best_ry,
            best_angle,
            best_colors,
        )

    def _score_rotated_ellipse_params_gpu(
        self,
        x: np.ndarray,
        y: np.ndarray,
        rx: np.ndarray,
        ry: np.ndarray,
        angle: np.ndarray,
        alpha_values: np.ndarray,
        canvas_full_sq: float,
        canvas_norm: float,
        return_params: bool = False,
        sample_stride: int = 1,
    ) -> tuple[object, ...]:
        return self._score_rotated_ellipse_params_device(
            self._to_device_float_tensor(x),
            self._to_device_float_tensor(y),
            self._to_device_float_tensor(rx),
            self._to_device_float_tensor(ry),
            self._to_device_float_tensor(angle),
            self._to_device_float_tensor(alpha_values),
            canvas_full_sq,
            canvas_norm,
            return_params=return_params,
            sample_stride=sample_stride,
        )

    def _score_rotated_ellipse_params_device(
        self,
        x: object,
        y: object,
        rx: object,
        ry: object,
        angle: object,
        alpha_values: object | None,
        canvas_full_sq: float,
        canvas_norm: float,
        return_params: bool = False,
        sample_stride: int = 1,
        fixed_half_alpha: bool = False,
    ) -> tuple[object, ...]:
        count = int(x.shape[0])
        torch = self._torch
        if count == 0:
            scores_out = torch.full((0,), float("inf"), device=self.device, dtype=torch.float32)
            colors_out = torch.zeros((0, 4), device=self.device, dtype=torch.int32)
            if return_params:
                empty = torch.zeros((0, 6), device=self.device, dtype=torch.float32)
                return scores_out, colors_out, empty
            return scores_out, colors_out
        x_t = self._to_device_float_tensor(x)
        y_t = self._to_device_float_tensor(y)
        rx_t = torch.clamp(self._to_device_float_tensor(rx), min=1e-6)
        ry_t = torch.clamp(self._to_device_float_tensor(ry), min=1e-6)
        angle_t = self._to_device_float_tensor(angle)
        alpha_t = None if fixed_half_alpha and alpha_values is None else self._to_device_float_tensor(alpha_values)
        angle_rad = angle_t * (math.pi / 180.0)
        if (
            fixed_half_alpha
            and alpha_t is None
            and self.alpha is None
            and not return_params
            and self._ensure_rotated_ellipse_cupy()
        ):
            return self._score_rotated_ellipse_params_device_cupy_fixed(
                x_t,
                y_t,
                rx_t,
                ry_t,
                angle_t,
                canvas_full_sq,
                canvas_norm,
                sample_stride=sample_stride,
            )
        abs_cos = torch.abs(torch.cos(angle_rad))
        abs_sin = torch.abs(torch.sin(angle_rad))
        ext_x = torch.sqrt((rx_t * abs_cos) * (rx_t * abs_cos) + (ry_t * abs_sin) * (ry_t * abs_sin))
        ext_y = torch.sqrt((rx_t * abs_sin) * (rx_t * abs_sin) + (ry_t * abs_cos) * (ry_t * abs_cos))
        x0 = torch.clamp(torch.floor(x_t - ext_x).to(torch.long), min=0)
        y0 = torch.clamp(torch.floor(y_t - ext_y).to(torch.long), min=0)
        x1 = torch.clamp(torch.ceil(x_t + ext_x + 1.0).to(torch.long), max=self.w)
        y1 = torch.clamp(torch.ceil(y_t + ext_y + 1.0).to(torch.long), max=self.h)
        width = torch.clamp(x1 - x0, min=0)
        height = torch.clamp(y1 - y0, min=0)
        self._canvas_full_sq_scalar.fill_(float(canvas_full_sq))
        pixel_budget = max(1, self.max_batch_pixels * max(1, sample_stride) * max(1, sample_stride))
        widths_cpu, heights_cpu, area_cpu, max_w, max_h = self._width_height_area_cpu(width, height)
        if max_w * max_h * count <= pixel_budget:
            scores, colors = self._score_rotated_ellipse_batch(
                x0,
                y0,
                width,
                height,
                x_t,
                y_t,
                rx_t,
                ry_t,
                angle_rad,
                alpha_t,
                canvas_norm,
                max_w,
                max_h,
                sample_stride,
                fixed_half_alpha=fixed_half_alpha,
            )
            if return_params:
                if alpha_t is None:
                    alpha_params = torch.full((count,), 128.0, device=self.device, dtype=torch.float32)
                else:
                    alpha_params = alpha_t
                params_t = torch.stack((x_t, y_t, rx_t, ry_t, angle_t, alpha_params), dim=1)
                return scores, colors, params_t
            return scores, colors
        scores_out = torch.full((count,), float("inf"), device=self.device, dtype=torch.float32)
        colors_out = torch.zeros((count, 4), device=self.device, dtype=torch.int32)
        indices_cpu = np.argsort(-area_cpu, kind="stable")
        indices = torch.as_tensor(indices_cpu, device=self.device, dtype=torch.long)
        x0_all = x0.index_select(0, indices)
        y0_all = y0.index_select(0, indices)
        width_all = width.index_select(0, indices)
        height_all = height.index_select(0, indices)
        shape_x_all = x_t.index_select(0, indices)
        shape_y_all = y_t.index_select(0, indices)
        rx_all = rx_t.index_select(0, indices)
        ry_all = ry_t.index_select(0, indices)
        angle_all = angle_t.index_select(0, indices) * (math.pi / 180.0)
        if alpha_t is None:
            alpha_stack = torch.full((count,), 128.0, device=self.device, dtype=torch.float32)
        else:
            alpha_stack = alpha_t
        alpha_all = alpha_stack.index_select(0, indices)
        start = 0
        while start < count:
            batch_area = max(1, int(area_cpu[indices_cpu[start]]))
            batch_len = max(1, pixel_budget // batch_area)
            end = min(count, start + batch_len)
            max_w = int(widths_cpu[indices_cpu[start:end]].max())
            max_h = int(heights_cpu[indices_cpu[start:end]].max())
            scores, colors = self._score_rotated_ellipse_batch(
                x0_all[start:end],
                y0_all[start:end],
                width_all[start:end],
                height_all[start:end],
                shape_x_all[start:end],
                shape_y_all[start:end],
                rx_all[start:end],
                ry_all[start:end],
                angle_all[start:end],
                alpha_all[start:end],
                canvas_norm,
                max_w,
                max_h,
                sample_stride,
                fixed_half_alpha=fixed_half_alpha,
            )
            scores_out.index_copy_(0, indices[start:end], scores)
            colors_out.index_copy_(0, indices[start:end], colors)
            start = end
        if return_params:
            if alpha_t is None:
                alpha_params = torch.full((count,), 128.0, device=self.device, dtype=torch.float32)
            else:
                alpha_params = alpha_t
            params_t = torch.stack((x_t, y_t, rx_t, ry_t, angle_t, alpha_params), dim=1)
            return scores_out, colors_out, params_t
        return scores_out, colors_out

    def _width_height_area_cpu(
        self,
        width: object,
        height: object,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
        torch = self._torch
        widths_cpu = width.to(device="cpu", dtype=torch.int32).numpy().astype(np.int64, copy=False)
        heights_cpu = height.to(device="cpu", dtype=torch.int32).numpy().astype(np.int64, copy=False)
        area_cpu = np.maximum(0, widths_cpu * heights_cpu)
        max_w = int(widths_cpu.max()) if widths_cpu.size else 0
        max_h = int(heights_cpu.max()) if heights_cpu.size else 0
        return widths_cpu, heights_cpu, area_cpu, max_w, max_h

    def _group_needs_sort(
        self,
        areas: list[int],
        pixel_limit: int | None = None,
        widths: list[int] | None = None,
        heights: list[int] | None = None,
    ) -> bool:
        if len(areas) <= 1:
            return False
        limit = self.max_batch_pixels if pixel_limit is None else max(1, int(pixel_limit))
        if widths is not None and heights is not None:
            return max(widths) * max(heights) * len(areas) > limit
        return max(areas) * len(areas) > limit

    def _split_batch_ranges(
        self,
        areas: list[int],
        pixel_limit: int | None = None,
        widths: list[int] | None = None,
        heights: list[int] | None = None,
    ) -> list[tuple[int, int]]:
        limit = self.max_batch_pixels if pixel_limit is None else max(1, int(pixel_limit))
        ranges: list[tuple[int, int]] = []
        start = 0
        max_area = 0
        max_w = 0
        max_h = 0
        for pos, raw_area in enumerate(areas):
            area = max(1, raw_area)
            next_max = max(max_area, area)
            if widths is not None and heights is not None:
                next_max_w = max(max_w, max(1, widths[pos]))
                next_max_h = max(max_h, max(1, heights[pos]))
                footprint = next_max_w * next_max_h * (pos - start + 1)
            else:
                next_max_w = max_w
                next_max_h = max_h
                footprint = next_max * (pos - start + 1)
            if pos > start and footprint > limit:
                ranges.append((start, pos))
                start = pos
                max_area = area
                max_w = max(1, widths[pos]) if widths is not None else 0
                max_h = max(1, heights[pos]) if heights is not None else 0
            else:
                max_area = next_max
                max_w = next_max_w
                max_h = next_max_h
        if start < len(areas):
            ranges.append((start, len(areas)))
        return ranges

    def _pack_shape_group(
        self,
        type_name: str,
        items: list[tuple[int, Shape, tuple[int, int, int, int], int, int, int]],
    ) -> dict[str, object]:
        torch = self._torch
        count = len(items)
        indices_np = np.empty(count, dtype=np.int64)
        x0_np = np.empty(count, dtype=np.int64)
        y0_np = np.empty(count, dtype=np.int64)
        width_np = np.empty(count, dtype=np.int64)
        height_np = np.empty(count, dtype=np.int64)
        fixed_half_alpha = items[0][1].color[3] == 128
        alpha_scale_np = None if fixed_half_alpha else np.empty(count, dtype=np.float32)
        alpha_int_np = None if fixed_half_alpha else np.empty(count, dtype=np.int32)
        if type_name in {"circle", "ellipse", "rotated_ellipse", "rotated_rectangle"}:
            shape_x_np = np.empty(count, dtype=np.float32)
            shape_y_np = np.empty(count, dtype=np.float32)
        if type_name == "circle":
            radius_np = np.empty(count, dtype=np.float32)
        if type_name in {"ellipse", "rotated_ellipse"}:
            inv_radius_x_sq_np = np.empty(count, dtype=np.float32)
            inv_radius_y_sq_np = np.empty(count, dtype=np.float32)
        if type_name in {"rotated_ellipse", "rotated_rectangle"}:
            cos_angle_np = np.empty(count, dtype=np.float32)
            sin_angle_np = np.empty(count, dtype=np.float32)
        if type_name == "rotated_rectangle":
            half_width_np = np.empty(count, dtype=np.float32)
            half_height_np = np.empty(count, dtype=np.float32)
        if type_name == "triangle":
            x1_np = np.empty(count, dtype=np.float32)
            y1_np = np.empty(count, dtype=np.float32)
            x2_minus_x1_np = np.empty(count, dtype=np.float32)
            y2_minus_y1_np = np.empty(count, dtype=np.float32)
            x3_minus_x2_np = np.empty(count, dtype=np.float32)
            y3_minus_y2_np = np.empty(count, dtype=np.float32)
            x1_minus_x3_np = np.empty(count, dtype=np.float32)
            y1_minus_y3_np = np.empty(count, dtype=np.float32)
            x2_np = np.empty(count, dtype=np.float32)
            y2_np = np.empty(count, dtype=np.float32)
            x3_np = np.empty(count, dtype=np.float32)
            y3_np = np.empty(count, dtype=np.float32)
        for pos, (idx, shape, bbox, _area, width, height) in enumerate(items):
            x0, y0, x1, y1 = bbox
            indices_np[pos] = idx
            x0_np[pos] = x0
            y0_np[pos] = y0
            width_np[pos] = width
            height_np[pos] = height
            if alpha_scale_np is not None:
                alpha = int(shape.color[3])
                alpha_scale_np[pos] = float(alpha / 255.0)
                alpha_int_np[pos] = alpha
            if type_name == "circle":
                shape_x_np[pos] = float(shape.x)
                shape_y_np[pos] = float(shape.y)
                radius_np[pos] = _get_circle_radius_sq(shape)
            elif type_name == "ellipse":
                shape_x_np[pos] = float(shape.x)
                shape_y_np[pos] = float(shape.y)
                inv_radius_x_sq_np[pos], inv_radius_y_sq_np[pos] = _get_ellipse_inverse_radii(shape)
            elif type_name == "rotated_ellipse":
                shape_x_np[pos] = float(shape.x)
                shape_y_np[pos] = float(shape.y)
                (
                    inv_radius_x_sq_np[pos],
                    inv_radius_y_sq_np[pos],
                    cos_angle_np[pos],
                    sin_angle_np[pos],
                ) = _get_rotated_ellipse_metrics(shape)
            elif type_name == "rotated_rectangle":
                shape_x_np[pos] = float(shape.x)
                shape_y_np[pos] = float(shape.y)
                half_width_np[pos] = float(shape.hw)
                half_height_np[pos] = float(shape.hh)
                cos_angle_np[pos], sin_angle_np[pos] = _get_rotated_rectangle_angle_metrics(shape)
            elif type_name == "triangle":
                x1 = float(shape.x1)
                y1 = float(shape.y1)
                x2 = float(shape.x2)
                y2 = float(shape.y2)
                x3 = float(shape.x3)
                y3 = float(shape.y3)
                x1_np[pos] = x1
                y1_np[pos] = y1
                x2_np[pos] = x2
                y2_np[pos] = y2
                x3_np[pos] = x3
                y3_np[pos] = y3
                x2_minus_x1_np[pos] = x2 - x1
                y2_minus_y1_np[pos] = y2 - y1
                x3_minus_x2_np[pos] = x3 - x2
                y3_minus_y2_np[pos] = y3 - y2
                x1_minus_x3_np[pos] = x1 - x3
                y1_minus_y3_np[pos] = y1 - y3
        common_np = np.stack((indices_np, x0_np, y0_np, width_np, height_np), axis=1)
        common_t = torch.as_tensor(common_np, device=self.device, dtype=torch.long)
        x0_t = common_t[:, 1]
        y0_t = common_t[:, 2]
        packed: dict[str, object] = {
            "indices": common_t[:, 0],
            "x0": x0_t,
            "x0_float": x0_t.to(dtype=torch.float32),
            "y0": y0_t,
            "y0_float": y0_t.to(dtype=torch.float32),
            "width": common_t[:, 3],
            "width_cpu": width_np,
            "height": common_t[:, 4],
            "height_cpu": height_np,
            "fixed_half_alpha": fixed_half_alpha,
        }
        if alpha_scale_np is not None:
            alpha_np = np.stack((alpha_scale_np, alpha_int_np.astype(np.float32)), axis=1)
            alpha_t = torch.as_tensor(alpha_np, device=self.device, dtype=torch.float32)
            packed["alpha_scale"] = alpha_t[:, 0]
            packed["alpha_int"] = alpha_t[:, 1].to(dtype=torch.int32)
        if type_name == "circle":
            circle_t = torch.as_tensor(
                np.stack((shape_x_np, shape_y_np, radius_np), axis=1),
                device=self.device,
                dtype=torch.float32,
            )
            packed["shape_x"] = circle_t[:, 0]
            packed["shape_y"] = circle_t[:, 1]
            packed["radius"] = circle_t[:, 2]
        elif type_name == "ellipse":
            ellipse_t = torch.as_tensor(
                np.stack((shape_x_np, shape_y_np, inv_radius_x_sq_np, inv_radius_y_sq_np), axis=1),
                device=self.device,
                dtype=torch.float32,
            )
            packed["shape_x"] = ellipse_t[:, 0]
            packed["shape_y"] = ellipse_t[:, 1]
            packed["inv_radius_x_sq"] = ellipse_t[:, 2]
            packed["inv_radius_y_sq"] = ellipse_t[:, 3]
        elif type_name == "rotated_ellipse":
            rot_ellipse_t = torch.as_tensor(
                np.stack(
                    (
                        shape_x_np,
                        shape_y_np,
                        inv_radius_x_sq_np,
                        inv_radius_y_sq_np,
                        cos_angle_np,
                        sin_angle_np,
                    ),
                    axis=1,
                ),
                device=self.device,
                dtype=torch.float32,
            )
            packed["shape_x"] = rot_ellipse_t[:, 0]
            packed["shape_y"] = rot_ellipse_t[:, 1]
            packed["inv_radius_x_sq"] = rot_ellipse_t[:, 2]
            packed["inv_radius_y_sq"] = rot_ellipse_t[:, 3]
            packed["cos_angle"] = rot_ellipse_t[:, 4]
            packed["sin_angle"] = rot_ellipse_t[:, 5]
        elif type_name == "rotated_rectangle":
            rot_rect_t = torch.as_tensor(
                np.stack((shape_x_np, shape_y_np, half_width_np, half_height_np, cos_angle_np, sin_angle_np), axis=1),
                device=self.device,
                dtype=torch.float32,
            )
            packed["shape_x"] = rot_rect_t[:, 0]
            packed["shape_y"] = rot_rect_t[:, 1]
            packed["half_width"] = rot_rect_t[:, 2]
            packed["half_height"] = rot_rect_t[:, 3]
            packed["cos_angle"] = rot_rect_t[:, 4]
            packed["sin_angle"] = rot_rect_t[:, 5]
        if type_name == "triangle":
            tri_t = torch.as_tensor(
                np.stack((
                    x1_np,
                    y1_np,
                    x2_np,
                    y2_np,
                    x3_np,
                    y3_np,
                    x2_minus_x1_np,
                    y2_minus_y1_np,
                    x3_minus_x2_np,
                    y3_minus_y2_np,
                    x1_minus_x3_np,
                    y1_minus_y3_np,
                ), axis=1),
                device=self.device,
                dtype=torch.float32,
            )
            packed["x1"] = tri_t[:, 0]
            packed["y1"] = tri_t[:, 1]
            packed["x2"] = tri_t[:, 2]
            packed["y2"] = tri_t[:, 3]
            packed["x3"] = tri_t[:, 4]
            packed["y3"] = tri_t[:, 5]
            packed["x2_minus_x1"] = tri_t[:, 6]
            packed["y2_minus_y1"] = tri_t[:, 7]
            packed["x3_minus_x2"] = tri_t[:, 8]
            packed["y3_minus_y2"] = tri_t[:, 9]
            packed["x1_minus_x3"] = tri_t[:, 10]
            packed["y1_minus_y3"] = tri_t[:, 11]
        return packed

    def _score_rotated_ellipse_group(
        self,
        items: list[tuple[int, Shape, tuple[int, int, int, int], int, int, int]],
        scores_out: object,
        colors_out: object,
        canvas_norm: float,
    ) -> None:
        torch = self._torch
        areas = [max(1, item[3]) for item in items]
        widths = [max(1, item[4]) for item in items]
        heights = [max(1, item[5]) for item in items]
        footprints = [widths[pos] * heights[pos] for pos in range(len(items))]
        fixed_half_alpha = items[0][1].color[3] == 128
        pixel_limit = self.max_batch_pixels
        if self._group_needs_sort(areas, pixel_limit, widths, heights):
            order = sorted(range(len(items)), key=footprints.__getitem__, reverse=True)
            items = [items[pos] for pos in order]
        count = len(items)
        indices_np = np.empty(count, dtype=np.int64)
        x0_np = np.empty(count, dtype=np.int64)
        y0_np = np.empty(count, dtype=np.int64)
        width_np = np.empty(count, dtype=np.int64)
        height_np = np.empty(count, dtype=np.int64)
        shape_x_np = np.empty(count, dtype=np.float32)
        shape_y_np = np.empty(count, dtype=np.float32)
        inv_rx_sq_np = np.empty(count, dtype=np.float32)
        inv_ry_sq_np = np.empty(count, dtype=np.float32)
        cos_angle_np = np.empty(count, dtype=np.float32)
        sin_angle_np = np.empty(count, dtype=np.float32)
        alpha_scale_np = None if fixed_half_alpha else np.empty(count, dtype=np.float32)
        alpha_int_np = None if fixed_half_alpha else np.empty(count, dtype=np.int32)
        for pos, (idx, shape, bbox, _area, width, height) in enumerate(items):
            x0, y0, x1, y1 = bbox
            indices_np[pos] = idx
            x0_np[pos] = x0
            y0_np[pos] = y0
            width_np[pos] = width
            height_np[pos] = height
            shape_x_np[pos] = float(shape.x)
            shape_y_np[pos] = float(shape.y)
            (
                inv_rx_sq_np[pos],
                inv_ry_sq_np[pos],
                cos_angle_np[pos],
                sin_angle_np[pos],
            ) = _get_rotated_ellipse_metrics(shape)
            if alpha_scale_np is not None:
                alpha = int(shape.color[3])
                alpha_scale_np[pos] = float(alpha / 255.0)
                alpha_int_np[pos] = alpha
        common_t = torch.as_tensor(
            np.stack((indices_np, x0_np, y0_np, width_np, height_np), axis=1),
            device=self.device,
            dtype=torch.long,
        )
        geom_t = torch.as_tensor(
            np.stack(
                (
                    shape_x_np,
                    shape_y_np,
                    inv_rx_sq_np,
                    inv_ry_sq_np,
                    cos_angle_np,
                    sin_angle_np,
                ),
                axis=1,
            ),
            device=self.device,
            dtype=torch.float32,
        )
        indices = common_t[:, 0]
        x0_all = common_t[:, 1]
        y0_all = common_t[:, 2]
        width_all = common_t[:, 3]
        height_all = common_t[:, 4]
        shape_x_all = geom_t[:, 0]
        shape_y_all = geom_t[:, 1]
        inv_rx_sq_all = geom_t[:, 2]
        inv_ry_sq_all = geom_t[:, 3]
        cos_angle_all = geom_t[:, 4]
        sin_angle_all = geom_t[:, 5]
        alpha_scale_all = None
        alpha_int_all = None
        if alpha_scale_np is not None:
            alpha_t = torch.as_tensor(
                np.stack((alpha_scale_np, alpha_int_np.astype(np.float32)), axis=1),
                device=self.device,
                dtype=torch.float32,
            )
            alpha_scale_all = alpha_t[:, 0]
            alpha_int_all = alpha_t[:, 1].to(dtype=torch.int32)
        start = 0
        while start < count:
            end = start
            max_w = 0
            max_h = 0
            while end < count:
                next_max_w = max(max_w, max(1, int(width_np[end])))
                next_max_h = max(max_h, max(1, int(height_np[end])))
                if end > start and next_max_w * next_max_h * (end - start + 1) > pixel_limit:
                    break
                max_w = next_max_w
                max_h = next_max_h
                end += 1
            batch_max_w = max_w if end > start else 0
            batch_max_h = max_h if end > start else 0
            scores, colors = self._score_rotated_ellipse_batch(
                x0_all[start:end],
                y0_all[start:end],
                width_all[start:end],
                height_all[start:end],
                shape_x_all[start:end],
                shape_y_all[start:end],
                None,
                None,
                None,
                None,
                canvas_norm,
                batch_max_w,
                batch_max_h,
                fixed_half_alpha=fixed_half_alpha,
                inv_rx_sq=inv_rx_sq_all[start:end],
                inv_ry_sq=inv_ry_sq_all[start:end],
                cos_angle=cos_angle_all[start:end],
                sin_angle=sin_angle_all[start:end],
                alpha_scale_values=None if alpha_scale_all is None else alpha_scale_all[start:end],
                alpha_int_values=None if alpha_int_all is None else alpha_int_all[start:end],
            )
            scores_out.index_copy_(0, indices[start:end], scores)
            colors_out.index_copy_(0, indices[start:end], colors)
            start = end

    def _score_rotated_ellipse_batch(
        self,
        x0: object,
        y0: object,
        width_t: object,
        height_t: object,
        shape_x: object,
        shape_y: object,
        rx: object | None,
        ry: object | None,
        angle: object | None,
        alpha_values: object | None,
        canvas_norm: float,
        max_w: int,
        max_h: int,
        sample_stride: int = 1,
        fixed_half_alpha: bool = False,
        inv_rx_sq: object | None = None,
        inv_ry_sq: object | None = None,
        cos_angle: object | None = None,
        sin_angle: object | None = None,
        alpha_scale_values: object | None = None,
        alpha_int_values: object | None = None,
    ) -> tuple[object, object]:
        torch = self._torch
        batch_size = int(x0.shape[0])
        if batch_size == 0:
            return (
                torch.full((0,), float("inf"), device=self.device, dtype=torch.float32),
                torch.zeros((0, 4), device=self.device, dtype=torch.int32),
            )
        if max_w <= 0 or max_h <= 0:
            return (
                torch.full((batch_size,), float("inf"), device=self.device, dtype=torch.float32),
                torch.zeros((batch_size, 4), device=self.device, dtype=torch.int32),
            )
        stride = max(1, int(sample_stride))
        x_len = max(1, (max_w + stride - 1) // stride)
        y_len = max(1, (max_h + stride - 1) // stride)
        x_base_i = (self._long_arange(x_len) * stride).view(1, 1, x_len)
        y_base_i = (self._long_arange(y_len) * stride).view(1, y_len, 1)
        x_base_f = self._float_arange(x_len, stride).view(1, 1, x_len)
        y_base_f = self._float_arange(y_len, stride).view(1, y_len, 1)
        valid = (x_base_i < width_t.view(-1, 1, 1)) & (y_base_i < height_t.view(-1, 1, 1))
        x_idx = (x0.view(-1, 1, 1) + x_base_i).clamp(max=self.w - 1)
        y_idx = (y0.view(-1, 1, 1) + y_base_i).clamp(max=self.h - 1)
        flat_idx = (y_idx * self.w + x_idx).reshape(-1)
        grid_shape = (x_idx.shape[0], y_idx.shape[1], x_idx.shape[2])
        with nullcontext():
            region_old_sq_base = self._canvas_minus_target_sqsum_flat.index_select(0, flat_idx).view(grid_shape[0], grid_shape[1], grid_shape[2])
            if fixed_half_alpha:
                region_d_base = self._target_minus_half_canvas_flat.index_select(0, flat_idx).view(grid_shape[0], grid_shape[1], grid_shape[2], 3)
                region_d_sq_base = self._target_minus_half_canvas_sqsum_flat.index_select(0, flat_idx).view(grid_shape[0], grid_shape[1], grid_shape[2])
            else:
                region_cur = self.canvas_flat.index_select(0, flat_idx).view(grid_shape[0], grid_shape[1], grid_shape[2], 3)
                region_tgt = self.target_flat.index_select(0, flat_idx).view(grid_shape[0], grid_shape[1], grid_shape[2], 3)
            x_abs = x0.view(-1, 1, 1) + x_base_f
            y_abs = y0.view(-1, 1, 1) + y_base_f
            x_rel = x_abs - shape_x.view(-1, 1, 1)
            y_rel = y_abs - shape_y.view(-1, 1, 1)
            if cos_angle is None or sin_angle is None:
                if angle is None:
                    raise ValueError("angle is required when cos_angle and sin_angle are not provided")
                cos_a = torch.cos(angle.view(-1, 1, 1))
                sin_a = torch.sin(angle.view(-1, 1, 1))
            else:
                cos_a = cos_angle.view(-1, 1, 1)
                sin_a = sin_angle.view(-1, 1, 1)
            xr = cos_a * x_rel + sin_a * y_rel
            yr = -sin_a * x_rel + cos_a * y_rel
            if inv_rx_sq is None or inv_ry_sq is None:
                if rx is None or ry is None:
                    raise ValueError("rx and ry are required when inverse radii are not provided")
                inv_rx_sq_t = torch.reciprocal((rx * rx).view(-1, 1, 1))
                inv_ry_sq_t = torch.reciprocal((ry * ry).view(-1, 1, 1))
            else:
                inv_rx_sq_t = inv_rx_sq.view(-1, 1, 1)
                inv_ry_sq_t = inv_ry_sq.view(-1, 1, 1)
            mask = (((xr * xr) * inv_rx_sq_t) + ((yr * yr) * inv_ry_sq_t)) <= 1.0
            mask = mask & valid
            if fixed_half_alpha:
                alpha_scale = self._half_alpha_scale
                alpha_scale_rgb = self._half_alpha_scale.view(1, 1)
            else:
                if alpha_scale_values is None and alpha_values is None:
                    raise ValueError("alpha_values is required when fixed_half_alpha is False")
                if alpha_scale_values is None:
                    alpha_scale = (alpha_values / 255.0).view(-1, 1, 1, 1)
                else:
                    alpha_scale = alpha_scale_values.view(-1, 1, 1, 1)
                alpha_scale_rgb = alpha_scale.view(-1, 1)
            valid_scores = (width_t > 0) & (height_t > 0)
            if fixed_half_alpha:
                d = region_d_base
            else:
                d = region_tgt - (1.0 - alpha_scale) * region_cur
            if self.alpha is None:
                weight = mask.sum(dim=(1, 2), dtype=torch.float32)
                valid_scores = valid_scores & (weight >= 0.5)
                if not fixed_half_alpha:
                    if alpha_int_values is not None:
                        valid_scores = valid_scores & (alpha_int_values > 0)
                    else:
                        valid_scores = valid_scores & (alpha_values > 0.0)
                safe_weight = weight.clamp_min(1e-6).view(-1, 1)
                mask_float = mask.unsqueeze(-1)
                sum_d = (d * mask_float).sum(dim=(1, 2))
                rgb_float = torch.floor(torch.clamp(sum_d / (alpha_scale_rgb * safe_weight), 0.0, 255.0))
                rgb = rgb_float.to(torch.int32)
                if fixed_half_alpha:
                    alpha_int = torch.full((rgb.shape[0], 1), 128, device=self.device, dtype=torch.int32)
                else:
                    if alpha_int_values is None:
                        alpha_int = alpha_values.to(torch.int32).unsqueeze(-1)
                    else:
                        alpha_int = alpha_int_values.unsqueeze(-1)
                colors = torch.cat([rgb, alpha_int], dim=1)
                applied = alpha_scale_rgb * rgb_float
                if fixed_half_alpha:
                    sum_d_sq_total = (region_d_sq_base * mask).sum(dim=(1, 2), dtype=torch.float32)
                else:
                    sum_d_sq = ((d * d) * mask_float).sum(dim=(1, 2))
                    sum_d_sq_total = sum_d_sq.sum(dim=1)
                inside_new_sq = sum_d_sq_total - 2.0 * (applied * sum_d).sum(dim=1) + weight * (applied * applied).sum(dim=1)
                if fixed_half_alpha:
                    inside_old_sq = (region_old_sq_base * mask).sum(dim=(1, 2), dtype=torch.float32)
                else:
                    inside_old_sq = (region_old_sq_base * mask).sum(dim=(1, 2), dtype=torch.float32)
            else:
                color_mask = mask.to(torch.float32)
                region_alpha_inside = self.alpha_inside_flat.index_select(0, flat_idx).view(grid_shape)
                region_alpha_scale = self.alpha_scale_flat.index_select(0, flat_idx).view(grid_shape)
                region_alpha_nonzero = self.alpha_nonzero_flat.index_select(0, flat_idx).view(grid_shape)
                body_total = mask.sum(dim=(1, 2), dtype=torch.float32)
                inside = (mask & region_alpha_inside).sum(dim=(1, 2), dtype=torch.float32)
                overlap_ok = (body_total > 0) & (inside >= body_total * STICKER_OVERLAP_MIN)
                effective_mask = color_mask * region_alpha_scale
                valid_scores = valid_scores & (body_total > 0) & overlap_ok
                weight = effective_mask.sum(dim=(1, 2))
                valid_scores = valid_scores & (weight >= 0.5)
                if not fixed_half_alpha:
                    if alpha_int_values is not None:
                        valid_scores = valid_scores & (alpha_int_values > 0)
                    else:
                        valid_scores = valid_scores & (alpha_values > 0.0)
                safe_weight = weight.clamp_min(1e-6).view(-1, 1)
                mask_float = color_mask.unsqueeze(-1)
                weighted_mask = region_alpha_scale.unsqueeze(-1) * mask_float
                sum_d = (d * weighted_mask).sum(dim=(1, 2))
                rgb_float = torch.floor(torch.clamp((sum_d / safe_weight) / alpha_scale_rgb, 0.0, 255.0))
                rgb = rgb_float.to(torch.int32)
                if fixed_half_alpha:
                    alpha_int = torch.full((rgb.shape[0], 1), 128, device=self.device, dtype=torch.int32)
                else:
                    if alpha_int_values is None:
                        alpha_int = alpha_values.to(torch.int32).unsqueeze(-1)
                    else:
                        alpha_int = alpha_int_values.unsqueeze(-1)
                colors = torch.cat([rgb, alpha_int], dim=1)
                applied = alpha_scale_rgb * rgb_float
                allowed_mask_f = (region_alpha_nonzero * color_mask).unsqueeze(-1)
                sum_d_allowed = (d * allowed_mask_f).sum(dim=(1, 2))
                sum_d_sq_allowed = ((d * d) * allowed_mask_f).sum(dim=(1, 2))
                allowed_count = allowed_mask_f[..., 0].sum(dim=(1, 2)).clamp_min(1e-6)
                inside_new_sq = (
                    sum_d_sq_allowed.sum(dim=1)
                    - 2.0 * (applied * sum_d_allowed).sum(dim=1)
                    + allowed_count * (applied * applied).sum(dim=1)
                )
                inside_old_sq = (region_old_sq_base * allowed_mask_f[..., 0]).sum(dim=(1, 2))
            sample_scale = float(stride * stride)
            total_sq = (self._canvas_full_sq_scalar / sample_scale) - inside_old_sq + inside_new_sq
            denom = max(canvas_norm / sample_scale, 1.0)
            scores = torch.sqrt(torch.clamp(total_sq, min=0.0) / denom)
            scores = scores.masked_fill(~valid_scores, float("inf"))
        return scores, colors

    def _score_batch(
        self,
        type_name: str,
        packed: dict[str, object],
        start: int,
        end: int,
        canvas_norm: float,
    ) -> tuple[object, object]:
        torch = self._torch
        batch_size = end - start
        scores_out = torch.full((batch_size,), float("inf"), device=self.device, dtype=torch.float32)
        colors_out = torch.zeros((batch_size, 4), device=self.device, dtype=torch.int32)
        if batch_size == 0:
            return scores_out, colors_out
        width_cpu = packed["width_cpu"][start:end]
        height_cpu = packed["height_cpu"][start:end]
        max_w = int(width_cpu.max()) if width_cpu.size else 0
        max_h = int(height_cpu.max()) if height_cpu.size else 0
        if max_w <= 0 or max_h <= 0:
            return scores_out, colors_out
        x0 = packed["x0"][start:end]
        y0 = packed["y0"][start:end]
        width_t = packed["width"][start:end]
        height_t = packed["height"][start:end]
        x_base_i = self._long_arange(max_w).view(1, 1, max_w)
        y_base_i = self._long_arange(max_h).view(1, max_h, 1)
        valid = (x_base_i < width_t.view(-1, 1, 1)) & (y_base_i < height_t.view(-1, 1, 1))
        x_idx = (x0.view(-1, 1, 1) + x_base_i).clamp(max=self.w - 1)
        y_idx = (y0.view(-1, 1, 1) + y_base_i).clamp(max=self.h - 1)
        flat_idx = (y_idx * self.w + x_idx).reshape(-1)
        grid_shape = (x_idx.shape[0], y_idx.shape[1], x_idx.shape[2])
        region_old_sq_base = self._canvas_minus_target_sqsum_flat.index_select(0, flat_idx).view(grid_shape[0], grid_shape[1], grid_shape[2])
        fixed_half_alpha = bool(packed.get("fixed_half_alpha", False))
        if fixed_half_alpha:
            region_d_base = self._target_minus_half_canvas_flat.index_select(0, flat_idx).view(grid_shape[0], grid_shape[1], grid_shape[2], 3)
            region_d_sq_base = self._target_minus_half_canvas_sqsum_flat.index_select(0, flat_idx).view(grid_shape[0], grid_shape[1], grid_shape[2])
        else:
            region_cur = self.canvas_flat.index_select(0, flat_idx).view(grid_shape[0], grid_shape[1], grid_shape[2], 3)
            region_tgt = self.target_flat.index_select(0, flat_idx).view(grid_shape[0], grid_shape[1], grid_shape[2], 3)
        if type_name == "rectangle":
            mask = valid
        else:
            x_base_f = self._float_arange(max_w, 1).view(1, 1, max_w)
            y_base_f = self._float_arange(max_h, 1).view(1, max_h, 1)
            x0_float = packed["x0_float"][start:end]
            y0_float = packed["y0_float"][start:end]
            x_abs = x0_float.view(-1, 1, 1) + x_base_f
            y_abs = y0_float.view(-1, 1, 1) + y_base_f
            mask = self._build_mask(type_name, packed, start, end, x_abs, y_abs, valid)
        alpha_scale_tensor = packed.get("alpha_scale")
        alpha_scale_values = None if alpha_scale_tensor is None else alpha_scale_tensor[start:end]
        alpha_int_tensor = packed.get("alpha_int")
        alpha_int_values = None if alpha_int_tensor is None else alpha_int_tensor[start:end]
        valid_scores = (width_t > 0) & (height_t > 0)
        if fixed_half_alpha:
            alpha_scale_rgb = self._half_alpha_scale.view(1, 1)
            d = region_d_base
        else:
            if alpha_scale_values is None or alpha_int_values is None:
                raise ValueError("alpha values are required when fixed_half_alpha is False")
            alpha_scale = alpha_scale_values.view(-1, 1, 1, 1)
            alpha_scale_rgb = alpha_scale.view(-1, 1)
            d = region_tgt - (1.0 - alpha_scale) * region_cur
        if self.alpha is not None:
            color_mask = mask.to(torch.float32)
            region_alpha_inside = self.alpha_inside_flat.index_select(0, flat_idx).view(grid_shape)
            region_alpha_scale = self.alpha_scale_flat.index_select(0, flat_idx).view(grid_shape)
            region_alpha_nonzero = self.alpha_nonzero_flat.index_select(0, flat_idx).view(grid_shape)
            body_total = mask.sum(dim=(1, 2), dtype=torch.float32)
            inside = (mask & region_alpha_inside).sum(dim=(1, 2), dtype=torch.float32)
            overlap_ok = (body_total > 0) & (inside >= body_total * STICKER_OVERLAP_MIN)
            effective_mask = color_mask * region_alpha_scale
            valid_scores = valid_scores & (body_total > 0) & overlap_ok
        else:
            region_alpha_nonzero = None
            effective_mask = mask.to(torch.float32)
        weight = effective_mask.sum(dim=(1, 2))
        valid_scores = valid_scores & (weight >= 0.5)
        if not fixed_half_alpha:
            valid_scores = valid_scores & (alpha_int_values > 0)
        safe_weight = weight.clamp_min(1e-6).view(-1, 1)
        sum_d = (d * effective_mask.unsqueeze(-1)).sum(dim=(1, 2))
        rgb_float = torch.floor(torch.clamp(sum_d / (alpha_scale_rgb * safe_weight), 0.0, 255.0))
        rgb = rgb_float.to(torch.int32)
        if fixed_half_alpha:
            alpha_int = torch.full((rgb.shape[0], 1), 128, device=self.device, dtype=torch.int32)
        else:
            alpha_int = alpha_int_values.unsqueeze(-1)
        colors = torch.cat([rgb, alpha_int], dim=1)
        if self.alpha is None:
            mask_float = mask.unsqueeze(-1)
            applied = alpha_scale_rgb * rgb_float
            if fixed_half_alpha:
                region_old_sq = (region_old_sq_base * mask).sum(dim=(1, 2), dtype=torch.float32)
                sum_d_sq_total = (region_d_sq_base * mask).sum(dim=(1, 2), dtype=torch.float32)
            else:
                region_old_sq = (region_old_sq_base * mask).sum(dim=(1, 2), dtype=torch.float32)
                sum_d_sq = ((d * d) * mask_float).sum(dim=(1, 2))
                sum_d_sq_total = sum_d_sq.sum(dim=1)
            region_new_sq = sum_d_sq_total - 2.0 * (applied * sum_d).sum(dim=1) + weight * (applied * applied).sum(dim=1)
        else:
            allowed_mask_f = (region_alpha_nonzero * color_mask).unsqueeze(-1)
            applied = alpha_scale_rgb * rgb_float
            sum_d_allowed = (d * allowed_mask_f).sum(dim=(1, 2))
            sum_d_sq_allowed = ((d * d) * allowed_mask_f).sum(dim=(1, 2))
            allowed_count = allowed_mask_f[..., 0].sum(dim=(1, 2)).clamp_min(1e-6)
            region_new_sq = (
                sum_d_sq_allowed.sum(dim=1)
                - 2.0 * (applied * sum_d_allowed).sum(dim=1)
                + allowed_count * (applied * applied).sum(dim=1)
            )
            region_old_sq = (region_old_sq_base * allowed_mask_f[..., 0]).sum(dim=(1, 2))
        total_sq = self._canvas_full_sq_scalar - region_old_sq + region_new_sq
        denom = max(canvas_norm, 1.0)
        scores = torch.sqrt(torch.clamp(total_sq, min=0.0) / denom)
        scores = scores.masked_fill(~valid_scores, float("inf"))
        return scores, colors

    def _score_batch_resilient(
        self,
        type_name: str,
        packed: dict[str, object],
        start: int,
        end: int,
        canvas_norm: float,
    ) -> tuple[object, object]:
        torch = self._torch
        try:
            return self._score_batch(type_name, packed, start, end, canvas_norm)
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or end - start <= 1:
                raise
            self._record_batch_oom()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            mid = start + max(1, (end - start) // 2)
            left_scores, left_colors = self._score_batch_resilient(type_name, packed, start, mid, canvas_norm)
            right_scores, right_colors = self._score_batch_resilient(type_name, packed, mid, end, canvas_norm)
            return torch.cat((left_scores, right_scores), dim=0), torch.cat((left_colors, right_colors), dim=0)

    def _score_quality_batch(
        self,
        type_name: str,
        packed: dict[str, object],
        base_colors: object,
        start: int,
        end: int,
        weighted_full_sq: float,
        weighted_norm: float,
        gradient_full_error: float,
        gradient_norm: float,
    ) -> tuple[object, object]:
        torch = self._torch
        batch_size = end - start
        scores_out = torch.full((batch_size,), float("inf"), device=self.device, dtype=torch.float32)
        colors_out = torch.zeros((batch_size, 4), device=self.device, dtype=torch.int32)
        if batch_size == 0:
            return scores_out, colors_out
        x0 = packed["x0"][start:end]
        y0 = packed["y0"][start:end]
        width_t = packed["width"][start:end]
        height_t = packed["height"][start:end]
        xg0 = torch.clamp(x0 - 1, min=0)
        yg0 = torch.clamp(y0 - 1, min=0)
        xg1 = torch.clamp(x0 + width_t + 1, max=self.w)
        yg1 = torch.clamp(y0 + height_t + 1, max=self.h)
        gwidth = torch.clamp(xg1 - xg0, min=0)
        gheight = torch.clamp(yg1 - yg0, min=0)
        width_cpu = packed["width_cpu"][start:end]
        height_cpu = packed["height_cpu"][start:end]
        max_w = min(self.w, int(width_cpu.max()) + 2) if width_cpu.size else 0
        max_h = min(self.h, int(height_cpu.max()) + 2) if height_cpu.size else 0
        if max_w <= 0 or max_h <= 0:
            return scores_out, colors_out
        x_base_i = self._long_arange(max_w).view(1, 1, max_w)
        y_base_i = self._long_arange(max_h).view(1, max_h, 1)
        valid = (x_base_i < gwidth.view(-1, 1, 1)) & (y_base_i < gheight.view(-1, 1, 1))
        x_idx = torch.minimum(xg0.view(-1, 1, 1) + x_base_i, (xg1 - 1).view(-1, 1, 1)).clamp(0, self.w - 1)
        y_idx = torch.minimum(yg0.view(-1, 1, 1) + y_base_i, (yg1 - 1).view(-1, 1, 1)).clamp(0, self.h - 1)
        flat_idx = (y_idx * self.w + x_idx).reshape(-1)
        grid_shape = (batch_size, max_h, max_w)
        region_cur = self.canvas_flat.index_select(0, flat_idx).view(batch_size, max_h, max_w, 3)
        region_tgt = self.target_flat.index_select(0, flat_idx).view(batch_size, max_h, max_w, 3)
        region_old_sqsum = self._canvas_minus_target_sqsum_flat.index_select(0, flat_idx).view(grid_shape)
        patch_weight = self._quality_edge_weight_flat.index_select(0, flat_idx).view(grid_shape) * valid.to(torch.float32)
        compute_gradient = float(self._quality_gradient_weight) > 0.0 and float(gradient_norm) >= 1.0
        if compute_gradient:
            target_gx = self._quality_target_gx_flat.index_select(0, flat_idx).view(grid_shape)
            target_gy = self._quality_target_gy_flat.index_select(0, flat_idx).view(grid_shape)
        x_abs = x_idx.to(torch.float32)
        y_abs = y_idx.to(torch.float32)
        if type_name == "rectangle":
            mask = (
                valid
                & (x_idx >= x0.view(-1, 1, 1))
                & (x_idx < (x0 + width_t).view(-1, 1, 1))
                & (y_idx >= y0.view(-1, 1, 1))
                & (y_idx < (y0 + height_t).view(-1, 1, 1))
            )
        else:
            mask = self._build_mask(type_name, packed, start, end, x_abs, y_abs, valid)
        color_indices = packed["indices"][start:end]
        colors = base_colors.index_select(0, color_indices).to(torch.int32)
        src_rgb = colors[:, :3].to(torch.float32)
        alpha_scale = (colors[:, 3].to(torch.float32) / 255.0).view(-1, 1, 1)
        inv_alpha_scale = 1.0 - alpha_scale
        new_r = alpha_scale * src_rgb[:, 0].view(-1, 1, 1) + inv_alpha_scale * region_cur[..., 0]
        new_g = alpha_scale * src_rgb[:, 1].view(-1, 1, 1) + inv_alpha_scale * region_cur[..., 1]
        new_b = alpha_scale * src_rgb[:, 2].view(-1, 1, 1) + inv_alpha_scale * region_cur[..., 2]
        new_sqsum = (
            (new_r - region_tgt[..., 0]) ** 2
            + (new_g - region_tgt[..., 1]) ** 2
            + (new_b - region_tgt[..., 2]) ** 2
        )
        new_sqsum = torch.where(mask, new_sqsum, region_old_sqsum)
        del src_rgb, alpha_scale, inv_alpha_scale
        old_weighted = (region_old_sqsum * patch_weight).sum(dim=(1, 2))
        new_weighted = (new_sqsum * patch_weight).sum(dim=(1, 2))
        weighted_total = torch.clamp(self._quality_weighted_full_sq_scalar - old_weighted + new_weighted, min=0.0)
        if float(weighted_norm) < 1.0:
            weighted_rms = torch.zeros_like(weighted_total)
        else:
            weighted_rms = torch.sqrt(weighted_total / float(weighted_norm))
        if compute_gradient:
            old_luma = (
                region_cur[..., 0] * 0.299
                + region_cur[..., 1] * 0.587
                + region_cur[..., 2] * 0.114
            )
            new_luma_inside = new_r * 0.299 + new_g * 0.587 + new_b * 0.114
            new_luma = torch.where(mask, new_luma_inside, old_luma)
            del new_r, new_g, new_b, new_sqsum
            both_gx, both_gy = self._sobel_xy_batch(torch.cat((old_luma, new_luma), dim=0))
            old_gx, new_gx = both_gx.split(batch_size, dim=0)
            old_gy, new_gy = both_gy.split(batch_size, dim=0)
            old_grad = (((old_gx - target_gx) ** 2 + (old_gy - target_gy) ** 2) * patch_weight).sum(dim=(1, 2))
            new_grad = (((new_gx - target_gx) ** 2 + (new_gy - target_gy) ** 2) * patch_weight).sum(dim=(1, 2))
            gradient_total = torch.clamp(self._quality_gradient_full_error_scalar - old_grad + new_grad, min=0.0)
            gradient_score = torch.sqrt(gradient_total / float(gradient_norm)) / 4.0
        else:
            del new_r, new_g, new_b, new_sqsum
            gradient_score = torch.zeros_like(weighted_total)
        valid_scores = (width_t > 0) & (height_t > 0) & (colors[:, 3] > 0) & (mask.sum(dim=(1, 2)) >= 1)
        scores = weighted_rms + float(self._quality_gradient_weight) * gradient_score
        scores = scores.masked_fill(~valid_scores, float("inf"))
        return scores, colors

    def _sobel_xy_batch(self, luma: object) -> tuple[object, object]:
        torch = self._torch
        padded = torch.nn.functional.pad(luma.unsqueeze(1), (1, 1, 1, 1), mode="replicate").squeeze(1)
        gx = (
            -padded[:, :-2, :-2]
            - 2.0 * padded[:, 1:-1, :-2]
            - padded[:, 2:, :-2]
            + padded[:, :-2, 2:]
            + 2.0 * padded[:, 1:-1, 2:]
            + padded[:, 2:, 2:]
        )
        gy = (
            -padded[:, :-2, :-2]
            - 2.0 * padded[:, :-2, 1:-1]
            - padded[:, :-2, 2:]
            + padded[:, 2:, :-2]
            + 2.0 * padded[:, 2:, 1:-1]
            + padded[:, 2:, 2:]
        )
        return gx, gy

    def _build_mask(self, type_name: str, packed, start: int, end: int, x_abs, y_abs, valid):
        torch = self._torch
        batch_slice = slice(start, end)
        if type_name == "circle":
            shape_x = packed["shape_x"][batch_slice].view(-1, 1, 1)
            shape_y = packed["shape_y"][batch_slice].view(-1, 1, 1)
            r_sq = packed["radius"][batch_slice].view(-1, 1, 1)
            dx = x_abs - shape_x
            dy = y_abs - shape_y
            mask = (dx * dx + dy * dy) <= r_sq
            return mask & valid
        if type_name == "ellipse":
            shape_x = packed["shape_x"][batch_slice].view(-1, 1, 1)
            shape_y = packed["shape_y"][batch_slice].view(-1, 1, 1)
            inv_rx_sq = packed["inv_radius_x_sq"][batch_slice].view(-1, 1, 1)
            inv_ry_sq = packed["inv_radius_y_sq"][batch_slice].view(-1, 1, 1)
            dx = x_abs - shape_x
            dy = y_abs - shape_y
            mask = ((dx * dx) * inv_rx_sq + (dy * dy) * inv_ry_sq) <= 1.0
            return mask & valid
        if type_name == "rotated_ellipse":
            shape_x = packed["shape_x"][batch_slice].view(-1, 1, 1)
            shape_y = packed["shape_y"][batch_slice].view(-1, 1, 1)
            inv_rx_sq = packed["inv_radius_x_sq"][batch_slice].view(-1, 1, 1)
            inv_ry_sq = packed["inv_radius_y_sq"][batch_slice].view(-1, 1, 1)
            x_rel = x_abs - shape_x
            y_rel = y_abs - shape_y
            cos_a = packed["cos_angle"][batch_slice].view(-1, 1, 1)
            sin_a = packed["sin_angle"][batch_slice].view(-1, 1, 1)
            xr = cos_a * x_rel + sin_a * y_rel
            yr = -sin_a * x_rel + cos_a * y_rel
            mask = (((xr * xr) * inv_rx_sq) + ((yr * yr) * inv_ry_sq)) <= 1.0
            return mask & valid
        if type_name == "rectangle":
            return valid
        if type_name == "rotated_rectangle":
            shape_x = packed["shape_x"][batch_slice].view(-1, 1, 1)
            shape_y = packed["shape_y"][batch_slice].view(-1, 1, 1)
            hw = packed["half_width"][batch_slice].view(-1, 1, 1)
            hh = packed["half_height"][batch_slice].view(-1, 1, 1)
            x_rel = x_abs - shape_x
            y_rel = y_abs - shape_y
            cos_a = packed["cos_angle"][batch_slice].view(-1, 1, 1)
            sin_a = packed["sin_angle"][batch_slice].view(-1, 1, 1)
            xr = cos_a * x_rel + sin_a * y_rel
            yr = -sin_a * x_rel + cos_a * y_rel
            mask = (xr.abs() <= hw) & (yr.abs() <= hh)
            return mask & valid
        if type_name == "triangle":
            x1 = packed["x1"][batch_slice].view(-1, 1, 1)
            y1 = packed["y1"][batch_slice].view(-1, 1, 1)
            x2 = packed["x2"][batch_slice].view(-1, 1, 1)
            y2 = packed["y2"][batch_slice].view(-1, 1, 1)
            x3 = packed["x3"][batch_slice].view(-1, 1, 1)
            y3 = packed["y3"][batch_slice].view(-1, 1, 1)
            d1 = packed["x2_minus_x1"][batch_slice].view(-1, 1, 1) * (y_abs - y1) - packed["y2_minus_y1"][batch_slice].view(-1, 1, 1) * (x_abs - x1)
            d2 = packed["x3_minus_x2"][batch_slice].view(-1, 1, 1) * (y_abs - y2) - packed["y3_minus_y2"][batch_slice].view(-1, 1, 1) * (x_abs - x2)
            d3 = packed["x1_minus_x3"][batch_slice].view(-1, 1, 1) * (y_abs - y3) - packed["y1_minus_y3"][batch_slice].view(-1, 1, 1) * (x_abs - x3)
            has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
            has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
            return (~(has_neg & has_pos)) & valid
        raise ValueError(f"Unsupported shape type for GPU backend: {type_name}")

    def _long_arange(self, n: int):
        if n <= int(self._long_base.shape[0]):
            return self._long_base[:n]
        tensor = self._long_cache.get(n)
        if tensor is None:
            tensor = self._torch.arange(n, device=self.device, dtype=self._torch.long)
            self._long_cache[n] = tensor
        return tensor

    def _float_arange(self, n: int, step: int):
        if n <= int(self._float_base.shape[0]):
            key = (int(self._float_base.shape[0]), step)
            tensor = self._float_cache.get(key)
            if tensor is None:
                tensor = self._float_base if step == 1 else (self._float_base * float(step))
                self._float_cache[key] = tensor
            return tensor[:n]
        key = (n, step)
        tensor = self._float_cache.get(key)
        if tensor is None:
            tensor = self._torch.arange(n, device=self.device, dtype=self._torch.float32)
            if step != 1:
                tensor = tensor * float(step)
            self._float_cache[key] = tensor
        return tensor
