from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class LineGuideResult:
    guide: np.ndarray | None
    source: str
    message: str


@dataclass(frozen=True)
class LineGuideModelConfig:
    input_color_order: str = "rgb"
    input_value_range: str = "0_1"
    input_mean: tuple[float, ...] = ()
    input_std: tuple[float, ...] = ()
    output_index: int = 0
    output_channel: int | None = None
    output_activation: str = "auto"
    output_invert: bool = False
    max_resolution: int | None = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _model_roots() -> list[Path]:
    roots: list[Path] = []
    env_root = os.environ.get("FD6_MODELS_DIR", "").strip()
    if env_root:
        roots.append(Path(env_root).expanduser())
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        roots.append(exe_dir / "models")
        meipass = getattr(sys, "_MEIPASS", "")
        if meipass:
            roots.append(Path(meipass).resolve() / "models")
        roots.append(exe_dir / "_internal" / "models")
    roots.extend([Path.cwd() / "models", _repo_root() / "models"])
    seen: set[str] = set()
    unique: list[Path] = []
    for root in roots:
        key = str(root)
        if key not in seen:
            unique.append(root)
            seen.add(key)
    return unique


def _candidate_paths(raw_path: str, source_path: Path | None = None) -> list[Path]:
    text = raw_path.strip()
    if not text:
        return []
    path = Path(text).expanduser()
    if path.is_absolute():
        return [path]
    roots: list[Path] = []
    if source_path is not None:
        roots.append(source_path.parent)
    roots.extend([Path.cwd(), _repo_root()])
    roots.extend(_model_roots())
    candidates: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        for candidate in (root / path, root / "models" / path):
            key = str(candidate)
            if key not in seen:
                candidates.append(candidate)
                seen.add(key)
    return candidates


def _find_existing_path(raw_path: str, source_path: Path | None = None) -> Path | None:
    for candidate in _candidate_paths(raw_path, source_path):
        if candidate.exists():
            return candidate
    return None


def _default_model_path() -> Path | None:
    for root in _model_roots():
        path = root / "line_guide.onnx"
        if path.exists():
            return path
    return None


def _model_config_path(model_path: Path) -> Path | None:
    candidates = [model_path.with_suffix(".json"), model_path.parent / "line_guide.json"]
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate
    return None


def _str_value(mapping: dict[str, Any], *keys: str, default: str) -> str:
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return str(value).strip().lower()
    return default


def _int_value(mapping: dict[str, Any], *keys: str, default: int) -> int:
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                return default
    return default


def _bool_value(mapping: dict[str, Any], *keys: str, default: bool) -> bool:
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            if isinstance(value, bool):
                return value
            text = str(value).strip().lower()
            if text in {"1", "true", "yes", "on"}:
                return True
            if text in {"0", "false", "no", "off"}:
                return False
    return default


def _float_tuple(value: Any) -> tuple[float, ...]:
    if value is None:
        return ()
    if isinstance(value, (int, float)):
        return (float(value),)
    if isinstance(value, (list, tuple)):
        result: list[float] = []
        for item in value:
            try:
                result.append(float(item))
            except (TypeError, ValueError):
                return ()
        return tuple(result)
    return ()


def load_line_guide_model_config(model_path: Path) -> LineGuideModelConfig:
    path = _model_config_path(model_path)
    if path is None:
        return LineGuideModelConfig()
    with path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, dict):
        return LineGuideModelConfig()
    input_raw = raw.get("input", {})
    output_raw = raw.get("output", {})
    input_cfg = input_raw if isinstance(input_raw, dict) else {}
    output_cfg = output_raw if isinstance(output_raw, dict) else {}
    output_channel_raw = output_cfg.get("channel", output_cfg.get("outputChannel"))
    output_channel = None
    if output_channel_raw is not None and str(output_channel_raw).strip().lower() != "auto":
        try:
            output_channel = int(output_channel_raw)
        except (TypeError, ValueError):
            output_channel = None
    max_resolution = raw.get("maxResolution", raw.get("max_resolution"))
    max_res_value = None
    if max_resolution is not None:
        try:
            max_res_value = max(0, int(max_resolution))
        except (TypeError, ValueError):
            max_res_value = None
    return LineGuideModelConfig(
        input_color_order=_str_value(input_cfg, "colorOrder", "color_order", default="rgb"),
        input_value_range=_str_value(input_cfg, "valueRange", "value_range", default="0_1"),
        input_mean=_float_tuple(input_cfg.get("mean")),
        input_std=_float_tuple(input_cfg.get("std")),
        output_index=max(0, _int_value(output_cfg, "index", "outputIndex", "output_index", default=0)),
        output_channel=output_channel,
        output_activation=_str_value(output_cfg, "activation", default="auto"),
        output_invert=_bool_value(output_cfg, "invert", "inverted", default=False),
        max_resolution=max_res_value,
    )


def _has_guide_signal(guide: np.ndarray) -> bool:
    return bool(guide.size) and float(np.nanmax(guide)) > 0.03


def _limit_image_size(img: Image.Image, max_resolution: int) -> Image.Image:
    cap = int(max_resolution)
    if cap <= 0:
        return img
    width, height = img.size
    long_side = max(width, height)
    if long_side <= cap:
        return img
    scale = cap / float(long_side)
    size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return img.resize(size, Image.LANCZOS)


def _resize_map(values: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    width, height = target_size
    src = np.clip(values.astype(np.float32, copy=False), 0.0, 1.0)
    if src.shape == (height, width):
        return src.astype(np.float32, copy=False)
    image = Image.fromarray((src * 255.0 + 0.5).astype(np.uint8), "L")
    resized = image.resize((width, height), Image.LANCZOS)
    return (np.asarray(resized, dtype=np.float32) / 255.0).astype(np.float32)


def _fit_to_target_geometry(
    img: Image.Image,
    target_size: tuple[int, int],
    *,
    source_size: tuple[int, int] | None = None,
    pad_to_square: bool = False,
) -> Image.Image:
    work = img
    if source_size is not None and pad_to_square and source_size[0] != source_size[1] and img.size == source_size:
        side = max(source_size)
        if "A" in img.getbands():
            background = Image.new("RGBA", (side, side), (255, 255, 255, 0))
            paste_img = img.convert("RGBA")
        else:
            background = Image.new("RGB", (side, side), (255, 255, 255))
            paste_img = img.convert("RGB")
        offset = ((side - source_size[0]) // 2, (side - source_size[1]) // 2)
        background.paste(paste_img, offset)
        work = background
    if work.size != target_size:
        work = work.resize(target_size, Image.LANCZOS)
    return work


def _gray_and_alpha(img: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    rgba = np.asarray(img.convert("RGBA"), dtype=np.float32) / 255.0
    gray = rgba[:, :, 0] * 0.299 + rgba[:, :, 1] * 0.587 + rgba[:, :, 2] * 0.114
    return gray.astype(np.float32), rgba[:, :, 3].astype(np.float32)


def normalise_line_strength(values: np.ndarray) -> np.ndarray:
    gray = np.asarray(values, dtype=np.float32)
    if gray.ndim == 3:
        gray = gray[:, :, 0] * 0.299 + gray[:, :, 1] * 0.587 + gray[:, :, 2] * 0.114
    if gray.size == 0:
        return np.zeros(gray.shape[:2], dtype=np.float32)
    if float(np.nanmax(gray)) > 1.5:
        gray = gray / 255.0
    gray = np.nan_to_num(np.clip(gray, 0.0, 1.0), nan=0.0, posinf=1.0, neginf=0.0)
    dark = 1.0 - gray
    light = gray

    def contrast_score(arr: np.ndarray) -> float:
        p50 = float(np.percentile(arr, 50.0))
        p99 = float(np.percentile(arr, 99.0))
        p997 = float(np.percentile(arr, 99.7))
        return max(p99, p997) - p50

    line = light if contrast_score(light) > contrast_score(dark) else dark
    lo = float(np.percentile(line, 50.0))
    hi = max(float(np.percentile(line, 99.0)), float(np.percentile(line, 99.7)), float(line.max()))
    if hi - lo < 0.02:
        return np.zeros(line.shape, dtype=np.float32)
    out = np.clip((line - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    out[out < 0.03] = 0.0
    return out


def load_line_guide_image(
    path: Path,
    target_size: tuple[int, int],
    *,
    max_resolution: int = 0,
    source_size: tuple[int, int] | None = None,
    pad_to_square: bool = False,
) -> np.ndarray:
    with Image.open(path) as img:
        img = _fit_to_target_geometry(
            img,
            target_size,
            source_size=source_size,
            pad_to_square=pad_to_square,
        )
        img = _limit_image_size(img, max_resolution)
        gray, alpha = _gray_and_alpha(img)
    guide = normalise_line_strength(gray) * alpha
    return _resize_map(guide, target_size)


def _shape_value(value: Any) -> int | None:
    return value if isinstance(value, int) and value > 0 else None


def _model_layout(input_shape: list[Any] | tuple[Any, ...]) -> tuple[str, int, int | None, int | None]:
    rank = len(input_shape)
    if rank == 4:
        c_first = _shape_value(input_shape[1])
        c_last = _shape_value(input_shape[3])
        if c_last in (1, 3) and c_first not in (1, 3):
            return "nhwc", c_last, _shape_value(input_shape[1]), _shape_value(input_shape[2])
        channels = c_first if c_first in (1, 3) else 3
        return "nchw", channels, _shape_value(input_shape[2]), _shape_value(input_shape[3])
    if rank == 3:
        c_first = _shape_value(input_shape[0])
        c_last = _shape_value(input_shape[2])
        if c_last in (1, 3) and c_first not in (1, 3):
            return "hwc", c_last, _shape_value(input_shape[0]), _shape_value(input_shape[1])
        channels = c_first if c_first in (1, 3) else 3
        return "chw", channels, _shape_value(input_shape[1]), _shape_value(input_shape[2])
    return "nchw", 3, None, None


def _dynamic_size(source_size: tuple[int, int], max_resolution: int) -> tuple[int, int]:
    width, height = source_size
    cap = int(max_resolution)
    if cap <= 0 or max(width, height) <= cap:
        return width, height
    scale = cap / float(max(width, height))
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def _onnx_input_array(
    image: Image.Image,
    layout: str,
    channels: int,
    size: tuple[int, int],
    tensor_type: str,
    config: LineGuideModelConfig | None = None,
) -> np.ndarray:
    img = image.convert("RGB").resize(size, Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    cfg = config or LineGuideModelConfig()
    if cfg.input_color_order == "bgr":
        arr = arr[:, :, ::-1]
    if channels == 1:
        gray = arr[:, :, 0] * 0.299 + arr[:, :, 1] * 0.587 + arr[:, :, 2] * 0.114
        arr = gray[:, :, None]
    elif cfg.input_color_order == "luma":
        gray = arr[:, :, 0] * 0.299 + arr[:, :, 1] * 0.587 + arr[:, :, 2] * 0.114
        arr = np.repeat(gray[:, :, None], channels, axis=2)
    value_range = cfg.input_value_range.replace("-", "_").replace(":", "_")
    if "uint8" in tensor_type:
        arr = np.clip(arr * 255.0 + 0.5, 0, 255)
    elif value_range in {"0_255", "255"}:
        arr = arr * 255.0
    elif value_range in {"minus1_1", "_1_1", "neg1_1"}:
        arr = arr * 2.0 - 1.0
    if "uint8" not in tensor_type:
        if cfg.input_mean:
            mean = _channel_params(cfg.input_mean, channels)
            arr = arr - mean.reshape((1, 1, channels))
        if cfg.input_std:
            std = np.maximum(_channel_params(cfg.input_std, channels), 1e-6)
            arr = arr / std.reshape((1, 1, channels))
    if layout in {"nchw", "chw"}:
        arr = np.transpose(arr, (2, 0, 1))
    if layout in {"nchw", "nhwc"}:
        arr = arr[None, ...]
    if "uint8" in tensor_type:
        return np.clip(arr, 0, 255).astype(np.uint8)
    if "float16" in tensor_type:
        return arr.astype(np.float16)
    return arr.astype(np.float32)


def _channel_params(values: tuple[float, ...], channels: int) -> np.ndarray:
    if not values:
        return np.zeros(channels, dtype=np.float32)
    if len(values) == channels:
        return np.asarray(values, dtype=np.float32)
    if len(values) == 1:
        return np.full(channels, float(values[0]), dtype=np.float32)
    if channels == 1:
        return np.asarray([float(values[0])], dtype=np.float32)
    if len(values) > channels:
        return np.asarray(values[:channels], dtype=np.float32)
    padded = list(values)
    while len(padded) < channels:
        padded.append(padded[-1])
    return np.asarray(padded, dtype=np.float32)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values.astype(np.float32, copy=False), -40.0, 40.0)
    return (1.0 / (1.0 + np.exp(-clipped))).astype(np.float32)


def _softmax_line_channel(values: np.ndarray, axis: int) -> np.ndarray:
    shifted = values.astype(np.float32, copy=False) - np.max(values, axis=axis, keepdims=True)
    exp_values = np.exp(np.clip(shifted, -40.0, 40.0))
    denom = np.maximum(exp_values.sum(axis=axis, keepdims=True), 1e-6)
    probs = exp_values / denom
    return np.take(probs, 1, axis=axis).astype(np.float32)


def _confidence_to_map(values: np.ndarray, activation: str = "auto", invert: bool = False) -> np.ndarray:
    arr = np.nan_to_num(values.astype(np.float32, copy=False), nan=0.0, posinf=1.0, neginf=0.0)
    if arr.size == 0:
        return np.zeros(arr.shape, dtype=np.float32)
    lo = float(arr.min())
    hi = float(arr.max())
    act = activation.strip().lower()
    if act == "sigmoid" or (act == "auto" and (lo < 0.0 or hi > 1.0)):
        arr = _sigmoid(arr)
    elif act in {"none", "linear"}:
        arr = arr.astype(np.float32, copy=False)
    if invert:
        arr = 1.0 - arr
    arr = np.nan_to_num(np.clip(arr, 0.0, 1.0), nan=0.0, posinf=1.0, neginf=0.0)
    lo = float(np.percentile(arr, 50.0))
    hi = max(float(np.percentile(arr, 99.0)), float(np.percentile(arr, 99.7)), float(arr.max()))
    if hi - lo < 0.02:
        return np.zeros(arr.shape, dtype=np.float32)
    out = np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    out[out < 0.03] = 0.0
    return out


def _take_output_channel(arr: np.ndarray, channel: int, activation: str) -> np.ndarray:
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        return arr
    axis = 0
    if arr.shape[-1] > 1 and (arr.shape[0] > 16 or arr.shape[-1] <= arr.shape[0]):
        axis = -1
    channel_count = arr.shape[axis]
    if channel < 0 or channel >= channel_count:
        raise ValueError(f"Output channel {channel} outside model output shape {arr.shape!r}")
    if activation.strip().lower() == "softmax":
        shifted = arr.astype(np.float32, copy=False) - np.max(arr, axis=axis, keepdims=True)
        exp_values = np.exp(np.clip(shifted, -40.0, 40.0))
        denom = np.maximum(exp_values.sum(axis=axis, keepdims=True), 1e-6)
        arr = exp_values / denom
    return np.take(arr, channel, axis=axis).astype(np.float32)


def _output_to_map(output: Any, config: LineGuideModelConfig | None = None) -> np.ndarray:
    cfg = config or LineGuideModelConfig()
    arr = np.asarray(output, dtype=np.float32)
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 4:
        arr = arr[0]
    if cfg.output_channel is not None:
        arr = _take_output_channel(arr, cfg.output_channel, cfg.output_activation)
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[-1] == 1:
            arr = arr[:, :, 0]
        elif arr.shape[0] == 2:
            arr = _softmax_line_channel(arr, axis=0)
        elif arr.shape[-1] == 2:
            arr = _softmax_line_channel(arr, axis=-1)
        elif arr.shape[0] <= 4:
            arr = arr.max(axis=0)
        elif arr.shape[-1] <= 4:
            arr = arr.max(axis=-1)
        else:
            arr = arr[0]
    if arr.ndim != 2:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Unsupported ONNX output shape: {np.asarray(output).shape!r}")
    return _confidence_to_map(arr, cfg.output_activation, cfg.output_invert)


def _create_session(ort: Any, model_path: Path, providers: list[str]) -> Any:
    return ort.InferenceSession(str(model_path), providers=providers)


def _session_providers(session: Any, fallback: list[str]) -> list[str]:
    get_providers = getattr(session, "get_providers", None)
    if get_providers is None:
        return fallback
    try:
        providers = list(get_providers())
    except Exception:
        return fallback
    return providers or fallback


def _run_session(
    session: Any,
    image: Image.Image,
    target_size: tuple[int, int],
    max_resolution: int,
    config: LineGuideModelConfig,
) -> np.ndarray:
    inputs = session.get_inputs()
    if not inputs:
        raise ValueError("ONNX model has no inputs")
    input_info = inputs[0]
    layout, channels, height, width = _model_layout(input_info.shape)
    dyn_width, dyn_height = _dynamic_size(image.size, max_resolution)
    size = (width or dyn_width, height or dyn_height)
    feed = _onnx_input_array(image, layout, channels, size, input_info.type, config)
    outputs = session.run(None, {input_info.name: feed})
    if not outputs:
        raise ValueError("ONNX model returned no outputs")
    output_index = min(config.output_index, len(outputs) - 1)
    guide = _output_to_map(outputs[output_index], config)
    return _resize_map(guide, target_size)


def load_line_guide_onnx(
    image: Image.Image,
    model_path: Path,
    target_size: tuple[int, int],
    *,
    max_resolution: int = 0,
    prefer_gpu: bool = True,
) -> tuple[np.ndarray, list[str]]:
    import onnxruntime as ort

    config = load_line_guide_model_config(model_path)
    if config.max_resolution is not None:
        max_resolution = config.max_resolution
    available = list(ort.get_available_providers())
    if prefer_gpu:
        preferred = [
            "CUDAExecutionProvider",
            "DmlExecutionProvider",
            "DirectMLExecutionProvider",
            "CPUExecutionProvider",
        ]
    else:
        preferred = ["CPUExecutionProvider"]
    providers = [name for name in preferred if name in available] or available
    try:
        session = _create_session(ort, model_path, providers)
    except Exception:
        if not prefer_gpu or "CPUExecutionProvider" not in available or providers == ["CPUExecutionProvider"]:
            raise
        providers = ["CPUExecutionProvider"]
        session = _create_session(ort, model_path, providers)
    try:
        actual_providers = _session_providers(session, providers)
        return _run_session(session, image, target_size, max_resolution, config), actual_providers
    except Exception:
        if not prefer_gpu or "CPUExecutionProvider" not in available or providers == ["CPUExecutionProvider"]:
            raise
        providers = ["CPUExecutionProvider"]
        session = _create_session(ort, model_path, providers)
        actual_providers = _session_providers(session, providers)
        return _run_session(session, image, target_size, max_resolution, config), actual_providers


def resolve_line_guide(
    source_image: Image.Image,
    target_size: tuple[int, int],
    profile: Any,
    source_path: Path | None = None,
    *,
    guide_source_size: tuple[int, int] | None = None,
    pad_guide_to_square: bool = False,
) -> LineGuideResult:
    if not bool(getattr(profile, "line_guide_enabled", False)):
        return LineGuideResult(None, "disabled", "Line guide: disabled.")
    max_resolution = max(0, int(getattr(profile, "line_guide_max_resolution", 0)))
    prefer_gpu = str(getattr(profile, "compute_backend", "auto") or "auto").strip().lower() != "cpu"
    messages: list[str] = []
    raw_model = str(getattr(profile, "line_guide_model_path", "") or "").strip()
    model_path = _find_existing_path(raw_model, source_path) if raw_model else _default_model_path()
    if model_path is not None:
        try:
            guide, providers = load_line_guide_onnx(
                source_image,
                model_path,
                target_size,
                max_resolution=max_resolution,
                prefer_gpu=prefer_gpu,
            )
            if not _has_guide_signal(guide):
                raise ValueError("ONNX output has no usable line signal")
            return LineGuideResult(
                guide,
                "onnx",
                f"Line guide: ONNX {model_path.name} ({', '.join(providers)}).",
            )
        except Exception as exc:
            messages.append(f"ONNX failed: {type(exc).__name__}: {exc}")
    elif raw_model:
        messages.append(f"ONNX model not found: {raw_model}")

    raw_image = str(getattr(profile, "line_guide_image_path", "") or "").strip()
    image_path = _find_existing_path(raw_image, source_path) if raw_image else None
    if image_path is not None:
        try:
            guide = load_line_guide_image(
                image_path,
                target_size,
                max_resolution=max_resolution,
                source_size=guide_source_size,
                pad_to_square=pad_guide_to_square,
            )
            if not _has_guide_signal(guide):
                raise ValueError("guide image has no usable line signal")
            prefix = "Line guide: image"
            if messages:
                prefix += f" after fallback ({'; '.join(messages)})"
            return LineGuideResult(guide, "image", f"{prefix}: {image_path.name}.")
        except Exception as exc:
            messages.append(f"image failed: {type(exc).__name__}: {exc}")
    elif raw_image:
        messages.append(f"image not found: {raw_image}")

    suffix = f" ({'; '.join(messages)})" if messages else " (no guide source found)"
    return LineGuideResult(None, "failed", f"Line guide: unavailable{suffix}; generation continues without it.")
