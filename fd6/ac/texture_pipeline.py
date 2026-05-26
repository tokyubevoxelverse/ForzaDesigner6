"""Image-to-texture preprocessing for AC livery export.

Takes the user's source image and produces a numpy RGBA array at the target
resolution and aspect ratio that ACC's livery loader expects. No DDS
encoding — ACC reads PNG files directly from the user's Customs folder, so
we keep everything in RGBA-uint8 land until the livery_writer serializes
it.

Aspect ratio handling: ACC's decals texture is normally square (e.g. 4096×4096).
If the source isn't square, we pad with transparent pixels so the source's
proportions are preserved (no stretching).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def _load_source_rgba(source: str | Path) -> np.ndarray:
    """Load source image as HxWx4 uint8 RGBA. Anything without alpha gets full opacity."""
    img = Image.open(str(source))
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    return np.asarray(img, dtype=np.uint8)


def _aspect_from_choice(choice: str, src_w: int, src_h: int) -> tuple[int, int]:
    """Resolve an aspect-ratio choice (Auto / 1:1 / 1:2 / 1:4) into (w_ratio, h_ratio)."""
    choice = (choice or "auto").lower().strip()
    if choice == "1:1":
        return (1, 1)
    if choice == "1:2":
        return (1, 2)
    if choice == "1:4":
        return (1, 4)
    # Auto — pick the canonical ACC aspect closest to the source.
    ratio = (src_w / src_h) if src_h > 0 else 1.0
    candidates = [(1, 1), (2, 1), (4, 1), (1, 2), (1, 4)]
    best = min(candidates, key=lambda ab: abs((ab[0] / ab[1]) - ratio))
    return best


def build_decal_texture(
    source: str | Path,
    target_long_edge: int = 4096,
    aspect_choice: str = "auto",
) -> tuple[np.ndarray, str]:
    """Produce a single decal PNG texture from a source image.

    Returns (rgba_array, applied_aspect_label).
    rgba_array is the texture ready to save via Pillow.
    applied_aspect_label is the AxB form (e.g. '1:1', '1:2') for logging.

    Pipeline:
      1. Load source as RGBA.
      2. Decide the output aspect ratio (per user choice or auto-detect).
      3. Compute (W, H) where the longer edge equals target_long_edge and
         W:H matches the chosen aspect ratio. Both are forced to powers of 2.
      4. Build a transparent canvas of (H, W).
      5. Scale source to fit inside the canvas preserving proportion, center it.
      6. Return the composed array.
    """
    src = _load_source_rgba(source)
    src_h, src_w = src.shape[:2]
    aw, ah = _aspect_from_choice(aspect_choice, src_w, src_h)

    # Determine final canvas size with long edge == target_long_edge.
    # target_long_edge is assumed already power-of-2 (UI restricts to 1024/2048/4096).
    if aw >= ah:
        out_w = target_long_edge
        out_h = target_long_edge * ah // aw
    else:
        out_h = target_long_edge
        out_w = target_long_edge * aw // ah
    # Snap both axes to nearest power-of-2 (ACC requirement). Round to nearest
    # rather than down so we don't shrink dramatically for ratios like 3:1.
    out_w = _nearest_pow2(out_w)
    out_h = _nearest_pow2(out_h)

    # Resize source into the canvas preserving aspect, center it.
    scale = min(out_w / src_w, out_h / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    src_img = Image.fromarray(src)
    resized = src_img.resize((new_w, new_h), Image.LANCZOS)
    resized_arr = np.asarray(resized, dtype=np.uint8)

    canvas = np.zeros((out_h, out_w, 4), dtype=np.uint8)
    pad_x = (out_w - new_w) // 2
    pad_y = (out_h - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w, :] = resized_arr

    return canvas, f"{aw}:{ah}"


def _nearest_pow2(value: int) -> int:
    """Round to the nearest power-of-2 (no smaller than 1)."""
    if value < 1:
        return 1
    lower = 1 << (value.bit_length() - 1)
    upper = lower << 1
    return lower if (value - lower) < (upper - value) else upper


def save_texture_png(rgba: np.ndarray, out_path: str | Path) -> Path:
    """Write the RGBA numpy array to disk as a PNG.

    ACC ignores PNG metadata — we only set the pixel data. The file size
    will be on the order of a few MB at 4096×4096 since most areas are
    likely transparent and PNG compresses runs of zeros well.
    """
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError("texture must be HxWx4 RGBA")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(str(out_path), format="PNG")
    return out_path
