"""Render a list of pre-built shapes onto a canvas (no optimization, no scoring).

Used by the GUI when a user uploads an existing JSON and we want to show what it
would look like in the preview pane without rerunning generation.
"""

from __future__ import annotations

import numpy as np

from forza_abyss_painter.shapegen.shapes import Shape


def _checkerboard(width: int, height: int, tile: int = 12) -> np.ndarray:
    """Light-grey/dark-grey checkerboard for transparency previews."""
    yy, xx = np.indices((height, width))
    mask = ((xx // tile) + (yy // tile)) & 1
    canvas = np.empty((height, width, 3), dtype=np.uint8)
    canvas[mask == 0] = (208, 208, 208)
    canvas[mask == 1] = (160, 160, 160)
    return canvas


def _is_stickerlike(shapes: list[Shape]) -> bool:
    """Heuristic: if a meaningful fraction of shapes have alpha < 255, treat as sticker."""
    if not shapes:
        return False
    n_transparent = 0
    for s in shapes:
        c = s.color
        if len(c) >= 4 and c[3] < 255:
            n_transparent += 1
    # If >5% of shapes are non-opaque, this is sticker-style content
    return n_transparent > max(5, len(shapes) // 20)


def render_shapes(
    shapes: list[Shape], width: int, height: int,
    background=(255, 255, 255), transparent_bg: bool = False,
) -> np.ndarray:
    """Composite all shapes (in order) onto a fresh canvas.

    Returns (H, W, 3) uint8 by default, or (H, W, 4) RGBA when
    `transparent_bg=True`. RGBA mode is used when reloading a sticker-mode
    JSON so the preview pane shows true transparency outside the painted
    silhouette instead of a solid white rectangle.
    """
    if transparent_bg:
        # Sticker preview path: paint shapes onto a neutral grey backdrop
        # (mirrors what the live engine uses in sticker mode) and build the
        # alpha channel as the union of every shape's coverage mask. ImageView
        # renders the result with Format_RGBA8888 so unpainted areas are truly
        # transparent against the pane background.
        canvas = np.full((height, width, 3), 40, dtype=np.uint8)
        alpha = np.zeros((height, width), dtype=np.uint8)
        for s in shapes:
            mask_local, bbox = s.rasterize_mask(width, height)
            x0, y0, x1, y1 = bbox
            if x1 <= x0 or y1 <= y0 or mask_local.size == 0:
                continue
            color = s.color
            a_shape = (color[3] / 255.0) if len(color) >= 4 else 1.0
            region_cur = canvas[y0:y1, x0:x1].astype(np.float32)
            src = np.array(color[:3], dtype=np.float32)
            m = (mask_local.astype(np.float32) / 255.0)[:, :, None]
            blended = m * (a_shape * src + (1.0 - a_shape) * region_cur) + (1.0 - m) * region_cur
            canvas[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)
            alpha[y0:y1, x0:x1] = np.maximum(alpha[y0:y1, x0:x1], mask_local)
        return np.dstack([canvas, alpha])

    if background == "auto":
        canvas = _checkerboard(width, height) if _is_stickerlike(shapes) \
                 else np.full((height, width, 3), 255, dtype=np.uint8)
    else:
        canvas = np.full((height, width, 3), background, dtype=np.uint8)
    for s in shapes:
        mask_local, bbox = s.rasterize_mask(width, height)
        x0, y0, x1, y1 = bbox
        if x1 <= x0 or y1 <= y0 or mask_local.size == 0:
            continue
        color = s.color  # use the shape's saved color (don't re-optimize)
        a = (color[3] / 255.0) if len(color) >= 4 else 1.0
        region_cur = canvas[y0:y1, x0:x1].astype(np.float32)
        src = np.array(color[:3], dtype=np.float32)
        m = (mask_local.astype(np.float32) / 255.0)[:, :, None]
        blended = m * (a * src + (1.0 - a) * region_cur) + (1.0 - m) * region_cur
        canvas[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)
    return canvas
