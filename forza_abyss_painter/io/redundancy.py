from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image

from forza_abyss_painter.io.exporter import load_json, save_json
from forza_abyss_painter.io.json_schema import FD6Document
from forza_abyss_painter.shapegen.scoring import composite, rms_error
from forza_abyss_painter.shapegen.shapes import Shape


def _replay(shapes: list[Shape], target: np.ndarray) -> tuple[np.ndarray, float]:
    avg = target.reshape(-1, 3).mean(axis=0).astype(np.uint8)
    canvas = np.tile(avg, (target.shape[0], target.shape[1], 1)).astype(np.uint8)
    for s in shapes:
        canvas, _ = composite(canvas, s, target)
    return canvas, rms_error(canvas, target)


def redundancy_pass(
    json_path: str | Path,
    image_path: str | Path,
    tolerance: float = 0.5,
) -> tuple[Path, int, int]:
    """Drop shapes whose removal changes RMS by less than `tolerance`.

    Returns (output_path, original_count, kept_count). Writes back to a sibling file
    with `_pruned` appended to the stem.
    """
    json_path = Path(json_path)
    image_path = Path(image_path)
    doc = load_json(json_path)
    shapes = doc.materialize_shapes()
    img = Image.open(image_path).convert("RGB")
    if img.size != tuple(doc.image_size):
        img = img.resize(doc.image_size, Image.LANCZOS)
    target = np.asarray(img, dtype=np.uint8)

    _, baseline_rms = _replay(shapes, target)
    kept: list[Shape] = []
    for i, s in enumerate(shapes):
        trial = kept + shapes[i + 1:]
        _, trial_rms = _replay(trial, target)
        if trial_rms - baseline_rms > tolerance:
            kept.append(s)
        # else: shape is redundant; drop it

    new_doc = FD6Document.from_engine(
        source_image=doc.source_image,
        image_size=doc.image_size,
        shapes=kept,
        profile_name=doc.profile + "+pruned",
    )
    out_path = json_path.with_name(json_path.stem + "_pruned.json")
    save_json(new_doc, out_path)
    return out_path, len(shapes), len(kept)
