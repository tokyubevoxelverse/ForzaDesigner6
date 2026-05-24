"""Strip wasted shapes from FD6-format vinyl JSONs.

Two passes, both enabled by default, both visually safe (the rendered final canvas
is essentially identical with or without them):

1. PADDING-WHITES — drop white shapes (RGB >= 230 all channels) whose center is in
   the 8% edge-padding margin around the source content. Targets the legacy bug
   (v0.1.1 and earlier) where the notebook filled opaque-mode padding with hardcoded
   solid white, causing greedy to spend ~100-300 shapes painting a phantom solid-
   white border that contributes nothing to the actual content. Fixed at the source
   in v0.1.2; this CLI cleans up any JSON generated before that fix.

2. DEAD SHAPES — drop shapes whose visible-pixel count in the final composite is
   below threshold (default 5). Computed by walking shapes top-to-bottom in z-order:
   each shape's "visible" mask = its full mask AND NOT yet claimed by a higher-z
   shape. With lock_alpha=True (the production default) every alpha=255 shape fully
   occludes its mask region, so shapes painted early can be completely buried by
   later commits — they contribute zero pixels to the final render but still eat a
   layer slot in the JSON and a write in the injector loop.

Usage:
    fap-clean input.json                      # → input_cleaned.json (both passes)
    fap-clean input.json -o output.json       # specify output path
    fap-clean input.json --in-place           # overwrite the input file
    fap-clean input.json --report             # print stats, don't write
    fap-clean input.json --no-padding-whites  # disable padding-white pass
    fap-clean input.json --no-dead-shapes     # disable dead-shape pass
    fap-clean input.json --min-visible-px 10  # tighter dead-shape threshold

Pure CPU — no torch / GPU required. For 3000 shapes on a 1263x765 canvas the
typical runtime is ~3-8 seconds (dominated by the per-shape mask rasterization).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from forza_abyss_painter.shapegen.shapes import shape_from_json


# Padding margin formula must match notebooks/build_colab_notebook.py's
# _load_image_bytes() — both compute pad_px = max(8, round(max(w,h) * 0.08)).
_BUFFER_FRAC = 0.08
_PAD_MIN = 8
# White threshold: all three RGB channels must be >= this to count as "white".
# Matches the user-observed cleanup pattern (off-white pearls/eyes inside content
# regions stay below this cutoff because they're typically 200-225).
_WHITE_THRESHOLD = 230


def _padding_margin_px(image_size: tuple[int, int]) -> int:
    w, h = image_size
    return max(_PAD_MIN, int(round(max(w, h) * _BUFFER_FRAC)))


def _is_white(color: list[int]) -> bool:
    return (color[0] >= _WHITE_THRESHOLD
            and color[1] >= _WHITE_THRESHOLD
            and color[2] >= _WHITE_THRESHOLD)


def _identify_padding_whites(shapes: list[dict], image_size: tuple[int, int]) -> set[int]:
    """Indices of shapes that are (a) white and (b) centered in the padding margin."""
    w, h = image_size
    pad = _padding_margin_px(image_size)
    inside_x0, inside_y0 = pad, pad
    inside_x1, inside_y1 = w - pad, h - pad
    drop: set[int] = set()
    for i, s in enumerate(shapes):
        if not _is_white(s["color"]):
            continue
        x, y = s["x"], s["y"]
        if not (inside_x0 <= x <= inside_x1 and inside_y0 <= y <= inside_y1):
            drop.add(i)
    return drop


def _compute_visible_pixel_counts(shapes: list[dict],
                                   image_size: tuple[int, int]) -> np.ndarray:
    """For each shape (in JSON order), count how many pixels would be visible in the
    final composite under z-order with alpha=255. Returns int64 array of length n.

    Walks top-down: starts at the LAST shape in JSON order (topmost in z), accumulates
    a `claimed` mask of pixels already painted. For each lower shape, visible = mask
    AND NOT claimed. Then claimed |= mask.

    For non-ellipse shapes (rare in production — almost all output is rotated_ellipse)
    the same logic applies — rasterize_mask returns a bool mask and a bbox; we paste
    into the claim array at the bbox."""
    w, h = image_size
    n = len(shapes)
    claimed = np.zeros((h, w), dtype=bool)
    visible_counts = np.zeros(n, dtype=np.int64)
    shape_objs = [shape_from_json(s) for s in shapes]
    for i in reversed(range(n)):
        mask, bbox = shape_objs[i].rasterize_mask(w, h)
        x0, y0, x1, y1 = bbox
        if x1 <= x0 or y1 <= y0:
            continue   # degenerate shape (off-canvas or zero-area)
        mask_bool = mask > 0
        claim_slice = claimed[y0:y1, x0:x1]
        visible_count = int(np.count_nonzero(mask_bool & ~claim_slice))
        visible_counts[i] = visible_count
        claim_slice |= mask_bool   # in-place — modifies the claimed array
    return visible_counts


def clean_doc(doc: dict,
              drop_padding_whites: bool = True,
              drop_dead_shapes: bool = True,
              min_visible_px: int = 5) -> tuple[dict, dict]:
    """Returns (cleaned_doc, report). Doesn't mutate input.

    report has int counts:
        input_count
        output_count
        dropped_total
        dropped_padding_whites_only
        dropped_dead_shapes_only
        dropped_both_conditions
    """
    shapes = doc.get("shapes", [])
    image_size_raw = doc.get("image_size")
    if not (isinstance(image_size_raw, (list, tuple)) and len(image_size_raw) == 2):
        raise ValueError(
            "doc is missing image_size [w, h] — required for padding-margin + visibility "
            "computations. Add image_size to the doc or skip this CLI."
        )
    image_size = (int(image_size_raw[0]), int(image_size_raw[1]))

    drop_pw: set[int] = (
        _identify_padding_whites(shapes, image_size)
        if drop_padding_whites else set()
    )
    if drop_dead_shapes:
        visible = _compute_visible_pixel_counts(shapes, image_size)
        drop_ds: set[int] = {i for i in range(len(shapes)) if visible[i] < min_visible_px}
    else:
        drop_ds = set()

    drop_all = drop_pw | drop_ds
    kept = [s for i, s in enumerate(shapes) if i not in drop_all]
    cleaned = dict(doc)
    cleaned["shapes"] = kept
    if "shape_count" in cleaned:
        cleaned["shape_count"] = len(kept)

    report = {
        "input_count": len(shapes),
        "output_count": len(kept),
        "dropped_total": len(drop_all),
        "dropped_padding_whites_only": len(drop_pw - drop_ds),
        "dropped_dead_shapes_only": len(drop_ds - drop_pw),
        "dropped_both_conditions": len(drop_pw & drop_ds),
    }
    return cleaned, report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fap-clean",
        description="Strip wasted shapes (padding-margin whites + fully-occluded "
                    "dead weight) from FD6-format vinyl JSONs. Pure cleanup — the "
                    "rendered final canvas is visually identical.",
    )
    parser.add_argument("input", help="path to input JSON")
    parser.add_argument("-o", "--output",
                       help="output path (default: <input>_cleaned.json)")
    parser.add_argument("--in-place", action="store_true",
                       help="overwrite the input file (mutually exclusive with -o)")
    parser.add_argument("--report", action="store_true",
                       help="print stats, don't write any output file")
    parser.add_argument("--no-padding-whites", dest="padding_whites",
                       action="store_false", default=True,
                       help="don't drop padding-margin white shapes")
    parser.add_argument("--no-dead-shapes", dest="dead_shapes",
                       action="store_false", default=True,
                       help="don't drop shapes with low visible-pixel count")
    parser.add_argument("--min-visible-px", type=int, default=5,
                       help="threshold for dead-shape pruning (default: 5)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    input_path = Path(args.input)

    if not input_path.exists():
        print(f"error: input file not found: {input_path}", file=sys.stderr)
        return 2

    if args.in_place and args.output:
        print("error: --in-place and -o/--output are mutually exclusive", file=sys.stderr)
        return 2

    doc = json.loads(input_path.read_text())
    try:
        cleaned, report = clean_doc(
            doc,
            drop_padding_whites=args.padding_whites,
            drop_dead_shapes=args.dead_shapes,
            min_visible_px=args.min_visible_px,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"input:   {input_path.name}  ({report['input_count']} shapes)")
    print(f"dropped: {report['dropped_total']}  "
          f"(padding-whites: {report['dropped_padding_whites_only']}, "
          f"dead-shapes: {report['dropped_dead_shapes_only']}, "
          f"both: {report['dropped_both_conditions']})")

    if args.report:
        print("(--report mode; no file written)")
        return 0

    if args.in_place:
        output_path = input_path
    elif args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_name(input_path.stem + "_cleaned.json")

    output_path.write_text(json.dumps(cleaned, indent=2))
    print(f"output:  {output_path.name}  ({report['output_count']} shapes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
