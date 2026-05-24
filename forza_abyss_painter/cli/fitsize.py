"""Local input-resizer: pick the optimal canvas long side for a GPU + preset.

Usage:
    python -m forza_abyss_painter.cli.fitsize <image-or-dir> --preset highres_3000 --card blackwell-96
    fd6-fitsize <image-or-dir> --preset highres_3000 --card blackwell-96   # after pip install -e .

Memory model (calibrated; valid for L <= 2048, above which the bbox crop caps at 256):
    peak_bytes ~= 5 * K * L^2     (K = random_samples, L = canvas long side)
Target long side is the smallest of three caps:
    device  = sqrt(budget / (5*K))     budget = usable_GiB * 2^30 * 0.85
    quality = sqrt(N) * 28             detail-matched ceiling (GPU_SHAPEGEN.md §5)
    source  = source long side         never upscale
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from PIL import Image

from forza_abyss_painter.shapegen.cards import usable_gib
from forza_abyss_painter.shapegen.presets import PRESETS

BUDGET_FACTOR = 0.85          # headroom for canvas/edge/gradient/fragmentation overhead
BYTES_PER_GIB = 1024 ** 3
PEAK_COEFF = 5                # peak_bytes ~= PEAK_COEFF * K * L^2
QUALITY_COEFF = 28            # detail-matched long side ~= sqrt(N) * 28


def peak_bytes(long_side: int, random_samples: int) -> float:
    """Calibrated GPU scoring-batch peak in bytes. Valid for long_side <= 2048."""
    return float(PEAK_COEFF * random_samples * long_side * long_side)


def preset_params(preset: str) -> tuple[int, int]:
    """(num_shapes, random_samples) for a preset, read from the frozen baseline."""
    p = PRESETS[preset]
    return p["num_shapes"], p["random_samples"]


def plan_size(
    source_long: int, num_shapes: int, random_samples: int, usable_gib: int
) -> tuple[int, str, dict[str, float]]:
    """Return (target_long, binding_cap_name, caps). target_long never exceeds source_long."""
    budget = usable_gib * BYTES_PER_GIB * BUDGET_FACTOR
    caps = {
        "device": math.sqrt(budget / (PEAK_COEFF * random_samples)),
        "quality": math.sqrt(num_shapes) * QUALITY_COEFF,
        "source": float(source_long),
    }
    # tie-break by insertion order: device > quality > source
    binding = min(caps, key=caps.get)
    return int(caps[binding]), binding, caps  # floor: stay within budget


def output_name(stem: str, preset: str, card: str, target_long: int) -> str:
    """Self-documenting filename: <stem>_<preset>_<card>_<L>px.png (preset/card separators stripped)."""
    ptag = preset.replace("_", "")
    ctag = card.replace("-", "")
    return f"{stem}_{ptag}_{ctag}_{target_long}px.png"


def resize_to(img: "Image.Image", target_long: int) -> tuple["Image.Image", bool]:
    """Downscale img so its long side == target_long. Returns (image, changed). Never upscales."""
    if max(img.size) <= target_long:
        return img, False  # caller's object — do not mutate

    out = img.copy()
    out.thumbnail((target_long, target_long), Image.LANCZOS)
    return out, True


def _gather_images(path: Path) -> list[Path]:
    """Return a sorted list of PNGs: [path] for a file, or glob for a directory."""
    if path.is_dir():
        return sorted(path.glob("*.png"))
    return [path]


def _report(src: Path, src_long: int, target: int, binding: str,
            caps: dict[str, float], random_samples: int) -> str:
    """Format a human-readable per-image sizing report string."""
    lines = [f"{src.name}: {src_long}px -> {target}px  (binding: {binding})"]
    for name in ("device", "quality", "source"):
        mark = " <-" if name == binding else ""
        lines.append(f"    {name:>7}: {int(caps[name]):7d}px{mark}")
    gb = peak_bytes(target, random_samples) / 1e9
    lines.append(f"    predicted peak VRAM at target: ~{gb:.0f} GB")
    if binding == "device":
        lines.append("    note: VRAM-limited (below the detail your shape budget could use). "
                     "Use a bigger card, fewer shapes, or a preset with lower RANDOM_SAMPLES.")
    elif binding == "source":
        lines.append("    note: source-limited. A higher-res source would help (up to the "
                     "quality cap).")
    elif binding == "quality":
        lines.append("    note: optimal (detail-matched), with VRAM to spare.")
    if target >= src_long:
        lines.append("    already within budget — copied through unchanged.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Entry point: parse args, size each image, write fit/ outputs. Returns exit code."""
    parser = argparse.ArgumentParser(
        prog="fd6-fitsize",
        description="Resize a source image to the optimal canvas long side for a GPU + preset.",
    )
    parser.add_argument("path", type=Path, help="image file or directory of PNGs")
    parser.add_argument("--preset", required=True, choices=sorted(PRESETS))
    parser.add_argument("--card", required=True, help="GPU card key (see forza_abyss_painter.shapegen.cards)")
    parser.add_argument("--out", type=Path, default=None,
                        help="output dir (default: a 'fit/' subfolder beside the source)")
    args = parser.parse_args(argv)

    try:
        gib = usable_gib(args.card)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    num_shapes, random_samples = preset_params(args.preset)
    images = _gather_images(args.path)
    if not images:
        print(f"error: no PNG images found at {args.path}", file=sys.stderr)
        return 2

    if not args.path.is_dir() and not args.path.is_file():
        print(f"error: file not found: {args.path}", file=sys.stderr)
        return 2

    base = args.path if args.path.is_dir() else args.path.parent
    out_dir = args.out or (base / "fit")
    out_dir.mkdir(parents=True, exist_ok=True)

    for src in images:
        img = Image.open(src).convert("RGBA")
        src_long = max(img.size)
        target, binding, caps = plan_size(src_long, num_shapes, random_samples, gib)
        out, _ = resize_to(img, target)
        out_path = out_dir / output_name(src.stem, args.preset, args.card, target)
        out.save(out_path)
        print(_report(src, src_long, target, binding, caps, random_samples))
        print(f"    wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
