from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fd6.shapegen.benchmark import benchmark_line_guide, load_benchmark_inputs, write_benchmark_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=None)
    parser.add_argument("--guide", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--output", default="docs/line_guide_measurements.json")
    parser.add_argument("--size", type=int, default=40)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--shapes", type=int, default=16)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--mutations", type=int, default=8)
    parser.add_argument("--backend", choices=("cpu", "gpu", "auto"), default="cpu")
    args = parser.parse_args()

    (
        target,
        guide,
        guide_source,
        guide_prepare_seconds,
        guide_prepare_vram_peak_mb,
    ) = load_benchmark_inputs(
        args.image,
        args.guide,
        size=args.size,
        model_path=args.model,
        prefer_gpu=args.backend != "cpu",
    )
    command = " ".join(sys.argv)
    report = benchmark_line_guide(
        target,
        guide,
        seed=args.seed,
        stop_at=args.shapes,
        random_samples=args.samples,
        mutated_samples=args.mutations,
        compute_backend=args.backend,
        guide_source=guide_source,
        guide_prepare_seconds=guide_prepare_seconds,
        guide_prepare_vram_peak_mb=guide_prepare_vram_peak_mb,
        command=command,
    )
    write_benchmark_report(report, args.output)
    print(json.dumps(report["delta"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
