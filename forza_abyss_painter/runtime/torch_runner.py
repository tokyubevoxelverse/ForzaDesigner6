"""Subprocess entry point for GPU shape-gen.

The EXE-side Phase 3 GPU pipeline runs the shape generator in an
isolated subprocess that lives inside the on-demand torch runtime
(see torch_installer.py). This module is what that subprocess
actually executes. From the main EXE:

    proc = Popen(
        [embedded_python_exe(), "-m",
         "forza_abyss_painter.runtime.torch_runner",
         "--config", str(config_path)],
        stderr=PIPE, stdout=PIPE,
    )
    for raw in iter(proc.stderr.readline, b""):
        event = json.loads(raw.decode("utf-8"))
        emit_qt_signal(event)
    proc.wait()
    if proc.returncode == 0:
        return load_json(cfg.output_json_path)

## IPC protocol (pinned by tests/test_torch_runner.py)

Config file is JSON written by the EXE before spawn. Required fields:
    image_path          str — absolute path to source RGB(A) image
    output_json_path    str — absolute path the runner writes final JSON to
    num_shapes          int — total shapes the engine commits
    max_resolution      int — long-side ceiling for engine downscale
    random_samples      int — seed-batch size per shape (the K in K*H*W*12)
    sticker_mode        bool — alpha-mask path (transparent areas skipped)

Optional fields with defaults that mirror GPUConfig defaults:
    seed                int (default 0 = time-based)
    edge_strength       float (default 0.0)
    posterize_levels    int (default 0)
    bbox_local          bool (default False)
    joint_polish_steps  int (default 0)
    lock_alpha          bool (default True; raising False is rejected)
    progress_every      int (default 0 — no progress events)
    checkpoint_every    int (default 0 — no checkpoint events)
    device              str (default "cuda"; "cpu" allowed for testing)

Stderr is the progress channel. Each event is one JSON object on its
own line, flushed immediately. Event kinds:

    {"kind": "started",  "cfg_summary": {…subset of config…}}
    {"kind": "progress", "shape_count": N, "total": T}
    {"kind": "checkpoint", "shape_count": N, "shapes": [...]}  # optional
    {"kind": "done", "output_path": "...", "shape_count": N}
    {"kind": "error", "stage": "<phase>", "message": "..."}

Exit codes:
    0  success (a "done" event preceded this exit)
    1  engine runtime error (e.g., CUDA OOM, image load failed)
    2  config-load error (bad JSON, missing required fields)
    3  import error (torch not installed in this Python — user must
       install the runtime via the runtime-install dialog first)

The main EXE treats any non-zero exit + an "error" event as a clean
failure that propagates a modal to the user. A non-zero exit WITHOUT
a preceding "error" event is a runner bug — the IPC contract requires
emitting "error" before non-zero exit.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path


# ----------------------------------------------------------- IPC: config


@dataclass
class RunConfig:
    """Parsed + validated config from the EXE.

    Mirrors the subset of GPUConfig the runner actually exposes. Fields
    that are GPUConfig-internal (shape_types, alpha_levels, etc.) stay
    on GPUConfig defaults — the runner doesn't expose them across the
    IPC boundary because the EXE-side presets fix them. Future fields
    that the EXE wants to expose are added here AND to to_dict / from_dict.
    """

    image_path: Path
    output_json_path: Path
    num_shapes: int
    max_resolution: int
    random_samples: int
    sticker_mode: bool = False

    seed: int = 0
    edge_strength: float = 0.0
    posterize_levels: int = 0
    bbox_local: bool = False
    joint_polish_steps: int = 0
    lock_alpha: bool = True
    progress_every: int = 0
    checkpoint_every: int = 0
    device: str = "cuda"
    # VRAM budget for chunked-K mode (0 = no budget = full K in one
    # pass; the engine uses this to derive chunk_size).
    vram_budget_gib: float = 0.0

    # Optional human-readable label echoed back in the started event.
    # Used by the EXE's status bar / progress label.
    preset_label: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "RunConfig":
        """Parse + validate. Raises ValueError on missing required fields,
        wrong types, or out-of-range values."""
        # Required fields.
        try:
            image_path = Path(d["image_path"])
            output_json_path = Path(d["output_json_path"])
            num_shapes = int(d["num_shapes"])
            max_resolution = int(d["max_resolution"])
            random_samples = int(d["random_samples"])
        except KeyError as exc:
            raise ValueError(f"missing required config field: {exc}") from exc
        # Range checks — silent garbage values would only manifest as
        # deep-engine errors that are confusing to triage.
        if num_shapes < 1:
            raise ValueError(f"num_shapes must be >= 1, got {num_shapes}")
        if max_resolution < 64:
            raise ValueError(f"max_resolution must be >= 64, got {max_resolution}")
        if random_samples < 1:
            raise ValueError(f"random_samples must be >= 1, got {random_samples}")
        # Optional with type coercion.
        lock_alpha = bool(d.get("lock_alpha", True))
        if not lock_alpha:
            # Hard system constraint — see GPUConfig.lock_alpha docstring.
            # Surface the violation EARLY so the EXE doesn't waste minutes
            # running a shape-gen whose output the injector would reject.
            raise ValueError(
                "lock_alpha=False is not a supported value — the Forza "
                "injector writes alpha=255 at inject time and any non-opaque "
                "JSON breaks preview/in-game parity."
            )
        device = str(d.get("device", "cuda"))
        if device not in ("cuda", "cpu"):
            raise ValueError(f"device must be 'cuda' or 'cpu', got {device!r}")
        return cls(
            image_path=image_path,
            output_json_path=output_json_path,
            num_shapes=num_shapes,
            max_resolution=max_resolution,
            random_samples=random_samples,
            sticker_mode=bool(d.get("sticker_mode", False)),
            seed=int(d.get("seed", 0)),
            edge_strength=float(d.get("edge_strength", 0.0)),
            posterize_levels=int(d.get("posterize_levels", 0)),
            bbox_local=bool(d.get("bbox_local", False)),
            joint_polish_steps=int(d.get("joint_polish_steps", 0)),
            vram_budget_gib=float(d.get("vram_budget_gib", 0.0)),
            lock_alpha=lock_alpha,
            progress_every=int(d.get("progress_every", 0)),
            checkpoint_every=int(d.get("checkpoint_every", 0)),
            device=device,
            preset_label=str(d.get("preset_label", "")),
        )

    def summary(self) -> dict:
        """Echoed back in the 'started' event so the EXE's progress label
        can show e.g., 'GPU 700 shapes @ 600px (preset: Headshot 700)'.
        Drops file paths (the EXE already knows them) + redundant defaults."""
        return {
            "num_shapes": self.num_shapes,
            "max_resolution": self.max_resolution,
            "random_samples": self.random_samples,
            "sticker_mode": self.sticker_mode,
            "device": self.device,
            "preset_label": self.preset_label,
        }


# ----------------------------------------------------------- IPC: emit


def emit(stream, event: dict) -> None:
    """Write one event to the IPC channel as a single JSON line + flush.

    Stream is parameterized (vs hardcoded sys.stderr) so tests can capture
    the output without monkey-patching sys.stderr globally. Production
    callers pass sys.stderr.
    """
    stream.write(json.dumps(event) + "\n")
    stream.flush()


# --------------------------------------------------------- runner core


def _load_image(image_path: Path):
    """Load + decode RGB(A) image. Returns (rgb_uint8_HxWx3, alpha_or_None).
    Imports PIL/numpy lazily so config-load errors don't crash on missing
    optional deps."""
    import numpy as np
    from PIL import Image

    img = Image.open(image_path)
    has_alpha = img.mode in ("RGBA", "LA") or (
        img.mode == "P" and "transparency" in img.info
    )
    if has_alpha:
        rgba = img.convert("RGBA")
        arr = np.asarray(rgba, dtype=np.uint8)
        rgb = arr[:, :, :3].copy()
        alpha = arr[:, :, 3].copy()
        return rgb, alpha
    return np.asarray(img.convert("RGB"), dtype=np.uint8), None


def _downscale_to_max_resolution(rgb, alpha, max_resolution: int):
    """Match the CPU worker's pre-engine downscale step so GPU + CPU
    outputs come from the same canvas size for the same preset."""
    import numpy as np
    from PIL import Image

    h, w = rgb.shape[:2]
    if max(h, w) <= max_resolution:
        return rgb, alpha
    scale = max_resolution / max(h, w)
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    rgb_resized = np.asarray(
        Image.fromarray(rgb, "RGB").resize(new_size, Image.LANCZOS),
        dtype=np.uint8,
    )
    if alpha is not None:
        alpha = np.asarray(
            Image.fromarray(alpha, "L").resize(new_size, Image.LANCZOS),
            dtype=np.uint8,
        )
    return rgb_resized, alpha


def _install_graceful_shutdown_handler(stream, logger) -> None:
    """Catch SIGINT (Unix) / CTRL_BREAK_EVENT (Windows) so the parent
    EXE can request a graceful stop. Handler emits a 'cancelled' event
    on stderr, asks torch to release the CUDA allocator + reset peak
    stats so the next run starts clean, then exits with a non-zero
    code. Without this handler, the only way to stop a runaway GPU
    run is the parent calling TerminateProcess — which doesn't give
    the runner a chance to free the cache → driver doesn't reclaim
    VRAM → GPU stuck until PC restart.
    """
    import signal as _signal

    def _shutdown(signum, frame):
        emit(stream, {
            "kind": "error", "stage": "cancelled",
            "message": (
                f"Runner received signal {signum} — releasing CUDA cache "
                f"and exiting."
            ),
        })
        logger.log("runner_cancelled_signal", signum=int(signum))
        # Best-effort cache release. If torch isn't loaded yet (signal
        # arrived during config-load), this is a no-op via ImportError.
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                logger.log("runner_cuda_cache_released")
        except Exception as exc:
            logger.log_exception("runner_cancel_cleanup_failed", exc)
        # Exit with code 1 (engine_run-class failure) so the parent's
        # IPC parser routes through the error path, not the done path.
        sys.exit(1)

    # Windows: CTRL_BREAK_EVENT (SIGBREAK) is what the parent sends to
    # CREATE_NEW_PROCESS_GROUP'd children. SIGTERM doesn't exist; if
    # the parent eventually escalates to TerminateProcess, we can't
    # catch that anyway (kernel-level kill).
    # Unix: SIGINT is what the parent sends; SIGTERM is the escalation.
    if sys.platform == "win32":
        _signal.signal(_signal.SIGBREAK, _shutdown)
    else:
        _signal.signal(_signal.SIGINT, _shutdown)
        _signal.signal(_signal.SIGTERM, _shutdown)


def run(cfg: RunConfig, stream=sys.stderr) -> int:
    """Execute the configured run. Returns exit code."""
    # Subprocess gets its own log session file (process_label='runner')
    # so the diagnostics bundle ships both halves of the run (main +
    # runner) side-by-side. Failures here often DON'T reach the main
    # process's stderr parser (e.g., torch import crash with no traceback),
    # so this on-disk log is the only diagnostic path.
    try:
        from forza_abyss_painter.runtime.gpu_logger import get_gpu_logger
        logger = get_gpu_logger(process_label="runner")
        logger.log("runner_started", cfg_summary=cfg.summary())
        _install_graceful_shutdown_handler(stream, logger)
    except Exception:   # pragma: no cover — defensive
        # If even gpu_logger fails to import (gpu_logger isn't in the
        # embedded site-packages copy), don't crash the runner. Fall
        # back to a no-op stub so the rest of run() still works.
        class _NoopLogger:
            def log(self, *a, **kw): pass
            def log_exception(self, *a, **kw): pass
            def start_phase(self, *a, **kw):
                from contextlib import nullcontext
                return nullcontext()
        logger = _NoopLogger()
    emit(stream, {"kind": "started", "cfg_summary": cfg.summary()})

    # Lazy-import the engine so a bad config (caught in main()) doesn't
    # need torch to be installed first. Also lets us emit a clean
    # 'import_engine' error event if the user clicked Generate before
    # the runtime installer completed.
    try:
        with logger.start_phase("import_engine"):
            from forza_abyss_painter.shapegen.gpu.engine import GPUConfig, run_gpu
    except ImportError as exc:
        emit(stream, {
            "kind": "error", "stage": "import_engine",
            "message": (
                f"{type(exc).__name__}: {exc}. The GPU runtime isn't "
                f"fully installed in this Python — run the runtime "
                f"installer from Tools → Generate shapes locally first."
            ),
        })
        return 3

    # Load + downscale image.
    try:
        with logger.start_phase("load_image", image_path=str(cfg.image_path)):
            rgb, alpha_mask = _load_image(cfg.image_path)
            logger.log("image_loaded",
                       shape=list(rgb.shape),
                       has_alpha=alpha_mask is not None)
            rgb, alpha_mask = _downscale_to_max_resolution(
                rgb, alpha_mask, cfg.max_resolution,
            )
            logger.log("image_downscaled",
                       final_shape=list(rgb.shape),
                       max_resolution=cfg.max_resolution)
    except (OSError, ValueError) as exc:
        emit(stream, {
            "kind": "error", "stage": "load_image",
            "message": f"{type(exc).__name__}: {exc}",
        })
        return 1

    # Build the engine config from the runner config.
    gpu_cfg = GPUConfig(
        num_shapes=cfg.num_shapes,
        random_samples=cfg.random_samples,
        seed=cfg.seed,
        edge_strength=cfg.edge_strength,
        posterize_levels=cfg.posterize_levels,
        bbox_local=cfg.bbox_local,
        joint_polish_steps=cfg.joint_polish_steps,
        vram_budget_gib=cfg.vram_budget_gib,
        lock_alpha=cfg.lock_alpha,
    )

    # Checkpoint callback streams progress to stderr. We use the engine's
    # checkpoint_cb (not progress_every — which is print-only) because
    # it carries the current shapes-so-far list, which is what the EXE's
    # progress bar wants for its periodic preview render.
    def _checkpoint_cb(shape_idx: int, shapes_so_far: list) -> None:
        emit(stream, {
            "kind": "checkpoint",
            "shape_count": shape_idx,
            "total": cfg.num_shapes,
        })

    # Drive the run.
    try:
        # progress_every and checkpoint_every BOTH drive emit() — at
        # different cadences. progress_every is cheap (just a number);
        # checkpoint_every includes the full shape list (heavier).
        with logger.start_phase("run_gpu",
                                 num_shapes=cfg.num_shapes,
                                 random_samples=cfg.random_samples):
            shapes_list, _final_canvas = run_gpu(
                target_rgb=rgb,
                cfg=gpu_cfg,
                alpha_mask=alpha_mask if cfg.sticker_mode else None,
                progress_every=cfg.progress_every,
                checkpoint_cb=_checkpoint_cb if cfg.checkpoint_every > 0 else None,
                checkpoint_every=cfg.checkpoint_every,
            )
            logger.log("run_gpu_done", shapes_count=len(shapes_list))
    except RuntimeError as exc:
        # run_gpu's OOM wrapper converts torch.cuda.OutOfMemoryError into
        # a RuntimeError with an actionable recipe — pass that through
        # unchanged so the EXE's modal can show the user the exact knob
        # values to lower.
        emit(stream, {
            "kind": "error", "stage": "engine_run",
            "message": f"{type(exc).__name__}: {exc}",
        })
        return 1
    except Exception as exc:  # pragma: no cover — broad catch for IPC safety
        emit(stream, {
            "kind": "error", "stage": "engine_run",
            "message": f"{type(exc).__name__}: {exc}",
        })
        return 1

    # Write final JSON via the canonical save_json path so it lands in the
    # exact same schema the CPU worker emits — load_json on the EXE side
    # will round-trip cleanly. See §3 of CLAUDE.md.
    try:
        with logger.start_phase("save_json",
                                 output_path=str(cfg.output_json_path)):
            from forza_abyss_painter.io.exporter import save_json
            from forza_abyss_painter.io.json_schema import FD6Document
            h, w = rgb.shape[:2]
            doc = FD6Document.from_engine(
                source_image=cfg.image_path.name,
                image_size=(w, h),   # JSON convention is (width, height)
                shapes=_shape_dicts_to_objects(shapes_list),
                profile_name=cfg.preset_label,
                sticker_mode=cfg.sticker_mode,
            )
            save_json(doc, cfg.output_json_path)
    except (OSError, ValueError, KeyError) as exc:
        emit(stream, {
            "kind": "error", "stage": "save_json",
            "message": f"{type(exc).__name__}: {exc}",
        })
        return 1

    logger.log("runner_done",
               output_path=str(cfg.output_json_path),
               shape_count=len(shapes_list))
    emit(stream, {
        "kind": "done",
        "output_path": str(cfg.output_json_path),
        "shape_count": len(shapes_list),
    })
    return 0


def _shape_dicts_to_objects(shape_dicts: list):
    """run_gpu returns shapes as JSON-ready dicts; FD6Document.from_engine
    expects Shape objects that have a .to_json() method. Re-materialize
    via the shape registry so the canonical schema applies."""
    from forza_abyss_painter.shapegen.shapes import shape_from_json
    return [shape_from_json(d) for d in shape_dicts]


# --------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="torch_runner",
        description="GPU shape-gen subprocess (Phase 3 EXE runtime).",
    )
    parser.add_argument("--config", required=True, type=Path,
                        help="path to JSON config (see module docstring "
                             "for required + optional fields)")
    args = parser.parse_args(argv)

    # Config load. Failures here exit 2 — distinct from runtime errors (1)
    # so the EXE's modal can tell the user "we mis-built the config"
    # vs. "the engine itself failed".
    try:
        data = json.loads(Path(args.config).read_text(encoding="utf-8"))
        cfg = RunConfig.from_dict(data)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        emit(sys.stderr, {
            "kind": "error", "stage": "config_load",
            "message": f"{type(exc).__name__}: {exc}",
        })
        return 2

    return run(cfg)


if __name__ == "__main__":
    sys.exit(main())
