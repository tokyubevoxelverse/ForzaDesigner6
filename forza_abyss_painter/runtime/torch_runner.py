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

Config file is JSON written by the EXE before spawn.

Always-required fields (both modes):
    image_path          str — absolute path to source RGB(A) image
    output_json_path    str — absolute path the runner writes final JSON to

Required in fresh mode only (mode='fresh', the default):
    num_shapes          int — total shapes the engine commits
    max_resolution      int — long-side ceiling for engine downscale
    random_samples      int — seed-batch size per shape (the K in K*H*W*12)

    sticker_mode        bool — alpha-mask path (transparent areas skipped)

Mode dispatch (#85 #86):
    mode                str (default "fresh") — "fresh" or "polish_only".

    In "fresh" mode (default): num_shapes, max_resolution, and
    random_samples are all required. The engine generates shapes from
    the supplied image.

    In "polish_only" mode: input_shapes_path is required;
    num_shapes/max_resolution/random_samples are NOT required and are
    ignored (canvas dims come from the loaded JSON's image_size). The
    engine skips generation and runs joint_polish() on the loaded
    shapes against the supplied image.

    input_shapes_path   str — (polish_only required) absolute path to
                        an existing fd6.shapes JSON file to polish.
    polish_steps_override  int >= 1 (optional) — overrides the default
                        150 polish steps. Only read in polish_only mode.

Optional fields with defaults that mirror GPUConfig defaults:
    seed                int (default 0 = time-based)
    edge_strength       float (default 0.0)
    posterize_levels    int (default 0)
    bbox_local          bool (default True — production EXE path; the
                        full_canvas path materializes (K, H, W) masks
                        in one shot inside rasterize_hard which OOMs
                        at K=8192 + 720px on 32 GiB cards; chunking is
                        only wired for scoring, not rasterize. Set to
                        False ONLY for multi-shape / non-ellipse runs
                        once strategy-B chunked-rasterize lands.)
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
from dataclasses import dataclass
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
    bbox_local: bool = True   # production EXE default; see field doc
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

    # --- Mode dispatch (#85 #86) ---------------------------------------
    # mode == "fresh"        → generate shapes from image (the default,
    #                          unchanged historic behavior)
    # mode == "polish_only"  → load shapes from input_shapes_path + run
    #                          joint_polish() on them with the supplied
    #                          image as target. num_shapes, max_resolution,
    #                          random_samples are not required in this
    #                          mode (canvas size comes from the loaded
    #                          doc.image_size).
    mode: str = "fresh"
    input_shapes_path: Path | None = None
    polish_steps_override: int | None = None

    # --- Resume support (#snapshot-resume) -----------------------------
    # When set in fresh mode: runner loads the partial doc, replays its
    # shapes onto the canvas, and continues the greedy loop from
    # len(seeded) → num_shapes. polish_only + seed = ValueError (resume
    # is fresh-only).
    seed_shapes_path: Path | None = None

    # --- Resume rescale support (#vram-honesty Task 6/7) ---------------
    # (width, height) of the canvas the seed_shapes_path doc was
    # generated on. When the resume run picks a SMALLER canvas (because
    # the user dropped max_resolution to fit headroom), engine.run_gpu
    # rescales the seeded coords from this size to the current canvas
    # before continuing. None = no rescale (snapshot ran at the same
    # canvas dims as the current run, or no resume at all).
    # JSON-roundtrip-safe: serializes to list, coerced back to tuple in
    # from_dict.
    seed_canvas_size: tuple[int, int] | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "RunConfig":
        """Parse + validate. Raises ValueError on missing required fields,
        wrong types, or out-of-range values. Conditional requirements
        depend on `mode`:

          mode='fresh' (default)  — image_path, output_json_path,
              num_shapes, max_resolution, random_samples are all required.
          mode='polish_only'       — image_path, output_json_path,
              input_shapes_path are required. The shape-budget fields are
              ignored (canvas dims come from the loaded JSON's image_size).
        """
        mode = str(d.get("mode", "fresh"))
        if mode not in ("fresh", "polish_only"):
            raise ValueError(
                f"unknown mode {mode!r} — must be 'fresh' or 'polish_only'"
            )

        # Always-required fields, regardless of mode.
        try:
            image_path = Path(d["image_path"])
            output_json_path = Path(d["output_json_path"])
        except KeyError as exc:
            raise ValueError(f"missing required config field: {exc}") from exc

        if mode == "polish_only":
            # polish_only fields.
            isp = d.get("input_shapes_path")
            if not isp:
                raise ValueError(
                    "input_shapes_path is required when mode='polish_only'"
                )
            input_shapes_path = Path(isp)
            if not input_shapes_path.is_file():
                raise ValueError(
                    f"input_shapes_path not found: {input_shapes_path}"
                )
            polish_steps_override = d.get("polish_steps_override")
            if polish_steps_override is not None:
                polish_steps_override = int(polish_steps_override)
                if polish_steps_override < 1:
                    raise ValueError(
                        f"polish_steps_override must be >= 1, "
                        f"got {polish_steps_override}"
                    )
            # Placeholders — polish_only never reads these (canvas dims
            # come from the loaded JSON's image_size). Kept as 0 (not None)
            # so the int field types stay non-optional.
            num_shapes = 0
            max_resolution = 0
            random_samples = 0
        else:
            # Fresh-mode required fields.
            try:
                num_shapes = int(d["num_shapes"])
                max_resolution = int(d["max_resolution"])
                random_samples = int(d["random_samples"])
            except KeyError as exc:
                raise ValueError(
                    f"missing required config field: {exc}"
                ) from exc
            if num_shapes < 1:
                raise ValueError(
                    f"num_shapes must be >= 1, got {num_shapes}"
                )
            if max_resolution < 64:
                raise ValueError(
                    f"max_resolution must be >= 64, got {max_resolution}"
                )
            if random_samples < 1:
                raise ValueError(
                    f"random_samples must be >= 1, got {random_samples}"
                )
            input_shapes_path = None
            polish_steps_override = None

        # seed_shapes_path: fresh-mode resume. Existence-check + mode
        # gate. Polish + seed combined is meaningless (polish replays
        # input_shapes_path; can't ALSO seed from somewhere else).
        ssp = d.get("seed_shapes_path")
        if ssp:
            if mode == "polish_only":
                raise ValueError(
                    "seed_shapes_path is not supported in polish_only "
                    "mode (resume is fresh-only)"
                )
            seed_shapes_path = Path(ssp)
            if not seed_shapes_path.is_file():
                raise ValueError(
                    f"seed_shapes_path not found: {seed_shapes_path}"
                )
        else:
            seed_shapes_path = None

        # seed_canvas_size: optional (w, h) tuple recording the canvas
        # dims of the seed_shapes_path doc. engine.run_gpu rescales
        # seeded coords when this differs from the current canvas. JSON
        # roundtrips tuples as lists, so coerce. Validate shape + types
        # so a malformed value never reaches the engine.
        scs = d.get("seed_canvas_size")
        if scs is None:
            seed_canvas_size = None
        else:
            if not isinstance(scs, (list, tuple)) or len(scs) != 2:
                raise ValueError(
                    f"seed_canvas_size must be a (width, height) "
                    f"pair, got {scs!r}"
                )
            try:
                sw, sh = int(scs[0]), int(scs[1])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"seed_canvas_size entries must be int-coercible, "
                    f"got {scs!r}"
                ) from exc
            if sw < 1 or sh < 1:
                raise ValueError(
                    f"seed_canvas_size dims must be >= 1, got "
                    f"({sw}, {sh})"
                )
            seed_canvas_size = (sw, sh)

        # Optional with type coercion (same across modes).
        lock_alpha = bool(d.get("lock_alpha", True))
        if not lock_alpha:
            raise ValueError(
                "lock_alpha=False is not a supported value — the Forza "
                "injector writes alpha=255 at inject time and any non-opaque "
                "JSON breaks preview/in-game parity."
            )
        device = str(d.get("device", "cuda"))
        if device not in ("cuda", "cpu"):
            raise ValueError(f"device must be 'cuda' or 'cpu', got {device!r}")
        ce = int(d.get("checkpoint_every", 0))
        if device == "cuda" and 0 < ce < 100:
            raise ValueError(
                f"checkpoint_every must be >= 100 on cuda (got {ce}); "
                f"set 0 to disable snapshots entirely"
            )
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
            bbox_local=bool(d.get("bbox_local", True)),
            joint_polish_steps=int(d.get("joint_polish_steps", 0)),
            vram_budget_gib=float(d.get("vram_budget_gib", 0.0)),
            lock_alpha=lock_alpha,
            progress_every=int(d.get("progress_every", 0)),
            checkpoint_every=ce,
            device=device,
            preset_label=str(d.get("preset_label", "")),
            mode=mode,
            input_shapes_path=input_shapes_path,
            polish_steps_override=polish_steps_override,
            seed_shapes_path=seed_shapes_path,
            seed_canvas_size=seed_canvas_size,
        )

    def summary(self) -> dict:
        """Echoed back in the 'started' event so the EXE's progress label
        can show e.g., 'GPU 700 shapes @ 600px (preset: Headshot 700)'.
        Drops file paths (the EXE already knows them) + redundant defaults."""
        return {
            "mode": self.mode,
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


def _run_polish_only(cfg: RunConfig, stream, logger) -> int:
    """Polish branch for mode='polish_only'. Loads the source image +
    the existing shapes JSON, builds the joint_polish target (matching
    engine.py's construction at lines 460-570 of engine.py), runs the
    polish optimizer with freeze_geometry=True, saves the refined
    shapes via the canonical save_json path.

    All non-ellipse shapes are rejected here so the user gets a clean
    error before any GPU work starts (joint_polish only handles
    rotated_ellipse — engine.py:697 has the matching gate).
    """
    # Import inside the function so the import-error exit code (3) still
    # applies if torch isn't installed.
    try:
        with logger.start_phase("import_engine"):
            import torch
            import numpy as np
            from forza_abyss_painter.shapegen.gpu.joint_polish import joint_polish
            from forza_abyss_painter.shapegen.gpu.engine import (
                _posterize, _edge_weight_map, DTYPE, get_device,
            )
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

    # Load shapes JSON first — canvas dims come from doc.image_size, NOT
    # from cfg.max_resolution. This is the key polish_only invariant:
    # polish operates on the canvas the shapes were generated for.
    try:
        with logger.start_phase("load_shapes_json",
                                  shapes_path=str(cfg.input_shapes_path)):
            from forza_abyss_painter.io.exporter import load_json, save_json
            from forza_abyss_painter.io.json_schema import FD6Document
            doc = load_json(str(cfg.input_shapes_path))
            shapes_json = list(doc.shapes)
            logger.log("shapes_loaded",
                       shape_count=len(shapes_json),
                       image_size=list(doc.image_size))
    except (OSError, ValueError, KeyError) as exc:
        emit(stream, {
            "kind": "error", "stage": "load_shapes_json",
            "message": f"{type(exc).__name__}: {exc}",
        })
        return 1

    # Polish only supports rotated_ellipse (joint_polish builds a
    # (N, 5) geometry tensor expecting x/y/rx/ry/angle). Reject other
    # shape types early.
    if not shapes_json:
        emit(stream, {
            "kind": "error", "stage": "polish_only_empty_input",
            "message": (
                f"input_shapes_path {cfg.input_shapes_path} has zero "
                f"shapes — nothing to polish."
            ),
        })
        return 1
    non_ellipse = [s for s in shapes_json if s.get("type") != "rotated_ellipse"]
    if non_ellipse:
        kinds = sorted({s.get("type", "?") for s in non_ellipse})
        emit(stream, {
            "kind": "error", "stage": "polish_only_unsupported_shape",
            "message": (
                f"polish_only supports rotated_ellipse only; found "
                f"{len(non_ellipse)} non-ellipse shape(s) of type(s) "
                f"{kinds} in {cfg.input_shapes_path.name}. Polish "
                f"skipped."
            ),
        })
        return 1

    if doc.image_size[0] <= 0 or doc.image_size[1] <= 0:
        emit(stream, {
            "kind": "error", "stage": "polish_only_invalid_canvas",
            "message": (
                f"loaded JSON has image_size={doc.image_size}; polish "
                f"requires a positive canvas size."
            ),
        })
        return 1
    canvas_w, canvas_h = int(doc.image_size[0]), int(doc.image_size[1])

    # Load + resize source image to the loaded JSON's canvas.
    try:
        with logger.start_phase("load_image", image_path=str(cfg.image_path)):
            from PIL import Image
            img = Image.open(cfg.image_path)
            sticker = cfg.sticker_mode or bool(getattr(doc, "sticker_mode", False))
            if sticker:
                rgba = img.convert("RGBA").resize(
                    (canvas_w, canvas_h), Image.LANCZOS)
                arr = np.asarray(rgba, dtype=np.uint8)
                rgb = arr[:, :, :3].copy()
                alpha_mask = arr[:, :, 3].copy()
            else:
                rgb = np.asarray(
                    img.convert("RGB").resize((canvas_w, canvas_h), Image.LANCZOS),
                    dtype=np.uint8,
                )
                alpha_mask = None
            logger.log("image_loaded_for_polish",
                       canvas_size=[canvas_w, canvas_h],
                       has_alpha=alpha_mask is not None)
    except (OSError, ValueError) as exc:
        emit(stream, {
            "kind": "error", "stage": "load_image",
            "message": f"{type(exc).__name__}: {exc}",
        })
        return 1

    # Build polish inputs mirroring engine.py's construction. Posterize +
    # alpha substrate fill happen the same way; the only difference is we
    # start from the FINAL canvas (rendered from the loaded shapes) so
    # joint_polish sees the same baseline the engine would have at this
    # point in a fresh run.
    try:
        with logger.start_phase("build_polish_inputs"):
            device = get_device() if cfg.device == "cuda" else "cpu"
            if cfg.posterize_levels:
                rgb = _posterize(rgb, cfg.posterize_levels)
            if alpha_mask is not None:
                opaque_mask3 = (alpha_mask > 0)[:, :, None].astype(np.uint8)
                substrate = np.full_like(rgb, 40)
                target_np = np.where(opaque_mask3 > 0, rgb, substrate)
            else:
                target_np = rgb
            target = torch.from_numpy(target_np).to(device)
            alpha_t = None
            alpha_mask_f = None
            if alpha_mask is not None:
                alpha_t = torch.from_numpy(alpha_mask).to(device)
                alpha_mask_f = alpha_t.to(DTYPE) / 255.0
            edge_weight = (_edge_weight_map(target, cfg.edge_strength)
                            if cfg.edge_strength > 0 else None)
            # canvas_init is the mean of the target (or substrate-40 for
            # sticker) — same construction as engine.py:566-570.
            if alpha_mask is not None:
                canvas_init = torch.full(
                    (canvas_h, canvas_w, 3), 40,
                    dtype=torch.uint8, device=device,
                )
            else:
                mean = target.to(DTYPE).reshape(-1, 3).mean(dim=0).round() \
                    .clamp(0, 255).to(torch.uint8)
                canvas_init = mean.view(1, 1, 3) \
                    .expand(canvas_h, canvas_w, 3).contiguous().clone()
    except Exception as exc:  # pragma: no cover — defensive
        logger.log_exception("build_polish_inputs_unexpected", exc)
        emit(stream, {
            "kind": "error", "stage": "build_polish_inputs",
            "message": f"{type(exc).__name__}: {exc}",
        })
        return 1

    steps = cfg.polish_steps_override if cfg.polish_steps_override is not None else 150
    emit(stream, {"kind": "progress", "shape_count": 0, "total": steps})

    try:
        with logger.start_phase("joint_polish",
                                  steps=steps,
                                  shape_count=len(shapes_json)):
            refined, _canvas_np = joint_polish(
                shapes_json, target, alpha_t, alpha_mask_f, edge_weight,
                canvas_init, canvas_h, canvas_w, steps,
                lock_alpha=cfg.lock_alpha,
                purity_penalty=0.0,
                freeze_geometry=True,
            )
            logger.log("joint_polish_done", refined_count=len(refined))
    except RuntimeError as exc:
        # joint_polish + OOM are converted to RuntimeError upstream;
        # surface verbatim like the fresh path, and append restart
        # guidance — polish_only also runs on GPU and can OOM on a
        # fragmented cache after a prior fresh-gen failure.
        emit(stream, {
            "kind": "error", "stage": "joint_polish",
            "message": (
                f"{type(exc).__name__}: {exc}\n\n"
                f"If this is a CUDA OOM, the GPU's cache may still hold "
                f"the failed allocation. Close FH6 if running, restart "
                f"the EXE to release the CUDA context, and try a "
                f"smaller preset (e.g. Medium 1000 instead of Hi-Res 3000)."
            ),
        })
        return 1
    except Exception as exc:  # pragma: no cover — defensive
        logger.log_exception("joint_polish_unexpected", exc)
        emit(stream, {
            "kind": "error", "stage": "joint_polish",
            "message": f"{type(exc).__name__}: {exc}",
        })
        return 1

    # Save via the canonical exporter so the validator hook (#100) runs.
    try:
        with logger.start_phase("save_json",
                                  output_path=str(cfg.output_json_path)):
            polished_doc = FD6Document(
                source_image=cfg.image_path.name,
                image_size=(canvas_w, canvas_h),
                shape_count=len(refined),
                generated_at=doc.generated_at,
                profile=f"{doc.profile} (polished)" if doc.profile else "polished",
                sticker_mode=sticker,
                shapes=refined,
            )
            save_json(polished_doc, cfg.output_json_path)
    except (OSError, ValueError, KeyError) as exc:
        emit(stream, {
            "kind": "error", "stage": "save_json",
            "message": f"{type(exc).__name__}: {exc}",
        })
        return 1

    logger.log("polish_only_done",
               output_path=str(cfg.output_json_path),
               shape_count=len(refined))
    emit(stream, {
        "kind": "done",
        "output_path": str(cfg.output_json_path),
        "shape_count": len(refined),
    })
    return 0


def _build_run_config_block(cfg: "RunConfig") -> dict:
    """Build the ``_run_config`` metadata block embedded in snapshot
    and final-output JSONs.  Carries the original run params so resume
    can replay with the same config.

    DRY: snapshot and final-output writers both call this — keeps the
    two sites in lockstep when fields are added/removed.
    """
    return {
        "target_shape_count": int(cfg.num_shapes),
        "random_samples": int(cfg.random_samples),
        "max_resolution": int(cfg.max_resolution),
        "edge_strength": float(cfg.edge_strength),
        "posterize_levels": int(cfg.posterize_levels),
        "sticker_mode": bool(cfg.sticker_mode),
        "lock_alpha": bool(cfg.lock_alpha),
        "bbox_local": bool(cfg.bbox_local),
        "joint_polish_steps": int(cfg.joint_polish_steps),
        "vram_budget_gib": float(cfg.vram_budget_gib),
        "preset_label": str(cfg.preset_label),
    }


def _write_snapshot(
    cfg: "RunConfig",
    shapes_list: list,
    count: int,
    image_w: int,
    image_h: int,
) -> "Path":
    """Save a partial snapshot to <output_stem>_<count>.json next to the
    final output. Embeds _run_config breadcrumb for resume.

    Writes the dict directly (not via save_json) because save_json
    expects an FD6Document dataclass that has no `_run_config` field.
    Atomic write: temp file + os.replace().
    """
    import json as _json
    import os as _os
    import time as _time
    from forza_abyss_painter.io.snapshots import snapshot_path_for

    snap_path = snapshot_path_for(cfg.output_json_path, count)
    tmp_path = snap_path.with_suffix(snap_path.suffix + ".tmp")

    doc = {
        "format": "fd6.shapes",
        "version": 1,
        "source_image": cfg.image_path.name,
        "image_size": [image_w, image_h],
        "shape_count": count,
        "generated_at": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
        "profile": cfg.preset_label,
        "sticker_mode": bool(cfg.sticker_mode),
        "shapes": list(shapes_list),
        "_run_config": _build_run_config_block(cfg),
    }
    tmp_path.write_text(_json.dumps(doc), encoding="utf-8")
    _os.replace(tmp_path, snap_path)
    return snap_path


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

    # Mode dispatch (#85 #86). polish_only branches off here; fresh
    # continues with the historic path below.
    if cfg.mode == "polish_only":
        return _run_polish_only(cfg, stream, logger)

    # Resume support: validate seed_shapes_path BEFORE importing torch so
    # a bad snapshot (missing file, corrupt JSON, non-ellipse shapes) fails
    # fast without downloading/importing the GPU runtime. load_json has no
    # torch dependency. polish_only already returned above, so this is
    # fresh-only by construction.
    seed_shapes: list[dict] | None = None
    if cfg.seed_shapes_path is not None:
        try:
            with logger.start_phase("load_seed_shapes",
                                    path=str(cfg.seed_shapes_path)):
                from forza_abyss_painter.io.exporter import load_json
                seed_doc = load_json(str(cfg.seed_shapes_path))
                seed_shapes = list(seed_doc.shapes)
        except (OSError, ValueError, KeyError) as exc:
            emit(stream, {
                "kind": "error", "stage": "load_seed_shapes",
                "message": f"{type(exc).__name__}: {exc}",
            })
            return 1
        if not seed_shapes:
            emit(stream, {
                "kind": "error", "stage": "resume_empty_seed",
                "message": f"seed_shapes_path "
                           f"{cfg.seed_shapes_path} has zero shapes",
            })
            return 1
        non_ell = [s for s in seed_shapes
                   if s.get("type") != "rotated_ellipse"]
        if non_ell:
            kinds = sorted({s.get("type", "?") for s in non_ell})
            emit(stream, {
                "kind": "error", "stage": "resume_unsupported_shape",
                "message": (
                    f"resume supports rotated_ellipse only; found "
                    f"{len(non_ell)} non-ellipse shape(s) of "
                    f"type(s) {kinds} in "
                    f"{cfg.seed_shapes_path.name}"
                ),
            })
            return 1
        logger.log("seed_loaded", count=len(seed_shapes))

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
        # Always emit the lightweight checkpoint event (progress bar).
        emit(stream, {
            "kind": "checkpoint",
            "shape_count": shape_idx,
            "total": cfg.num_shapes,
        })
        # Best-effort snapshot write + snapshot event. Disk failures
        # don't crash the run — the snapshot is a safety net, not a
        # hard requirement.
        try:
            snap_path = _write_snapshot(
                cfg, shapes_so_far, shape_idx,
                image_w=rgb.shape[1], image_h=rgb.shape[0],
            )
        except OSError as exc:
            logger.log_exception("snapshot_write_failed", exc)
            return
        emit(stream, {
            "kind": "snapshot",
            "shape_count": shape_idx,
            "total": cfg.num_shapes,
            "path": str(snap_path),
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
                seed_shapes=seed_shapes,
                seed_canvas_size=cfg.seed_canvas_size,
            )
            logger.log("run_gpu_done", shapes_count=len(shapes_list))
    except RuntimeError as exc:
        # run_gpu's OOM wrapper converts torch.cuda.OutOfMemoryError into
        # a RuntimeError with an actionable recipe — pass that through
        # AND append restart guidance so users know how to recover.
        # After an OOM the CUDA cache may still hold the failed
        # allocation; subsequent runs hit the same OOM until the EXE
        # restarts.
        emit(stream, {
            "kind": "error", "stage": "engine_run",
            "message": (
                f"{type(exc).__name__}: {exc}\n\n"
                f"If this is a CUDA OOM, the GPU's cache may still hold "
                f"the failed allocation. Close FH6 if running, restart "
                f"the EXE to release the CUDA context, and try a "
                f"smaller preset (e.g. Medium 1000 instead of Hi-Res 3000)."
            ),
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

    # Embed _run_config in the FINAL output too (mirrors snapshots).
    # save_json was the canonical schema-checked path; for the
    # metadata we read+inject+rewrite — small file, negligible cost.
    try:
        import json as _json
        doc_dict = _json.loads(
            cfg.output_json_path.read_text(encoding="utf-8")
        )
        doc_dict["_run_config"] = _build_run_config_block(cfg)
        cfg.output_json_path.write_text(
            _json.dumps(doc_dict), encoding="utf-8",
        )
    except OSError as exc:
        logger.log_exception("final_run_config_inject_failed", exc)
        # Don't fail the whole run for a metadata-injection error —
        # the JSON itself is valid + saved. Just log + continue.

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
