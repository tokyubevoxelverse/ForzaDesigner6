"""Builds notebooks/fap_gpu_colab.ipynb from inline cell sources.

Re-run after edits to engine.py / scoring.py / rasterize.py to refresh the notebook's
copy of the Forza Abyss Painter GPU implementation.

Usage:
    python notebooks/build_colab_notebook.py
"""
from __future__ import annotations

import json
from pathlib import Path
from forza_abyss_painter.shapegen.presets import PRESETS

ROOT = Path(__file__).resolve().parent.parent


def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


def md(src: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": src.splitlines(keepends=True),
    }


# GitHub raw URL for the logo. Works once the repo is public; loads with zero
# notebook-size overhead (vs base64-embedding which would add ~470 KB per
# notebook × 6 = ~3 MB). The `?raw=true` form is canonical for GitHub-hosted
# binary assets and is what Colab markdown renders correctly.
_LOGO_RAW_URL = (
    "https://raw.githubusercontent.com/whykusanagi/"
    "forza-abyss-painter/main/assets/forza_abyss_painter_logo.png"
)
_REPO_URL = "https://github.com/whykusanagi/forza-abyss-painter"

CELL_INTRO = f"""<div align="center" style="background: linear-gradient(135deg, #0a0a0a 0%, #1a0a1f 100%); padding: 32px 24px; border-radius: 8px; border: 1px solid #3a2555; margin-bottom: 16px;">
  <img src="{_LOGO_RAW_URL}" alt="Forza Abyss Painter" width="180" style="margin-bottom: 16px;"/>
  <h1 style="color: #d94f90; margin: 0; font-family: -apple-system, BlinkMacSystemFont, sans-serif; font-weight: 700; letter-spacing: 0.5px;">Forza Abyss Painter</h1>
  <p style="color: #a78bfa; margin: 8px 0 0 0; font-size: 14px; font-family: -apple-system, BlinkMacSystemFont, monospace;">GPU Shape-Gen · Colab Edition</p>
  <p style="color: #6a6a6a; margin: 4px 0 0 0; font-size: 12px;"><a href="{_REPO_URL}" style="color: #8b5cf6; text-decoration: none;">github.com/whykusanagi/forza-abyss-painter</a></p>
</div>

This notebook runs the Forza Abyss Painter GPU shape-generator on CUDA.
It mirrors the desktop CLI (`forza_abyss_painter.cli.oneshot`) but is self-contained — no clone, no install
beyond the pre-installed Colab stack.

**Before running:** switch the runtime to a GPU.
> Runtime → Change runtime type → Hardware accelerator: **GPU** (T4 is fine; V100/A100 is faster)

**Workflow:**
1. Run the *Setup* cell (verifies CUDA, defines the engine).
2. Run the *Upload* cell and pick a PNG/JPG. For stickers (RGBA with transparency you want
   preserved), set `STICKER_MODE=True` in the *Run* cell.
3. Run the *Run* cell — it generates shapes and writes `<stem>_<NUM_SHAPES>.json`
   + `<stem>_<NUM_SHAPES>_render.png` so you can tell at a glance which budget tier this
   JSON belongs to (eg `my_image_3000.json` = high-detail vs `my_image_400.json` = lineart).
4. The result is displayed inline. Download the JSON to ship to your Windows FH6 injector.

**If you hit a CUDA OOM:** run the **Cleanup** cell (Section 3) and lower `RANDOM_SAMPLES`
or `MUTATIONS_PER_ROUND`. Memory cost scales as `K * H * W * 12 bytes` per intermediate,
with 2-3 intermediates live. Restart runtime if the cleanup cell can't get allocation under
~10 GB.
"""

# Footer cell appended to every notebook — small credit banner with the
# corrupted-theme colors so the visual identity bookends the workflow.
CELL_FOOTER = f"""<div align="center" style="margin-top: 32px; padding: 16px; border-top: 1px solid #3a2555; color: #6a6a6a; font-size: 12px;">
  <p style="margin: 0;">made with <strong style="color: #d94f90;">Forza Abyss Painter</strong></p>
  <p style="margin: 4px 0 0 0;"><a href="{_REPO_URL}" style="color: #8b5cf6; text-decoration: none;">github.com/whykusanagi/forza-abyss-painter</a> · MIT</p>
</div>
"""

CELL_SETUP_DEPS = """# --- Environment check ---
import sys, time
import numpy as np
import torch
from PIL import Image

print(f"Python:  {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA:    available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU:     {name}  ({vram_gb:.1f} GB VRAM)")
else:
    print("WARNING: No CUDA. Runtime > Change runtime type > Hardware accelerator: GPU")

DTYPE = torch.float32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""

# The engine: rasterize + score + run_gpu, all inlined into one Colab cell.
# Source is kept in sync with forza_abyss_painter/shapegen/gpu/{rasterize,scoring,engine}.py.
# CUDA-first instead of MPS-first; otherwise the math is identical.
GPU_PKG = ROOT / "forza_abyss_painter" / "shapegen" / "gpu"
_PROVIDED_IMPORTS = {
    "import torch", "import time", "import numpy as np",
}


def _strip_module(path: Path) -> str:
    """Return a module's source with cross-package and already-provided imports removed."""
    out = []
    for ln in path.read_text().splitlines():
        s = ln.strip()
        if s.startswith("from __future__"):
            continue
        if s.startswith("from forza_abyss_painter.shapegen.gpu"):
            continue
        if s.startswith("from dataclasses import") or s.startswith("from typing import"):
            continue
        if s in _PROVIDED_IMPORTS:
            continue
        out.append(ln)
    return "\n".join(out).strip()


def _inline_package_engine() -> str:
    """Build the engine setup cell by inlining the package GPU modules verbatim. This is
    THE anti-drift mechanism: the notebook always reflects forza_abyss_painter/shapegen/gpu/ exactly."""
    preamble = (
        "# === Forza Abyss Painter GPU engine — inlined verbatim from forza_abyss_painter/shapegen/gpu/ ===\n"
        "# Regenerated by notebooks/build_colab_notebook.py. DO NOT edit here; edit the\n"
        "# package modules and rebuild so the notebook and CLI never drift apart.\n"
        "from __future__ import annotations\n"
        "from dataclasses import dataclass, field, replace\n"
        "from typing import Callable\n"
        "import time\n"
        "import numpy as np\n"
        "import torch\n"
    )
    parts = [preamble]
    # Dependency order: engine imports all the others; bbox_score needs scoring; joint_polish
    # needs shapes_gpu/rasterize/scoring; refill needs joint_polish + bbox_score + shapes_gpu
    # + rasterize + scoring + device, AND engine.run_gpu calls refill.clean_and_refill. Keep
    # this list complete or the notebook NameErrors at runtime — exactly what task #74 was
    # logged for (`name 'clean_and_refill' is not defined` after a 10-minute greedy run).
    # Inline order matters: each module's source is concatenated raw, so anything `engine`
    # references must be inlined before `engine`.
    for name in ["device", "rasterize", "scoring", "shapes_gpu", "bbox_score",
                 "joint_polish", "refill", "engine"]:
        parts.append(f"# ----- {name}.py -----\n" + _strip_module(GPU_PKG / f"{name}.py"))
    parts.append('DEVICE = get_device()\nprint(f"Engine loaded on {DEVICE}. Ready for image upload.")')
    return "\n\n".join(parts)


CELL_SETUP_ENGINE = _inline_package_engine()

CELL_UPLOAD = '''# --- Upload an image ---
from google.colab import files
import io

print("Pick a PNG or JPG to convert into Forza vinyl shapes.")
_uploaded = files.upload()
if not _uploaded:
    raise SystemExit("No file uploaded.")

# Take the first uploaded file
_first_name, _first_bytes = next(iter(_uploaded.items()))
SOURCE_IMAGE_NAME = _first_name
SOURCE_IMAGE_BYTES = _first_bytes
_img_preview = Image.open(io.BytesIO(SOURCE_IMAGE_BYTES))
_sw, _sh = _img_preview.size
print(f"Loaded {SOURCE_IMAGE_NAME}: {_sw}x{_sh}, mode={_img_preview.mode}")
print("Set your knobs next, then the Resolution Planner will help you choose MAX_RESOLUTION.")
_img_preview
'''

_KNOBS_TMPL = '''# --- Knobs ({preset_label}) ---
# Preset tuned for this shape budget. Everything is editable; the Resolution Planner below
# validates MAX_RESOLUTION against your GPU before you launch.
REFINE_MODE        = "gradient"   # "gradient" (Adam refine) | "hillclimb" (legacy mutations)

# Candidate shape types. The FH6 injector currently renders only ellipses — triangle/
# rotated_rectangle render in PREVIEW but won't inject today (dev roadmap). Multi-shape is
# gradient-only, falls back to full-canvas scoring (lower RANDOM_SAMPLES), and skips
# joint-polish. The `shapes_*` presets enable all three for evaluating the quality ceiling.
SHAPE_TYPES = {shape_types}

# Per-shape opacity search (crisp black lines / dark detail vs smooth gradients). The chosen
# opacity is written into each shape's JSON color. Fewer levels = faster.
ALPHA_LEVELS = [96, 160, 255]

# Edge-weighted loss: prioritize high-contrast detail (eyes, lineart) over flat regions.
# 0 = off; useful range 1-4 (higher over-focuses edges).
EDGE_STRENGTH = {edge_strength}

# Posterize the target to N color levels before fitting (0 = off). Flattens gradients into
# bands → cleaner fills, crisper edges (the geometrize trick). 16-32 is a good range.
POSTERIZE_LEVELS = {posterize_levels}

# bbox-local scoring: score each random candidate over its bounding-box crop, not the full
# canvas (~10x+ cheaper) — this is what makes high RANDOM_SAMPLES affordable.
BBOX_LOCAL = True

NUM_SHAPES         = {num_shapes}   # FH6 vinyl-group limit is ~3000
RANDOM_SAMPLES     = {random_samples}  # seed candidates/shape. forza-painter's #1 fix for
                            # "blurry" is RAISING this (they use 50k-350k). bbox-local makes
                            # it affordable; push higher to chase parity.
MAX_RESOLUTION     = {max_resolution}   # matched to NUM_SHAPES (~sqrt(N)*28 px long side).
                            # Higher wastes compute on detail the shape budget can't render.
UPLOAD_MAX_LONG_SIDE = 720  # HARD SAFETY CAP on input image's long side. Applied at upload
                            # time, BEFORE the engine sees the image. Effective resize ceiling
                            # = min(MAX_RESOLUTION, UPLOAD_MAX_LONG_SIDE). Even on a 102 GB
                            # Blackwell 6000 we hit OOM at native resolutions >1600 because
                            # the bbox-local scorer's per-candidate crop area scales with
                            # max(rx,ry)² ~= (canvas_long_side/8)² — 1600² is 4× the cost
                            # of 720² and pushes K=6144 candidates past the VRAM budget.
                            # 720 keeps every preset comfortably under 20 GB peak. Raise
                            # ONLY if you have measured your GPU's headroom and accept OOM risk.
SEED               = 42     # 0 = time-based
STICKER_MODE       = True   # True if your image has transparency to preserve
ALPHA_THRESHOLD    = {alpha_threshold}      # 0 = keep soft alpha. If you see a WHITE HALO around the figure
                            # (common on stickers exported over a white background — the
                            # anti-aliased edge carries white RGB), set 128 to binarize the
                            # silhouette and drop the fringe. Raise to 160-200 if it persists.
CHECKPOINT_EVERY   = {checkpoint_every}    # write a recoverable JSON to Drive every N shapes
                            # (survives a mid-run disconnect). 0 = off.
# NOTE: lock_alpha is NOT exposed as a user knob — it's a HARD SYSTEM CONSTRAINT.
# The Forza injector writes alpha=255 unconditionally; soft-alpha JSONs would render
# one way in this notebook's preview and another way in-game. run_gpu raises ValueError
# if cfg.lock_alpha is False. Hardcoded True below in the cfg instantiation.
POLISH_FREEZE_GEOMETRY = {polish_freeze_geometry}   # True (PRODUCTION DEFAULT): joint_polish refines
                                     # colors only, leaving shape geometry bit-identical to greedy.
                                     # Prevents the inflate/collapse exploits Adam was finding on
                                     # sparkle content (verified at 1000 shapes: 1000/1000 geom
                                     # frozen, engine↔upstream parity 0.000). Set False for the
                                     # legacy joint-geom-and-color polish (escape hatch for future
                                     # structured-refinement experiments).

# --- Gradient-refinement knobs (cheap on memory) ---
GRAD_STARTS        = 16     # multi-start diversity → escapes poor local optima
GRAD_STEPS         = 50     # per-shape convergence depth
GRAD_LR            = 2.0    # Adam learning rate

# Joint-polish: after greedy, gradient-optimize ALL shapes at once vs the full target —
# breaks greedy myopia (early shapes get un-stuck). Fewer shapes benefit from MORE steps
# (each shape carries more responsibility). Watch the printed loss; stop raising when it
# plateaus. 0 = off.
JOINT_POLISH_STEPS = {joint_polish_steps}

# --- Hill-climb knobs (only used if REFINE_MODE == "hillclimb") ---
MUTATION_ROUNDS    = 16
MUTATIONS_PER_ROUND = 64

print(f"knobs set: shapes={{NUM_SHAPES}}, samples={{RANDOM_SAMPLES}}, max_res={{MAX_RESOLUTION}}, "
      f"edge={{EDGE_STRENGTH}}, joint_polish={{JOINT_POLISH_STEPS}}")
print("Next: run the Resolution Planner to validate MAX_RESOLUTION against your VRAM.")
'''


def cell_knobs(preset):
    p = dict(preset)
    p.setdefault("lock_alpha", True)    # injector-safety default — the injector forces binary
                                        # alpha at write time, so a False default would produce
                                        # JSONs whose engine PNG renders differently than the
                                        # game does. NEVER set False without explicit reason.
    p.setdefault("polish_freeze_geometry", True)    # production default; multi-shape eval presets
                                                     # don't run polish so it's moot for them either way
    return _KNOBS_TMPL.format(**p)


CELL_POSTERIZE_PREVIEW = '''# --- Posterize preview: pick POSTERIZE_LEVELS for THIS image ---
# Posterize flattens color gradients into bands → cleaner fills + crisper edges, BUT too few
# levels washes out faint detail: subtle color shifts smaller than the quantization step
# vanish (this is what made the neck tattoos disappear). The panels below show 4 levels;
# the ★ recommendation is the lowest level whose step is fine enough to preserve THIS image's
# fine detail. Eyeball them, then set POSTERIZE_LEVELS in the Configure cell.
import io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter

_LEVELS = [16, 24, 32, 48]

_im = Image.open(io.BytesIO(SOURCE_IMAGE_BYTES))
_im = _im.convert("RGB") if _im.mode not in ("RGB",) else _im
_w0, _h0 = _im.size
_s = 512 / max(_w0, _h0)
if _s < 1:
    _im = _im.resize((max(1, int(_w0 * _s)), max(1, int(_h0 * _s))))
_rgb = np.asarray(_im)

def _posterize_np(a, L):
    if L < 2 or L >= 256:
        return a
    f = a.astype(np.float32) / 255.0
    return (np.round(f * (L - 1)) / (L - 1) * 255.0).round().clip(0, 255).astype(np.uint8)

# Fine-detail amplitude: typical high-pass magnitude where the image carries detail. The
# posterize step (255/(L-1)) must be <= this or that detail gets quantized away.
_blur = np.asarray(_im.filter(ImageFilter.BoxBlur(2))).astype(np.float32)
_hp = np.abs(_rgb.astype(np.float32) - _blur).max(axis=2)
_det = _hp[_hp > 1.0]
_amp = float(np.percentile(_det, 40)) if _det.size > 50 else 24.0
_need = 255.0 / max(_amp, 1e-3) + 1.0
_rec = _LEVELS[-1]
for _L in _LEVELS:
    if _L >= _need:
        _rec = _L
        break

_fig, _ax = plt.subplots(1, len(_LEVELS) + 1, figsize=(3.2 * (len(_LEVELS) + 1), 3.4))
_ax[0].imshow(_rgb); _ax[0].set_title("original"); _ax[0].axis("off")
for _i, _L in enumerate(_LEVELS):
    _ax[_i + 1].imshow(_posterize_np(_rgb, _L))
    _ax[_i + 1].set_title(f"posterize {_L}" + ("  \\u2605REC" if _L == _rec else ""))
    _ax[_i + 1].axis("off")
plt.tight_layout(); plt.show()

print(f"Recommended POSTERIZE_LEVELS = {_rec}")
print(f"  (fine-detail amplitude ~{_amp:.0f}/255; levels below ~{_need:.0f} start washing out")
print(f"   detail that small. Lower = cleaner banding; higher = preserve faint colored detail")
print(f"   like tattoos. 0 disables posterize entirely.)")
print("Set POSTERIZE_LEVELS in the Configure cell to your pick — the \\u2605 is a starting point.")
'''

CELL_RESOLUTION_PLANNER = '''# --- Resolution Planner (decide MAX_RESOLUTION here) ---
# Your sources are high-res PSD exports. Downscaling to fit VRAM throws away detail, so
# pick the LARGEST MAX_RESOLUTION that fits comfortably. This cell shows the tradeoff and
# HARD-STOPS if your current MAX_RESOLUTION would OOM — so you never launch a doomed run.
import torch

_sw, _sh = Image.open(io.BytesIO(SOURCE_IMAGE_BYTES)).size
_long = max(_sw, _sh)
if torch.cuda.is_available():
    _props = torch.cuda.get_device_properties(0)
    _vram_total = _props.total_memory / 1e9
    _vram_free = (_props.total_memory - torch.cuda.memory_allocated()) / 1e9
    _gpu = _props.name
else:
    _vram_total = _vram_free = 0.0
    _gpu = "CPU (no CUDA)"

# Helper used by the probe table AND the hard breakpoint below. MUST stay in
# sync with _load_image_bytes in the Run cell: the final canvas dims it returns
# are exactly what gets handed to run_gpu, which is what the VRAM formula needs.
# Math: effective_max = min(max_resolution, upload_cap); reserve padding budget
# within that (PAD_FACTOR = 1 + 2*BUFFER_FRAC), scale source down to fit the
# reserved input cap, then add pad_px to get the canvas the engine sees.
_BUFFER_FRAC = 0.08
_PAD_FACTOR = 1.0 + 2.0 * _BUFFER_FRAC
def _capped_padded_dims(sw, sh, max_resolution, upload_cap):
    eff_max = max_resolution if upload_cap <= 0 else min(max_resolution, upload_cap)
    eff_in_max = max(1, int(eff_max / _PAD_FACTOR))
    long = max(sw, sh)
    scale = min(1.0, eff_in_max / long)
    iw, ih = max(1, int(sw * scale)), max(1, int(sh * scale))
    pad_px = max(8, int(round(max(iw, ih) * _BUFFER_FRAC)))
    return iw + 2 * pad_px, ih + 2 * pad_px, scale < 1.0 or eff_max < max_resolution

_base_pw, _base_ph, _ = _capped_padded_dims(_sw, _sh, 1200, UPLOAD_MAX_LONG_SIDE)
_base_px = _base_pw * _base_ph

_bbox_on = BBOX_LOCAL and REFINE_MODE == "gradient" and SHAPE_TYPES == ["rotated_ellipse"]
# Effective batch size for memory: bbox scores all RANDOM_SAMPLES ellipses at once; the
# full-canvas multi-kind path scores each shape type's RANDOM_SAMPLES//n_kinds batch
# sequentially, so peak is set by the per-kind batch, not the total.
_eff_k = RANDOM_SAMPLES if _bbox_on else max(1, RANDOM_SAMPLES // max(1, len(SHAPE_TYPES)))
# Peak holds several simultaneous (K, footprint, 3) float32 intermediates. The bbox path
# (gather crops + per-alpha diff + its square) peaks higher than the full-canvas SSE-
# decomposition, so it needs a larger multiplier. These are calibrated against observed peaks.
_SAFETY = 5.5 if _bbox_on else 3.5
print(f"GPU: {_gpu}   VRAM {_vram_total:.0f} GB total, ~{_vram_free:.0f} GB free")
print(f"Source: {_sw}x{_sh} (long side {_long}px)   STICKER_MODE={STICKER_MODE}   "
      f"RANDOM_SAMPLES={RANDOM_SAMPLES}   bbox_local={_bbox_on}")
print(f"Upload cap: {UPLOAD_MAX_LONG_SIDE}px final padded long side (set UPLOAD_MAX_LONG_SIDE=0 to disable).\\n")
print(f"{'MAX_RES':>8} | {'final canvas':>13} | {'downscale':>9} | {'peak VRAM':>9} | {'time vs1200':>11} | fit")
print("-" * 76)

_rec = 1200
for _mr in (1200, 1600, 2000, 2400, 3000, 4000, 5000):
    # Final canvas dims = exactly what the engine sees (cap + reserve-for-pad + pad-added).
    _pw, _ph, _capped = _capped_padded_dims(_sw, _sh, _mr, UPLOAD_MAX_LONG_SIDE)
    _px = _pw * _ph
    # bbox-local scores each candidate over a crop (~max-radius sized), not the full canvas,
    # so the per-candidate memory footprint is the crop area, not _px.
    if _bbox_on:
        _crop_e = min(256, max(_pw, _ph) // 8)
        _foot = min((2 * _crop_e + 1) ** 2, _px)
    else:
        _foot = _px
    _peak = _eff_k * _foot * 3 * 4 * _SAFETY / 1e9
    _ratio = _long / max(_pw, _ph)
    _fits = (_vram_free <= 0) or (_peak < _vram_free * 0.85)
    _is_upscale = _mr > _long and not _capped
    if _fits and not _is_upscale:
        _rec = _mr
    _flag = ("OK " if _fits else "OOM") + (" (capped)" if _capped else (" (>=source)" if _is_upscale else ""))
    print(f"{_mr:>8} | {_pw:>5}x{_ph:<7} | {_ratio:>7.1f}x | {_peak:>7.1f}GB | {_px/_base_px:>9.1f}x | {_flag}")
print("-" * 76)
# Detail saturates at the point where the smallest shapes map to real pixels. Past this,
# higher resolution chases detail the shape budget can't reproduce — pure wasted compute.
# Rule of thumb from the geometrize/primitive family: matched long side ~ sqrt(N) * 28.
_matched = int((NUM_SHAPES ** 0.5) * 28)
print(f"Detail saturation for {NUM_SHAPES} shapes: ~{_matched}px long side. Beyond this,")
print(f"  higher MAX_RESOLUTION mostly costs time (raise NUM_SHAPES to exploit a bigger canvas).")
print(f"VRAM-fit recommendation (largest that fits, no upscaling): MAX_RESOLUTION = {_rec}")
print(f"Quality-matched suggestion: MAX_RESOLUTION ~= min({_rec}, {_matched}) = {min(_rec, _matched)}")
print(f"You currently set: MAX_RESOLUTION = {MAX_RESOLUTION}")

# --- Hard breakpoint ---
_pw, _ph, _ = _capped_padded_dims(_sw, _sh, MAX_RESOLUTION, UPLOAD_MAX_LONG_SIDE)
if _bbox_on:
    _crop_e = min(256, max(_pw, _ph) // 8)
    _foot = min((2 * _crop_e + 1) ** 2, _pw * _ph)
else:
    _foot = _pw * _ph
_peak = _eff_k * _foot * 3 * 4 * _SAFETY / 1e9
if _vram_free > 0 and _peak > _vram_free * 0.85:
    raise SystemExit(
        f"\\nSTOP: MAX_RESOLUTION={MAX_RESOLUTION} needs ~{_peak:.0f} GB but only ~{_vram_free:.0f} GB "
        f"is free.\\n  Fix: set MAX_RESOLUTION={_rec} (or lower RANDOM_SAMPLES) in the Knobs cell, "
        f"re-run it, then re-run this planner."
    )
print(f"\\nOK: MAX_RESOLUTION={MAX_RESOLUTION} -> final canvas {_pw}x{_ph}, peak ~{_peak:.0f} GB. Proceed to Run.")
if _long > MAX_RESOLUTION:
    print(f"Detail note: downscaling {_long/MAX_RESOLUTION:.1f}x — features finer than "
          f"~{_long/MAX_RESOLUTION:.0f}px in the source won't survive. Bump MAX_RESOLUTION if you can afford it.")
'''

CELL_RUN = '''# --- Run ---
def _load_image_bytes(name, raw, max_resolution, sticker, upload_cap=720):
    img = Image.open(io.BytesIO(raw))
    has_alpha = img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info)
    alpha_mask = None
    if sticker:
        if not has_alpha:
            raise ValueError(f"STICKER_MODE=True but {name} has no alpha (mode={img.mode!r})")
        rgba = np.asarray(img.convert("RGBA"), dtype=np.uint8)
        rgb_img = Image.fromarray(rgba[:, :, :3])  # mode inferred from HxWx3 shape
        alpha_mask = rgba[:, :, 3].copy()
        img = rgb_img
    elif has_alpha:
        rgba = img.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.split()[3])
        img = bg
    else:
        img = img.convert("RGB")
    # Apply the VRAM safety cap BEFORE the preset's max_resolution check. The bbox-local
    # scorer's peak VRAM scales as K * (canvas_long_side / 8)^2 * 3 channels — quadratic in
    # the long side. Even a 102 GB Blackwell 6000 OOMs above ~1600 long-side at K=6144;
    # 720 keeps every production preset comfortably under 20 GB peak. Effective ceiling =
    # min(preset's max_resolution, upload_cap). Setting upload_cap=0 disables the cap.
    effective_max = max_resolution if upload_cap <= 0 else min(max_resolution, upload_cap)
    # The cap is meant as a ceiling on the FINAL PADDED canvas (that's what hits VRAM),
    # not on the input image. Padding below adds ~2*BUFFER_FRAC*long pixels on top of
    # the input. Reserve that budget within the cap so the post-padding canvas fits.
    # Without this, a 720 cap with 8% padding leaked the final canvas up to ~836 px
    # (720 + 2*58) and OOM'd a 102 GB GPU at ~99 GB peak vs the probe's predicted 65 GB.
    BUFFER_FRAC = 0.08
    PAD_FACTOR = 1.0 + 2.0 * BUFFER_FRAC
    effective_input_max = max(1, int(effective_max / PAD_FACTOR))
    original_long = max(img.size)
    if original_long > effective_input_max:
        if effective_input_max < max_resolution:
            print(f"[upload cap] long side {original_long} > {effective_input_max} "
                  f"(input reserved under {effective_max}px final cap; "
                  f"preset max_resolution={max_resolution}). Auto-resizing input to {effective_input_max}.")
    if max(img.size) > effective_input_max:
        scale = effective_input_max / max(img.size)
        new_size = (max(1, int(img.size[0] * scale)), max(1, int(img.size[1] * scale)))
        img = img.resize(new_size, Image.LANCZOS)
        if alpha_mask is not None:
            am_img = Image.fromarray(alpha_mask).resize(new_size, Image.LANCZOS)  # mode inferred
            alpha_mask = np.asarray(am_img, dtype=np.uint8)

    # Edge-buffer padding (matches upstream forza_abyss_painter/shapegen/worker.py). FH6's vinyl renderer
    # treats shapes whose extents touch the canvas edge as unbounded, producing large
    # smears and corner artifacts after injection. Pad the source ~8% per side so even
    # shapes that land on the outermost rows/cols of the content stay several pixels
    # inside the actual canvas edge.
    pad_px = max(8, int(round(max(img.size) * BUFFER_FRAC)))
    new_w = img.size[0] + 2 * pad_px
    new_h = img.size[1] + 2 * pad_px
    if sticker:
        buffered = Image.new("RGB", (new_w, new_h), (0, 0, 0))
        buffered.paste(img, (pad_px, pad_px))
        img = buffered
        if alpha_mask is not None:
            padded_alpha = np.zeros((new_h, new_w), dtype=np.uint8)
            src_h, src_w = alpha_mask.shape[:2]
            padded_alpha[pad_px:pad_px + src_h, pad_px:pad_px + src_w] = alpha_mask
            alpha_mask = padded_alpha
    else:
        # Pad with the SOURCE's mean color, not white. The engine inits the opaque canvas
        # to mean(target) (engine.py:359), so a (255,255,255) margin creates ~33% of the
        # padded canvas as a phantom solid-white region that greedy + polish dutifully
        # spend ~100+ shapes painting before reaching the actual content — pure shape-
        # budget waste. Filling the margin with the source's mean color makes the padded
        # region match the canvas init exactly: per-pixel loss = 0 in the padding, so
        # greedy never places candidates there and the full shape budget goes into the
        # content. (Sticker mode is unaffected; its alpha-mask gate already rejects
        # shapes that don't overlap the silhouette, so its (0,0,0) pad never gets shapes.)
        src_arr = np.asarray(img, dtype=np.uint8)
        src_mean = tuple(int(c) for c in src_arr.reshape(-1, 3).mean(axis=0).round().clip(0, 255))
        buffered = Image.new("RGB", (new_w, new_h), src_mean)
        buffered.paste(img, (pad_px, pad_px))
        img = buffered
    return np.asarray(img, dtype=np.uint8), alpha_mask


target_rgb, alpha_mask = _load_image_bytes(
    SOURCE_IMAGE_NAME, SOURCE_IMAGE_BYTES, MAX_RESOLUTION, STICKER_MODE,
    upload_cap=UPLOAD_MAX_LONG_SIDE,
)
h, w = target_rgb.shape[:2]
mode_tag = "sticker" if STICKER_MODE else "opaque"
print(f"target: {w}x{h} ({mode_tag} mode), refine={REFINE_MODE}, alpha_mask={'yes' if alpha_mask is not None else 'no'}")

cfg = GPUConfig(
    num_shapes=NUM_SHAPES,
    random_samples=RANDOM_SAMPLES,
    seed=SEED,
    refine_mode=REFINE_MODE,
    shape_types=SHAPE_TYPES,
    alpha_levels=ALPHA_LEVELS,
    edge_strength=EDGE_STRENGTH,
    posterize_levels=POSTERIZE_LEVELS,
    bbox_local=BBOX_LOCAL,
    joint_polish_steps=JOINT_POLISH_STEPS,
    alpha_threshold=ALPHA_THRESHOLD,
    grad_starts=GRAD_STARTS,
    grad_steps=GRAD_STEPS,
    grad_lr=GRAD_LR,
    mutation_rounds=MUTATION_ROUNDS,
    mutations_per_round=MUTATIONS_PER_ROUND,
    lock_alpha=True,   # SYSTEM CONSTRAINT — see notebook comment above. Not user-tunable.
    polish_freeze_geometry=POLISH_FREEZE_GEOMETRY,
)
import json
from pathlib import Path

stem = Path(SOURCE_IMAGE_NAME).stem
out_dir = Path(OUTPUT_DIR) / stem
out_dir.mkdir(parents=True, exist_ok=True)
mode_str = f"gpu_colab({mode_tag},shapes={cfg.num_shapes},samples={cfg.random_samples})"

def _doc(shapes):
    # MUST stay aligned with FD6Document.to_dict() in forza_abyss_painter/io/json_schema.py — every field
    # the CPU engine writes, the GPU/notebook must also write. Dropping sticker_mode was
    # the cause of the "looks fine in the engine render, looks wrong in the ForzaAbyssPainter.exe
    # preview/inject" mismatch: the engine optimizes shape colors against a grey substrate
    # when sticker mode is on, and downstream tools read sticker_mode from the JSON to pick
    # the matching backdrop. Without that flag the desktop app paints on white and the
    # grey-substrate-optimized colors look bad.
    return {
        "format": "fd6.shapes", "version": 1,
        "source_image": SOURCE_IMAGE_NAME, "image_size": [w, h],
        "shape_count": len(shapes),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "profile": mode_str,
        "sticker_mode": bool(STICKER_MODE),
        "shapes": shapes,
    }

# Output naming convention: include NUM_SHAPES (the shape budget) in every filename so
# users can identify quality level at a glance — eg "my_image_3000.json" is a high-detail
# render vs "my_image_400.json" for a lineart-density render. Budget is the planned shape
# count even if the final actual count is slightly lower (sticker constraints can exhaust).
_BUDGET_TAG = str(NUM_SHAPES)

# Checkpoint to Drive during the run so a session reset mid-run still leaves a recoverable
# JSON. Writes greedy progress every CHECKPOINT_EVERY shapes.
def _checkpoint(shape_idx, shapes_so_far):
    p = out_dir / f"{stem}_{_BUDGET_TAG}_ckpt_{shape_idx}.json"
    p.write_text(json.dumps(_doc(shapes_so_far)))

t0 = time.perf_counter()
shapes_json, final_canvas = run_gpu(
    target_rgb, cfg, alpha_mask=alpha_mask, progress_every=50,
    checkpoint_cb=_checkpoint, checkpoint_every=CHECKPOINT_EVERY,
)
elapsed = time.perf_counter() - t0

# Persist FINAL outputs to Drive IMMEDIATELY — before any variable reset / disconnect can
# lose them. This is the whole point of the Drive version. Filenames include the shape
# budget (NUM_SHAPES) so 400/700/1000/3000-shape renders are distinguishable at a glance.
json_path = out_dir / f"{stem}_{_BUDGET_TAG}.json"
png_path = out_dir / f"{stem}_{_BUDGET_TAG}_render.png"
json_path.write_text(json.dumps(_doc(shapes_json), indent=2))
Image.fromarray(final_canvas, "RGB").save(png_path)

print(f"\\ndone: {len(shapes_json)} shapes in {elapsed:.2f}s ({elapsed/max(1,len(shapes_json))*1000:.1f} ms/shape)")
print(f"SAVED -> {json_path}")
print(f"SAVED -> {png_path}")
if len(shapes_json) < NUM_SHAPES:
    print(f"WARNING: only {len(shapes_json)} of {NUM_SHAPES} shapes committed (sticker constraint exhausted attempts)")
'''

CELL_DISPLAY = '''# --- Display source vs render ---
import matplotlib.pyplot as plt

_src_img = Image.open(io.BytesIO(SOURCE_IMAGE_BYTES))
if _src_img.mode != "RGB":
    _src_for_show = _src_img.convert("RGBA") if STICKER_MODE else _src_img.convert("RGB")
else:
    _src_for_show = _src_img

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
axes[0].imshow(_src_for_show)
axes[0].set_title(f"source ({SOURCE_IMAGE_NAME})")
axes[0].axis("off")
axes[1].imshow(final_canvas)
axes[1].set_title(f"Forza Abyss Painter render ({len(shapes_json)} shapes)")
axes[1].axis("off")
plt.tight_layout()
plt.show()
'''

CELL_MOUNT_DRIVE = '''# --- Mount Google Drive (persistent output; survives session resets) ---
# The Run cell writes the final JSON + PNG here the instant generation finishes, plus
# recoverable checkpoints during the run. Even if the Colab session resets afterward, your
# output is already on Drive — no more losing results to a disconnect.
USE_DRIVE    = True            # False → save to /content only (lost on reset)
DRIVE_FOLDER = "forza_abyss_painter_output"    # subfolder under MyDrive/

import os
if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    OUTPUT_DIR = f"/content/drive/MyDrive/{DRIVE_FOLDER}"
else:
    OUTPUT_DIR = "/content/forza_abyss_painter_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Outputs -> {OUTPUT_DIR}")
print("Per image, results land in <OUTPUT_DIR>/<image_stem>/  (final + numbered checkpoints).")
'''

CELL_DOWNLOAD = '''# --- (Optional) also download to your browser ---
# Outputs are ALREADY saved to Drive by the Run cell. This just additionally pushes the
# final files to your browser's downloads if you want a local copy. Safe to skip.
try:
    from google.colab import files
    files.download(str(json_path))
    files.download(str(png_path))
except Exception as _e:
    print(f"Browser download skipped ({_e}). Your files are on Drive at: {out_dir}")
'''


CELL_CLEANUP = '''# --- Cleanup: free CUDA memory between runs ---
# Run this cell after every failed/successful run when you want to change parameters
# and re-run. Clears the run artifacts AND the IPython output history that pins them
# (the most common reason a "fresh" run still OOMs).
import sys, gc, torch

# 1) Clear Jupyter's last-traceback (holds frame locals = live tensors after a crash)
sys.last_traceback = None
sys.last_value = None
sys.last_type = None

# 2) Drop user-level variables that hold GPU tensors or run outputs
for _name in (
    # Run outputs
    "shapes_json", "final_canvas", "cfg",
    # Preprocessed inputs (CPU but worth clearing to drop the IPython _N hold)
    "target_rgb", "alpha_mask", "h", "w", "mode_tag",
    # Anything that might have leaked from a previous engine call
    "target", "alpha_t", "alpha_mask_f", "canvas",
    "masks", "scores", "colors", "params", "params_cpu",
    "mut_masks", "mut_scores", "mut_colors", "mut", "mut_cpu",
    "best_params", "best_color",
):
    globals().pop(_name, None)

# 3) Clear IPython output history (Out[i], _, __, ___, _oh) — biggest hidden leak
try:
    ip = get_ipython()
    ip.user_ns.pop("_", None)
    ip.user_ns.pop("__", None)
    ip.user_ns.pop("___", None)
    if "_oh" in ip.user_ns:
        ip.user_ns["_oh"].clear()
    if "Out" in ip.user_ns:
        ip.user_ns["Out"].clear()
except (NameError, AttributeError):
    pass

# 4) Force GC and release the allocator pool
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    alloc = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"after cleanup: allocated={alloc:.2f} GB, reserved={reserved:.2f} GB")
    if alloc > 1.0:
        print("WARNING: >1 GB still live. Restart the runtime if you need a fully clean slate.")
else:
    print("(no CUDA — nothing to free)")
'''


def build(preset_key):
    preset = PRESETS[preset_key]
    out = ROOT / "notebooks" / f"fap_gpu_colab_{preset_key}.ipynb"
    notebook = {
        "cells": [
            md(CELL_INTRO),
            md(f"## Preset: **{preset['preset_label']}**\n\nThis variant ships tuned defaults for "
               f"a {preset['num_shapes']}-shape budget. Edit any knob in the Configure cell."),
            md("## 1. Setup — verify CUDA"),
            code(CELL_SETUP_DEPS),
            md("## 2. Setup — load the Forza Abyss Painter GPU engine\n\nRun this cell once per session. It defines `run_gpu(...)` and helpers."),
            code(CELL_SETUP_ENGINE),
            md("## 3. 🧹 Cleanup (run between attempts)\n\nUse this if a previous run OOM'd or you want to start fresh with different parameters. If `allocated` stays >1 GB after this cell, restart the runtime (Runtime → Restart runtime)."),
            code(CELL_CLEANUP),
            md("## 4. Mount Google Drive (persistent output)\n\n**This is the key cell for not losing results.** Outputs save straight to Drive the instant generation finishes, plus checkpoints during the run — so a session reset / disconnect can't lose your JSON + PNG. You'll be prompted to authorize Drive access."),
            code(CELL_MOUNT_DRIVE),
            md("## 5. Upload an image"),
            code(CELL_UPLOAD),
            md("## 6. Posterize preview — **pick POSTERIZE_LEVELS**\n\nShows your image at 4 posterize levels with a ★ recommendation (the lowest level that won't wash out this image's fine detail). Decide your value here, then set it in the Configure cell."),
            code(CELL_POSTERIZE_PREVIEW),
            md("## 7. Configure — set knobs\n\nPreset defaults below. Set `POSTERIZE_LEVELS` from the preview above. `STICKER_MODE=True` requires an alpha channel. Re-run this cell after changing anything."),
            code(cell_knobs(preset)),
            md("## 8. Resolution Planner — **decide MAX_RESOLUTION**\n\nShows the detail/VRAM/time tradeoff for your specific image + GPU, recommends the best fit, and **hard-stops if your chosen resolution would OOM**. Re-run the Knobs cell then this one until it says *Proceed to Run*."),
            code(CELL_RESOLUTION_PLANNER),
            md("## 9. Run\n\nSaves the final JSON + PNG to Drive the moment it finishes (and checkpoints every `CHECKPOINT_EVERY` shapes). Progress prints every 50 shapes."),
            code(CELL_RUN),
            md("## 10. View result"),
            code(CELL_DISPLAY),
            md("## 11. (Optional) browser download\n\nYour files are already on Drive. This just pushes a local copy to your browser too — safe to skip, and it won't matter if the session later resets."),
            code(CELL_DOWNLOAD),
            md(CELL_FOOTER),
        ],
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": out.name, "provenance": [], "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out.write_text(json.dumps(notebook, indent=1))
    print(f"wrote {out}  ({out.stat().st_size/1024:.1f} KB, {len(notebook['cells'])} cells)")


if __name__ == "__main__":
    for _key in PRESETS:
        build(_key)
