"""Frozen per-preset quality baselines (num_shapes, random_samples, etc.).

SINGLE SOURCE OF TRUTH. Imported by notebooks/build_colab_notebook.py (to build the
Colab notebooks) and by fd6/cli/fitsize.py (to size inputs). Do NOT duplicate these
values elsewhere — they are A/B-judged and must not drift. See the
feedback_fixed_quality_params constraint.
"""
from __future__ import annotations

# --- Presets: matched-resolution + tuned per shape budget (see docs/GPU_SHAPEGEN.md) ---
PRESETS = {
    # NOTE: these per-preset values are a FROZEN quality baseline. Do NOT change anything that
    # affects output (random_samples, num_shapes, edge_strength, posterize_levels,
    # joint_polish_steps, alpha_levels) without explicitly flagging it — the user A/B-judges
    # output across runs and a silent change invalidates the comparison. (See the feedback
    # memory.) random_samples differs per preset ON PURPOSE: the bbox crop grows with
    # resolution, so 24576 fits at the medium preset's ≤1000px but not at highres's 1600px.
    "highres_3000": {
        "preset_label": "HIGH-RES / 3000 shapes (ellipse-only, injector-ready)",
        "num_shapes": 3000,
        "random_samples": 6144,    # LOCKED. The most that fits at 1600px (~41 GB; the bbox
                                   # crop is ~5x larger than at 720px). Quality is gradient-
                                   # driven, so this is not the quality bottleneck.
        "max_resolution": 1600,    # ~sqrt(3000)*28
        "edge_strength": 2.0,
        "posterize_levels": 24,
        "joint_polish_steps": 150,  # 3000-shape loss plateaus ~150
        "checkpoint_every": 250,
        "shape_types": '["rotated_ellipse"]',
        "alpha_threshold": 0,
        "lock_alpha": True,
    },
    "medium_1000": {
        "preset_label": "MEDIUM / 1000 shapes (ellipse-only, injector-ready)",
        "num_shapes": 1000,
        "random_samples": 24576,   # LOCKED — the validated baseline that produced the good
                                   # bodycon 720->1000 render. Peaks ~53 GB at 720px bbox, so
                                   # needs a CLEAN 80GB A100 (restart the runtime to free VRAM
                                   # if a prior run is holding it).
        "max_resolution": 1000,    # ~sqrt(1000)*28 ≈ 885, rounded up
        "edge_strength": 3.0,      # push detail harder with a tighter shape budget
        "posterize_levels": 24,
        "joint_polish_steps": 250,  # fewer shapes carry more weight → more correction
        "checkpoint_every": 100,
        "shape_types": '["rotated_ellipse"]',
        "alpha_threshold": 0,
        "lock_alpha": True,
    },
    # --- Low-shape presets for SIMPLE subjects (ellipse-only, injector-ready) ---
    # These exist because flat lineart / single-face busts don't need a 1000+ shape budget —
    # they're mostly transparent or flat color, so the shapes-per-feature density is high even
    # at a few hundred shapes. Fast + cheap (fit a T4/V100), and the JSON ships to the injector
    # like the other ellipse presets. NOT a relaxed version of medium_1000 — different tuning.
    "lineart_400": {
        "preset_label": "LINE-ART / 400 shapes (flat onomatopoeia, kanji, logos — injector-ready)",
        "num_shapes": 400,         # onomatopoeia/logos read cleanly at 250-500; 400 is the
                                   # sweet spot (most shapes go to tracing the thin outlines).
        "random_samples": 8192,    # LOCKED. bbox-local at this low res is cheap, so this fits a
                                   # T4. High sample volume helps trace thin curved strokes.
        "max_resolution": 720,     # flat color → no fine gradient to preserve; 720 keeps the
                                   # outlines crisp without wasting compute (matched ~560px).
        "edge_strength": 4.0,      # the whole image IS edges (outlines on flat fills) — push
                                   # the search hard onto them. Higher than any other preset.
        "posterize_levels": 16,    # near-flat already; low levels clean the AA without loss.
        "joint_polish_steps": 300,  # few shapes each carry a lot → un-stick the outline shapes.
        "checkpoint_every": 100,
        "shape_types": '["rotated_ellipse"]',
        "alpha_threshold": 128,    # these are genuinely transparent over white; binarize the
                                   # mask to kill the white fringe on the anti-aliased edge.
        "lock_alpha": True,
    },
    "headshots_700": {
        "preset_label": "HEADSHOTS / 700 shapes (single transparent face/bust — injector-ready)",
        "num_shapes": 700,         # a cropped single face needs far less than a full body; 700
                                   # sits below medium_1000 ("for less") but above flat lineart
                                   # because faces carry shaded skin + eye/mouth detail.
        "random_samples": 8192,    # LOCKED. Fits a clean A100 with headroom at this res.
        "max_resolution": 900,     # eyes/mouth need pixels; matched for 700 is ~740, nudged up.
        "edge_strength": 3.0,      # emphasize lineart + eyes, but not the extreme flat-lineart 4.
        "posterize_levels": 24,    # skin has soft gradients — keep 24 so shading doesn't band.
        "joint_polish_steps": 300,
        "checkpoint_every": 100,
        "shape_types": '["rotated_ellipse"]',
        "alpha_threshold": 0,      # soft hair/edge silhouette — do NOT binarize (would harden
                                   # the fringe). Raise only if a specific export shows a halo.
        "lock_alpha": True,
    },
    # --- Multi-shape EVALUATION presets (triangle + rotated_rectangle enabled) ---
    # Purpose: see whether richer shapes (sharp corners, straight edges) lift quality before
    # the injector supports them. NOT injector-ready today — preview/eval only. Multi-shape is
    # gradient-only + full-canvas scoring (no bbox) + no joint-polish, so RANDOM_SAMPLES is
    # much lower (it's split across 3 kinds and each kind is scored full-canvas). These are
    # NEW notebooks, not the locked ellipse baselines.
    "shapes_highres_3000": {
        "preset_label": "HIGH-RES / 3000 shapes (multi-shape EVAL — not injectable yet)",
        "num_shapes": 3000,
        "random_samples": 1536,    # 512/kind full-canvas at 1600px (~40 GB on clean A100)
        "max_resolution": 1600,
        "edge_strength": 2.0,
        "posterize_levels": 24,
        "joint_polish_steps": 0,    # joint-polish is ellipse-only; auto-skipped for mixed shapes
        "checkpoint_every": 250,
        "shape_types": '["rotated_ellipse", "triangle", "rotated_rectangle"]',
        "alpha_threshold": 0,
    },
    "shapes_medium_1000": {
        "preset_label": "MEDIUM / 1000 shapes (multi-shape EVAL — not injectable yet)",
        "num_shapes": 1000,
        "random_samples": 3072,    # 1024/kind full-canvas at ~1000px (~19 GB)
        "max_resolution": 1000,
        "edge_strength": 3.0,
        "posterize_levels": 24,
        "joint_polish_steps": 0,
        "checkpoint_every": 100,
        "shape_types": '["rotated_ellipse", "triangle", "rotated_rectangle"]',
        "alpha_threshold": 0,
    },
}
