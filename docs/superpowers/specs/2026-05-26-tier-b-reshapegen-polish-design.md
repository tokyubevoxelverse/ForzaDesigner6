# Tier B short-term: Re-shape-gen (#85) + Polish loaded JSON (#86) — Design

**Date:** 2026-05-26
**Status:** Approved for implementation planning
**Scope:** EXE / consumer GPU path only. Colab/FD6 not affected.
**Tasks closed:** #85, #86

---

## 1. Purpose

After a user clicks **Upload JSON** and loads an existing `fd6.shapes` document,
they currently have no way to:

- **#85** re-run shape-gen against the same source image at a higher budget
  (e.g. 1000 → 3000 shapes) without manually re-picking the image and a preset.
- **#86** polish the colors/alpha of the shapes already in that JSON without
  generating new geometry.

Both are filed in CLAUDE.md §1's planning matrix as "Tier B, ready to start
(no Cursor blocker)". This spec covers shipping both together in one session.

## 2. Non-goals

- **No JSON schema change.** Both features round-trip the existing
  `fd6.shapes` v1 schema. The `source_image` field stays "filename only";
  full-path storage is explicitly rejected (see §4.1).
- **No new quality-knob defaults.** `RANDOM_SAMPLES`, `max_resolution`,
  `UPLOAD_MAX_LONG_SIDE=720`, posterize levels, etc. are unchanged
  (CLAUDE.md §8a).
- **No geometry mutation in polish.** `freeze_geometry=True` is hardcoded for
  #86's polish path. Geometry-mutating polish is reserved for a future
  expert-mode session.
- **No new shape types.** Both features only round-trip existing shape types
  that the injector already understands (CLAUDE.md §3).
- **Colab/FD6 untouched.** This is EXE-side plumbing only.

## 3. User-facing behavior

After **Upload JSON** succeeds and a JSON is loaded into `MainWindow`:

- Two new buttons appear on `upload_panel`:
  - `Re-shape-gen at higher budget…`
  - `Polish loaded JSON…`
- Both buttons are hidden when no JSON is loaded.
- Both buttons are hidden when their feature flag is `False` (see §6).

### 3.1 #85 — Re-shape-gen flow

1. User clicks **Re-shape-gen at higher budget…**.
2. The source image path is resolved (§4.1). If unresolved, the flow aborts.
3. The **existing** `generate_dialog` opens, pre-populated with:
   - `source_path` = resolved source image
   - `output_path` = `<source_stem>_<num_shapes>.json` (same as fresh-gen default)
4. User picks any preset and confirms. Existing fresh-generation pipeline runs.
5. Output is a new JSON file; the loaded JSON is untouched on disk.

### 3.2 #86 — Polish loaded JSON flow

1. User clicks **Polish loaded JSON…**.
2. The source image path is resolved (§4.1). If unresolved, the flow aborts.
3. A new small dialog (`polish_dialog`) appears with two controls:
   - **Polish iterations** — `QSpinBox`, default `150`, range `50–500`,
     step `10`.
   - **Lock alpha to 255** — `QCheckBox`, default `True`.
4. User clicks **Polish**. Confirms output path defaulting to
   `<input>_polished.json` next to the loaded JSON.
5. `GpuGenWorker` spawns `torch_runner` with the new `mode: "polish_only"`
   config. Progress events stream through the existing event channel.
6. Runner saves output via the existing `save_json()` path. The #100
   validator hook runs before the file lands on disk.
7. The loaded JSON on disk is **not** modified.

## 4. Cross-cutting design decisions

### 4.1 Source-image resolution

The loaded JSON's `source_image` field is a bare filename (e.g. `nikke.png`),
not a path. Both features need the full source image. Resolution policy:

1. Try `<json_path.parent> / doc.source_image`. If it exists, use it.
2. Otherwise, open a modal: *"Source image `<filename>` not found next to
   the JSON. Pick it manually?"* and a file picker.
3. Picker cancelled → flow exits silently. No progress dialog appears.

A small helper `_resolve_source_image(json_path, doc) -> Path | None` lives
in `forza_abyss_painter/gui/main_window.py` (or a sibling module) and is
unit-tested.

### 4.2 Output naming

- **Re-shape-gen:** `<source_stem>_<num_shapes>.json` in the directory the
  user chooses in `generate_dialog` (existing behavior — no change).
- **Polish:** `<loaded_json_stem>_polished.json` in the same directory as
  the loaded JSON. If that path already exists, suffix with `_1`, `_2`, …
  using the existing exporter behavior (verify in implementation;
  add a test).

### 4.3 Validator integration

Both outputs are written via the same `save_json()` path. The #100 validator
hook already blocks invalid output from landing on disk and surfaces
warnings in the GUI. No new validation code is added.

### 4.4 Feature flags

`forza_abyss_painter/gui/feature_flags.py` gains:

```python
RESHAPE_GEN_AVAILABLE: bool = False
POLISH_LOADED_AVAILABLE: bool = False
```

Per CLAUDE.md §1c, defaults stay `False` during development. They are
flipped to `True` in the **same commit** that lands the smoke-tested
plumbing. A pinning test enforces the default values (mirrors
`tests/test_gpu_phase3_flag.py`) so the flag value cannot drift
independently of the plumbing.

## 5. Runner architecture

### 5.1 RunConfig extensions

`forza_abyss_painter/runtime/torch_runner.py` adds three optional fields
to `RunConfig`:

| Field | Type | Default | Used when |
|---|---|---|---|
| `mode` | `str` | `"fresh"` | always; validated against `{"fresh", "polish_only"}` |
| `input_shapes_path` | `Path \| None` | `None` | required when `mode == "polish_only"` |
| `polish_steps_override` | `int \| None` | `None` | optional in `polish_only`; overrides preset polish steps |

Validation in `RunConfig.from_dict`:

- `mode` must be one of the allowed values; unknown → `ValueError`.
- `mode == "polish_only"` requires `input_shapes_path` non-None and existing.
- `mode == "polish_only"` does not require `num_shapes`, `random_samples`,
  or `max_resolution`. To support this, the runner config schema relaxes
  these three fields to optional **only when `mode == "polish_only"`**.
  In `mode == "fresh"` they remain required (current behavior preserved).
  The `RunConfig.from_dict` validator gates this conditional requirement.
- `mode == "fresh"` ignores `input_shapes_path`, `polish_steps_override`.

### 5.2 run() dispatch

`run()` adds a single branch at the top after `RunConfig.from_dict`:

```python
if cfg.mode == "polish_only":
    return _run_polish_only(cfg)
# existing fresh path below, unchanged
```

The `_run_polish_only(cfg)` helper:

1. Loads the input shapes via `load_json(cfg.input_shapes_path)` first, so
   the canvas dimensions are known.
2. Loads the source image and resizes it to **the loaded JSON's
   `doc.image_size`** — not `max_resolution`. The shapes' coordinates are
   relative to that canvas; polishing at any other resolution would
   misalign every shape. `max_resolution` in cfg is ignored in
   polish_only mode.
3. Builds the polish inputs: posterized `target`, `alpha_t`, `alpha_mask_f`,
   `edge_weight`, `canvas_init` — same construction the engine uses before
   calling `joint_polish()`. Uses the same `posterize_levels`, `edge_strength`,
   and `sticker_mode` values that were used to generate the loaded JSON
   (read from `doc` if persisted there; otherwise fall back to the cfg
   values, which the GUI populates from the loaded JSON's metadata where
   available). Extract into a shared helper if duplication is meaningful;
   otherwise inline.
4. Calls `joint_polish(...)` with:
   - `shapes_json` = loaded shapes (materialized to the dict form `joint_polish`
     expects).
   - `steps` = `cfg.polish_steps_override` if set, else the preset default
     for the shape count (use the same lookup the fresh path uses).
   - `lock_alpha` = `cfg.lock_alpha` (already in RunConfig).
   - `freeze_geometry=True` — hardcoded; geometry mutation is a non-goal.
   - `purity_penalty=0.0` — irrelevant when `freeze_geometry=True`.
5. Saves the refined shapes via `save_json(cfg.output_json_path, doc)`. The
   #100 validator hook runs in `save_json`.
6. Emits progress events identical to the fresh path: `start`, `tick`,
   `done` with the same event schema `gpu_gen_worker` already parses.

### 5.3 IPC schema additions

`gpu_gen_worker.build_run_config()` gains an optional `mode` kwarg
(default `"fresh"`). A sibling helper `build_polish_config(...)` in the
same module constructs a polish_only config with the right fields. Both
write JSON to the same temp path; `GpuGenWorker` itself does not change.

## 6. GUI changes

### 6.1 upload_panel.py

- Two `QPushButton` instances appended after the existing Upload JSON
  controls.
- Visibility tied to:
  - `feature_flags.RESHAPE_GEN_AVAILABLE` and `POLISH_LOADED_AVAILABLE`
    (set at construction time; flags are static).
  - The loaded-JSON state (toggled via a `set_json_loaded(path: Path | None)`
    slot the panel exposes).
- Each button emits a new signal: `reshape_requested` and `polish_requested`.

### 6.2 main_window.py

- Connects `upload.reshape_requested` → `_on_reshape_requested`.
- Connects `upload.polish_requested` → `_on_polish_requested`.
- `_on_json_loaded_for_preview` calls `upload.set_json_loaded(json_path)`
  after a successful load; clears it on load error.
- `_on_reshape_requested`:
  - Resolves source image (§4.1). Abort if unresolved.
  - Constructs a `GenerateDialog` with `initial_source_path=resolved`.
  - Opens it modally. From here, fresh-gen flow is identical to today.
- `_on_polish_requested`:
  - Resolves source image (§4.1). Abort if unresolved.
  - Constructs a `PolishDialog` with `initial_steps=150`, `initial_lock_alpha=True`.
  - On accept, calls `build_polish_config(...)` and dispatches to
    `GpuGenWorker` exactly as fresh-gen does.

### 6.3 polish_dialog.py (new)

Single QDialog:

- Title: *"Polish loaded JSON"*
- Body shows the resolved source image filename and the input JSON filename
  for confirmation.
- Polish iterations spinbox (50–500, step 10, default 150).
- Lock-alpha checkbox (default checked).
- Output path display (auto-computed, read-only label with a
  *"Choose output…"* button if user wants to override).
- Polish / Cancel buttons.
- `accept()` exposes `.values() -> {steps, lock_alpha, output_path}`.

### 6.4 generate_dialog.py

Add a single optional constructor kwarg: `initial_source_path: Path | None`.
When set, pre-populates the source-image field and skips the user picking
it again. Output path is recomputed from the source stem like fresh-gen.

## 7. Testing strategy

All tests follow the CLAUDE.md §1a discipline: **no mocks for things that
have a real implementation available**. Joint_polish runs for real on the
host's torch install; subprocess tests spawn the real runner.

### 7.1 Unit / integration tests

| Test file | Covers |
|---|---|
| `tests/test_torch_runner_polish_mode.py` | `RunConfig.from_dict({"mode": "polish_only", ...})` parses; `mode == "polish_only"` requires `input_shapes_path`; unknown mode raises; default mode is `"fresh"`. |
| `tests/test_polish_runner_integration.py` | Subprocess spawn with a 3-ellipse fixture JSON + 32×32 image; output JSON exists, shape count preserved, validator-clean, geometry unchanged (frozen), color tensor differs from input by at least one pixel (proves polish ran). |
| `tests/test_upload_panel_reshape_polish_buttons.py` | Offscreen Qt; flags ON + JSON loaded → both buttons visible. Flag OFF → button hidden. JSON cleared → button hidden. |
| `tests/test_reshape_polish_flags_gating.py` | Pins `RESHAPE_GEN_AVAILABLE = False`, `POLISH_LOADED_AVAILABLE = False`. Documents the checklist for flipping. Mirrors `tests/test_gpu_phase3_flag.py`. |
| `tests/test_resolve_source_image.py` | Sibling exists → returns sibling path. Sibling missing → returns `None`. JSON parent unreadable → returns `None`. |
| `tests/test_polish_dialog_values.py` | `PolishDialog` constructs; default steps and lock_alpha are correct; `.values()` returns the right dict; user output-path override is respected. |
| `tests/test_generate_dialog_initial_source.py` | `GenerateDialog(initial_source_path=...)` pre-fills the source field; output default tracks the source stem. |

### 7.2 Test-fixture discipline (CLAUDE.md §8f)

All fixtures that contain `fd6.shapes` JSON go through `load_json()` /
`save_json()` round-trip in a `conftest.py` helper before any assertion runs
against their contents. No hand-typed schema strings live in test files.

### 7.3 Local smoke (CLAUDE.md §1b)

Single offscreen-Qt smoke that constructs a real `MainWindow()`, loads a
real test JSON + image, clicks the Polish button (or signals through the
slot directly to avoid an event-loop), runs polish_only to completion, and
asserts:

- output file exists at the expected `_polished.json` path,
- validator passes on the output,
- shape count matches input,
- progress events fired through to the GUI (status changed at least once).

Only ONE `MainWindow` constructed per process (§8h).

A second smoke covers re-shape-gen: load JSON, click Re-shape-gen, confirm
the existing fresh-gen smoke path completes end-to-end with the pre-filled
source image.

## 8. Risks and out-of-scope items

### 8.1 Risks

| Risk | Mitigation |
|---|---|
| Source image silently changed between JSON save and polish (user replaced the file) | Out of scope. Polish operates on whatever image is at the resolved path; we do not hash/verify. The output JSON's `source_image` field will reflect the actually-used filename. |
| Polish steps too high → VRAM headroom shrinks below #125 guard | #125 guard already checks free VRAM before runner start. Polish memory is a fraction of full shape-gen, so collision is unlikely in practice. If it happens, the existing OOM modal fires. |
| User polishes a multi-shape JSON with non-injector-safe types (rectangle/triangle) | `joint_polish` currently optimizes only `rotated_ellipse` shapes (verify in implementation). If it touches other types, scope is widened — either restrict polish to ellipses-only or error out cleanly. **Decision required during implementation, not deferred.** |
| `generate_dialog` does not accept an `initial_source_path` kwarg today | Add one. Backward compatible (kwarg defaults to None). |
| QSettings drift between sessions | Not used here. Both features are stateless per-invocation. |

### 8.2 Out of scope for this spec

- Geometry-mutating polish (expert mode with `freeze_geometry=False`).
- Polishing only a subset of shapes (e.g. "polish shapes 100–200").
- Re-shape-gen from a different source image (use fresh-gen for that — the
  whole point of this button is "same image, different budget").
- Schema change to store full source-image paths.
- Multi-shape support (#65 / #129 / #130 are separate strategic tasks).
- Auto-OOM recovery (#124, separate).

## 9. Acceptance criteria

This spec is "done" (mergeable to main) when:

1. All seven test files in §7.1 are green.
2. Local smoke (§7.3) is green for both features.
3. Feature flags flipped to `True` in the same commit as the smoke.
4. SMB sync per CLAUDE.md §6e — sync after local smoke green, before
   asking QUASAR/Cursor to validate. EXE rebuild succeeds.
5. Cursor confirms the two new buttons render and run end-to-end on a
   real `fd6.shapes` JSON on QUASAR.
6. CLAUDE.md is left unchanged — these features are routine wiring, not
   new failure patterns.

## 10. File-touch summary

**New:**
- `forza_abyss_painter/gui/polish_dialog.py`
- `tests/test_torch_runner_polish_mode.py`
- `tests/test_polish_runner_integration.py`
- `tests/test_upload_panel_reshape_polish_buttons.py`
- `tests/test_reshape_polish_flags_gating.py`
- `tests/test_resolve_source_image.py`
- `tests/test_polish_dialog_values.py`
- `tests/test_generate_dialog_initial_source.py`

**Modified:**
- `forza_abyss_painter/runtime/torch_runner.py` — RunConfig fields, `_run_polish_only`, run() dispatch.
- `forza_abyss_painter/gui/upload_panel.py` — two buttons + signals + `set_json_loaded`.
- `forza_abyss_painter/gui/main_window.py` — two slots + source-image resolver + dialog wiring.
- `forza_abyss_painter/gui/generate_dialog.py` — `initial_source_path` kwarg.
- `forza_abyss_painter/gui/gpu_gen_worker.py` — `build_polish_config()` helper; optional `mode` in `build_run_config()`.
- `forza_abyss_painter/gui/feature_flags.py` — two new flags.
- `tests/conftest.py` (if needed) — round-trip helper per §7.2.

**Untouched (deliberately):**
- `forza_abyss_painter/io/json_schema.py` (no schema change).
- `forza_abyss_painter/io/validator.py` (validator path already covers polish output).
- `forza_abyss_painter/shapegen/gpu/joint_polish.py` (used as-is; signature already fits).
- `forza_abyss_painter/shapegen/presets.py` (no quality-knob changes).
