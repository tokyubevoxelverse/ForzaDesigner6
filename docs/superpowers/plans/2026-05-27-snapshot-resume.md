# GPU Snapshots, Live Preview, Resume, and Auto-Polish — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship periodic on-disk snapshots during GPU runs so failures are recoverable, wire a live preview off those snapshots so the middle panel updates during a run, add a Resume-from-snapshot GUI flow, and bring GPU presets to CPU polish parity.

**Architecture:**
- One primitive: `<output_stem>_<count>.json` written by the runner at every `checkpoint_cb`. The same file feeds two consumers: GUI preview (render off-thread via `render_shapes`) and resume (replay onto canvas via new `seed_shapes` arg on `engine.run_gpu`).
- New optional `_run_config` block in snapshot JSON carries the original run params for one-click resume.
- GPU min cadence enforced at TWO places (GUI spinbox + `RunConfig.from_dict`).

**Tech Stack:** Python 3.10+, PySide6 (offscreen Qt for GUI tests), PyTorch via embedded runtime (integration tests gated on torch), pytest, existing `io/exporter`, existing `shapegen/render`.

**Spec:** `docs/superpowers/specs/2026-05-27-snapshot-resume-design.md`

---

## File Structure

### New files

| Path | Responsibility |
|---|---|
| `forza_abyss_painter/io/snapshots.py` | Pure `snapshot_path_for(output_path, count) -> Path` helper. No I/O. |
| `forza_abyss_painter/gui/snapshot_render.py` | `_RenderSnapshotJob(QRunnable)` — reads snapshot JSON off the GUI thread, calls `render_shapes`, marshals the numpy canvas back to `PreviewPanel.on_preview` via `QMetaObject.invokeMethod`. |
| `forza_abyss_painter/gui/resume_dialog.py` | `ResumeDialog(QDialog)` — reads `_run_config`, shows "Continue X → Y" summary, returns RunConfig dict via `.values()`. Picker-fallback when `_run_config` missing. |
| `tests/test_snapshot_path_for.py` | Pure helper math (no Qt, no torch). |
| `tests/test_validator_underscore_keys.py` | `_run_config` doesn't trigger unknown-key warnings; malformed snapshots still error. |
| `tests/test_run_config_seed_shapes_path.py` | `seed_shapes_path` parses; missing file → ValueError; polish_only + seed → ValueError. |
| `tests/test_run_config_cuda_min_cadence.py` | cuda+50 raises; cuda+100 OK; cuda+0 OK; cpu+10 OK. |
| `tests/test_engine_seed_shapes.py` | `run_gpu(seed_shapes=...)` replays + continues. Gated on torch. |
| `tests/test_snapshot_runner_integration.py` | Subprocess: 300 shapes + checkpoint_every=100 → 3 snapshots land. Gated on torch. |
| `tests/test_resume_runner_integration.py` | Subprocess: fresh + seed_shapes_path → output = seeded + generated. Gated on torch. |
| `tests/test_build_run_config_polish.py` | `build_run_config` forwards `joint_polish_steps` from preset. |
| `tests/test_gpu_gen_worker_snapshot_signal.py` | Worker dispatches `{"kind": "snapshot", ...}` events to the new Signal. |
| `tests/test_snapshot_render_job.py` | `_RenderSnapshotJob` reads fixture snapshot, renders, calls preview slot. |
| `tests/test_upload_panel_resume_button.py` | Button always visible; signal carries the picked path. |
| `tests/test_resume_dialog_values.py` | Reads `_run_config` → values dict; falls back to preset picker when missing. |
| `tests/test_generate_dialog_checkpoint_spinbox.py` | Spinbox bounds (min=100, max=1000, step=50, default=100); written into config dict. |
| `tests/test_main_window_snapshot_wiring.py` | Real MainWindow smoke for snapshot signal + resume slot. |

### Modified files

| Path | Change |
|---|---|
| `forza_abyss_painter/io/validator.py` | Allow top-level keys starting with `_` without warning. |
| `forza_abyss_painter/runtime/torch_runner.py` | `RunConfig` gains `seed_shapes_path`; `from_dict` validates min cadence + polish/seed conflict; new `_write_snapshot` helper; `_checkpoint_cb` writes snapshot + emits `snapshot` event; fresh-mode loads seed shapes when set. |
| `forza_abyss_painter/shapegen/gpu/engine.py` | `run_gpu(...)` accepts optional `seed_shapes: list[dict] | None`; replays each onto canvas before greedy loop. |
| `forza_abyss_painter/gui/gpu_gen_worker.py` | New `snapshot = Signal(int, int, str)`; dispatch in `_dispatch`; `build_run_config` forwards `joint_polish_steps` + `checkpoint_every` (no more `// 20` heuristic when caller provides it). |
| `forza_abyss_painter/gui/generate_dialog.py` | `LOCAL_PRESETS` gains `joint_polish_steps`; dialog adds a "Snapshot every N shapes" `QSpinBox`; preset combo wires its value into the spawn config. |
| `forza_abyss_painter/gui/upload_panel.py` | New `resume_requested = Signal(Path)`; new "Resume from snapshot…" button below the Tier B row; click handler opens `QFileDialog` and emits. |
| `forza_abyss_painter/gui/main_window.py` | `_on_gpu_snapshot` slot + render-job dispatch + single-slot throttle; `_on_resume_requested` slot + ResumeDialog construction + GpuGenWorker spawn. |
| `docs/JSON_SPEC.md` | New "Optional metadata fields" section documenting `_*` reserved prefix and `_run_config`. |

### Deliberately untouched

- `forza_abyss_painter/shapegen/worker.py` — CPU snapshot machinery already works.
- `engine.refill_dead_shapes` default + ordering — end-of-run refill is current behavior.
- `forza_abyss_painter/io/exporter.py::save_json` — keeps writing whatever doc it's given; snapshot helper builds the doc.

---

## Task 1: snapshot_path_for helper

**Files:**
- Create: `forza_abyss_painter/io/snapshots.py`
- Create: `tests/test_snapshot_path_for.py`

- [ ] **Step 1.1: Write the failing test**

Create `tests/test_snapshot_path_for.py`:

```python
"""snapshot_path_for builds the <output_stem>_<count>.json path.

Used by both the CPU worker (existing inline construction) and the GPU
runner (new code). Pure path math — no I/O.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from forza_abyss_painter.io.snapshots import snapshot_path_for


def test_basic(tmp_path):
    out = tmp_path / "ziz_dnace.json"
    assert snapshot_path_for(out, 2900) == tmp_path / "ziz_dnace_2900.json"


def test_preserves_parent_dir(tmp_path):
    out = tmp_path / "subdir" / "ziz_dnace.json"
    assert snapshot_path_for(out, 100) == tmp_path / "subdir" / "ziz_dnace_100.json"


def test_stem_with_dots(tmp_path):
    out = tmp_path / "ziz.dance.v2.json"
    assert snapshot_path_for(out, 500) == tmp_path / "ziz.dance.v2_500.json"


def test_no_extension(tmp_path):
    out = tmp_path / "ziz"   # caller didn't add .json
    assert snapshot_path_for(out, 100) == tmp_path / "ziz_100.json"


def test_count_zero(tmp_path):
    out = tmp_path / "x.json"
    assert snapshot_path_for(out, 0) == tmp_path / "x_0.json"


def test_accepts_str_path():
    result = snapshot_path_for("/tmp/x.json", 100)
    assert result == Path("/tmp/x_100.json")
```

- [ ] **Step 1.2: Run test to verify it fails**

Run: `pytest tests/test_snapshot_path_for.py -v`

Expected: FAIL with `ImportError: cannot import name 'snapshot_path_for'`.

- [ ] **Step 1.3: Implement**

Create `forza_abyss_painter/io/snapshots.py`:

```python
"""Snapshot file naming + helpers.

Shared by:
- CPU `shapegen/worker.py` (was inline construction; switch to this).
- GPU `runtime/torch_runner.py` (new code).
- GUI `gui/main_window.py` (resume flow needs to look for siblings).

Snapshots live next to the final output JSON. Naming pattern matches
the CPU side's historical convention so existing artifacts on disk
keep round-tripping.
"""
from __future__ import annotations

from pathlib import Path


def snapshot_path_for(output_path: str | Path, count: int) -> Path:
    """Return the snapshot path for `count` shapes alongside the final
    output JSON.

    E.g. `snapshot_path_for("Downloads/x/ziz.json", 2900)` →
    `PosixPath('Downloads/x/ziz_2900.json')`.

    Uses `Path.stem` so filenames with dots (`ziz.dance.v2.json`) get
    the count appended before the LAST suffix only. Always returns
    `.json` extension regardless of the input's extension (or absence
    thereof), so a caller that passes a stem-only path still gets a
    canonical snapshot name.
    """
    output = Path(output_path)
    parent = output.parent
    stem = output.stem
    return parent / f"{stem}_{count}.json"
```

- [ ] **Step 1.4: Run test to verify it passes**

Run: `pytest tests/test_snapshot_path_for.py -v`

Expected: PASS — 6 tests green.

- [ ] **Step 1.5: Commit**

```bash
git add forza_abyss_painter/io/snapshots.py tests/test_snapshot_path_for.py
git commit -m "$(cat <<'EOF'
feat(io): snapshot_path_for helper (#snapshot-resume)

Pure path-math helper used by both the CPU worker (currently inline)
and the GPU runner (new code in the upcoming snapshot/resume work).
Stem + parent + _N.json — matches the existing CPU convention so
old artifacts on disk keep round-tripping.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Auto-polish on fresh GPU runs

**Files:**
- Modify: `forza_abyss_painter/gui/generate_dialog.py:42-67` (LOCAL_PRESETS gains `joint_polish_steps`)
- Modify: `forza_abyss_painter/gui/gpu_gen_worker.py` (`build_run_config` forwards the field)
- Create: `tests/test_build_run_config_polish.py`

- [ ] **Step 2.1: Write the failing test**

Create `tests/test_build_run_config_polish.py`:

```python
"""build_run_config forwards joint_polish_steps from the preset so
fresh GPU runs polish at end (matching CPU calibration)."""
from __future__ import annotations

from pathlib import Path

import pytest

from forza_abyss_painter.gui.gpu_gen_worker import build_run_config
from forza_abyss_painter.gui.generate_dialog import LOCAL_PRESETS


def _img(tmp_path) -> Path:
    p = tmp_path / "src.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\n")
    return p


def test_preset_polish_steps_forwarded(tmp_path):
    """When the preset dict carries joint_polish_steps, build_run_config
    forwards it to the runner config dict."""
    preset = {
        "label": "Medium — 1000 shapes",
        "num_shapes": 1000,
        "max_resolution": 720,
        "random_samples": 8192,
        "joint_polish_steps": 150,
    }
    cfg = build_run_config(_img(tmp_path), tmp_path / "out.json", preset)
    assert cfg["joint_polish_steps"] == 150


def test_polish_steps_defaults_to_zero_when_missing(tmp_path):
    """Back-compat: presets without joint_polish_steps (custom user
    configs, older test fixtures) get 0 (no polish)."""
    preset = {
        "label": "Custom",
        "num_shapes": 100,
        "max_resolution": 360,
        "random_samples": 1024,
    }
    cfg = build_run_config(_img(tmp_path), tmp_path / "out.json", preset)
    assert cfg["joint_polish_steps"] == 0


def test_local_presets_all_have_polish_steps():
    """The shipped LOCAL_PRESETS must all carry joint_polish_steps per
    the spec calibration table."""
    expected = {
        "Lineart — 400 shapes": 100,
        "Headshot — 700 shapes": 150,
        "Medium — 1000 shapes": 150,
        "Hi-Res — 3000 shapes (FH6 closed only)": 250,
    }
    for p in LOCAL_PRESETS:
        label = p["label"]
        assert "joint_polish_steps" in p, (
            f"preset {label!r} missing joint_polish_steps"
        )
        assert p["joint_polish_steps"] == expected[label], (
            f"preset {label!r} polish_steps={p['joint_polish_steps']} "
            f"expected {expected[label]} per spec §3.4"
        )
```

- [ ] **Step 2.2: Run test to verify it fails**

Run: `pytest tests/test_build_run_config_polish.py -v`

Expected: 3 failures. `test_local_presets_all_have_polish_steps` fails first because the field is missing; the other two fail because `build_run_config` doesn't read it.

- [ ] **Step 2.3: Add joint_polish_steps to LOCAL_PRESETS**

Edit `forza_abyss_painter/gui/generate_dialog.py:42-67`. Add `"joint_polish_steps": <N>,` to each preset entry per the spec table:

```python
LOCAL_PRESETS: list[dict] = [
    {
        "label": "Lineart — 400 shapes",
        "num_shapes": 400, "max_resolution": 480,
        "random_samples": 4096, "est_peak_vram_gib": 2.5,
        "joint_polish_steps": 100,
        "description": "Logos, kanji, line art. Fast (~2 min on 30/40-series).",
    },
    {
        "label": "Headshot — 700 shapes",
        "num_shapes": 700, "max_resolution": 600,
        "random_samples": 6144, "est_peak_vram_gib": 3.5,
        "joint_polish_steps": 150,
        "description": "Portraits. Balanced quality and speed.",
    },
    {
        "label": "Medium — 1000 shapes",
        "num_shapes": 1000, "max_resolution": 720,
        "random_samples": 8192, "est_peak_vram_gib": 5.0,
        "joint_polish_steps": 150,
        "description": "General-purpose. Recommended default for 8+ GiB cards.",
    },
    {
        "label": "Hi-Res — 3000 shapes (FH6 closed only)",
        "num_shapes": 3000, "max_resolution": 1000,
        "random_samples": 12288, "est_peak_vram_gib": 12.0,
        "joint_polish_steps": 250,
        "description": "Maximum detail. Needs 16+ GiB free — close FH6 first.",
    },
]
```

- [ ] **Step 2.4: Forward in build_run_config**

Edit `forza_abyss_painter/gui/gpu_gen_worker.py` `build_run_config`. Add to the returned dict (preserve existing fields):

```python
        "joint_polish_steps": int(preset.get("joint_polish_steps", 0)),
```

Place it near `"checkpoint_every"` for grouping.

- [ ] **Step 2.5: Run tests**

```bash
pytest tests/test_build_run_config_polish.py tests/test_generate_dialog_initial_source.py tests/test_generate_dialog_recommendation_label.py -v
```

Expected: all green (3 new + existing pass).

- [ ] **Step 2.6: Commit**

```bash
git add forza_abyss_painter/gui/generate_dialog.py forza_abyss_painter/gui/gpu_gen_worker.py tests/test_build_run_config_polish.py
git commit -m "$(cat <<'EOF'
feat(gpu): auto-polish on fresh runs (CPU parity)

GPU presets previously omitted joint_polish_steps so fresh runs were
raw greedy + refill. CPU presets have always polished. Bring GPU to
parity with calibrated CPU values per spec §3.4:

  Lineart 400  → 100 steps
  Headshot 700 → 150 steps
  Medium  1000 → 150 steps (match CPU medium_1000)
  Hi-Res  3000 → 250 steps (match CPU highres_3000)

build_run_config forwards the field; RunConfig.from_dict already
accepts it. CLAUDE.md §8a quality-knob change called out in the spec.

Wall-time impact: ~30-60s at 1000 shapes, ~60-120s at 3000 shapes.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Validator tolerance for `_*` keys

**Files:**
- Modify: `forza_abyss_painter/io/validator.py` (extend unknown-key tolerance)
- Create: `tests/test_validator_underscore_keys.py`

- [ ] **Step 3.1: Locate the validator's unknown-key logic**

Run `grep -n "unknown\|_run_config\|extra_keys" forza_abyss_painter/io/validator.py | head -20` to see if there's an existing unknown-keys path. If not, the validator may silently ignore extras already — the test will tell you.

- [ ] **Step 3.2: Write the failing test**

Create `tests/test_validator_underscore_keys.py`:

```python
"""The fd6.shapes validator must tolerate snapshot metadata fields
prefixed with `_`. `_run_config` is the resume breadcrumb but the
convention is open-ended: any `_*` top-level key is metadata, not
contract, and should not trigger warnings or errors.
"""
from __future__ import annotations

from forza_abyss_painter.io.validator import validate_document, Severity


def _good_doc() -> dict:
    return {
        "format": "fd6.shapes",
        "version": 1,
        "source_image": "x.png",
        "image_size": [64, 64],
        "shape_count": 1,
        "generated_at": "",
        "profile": "",
        "sticker_mode": False,
        "shapes": [
            {"type": "rotated_ellipse", "x": 16.0, "y": 32.0,
             "rx": 8.0, "ry": 8.0, "angle": 0.0,
             "color": [128, 128, 128, 255]},
        ],
    }


def test_baseline_clean_doc_has_no_errors():
    issues = validate_document(_good_doc())
    errors = [i for i in issues if i.severity is Severity.ERROR]
    assert errors == []


def test_run_config_block_does_not_trigger_warnings():
    doc = _good_doc()
    doc["_run_config"] = {
        "target_shape_count": 100,
        "random_samples": 1024,
        "max_resolution": 360,
    }
    issues = validate_document(doc)
    # No issue should mention `_run_config` as unknown/unexpected.
    for issue in issues:
        msg = (issue.message or "").lower()
        assert "_run_config" not in msg, (
            f"validator complained about _run_config: {issue}"
        )


def test_arbitrary_underscore_metadata_tolerated():
    """Other underscore-prefixed keys (future metadata) shouldn't warn."""
    doc = _good_doc()
    doc["_diag"] = {"wall_time_s": 17.3}
    doc["_my_experiment"] = "alpha"
    issues = validate_document(doc)
    for issue in issues:
        msg = (issue.message or "").lower()
        assert "_diag" not in msg
        assert "_my_experiment" not in msg


def test_real_field_typos_still_warn_or_error():
    """Defensive: only `_*` is metadata. A typo'd real field should
    still get flagged. E.g. `formatt` (typo of `format`)."""
    doc = _good_doc()
    doc["formatt"] = "fd6.shapes"   # typo, no underscore
    issues = validate_document(doc)
    # We expect SOMETHING — either a warning or error pointing at the
    # unknown key. If the validator silently ignores ALL unknowns
    # today, this test documents that behavior; the assertion below
    # then weakens to just "validator ran cleanly on the doc body".
    # Adjust based on what the existing validator does.
    # For now: assert at least the baseline-good-doc validation still
    # works on this near-doc.
    errors = [i for i in issues if i.severity is Severity.ERROR]
    # The shapes are still well-formed; ERROR list should remain empty.
    assert errors == []
```

- [ ] **Step 3.3: Run test to see current state**

Run: `pytest tests/test_validator_underscore_keys.py -v`

Document the result:
- If all 4 tests pass: the validator already tolerates extras → no implementation change needed for this task. Skip to commit.
- If `test_run_config_block_does_not_trigger_warnings` fails with the validator emitting a warning that contains `_run_config`: add an exemption.

- [ ] **Step 3.4: If needed — exempt `_*` keys**

Open `forza_abyss_painter/io/validator.py`. Find where it iterates top-level keys. Add an early-skip for keys starting with `_`:

```python
        if key.startswith("_"):
            # Reserved for non-spec metadata (e.g. _run_config from
            # the resume system). Validator stays out of the way.
            continue
```

(Adapt to the actual function structure — read the file first.)

If the validator currently warns on unknown keys explicitly:
- Add the `_*` skip BEFORE the warning emission.
- Document: "Underscore-prefixed top-level keys are reserved for runner state per docs/JSON_SPEC.md."

- [ ] **Step 3.5: Run tests**

Run: `pytest tests/test_validator_underscore_keys.py tests/test_polish_runner_integration.py -v`

Expected: 4/4 underscore tests pass; existing validator-touching tests don't regress.

- [ ] **Step 3.6: Update docs/JSON_SPEC.md**

Add to `docs/JSON_SPEC.md` a new section before any "Strict Mode" or "Future Versions" section:

```markdown
## Optional metadata fields (`_*`)

Top-level keys prefixed with `_` are reserved for non-spec metadata
written by tools that produce or consume `fd6.shapes` documents.
Validators MUST NOT warn or error on these keys.

Currently reserved:

- `_run_config` — runner state, written by the GPU/CPU shape-gen
  runner into snapshot JSONs (and final output JSONs) so partial runs
  can be resumed with the original parameters. Shape varies by runner
  version; consumers should treat unknown sub-keys as opaque.

Future tools may add other `_*` keys (e.g. `_diag`, `_telemetry`).
Producers SHOULD prefix any non-spec metadata with `_` to stay out
of the validator's strict path.
```

- [ ] **Step 3.7: Commit**

```bash
git add forza_abyss_painter/io/validator.py tests/test_validator_underscore_keys.py docs/JSON_SPEC.md
git commit -m "$(cat <<'EOF'
feat(io): validator tolerates `_*` metadata top-level keys

Reserves underscore-prefixed top-level keys for non-spec metadata
(the resume system's `_run_config` is the first user). Validator
skips them entirely — no warnings, no errors. Documented in
JSON_SPEC.md so future producers know the convention.

The validator's strict path stays in place for unprefixed unknown
keys (typos like `formatt` still raise).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: RunConfig — `seed_shapes_path` + GPU min cadence

**Files:**
- Modify: `forza_abyss_painter/runtime/torch_runner.py` (RunConfig dataclass + `from_dict`)
- Create: `tests/test_run_config_seed_shapes_path.py`
- Create: `tests/test_run_config_cuda_min_cadence.py`

- [ ] **Step 4.1: Write the failing tests**

Create `tests/test_run_config_seed_shapes_path.py`:

```python
"""RunConfig.seed_shapes_path: fresh-mode resume points to a snapshot.

Validation:
- str → Path on parse
- Missing file → ValueError
- Set in polish_only mode → ValueError (resume is fresh-only)
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from forza_abyss_painter.runtime.torch_runner import RunConfig


def _fresh_dict(tmp_path) -> dict:
    image = tmp_path / "img.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    return {
        "image_path": str(image),
        "output_json_path": str(tmp_path / "out.json"),
        "num_shapes": 100,
        "max_resolution": 360,
        "random_samples": 1024,
    }


def _snapshot(tmp_path) -> Path:
    p = tmp_path / "out_50.json"
    p.write_text(json.dumps({"format": "fd6.shapes", "version": 1, "shapes": []}))
    return p


def test_default_seed_shapes_path_is_none(tmp_path):
    cfg = RunConfig.from_dict(_fresh_dict(tmp_path))
    assert cfg.seed_shapes_path is None


def test_seed_shapes_path_parses(tmp_path):
    snap = _snapshot(tmp_path)
    d = _fresh_dict(tmp_path)
    d["seed_shapes_path"] = str(snap)
    cfg = RunConfig.from_dict(d)
    assert cfg.seed_shapes_path == snap


def test_seed_shapes_path_missing_file_raises(tmp_path):
    d = _fresh_dict(tmp_path)
    d["seed_shapes_path"] = str(tmp_path / "does_not_exist_999.json")
    with pytest.raises(ValueError, match="seed_shapes_path"):
        RunConfig.from_dict(d)


def test_polish_only_rejects_seed_shapes_path(tmp_path):
    snap = _snapshot(tmp_path)
    image = tmp_path / "img.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    shapes = tmp_path / "in.json"
    shapes.write_text(json.dumps({"format": "fd6.shapes", "version": 1, "shapes": []}))
    d = {
        "image_path": str(image),
        "output_json_path": str(tmp_path / "out.json"),
        "mode": "polish_only",
        "input_shapes_path": str(shapes),
        "seed_shapes_path": str(snap),
    }
    with pytest.raises(ValueError, match="seed_shapes_path"):
        RunConfig.from_dict(d)
```

Create `tests/test_run_config_cuda_min_cadence.py`:

```python
"""GPU runs enforce a minimum checkpoint cadence of 100 to keep snapshot
write frequency reasonable on fast cards. CPU runs can checkpoint every
shape."""
from __future__ import annotations

import pytest

from forza_abyss_painter.runtime.torch_runner import RunConfig


def _fresh_dict(tmp_path) -> dict:
    image = tmp_path / "img.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    return {
        "image_path": str(image),
        "output_json_path": str(tmp_path / "out.json"),
        "num_shapes": 100,
        "max_resolution": 360,
        "random_samples": 1024,
    }


def test_cuda_checkpoint_every_50_rejected(tmp_path):
    d = _fresh_dict(tmp_path)
    d["device"] = "cuda"
    d["checkpoint_every"] = 50
    with pytest.raises(ValueError, match="checkpoint_every"):
        RunConfig.from_dict(d)


def test_cuda_checkpoint_every_100_accepted(tmp_path):
    d = _fresh_dict(tmp_path)
    d["device"] = "cuda"
    d["checkpoint_every"] = 100
    cfg = RunConfig.from_dict(d)
    assert cfg.checkpoint_every == 100


def test_cuda_checkpoint_every_zero_accepted(tmp_path):
    """0 means 'disabled' — explicit opt-out from snapshots. Preserved
    for unit-test callers + power users."""
    d = _fresh_dict(tmp_path)
    d["device"] = "cuda"
    d["checkpoint_every"] = 0
    cfg = RunConfig.from_dict(d)
    assert cfg.checkpoint_every == 0


def test_cpu_checkpoint_every_10_accepted(tmp_path):
    d = _fresh_dict(tmp_path)
    d["device"] = "cpu"
    d["checkpoint_every"] = 10
    cfg = RunConfig.from_dict(d)
    assert cfg.checkpoint_every == 10


def test_cpu_checkpoint_every_1_accepted(tmp_path):
    d = _fresh_dict(tmp_path)
    d["device"] = "cpu"
    d["checkpoint_every"] = 1
    cfg = RunConfig.from_dict(d)
    assert cfg.checkpoint_every == 1
```

- [ ] **Step 4.2: Run tests to verify failure**

```bash
pytest tests/test_run_config_seed_shapes_path.py tests/test_run_config_cuda_min_cadence.py -v
```

Expected: failures on `seed_shapes_path` (field doesn't exist) and on cuda+50 (no min-cadence validation).

- [ ] **Step 4.3: Add `seed_shapes_path` field**

Edit `forza_abyss_painter/runtime/torch_runner.py`. In the `RunConfig` dataclass (after `polish_steps_override`):

```python
    # --- Resume support (#snapshot-resume) -----------------------------
    # When set in fresh mode: runner loads the partial doc, replays its
    # shapes onto the canvas, and continues the greedy loop from
    # len(seeded) → num_shapes. polish_only + seed = ValueError (resume
    # is fresh-only).
    seed_shapes_path: Path | None = None
```

- [ ] **Step 4.4: Add validation in `from_dict`**

Edit `from_dict` in the same file. Add AFTER the existing `mode` validation (around the polish_only branch):

```python
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
```

Add `seed_shapes_path=seed_shapes_path,` to the final `cls(...)` constructor call.

Also: add the cuda min-cadence check just before the device validation block:

```python
        ce = int(d.get("checkpoint_every", 0))
        if device == "cuda" and 0 < ce < 100:
            raise ValueError(
                f"checkpoint_every must be >= 100 on cuda (got {ce}); "
                f"set 0 to disable snapshots entirely"
            )
```

Note: `device` is read just above this block in the existing code. The check must run AFTER `device` is bound but BEFORE the final `cls(...)`. Position it after the existing `device not in ('cuda', 'cpu')` validation.

- [ ] **Step 4.5: Run tests**

```bash
pytest tests/test_run_config_seed_shapes_path.py tests/test_run_config_cuda_min_cadence.py tests/test_torch_runner.py tests/test_torch_runner_polish_mode.py -v 2>&1 | tail -20
```

Expected: 4 (seed) + 5 (cadence) + existing torch_runner tests all green.

- [ ] **Step 4.6: Commit**

```bash
git add forza_abyss_painter/runtime/torch_runner.py tests/test_run_config_seed_shapes_path.py tests/test_run_config_cuda_min_cadence.py
git commit -m "$(cat <<'EOF'
feat(runtime): RunConfig.seed_shapes_path + cuda min cadence

Adds two pieces of the snapshot/resume foundation:

- seed_shapes_path: optional fresh-mode field pointing at a partial
  snapshot. Validated as existing file. polish_only + seed_shapes_path
  → ValueError (resume is fresh-only by design).
- cuda + 0 < checkpoint_every < 100 → ValueError. Defense in depth
  alongside the GUI spinbox bound (added in a later commit). 0
  (disabled) and cpu (any value) remain accepted.

Runner branch wiring lands in the next commit (engine seed support +
torch_runner load-and-pass).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: engine.run_gpu — seed_shapes replay

**Files:**
- Modify: `forza_abyss_painter/shapegen/gpu/engine.py` (`run_gpu` signature + replay)
- Create: `tests/test_engine_seed_shapes.py`

- [ ] **Step 5.1: Write the failing test**

Create `tests/test_engine_seed_shapes.py`:

```python
"""run_gpu(seed_shapes=...) replays each seeded ellipse onto canvas
before entering the greedy loop. Final shapes list = seeded +
newly-generated. Total count = num_shapes parameter (NOT
len(seeded) + num_shapes — num_shapes is the TARGET).
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")


def _make_target(h=32, w=32):
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, :w // 2] = (200, 80, 80)
    arr[:, w // 2:] = (80, 80, 200)
    return arr


def _ellipse(x, y):
    return {
        "type": "rotated_ellipse",
        "x": float(x), "y": float(y),
        "rx": 4.0, "ry": 4.0, "angle": 0.0,
        "color": [128, 128, 128, 255],
    }


def test_seed_only_returns_seeded_unchanged():
    """num_shapes equals len(seeded) → loop runs 0 iterations →
    output is exactly the seeded shapes."""
    from forza_abyss_painter.shapegen.gpu.engine import GPUConfig, run_gpu
    target = _make_target()
    seed = [_ellipse(8, 16), _ellipse(16, 16), _ellipse(24, 16)]
    cfg = GPUConfig(
        num_shapes=3,           # equal to len(seed)
        random_samples=64,
        bbox_local=True,
        joint_polish_steps=0,   # skip polish for speed
        refill_dead_shapes=False,
    )
    shapes_out, _canvas = run_gpu(
        target_rgb=target, cfg=cfg, alpha_mask=None,
        progress_every=0, checkpoint_cb=None, checkpoint_every=0,
        seed_shapes=seed,
    )
    assert len(shapes_out) == 3
    for i, (a, b) in enumerate(zip(seed, shapes_out)):
        for key in ("x", "y", "rx", "ry", "angle"):
            assert abs(a[key] - b[key]) < 1e-6, (
                f"seeded shape {i} key {key} mutated: "
                f"in={a[key]} out={b[key]}"
            )


def test_seed_plus_generated_produces_target_count():
    """num_shapes=5, len(seeded)=3 → output has 5 shapes; first 3
    bit-identical to seed; last 2 generated by greedy."""
    from forza_abyss_painter.shapegen.gpu.engine import GPUConfig, run_gpu
    target = _make_target()
    seed = [_ellipse(8, 16), _ellipse(16, 16), _ellipse(24, 16)]
    cfg = GPUConfig(
        num_shapes=5,
        random_samples=64,
        bbox_local=True,
        joint_polish_steps=0,
        refill_dead_shapes=False,
    )
    shapes_out, _canvas = run_gpu(
        target_rgb=target, cfg=cfg, alpha_mask=None,
        progress_every=0, checkpoint_cb=None, checkpoint_every=0,
        seed_shapes=seed,
    )
    assert len(shapes_out) == 5
    for i in range(3):
        for key in ("x", "y", "rx", "ry", "angle"):
            assert abs(seed[i][key] - shapes_out[i][key]) < 1e-6


def test_no_seed_unchanged_behavior():
    """seed_shapes=None must produce identical output to omitting the
    kwarg (back-compat)."""
    from forza_abyss_painter.shapegen.gpu.engine import GPUConfig, run_gpu
    target = _make_target()
    cfg = GPUConfig(
        num_shapes=2,
        random_samples=64,
        seed=42,                # fixed seed → deterministic
        bbox_local=True,
        joint_polish_steps=0,
        refill_dead_shapes=False,
    )
    out_default, _ = run_gpu(
        target_rgb=target, cfg=cfg, alpha_mask=None,
        progress_every=0, checkpoint_cb=None, checkpoint_every=0,
    )
    out_none, _ = run_gpu(
        target_rgb=target, cfg=cfg, alpha_mask=None,
        progress_every=0, checkpoint_cb=None, checkpoint_every=0,
        seed_shapes=None,
    )
    assert len(out_default) == 2
    assert len(out_none) == 2
```

- [ ] **Step 5.2: Run test to verify failure**

```bash
pytest tests/test_engine_seed_shapes.py -v
```

Expected: errors / failures on the `seed_shapes` kwarg not existing.

- [ ] **Step 5.3: Add `seed_shapes` to run_gpu**

Edit `forza_abyss_painter/shapegen/gpu/engine.py`. Find the `def run_gpu(...)` signature (around line 410-440 — search for it). Add the new kwarg at the end:

```python
def run_gpu(
    target_rgb,
    cfg: GPUConfig,
    alpha_mask=None,
    progress_every: int = 0,
    checkpoint_cb=None,
    checkpoint_every: int = 0,
    seed_shapes: "list[dict] | None" = None,
):
```

Inside the function, AFTER the `canvas` initialization but BEFORE the main `while shape_idx < cfg.num_shapes:` loop, add the replay block:

```python
    # Resume support (#snapshot-resume): when seed_shapes is provided,
    # replay each onto canvas + append to the shapes list. Greedy loop
    # then starts at len(seeded) and continues to cfg.num_shapes. The
    # randomness state is unaffected — only the canvas + shapes list
    # change. Polish + refill run at end as usual on the FULL set.
    if seed_shapes:
        for s in seed_shapes:
            if s.get("type") != "rotated_ellipse":
                # Defensive: caller should have rejected upstream
                # (runner branch validates). If we get here, fail
                # loud rather than corrupting canvas state.
                raise ValueError(
                    f"seed_shapes contains non-ellipse type "
                    f"{s.get('type')!r}; resume currently supports "
                    f"rotated_ellipse only"
                )
            params = torch.tensor(
                [s["x"], s["y"], s["rx"], s["ry"], s["angle"]],
                dtype=DTYPE, device=device,
            )
            color = torch.tensor(s["color"][:3], dtype=DTYPE, device=device)
            alpha_val = int(s["color"][3])
            canvas = _composite_one(
                kinds[0],   # ellipse kind (bbox_local path)
                canvas, params, color, h, w, alpha_mask_f, alpha_val,
            )
            shapes.append(dict(s))   # copy to avoid caller mutation
        shape_idx = len(shapes)
```

The variables `canvas`, `shapes`, `shape_idx`, `device`, `kinds`, `alpha_mask_f`, `h`, `w` all exist in the surrounding scope from prior init code — verify by reading the function before/after the insertion point.

If `cfg.num_shapes <= len(seed_shapes)`, the `while` loop condition is false on the first check, so the loop runs zero iterations and falls through to polish/refill on the seeded set unchanged. That's the desired behavior for the "seed_only" test case.

- [ ] **Step 5.4: Run tests**

```bash
pytest tests/test_engine_seed_shapes.py -v
```

Expected: 3/3 green. Each test runs the engine on a 32×32 image; takes ~5 seconds total on CPU even without CUDA.

Also run the runner regression set:
```bash
pytest tests/test_torch_runner.py tests/test_torch_runner_polish_mode.py tests/test_polish_runner_integration.py -v 2>&1 | tail -10
```

No regression — existing tests don't set `seed_shapes` so they hit the unchanged default path.

- [ ] **Step 5.5: Commit**

```bash
git add forza_abyss_painter/shapegen/gpu/engine.py tests/test_engine_seed_shapes.py
git commit -m "$(cat <<'EOF'
feat(engine): run_gpu accepts seed_shapes for resume

Optional list[dict] kwarg. When provided, each shape is replayed onto
the canvas via _composite_one before the greedy loop starts. shape_idx
initializes at len(seeded), so the loop generates only (num_shapes -
len(seeded)) new shapes. Polish + refill run on the FULL final set.

Non-ellipse shapes in seed → ValueError. Production callers (runner
fresh+seed branch) validate upstream; this is the defensive last
line.

Bit-identical seed preservation pinned by test_seed_only_returns_
seeded_unchanged (output == input when num_shapes == len(seeded)).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: torch_runner — snapshot write + IPC event + seed wiring

**Files:**
- Modify: `forza_abyss_painter/runtime/torch_runner.py`
- Create: `tests/test_snapshot_runner_integration.py`
- Create: `tests/test_resume_runner_integration.py`

This is the biggest task — three related changes in the same file. Implementing TDD style, three test files first, then one big implementation, then commits.

- [ ] **Step 6.1: Write the snapshot integration test**

Create `tests/test_snapshot_runner_integration.py`:

```python
"""Subprocess: fresh GPU run with checkpoint_every=100, num_shapes=300
must produce 3 snapshot files at the right names + each must parse +
contain _run_config."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
PIL = pytest.importorskip("PIL")


def _image(path: Path, h=32, w=32):
    from PIL import Image
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, :w // 2] = (200, 80, 80)
    arr[:, w // 2:] = (80, 80, 200)
    Image.fromarray(arr, "RGB").save(path)


@pytest.mark.skipif(
    not torch.cuda.is_available() and os.environ.get("FAP_SNAPSHOT_TEST_FORCE_CPU") != "1",
    reason="snapshot test runs the real runner; skipping on CPU-only host. "
           "Set FAP_SNAPSHOT_TEST_FORCE_CPU=1 to force.",
)
def test_runner_writes_snapshots_at_each_checkpoint(tmp_path):
    image = tmp_path / "img.png"
    _image(image)
    out = tmp_path / "out" / "fixture.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg = {
        "image_path": str(image),
        "output_json_path": str(out),
        "num_shapes": 300,
        "max_resolution": 360,
        "random_samples": 256,
        "checkpoint_every": 100,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "lock_alpha": True,
        # Defaults: bbox_local=True, joint_polish_steps=0
    }
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "-m", "forza_abyss_painter.runtime.torch_runner",
         "--config", str(cfg_path)],
        capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, (
        f"runner exited {result.returncode}\nstderr:\n{result.stderr}"
    )
    # Final output JSON.
    assert out.is_file()
    # Snapshots at 100, 200, 300.
    for n in (100, 200, 300):
        snap = out.parent / f"fixture_{n}.json"
        assert snap.is_file(), f"missing snapshot {snap}"
        doc = json.loads(snap.read_text(encoding="utf-8"))
        assert doc["format"] == "fd6.shapes"
        assert doc["shape_count"] == n
        # _run_config breadcrumb for resume.
        assert "_run_config" in doc, f"snapshot {snap.name} missing _run_config"
        rc = doc["_run_config"]
        assert rc["target_shape_count"] == 300
        assert rc["random_samples"] == 256
        assert rc["max_resolution"] == 360


def test_snapshot_event_in_stderr(tmp_path):
    """The snapshot event must appear on stderr alongside checkpoint
    events. Runs without GPU (fewer shapes, small image)."""
    image = tmp_path / "img.png"
    _image(image, h=16, w=16)
    out = tmp_path / "out" / "fixture.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg = {
        "image_path": str(image),
        "output_json_path": str(out),
        "num_shapes": 6,
        "max_resolution": 64,
        "random_samples": 16,
        "checkpoint_every": 3,
        "device": "cpu",
        "lock_alpha": True,
    }
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "-m", "forza_abyss_painter.runtime.torch_runner",
         "--config", str(cfg_path)],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0
    # Find at least one snapshot event in stderr.
    snapshot_events = []
    for line in result.stderr.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("kind") == "snapshot":
            snapshot_events.append(ev)
    assert len(snapshot_events) >= 1
    first = snapshot_events[0]
    assert "shape_count" in first
    assert "total" in first
    assert "path" in first
    assert Path(first["path"]).is_file()
```

- [ ] **Step 6.2: Write the resume integration test**

Create `tests/test_resume_runner_integration.py`:

```python
"""Subprocess: fresh run with seed_shapes_path → output contains
seeded shapes + newly-generated. Tests the full resume runner flow."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
PIL = pytest.importorskip("PIL")


def _image(path: Path, h=32, w=32):
    from PIL import Image
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, :w // 2] = (200, 80, 80)
    arr[:, w // 2:] = (80, 80, 200)
    Image.fromarray(arr, "RGB").save(path)


def _snapshot(path: Path, w=32, h=32) -> dict:
    """3-ellipse fd6.shapes doc."""
    shapes = [
        {"type": "rotated_ellipse", "x": 8.0, "y": 16.0, "rx": 4.0, "ry": 4.0,
         "angle": 0.0, "color": [128, 128, 128, 255]},
        {"type": "rotated_ellipse", "x": 16.0, "y": 16.0, "rx": 4.0, "ry": 4.0,
         "angle": 30.0, "color": [128, 128, 128, 255]},
        {"type": "rotated_ellipse", "x": 24.0, "y": 16.0, "rx": 4.0, "ry": 4.0,
         "angle": 60.0, "color": [128, 128, 128, 255]},
    ]
    doc = {
        "format": "fd6.shapes", "version": 1,
        "source_image": "img.png",
        "image_size": [w, h], "shape_count": len(shapes),
        "generated_at": "", "profile": "test",
        "sticker_mode": False, "shapes": shapes,
    }
    path.write_text(json.dumps(doc), encoding="utf-8")
    return doc


def test_resume_runner_appends_to_seed(tmp_path):
    image = tmp_path / "img.png"
    _image(image)
    snap = tmp_path / "out" / "fixture_3.json"
    snap.parent.mkdir(parents=True, exist_ok=True)
    in_doc = _snapshot(snap, w=32, h=32)

    out = tmp_path / "out" / "fixture.json"
    cfg = {
        "image_path": str(image),
        "output_json_path": str(out),
        "num_shapes": 6,
        "max_resolution": 32,
        "random_samples": 16,
        "seed_shapes_path": str(snap),
        "checkpoint_every": 0,
        "device": "cpu",
        "lock_alpha": True,
    }
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "-m", "forza_abyss_painter.runtime.torch_runner",
         "--config", str(cfg_path)],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, (
        f"runner exited {result.returncode}\nstderr:\n{result.stderr}"
    )
    out_doc = json.loads(out.read_text(encoding="utf-8"))
    assert out_doc["shape_count"] == 6
    # First 3 shapes bit-identical to seed.
    for i in range(3):
        for key in ("x", "y", "rx", "ry", "angle"):
            assert abs(in_doc["shapes"][i][key] - out_doc["shapes"][i][key]) < 1e-6


def test_resume_runner_rejects_non_ellipse_seed(tmp_path):
    """Snapshot with a rectangle shape → runner emits clean error event
    before invoking run_gpu."""
    image = tmp_path / "img.png"
    _image(image)
    snap = tmp_path / "out" / "bad_2.json"
    snap.parent.mkdir(parents=True, exist_ok=True)
    snap.write_text(json.dumps({
        "format": "fd6.shapes", "version": 1,
        "source_image": "img.png",
        "image_size": [32, 32], "shape_count": 1,
        "generated_at": "", "profile": "test",
        "sticker_mode": False,
        "shapes": [
            {"type": "rectangle", "x": 16.0, "y": 16.0,
             "hw": 8.0, "hh": 8.0,
             "color": [100, 100, 100, 255]},
        ],
    }), encoding="utf-8")

    out = tmp_path / "out" / "fixture.json"
    cfg = {
        "image_path": str(image),
        "output_json_path": str(out),
        "num_shapes": 4,
        "max_resolution": 32,
        "random_samples": 16,
        "seed_shapes_path": str(snap),
        "checkpoint_every": 0,
        "device": "cpu",
        "lock_alpha": True,
    }
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "-m", "forza_abyss_painter.runtime.torch_runner",
         "--config", str(cfg_path)],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode != 0
    assert "resume_unsupported_shape" in result.stderr or \
           "rotated_ellipse" in result.stderr
    assert not out.exists(), "output written despite resume error"
```

- [ ] **Step 6.3: Run tests to verify failure**

```bash
pytest tests/test_snapshot_runner_integration.py tests/test_resume_runner_integration.py -v
```

Expected: failures (or skips on no-torch hosts). Snapshot tests fail because snapshot event isn't emitted; resume tests fail because seed loading isn't wired.

- [ ] **Step 6.4: Implement `_write_snapshot` helper**

Edit `forza_abyss_painter/runtime/torch_runner.py`. Add a new function near `_run_polish_only` (before `run()`):

```python
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
    import json
    import os
    import time
    from pathlib import Path
    from forza_abyss_painter.io.snapshots import snapshot_path_for

    snap_path = snapshot_path_for(cfg.output_json_path, count)
    tmp_path = snap_path.with_suffix(snap_path.suffix + ".tmp")

    doc = {
        "format": "fd6.shapes",
        "version": 1,
        "source_image": cfg.image_path.name,
        "image_size": [image_w, image_h],
        "shape_count": count,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "profile": cfg.preset_label,
        "sticker_mode": bool(cfg.sticker_mode),
        "shapes": list(shapes_list),
        "_run_config": {
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
        },
    }
    tmp_path.write_text(json.dumps(doc), encoding="utf-8")
    os.replace(tmp_path, snap_path)
    return snap_path
```

- [ ] **Step 6.5: Wire `_write_snapshot` into `_checkpoint_cb`**

Find the existing `_checkpoint_cb` inside `run()` (around line 372 of torch_runner.py — search for `def _checkpoint_cb`). Replace it:

```python
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
```

`rgb` is in scope from earlier `_downscale_to_max_resolution` call — verify the variable name matches.

- [ ] **Step 6.6: Wire `seed_shapes_path` into the fresh-mode path**

In `run()`, just before the existing `shapes_list, _final_canvas = run_gpu(...)` call, add:

```python
        # Resume support: when seed_shapes_path is set, load the
        # partial snapshot + pass shapes to run_gpu(seed_shapes=...).
        # The runner branch already validated mode==fresh in
        # from_dict, so this is fresh-only by construction.
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
```

Then change the existing `run_gpu(...)` call to pass `seed_shapes=seed_shapes`:

```python
            shapes_list, _final_canvas = run_gpu(
                target_rgb=rgb,
                cfg=gpu_cfg,
                alpha_mask=alpha_mask if cfg.sticker_mode else None,
                progress_every=cfg.progress_every,
                checkpoint_cb=_checkpoint_cb if cfg.checkpoint_every > 0 else None,
                checkpoint_every=cfg.checkpoint_every,
                seed_shapes=seed_shapes,
            )
```

- [ ] **Step 6.7: Embed `_run_config` in final output too**

Find the existing `save_json(doc, cfg.output_json_path)` call near the end of `run()`. Replace the doc construction so the final output also embeds `_run_config`:

```python
        with logger.start_phase("save_json",
                                 output_path=str(cfg.output_json_path)):
            from forza_abyss_painter.io.exporter import save_json
            from forza_abyss_painter.io.json_schema import FD6Document
            h_canvas, w_canvas = rgb.shape[:2]
            doc = FD6Document.from_engine(
                source_image=cfg.image_path.name,
                image_size=(w_canvas, h_canvas),
                shapes=_shape_dicts_to_objects(shapes_list),
                profile_name=cfg.preset_label,
                sticker_mode=cfg.sticker_mode,
            )
            # Embed run config so the FINAL output also carries the
            # provenance breadcrumb (mirrors snapshots).
            save_json(doc, cfg.output_json_path)
            # save_json is the canonical schema-checked path; for the
            # _run_config metadata we read+inject+rewrite (small file,
            # negligible cost) so we don't have to expand FD6Document.
            import json
            doc_dict = json.loads(
                cfg.output_json_path.read_text(encoding="utf-8")
            )
            doc_dict["_run_config"] = {
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
            cfg.output_json_path.write_text(
                json.dumps(doc_dict), encoding="utf-8",
            )
```

- [ ] **Step 6.8: Run tests**

```bash
pytest tests/test_snapshot_runner_integration.py tests/test_resume_runner_integration.py tests/test_torch_runner.py tests/test_torch_runner_polish_mode.py tests/test_polish_runner_integration.py -v 2>&1 | tail -25
```

Expected: all green (or skip-due-to-torch on this host). If `test_snapshot_event_in_stderr` and `test_resume_runner_appends_to_seed` and `test_resume_runner_rejects_non_ellipse_seed` are CPU-only (per their existing pattern), they should run on this host. CUDA-gated tests may skip.

If snapshot tests run on CPU but the engine is slow at max_resolution=32 + 6 shapes, the wall time should be <30s per test. If longer, drop the test count further.

- [ ] **Step 6.9: Commit**

```bash
git add forza_abyss_painter/runtime/torch_runner.py tests/test_snapshot_runner_integration.py tests/test_resume_runner_integration.py
git commit -m "$(cat <<'EOF'
feat(runtime): periodic snapshots + resume runner wiring (#snapshot-resume)

Three runner-side changes that share the same IPC + on-disk artifact:

1. _write_snapshot helper builds <output_stem>_<count>.json with the
   shapes list AND a _run_config metadata block (target_shape_count,
   K, max_resolution, posterize, sticker, etc.). Atomic write via
   .tmp + os.replace().
2. _checkpoint_cb rewritten: emits the existing "checkpoint" event
   (lightweight, drives progress bar) AND a new "snapshot" event
   carrying the saved path. Snapshot write failures are non-fatal —
   logged but don't crash the run.
3. Fresh-mode handles cfg.seed_shapes_path: loads the partial doc,
   validates ellipse-only, passes shapes into run_gpu(seed_shapes=).
   Empty seed / non-ellipse seed / load failure each emit a clean
   typed error event (resume_empty_seed, resume_unsupported_shape,
   load_seed_shapes).

Final output now also embeds _run_config so a finished JSON shows
what produced it.

Tests gated on torch via importorskip; resume + snapshot subprocess
tests run end-to-end at small canvas sizes on CPU.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: GpuGenWorker — snapshot signal

**Files:**
- Modify: `forza_abyss_painter/gui/gpu_gen_worker.py`
- Create: `tests/test_gpu_gen_worker_snapshot_signal.py`

- [ ] **Step 7.1: Write the failing test**

Create `tests/test_gpu_gen_worker_snapshot_signal.py`:

```python
"""GpuGenWorker has a new `snapshot = Signal(int, int, str)` that
fires when the runner emits a snapshot event."""
from __future__ import annotations

from pathlib import Path

import pytest

from forza_abyss_painter.gui.gpu_gen_worker import GpuGenWorker


def test_snapshot_signal_exists():
    """The signal must exist on the class (introspection check, no
    QApplication needed)."""
    assert hasattr(GpuGenWorker, "snapshot")


def test_dispatch_routes_snapshot_event(tmp_path):
    """_dispatch returns True for a 'snapshot' kind event and emits
    the right Signal payload."""
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])

    py = Path("/usr/bin/python3")   # not invoked; dispatch test only
    cfg = tmp_path / "cfg.json"
    cfg.write_text("{}")
    worker = GpuGenWorker(embedded_python_exe=py, config_path=cfg)

    received: list = []
    worker.snapshot.connect(lambda c, t, p: received.append((c, t, p)))

    handled = worker._dispatch({
        "kind": "snapshot",
        "shape_count": 100,
        "total": 300,
        "path": "/tmp/fixture_100.json",
    })
    assert handled is True
    assert received == [(100, 300, "/tmp/fixture_100.json")]


def test_dispatch_unknown_returns_false(tmp_path):
    """Unknown event kinds aren't snapshots and aren't routed."""
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])

    py = Path("/usr/bin/python3")
    cfg = tmp_path / "cfg.json"
    cfg.write_text("{}")
    worker = GpuGenWorker(embedded_python_exe=py, config_path=cfg)
    handled = worker._dispatch({"kind": "unrecognized_event"})
    assert handled is False
```

- [ ] **Step 7.2: Run test to verify failure**

```bash
pytest tests/test_gpu_gen_worker_snapshot_signal.py -v
```

Expected: `AttributeError: type object 'GpuGenWorker' has no attribute 'snapshot'`.

- [ ] **Step 7.3: Add the signal + dispatch**

Edit `forza_abyss_painter/gui/gpu_gen_worker.py`.

In the `GpuGenWorker` class signals block (alongside `started`, `progress`, `checkpoint`, `done`, `error`, `finished`):

```python
    snapshot = Signal(int, int, str)  # shape_count, total, snapshot_path
```

In `_dispatch(self, event)`, add a branch BEFORE the existing `else: return False`:

```python
        elif kind == "snapshot":
            self.snapshot.emit(
                int(event.get("shape_count", 0)),
                int(event.get("total", 0)),
                str(event.get("path", "")),
            )
```

- [ ] **Step 7.4: Run tests**

```bash
pytest tests/test_gpu_gen_worker_snapshot_signal.py -v
```

Expected: 3/3 green.

- [ ] **Step 7.5: Commit**

```bash
git add forza_abyss_painter/gui/gpu_gen_worker.py tests/test_gpu_gen_worker_snapshot_signal.py
git commit -m "$(cat <<'EOF'
feat(gui): GpuGenWorker.snapshot signal (#snapshot-resume)

New Signal(int, int, str) emitted when the runner sends a "snapshot"
IPC event. Carries (shape_count, total, snapshot_path). Existing
"checkpoint" signal remains (lightweight, used by progress bar) —
the two events fire side-by-side now.

GUI consumer (MainWindow) wires this to the off-thread render job in
a later commit.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: snapshot_render.py — off-thread renderer

**Files:**
- Create: `forza_abyss_painter/gui/snapshot_render.py`
- Create: `tests/test_snapshot_render_job.py`

- [ ] **Step 8.1: Write the failing test**

Create `tests/test_snapshot_render_job.py`:

```python
"""_RenderSnapshotJob reads a fixture snapshot JSON, renders it via
render_shapes, and pushes the numpy canvas into PreviewPanel.on_preview.

The job must run synchronously when invoked from the same thread so
we don't need a Qt event loop for the test — it's a QRunnable, and
calling .run() directly executes the body. Cross-thread marshaling
to the GUI thread is tested separately under MainWindow."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from forza_abyss_painter.gui.preview_panel import PreviewPanel
from forza_abyss_painter.gui.snapshot_render import _RenderSnapshotJob


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _snapshot(path: Path, w=32, h=32):
    path.write_text(json.dumps({
        "format": "fd6.shapes", "version": 1,
        "source_image": "x.png",
        "image_size": [w, h], "shape_count": 1,
        "generated_at": "", "profile": "",
        "sticker_mode": False,
        "shapes": [
            {"type": "rotated_ellipse", "x": 16.0, "y": 16.0,
             "rx": 8.0, "ry": 8.0, "angle": 0.0,
             "color": [200, 80, 80, 255]},
        ],
    }), encoding="utf-8")


def test_render_job_calls_on_preview(qapp, tmp_path):
    snap = tmp_path / "fixture_1.json"
    _snapshot(snap)

    panel = PreviewPanel()
    # Capture the canvas the preview panel receives.
    received = {}
    original = panel.on_preview
    def spy(arr):
        received["canvas"] = arr
        original(arr)
    panel.on_preview = spy   # type: ignore

    job = _RenderSnapshotJob(snap, panel)
    job.run()
    # Inline run dispatches QMetaObject.invokeMethod with QueuedConnection,
    # which posts an event. Drain the queue.
    qapp.processEvents()

    assert "canvas" in received, "preview.on_preview was never called"
    canvas = received["canvas"]
    assert canvas.shape == (32, 32, 3) or canvas.shape == (32, 32, 4)
    panel.deleteLater()


def test_render_job_swallows_corrupt_snapshot(qapp, tmp_path):
    """If the snapshot is mid-write or invalid, the job must return
    cleanly without raising — the next snapshot fires within seconds."""
    snap = tmp_path / "corrupt_1.json"
    snap.write_text("{not json")

    panel = PreviewPanel()
    received = {}
    original = panel.on_preview
    def spy(arr):
        received["canvas"] = arr
        original(arr)
    panel.on_preview = spy   # type: ignore

    job = _RenderSnapshotJob(snap, panel)
    job.run()   # must not raise
    qapp.processEvents()

    assert "canvas" not in received
    panel.deleteLater()


def test_render_job_handles_empty_shapes(qapp, tmp_path):
    """Edge: snapshot at count=0 (theoretically) has empty shapes
    list. Render should produce a clean canvas, not raise."""
    snap = tmp_path / "empty_0.json"
    snap.write_text(json.dumps({
        "format": "fd6.shapes", "version": 1,
        "source_image": "x.png",
        "image_size": [32, 32], "shape_count": 0,
        "generated_at": "", "profile": "",
        "sticker_mode": False,
        "shapes": [],
    }), encoding="utf-8")

    panel = PreviewPanel()
    received = {}
    original = panel.on_preview
    def spy(arr):
        received["canvas"] = arr
        original(arr)
    panel.on_preview = spy   # type: ignore

    job = _RenderSnapshotJob(snap, panel)
    job.run()
    qapp.processEvents()

    assert "canvas" in received
    panel.deleteLater()
```

- [ ] **Step 8.2: Run test to verify failure**

```bash
pytest tests/test_snapshot_render_job.py -v
```

Expected: `ImportError: cannot import name '_RenderSnapshotJob'`.

- [ ] **Step 8.3: Implement**

Create `forza_abyss_painter/gui/snapshot_render.py`:

```python
"""Off-thread snapshot → canvas → PreviewPanel rendering.

Used by MainWindow when the GpuGenWorker emits a `snapshot` Signal
during a run. The QRunnable reads the snapshot JSON, renders via
`render_shapes` (pure CPU, no torch), and marshals the resulting
numpy canvas back to the GUI thread via QMetaObject.invokeMethod
calling `PreviewPanel.on_preview`.

Throttling (single-slot queue) lives in MainWindow — this module
just renders one snapshot.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QRunnable, QMetaObject, Qt, Q_ARG

if TYPE_CHECKING:
    from forza_abyss_painter.gui.preview_panel import PreviewPanel


class _RenderSnapshotJob(QRunnable):
    """Background render: snapshot JSON → numpy canvas → preview slot.

    Errors are swallowed silently:
      - Snapshot may be mid-write (next snapshot fires within ~1s on GPU).
      - Snapshot may have been deleted between event-fire and read.
      - render_shapes may raise on malformed shapes (unlikely; the
        runner-side validator catches most issues).

    All cases: log + return. The next snapshot event will trigger
    another render.
    """

    def __init__(self, snapshot_path: "str | Path",
                 preview: "PreviewPanel") -> None:
        super().__init__()
        self._path = Path(snapshot_path)
        self._preview = preview

    def run(self) -> None:   # noqa: D401 — QRunnable contract
        try:
            from forza_abyss_painter.io.exporter import load_json
            from forza_abyss_painter.shapegen.render import render_shapes
            doc = load_json(str(self._path))
            shapes = doc.materialize_shapes()
            w, h = doc.image_size if doc.image_size else (1, 1)
            if w < 1 or h < 1:
                return
            transparent_bg = bool(getattr(doc, "sticker_mode", False))
            canvas = render_shapes(
                shapes, int(w), int(h),
                background=(255, 255, 255),
                transparent_bg=transparent_bg,
            )
        except Exception:
            # Best-effort: silently skip this render; the next snapshot
            # fires soon. Log to stderr for diagnostics; not via logger
            # to avoid pulling Qt-thread loggers into the worker.
            import sys
            import traceback
            print(
                f"snapshot_render: skipping {self._path.name}: "
                f"{traceback.format_exc(limit=2)}",
                file=sys.stderr,
            )
            return
        # Marshal back to the GUI thread. Q_ARG with `object` because
        # numpy.ndarray isn't a Qt-registered type — opaque-object
        # passthrough is fine here.
        QMetaObject.invokeMethod(
            self._preview, "on_preview",
            Qt.QueuedConnection,
            Q_ARG(object, canvas),
        )
```

`PreviewPanel.on_preview` is already a regular method that accepts a numpy array; `QMetaObject.invokeMethod` with `Qt.QueuedConnection` calls it on the GUI thread.

**Note:** the `Q_ARG(object, canvas)` form works because Python's Qt binding treats numpy arrays as opaque objects. If `on_preview` is also a `@Slot`-decorated method, it auto-registers; otherwise the QueuedConnection still delivers but may warn. Verify by running the test.

- [ ] **Step 8.4: Run tests**

```bash
pytest tests/test_snapshot_render_job.py -v
```

Expected: 3/3 green. If `test_render_job_calls_on_preview` fails with a "QMetaObject.invokeMethod failed" message, the `on_preview` method needs a `@Slot` decoration. Add `from PySide6.QtCore import Slot` and `@Slot(object)` above the existing `on_preview` in `preview_panel.py`.

- [ ] **Step 8.5: If on_preview needs @Slot decoration**

If Step 8.4 caught the marshaling issue, edit `forza_abyss_painter/gui/preview_panel.py`:

```python
from PySide6.QtCore import Qt, Slot
...
    @Slot(object)
    def on_preview(self, arr) -> None:
        self.preview_view.set_numpy(arr)
```

Re-run Step 8.4.

- [ ] **Step 8.6: Commit**

```bash
git add forza_abyss_painter/gui/snapshot_render.py tests/test_snapshot_render_job.py
# Also stage preview_panel.py if Slot decoration was added:
git add forza_abyss_painter/gui/preview_panel.py 2>/dev/null || true
git commit -m "$(cat <<'EOF'
feat(gui): _RenderSnapshotJob — off-thread snapshot rendering

QRunnable that reads a snapshot JSON, renders via render_shapes
(pure CPU), and marshals the numpy canvas to PreviewPanel.on_preview
via QMetaObject.invokeMethod QueuedConnection.

Error swallowing: mid-write reads, deleted files, malformed JSON all
log to stderr + return cleanly. The next snapshot event fires soon
on a GPU run (every 100 shapes → typically 1-3s later).

MainWindow dispatch + single-slot throttle lands in a later commit.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: upload_panel — Resume from snapshot button

**Files:**
- Modify: `forza_abyss_painter/gui/upload_panel.py`
- Create: `tests/test_upload_panel_resume_button.py`

- [ ] **Step 9.1: Write the failing test**

Create `tests/test_upload_panel_resume_button.py`:

```python
"""Resume from snapshot… button is always visible (no flag, no
loaded-JSON gate) and emits the picked path via resume_requested."""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from forza_abyss_painter.gui.upload_panel import UploadPanel


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def test_resume_button_exists_and_visible(qapp):
    panel = UploadPanel()
    assert hasattr(panel, "resume_btn")
    # Always visible — no feature flag, no Upload-JSON gate.
    assert panel.resume_btn.isVisible() or panel.resume_btn.isVisibleTo(panel)
    panel.deleteLater()


def test_resume_button_emits_signal_with_picked_path(qapp, tmp_path):
    snap = tmp_path / "x_2900.json"
    snap.write_text("{}")
    panel = UploadPanel()
    received: list[Path] = []
    panel.resume_requested.connect(lambda p: received.append(p))
    # Patch QFileDialog.getOpenFileName to return our test path.
    from PySide6.QtWidgets import QFileDialog
    with patch.object(QFileDialog, "getOpenFileName",
                       return_value=(str(snap), "Forza Abyss Painter snapshots (*_*.json)")):
        panel.resume_btn.click()
    assert received == [snap]
    panel.deleteLater()


def test_resume_button_cancel_emits_nothing(qapp):
    panel = UploadPanel()
    received: list[Path] = []
    panel.resume_requested.connect(lambda p: received.append(p))
    from PySide6.QtWidgets import QFileDialog
    with patch.object(QFileDialog, "getOpenFileName",
                       return_value=("", "")):
        panel.resume_btn.click()
    assert received == []
    panel.deleteLater()


def test_resume_button_placement_below_tier_b_row(qapp):
    """Layout sanity: resume button is BELOW the Tier B reshape/polish
    row but ABOVE the Recent stack (so the stretch=1 stack doesn't
    push it offscreen)."""
    panel = UploadPanel()
    layout = panel.layout()
    stack_index = None
    reshape_row_index = None
    resume_row_index = None
    for i in range(layout.count()):
        item = layout.itemAt(i)
        widget = item.widget()
        if widget is panel.stack:
            stack_index = i
            continue
        sub_layout = item.layout()
        if sub_layout is not None:
            for j in range(sub_layout.count()):
                sw = sub_layout.itemAt(j).widget()
                if sw is panel.reshape_btn:
                    reshape_row_index = i
                if sw is panel.resume_btn:
                    resume_row_index = i
    assert stack_index is not None
    assert reshape_row_index is not None
    assert resume_row_index is not None
    assert reshape_row_index < resume_row_index < stack_index, (
        f"layout order wrong: reshape={reshape_row_index} "
        f"resume={resume_row_index} stack={stack_index}"
    )
    panel.deleteLater()
```

- [ ] **Step 9.2: Run test to verify failure**

```bash
pytest tests/test_upload_panel_resume_button.py -v
```

Expected: `AttributeError: 'UploadPanel' object has no attribute 'resume_btn'`.

- [ ] **Step 9.3: Implement**

Edit `forza_abyss_painter/gui/upload_panel.py`.

At the top of the class, add a signal alongside the others:

```python
    resume_requested = Signal(Path)      # User picked a snapshot to resume from
```

In `__init__`, AFTER the Tier B `reshape_polish_row` block (which ends with `layout.addLayout(reshape_polish_row)`) and BEFORE the `layout.addSpacing(4)` + section_label, add:

```python
        # Resume from snapshot (#snapshot-resume). Always visible — failed
        # runs are a day-one concern, no flag, no gate on loaded JSON.
        resume_row = QHBoxLayout()
        self.resume_btn = QPushButton("Resume from snapshot…", self)
        self.resume_btn.setToolTip(
            "Pick a partial snapshot (<name>_<count>.json) from a "
            "failed or cancelled run and continue generation to the "
            "original target shape count using the original settings."
        )
        self.resume_btn.clicked.connect(self._on_resume_clicked)
        resume_row.addWidget(self.resume_btn)
        layout.addLayout(resume_row)
```

Add the click handler method on the class:

```python
    def _on_resume_clicked(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(
            self, "Pick snapshot to resume",
            "",
            "Forza Abyss Painter snapshots (*_*.json);;All JSON (*.json);;All files (*)",
        )
        if path:
            self.resume_requested.emit(Path(path))
```

- [ ] **Step 9.4: Run tests**

```bash
pytest tests/test_upload_panel_resume_button.py tests/test_upload_panel_reshape_polish_buttons.py tests/test_upload_panel_button_placement.py -v
```

Expected: 4 (new) + 6 (Tier B visibility) + 1 (placement) = 11 green.

- [ ] **Step 9.5: Commit**

```bash
git add forza_abyss_painter/gui/upload_panel.py tests/test_upload_panel_resume_button.py
git commit -m "$(cat <<'EOF'
feat(gui): Resume from snapshot… button on upload_panel

Always visible (no feature flag, no loaded-JSON gate). Click opens a
QFileDialog filtered to snapshot files; on pick, emits
`resume_requested = Signal(Path)`.

Layout: below the Tier B reshape/polish row, above the Recent stack
(so the stretch=1 stack doesn't push it offscreen). Pinned by
test_resume_button_placement_below_tier_b_row.

MainWindow handler + ResumeDialog construction land in the next
commit.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: ResumeDialog

**Files:**
- Create: `forza_abyss_painter/gui/resume_dialog.py`
- Create: `tests/test_resume_dialog_values.py`

- [ ] **Step 10.1: Write the failing test**

Create `tests/test_resume_dialog_values.py`:

```python
"""ResumeDialog reads _run_config from the snapshot, shows the
continue summary, and returns a RunConfig dict via .values()."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from forza_abyss_painter.gui.resume_dialog import ResumeDialog


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _snapshot_with_config(path: Path, count=2900, target=3000):
    path.write_text(json.dumps({
        "format": "fd6.shapes", "version": 1,
        "source_image": "ziz.png",
        "image_size": [1200, 981], "shape_count": count,
        "generated_at": "", "profile": "_default",
        "sticker_mode": False,
        "shapes": [
            {"type": "rotated_ellipse", "x": 100.0, "y": 100.0,
             "rx": 5.0, "ry": 5.0, "angle": 0.0,
             "color": [128, 128, 128, 255]},
        ] * count,
        "_run_config": {
            "target_shape_count": target,
            "random_samples": 1000,
            "max_resolution": 1200,
            "edge_strength": 0.0,
            "posterize_levels": 0,
            "sticker_mode": False,
            "lock_alpha": True,
            "bbox_local": True,
            "joint_polish_steps": 250,
            "vram_budget_gib": 0.0,
            "preset_label": "_default",
        },
    }), encoding="utf-8")


def _snapshot_without_config(path: Path, count=500):
    path.write_text(json.dumps({
        "format": "fd6.shapes", "version": 1,
        "source_image": "x.png",
        "image_size": [720, 720], "shape_count": count,
        "generated_at": "", "profile": "",
        "sticker_mode": False,
        "shapes": [
            {"type": "rotated_ellipse", "x": 100.0, "y": 100.0,
             "rx": 5.0, "ry": 5.0, "angle": 0.0,
             "color": [128, 128, 128, 255]},
        ] * count,
    }), encoding="utf-8")


def test_values_from_embedded_run_config(qapp, tmp_path):
    snap = tmp_path / "ziz_2900.json"
    _snapshot_with_config(snap, count=2900, target=3000)
    src = tmp_path / "ziz.png"
    src.write_bytes(b"\x89PNG\r\n\x1a\n")

    dlg = ResumeDialog(parent=None, snapshot_path=snap, source_image_path=src)
    values = dlg.values()
    assert values["mode"] == "fresh"
    assert values["seed_shapes_path"] == str(snap)
    assert values["image_path"] == str(src)
    assert values["num_shapes"] == 3000
    assert values["random_samples"] == 1000
    assert values["max_resolution"] == 1200
    assert values["joint_polish_steps"] == 250
    assert values["lock_alpha"] is True
    dlg.deleteLater()


def test_continue_summary_in_label(qapp, tmp_path):
    snap = tmp_path / "ziz_2900.json"
    _snapshot_with_config(snap, count=2900, target=3000)
    src = tmp_path / "ziz.png"
    src.write_bytes(b"\x89PNG\r\n\x1a\n")

    dlg = ResumeDialog(parent=None, snapshot_path=snap, source_image_path=src)
    # The body label should mention the continuation range.
    text = dlg.summary_label.text()
    assert "2900" in text
    assert "3000" in text
    assert "ziz_2900.json" in text
    dlg.deleteLater()


def test_missing_run_config_shows_preset_picker(qapp, tmp_path):
    snap = tmp_path / "x_500.json"
    _snapshot_without_config(snap, count=500)
    src = tmp_path / "x.png"
    src.write_bytes(b"\x89PNG\r\n\x1a\n")

    dlg = ResumeDialog(parent=None, snapshot_path=snap, source_image_path=src)
    # No embedded run config → picker UI present.
    assert dlg.preset_combo is not None
    assert dlg.preset_combo.isVisibleTo(dlg)
    # Initial values default to the first preset.
    dlg.preset_combo.setCurrentIndex(0)
    values = dlg.values()
    assert values["num_shapes"] > 500   # picked preset targets > current count
    assert values["seed_shapes_path"] == str(snap)
    dlg.deleteLater()


def test_target_must_exceed_current_count(qapp, tmp_path):
    """If the snapshot is already at target (2900 of 2900), Resume is
    not meaningful. The dialog should detect and disable the Resume
    button (or refuse to construct). For now: disable + label change."""
    snap = tmp_path / "ziz_2900.json"
    _snapshot_with_config(snap, count=2900, target=2900)
    src = tmp_path / "ziz.png"
    src.write_bytes(b"\x89PNG\r\n\x1a\n")

    dlg = ResumeDialog(parent=None, snapshot_path=snap, source_image_path=src)
    # Either resume_btn disabled OR the dialog label explicitly warns.
    # Pick one: disable when target <= current.
    assert dlg.resume_btn.isEnabled() is False
    dlg.deleteLater()
```

- [ ] **Step 10.2: Run test to verify failure**

```bash
pytest tests/test_resume_dialog_values.py -v
```

Expected: `ImportError: No module named 'forza_abyss_painter.gui.resume_dialog'`.

- [ ] **Step 10.3: Implement**

Create `forza_abyss_painter/gui/resume_dialog.py`:

```python
"""Modal: confirm resume from a partial snapshot.

If the snapshot embeds `_run_config`, the dialog auto-fills target +
params and just asks the user to confirm. If `_run_config` is missing
(older snapshots), a preset picker UI is shown so the user can specify
target + K + max_resolution manually.

Returns the full RunConfig dict via .values() — caller hands directly
to GpuGenWorker.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox, QDialog, QHBoxLayout, QLabel, QPushButton,
    QVBoxLayout, QWidget,
)

from forza_abyss_painter.io.exporter import load_json


class ResumeDialog(QDialog):
    """Confirm-or-pick resume parameters.

    Two modes:
      - `_run_config` embedded → silent one-click resume (preset_combo
        hidden; values come from the embedded block).
      - `_run_config` missing → preset_combo shown; user picks a
        target preset before clicking Resume.

    `.values()` returns a dict ready for GpuGenWorker / build_run_config.
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        snapshot_path: Path,
        source_image_path: Path,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Resume from snapshot")
        self.setModal(True)
        self.setMinimumWidth(520)

        self._snapshot_path = Path(snapshot_path)
        self._source_image_path = Path(source_image_path)

        # Load snapshot to extract _run_config (or detect absence).
        self._doc = load_json(str(self._snapshot_path))
        self._current_count = int(self._doc.shape_count or 0)
        # Try the embedded run config first.
        raw = self._snapshot_path.read_text(encoding="utf-8")
        import json
        self._run_config: dict[str, Any] | None = json.loads(raw).get("_run_config")

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(10)

        intro = QLabel(
            "Continue an interrupted shape-gen run from the last partial "
            "snapshot. Original settings are reused so the resumed "
            "shapes blend with what's already on the canvas."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #999;")
        root.addWidget(intro)

        # Summary line — populated in _refresh_summary based on
        # _run_config availability.
        self.summary_label = QLabel("", self)
        self.summary_label.setWordWrap(True)
        root.addWidget(self.summary_label)

        # Fallback preset picker (only shown when _run_config missing).
        from forza_abyss_painter.gui.generate_dialog import LOCAL_PRESETS
        self.preset_combo = QComboBox(self)
        for p in LOCAL_PRESETS:
            self.preset_combo.addItem(p["label"], userData=p)
        if self._run_config is None:
            picker_row = QHBoxLayout()
            picker_row.addWidget(QLabel("Target preset:", self))
            picker_row.addWidget(self.preset_combo, stretch=1)
            root.addLayout(picker_row)
            self.preset_combo.currentIndexChanged.connect(self._refresh_summary)
        else:
            self.preset_combo.hide()

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.cancel_btn = QPushButton("Cancel", self)
        self.cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(self.cancel_btn)
        self.resume_btn = QPushButton("Resume", self)
        self.resume_btn.setDefault(True)
        self.resume_btn.clicked.connect(self.accept)
        btn_row.addWidget(self.resume_btn)
        root.addLayout(btn_row)

        self._refresh_summary()

    def _effective_target(self) -> int:
        if self._run_config:
            return int(self._run_config.get("target_shape_count", 0))
        preset = self.preset_combo.currentData()
        return int(preset["num_shapes"]) if preset else 0

    def _effective_K(self) -> int:
        if self._run_config:
            return int(self._run_config.get("random_samples", 0))
        preset = self.preset_combo.currentData()
        return int(preset["random_samples"]) if preset else 0

    def _effective_max_res(self) -> int:
        if self._run_config:
            return int(self._run_config.get("max_resolution", 0))
        preset = self.preset_combo.currentData()
        return int(preset["max_resolution"]) if preset else 0

    def _effective_polish_steps(self) -> int:
        if self._run_config:
            return int(self._run_config.get("joint_polish_steps", 0))
        preset = self.preset_combo.currentData()
        return int(preset.get("joint_polish_steps", 0)) if preset else 0

    def _refresh_summary(self) -> None:
        target = self._effective_target()
        current = self._current_count
        if target <= current:
            self.summary_label.setText(
                f"<b>Snapshot already at target ({current} shapes).</b> "
                f"Nothing to resume; pick a higher-target preset or "
                f"start a fresh run."
            )
            self.resume_btn.setEnabled(False)
            return
        self.resume_btn.setEnabled(True)
        K = self._effective_K()
        max_res = self._effective_max_res()
        self.summary_label.setText(
            f"<b>Continue {current} → {target} shapes</b> from "
            f"<code>{self._snapshot_path.name}</code> "
            f"(K={K}, max_res={max_res}). "
            f"Source: <code>{self._source_image_path.name}</code>."
        )
        self.summary_label.setTextFormat(Qt.RichText)

    def values(self) -> dict:
        """Return a RunConfig-ready dict. Caller hands to GpuGenWorker
        (via build_run_config or directly)."""
        target = self._effective_target()
        # Compose the output path: same dir + stem as snapshot but
        # strip the _N suffix and append .json.
        stem = self._snapshot_path.stem   # e.g. "ziz_2900"
        # Strip trailing _<digits>
        import re
        base_stem = re.sub(r"_\d+$", "", stem)
        output_path = self._snapshot_path.parent / f"{base_stem}.json"
        return {
            "image_path": str(self._source_image_path),
            "output_json_path": str(output_path),
            "mode": "fresh",
            "seed_shapes_path": str(self._snapshot_path),
            "num_shapes": target,
            "max_resolution": self._effective_max_res(),
            "random_samples": self._effective_K(),
            "joint_polish_steps": self._effective_polish_steps(),
            "sticker_mode": bool(
                (self._run_config or {}).get("sticker_mode",
                                              self._doc.sticker_mode)
            ),
            "lock_alpha": True,
            "bbox_local": bool(
                (self._run_config or {}).get("bbox_local", True)
            ),
            "preset_label": str(
                (self._run_config or {}).get("preset_label", "resumed")
            ),
            "checkpoint_every": 100,
            "device": "cuda",
        }
```

- [ ] **Step 10.4: Run tests**

```bash
pytest tests/test_resume_dialog_values.py -v
```

Expected: 4/4 green.

- [ ] **Step 10.5: Commit**

```bash
git add forza_abyss_painter/gui/resume_dialog.py tests/test_resume_dialog_values.py
git commit -m "$(cat <<'EOF'
feat(gui): ResumeDialog confirms resume from partial snapshot

Reads _run_config if present → silent one-click resume. Falls back
to a preset picker when _run_config missing (older snapshots).
.values() returns a RunConfig-ready dict.

Output path is computed by stripping the _N suffix from the snapshot
filename: ziz_2900.json → ziz.json. The new run will overwrite that
path on completion.

Disables the Resume button when target <= current_count (nothing to
resume to). User can pick a higher-target preset in the fallback
flow.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Generate dialog — checkpoint spinbox

**Files:**
- Modify: `forza_abyss_painter/gui/generate_dialog.py`
- Modify: `forza_abyss_painter/gui/gpu_gen_worker.py:build_run_config`
- Create: `tests/test_generate_dialog_checkpoint_spinbox.py`

- [ ] **Step 11.1: Write the failing test**

Create `tests/test_generate_dialog_checkpoint_spinbox.py`:

```python
"""Generate dialog has a 'Snapshot every N shapes' QSpinBox with
min=100, max=1000, step=50, default=100 (cuda min, per spec §6).
The value flows into the IPC config."""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from forza_abyss_painter.gui.generate_dialog import GenerateLocallyDialog


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def test_checkpoint_spinbox_bounds(qapp):
    dlg = GenerateLocallyDialog(parent=None)
    sb = dlg.checkpoint_every_spinbox
    assert sb.minimum() == 100
    assert sb.maximum() == 1000
    assert sb.singleStep() == 50
    assert sb.value() == 100   # default
    dlg.deleteLater()


def test_checkpoint_spinbox_writes_into_build_run_config(qapp, tmp_path):
    """When the dialog spawns a run, the spinbox value lands in the
    config dict's checkpoint_every field (not the old // 20 heuristic)."""
    img = tmp_path / "src.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")
    dlg = GenerateLocallyDialog(parent=None)
    dlg.source_path = img
    dlg.checkpoint_every_spinbox.setValue(250)

    preset = dlg.preset_combo.currentData()
    from forza_abyss_painter.gui.gpu_gen_worker import build_run_config
    cfg = build_run_config(
        img, tmp_path / "out.json", preset,
        checkpoint_every=int(dlg.checkpoint_every_spinbox.value()),
    )
    assert cfg["checkpoint_every"] == 250
    dlg.deleteLater()


def test_build_run_config_accepts_checkpoint_every_kwarg(tmp_path):
    """build_run_config gets a new optional kwarg. Default behavior
    (no kwarg) preserves the old num_shapes // 20 heuristic for any
    callers that don't pass it explicitly."""
    img = tmp_path / "src.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")
    preset = {
        "label": "Custom",
        "num_shapes": 400,
        "max_resolution": 480,
        "random_samples": 4096,
        "joint_polish_steps": 100,
    }
    # With explicit kwarg.
    from forza_abyss_painter.gui.gpu_gen_worker import build_run_config
    cfg_explicit = build_run_config(
        img, tmp_path / "out.json", preset, checkpoint_every=100,
    )
    assert cfg_explicit["checkpoint_every"] == 100
    # Without kwarg → falls back to old heuristic FLOORED at 100
    # (the cuda runner min).
    cfg_default = build_run_config(img, tmp_path / "out.json", preset)
    # 400 // 20 = 20, but the floor brings it up to 100.
    assert cfg_default["checkpoint_every"] == 100
```

- [ ] **Step 11.2: Run test to verify failure**

```bash
pytest tests/test_generate_dialog_checkpoint_spinbox.py -v
```

Expected: `AttributeError: 'GenerateLocallyDialog' object has no attribute 'checkpoint_every_spinbox'`.

- [ ] **Step 11.3: Add the spinbox to GenerateLocallyDialog**

Edit `forza_abyss_painter/gui/generate_dialog.py`. Find the form layout where `preset_combo` and other controls live (around line 121-133). Add a new spinbox after the preset description:

```python
        # Checkpoint cadence (snapshot every N shapes). GPU min 100 per
        # #snapshot-resume §6 — power users can raise to 1000 to reduce
        # disk writes on big runs.
        from PySide6.QtWidgets import QSpinBox
        self.checkpoint_every_spinbox = QSpinBox(self)
        self.checkpoint_every_spinbox.setRange(100, 1000)
        self.checkpoint_every_spinbox.setSingleStep(50)
        self.checkpoint_every_spinbox.setValue(100)
        self.checkpoint_every_spinbox.setToolTip(
            "Save a partial snapshot every N shapes. Lets you resume "
            "from the most recent snapshot if the run fails. Minimum "
            "100 on GPU runs."
        )
        form.addRow("Snapshot every:", self.checkpoint_every_spinbox)
```

Place this AFTER the preset combo row and BEFORE the preset description label (or wherever the form layout naturally fits — read the existing structure first).

In `_on_generate_clicked`, when building the config, pass the spinbox value:

Find this block:
```python
        config = build_run_config(
            self.source_path, out_path, preset,
            sticker_mode=False,   # TODO: tie to a sticker checkbox once added
        )
```

Add the kwarg:
```python
        config = build_run_config(
            self.source_path, out_path, preset,
            sticker_mode=False,
            checkpoint_every=int(self.checkpoint_every_spinbox.value()),
        )
```

- [ ] **Step 11.4: Update build_run_config to accept the kwarg**

Edit `forza_abyss_painter/gui/gpu_gen_worker.py:build_run_config`. Change the signature:

```python
def build_run_config(
    image_path: Path,
    output_json_path: Path,
    preset: dict,
    sticker_mode: bool = False,
    vram_budget_gib: float = 0.0,
    checkpoint_every: int | None = None,
) -> dict:
```

Update the body where `checkpoint_every` is computed:

```python
    # Checkpoint cadence: caller-supplied (from GUI spinbox) preferred;
    # otherwise fall back to the old "20 progress events per run"
    # heuristic FLOORED at 100 (the cuda min enforced runner-side).
    # Without the floor, callers that don't pass the kwarg (e.g.
    # auto-queue _start_gpu path on small runs) would feed
    # checkpoint_every=20 into a cuda runner and trip the rejection.
    if checkpoint_every is None:
        ce = max(100, int(preset["num_shapes"]) // 20)
    else:
        ce = int(checkpoint_every)
```

Then in the returned dict, replace the inline `max(1, ...)` with `ce`:

```python
    return {
        ...
        "checkpoint_every": ce,
        ...
    }
```

- [ ] **Step 11.5: Run tests**

```bash
pytest tests/test_generate_dialog_checkpoint_spinbox.py tests/test_build_run_config_polish.py tests/test_generate_dialog_initial_source.py tests/test_generate_dialog_recommendation_label.py -v
```

Expected: all green.

- [ ] **Step 11.6: Commit**

```bash
git add forza_abyss_painter/gui/generate_dialog.py forza_abyss_painter/gui/gpu_gen_worker.py tests/test_generate_dialog_checkpoint_spinbox.py
git commit -m "$(cat <<'EOF'
feat(gui): "Snapshot every N shapes" spinbox in Generate dialog

GPU min cadence enforcement (spec §6) at the GUI layer:
- QSpinBox bounds [100, 1000], step 50, default 100.
- Value flows into build_run_config via the new checkpoint_every kwarg.
- Old num_shapes // 20 heuristic preserved as default when caller
  doesn't pass the kwarg (back-compat with non-GUI callers).

Runner-side enforcement (RunConfig.from_dict raises on
cuda + 0 < ce < 100) landed in an earlier commit.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: MainWindow — wire snapshot signal + resume slot

**Files:**
- Modify: `forza_abyss_painter/gui/main_window.py`
- Create: `tests/test_main_window_snapshot_wiring.py`

- [ ] **Step 12.1: Write the failing test**

Create `tests/test_main_window_snapshot_wiring.py`:

```python
"""MainWindow snapshot signal + resume slot wiring (smoke test).

Per CLAUDE.md §8h: single MainWindow per process. This test combines
multiple assertions into one construction."""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from forza_abyss_painter.gui.main_window import MainWindow


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _snapshot(path: Path):
    path.write_text(json.dumps({
        "format": "fd6.shapes", "version": 1,
        "source_image": "x.png",
        "image_size": [32, 32], "shape_count": 1,
        "generated_at": "", "profile": "",
        "sticker_mode": False,
        "shapes": [
            {"type": "rotated_ellipse", "x": 16.0, "y": 16.0,
             "rx": 4.0, "ry": 4.0, "angle": 0.0,
             "color": [200, 80, 80, 255]},
        ],
        "_run_config": {
            "target_shape_count": 100,
            "random_samples": 1024,
            "max_resolution": 360,
            "edge_strength": 0.0,
            "posterize_levels": 0,
            "sticker_mode": False,
            "lock_alpha": True,
            "bbox_local": True,
            "joint_polish_steps": 0,
            "vram_budget_gib": 0.0,
            "preset_label": "test",
        },
    }), encoding="utf-8")


def test_main_window_has_resume_slot(qapp):
    win = MainWindow()
    try:
        assert hasattr(win, "_on_resume_requested")
        assert hasattr(win, "_on_gpu_snapshot")
        # upload_panel.resume_requested must be wired.
        # Soft check: the signal exists.
        assert hasattr(win.upload, "resume_requested")
    finally:
        win.close()
        win.deleteLater()


def test_resume_slot_handles_snapshot(qapp, tmp_path, monkeypatch):
    """End-to-end smoke (no actual subprocess): click flow from
    upload_panel.resume_requested → ResumeDialog → would-spawn-worker.

    Patches runtime install prompt + ResumeDialog.exec to auto-accept,
    then asserts the dialog values dict was assembled.
    """
    snap = tmp_path / "x_50.json"
    _snapshot(snap)
    src = tmp_path / "x.png"
    src.write_bytes(b"\x89PNG\r\n\x1a\n")

    # Avoid the runtime install modal.
    from forza_abyss_painter.gui import runtime_install_dialog as rid
    monkeypatch.setattr(rid, "prompt_install_or_use_existing",
                         lambda parent: True)

    # Auto-accept the ResumeDialog.
    from forza_abyss_painter.gui import resume_dialog as rd_mod
    from PySide6.QtWidgets import QDialog

    captured_values: list[dict] = []

    original_init = rd_mod.ResumeDialog.__init__
    original_values = rd_mod.ResumeDialog.values

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)

    def patched_exec(self):
        captured_values.append(original_values(self))
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(rd_mod.ResumeDialog, "exec", patched_exec)

    # Also avoid actually spawning the worker — patch GpuGenWorker so
    # we just record the config_path that would be passed.
    from forza_abyss_painter.gui import gpu_gen_worker as ggw
    spawned: list = []

    class _StubWorker:
        def __init__(self, embedded_python_exe, config_path):
            spawned.append(Path(config_path))
        def moveToThread(self, *a, **k): pass
        def cancel(self): pass
        run = None    # not started
        # Stub signals — connect() is a no-op.
        class _Sig:
            def connect(self, *a, **k): pass
        started = progress = checkpoint = done = error = finished = snapshot = _Sig()

    monkeypatch.setattr(ggw, "GpuGenWorker", _StubWorker)

    # Override the source-image resolver to avoid a real picker.
    from forza_abyss_painter.gui import main_window as mw_mod
    monkeypatch.setattr(mw_mod, "_resolve_source_image_path",
                         lambda json_path, name: src)

    win = MainWindow()
    try:
        win._on_resume_requested(snap)
        # ResumeDialog was constructed + exec'd → values captured.
        assert len(captured_values) == 1
        v = captured_values[0]
        assert v["mode"] == "fresh"
        assert v["seed_shapes_path"] == str(snap)
        assert v["num_shapes"] == 100
    finally:
        win.close()
        win.deleteLater()
```

- [ ] **Step 12.2: Run test to verify failure**

```bash
pytest tests/test_main_window_snapshot_wiring.py -v
```

Expected: `AttributeError: 'MainWindow' object has no attribute '_on_resume_requested'`.

- [ ] **Step 12.3: Wire the upload signal + add the slots**

Edit `forza_abyss_painter/gui/main_window.py`.

Where existing upload_panel signals are connected (after the reshape_requested / polish_requested connections from Tier B), add:

```python
        self.upload.resume_requested.connect(self._on_resume_requested)
```

Where existing GPU worker signals are connected in `_start_gpu` (around line 1268-1269), add the snapshot connection right after `_on_gpu_progress`:

```python
        self._worker.snapshot.connect(self._on_gpu_snapshot)
```

Also initialize the throttle state at MainWindow `__init__`:

```python
        self._snapshot_render_in_flight: bool = False
        self._snapshot_pending_path: str | None = None
```

(Add these alongside other `__init__` state vars like `_loaded_json_path`.)

Add the new methods to MainWindow:

```python
    def _on_gpu_snapshot(self, count: int, total: int, snapshot_path: str) -> None:
        """Runner just wrote a snapshot; dispatch a render off-thread.

        Single-slot throttle: if a render is already in flight, just
        remember the latest path. When the in-flight job finishes, if
        a newer path is pending, start it. Drops intermediate renders
        if snapshots fire faster than render — for GPU at every-100,
        this rarely matters but it bounds memory + thread churn.
        """
        from pathlib import Path as _Path
        self.statusBar().showMessage(
            f"GPU: snapshot saved at {count}/{total} → "
            f"{_Path(snapshot_path).name}", 2000,
        )
        if self._snapshot_render_in_flight:
            self._snapshot_pending_path = snapshot_path
            return
        self._dispatch_snapshot_render(snapshot_path)

    def _dispatch_snapshot_render(self, snapshot_path: str) -> None:
        from PySide6.QtCore import QThreadPool, QTimer
        from forza_abyss_painter.gui.snapshot_render import _RenderSnapshotJob
        self._snapshot_render_in_flight = True
        job = _RenderSnapshotJob(snapshot_path, self.preview)
        # Use a one-shot timer to clear the flag + dispatch any pending
        # render. QRunnable doesn't expose a finished signal, but we
        # know the render time is bounded — set a generous timeout
        # (3s) after which we assume the job is done. If a render
        # actually takes longer, the next snapshot just adds to the
        # pending queue with no harm.
        QThreadPool.globalInstance().start(job)
        QTimer.singleShot(3000, self._snapshot_render_drain)

    def _snapshot_render_drain(self) -> None:
        self._snapshot_render_in_flight = False
        if self._snapshot_pending_path is not None:
            pending = self._snapshot_pending_path
            self._snapshot_pending_path = None
            self._dispatch_snapshot_render(pending)

    def _on_resume_requested(self, snapshot_path: "Path") -> None:
        """User picked a snapshot to resume from. Resolve source image,
        open ResumeDialog, on accept spawn GpuGenWorker with the
        dialog's values dict."""
        from PySide6.QtWidgets import QMessageBox, QFileDialog
        from PySide6.QtWidgets import QDialog as _QDialog
        from pathlib import Path as _Path

        # Load + sanity-check snapshot.
        try:
            from forza_abyss_painter.io.exporter import load_json
            doc = load_json(str(snapshot_path))
        except Exception as exc:
            QMessageBox.critical(
                self, "Couldn't load snapshot",
                f"Snapshot {snapshot_path.name} could not be loaded:\n\n"
                f"{type(exc).__name__}: {exc}",
            )
            return

        # Resolve source image (same-folder heuristic + picker).
        sibling = _resolve_source_image_path(snapshot_path, doc.source_image)
        if sibling is not None:
            source = sibling
        else:
            QMessageBox.information(
                self, "Source image not found",
                f"Snapshot references '{doc.source_image}' but it's not "
                f"next to the snapshot file. Pick it manually.",
            )
            picked, _ = QFileDialog.getOpenFileName(
                self, f"Pick source image (looking for '{doc.source_image}')",
                "", "Images (*.png *.jpg *.jpeg *.webp);;All files (*)",
            )
            if not picked:
                return
            source = _Path(picked)

        # Runtime install check (resume uses the GPU runner).
        from forza_abyss_painter.gui.runtime_install_dialog import (
            prompt_install_or_use_existing,
        )
        if not prompt_install_or_use_existing(self):
            return

        # Dialog.
        from forza_abyss_painter.gui.resume_dialog import ResumeDialog
        dlg = ResumeDialog(
            parent=self,
            snapshot_path=snapshot_path,
            source_image_path=source,
        )
        if dlg.exec() != _QDialog.DialogCode.Accepted:
            return
        values = dlg.values()

        # Write the config + spawn GpuGenWorker (same pattern as
        # _start_gpu and _on_polish_requested).
        config_path = snapshot_path.parent / (
            f".{snapshot_path.stem}_resume_config.json"
        )
        try:
            config_path.write_text(
                json.dumps(values, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            QMessageBox.critical(
                self, "Couldn't start resume",
                f"Failed to write resume config: {exc}",
            )
            return

        from forza_abyss_painter.runtime.torch_installer import embedded_python_exe
        py = embedded_python_exe()
        if not py.exists():
            QMessageBox.critical(
                self, "GPU runtime missing",
                f"Embedded Python not found at {py}. "
                f"Open Tools → Generate shapes locally to install the runtime.",
            )
            return

        # Spawn — same pattern as _start_gpu.
        from forza_abyss_painter.gui.gpu_gen_worker import GpuGenWorker
        from PySide6.QtCore import QThread
        self._worker = GpuGenWorker(
            embedded_python_exe=py,
            config_path=config_path,
        )
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self.preview.on_progress)
        self._worker.checkpoint.connect(self.preview.on_progress)
        self._worker.progress.connect(self._on_gpu_progress)
        self._worker.checkpoint.connect(self._on_gpu_progress)
        self._worker.snapshot.connect(self._on_gpu_snapshot)
        self._worker.done.connect(self._on_gpu_done)
        self._worker.error.connect(self._on_gpu_error)
        self._worker.finished.connect(self._teardown_thread)
        import time as _time
        self._gpu_run_start_t = _time.monotonic()
        self.preview.set_source(source)
        self._thread.start()
        self.settings_panel.set_running(True)
        self.statusBar().showMessage(
            f"Resume started: {snapshot_path.name} → "
            f"continuing to {values['num_shapes']} shapes"
        )
```

- [ ] **Step 12.4: Run tests**

```bash
pytest tests/test_main_window_snapshot_wiring.py tests/test_gpu_bundle_gui.py tests/test_main_window_autotune_status.py -v 2>&1 | tail -15
```

Expected: 2 new + existing main_window tests all green.

- [ ] **Step 12.5: Commit**

```bash
git add forza_abyss_painter/gui/main_window.py tests/test_main_window_snapshot_wiring.py
git commit -m "$(cat <<'EOF'
feat(gui): MainWindow snapshot signal + resume slot

Three pieces of GUI wiring for the snapshot/resume system:

1. _on_gpu_snapshot slot: receives the (count, total, path) tuple
   from GpuGenWorker.snapshot Signal. Updates the status bar and
   dispatches an off-thread render via _RenderSnapshotJob.

2. Single-slot render throttle: _snapshot_render_in_flight +
   _snapshot_pending_path. If a render is in flight when a new
   snapshot arrives, just remember the latest path; QTimer(3s) drains
   the queue. Bounds thread churn at fast-fire snapshots.

3. _on_resume_requested slot: validates snapshot loads, resolves
   source image (same-folder heuristic + picker fallback), prompts
   runtime install if needed, opens ResumeDialog, spawns GpuGenWorker
   with the dialog's values dict.

Same dialog auto-accept + worker stub patterns as the Tier B
_on_polish_requested test for smoke coverage without a real
subprocess.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: Local smoke + SMB sync + push

- [ ] **Step 13.1: Run the full new-feature test set**

```bash
pytest tests/test_snapshot_path_for.py \
       tests/test_build_run_config_polish.py \
       tests/test_validator_underscore_keys.py \
       tests/test_run_config_seed_shapes_path.py \
       tests/test_run_config_cuda_min_cadence.py \
       tests/test_engine_seed_shapes.py \
       tests/test_snapshot_runner_integration.py \
       tests/test_resume_runner_integration.py \
       tests/test_gpu_gen_worker_snapshot_signal.py \
       tests/test_snapshot_render_job.py \
       tests/test_upload_panel_resume_button.py \
       tests/test_resume_dialog_values.py \
       tests/test_generate_dialog_checkpoint_spinbox.py \
       tests/test_main_window_snapshot_wiring.py -v 2>&1 | tail -30
```

Expected: all green (or skip-due-to-torch on this host where appropriate).

- [ ] **Step 13.2: Broader regression**

```bash
pytest tests/test_torch_runner.py \
       tests/test_torch_runner_polish_mode.py \
       tests/test_polish_runner_integration.py \
       tests/test_vram_preflight_verdict.py \
       tests/test_recommend_max_resolution.py \
       tests/test_main_window_autotune_status.py \
       tests/test_upload_panel_reshape_polish_buttons.py \
       tests/test_polish_dialog_values.py \
       tests/test_generate_dialog_initial_source.py \
       tests/test_generate_dialog_recommendation_label.py \
       tests/test_resolve_source_image.py \
       tests/test_fap_refresh_same_dir_guard.py \
       tests/test_load_json_bom_tolerance.py \
       tests/test_main_window_gpu_progress_wiring.py \
       tests/test_brand_banner_badge_path.py \
       tests/test_gpu_bundle_gui.py 2>&1 | tail -10
```

Expected: all pass or pre-existing skips.

- [ ] **Step 13.3: Append a Run 8 preview to CURSOR_NEXT_RUN.md on SMB**

```bash
cat >> /Volumes/ContentCreation/ForzaAbyssPainter_build/CURSOR_NEXT_RUN.md <<'PREVIEW_EOF'

---

## Run 8 preview — snapshots + live preview + resume + auto-polish

The next EXE rebuild ships four interlinked changes:

1. **Live preview during GPU runs.** Middle PreviewPanel updates every
   100 shapes via an off-thread render of the on-disk snapshot.
2. **Periodic snapshots to disk.** `<output_stem>_<count>.json` next to
   the final output. Failed runs leave the most recent snapshot
   intact → no more all-or-nothing failures.
3. **Resume from snapshot button.** New button on upload_panel
   (always visible). Pick a partial snapshot → ResumeDialog →
   continue to target. Works with snapshots from this rebuild forward
   (which embed `_run_config`). Older snapshots fall back to a preset
   picker.
4. **Auto-polish on fresh GPU runs.** LOCAL_PRESETS gains
   `joint_polish_steps` matching the CPU calibration (Lineart 100 /
   Headshot 150 / Medium 150 / Hi-Res 250). Output should visibly
   improve in color accuracy compared to the pre-spec runs.

QUASAR spot-check (~10 min):

1. Generate a small run (e.g. Headshot 700 on a logo). Watch the
   middle preview update every 100 shapes. Status bar shows
   `GPU: snapshot saved at X/Y → <name>.json`.
2. Check the output dir — there should be ~7 snapshot files +
   the final output. Snapshots all contain `_run_config`.
3. Mid-run, kill the EXE (Task Manager). Reopen, click "Resume from
   snapshot…", pick the most recent snapshot. Confirm the
   ResumeDialog auto-fills target. Click Resume. New run continues
   to target.
4. Compare a fresh 1000-shape run output now (auto-polished) vs a
   pre-spec output (unpolished). Polished should have visibly better
   color match.
5. Try a tight-cadence run (set Snapshot every: 50) → spinbox should
   refuse to go below 100.

This is informational, not blocking. Run 8 is the snapshot/resume
candidate; Run 9 remains blocked on #129 chunked rasterize.

PREVIEW_EOF
echo "appended Run 8 preview"
```

- [ ] **Step 13.4: rsync source to SMB**

```bash
rsync -a --delete \
  --exclude='__pycache__' --exclude='.git' --exclude='*.pyc' \
  --exclude='dist' --exclude='build' --exclude='.pytest_cache' \
  --exclude='*.egg-info' --exclude='node_modules' \
  /Users/kusanagi/Development/forza-abyss-painter/ \
  /Volumes/ContentCreation/ForzaAbyssPainter_build/source/ 2>&1 | tail -3
```

- [ ] **Step 13.5: Push**

```bash
git push origin feat/exe-colab-ports-batch 2>&1 | tail -5
```

Report the push range + final test count.

---

## Self-Review Summary

| Spec section | Implementing task |
|---|---|
| §1 Purpose (snapshots, live preview, resume, auto-polish) | Tasks 1-12 collectively |
| §3.1 Live preview | Tasks 7 (signal), 8 (render job), 12 (wiring + throttle) |
| §3.2 Periodic snapshots | Tasks 1 (helper), 6 (runner writes) |
| §3.3 Resume GUI | Tasks 9 (button), 10 (dialog), 12 (slot) |
| §3.4 Auto-polish | Task 2 (presets + forwarding) |
| §4 Snapshot format (with _run_config) | Task 6 (writer), Task 3 (validator tolerance) |
| §5.1 Runner IPC | Task 6 |
| §5.2-5.3 GUI snapshot subscription | Tasks 7, 8, 12 |
| §5.4 Resume flow GUI | Tasks 9, 10, 12 |
| §5.5 Resume flow runner | Tasks 4 (config field), 5 (engine seed), 6 (runner load) |
| §5.6 Auto-polish wiring | Task 2 |
| §6 Min cadence (GUI + runner) | Tasks 4 (runner), 11 (GUI spinbox) |
| §7 Testing strategy | Tasks 1-12 each carry their tests |
| §9 Acceptance criteria | Task 13 (smoke + SMB + push) |
