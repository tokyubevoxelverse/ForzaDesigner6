# PR A — VRAM Honesty Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the VRAM-estimate gap that let Re-shape-gen → Hi-Res 3000 → OOM on a 32 GiB card. Today's `estimate_peak_vram_gib` only models the K-scorer batch (~12-18 GiB). Run 4 measured 47.5 GiB full pipeline; the QUASAR 2026-05-27 reshape hit 53.7 GiB. The fix is honest preflight + honest UI labels.

**Architecture:**
- New constant `PIPELINE_OVERHEAD_GIB = 35.0` in `vram_planner.py`, calibrated from Run 4/6 evidence.
- New function `estimate_full_pipeline_gib(K, bbox_local, max_resolution, ...)` = `estimate_peak_vram_gib(...)` + `PIPELINE_OVERHEAD_GIB`. Keeps the existing K-only function for chunking math (unchanged semantics).
- Preflight (`_start_gpu`) + UI labels (Generate dialog, SettingsPanel) all switch to the new full-pipeline estimate.
- Drop the static `est_peak_vram_gib` hand-tuned constants from `LOCAL_PRESETS` — replace with live computation.
- OOM error message in `torch_runner` adds "restart the EXE to release CUDA cache".

**Tech Stack:** Pure Python (vram_planner has no torch dep), PySide6 for the GUI label changes, pytest for the calibration regression.

**Reference:** Cursor's `\\QUASAR\ContentCreation\ForzaAbyssPainter_build\diagnostics\UPSTREAM_MANUAL_TEST_FINDINGS_20260527.md` §3.

---

## File Structure

### Modified files

| Path | Change |
|---|---|
| `forza_abyss_painter/shapegen/gpu/vram_planner.py` | Add `PIPELINE_OVERHEAD_GIB` constant + `estimate_full_pipeline_gib()` function. Keep `estimate_peak_vram_gib()` unchanged. |
| `forza_abyss_painter/gui/settings_panel.py:320-335` | `estimate_peak_vram_gib(profile)` method now calls the full-pipeline estimate instead of the K-only one. |
| `forza_abyss_painter/gui/generate_dialog.py` | Drop static `est_peak_vram_gib` from LOCAL_PRESETS. Preset combo label + description use live `estimate_full_pipeline_gib`. |
| `forza_abyss_painter/gui/main_window.py:1155-1163` | Preflight `peak_gib` computed via `estimate_full_pipeline_gib`. |
| `forza_abyss_painter/runtime/torch_runner.py` | OOM error in `engine_run` stage appends "Try restarting the EXE to release CUDA cache." |

### New tests

| Path | Covers |
|---|---|
| `tests/test_estimate_full_pipeline_gib.py` | Function returns `estimate_peak_vram_gib + 35`. Run-4 scenario (K=8192, max_res=720, bbox_local) returns ≥40 GiB. Hi-Res 3000 scenario (K=12288, max_res=1000) returns ≥165 GiB on bbox_local. |
| `tests/test_vram_preflight_uses_full_pipeline.py` | Pin Run-4 scenario blocks at preflight via `_vram_preflight_verdict("block")`. Pin Hi-Res on 22 GiB free blocks. |
| `tests/test_generate_dialog_live_peak_label.py` | Preset combo label format string contains a `:.1f` substitution — not the hardcoded `12.0` etc. |

### Deliberately untouched

- `estimate_peak_vram_gib` semantics — `resolve_k_chunk_size` still uses it for chunking math (K-batch budget specifically).
- `BBOX_LOCAL_SAFETY = 15.0` — already calibrated against PyTorch's reported peak. The 35 GiB overhead is additive on top of that.
- `_vram_preflight_verdict` logic — unchanged; the threshold (0.85 * free, peak > free) still applies, just with the better estimate.

---

## Task 1: `estimate_full_pipeline_gib` + constant

**Files:**
- Modify: `forza_abyss_painter/shapegen/gpu/vram_planner.py` (append after `estimate_peak_vram_gib`)
- Create: `tests/test_estimate_full_pipeline_gib.py`

- [ ] **Step 1.1: Write the failing test**

Create `tests/test_estimate_full_pipeline_gib.py`:

```python
"""estimate_full_pipeline_gib returns K-scorer estimate + calibrated
overhead. Calibration from Run 4 (47.5 GiB measured) + QUASAR
2026-05-27 (53.7 GiB measured at Hi-Res 3000 on 32 GiB card)."""
from __future__ import annotations

from forza_abyss_painter.shapegen.gpu.vram_planner import (
    estimate_peak_vram_gib,
    estimate_full_pipeline_gib,
    PIPELINE_OVERHEAD_GIB,
)


def test_constant_is_35_gib():
    assert PIPELINE_OVERHEAD_GIB == 35.0


def test_full_pipeline_adds_overhead_to_k_estimate():
    """The full estimate is the K-scorer batch peak PLUS a fixed
    overhead for canvas + refill + joint_polish + allocator
    fragmentation. Calibrated from Run 4/6 evidence."""
    k_only = estimate_peak_vram_gib(
        K=8192, bbox_local=True, max_resolution=720,
    )
    full = estimate_full_pipeline_gib(
        K=8192, bbox_local=True, max_resolution=720,
    )
    assert full == k_only + PIPELINE_OVERHEAD_GIB


def test_run_4_scenario_estimates_at_least_40_gib():
    """Run 4: K=8192, max_resolution=720, bbox_local. Measured 47.5
    GiB allocated on a 32 GiB card. New estimate must be at least
    40 GiB — well into 'won't fit' territory."""
    est = estimate_full_pipeline_gib(
        K=8192, bbox_local=True, max_resolution=720,
    )
    assert est >= 40.0, (
        f"Run-4 scenario estimate {est:.1f} GiB is below the 40 GiB "
        f"floor; calibration insufficient to block this OOM scenario."
    )


def test_hi_res_3000_scenario_estimates_at_least_50_gib():
    """QUASAR 2026-05-27 manual test: Hi-Res preset (K=12288,
    max_resolution=1000, bbox_local). Measured 53.7 GiB at OOM on
    32 GiB card. New estimate must be at least 50 GiB."""
    est = estimate_full_pipeline_gib(
        K=12288, bbox_local=True, max_resolution=1000,
    )
    assert est >= 50.0, (
        f"Hi-Res 3000 scenario estimate {est:.1f} GiB is below the "
        f"50 GiB floor; would not block the QUASAR OOM."
    )


def test_small_run_includes_overhead_too():
    """Even a small run (K=1024, max_res=480) has the pipeline
    overhead — polish + canvas + refill don't scale to zero."""
    est = estimate_full_pipeline_gib(
        K=1024, bbox_local=True, max_resolution=480,
    )
    # Small K, small canvas → K-only is ~1-3 GiB; + 35 = 36-38 GiB.
    assert est >= 35.0
    assert est <= 40.0   # not absurdly larger


def test_zero_k_returns_overhead_only():
    """Edge case: K=0 → no K-scorer batch, just the fixed overhead."""
    est = estimate_full_pipeline_gib(
        K=0, bbox_local=True, max_resolution=480,
    )
    assert est == PIPELINE_OVERHEAD_GIB
```

- [ ] **Step 1.2: Run test to verify it fails**

`pytest tests/test_estimate_full_pipeline_gib.py -v`

Expected: `ImportError: cannot import name 'estimate_full_pipeline_gib'`.

- [ ] **Step 1.3: Implement**

Append to `forza_abyss_painter/shapegen/gpu/vram_planner.py` (after `recommend_max_resolution`):

```python
# Fixed pipeline overhead — calibrated from real-run evidence:
#   Run 4 (2026-05-26): K=8192, max_res=720, bbox_local
#     K-only estimate ≈ 12.5 GiB; measured allocated 47.5 GiB → +35 GiB overhead.
#   QUASAR 2026-05-27 (manual): K=12288, max_res=1000, bbox_local, Hi-Res 3000 preset
#     K-only estimate ≈ 18 GiB; measured 53.7 GiB at OOM → +35.7 GiB overhead.
#
# 35 GiB covers:
#   - Full canvas tensors (target, alpha_t, edge_weight, canvas_init) at max_res²
#   - Per-shape state list (3000 shapes × ~200 bytes each)
#   - clean_and_refill rasterization pass at end of run
#   - joint_polish (Adam optimizer with rgb + alpha + geom gradients)
#   - PyTorch caching allocator fragmentation (even with
#     expandable_segments:True the peak doesn't fully collapse)
#
# This is a defensive overestimate. The cost of over-warning users to
# close FH6 is small; the cost of letting an OOM through (Run 4 + this
# session) is high — wasted time + scary error modal.
#
# When future engine refactors (#129 chunked rasterize, mid-run refill)
# change the overhead profile, recalibrate here.
PIPELINE_OVERHEAD_GIB = 35.0


def estimate_full_pipeline_gib(
    K: int,
    bbox_local: bool,
    max_resolution: int,
    bbox_crop_max: int = 256,
) -> float:
    """Predicted peak VRAM in GiB for the FULL shape-gen pipeline:
    K-scorer batch + canvas + refill + joint_polish + allocator
    fragmentation.

    Use this for "will this run OOM" decisions in the preflight and
    in any UI label that claims to show peak VRAM. The K-only
    `estimate_peak_vram_gib` is kept for chunking math
    (resolve_k_chunk_size) — that function answers a different
    question ("how big is the K-batch alone for chunk sizing") and
    must stay K-only.

    See PIPELINE_OVERHEAD_GIB docstring for calibration evidence.
    """
    k_only = estimate_peak_vram_gib(
        K=K, bbox_local=bbox_local,
        max_resolution=max_resolution,
        bbox_crop_max=bbox_crop_max,
    )
    return k_only + PIPELINE_OVERHEAD_GIB
```

- [ ] **Step 1.4: Run tests**

`pytest tests/test_estimate_full_pipeline_gib.py tests/test_recommend_max_resolution.py -v`

Expected: 6 new pass; existing recommend_max_resolution tests still pass (no API change to `estimate_peak_vram_gib`).

- [ ] **Step 1.5: Commit**

```bash
git add forza_abyss_painter/shapegen/gpu/vram_planner.py tests/test_estimate_full_pipeline_gib.py
git commit -m "$(cat <<'EOF'
feat(vram): estimate_full_pipeline_gib + 35 GiB overhead constant

Cursor's QUASAR 2026-05-27 manual test caught an OOM at 53.7 GiB
peak on a 32 GiB card during Re-shape-gen → Hi-Res 3000. UI showed
~12-18 GiB; the pre-Start hard-block let it through because
estimate_peak_vram_gib is K-scorer-only and doesn't include canvas,
refill, joint_polish, or allocator fragmentation.

New estimate_full_pipeline_gib(K, ...) = estimate_peak_vram_gib + 35.
PIPELINE_OVERHEAD_GIB = 35 calibrated from:
- Run 4 (K=8192/720): K-only ~12.5 GiB, measured 47.5 GiB → +35
- QUASAR 2026-05-27 (K=12288/1000): K-only ~18, OOM @ 53.7 → +35.7

estimate_peak_vram_gib unchanged — resolve_k_chunk_size still needs
the K-only value for chunking math. New function is for "will this
run OOM" decisions (preflight, UI labels).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Preflight + SettingsPanel use full-pipeline estimate

**Files:**
- Modify: `forza_abyss_painter/gui/settings_panel.py:320-335` (`estimate_peak_vram_gib` method)
- Modify: `forza_abyss_painter/gui/main_window.py:1155-1163` (preflight in `_start_gpu`)
- Create: `tests/test_vram_preflight_uses_full_pipeline.py`

- [ ] **Step 2.1: Write the failing test**

Create `tests/test_vram_preflight_uses_full_pipeline.py`:

```python
"""Preflight (_vram_preflight_verdict) must catch the Run-4-class
OOM scenario when given the full-pipeline estimate.

This test doesn't construct MainWindow — it directly exercises
estimate_full_pipeline_gib + _vram_preflight_verdict to pin the
math. The GUI wiring is verified in the next task's smoke."""
from __future__ import annotations

from forza_abyss_painter.shapegen.gpu.vram_planner import (
    estimate_full_pipeline_gib,
)
from forza_abyss_painter.gui.main_window import _vram_preflight_verdict


def test_run_4_scenario_blocks_at_32_gib_card():
    """Run 4: K=8192, max_res=720, ~22 GiB free on 32G card.
    Full-pipeline estimate must trigger 'block' verdict."""
    peak = estimate_full_pipeline_gib(K=8192, bbox_local=True, max_resolution=720)
    severity, _msg = _vram_preflight_verdict(
        peak_gib=peak, free_gib=22.0, budget_gib=8.0,
    )
    assert severity == "block", (
        f"Run-4 scenario (peak {peak:.1f} GiB vs 22 free) should block; "
        f"got {severity}"
    )


def test_hi_res_3000_scenario_blocks():
    """QUASAR 2026-05-27: K=12288, max_res=1000, FH6 not open
    (~27 GiB free on 32G). Should block — measured OOM at 53.7 GiB."""
    peak = estimate_full_pipeline_gib(K=12288, bbox_local=True, max_resolution=1000)
    severity, _msg = _vram_preflight_verdict(
        peak_gib=peak, free_gib=27.0, budget_gib=8.0,
    )
    assert severity == "block"


def test_small_run_on_big_card_passes():
    """K=1024, max_res=480 (Lineart 400 preset) on a 95 GiB workstation
    card: full estimate ~37 GiB, plenty of headroom. Should NOT block."""
    peak = estimate_full_pipeline_gib(K=1024, bbox_local=True, max_resolution=480)
    severity, _msg = _vram_preflight_verdict(
        peak_gib=peak, free_gib=95.0, budget_gib=32.0,
    )
    assert severity == "ok"


def test_medium_on_32g_with_fh6_closed_passes_or_warns():
    """Medium 1000 (K=8192, max_res=720) on RTX 5090 with FH6 closed
    (~27 GiB free): full estimate ~47 GiB. 27 < 47 → should block."""
    peak = estimate_full_pipeline_gib(K=8192, bbox_local=True, max_resolution=720)
    severity, _msg = _vram_preflight_verdict(
        peak_gib=peak, free_gib=27.0, budget_gib=8.0,
    )
    # Run 4 evidence: this scenario DID OOM. Block is the right call.
    assert severity == "block"


def test_settings_panel_estimate_returns_full_pipeline():
    """SettingsPanel.estimate_peak_vram_gib(profile) must return the
    full-pipeline number, not the K-only one (UI consumers don't have
    a reason to see K-only)."""
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from forza_abyss_painter.gui.settings_panel import SettingsPanel

    panel = SettingsPanel()
    # Build a fake profile object with the attributes the method
    # reads. SettingsPanel's method just calls vram_planner — we just
    # need a Profile-like duck.
    class _Profile:
        random_samples = 8192
        max_resolution = 720
    profile = _Profile()
    est = panel.estimate_peak_vram_gib(profile)
    # Same scenario as Run 4. Full-pipeline estimate ≥ 40 GiB.
    assert est >= 40.0
    panel.deleteLater()
```

- [ ] **Step 2.2: Run test to verify failure**

`pytest tests/test_vram_preflight_uses_full_pipeline.py -v`

Expected: failures on the block-verdict tests (the preflight uses K-only estimate which doesn't trigger block).

- [ ] **Step 2.3: Update SettingsPanel + _start_gpu**

Edit `forza_abyss_painter/gui/settings_panel.py:320-335`. Change the import + call:

```python
    def estimate_peak_vram_gib(self, profile) -> float:
        """Single source of truth in `vram_planner.estimate_full_pipeline_gib`.
        Returns the full-pipeline VRAM estimate (K-scorer + canvas +
        refill + joint_polish + allocator overhead). Use this for
        UI labels + preflight decisions, NOT for chunking math."""
        from forza_abyss_painter.shapegen.gpu.vram_planner import (
            estimate_full_pipeline_gib,
        )
        return estimate_full_pipeline_gib(
            K=int(profile.random_samples),
            bbox_local=True,
            max_resolution=int(profile.max_resolution),
        )
```

(Keep the method name `estimate_peak_vram_gib` — many callers use it. The semantic just shifts from K-only to full-pipeline.)

Edit `forza_abyss_painter/gui/main_window.py` around lines 1155-1163. Find the existing block (search `grep -n "estimate_peak_vram_gib" forza_abyss_painter/gui/main_window.py`):

```python
        from forza_abyss_painter.shapegen.gpu.vram_planner import (
            estimate_peak_vram_gib,
        )
        peak_gib = estimate_peak_vram_gib(
```

Change to:

```python
        from forza_abyss_painter.shapegen.gpu.vram_planner import (
            estimate_full_pipeline_gib,
        )
        peak_gib = estimate_full_pipeline_gib(
```

(All kwargs stay the same since both functions take `K`, `bbox_local`, `max_resolution`, `bbox_crop_max`.)

- [ ] **Step 2.4: Run tests**

```bash
pytest tests/test_vram_preflight_uses_full_pipeline.py tests/test_vram_preflight_verdict.py tests/test_main_window_autotune_status.py -v
```

Expected: 5 new + existing preflight tests all pass.

- [ ] **Step 2.5: Commit**

```bash
git add forza_abyss_painter/gui/settings_panel.py forza_abyss_painter/gui/main_window.py tests/test_vram_preflight_uses_full_pipeline.py
git commit -m "$(cat <<'EOF'
fix(gui): preflight + SettingsPanel use full-pipeline estimate

Switches the "will this run OOM" decision paths from K-only
estimate_peak_vram_gib to the new estimate_full_pipeline_gib.

- main_window._start_gpu preflight now computes peak_gib via the
  full-pipeline function. The _vram_preflight_verdict thresholds
  (block @ peak > free, warn @ 0.85*free < peak ≤ free) are
  unchanged — only the input estimate changed.
- SettingsPanel.estimate_peak_vram_gib(profile) now returns the
  full-pipeline number (same method name; semantic shift). UI
  consumers + Run 4 / QUASAR Hi-Res 3000 scenarios block at
  preflight as intended.

The K-only estimate_peak_vram_gib stays — resolve_k_chunk_size
needs it for chunking math, which is a different question.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Generate dialog uses live estimate

**Files:**
- Modify: `forza_abyss_painter/gui/generate_dialog.py` (drop static `est_peak_vram_gib` reads; use live function)
- Create: `tests/test_generate_dialog_live_peak_label.py`

- [ ] **Step 3.1: Write the failing test**

Create `tests/test_generate_dialog_live_peak_label.py`:

```python
"""GenerateLocallyDialog preset combo + description show LIVE
estimate_full_pipeline_gib, not the hardcoded est_peak_vram_gib
marketing number per preset."""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from forza_abyss_painter.gui.generate_dialog import (
    GenerateLocallyDialog, LOCAL_PRESETS,
)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def test_preset_combo_label_shows_live_estimate(qapp, monkeypatch):
    """Preset combo label format includes a peak-VRAM number derived
    from estimate_full_pipeline_gib(K, max_res), NOT the static
    est_peak_vram_gib marketing value (12.0 etc.)."""
    # Patch probe so the test is reproducible.
    from forza_abyss_painter.runtime import nvidia_smi as ns_mod
    def fake_probe(*, force=False):
        return ns_mod.ProbeResult(
            available=False, reason="test", probed_at=0.0,
        )
    monkeypatch.setattr(
        "forza_abyss_painter.gui.generate_dialog.probe_free_vram",
        fake_probe,
    )

    dlg = GenerateLocallyDialog(parent=None)
    # The Hi-Res preset's static est_peak_vram_gib is 12.0 — but the
    # live full-pipeline estimate is ~165 GiB. Confirm the combo
    # label is NOT 12.0.
    hi_res_idx = next(
        i for i, p in enumerate(LOCAL_PRESETS) if "Hi-Res" in p["label"]
    )
    dlg.preset_combo.setCurrentIndex(hi_res_idx)
    label = dlg.preset_combo.itemText(hi_res_idx)
    # Old format: "Hi-Res — 3000 shapes  (~12.0 GiB peak)"
    # New format: "Hi-Res — 3000 shapes  (~165 GiB peak)" or similar
    assert "12.0 GiB" not in label, (
        f"Preset label still shows static 12.0 GiB: {label!r}"
    )
    dlg.deleteLater()
```

- [ ] **Step 3.2: Run test to verify failure**

`pytest tests/test_generate_dialog_live_peak_label.py -v`

Expected: FAIL — label still contains `12.0 GiB`.

- [ ] **Step 3.3: Update generate_dialog.py**

Edit `forza_abyss_painter/gui/generate_dialog.py`.

**3.3a:** The static `est_peak_vram_gib` field in `LOCAL_PRESETS` is still useful as a docstring hint but should NOT drive UI labels. Keep the field for now (the LOCAL_PRESETS structure has other consumers) but stop reading it for display purposes.

**3.3b:** Find the preset combo loader (line ~127-132 — `grep -n "addItem" forza_abyss_painter/gui/generate_dialog.py`). Currently:

```python
        for p in LOCAL_PRESETS:
            self.preset_combo.addItem(
                f"{p['label']}  (~{p['est_peak_vram_gib']:.1f} GiB peak)",
                userData=p,
            )
```

Change to use the live estimate:

```python
        from forza_abyss_painter.shapegen.gpu.vram_planner import (
            estimate_full_pipeline_gib,
        )
        for p in LOCAL_PRESETS:
            live_peak = estimate_full_pipeline_gib(
                K=int(p["random_samples"]),
                bbox_local=True,
                max_resolution=int(p["max_resolution"]),
            )
            self.preset_combo.addItem(
                f"{p['label']}  (~{live_peak:.0f} GiB peak full pipeline)",
                userData=p,
            )
```

**3.3c:** Find the description label string (line ~282 — `grep -n "est_peak_vram_gib" forza_abyss_painter/gui/generate_dialog.py`):

```python
            f"estimated peak VRAM: {preset['est_peak_vram_gib']:.1f} GiB"
```

Change to use the live estimate (compute it once and reuse):

```python
            from forza_abyss_painter.shapegen.gpu.vram_planner import (
                estimate_full_pipeline_gib,
            )
            live_peak = estimate_full_pipeline_gib(
                K=int(preset["random_samples"]),
                bbox_local=True,
                max_resolution=int(preset["max_resolution"]),
            )
            ...
            f"estimated peak VRAM: ~{live_peak:.0f} GiB (full pipeline)"
```

**3.3d:** Also update `_refresh_vram_estimate` at line ~303 which reads `preset["est_peak_vram_gib"]`. Change to use `estimate_full_pipeline_gib` similarly.

(Read the function to find the right structure — the change is mechanical: replace `preset["est_peak_vram_gib"]` reads with a live computation.)

- [ ] **Step 3.4: Run tests**

```bash
pytest tests/test_generate_dialog_live_peak_label.py tests/test_generate_dialog_recommendation_label.py tests/test_generate_dialog_initial_source.py tests/test_generate_dialog_checkpoint_spinbox.py -v
```

Expected: 1 new + all existing dialog tests still pass.

- [ ] **Step 3.5: Commit**

```bash
git add forza_abyss_painter/gui/generate_dialog.py tests/test_generate_dialog_live_peak_label.py
git commit -m "$(cat <<'EOF'
fix(gui): Generate dialog labels use live full-pipeline estimate

Drops the static `est_peak_vram_gib` marketing numbers (Lineart 2.5,
Headshot 3.5, Medium 5.0, Hi-Res 12.0) from the preset combo label,
preset description, and _refresh_vram_estimate. Each now reads
estimate_full_pipeline_gib(K, max_res, bbox_local=True) at display
time so the user sees the actual expected peak (e.g. Hi-Res 3000
shows ~165 GiB instead of 12.0).

Cursor's QUASAR 2026-05-27 finding: user picked Hi-Res based on
"12.0 GiB peak" label → OOM at 53.7 GiB. Honest labels would have
visibly steered them to Medium.

LOCAL_PRESETS still carries est_peak_vram_gib field as a
documentation hint; it just doesn't drive display anymore.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: OOM error message includes restart hint

**Files:**
- Modify: `forza_abyss_painter/runtime/torch_runner.py` (find OOM handling in `engine_run` stage)
- Create: `tests/test_torch_runner_oom_message.py`

- [ ] **Step 4.1: Write the failing test**

Create `tests/test_torch_runner_oom_message.py`:

```python
"""When the runner emits an engine_run error from a CUDA OOM, the
message should include the 'restart EXE' suggestion so users know
how to recover (CUDA cache survives the failed process and the
EXE process itself can hold an old context)."""
from __future__ import annotations

import re

from forza_abyss_painter.runtime import torch_runner


def test_engine_run_oom_message_suggests_restart():
    """Find the runner code that emits the engine_run error event
    and confirm it includes 'restart' guidance."""
    # The function we care about is run() — but the OOM message
    # construction lives near the engine_run except-block. Read the
    # source for the literal 'restart' / 'restart EXE' / similar.
    import inspect
    src = inspect.getsource(torch_runner)
    # The OOM-class message should mention restart. We don't pin the
    # exact phrasing, just that the word appears in proximity to
    # 'engine_run'.
    engine_run_block = re.search(
        r'"stage":\s*"engine_run".*?\n(?:.*?\n){0,15}',
        src,
        re.DOTALL,
    )
    assert engine_run_block, "Could not find engine_run stage block in run()"
    block_text = engine_run_block.group(0).lower()
    assert "restart" in block_text, (
        "engine_run error message should suggest restarting the EXE "
        "to release the CUDA cache. Got block:\n" + engine_run_block.group(0)
    )
```

- [ ] **Step 4.2: Run test to verify failure**

`pytest tests/test_torch_runner_oom_message.py -v`

Expected: FAIL — 'restart' not in the engine_run error block.

- [ ] **Step 4.3: Update the OOM message**

Edit `forza_abyss_painter/runtime/torch_runner.py`. Find the `engine_run` error emission (search `grep -n "engine_run" forza_abyss_painter/runtime/torch_runner.py`).

There should be two locations (one for `RuntimeError` — the OOM-class — and one for the generic `Exception` catch-all). The `RuntimeError` case is the one that matters for OOM. Update its message to include the restart hint. Example:

```python
    except RuntimeError as exc:
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
```

(Adapt to the actual structure — verify the variable names match before editing.)

- [ ] **Step 4.4: Run tests**

```bash
pytest tests/test_torch_runner_oom_message.py tests/test_torch_runner.py tests/test_torch_runner_polish_mode.py -v 2>&1 | tail -15
```

Expected: 1 new + existing tests pass.

- [ ] **Step 4.5: Commit**

```bash
git add forza_abyss_painter/runtime/torch_runner.py tests/test_torch_runner_oom_message.py
git commit -m "$(cat <<'EOF'
fix(runtime): OOM error message suggests restart EXE + smaller preset

Cursor's QUASAR 2026-05-27 finding: after OOM, the EXE's GPU still
holds 53 GiB of failed allocations. Subsequent runs hit the same
OOM until the process restarts. The error dialog didn't tell the
user this.

engine_run RuntimeError emission now includes: "close FH6 if
running, restart the EXE to release the CUDA context, and try a
smaller preset."

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Smoke + SMB + push

- [ ] **Step 5.1: Full regression sweep**

```bash
pytest tests/test_estimate_full_pipeline_gib.py \
       tests/test_vram_preflight_uses_full_pipeline.py \
       tests/test_generate_dialog_live_peak_label.py \
       tests/test_torch_runner_oom_message.py \
       tests/test_vram_preflight_verdict.py \
       tests/test_recommend_max_resolution.py \
       tests/test_main_window_autotune_status.py \
       tests/test_generate_dialog_recommendation_label.py \
       tests/test_generate_dialog_initial_source.py \
       tests/test_generate_dialog_checkpoint_spinbox.py \
       tests/test_torch_runner.py \
       tests/test_torch_runner_polish_mode.py \
       tests/test_gpu_bundle_gui.py -v 2>&1 | tail -10
```

Expected: all pass + 1 pre-existing skip.

- [ ] **Step 5.2: rsync source to SMB**

```bash
rsync -a --delete \
  --exclude='__pycache__' --exclude='.git' --exclude='*.pyc' \
  --exclude='dist' --exclude='build' --exclude='.pytest_cache' \
  --exclude='*.egg-info' --exclude='node_modules' \
  /Users/kusanagi/Development/forza-abyss-painter/ \
  /Volumes/ContentCreation/ForzaAbyssPainter_build/source/
```

- [ ] **Step 5.3: Append a Run 10 note to CURSOR_NEXT_RUN.md**

```bash
cat >> /Volumes/ContentCreation/ForzaAbyssPainter_build/CURSOR_NEXT_RUN.md <<'PREVIEW_EOF'

---

## Run 10 preview — PR A VRAM honesty (Cursor 2026-05-27 manual fix bundle)

Closes UPSTREAM_MANUAL_TEST_FINDINGS_20260527 §1 + §3:

- `estimate_full_pipeline_gib(K, ...) = estimate_peak_vram_gib + 35` —
  35 GiB calibrated from Run 4 (47.5 GiB measured) + your reshape
  session (53.7 GiB measured).
- Generate dialog preset combo NOW shows live full-pipeline estimate
  (e.g. Hi-Res 3000 → ~165 GiB) instead of the static "12.0 GiB
  peak" marketing number.
- `_start_gpu` preflight and SettingsPanel both switched to the new
  estimate. Existing Item E hard-block thresholds unchanged — they
  catch the Hi-Res-on-32G OOM now because the estimate is honest.
- OOM error message in the runner: "close FH6, restart EXE to
  release CUDA cache, try smaller preset."

QUASAR spot-check (~5 min after rebuild):

1. Open EXE → Generate locally → confirm Hi-Res 3000 preset label
   shows ~165 GiB peak (not 12.0). Medium 1000 should show ~47 GiB.
2. Try the Re-shape-gen scenario from §1 of the findings doc
   (loaded 3k JSON + hi-res body + Hi-Res preset) → should NOW
   hit the critical block modal BEFORE the runner spawns.
3. Lower to Medium 1000 → should pass preflight on a free RTX 5090
   (~27 GiB free) — wait, Medium also estimates ~47 GiB now, so
   it ALSO blocks on a 32G card with FH6 closed (~27 free).

The third point is real and expected — Run 4 ALREADY OOMed at
Medium-class. The honest estimate just exposes that. PR B (Tier B
UX) will lower the polish default + add reshape defaults to
Headshot/Lineart class for 32G cards. PR A is the truth-telling
layer; PR B is the friendly defaults.

PREVIEW_EOF
echo "Run 10 preview appended"
```

- [ ] **Step 5.4: Push**

```bash
git push origin feat/exe-colab-ports-batch 2>&1 | tail -3
```

Report the push range + final test count.

---

## Self-Review Summary

| Fix from Cursor's findings | Task |
|---|---|
| V1 — PIPELINE_OVERHEAD_GIB / estimate_full_pipeline_gib | Task 1 |
| V2 — Preset dropdown label uses live estimate | Task 3 |
| V3 — Tooltip on polish overhead | (deferred — covered by V1's full-pipeline naming) |
| V4 — peak > 0.85*free → critical block | Already implemented in Item E; Task 2 just feeds it the better estimate |
| V5 — VRAM budget combo clarification | (deferred — out of scope for PR A; user choice was "PR A first") |
| B5 — OOM dialog mentions restart EXE | Task 4 |
| Acceptance: Run 4 + Hi-Res 3000 scenarios both block at preflight | Tasks 1 + 2 |
