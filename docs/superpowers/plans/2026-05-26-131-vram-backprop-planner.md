# #131 VRAM Back-Prop Resolution Planner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the static 720-pixel cap with a dynamic recommendation derived from `probe_free_vram()` + the existing `vram_planner` math, so big-VRAM cards stop throwing away 80%+ of their headroom on the default `Nikke 2048×2048 → 720` downscale.

**Architecture:**
- New pure function `vram_planner.recommend_max_resolution(free_gib, K, …) -> int` that inverts `_peak_bytes_per_candidate`'s footprint formula. Returns `safety_floor_px` (720) when probe unavailable or math says <720; otherwise the largest fitting canvas.
- `worker._apply_upload_cap` gains `upload_cap_override` kwarg (backward-compatible default None).
- `GenerateLocallyDialog` and `_start_gpu` slot probe at dialog-open / generate-time, compute recommendation, surface in label/status-bar, and pass through to the runner via the preset dict.
- Item E hard-block remains the safety net for "even the recommendation won't fit".

**Tech Stack:** Python 3.10+, PySide6 (Qt), pytest (offscreen Qt for GUI tests), nvidia-smi probe (#125).

**Spec:** `docs/superpowers/specs/2026-05-26-131-vram-backprop-planner-design.md`

---

## File Structure

### New files

| Path | Purpose |
|---|---|
| `tests/test_recommend_max_resolution.py` | Pure math — Regime A solving, Regime B short-circuit, floor clamp, ceiling clamp, probe-unavailable. |
| `tests/test_apply_upload_cap_override.py` | `worker._apply_upload_cap(upload_cap_override=N)` uses N; default kwarg unchanged. |
| `tests/test_generate_dialog_recommendation_label.py` | Offscreen Qt — dialog shows "Recommended: X px" label with right number for canned probe values. |
| `tests/test_main_window_autotune_status.py` | Offscreen Qt — `_start_gpu` writes auto-tune line to status bar; preset's max_resolution gets bumped. |

### Modified files

| Path | Change |
|---|---|
| `forza_abyss_painter/shapegen/gpu/vram_planner.py` | Append `recommend_max_resolution` function. |
| `forza_abyss_painter/shapegen/worker.py` | Add `upload_cap_override` kwarg to `_apply_upload_cap`. |
| `forza_abyss_painter/gui/generate_dialog.py` | Probe + label + preset-bump on dialog open / preset change. |
| `forza_abyss_painter/gui/main_window.py` | In `_start_gpu` around line 1170-1180, compute recommendation, bump preset's `max_resolution`, status-bar log. |

### Deliberately untouched

- `DEFAULT_UPLOAD_CAP_PX = 720` — stays as the floor constant; only its usage gains an optional override.
- `BBOX_LOCAL_SAFETY = 15.0` — no recalibration in this spec.
- Item E preflight in `_start_gpu` (lines 1057-1086) — unchanged. Back-prop runs BEFORE the preflight; preflight catches "still doesn't fit".

---

## Task 1: `recommend_max_resolution` pure function

**Files:**
- Modify: `forza_abyss_painter/shapegen/gpu/vram_planner.py` (append at end of file)
- Create: `tests/test_recommend_max_resolution.py`

- [ ] **Step 1.1: Write the failing test**

Create `tests/test_recommend_max_resolution.py`:

```python
"""Pure math tests for recommend_max_resolution.

Inverts _peak_bytes_per_candidate's footprint formula: given a free
VRAM budget + K, returns the largest max_resolution that fits.
Returns safety_floor_px when probe is unavailable or math says <floor.
"""
from __future__ import annotations

import math

import pytest

from forza_abyss_painter.shapegen.gpu.vram_planner import (
    recommend_max_resolution,
    estimate_peak_vram_gib,
    BBOX_LOCAL_SAFETY,
)


def test_probe_unavailable_returns_floor():
    assert recommend_max_resolution(free_gib=0.0, K=8192) == 720
    assert recommend_max_resolution(free_gib=-1.0, K=8192) == 720


def test_tiny_budget_clamps_to_floor():
    """Tight free VRAM (e.g. FH6 squatting on a 32 GiB card) →
    return the safety floor; don't suggest something even smaller."""
    rec = recommend_max_resolution(free_gib=1.0, K=8192)
    assert rec == 720


def test_big_card_idle_solves_regime_a():
    """RTX PRO 6000 Blackwell at K=8192, ~95 GiB free →
    Regime A math solves for a max_res well above 720."""
    rec = recommend_max_resolution(free_gib=95.0, K=8192)
    assert rec > 720
    # Sanity: the returned value must actually fit in the budget at
    # 0.85 safety margin.
    peak = estimate_peak_vram_gib(K=8192, bbox_local=True, max_resolution=rec)
    assert peak <= 95.0 * 0.85, (
        f"recommendation {rec}px → peak {peak:.2f} GiB > budget"
    )


def test_regime_b_short_circuit():
    """When crop_e is pinned at bbox_crop_max=256, increasing
    max_resolution beyond 8*256=2048 doesn't change peak_bytes
    (footprint becomes constant (513)² · K · 12 · 15). So if Regime B
    fits at all, we can return the absolute ceiling."""
    # K small enough that Regime B fits comfortably
    rec = recommend_max_resolution(free_gib=200.0, K=1024,
                                     absolute_ceiling_px=4096)
    assert rec == 4096


def test_absolute_ceiling_clamps_high_recommendation():
    """When math says > absolute_ceiling, clamp."""
    rec = recommend_max_resolution(free_gib=500.0, K=512,
                                     absolute_ceiling_px=2048)
    assert rec <= 2048


def test_monotonic_in_free_gib():
    """More free VRAM → at-least-as-much recommended resolution."""
    rec_low = recommend_max_resolution(free_gib=10.0, K=8192)
    rec_mid = recommend_max_resolution(free_gib=40.0, K=8192)
    rec_high = recommend_max_resolution(free_gib=95.0, K=8192)
    assert rec_low <= rec_mid <= rec_high


def test_monotonic_inverse_in_k():
    """At fixed budget, lower K → at-least-as-much recommended res
    (each candidate is cheaper)."""
    rec_low_k = recommend_max_resolution(free_gib=40.0, K=2048)
    rec_high_k = recommend_max_resolution(free_gib=40.0, K=16384)
    assert rec_low_k >= rec_high_k


def test_floor_can_be_overridden():
    """Caller can pass a lower safety floor (e.g. for testing)."""
    rec = recommend_max_resolution(free_gib=1.0, K=8192,
                                     safety_floor_px=128)
    assert rec >= 128


def test_returned_value_actually_fits():
    """For a range of (free_gib, K), the returned max_res must keep
    estimate_peak_vram_gib at or below free_gib * 0.85."""
    for free_gib in (5.0, 12.0, 27.0, 48.0, 95.0):
        for K in (1024, 4096, 8192, 16384):
            rec = recommend_max_resolution(free_gib=free_gib, K=K)
            # Only check when recommendation is in Regime A territory;
            # Regime B returns the absolute_ceiling which by construction
            # fits.
            peak = estimate_peak_vram_gib(K=K, bbox_local=True,
                                            max_resolution=rec)
            # Floor case is allowed to over-shoot the budget — Item E's
            # hard-block catches that scenario separately.
            if rec > 720:
                assert peak <= free_gib * 0.85 + 1e-6, (
                    f"free={free_gib} K={K} rec={rec} peak={peak:.2f} "
                    f"exceeds 0.85*{free_gib}={free_gib*0.85:.2f}"
                )


def test_full_canvas_path():
    """full_canvas=False uses FULL_CANVAS_SAFETY (8.0). For the same
    budget, full_canvas yields LOWER max_res than bbox_local (per-cand
    cost is much higher)."""
    bbox = recommend_max_resolution(free_gib=95.0, K=4096, bbox_local=True)
    full = recommend_max_resolution(free_gib=95.0, K=4096, bbox_local=False)
    assert full < bbox or full == 720, (
        f"full_canvas={full} should be <= bbox={bbox} (or both at floor)"
    )
```

Run `pytest tests/test_recommend_max_resolution.py -v` — expect FAIL with `ImportError: cannot import name 'recommend_max_resolution'`.

- [ ] **Step 1.2: Implement the function**

Append to `forza_abyss_painter/shapegen/gpu/vram_planner.py` (after `resolve_k_chunk_size`):

```python
def recommend_max_resolution(
    free_gib: float,
    K: int,
    *,
    bbox_local: bool = True,
    bbox_crop_max: int = 256,
    safety_floor_px: int = 720,
    safety_margin: float = 0.85,
    absolute_ceiling_px: int = 4096,
) -> int:
    """Return the largest max_resolution that fits in
    `safety_margin * free_gib` of VRAM at the given K, in the bbox_local
    (or full_canvas) scoring regime.

    Always returns at least `safety_floor_px` (default 720). Probe-
    unavailable callers (free_gib <= 0) get the floor. When the math
    indicates a value above `absolute_ceiling_px` (i.e., Regime B is
    fully satisfied), the ceiling is returned — callers separately
    clamp by the source image's long side.

    The math is documented in
    `docs/superpowers/specs/2026-05-26-131-vram-backprop-planner-design.md`
    §4. In short, the K-batch footprint formula has two regimes:

      Regime A — crop_e = max_resolution // 8 (small canvas):
          footprint = (2 * crop_e + 1)²       quadratic in max_res
      Regime B — crop_e = bbox_crop_max       (big canvas, capped):
          footprint = (2 * bbox_crop_max + 1)²   constant

    Regime B is the "we're past the bbox cap; canvas size no longer
    affects K-peak" regime — if it fits, ceiling out.
    """
    if free_gib <= 0 or K < 1:
        return safety_floor_px

    target_bytes = free_gib * safety_margin * 1e9
    if bbox_local:
        safety = BBOX_LOCAL_SAFETY
    else:
        safety = FULL_CANVAS_SAFETY
    bytes_per_pixel = 12.0 * safety   # 3 channels × 4 bytes × safety

    # Regime B check: at crop_e = bbox_crop_max, footprint is constant.
    # If THAT fits, then any max_res >= bbox_crop_max * 8 is fine.
    if bbox_local:
        regime_b_footprint = (2 * bbox_crop_max + 1) ** 2
        regime_b_total = K * regime_b_footprint * bytes_per_pixel
        if regime_b_total <= target_bytes:
            return int(absolute_ceiling_px)

    # Regime A: footprint = (2 * (max_res // 8) + 1)². Solve for max_res.
    # Solve: K * (2 * (m // 8) + 1)² * bytes_per_pixel <= target_bytes
    # i.e.: (2 * (m // 8) + 1) <= sqrt(target_bytes / (K * bytes_per_pixel))
    # i.e.: m <= 8 * ((sqrt(...) - 1) / 2)
    import math as _math
    if K * bytes_per_pixel <= 0:
        return safety_floor_px
    inner = target_bytes / (K * bytes_per_pixel)
    if inner <= 1:
        return safety_floor_px
    sqrt_inner = _math.sqrt(inner)
    if sqrt_inner <= 1:
        return safety_floor_px
    max_res = int(8 * (sqrt_inner - 1) / 2)

    # full_canvas path uses the canvas² footprint (no bbox cap), so the
    # solve is just: max_res² <= target_bytes / (K * bytes_per_pixel).
    if not bbox_local:
        max_res = int(_math.sqrt(target_bytes / (K * bytes_per_pixel)))

    # Floor + ceiling clamps.
    max_res = max(safety_floor_px, max_res)
    max_res = min(absolute_ceiling_px, max_res)
    return max_res
```

- [ ] **Step 1.3: Run tests**

Run `pytest tests/test_recommend_max_resolution.py -v` — expect 9/9 pass.

Run `pytest tests/test_torch_runner.py tests/test_torch_runner_polish_mode.py -v 2>&1 | tail -10` — no regression.

- [ ] **Step 1.4: Commit**

```bash
git add forza_abyss_painter/shapegen/gpu/vram_planner.py tests/test_recommend_max_resolution.py
git commit -m "$(cat <<'EOF'
feat(vram): recommend_max_resolution — back-prop free VRAM → canvas (#131)

Pure function that inverts _peak_bytes_per_candidate's footprint
formula: given free VRAM + K, returns the largest max_resolution that
fits at the 0.85 safety margin. Handles Regime A (canvas-driven
crop_e), Regime B (bbox_crop_max-pinned crop_e), and the
probe-unavailable / tight-budget cases (returns safety_floor_px = 720).

No surface integration yet — that's the next commits (worker,
generate_dialog, main_window).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `_apply_upload_cap` override kwarg

**Files:**
- Modify: `forza_abyss_painter/shapegen/worker.py:22-67`
- Create: `tests/test_apply_upload_cap_override.py`

- [ ] **Step 2.1: Write the failing test**

Create `tests/test_apply_upload_cap_override.py`:

```python
"""upload_cap_override kwarg on _apply_upload_cap lets callers pass a
back-prop-derived value instead of the static DEFAULT_UPLOAD_CAP_PX.

Backward-compat: default kwarg is None → behavior unchanged.
Override never goes BELOW the static floor (720) — back-prop only
raises the cap, never lowers it (the safety floor is the consumer-card
production stance).
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from forza_abyss_painter.shapegen.worker import (
    _apply_upload_cap,
    DEFAULT_UPLOAD_CAP_PX,
)


def _src(w: int, h: int) -> Image.Image:
    return Image.new("RGB", (w, h), color=(128, 128, 128))


def test_default_kwarg_preserves_old_behavior():
    """Override=None → uses DEFAULT_UPLOAD_CAP_PX (720). A 2000-px
    source should get downscaled to a value derived from 720 / pad_factor."""
    img, _, was_resized = _apply_upload_cap(
        _src(2000, 2000), None,
        profile_max_resolution=720,
        upload_cap_px=DEFAULT_UPLOAD_CAP_PX,
    )
    assert was_resized is True
    assert max(img.size) <= 720


def test_override_raises_cap():
    """Override=1200 + profile_max_resolution=1200 → 2000-px source
    survives at higher resolution than 720 would allow."""
    img_default, _, _ = _apply_upload_cap(
        _src(2000, 2000), None,
        profile_max_resolution=1200,
        upload_cap_px=DEFAULT_UPLOAD_CAP_PX,
        upload_cap_override=None,
    )
    img_override, _, _ = _apply_upload_cap(
        _src(2000, 2000), None,
        profile_max_resolution=1200,
        upload_cap_px=DEFAULT_UPLOAD_CAP_PX,
        upload_cap_override=1200,
    )
    assert max(img_default.size) <= 720
    assert max(img_override.size) > max(img_default.size)
    assert max(img_override.size) <= 1200


def test_override_never_below_floor():
    """upload_cap_override < DEFAULT_UPLOAD_CAP_PX → use the floor.
    Back-prop must never lower the cap below the safety floor."""
    img, _, _ = _apply_upload_cap(
        _src(2000, 2000), None,
        profile_max_resolution=720,
        upload_cap_px=DEFAULT_UPLOAD_CAP_PX,
        upload_cap_override=480,  # below floor
    )
    # Behavior should match upload_cap_override=None (uses 720 floor).
    img_ref, _, _ = _apply_upload_cap(
        _src(2000, 2000), None,
        profile_max_resolution=720,
        upload_cap_px=DEFAULT_UPLOAD_CAP_PX,
        upload_cap_override=None,
    )
    assert img.size == img_ref.size


def test_override_zero_disables_cap_like_legacy():
    """upload_cap_override=0 → disable cap (legacy upload_cap_px=0 behavior).
    The override is OPT-OUT compatible: passing 0 means 'no cap'."""
    img, _, was_resized = _apply_upload_cap(
        _src(2000, 2000), None,
        profile_max_resolution=2000,
        upload_cap_px=0,
        upload_cap_override=0,
    )
    assert was_resized is False
    assert img.size == (2000, 2000)


def test_alpha_mask_resizes_with_override(tmp_path):
    """When override raises the cap, an alpha mask should resize to
    the same dimensions as the RGB image."""
    img, alpha_out, was_resized = _apply_upload_cap(
        _src(2000, 2000),
        np.zeros((2000, 2000), dtype=np.uint8),
        profile_max_resolution=1200,
        upload_cap_px=DEFAULT_UPLOAD_CAP_PX,
        upload_cap_override=1200,
    )
    assert was_resized is True
    assert alpha_out is not None
    assert alpha_out.shape == (img.size[1], img.size[0])
```

Run `pytest tests/test_apply_upload_cap_override.py -v` — expect FAIL with `TypeError: _apply_upload_cap() got an unexpected keyword argument 'upload_cap_override'`.

- [ ] **Step 2.2: Implement the kwarg**

Edit `forza_abyss_painter/shapegen/worker.py`. Change the signature at line 22 and the function body:

```python
def _apply_upload_cap(
    img: Image.Image,
    alpha_mask: np.ndarray | None,
    *,
    profile_max_resolution: int,
    upload_cap_px: int,
    buffer_frac: float = 0.08,
    upload_cap_override: int | None = None,
) -> tuple[Image.Image, np.ndarray | None, bool]:
    """Downscale img so the FINAL PADDED canvas stays under the upload cap.

    Ports v0.1.5 from the colab pipeline (sister repo's
    `notebooks/build_colab_notebook.py` -> `_load_image_bytes`). The trick
    is to reserve padding budget up front: the cap applies to the final
    padded output, so we divide by (1 + 2*buffer_frac) to derive the
    effective INPUT max — that way after `_pad_with_source_mean` fires,
    the post-pad long side lands at or below `effective_max`.

    `upload_cap_px <= 0` disables the cap entirely. Callers wanting the
    pre-v0.1.5 behavior (no upload-time safety net, only the
    profile.max_resolution downscale that runs post-padding) pass 0.

    The cap is `min(profile.max_resolution, upload_cap_px)` — there's no
    point shrinking below the preset's own ceiling, that would just lose
    detail without speeding anything up.

    `upload_cap_override` (#131): when set, replaces upload_cap_px for
    this call (subject to the floor min) — used by the back-prop
    planner to raise the cap on big-VRAM cards. None preserves the
    legacy behavior. `upload_cap_override=0` means "disable cap"
    (same semantic as upload_cap_px=0). Otherwise the effective cap is
    `max(upload_cap_px, upload_cap_override)` — back-prop only raises.

    Returns (img, alpha_mask, was_resized) so callers can log when the
    cap actually fired (useful for the UI: "auto-resized your 4K input
    to 720px for safety").
    """
    if upload_cap_override is not None:
        if upload_cap_override == 0:
            effective_cap = 0   # explicit disable
        else:
            effective_cap = max(upload_cap_px, upload_cap_override)
    else:
        effective_cap = upload_cap_px

    if effective_cap <= 0:
        return img, alpha_mask, False
    effective_max = min(profile_max_resolution, effective_cap)
    pad_factor = 1.0 + 2.0 * buffer_frac
    effective_input_max = max(1, int(effective_max / pad_factor))
    if max(img.size) <= effective_input_max:
        return img, alpha_mask, False
    scale = effective_input_max / max(img.size)
    new_size = (
        max(1, int(img.size[0] * scale)),
        max(1, int(img.size[1] * scale)),
    )
    img = img.resize(new_size, Image.LANCZOS)
    if alpha_mask is not None:
        am_img = Image.fromarray(alpha_mask, "L").resize(new_size, Image.LANCZOS)
        alpha_mask = np.asarray(am_img, dtype=np.uint8)
    return img, alpha_mask, True
```

- [ ] **Step 2.3: Run tests**

Run `pytest tests/test_apply_upload_cap_override.py -v` — expect 5/5 pass.

Find existing tests that call `_apply_upload_cap` and confirm no regression:
```bash
grep -rl "_apply_upload_cap" tests/ | head -5
```
Run any that exist with `-v`.

- [ ] **Step 2.4: Commit**

```bash
git add forza_abyss_painter/shapegen/worker.py tests/test_apply_upload_cap_override.py
git commit -m "$(cat <<'EOF'
feat(worker): _apply_upload_cap accepts upload_cap_override kwarg (#131)

Backward-compatible: omitting the kwarg or passing None preserves the
static DEFAULT_UPLOAD_CAP_PX=720 behavior. When set, effective cap is
max(upload_cap_px, upload_cap_override) — back-prop only RAISES the
cap, never lowers it (the floor stays the consumer-card production
stance). upload_cap_override=0 means "disable cap" matching the
existing upload_cap_px=0 semantic.

Caller integration follows in the GenerateLocallyDialog + _start_gpu
commits.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `GenerateLocallyDialog` recommendation label + preset bump

**Files:**
- Modify: `forza_abyss_painter/gui/generate_dialog.py` (`_on_preset_changed` + dialog state)
- Create: `tests/test_generate_dialog_recommendation_label.py`

- [ ] **Step 3.1: Write the failing test**

Create `tests/test_generate_dialog_recommendation_label.py`:

```python
"""GenerateLocallyDialog shows the back-prop recommendation in the
preset description label and bumps the preset's effective
max_resolution (#131)."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from forza_abyss_painter.gui.generate_dialog import GenerateLocallyDialog  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def big_card_probe(monkeypatch):
    """Monkey-patch probe_free_vram to return canned values matching an
    idle RTX PRO 6000 Blackwell — back-prop should suggest a value
    well above 720."""
    from forza_abyss_painter.runtime import nvidia_smi as ns_mod

    def fake_probe(*, force=False):
        return ns_mod.ProbeResult(
            available=True,
            free_mib=95 * 1024, total_mib=98 * 1024,
            name="NVIDIA RTX PRO 6000 Blackwell",
            driver_version="555.85",
            probed_at=0.0,
        )

    monkeypatch.setattr(
        "forza_abyss_painter.gui.generate_dialog.probe_free_vram",
        fake_probe,
    )


@pytest.fixture
def probe_unavailable(monkeypatch):
    from forza_abyss_painter.runtime import nvidia_smi as ns_mod

    def fake_probe(*, force=False):
        return ns_mod.ProbeResult(
            available=False, reason="nvidia-smi not on PATH",
            probed_at=0.0,
        )

    monkeypatch.setattr(
        "forza_abyss_painter.gui.generate_dialog.probe_free_vram",
        fake_probe,
    )


def test_label_shows_recommendation_on_supported_gpu(qapp, big_card_probe):
    dlg = GenerateLocallyDialog(parent=None)
    desc_text = dlg.preset_desc.text()
    assert "Recommended" in desc_text
    # Big-card scenario: back-prop should raise above 720.
    # The label includes the free-GiB context.
    assert "GiB free" in desc_text
    dlg.deleteLater()


def test_label_floor_message_when_probe_unavailable(qapp, probe_unavailable):
    dlg = GenerateLocallyDialog(parent=None)
    desc_text = dlg.preset_desc.text()
    assert "Recommended" in desc_text
    # Probe-unavailable scenario: should show the floor message.
    assert "safety floor" in desc_text or "probe unavailable" in desc_text.lower()
    dlg.deleteLater()


def test_preset_data_carries_effective_max_resolution(qapp, big_card_probe):
    """The preset combo's currentData() should reflect the BUMPED
    max_resolution so build_run_config gets the back-prop value."""
    dlg = GenerateLocallyDialog(parent=None)
    preset = dlg.preset_combo.currentData()
    assert preset is not None
    # Big card → max_resolution should be at least the preset's baked
    # value (back-prop never lowers).
    baked = preset.get("baked_max_resolution", preset["max_resolution"])
    assert preset["max_resolution"] >= baked
    dlg.deleteLater()


def test_preset_change_recomputes_recommendation(qapp, big_card_probe):
    """Changing the preset (different K) triggers a new probe + bump."""
    dlg = GenerateLocallyDialog(parent=None)
    # Lineart (K=4096) vs Hi-Res (K=12288) at the same probe → different
    # recommendations (lower K → more headroom per candidate → higher max_res).
    dlg.preset_combo.setCurrentIndex(0)   # Lineart
    lineart_max = dlg.preset_combo.currentData()["max_resolution"]
    dlg.preset_combo.setCurrentIndex(3)   # Hi-Res
    hires_max = dlg.preset_combo.currentData()["max_resolution"]
    # Both should be ≥ their baked values; lineart should generally be
    # higher because K is smaller.
    assert lineart_max >= 480   # Lineart baked
    assert hires_max >= 1000    # Hi-Res baked
    dlg.deleteLater()
```

Run `pytest tests/test_generate_dialog_recommendation_label.py -v` — expect FAIL.

- [ ] **Step 3.2: Implement**

Edit `forza_abyss_painter/gui/generate_dialog.py`. At the top imports, add:

```python
from forza_abyss_painter.runtime.nvidia_smi import probe_free_vram
from forza_abyss_painter.shapegen.gpu.vram_planner import recommend_max_resolution
```

Replace the existing `_on_preset_changed` method (around lines 203-222) with:

```python
    def _on_preset_changed(self, idx: int) -> None:
        self._selected_preset_idx = idx
        preset = self.preset_combo.currentData()
        if preset is None:
            return
        # Probe free VRAM (#125) and compute the back-prop max_resolution
        # recommendation (#131). The probe is cached for 5s so flipping
        # presets doesn't spawn nvidia-smi every change.
        probe = probe_free_vram()
        baked_max_res = preset.get("baked_max_resolution",
                                     preset["max_resolution"])
        if probe.available and probe.free_gib is not None:
            recommended = recommend_max_resolution(
                free_gib=probe.free_gib,
                K=int(preset["random_samples"]),
                bbox_local=True,
            )
            # Back-prop never LOWERS the baked preset value — the preset
            # author chose that as a quality/speed default.
            effective_max_res = max(baked_max_res, recommended)
            rec_line = (
                f"<br><b>Recommended max_resolution:</b> "
                f"{effective_max_res} px "
                f"(auto-tuned to fit {probe.free_gib:.1f} GiB free on "
                f"{probe.name or 'GPU'}). Floor: 720."
            )
        else:
            effective_max_res = baked_max_res
            rec_line = (
                f"<br><b>Recommended max_resolution:</b> "
                f"{effective_max_res} px (VRAM probe unavailable; using "
                f"safety floor)."
            )

        # Persist the effective value back into the preset dict so
        # downstream build_run_config sees the bumped number.
        preset.setdefault("baked_max_resolution", baked_max_res)
        preset["max_resolution"] = effective_max_res

        self.preset_desc.setText(
            f"<b>{preset['label']}</b><br>"
            f"{preset['description']}<br><br>"
            f"<b>Settings:</b> "
            f"max_resolution={preset['max_resolution']}, "
            f"random_samples={preset['random_samples']}, "
            f"estimated peak VRAM: {preset['est_peak_vram_gib']:.1f} GiB"
            f"{rec_line}"
        )
        self.preset_desc.setTextFormat(Qt.RichText)
        self._refresh_vram_estimate()
        if self.source_path and preset:
            stem = self.source_path.stem
            suggested = (self.source_path.parent /
                         f"{stem}_{preset['num_shapes']}.json")
            self.output_field.setPlaceholderText(str(suggested))
```

- [ ] **Step 3.3: Run tests**

Run `pytest tests/test_generate_dialog_recommendation_label.py -v` — expect 4/4 pass.

Run `pytest tests/test_generate_dialog_initial_source.py -v` — confirm no regression on the Task 2 (#85) tests.

- [ ] **Step 3.4: Commit**

```bash
git add forza_abyss_painter/gui/generate_dialog.py tests/test_generate_dialog_recommendation_label.py
git commit -m "$(cat <<'EOF'
feat(gui): GenerateLocallyDialog shows back-prop max_res recommendation (#131)

On preset change (or dialog open), probes free VRAM via #125's
probe_free_vram and computes the recommended max_resolution via
recommend_max_resolution. Surfaces in the preset description label
("Recommended max_resolution: 1200 px (27.4 GiB free on RTX 5090).
Floor: 720.") and writes the effective value back into the preset
dict so build_run_config consumes the bumped number.

Probe-unavailable path (non-NVIDIA, WSL no passthrough) falls back to
the baked preset value with a clear label message.

baked_max_resolution preserved on the preset dict so subsequent preset
changes don't accumulate bumps.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: `main_window._start_gpu` auto-tune + status bar

**Files:**
- Modify: `forza_abyss_painter/gui/main_window.py` around lines 1170-1180 (the `_start_gpu` slot, where the preset dict is built)
- Create: `tests/test_main_window_autotune_status.py`

- [ ] **Step 4.1: Write the failing test**

Create `tests/test_main_window_autotune_status.py`:

```python
"""When _start_gpu builds its preset dict, it must apply the back-prop
recommendation and write an auto-tune line to the status bar (#131).

Cannot test the actual GPU spawn (no torch on test host), so we exercise
the preset-dict construction via a helper or by intercepting
build_run_config.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from forza_abyss_painter.gui import main_window as main_window_mod  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def big_card_probe(monkeypatch):
    from forza_abyss_painter.runtime import nvidia_smi as ns_mod

    def fake_probe(*, force=False):
        return ns_mod.ProbeResult(
            available=True,
            free_mib=95 * 1024, total_mib=98 * 1024,
            name="NVIDIA RTX PRO 6000 Blackwell",
            driver_version="555.85",
            probed_at=0.0,
        )

    monkeypatch.setattr(
        "forza_abyss_painter.gui.main_window.probe_free_vram",
        fake_probe,
    )


def test_apply_autotune_to_preset_returns_bumped_dict(big_card_probe):
    """Pure helper: given a preset dict + probe, returns the dict with
    bumped max_resolution and an auto-tune message."""
    preset = {
        "label": "Lineart — 400 shapes",
        "num_shapes": 400,
        "max_resolution": 480,
        "random_samples": 4096,
    }
    bumped, message = main_window_mod._apply_autotune_to_preset(preset)
    assert bumped["max_resolution"] >= 480, "must never lower baked value"
    assert "auto-tuned" in message.lower()
    assert "GiB free" in message
    assert "95" in message   # the free-GiB number is present


def test_apply_autotune_preserves_baked_value_on_probe_failure(monkeypatch):
    from forza_abyss_painter.runtime import nvidia_smi as ns_mod

    def fake_probe(*, force=False):
        return ns_mod.ProbeResult(
            available=False, reason="nvidia-smi missing", probed_at=0.0,
        )

    monkeypatch.setattr(
        "forza_abyss_painter.gui.main_window.probe_free_vram",
        fake_probe,
    )
    preset = {
        "label": "Hi-Res — 3000 shapes",
        "num_shapes": 3000,
        "max_resolution": 1000,
        "random_samples": 12288,
    }
    bumped, message = main_window_mod._apply_autotune_to_preset(preset)
    assert bumped["max_resolution"] == 1000   # unchanged
    assert "probe unavailable" in message.lower() or \
           "safety floor" in message.lower()


def test_apply_autotune_does_not_mutate_input(big_card_probe):
    preset = {
        "label": "Test",
        "num_shapes": 1000,
        "max_resolution": 720,
        "random_samples": 8192,
    }
    bumped, _ = main_window_mod._apply_autotune_to_preset(preset)
    # Input dict must be untouched (caller may want to retry).
    assert preset["max_resolution"] == 720
    # Output dict may have the same max_resolution if back-prop returned
    # the floor — that's fine.
    assert "max_resolution" in bumped
```

Run `pytest tests/test_main_window_autotune_status.py -v` — expect FAIL.

- [ ] **Step 4.2: Implement**

Edit `forza_abyss_painter/gui/main_window.py`. Add a module-level helper near the other module-level helpers (`_resolve_source_image_path`, `_vram_preflight_verdict`):

```python
def _apply_autotune_to_preset(preset: dict) -> tuple[dict, str]:
    """Apply the #131 back-prop recommendation to a GPU preset dict.

    Returns a NEW dict (input is not mutated) with `max_resolution`
    bumped to the back-prop value when bigger than the baked preset
    value (never lower — the preset author's choice is the floor for
    that preset). The second element is a human-readable status line
    the caller can log to the status bar.

    `probe_free_vram` is called with force=True to bypass the 5s cache
    — the user just clicked Start, so fresh data is worth the
    subprocess spawn cost.
    """
    bumped = dict(preset)   # shallow copy; preset's leaf values are scalars
    baked_max_res = int(bumped["max_resolution"])
    K = int(bumped.get("random_samples", 0) or 0)
    probe = probe_free_vram(force=True)
    if probe.available and probe.free_gib is not None and K > 0:
        from forza_abyss_painter.shapegen.gpu.vram_planner import (
            recommend_max_resolution,
        )
        recommended = recommend_max_resolution(
            free_gib=probe.free_gib, K=K, bbox_local=True,
        )
        effective = max(baked_max_res, recommended)
        bumped["max_resolution"] = effective
        if effective > baked_max_res:
            message = (
                f"Auto-tuned max_res to {effective} px "
                f"({probe.name or 'GPU'}, {probe.free_gib:.1f} GiB free)"
            )
        else:
            message = (
                f"Using baked max_res {baked_max_res} px "
                f"({probe.free_gib:.1f} GiB free; back-prop says "
                f"{recommended} px — using floor)"
            )
    else:
        bumped["max_resolution"] = baked_max_res
        message = (
            f"Using baked max_res {baked_max_res} px "
            f"(VRAM probe unavailable; using safety floor)"
        )
    return bumped, message
```

**Promote `probe_free_vram` to a module-level import.** The Item E
preflight code from the earlier session imports it LOCALLY inside the
`_start_gpu` slot:

```python
        from forza_abyss_painter.runtime.nvidia_smi import probe_free_vram
        probe = probe_free_vram(force=True)
```

Move that import to the top of `main_window.py` alongside the other
runtime imports, AND remove the local import line inside the slot. This
is required so `tests/test_main_window_autotune_status.py` can
`monkeypatch.setattr("forza_abyss_painter.gui.main_window.probe_free_vram", ...)` — the
patch only works when the name lives at module level.

Then in `_start_gpu` around lines 1175-1180, replace the preset-dict construction:

```python
        # Build a preset dict matching gpu_gen_worker.build_run_config()'s
        # expected fields. Source values from the SettingsPanel so what
        # the user picked in CPU mode carries over to GPU. Apply the
        # #131 back-prop recommendation BEFORE handing to the runner so
        # big-VRAM cards get the bumped max_resolution.
        preset_baked = {
            "label": profile.name,
            "num_shapes": profile.stop_at,
            "max_resolution": profile.max_resolution,
            "random_samples": profile.random_samples,
        }
        preset, autotune_message = _apply_autotune_to_preset(preset_baked)
        self.statusBar().showMessage(autotune_message, 6000)
```

Then the existing `build_run_config(image_path, output_path, preset, ...)` call below consumes `preset` directly (the bumped one).

- [ ] **Step 4.3: Run tests**

Run `pytest tests/test_main_window_autotune_status.py -v` — expect 3/3 pass.

Run the regression set:
```bash
pytest tests/test_vram_preflight_verdict.py tests/test_resolve_source_image.py tests/test_gpu_bundle_gui.py -v 2>&1 | tail -15
```
— confirm no regression on the existing main_window helpers.

- [ ] **Step 4.4: Commit**

```bash
git add forza_abyss_painter/gui/main_window.py tests/test_main_window_autotune_status.py
git commit -m "$(cat <<'EOF'
feat(gui): _start_gpu applies #131 back-prop max_res + status-bar log

New module-level helper _apply_autotune_to_preset() takes the preset
dict the slot would have sent to build_run_config, runs
recommend_max_resolution() against the live free-VRAM probe, and
returns the bumped dict + a status-bar message. Helper is pure (no
mutation of input) and unit-tested independent of GPU availability.

_start_gpu now calls the helper after the preflight passes; the
status bar logs the auto-tune outcome ("Auto-tuned max_res to 1200 px
(RTX 5090, 27.4 GiB free)" on big cards; "Using baked max_res 720 px
(VRAM probe unavailable...)" on probe-failure paths).

Item E hard-block remains in front — back-prop runs after the
preflight but before the build_run_config call, so a peak > free
condition still triggers the modal.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Final smoke + push

No feature flag for #131 — back-prop is a pure UX improvement gated only by the existing nvidia-smi probe.

- [ ] **Step 5.1: Run the full #131 test suite**

```bash
pytest tests/test_recommend_max_resolution.py \
       tests/test_apply_upload_cap_override.py \
       tests/test_generate_dialog_recommendation_label.py \
       tests/test_main_window_autotune_status.py -v 2>&1 | tail -30
```

Expect: all green. If any fail, fix BEFORE proceeding.

- [ ] **Step 5.2: Run the broader regression set**

```bash
pytest tests/test_vram_preflight_verdict.py \
       tests/test_torch_runner.py \
       tests/test_torch_runner_polish_mode.py \
       tests/test_cli_generate_bbox_flags.py \
       tests/test_upload_panel_reshape_polish_buttons.py \
       tests/test_polish_dialog_values.py \
       tests/test_generate_dialog_initial_source.py \
       tests/test_resolve_source_image.py \
       tests/test_fap_refresh_same_dir_guard.py \
       tests/test_load_json_bom_tolerance.py \
       tests/test_gpu_bundle_gui.py -v 2>&1 | tail -25
```

Expect: all pass or pre-existing skips (no new failures).

- [ ] **Step 5.3: Update CURSOR_NEXT_RUN.md on SMB with a Run 7 candidate section**

The actual Run 7 is blocked on #129 chunked rasterize per Cursor's plan, but back-prop is a UX feature Cursor can spot-check on QUASAR independently. Add a small section at the bottom of `/Volumes/ContentCreation/ForzaAbyssPainter_build/CURSOR_NEXT_RUN.md` (or a new file `CURSOR_RUN_7_PREVIEW.md`):

```bash
cat >> /Volumes/ContentCreation/ForzaAbyssPainter_build/CURSOR_NEXT_RUN.md <<'PREVIEW_EOF'

---

## Run 7 preview — #131 back-prop max_resolution

After your manual Run 6 items finish, the next EXE rebuild from this
branch also includes the #131 VRAM back-prop planner:

- Generate dialog now shows "Recommended max_resolution: X px
  (Y GiB free)" + auto-bumps the preset's max_res on supported GPUs.
- Status bar logs the auto-tune line when Start is clicked.
- CPU upload cap (worker._apply_upload_cap) accepts the override too.

QUASAR spot-check (~5 min):

1. Open EXE → Generate locally → confirm the preset description label
   includes a "Recommended max_resolution" line.
2. Cycle through presets (Lineart / Headshot / Medium / Hi-Res) —
   the recommendation should change (lower K → higher recommendation).
3. Start a generation with a 2k+ source image — confirm the status bar
   shows the auto-tune line + the run uses the bumped max_res (visible
   in the chunked-K log line: chunk_size at the new resolution).
4. Disconnect nvidia-smi (or run on a CPU-only box) → confirm the
   probe-unavailable fallback shows "safety floor" + baked preset value.

This is informational, not blocking — Run 7's main gate is still
#129 chunked rasterize.

PREVIEW_EOF
```

- [ ] **Step 5.4: rsync source to SMB**

```bash
rsync -a --delete \
  --exclude='__pycache__' --exclude='.git' --exclude='*.pyc' \
  --exclude='dist' --exclude='build' --exclude='.pytest_cache' \
  --exclude='*.egg-info' --exclude='node_modules' \
  /Users/kusanagi/Development/forza-abyss-painter/ \
  /Volumes/ContentCreation/ForzaAbyssPainter_build/source/
```

- [ ] **Step 5.5: Push**

```bash
git push origin feat/exe-colab-ports-batch
```

Report the push range + final test count to the user.

---

## Self-Review Summary

| Spec section | Implementing task |
|---|---|
| §1 Purpose (replace 720 cap dynamically) | Tasks 1–4 |
| §3.1 GenerateLocallyDialog UX | Task 3 |
| §3.2 worker.py CPU path | Task 2 |
| §4 Math (Regime A / Regime B) | Task 1 |
| §5.1 `recommend_max_resolution` API | Task 1 |
| §5.2 Touched callers | Tasks 2, 3, 4 |
| §5.3 Tests | Tasks 1, 2, 3, 4 |
| §6 Behavior matrix | Verified via `test_returned_value_actually_fits` (Task 1) |
| §9 Acceptance criteria | Tasks 1–5 |
