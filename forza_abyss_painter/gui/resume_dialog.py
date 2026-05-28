"""Modal: confirm resume from a partial snapshot.

If the snapshot embeds `_run_config`, the dialog auto-fills target +
params and just asks the user to confirm. If `_run_config` is missing
(older snapshots), a preset picker UI is shown so the user can specify
target + K + max_resolution manually.

Returns the full RunConfig dict via .values() — caller hands directly
to GpuGenWorker.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox, QDialog, QHBoxLayout, QLabel, QPushButton,
    QVBoxLayout, QWidget,
)

from forza_abyss_painter.io.exporter import load_json
from forza_abyss_painter.shapegen.gpu.vram_planner import (
    recommend_max_resolution,
)


SAFETY_FLOOR_PX = 256


def _probe_free_gib(budget_gib: float) -> float:
    """nvidia-smi free-VRAM probe. Mirrors gpu_preflight._probe_free_gib
    so monkeypatch parity is preserved for tests. Returns budget_gib
    if the probe is unavailable on this host."""
    from forza_abyss_painter.runtime.nvidia_smi import probe_free_vram
    probe = probe_free_vram(force=True)
    if probe.available and probe.free_gib is not None:
        return float(probe.free_gib)
    return float(budget_gib)


def _recommend(K: int, free_gib: float) -> int:
    return recommend_max_resolution(K=K, free_gib=free_gib, bbox_local=True)


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
        gpu_budget_gib: float = 24.0,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Resume from snapshot")
        self.setModal(True)
        self.setMinimumWidth(520)

        self._snapshot_path = Path(snapshot_path)
        self._source_image_path = Path(source_image_path)
        self._gpu_budget_gib = float(gpu_budget_gib)

        # Load snapshot to extract _run_config (or detect absence).
        self._doc = load_json(str(self._snapshot_path))
        self._current_count = int(self._doc.shape_count or 0)
        # Try the embedded run config first.
        raw = self._snapshot_path.read_text(encoding="utf-8")
        self._raw_doc = json.loads(raw)
        self._run_config: dict[str, Any] | None = self._raw_doc.get("_run_config")

        # Decide whether the snapshot's baked max_resolution fits the
        # current card's free VRAM. If not, the engine will re-rasterize
        # the seeded shapes via seed_canvas_size (Tasks 6+7).
        self._lower_decision = self._compute_lower_decision()

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

        # VRAM mismatch warning (hidden when snapshot fits).
        self.lower_warning_label = QLabel("", self)
        self.lower_warning_label.setWordWrap(True)
        self.lower_warning_label.setTextFormat(Qt.RichText)
        self.lower_warning_label.setStyleSheet(
            "color: #d94f90; background: #1a0a14; "
            "border: 1px solid #d94f90; padding: 8px; border-radius: 4px;"
        )
        if self._lower_decision.get("lowered"):
            baked = self._lower_decision["baked"]
            lowered = self._lower_decision["max_resolution"]
            free_gib = self._lower_decision.get("free_gib", 0.0)
            self.lower_warning_label.setText(
                f"<b>VRAM mismatch — resume will run at lower resolution.</b><br>"
                f"Snapshot was rendered at max_resolution=<b>{baked}</b>, but "
                f"the current card has only <b>{free_gib:.1f} GiB</b> free, "
                f"which fits max_resolution=<b>{lowered}</b>. "
                f"Seeded shapes will be re-rasterized to the lower canvas "
                f"before resume continues."
            )
            self.lower_warning_label.setVisible(True)
        else:
            self.lower_warning_label.setVisible(False)
        root.addWidget(self.lower_warning_label)

        # Fallback preset picker (only shown when _run_config missing).
        # Only include presets whose target exceeds the current shape count
        # so every item in the combo represents a meaningful resume target.
        from forza_abyss_painter.gui.generate_dialog import LOCAL_PRESETS
        self.preset_combo = QComboBox(self)
        for p in LOCAL_PRESETS:
            if int(p["num_shapes"]) > self._current_count:
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

    def _snapshot_canvas_size(self) -> tuple[int, int]:
        """Return the (W, H) canvas the snapshot was rendered at.
        Preference order: _run_config.image_size → doc.image_size →
        (max_res, max_res) fallback."""
        rc = self._run_config or {}
        img_size_raw = (
            rc.get("image_size")
            or self._raw_doc.get("image_size")
        )
        if img_size_raw and len(img_size_raw) >= 2:
            return (int(img_size_raw[0]), int(img_size_raw[1]))
        baked = self._effective_max_res() or 720
        return (baked, baked)

    def _compute_lower_decision(self) -> dict:
        """Decide whether to lower max_resolution for this resume.

        Compares the snapshot's baked max_resolution to what the current
        card's free VRAM can fit (via nvidia-smi probe + recommender).
        If the recommended is below the baked, returns a lowered
        decision; the engine will re-rasterize seeded shapes to the
        new canvas via seed_canvas_size.
        """
        rc = self._run_config or {}
        baked = int(
            rc.get("max_resolution")
            or self._raw_doc.get("image_size", [720])[0]
        )
        K = int(rc.get("random_samples") or 1000)
        snap_canvas = self._snapshot_canvas_size()

        free_gib = _probe_free_gib(self._gpu_budget_gib)
        recommended = _recommend(K=K, free_gib=free_gib)

        if recommended >= baked:
            return {
                "max_resolution": baked,
                "seed_canvas_size": snap_canvas,
                "lowered": False,
            }

        if recommended < SAFETY_FLOOR_PX:
            # Below the floor we don't have a sane lower target; leave
            # the baked value and let Task 9's preflight surface the
            # block to the user.
            return {
                "max_resolution": baked,
                "seed_canvas_size": snap_canvas,
                "lowered": False,
                "blocked": True,
            }

        return {
            "max_resolution": recommended,
            "seed_canvas_size": snap_canvas,
            "lowered": True,
            "free_gib": free_gib,
            "baked": baked,
        }

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
        stem = self._snapshot_path.stem   # e.g. "ziz_2900"
        # Strip trailing _<digits>
        base_stem = re.sub(r"_\d+$", "", stem)
        if not base_stem:
            # Pathological case: snapshot stem was `_N` only — fall
            # back to the original stem to avoid producing `.json`
            # as a literal output filename.
            base_stem = stem
        output_path = self._snapshot_path.parent / f"{base_stem}.json"
        # Lower-decision overrides max_resolution and surfaces the
        # original snapshot canvas so engine.run_gpu can re-rasterize
        # seeded shape coords (Task 6 + Task 7).
        return {
            "image_path": str(self._source_image_path),
            "output_json_path": str(output_path),
            "mode": "fresh",
            "seed_shapes_path": str(self._snapshot_path),
            "num_shapes": target,
            "max_resolution": int(self._lower_decision["max_resolution"]),
            "seed_canvas_size": tuple(self._lower_decision["seed_canvas_size"]),
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
