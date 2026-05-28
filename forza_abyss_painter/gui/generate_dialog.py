"""Local GPU shape-generation workflow dialog.

Phase 2 of EXE GPU bundle (task #62) — the UI scaffolding. Phase 3 will
wire the Generate button to subprocess-invoke the embedded-Python runtime
that runs `forza_abyss_painter.shapegen.gpu.engine.run_gpu`. For now the
button reports "Phase 3 not yet shipped."

UX flow:
  1. User picks source image (PNG/JPG) via file picker
  2. User picks preset (Lineart 400 / Headshot 700 / Medium 1000 / HiRes 3000)
  3. Dialog shows estimated peak VRAM vs free VRAM (probe + warning if tight)
  4. Generate button enabled when source + preset both set
  5. Progress bar + cancel during run
  6. On success: auto-load the generated JSON into the main window's
     preview panel (caller wires this via the dialog's `output_path`
     attribute after exec returns Accepted)

Cancel mid-run cleanly terminates the subprocess (Phase 3 plumbing).
"""
from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Qt, QThread
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QFrame,
    QHBoxLayout, QLabel, QLineEdit, QMessageBox, QProgressBar, QPushButton,
    QSpinBox, QVBoxLayout,
)

from forza_abyss_painter.gui.gpu_gen_worker import GpuGenWorker, build_run_config
from forza_abyss_painter.gui.gpu_preflight import gpu_run_preflight
from forza_abyss_painter.runtime.nvidia_smi import probe_free_vram
from forza_abyss_painter.runtime.torch_installer import embedded_python_exe
from forza_abyss_painter.shapegen.gpu.vram_planner import (
    estimate_peak_vram_gib,
    recommend_max_resolution,
)


# Local-GPU preset table. Mirrors the Colab notebook lineup but with
# defaults tuned for consumer cards. Each entry: label, num_shapes,
# max_resolution, random_samples, est_peak_vram_gib.
# Tuned conservative — consumer cards often share VRAM with FH6 or
# other apps. Estimates assume bbox-local scoring (the production path).
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


class GenerateLocallyDialog(QDialog):
    """Modal: pick source + preset, run shape-gen in a worker thread,
    return path to generated JSON via `self.output_path` after Accepted.
    """

    def __init__(
        self,
        parent=None,
        initial_source_path: Path | None = None,
        gpu_budget_gib: float = 24.0,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Generate shapes locally (GPU)")
        self.setModal(True)
        self.setMinimumWidth(620)

        self.source_path: Path | None = None
        self.output_path: Path | None = None     # set on successful run
        self._selected_preset_idx = 0            # default: Lineart
        # VRAM budget for the preflight gate. Caller passes the value from
        # SettingsPanel.selected_vram_budget_gib() so the gate knows how
        # tight the card is. Default 24.0 keeps headless tests working;
        # the real call site in main_window overrides it.
        self._gpu_budget_gib: float = float(gpu_budget_gib)

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(12)

        # Header
        hdr = QLabel("Generate shapes locally")
        hf = QFont(); hf.setBold(True); hf.setPointSize(13)
        hdr.setFont(hf)
        root.addWidget(hdr)

        sub = QLabel(
            "Runs the GPU shape-generator on your local CUDA card. Output JSON "
            "loads automatically into the preview after generation."
        )
        sub.setStyleSheet("color: #999;")
        sub.setWordWrap(True)
        root.addWidget(sub)

        # Form: source image picker, preset dropdown, output location.
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        # Source image row.
        src_row = QHBoxLayout()
        self.source_field = QLineEdit(self)
        self.source_field.setPlaceholderText("Pick a PNG or JPG…")
        self.source_field.setReadOnly(True)
        src_row.addWidget(self.source_field)
        self.source_browse_btn = QPushButton("Browse…", self)
        self.source_browse_btn.clicked.connect(self._on_browse_source)
        src_row.addWidget(self.source_browse_btn)
        src_wrap = QFrame(self)
        src_wrap.setLayout(src_row)
        form.addRow("Source image:", src_wrap)

        # Preset dropdown.
        self.preset_combo = QComboBox(self)
        for p in LOCAL_PRESETS:
            live_peak = estimate_peak_vram_gib(
                K=int(p["random_samples"]),
                bbox_local=True,
                max_resolution=int(p["max_resolution"]),
            )
            self.preset_combo.addItem(
                f"{p['label']}  (~{live_peak:.0f} GiB peak)",
                userData=p,
            )
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        form.addRow("Preset:", self.preset_combo)

        # Output location (defaults to source dir, same stem + _N.json).
        self.output_field = QLineEdit(self)
        self.output_field.setPlaceholderText("(defaults to source folder)")
        form.addRow("Output to:", self.output_field)

        # Checkpoint cadence (snapshot every N shapes). GPU min 100 per
        # #snapshot-resume §6 — power users can raise to 1000 to reduce
        # disk writes on big runs.
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

        root.addLayout(form)

        # Preset description box. Created BEFORE vram_info so the order of
        # widget addition reads top-to-bottom in the dialog layout. The
        # initial _on_preset_changed() populate call comes LAST (after all
        # widgets that the refresh touches are constructed) to avoid an
        # init-order AttributeError on self.vram_info.
        self.preset_desc = QLabel("")
        self.preset_desc.setWordWrap(True)
        self.preset_desc.setStyleSheet(
            "background: #1a0a1f; border: 1px solid #3a2555; border-radius: 4px; "
            "padding: 8px; color: #cccccc; font-size: 11px;"
        )
        root.addWidget(self.preset_desc)

        # VRAM probe + warning area. Populated by _refresh_vram_estimate
        # at construction and every preset change.
        self.vram_info = QLabel("")
        self.vram_info.setWordWrap(True)
        self.vram_info.setStyleSheet("color: #888; font-size: 11px;")
        root.addWidget(self.vram_info)

        # Both target widgets exist now — safe to populate.
        self._on_preset_changed(0)

        # Progress (hidden until Generate clicked).
        self.progress = QProgressBar(self)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setVisible(False)
        root.addWidget(self.progress)
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #888; font-size: 11px;")
        self.status_label.setVisible(False)
        root.addWidget(self.status_label)

        # Buttons.
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.cancel_btn = QPushButton("Cancel", self)
        self.cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(self.cancel_btn)
        self.generate_btn = QPushButton("Generate", self)
        self.generate_btn.setDefault(True)
        self.generate_btn.setEnabled(False)   # gated on source-picked
        self.generate_btn.clicked.connect(self._on_generate_clicked)
        btn_row.addWidget(self.generate_btn)
        root.addLayout(btn_row)

        # Pre-fill source from #85 re-shape-gen flow if caller provided it
        # AND the file exists. Missing-file case falls through silently
        # so the user re-picks (avoids running on a stale path).
        if initial_source_path is not None and Path(initial_source_path).is_file():
            self.source_path = Path(initial_source_path)
            self.source_field.setText(str(self.source_path))
            self.generate_btn.setEnabled(True)
            preset = self.preset_combo.currentData()
            if preset:
                stem = self.source_path.stem
                suggested = (self.source_path.parent /
                             f"{stem}_{preset['num_shapes']}.json")
                self.output_field.setPlaceholderText(str(suggested))

    # ----------------------------------------------------- ui event handlers

    def _on_browse_source(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(
            self, "Pick a source image",
            "", "Images (*.png *.jpg *.jpeg *.webp)",
        )
        if path:
            self.source_path = Path(path)
            self.source_field.setText(path)
            self.generate_btn.setEnabled(True)
            # Suggest default output path next to source.
            if not self.output_field.text():
                preset = self.preset_combo.currentData()
                stem = self.source_path.stem
                suggested = (self.source_path.parent /
                             f"{stem}_{preset['num_shapes']}.json")
                self.output_field.setPlaceholderText(str(suggested))

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

        live_peak = estimate_peak_vram_gib(
            K=int(preset["random_samples"]),
            bbox_local=True,
            max_resolution=int(preset["max_resolution"]),
        )
        self.preset_desc.setText(
            f"<b>{preset['label']}</b><br>"
            f"{preset['description']}<br><br>"
            f"<b>Settings:</b> "
            f"max_resolution={preset['max_resolution']}, "
            f"random_samples={preset['random_samples']}, "
            f"estimated peak VRAM: ~{live_peak:.0f} GiB (full pipeline)"
            f"{rec_line}"
        )
        self.preset_desc.setTextFormat(Qt.RichText)
        self._refresh_vram_estimate()
        if self.source_path and preset:
            stem = self.source_path.stem
            suggested = (self.source_path.parent /
                         f"{stem}_{preset['num_shapes']}.json")
            self.output_field.setPlaceholderText(str(suggested))

    def _refresh_vram_estimate(self) -> None:
        """Probe free VRAM via the runtime (if installed) and compare to the
        preset's estimated peak. Surface a clear OK/tight/risky label."""
        preset = self.preset_combo.currentData()
        if not preset:
            self.vram_info.setText("")
            return
        # Phase 2: VRAM probe needs the installed runtime to import torch.
        # Phase 3 will subprocess-call torch_runner to do the probe. For
        # now show the live K-only peak estimate, with a generic
        # recommendation.
        est = estimate_peak_vram_gib(
            K=int(preset["random_samples"]),
            bbox_local=True,
            max_resolution=int(preset["max_resolution"]),
        )
        self.vram_info.setText(
            f"Estimated peak VRAM: <b>~{est:.0f} GiB</b> (peak). "
            f"Make sure your card has at least that much FREE (close FH6 + "
            f"other GPU apps if tight). Phase 3 will probe your card "
            f"directly and warn before launch."
        )
        self.vram_info.setTextFormat(Qt.RichText)

    def values(self) -> dict:
        """Return the preset dict for the currently selected item.

        Chunked-K handles VRAM fit at scoring time inside the engine, so
        the preflight gate never modifies the preset and .values() flows
        through unchanged.
        """
        preset = self.preset_combo.currentData()
        if preset is None:
            return {}
        return dict(preset)

    def _on_generate_clicked(self) -> None:
        """Lock the form, spawn the GPU subprocess via GpuGenWorker on a
        QThread, route its IPC events through Qt signals to update the
        progress bar + status label. On done: read the output JSON path
        from the event, set self.output_path, accept the dialog.
        On error: surface a modal, leave the dialog open so the user
        can adjust + retry.
        """
        if not self.source_path:
            return
        # Centralized chunk-aware preflight (chunked-K Task 2). The
        # helper probes free VRAM, asks the chunk-aware estimator for
        # the actual runtime peak, then blocks/warns/proceeds. The
        # preset is NEVER modified -- chunking handles fit inside the
        # engine at scoring time, so .values() flows through unchanged.
        preset_baked = self.preset_combo.currentData()
        proceed, _info = gpu_run_preflight(
            parent=self,
            preset=preset_baked,
            budget_gib=float(self._gpu_budget_gib),
            context="Generate locally",
        )
        if not proceed:
            return
        preset = self.values()
        out_path = self._resolve_output_path(preset)

        # Write the IPC config alongside the source image so it's easy
        # to find for post-mortem inspection if anything goes wrong.
        config = build_run_config(
            self.source_path, out_path, preset,
            sticker_mode=False,   # TODO: tie to a sticker checkbox once added
            checkpoint_every=int(self.checkpoint_every_spinbox.value()),
        )
        config_path = self.source_path.parent / (
            f".{self.source_path.stem}_gpu_config.json"
        )
        try:
            config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        except OSError as exc:
            QMessageBox.critical(
                self, "Couldn't start",
                f"Failed to write GPU config to {config_path}:\n\n"
                f"{type(exc).__name__}: {exc}",
            )
            return

        # Pre-flight: embedded python must exist. If the user is here it
        # means the install path was confirmed by `prompt_install_or_use_existing`
        # but a partial install (or someone manually deleting the runtime
        # dir between sessions) can drop the binary while leaving the
        # marker. Catch BEFORE spawning so the error message is precise.
        py = embedded_python_exe()
        if not py.exists():
            QMessageBox.critical(
                self, "GPU runtime missing",
                f"Embedded Python not found at {py}.\n\n"
                f"Re-run Tools → Generate shapes locally and accept the "
                f"runtime install when prompted.",
            )
            return

        self._lock_ui_for_run()
        self._cancel_requested = False

        # Build worker + move to a dedicated QThread so the GUI stays
        # responsive while torch crunches. Hold both as instance attrs
        # so they don't garbage-collect mid-run.
        self._thread = QThread(self)
        self._worker = GpuGenWorker(
            embedded_python_exe=py,
            config_path=config_path,
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.started.connect(self._on_worker_started)
        # Treat fine-grained progress + checkpoint events identically for
        # the progress bar — both carry (shape_count, total). The dialog
        # doesn't currently surface checkpoint shape lists; future preview
        # render would subscribe to the checkpoint signal specifically.
        self._worker.progress.connect(self._on_worker_progress)
        self._worker.checkpoint.connect(self._on_worker_progress)
        self._worker.done.connect(self._on_worker_done)
        self._worker.error.connect(self._on_worker_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    # ----- worker-event slots (run on the GUI thread) -----

    def _on_worker_started(self, cfg_summary: dict) -> None:
        self.status_label.setText(
            f"Running on GPU — preset: "
            f"{cfg_summary.get('preset_label', '(unknown)')}, "
            f"target {cfg_summary.get('num_shapes', '?')} shapes"
        )

    def _on_worker_progress(self, shape_count: int, total: int) -> None:
        if total > 0:
            pct = max(0, min(100, int(100 * shape_count / total)))
            self.progress.setValue(pct)
        self.status_label.setText(f"Shape {shape_count} of {total}")

    def _on_worker_done(self, output_path: str, shape_count: int) -> None:
        self.output_path = Path(output_path)
        self.status_label.setText(
            f"Done — {shape_count} shapes written to {Path(output_path).name}"
        )
        self.progress.setValue(100)
        # Brief pause before auto-accept so the user sees the success
        # state. Qt's accept() can fire immediately — but the dialog
        # closing instantly looks like a crash; let the success render.
        from PySide6.QtCore import QTimer
        QTimer.singleShot(500, self.accept)

    def _on_worker_error(self, stage: str, message: str) -> None:
        # Re-enable form so the user can adjust + retry without re-opening
        # the dialog from scratch. Cancelled stage doesn't pop a modal
        # (user already knows what they did).
        self._unlock_ui_for_run()
        if stage != "cancelled":
            QMessageBox.critical(
                self, f"GPU generation failed — {stage}",
                f"Stage: {stage}\n\n{message}",
            )
        self.status_label.setText(
            f"Failed at {stage} — adjust settings and try again."
            if stage != "cancelled"
            else "Cancelled."
        )

    # ----- UI state helpers -----

    def _resolve_output_path(self, preset: dict) -> Path:
        """Honor the user's typed output path if set; else fall back to
        the placeholder (source stem + _N.json next to source)."""
        text = self.output_field.text().strip()
        if text:
            return Path(text)
        stem = self.source_path.stem
        return self.source_path.parent / f"{stem}_{preset['num_shapes']}.json"

    def _lock_ui_for_run(self) -> None:
        self.source_browse_btn.setEnabled(False)
        self.preset_combo.setEnabled(False)
        self.output_field.setEnabled(False)
        self.generate_btn.setEnabled(False)
        self.cancel_btn.setText("Cancel run")
        # Wire Cancel button to worker.cancel — only valid during a run.
        # Disconnect on _unlock so the next Cancel reverts to reject().
        try:
            self.cancel_btn.clicked.disconnect()
        except (RuntimeError, TypeError):
            pass
        self.cancel_btn.clicked.connect(self._on_cancel_run)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.status_label.setVisible(True)
        self.status_label.setText("Preparing GPU runtime…")

    def _unlock_ui_for_run(self) -> None:
        self.source_browse_btn.setEnabled(True)
        self.preset_combo.setEnabled(True)
        self.output_field.setEnabled(True)
        self.generate_btn.setEnabled(self.source_path is not None)
        self.cancel_btn.setText("Cancel")
        try:
            self.cancel_btn.clicked.disconnect()
        except (RuntimeError, TypeError):
            pass
        self.cancel_btn.clicked.connect(self.reject)

    def _on_cancel_run(self) -> None:
        """Cancel-during-run: ask the worker to terminate the subprocess.
        The worker emits an error event with stage='cancelled' which
        routes through _on_worker_error → _unlock_ui_for_run."""
        if hasattr(self, "_worker") and self._worker is not None:
            self._worker.cancel()


def open_generate_dialog_if_runtime_ready(
    parent, gpu_budget_gib: float = 24.0,
) -> Path | None:
    """Convenience entry point used by the Tools menu. Checks runtime is
    installed; if not, prompts the install flow; if so (or after install
    completes), opens the generate dialog. Returns the generated JSON
    path on success, None on cancel/skip.

    `gpu_budget_gib` flows through to the dialog so the preflight gate
    knows how tight the card is.
    """
    from forza_abyss_painter.gui.runtime_install_dialog import (
        prompt_install_or_use_existing,
    )
    if not prompt_install_or_use_existing(parent):
        return None
    dlg = GenerateLocallyDialog(parent, gpu_budget_gib=gpu_budget_gib)
    if dlg.exec() == QDialog.Accepted and dlg.output_path:
        return dlg.output_path
    return None
