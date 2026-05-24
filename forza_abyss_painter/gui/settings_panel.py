from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QPushButton, QSpinBox, QVBoxLayout, QWidget
)

from forza_abyss_painter.shapegen.profile import Profile, load_profile_from_file, list_bundled_profiles


SHAPE_TYPE_CHOICES = [
    ("rotated_ellipse", "Rotated Ellipse (default)"),
    ("rectangle", "Rectangle (coming soon)"),
    ("rotated_rectangle", "Rotated Rectangle (coming soon)"),
    ("ellipse", "Ellipse (coming soon)"),
    ("circle", "Circle (coming soon)"),
    ("triangle", "Triangle (coming soon)"),
]


class SettingsPanel(QWidget):
    """Profile picker + advanced knobs. Emits profile_changed when the user edits anything."""

    profile_changed = Signal(object)  # Profile
    start_clicked = Signal()
    pause_clicked = Signal()
    stop_clicked = Signal()
    inject_clicked = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Profile picker
        prof_row = QHBoxLayout()
        prof_label = QLabel("Profile:")
        prof_label.setToolTip(
            "A profile is a saved bundle of all the settings below. Pick one to "
            "fill the values automatically — for example, '_default' uses 3000 "
            "shapes at 1200 px which is a good general-purpose starting point. "
            "Adjust any setting after selecting a profile and your changes stay "
            "for this session."
        )
        prof_row.addWidget(prof_label)
        self.profile_combo = QComboBox(self)
        self.profile_combo.setToolTip(prof_label.toolTip())
        self._populate_profiles()
        self.profile_combo.currentIndexChanged.connect(self._on_profile_changed)
        prof_row.addWidget(self.profile_combo, stretch=1)
        layout.addLayout(prof_row)

        # Advanced group — every spinbox gets a plain-language tooltip so
        # non-technical users can hover and know what each number does.
        adv = QGroupBox("Advanced", self)
        form = QFormLayout(adv)
        self.stop_at = QSpinBox(); self.stop_at.setRange(10, 50000); self.stop_at.setValue(3000)
        self.stop_at.setToolTip(
            "How many shapes to generate before stopping. Set this to match the "
            "number of layers in your open Forza vinyl group (typical sphere "
            "templates have 500, 1500, or 3000 layers). If this number is "
            "larger than your template's layer count, the injection will fail "
            "because there aren't enough slots."
        )
        self.random_samples = QSpinBox(); self.random_samples.setRange(10, 50000); self.random_samples.setValue(1000)
        self.random_samples.setToolTip(
            "Per shape: how many random candidate shapes the generator tries "
            "before picking the best one. Higher = better quality but slower. "
            "1000 is a good default; drop to 500 for a fast preview, raise to "
            "2000+ for a final pass on a picky source image."
        )
        self.mutated_samples = QSpinBox(); self.mutated_samples.setRange(1, 5000); self.mutated_samples.setValue(200)
        self.mutated_samples.setToolTip(
            "After the random search picks a winner, how many small tweaks to "
            "try in order to refine it. Higher = each shape is positioned more "
            "precisely but generation is slower. 200 is a good default."
        )
        self.max_resolution = QSpinBox(); self.max_resolution.setRange(100, 4096); self.max_resolution.setValue(1200)
        self.max_resolution.setToolTip(
            "The biggest the image will be processed at, in pixels along the "
            "longer side. Higher = more accurate shape placement but uses way "
            "more memory and time. 1200 px is the sweet spot for most images. "
            "Push to 2048 or 4096 only if the source is highly detailed."
        )
        self.max_threads = QSpinBox(); self.max_threads.setRange(0, 64); self.max_threads.setValue(0)
        self.max_threads.setToolTip(
            "How many CPU cores the generator uses in parallel. Leave at 0 to "
            "let FD6 auto-pick a safe number based on your CPU and RAM. Only "
            "override this if you want to free up cores for something else "
            "while generation runs (e.g. set it to half your core count)."
        )
        self.preview_every = QSpinBox(); self.preview_every.setRange(1, 100); self.preview_every.setValue(1)
        self.preview_every.setToolTip(
            "How often to refresh the live preview pane during generation. "
            "1 = redraw after every shape (smoothest, slight CPU cost). "
            "10 = redraw every 10 shapes (faster, choppier). Doesn't affect "
            "the final result, only what you see while it's running."
        )
        # QFormLayout auto-creates QLabel widgets for the left column. Those
        # labels do NOT inherit tooltips from their paired field, so hovering
        # the text "Stop at shapes" would show nothing. Create the labels
        # explicitly and mirror each field's tooltip onto its label.
        for label_text, field in (
            ("Stop at shapes", self.stop_at),
            ("Random samples", self.random_samples),
            ("Mutated samples", self.mutated_samples),
            ("Max resolution (px)", self.max_resolution),
            ("Threads (0=auto)", self.max_threads),
            ("Preview every N", self.preview_every),
        ):
            row_label = QLabel(label_text, adv)
            row_label.setToolTip(field.toolTip())
            form.addRow(row_label, field)
            field.valueChanged.connect(self._on_adv_changed)
        layout.addWidget(adv)

        # Sticker mode toggle
        sticker_group = QGroupBox("Image options", self)
        sticker_group.setToolTip(
            "How FD6 should handle source images. Affects only PNGs with "
            "transparency — regular JPEG / PNG without alpha use the same "
            "code path either way."
        )
        sg_layout = QVBoxLayout(sticker_group)
        self.sticker_mode_cb = QCheckBox("Add white background to transparent images", sticker_group)
        self.sticker_mode_cb.setChecked(True)  # ON = current default behavior (composite onto white)
        self.sticker_mode_cb.setToolTip(
            "ON (default, recommended): see-through areas of a PNG get filled with "
            "white before generation. Use this for normal images.\n\n"
            "OFF (sticker mode): see-through areas stay see-through and shapes "
            "are only placed inside the visible part of the image. Use this for "
            "logos / stickers where you want the background of the vinyl to "
            "stay empty (the rest of the Forza vinyl group shows through)."
        )
        sg_layout.addWidget(self.sticker_mode_cb)
        layout.addWidget(sticker_group)

        # Shape types. Only rotated_ellipse is confirmed-working for the current
        # FH6 build; remaining primitives are disabled pending further work.
        supported_codes = {"rotated_ellipse"}
        supported_tooltips = {
            "rotated_ellipse": (
                "An oval that can be rotated to any angle. Fits organic / "
                "curvy content (faces, smoke, foliage) best."
            ),
        }
        types_group = QGroupBox("Shape types", self)
        types_group.setToolTip(
            "Which shapes the generator is allowed to use. When more than one "
            "is checked, FD6 rotates between them so each enabled type gets "
            "dedicated shape slots in the output."
        )
        tg_layout = QVBoxLayout(types_group)
        self._shape_checks: dict[str, QCheckBox] = {}
        generic_unsupported = "Not currently supported - planned for a future implementation"
        for code, label in SHAPE_TYPE_CHOICES:
            cb = QCheckBox(label, types_group)
            cb.setChecked(code == "rotated_ellipse")
            if code in supported_codes:
                cb.setToolTip(supported_tooltips.get(code, ""))
            else:
                cb.setEnabled(False)
                cb.setToolTip(generic_unsupported)
            cb.stateChanged.connect(self._on_adv_changed)
            tg_layout.addWidget(cb)
            self._shape_checks[code] = cb
        layout.addWidget(types_group)

        # Action buttons
        btn_row = QHBoxLayout()
        self.start_btn = QPushButton("Start"); self.start_btn.setMinimumHeight(36)
        self.start_btn.setToolTip(
            "Begin shape generation on the next image in the queue using the "
            "settings above. The preview pane shows the result building up "
            "shape-by-shape. You can press Pause to hold and Stop to abandon."
        )
        self.pause_btn = QPushButton("Pause"); self.pause_btn.setCheckable(True); self.pause_btn.setEnabled(False)
        self.pause_btn.setToolTip(
            "Temporarily pause generation. Click again to resume from where it "
            "left off — no shapes are lost."
        )
        self.stop_btn = QPushButton("Stop"); self.stop_btn.setEnabled(False)
        self.stop_btn.setToolTip(
            "Stop generation early. The shapes generated so far are kept and "
            "saved to JSON — you can still inject the partial result."
        )
        self.start_btn.clicked.connect(self.start_clicked.emit)
        self.pause_btn.clicked.connect(self.pause_clicked.emit)
        self.stop_btn.clicked.connect(self.stop_clicked.emit)
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.pause_btn)
        btn_row.addWidget(self.stop_btn)
        layout.addLayout(btn_row)

        # Target game picker — FH6 is the validated default. FH5/FH4 are beta.
        from forza_abyss_painter.inject.game_profiles import list_profiles
        target_row = QHBoxLayout()
        target_label = QLabel("Target:")
        target_label.setToolTip(
            "Which Forza title to inject into. FH6 is fully validated. "
            "FH5 / FH4 / FH3 use the same memory layout per public research but have "
            "not been independently verified — test on a throwaway vinyl group first."
        )
        target_row.addWidget(target_label)
        self.target_combo = QComboBox(self)
        self._target_profiles = list_profiles()
        for prof in self._target_profiles:
            self.target_combo.addItem(prof.label, prof.key)
        self.target_combo.setCurrentIndex(0)  # FH6 by default
        self.target_combo.setToolTip(
            "Which Forza title to inject into. FH6 is fully validated. "
            "FH5 / FH4 use the same memory layout per public research but have "
            "not been independently verified — test on a throwaway vinyl group first."
        )
        self.target_combo.currentIndexChanged.connect(self._on_target_changed)
        target_row.addWidget(self.target_combo, stretch=1)
        layout.addLayout(target_row)

        # Inject button — label updates with target selection
        self.inject_btn = QPushButton("Inject into Forza Horizon 6")
        self.inject_btn.setEnabled(False)
        self.inject_btn.setToolTip(
            "Push the most-recent generated/loaded shapes JSON into the selected Forza title's "
            "active vinyl group. Make sure the in-game vinyl editor is open with a fresh "
            "sphere-template group before clicking."
        )
        self.inject_btn.clicked.connect(self.inject_clicked.emit)
        layout.addWidget(self.inject_btn)

        layout.addStretch()

        # Apply initial profile
        self._on_profile_changed(self.profile_combo.currentIndex())

    def selected_target_profile_key(self) -> str:
        """Return the key ('fh6'/'fh5'/'fh4') of the currently picked injection target."""
        data = self.target_combo.currentData()
        return str(data) if data else "fh6"

    def _on_target_changed(self, idx: int) -> None:
        if idx < 0 or idx >= len(self._target_profiles):
            return
        prof = self._target_profiles[idx]
        # Strip the "(BETA)" suffix for the button label so it stays clean.
        clean_label = prof.label.replace(" (BETA)", "")
        self.inject_btn.setText(f"Inject into {clean_label}")
        if prof.beta:
            tooltip = (
                f"BETA target: {prof.label}.\n\n{prof.beta_note}\n\n"
                "Make sure the in-game vinyl editor is open with a fresh sphere-template group."
            )
        else:
            tooltip = (
                "Push the most-recent generated/loaded shapes JSON into the selected Forza title's "
                "active vinyl group. Make sure the in-game vinyl editor is open with a fresh "
                "sphere-template group before clicking."
            )
        self.inject_btn.setToolTip(tooltip)

    def _populate_profiles(self) -> None:
        self.profile_combo.clear()
        for path in list_bundled_profiles():
            self.profile_combo.addItem(path.stem, str(path))
        if self.profile_combo.count() == 0:
            self.profile_combo.addItem("default", "")

    def _on_profile_changed(self, idx: int) -> None:
        path = self.profile_combo.itemData(idx)
        if not path:
            return
        try:
            prof = load_profile_from_file(path)
        except Exception:
            return
        # Mirror into advanced widgets without re-emitting per-spinbox.
        for w in (self.stop_at, self.random_samples, self.mutated_samples, self.max_resolution, self.max_threads, self.preview_every):
            w.blockSignals(True)
        self.stop_at.setValue(prof.stop_at)
        self.random_samples.setValue(prof.random_samples)
        self.mutated_samples.setValue(prof.mutated_samples)
        self.max_resolution.setValue(prof.max_resolution)
        self.max_threads.setValue(prof.max_threads)
        self.preview_every.setValue(prof.preview_every)
        for w in (self.stop_at, self.random_samples, self.mutated_samples, self.max_resolution, self.max_threads, self.preview_every):
            w.blockSignals(False)
        for code, cb in self._shape_checks.items():
            cb.blockSignals(True)
            # Disabled shape types (everything except rotated_ellipse in v0.3.5)
            # stay unchecked regardless of what the loaded profile prefers,
            # so a profile that requests triangles can't sneak past the gray-out.
            if cb.isEnabled():
                cb.setChecked(code in prof.shape_types)
            else:
                cb.setChecked(False)
            cb.blockSignals(False)
        self.profile_changed.emit(self.build_profile())

    def _on_adv_changed(self, *_args) -> None:
        self.profile_changed.emit(self.build_profile())

    def build_profile(self) -> Profile:
        idx = self.profile_combo.currentIndex()
        path = self.profile_combo.itemData(idx) or ""
        base = Profile(name=self.profile_combo.itemText(idx) or "custom")
        if path:
            try:
                base = load_profile_from_file(path)
            except Exception:
                pass
        base.stop_at = self.stop_at.value()
        base.random_samples = self.random_samples.value()
        base.mutated_samples = self.mutated_samples.value()
        base.max_resolution = self.max_resolution.value()
        base.max_threads = self.max_threads.value()
        base.preview_every = self.preview_every.value()
        base.shape_types = [code for code, cb in self._shape_checks.items() if cb.isChecked()] or ["rotated_ellipse"]
        return base

    def set_running(self, running: bool) -> None:
        self.start_btn.setEnabled(not running)
        self.pause_btn.setEnabled(running)
        self.stop_btn.setEnabled(running)
