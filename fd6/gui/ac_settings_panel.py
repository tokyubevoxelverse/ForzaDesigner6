"""AC-mode settings panel — replaces the Forza generation panel when the
SuiteMode is AC. Surfaces car picker, output resolution, slot assignment,
and the Export button.

Strict separation: this panel imports from fd6.ac.* only — never from
fd6.inject, fd6.shapegen, or fd6.io.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

from fd6.ac.car_catalog import CarEntry, list_cars
from fd6.ac.profiles import ACTitleProfile, list_profiles
from fd6.gui.ac_paint_picker import ACPaintPanel


class ACSettingsPanel(QWidget):
    """Right-side panel for AC mode. Emits export_clicked when user is ready."""

    target_changed = Signal(object)        # ACTitleProfile
    export_clicked = Signal(dict)           # see _gather_export_config()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Target title dropdown
        target_row = QHBoxLayout()
        target_row.addWidget(QLabel("Target:"))
        self.target_combo = QComboBox(self)
        self._title_profiles = list_profiles()
        for prof in self._title_profiles:
            label = prof.label + ("" if prof.implemented else " (Coming Soon)")
            self.target_combo.addItem(label, prof.key)
        self.target_combo.setCurrentIndex(0)
        self.target_combo.currentIndexChanged.connect(self._on_target_changed)
        target_row.addWidget(self.target_combo, stretch=1)
        layout.addLayout(target_row)

        # Car picker
        car_row = QHBoxLayout()
        car_row.addWidget(QLabel("Car:"))
        self.car_combo = QComboBox(self)
        car_row.addWidget(self.car_combo, stretch=1)
        layout.addLayout(car_row)

        # Output resolution + aspect
        res_group = QGroupBox("Output", self)
        rf = QFormLayout(res_group)
        self.resolution_combo = QComboBox(self)
        for px in (4096, 2048, 1024):
            self.resolution_combo.addItem(f"{px} × {px}", px)
        self.resolution_combo.setCurrentIndex(0)
        rf.addRow("Resolution:", self.resolution_combo)
        self.aspect_combo = QComboBox(self)
        for label in ("Auto", "1:1", "1:2", "1:4"):
            self.aspect_combo.addItem(label, label.lower())
        rf.addRow("Aspect ratio:", self.aspect_combo)
        layout.addWidget(res_group)

        # Slot assignment
        slot_group = QGroupBox("Decal slot assignment", self)
        sl = QVBoxLayout(slot_group)
        self.auto_slot_cb = QCheckBox("Auto-assign slots (recommended)", slot_group)
        self.auto_slot_cb.setChecked(True)
        self.auto_slot_cb.stateChanged.connect(self._on_auto_slot_changed)
        sl.addWidget(self.auto_slot_cb)
        # Main decals + sponsors checkboxes — disabled until auto is unchecked.
        self.main_slot_cbs: dict[str, QCheckBox] = {}
        main_sub = QVBoxLayout()
        main_label = QLabel("Main decals:", slot_group)
        main_label.setStyleSheet("color: #aaa; padding-left: 8px;")
        main_sub.addWidget(main_label)
        for slot in ("decals.png", "decals_0.png", "decals_1.png"):
            cb = QCheckBox(slot, slot_group)
            cb.setEnabled(False)
            cb.setChecked(slot == "decals.png")
            cb.setStyleSheet("padding-left: 24px;")
            main_sub.addWidget(cb)
            self.main_slot_cbs[slot] = cb
        sl.addLayout(main_sub)
        self.sponsor_slot_cbs: dict[str, QCheckBox] = {}
        spon_sub = QVBoxLayout()
        spon_label = QLabel("Sponsors (optional):", slot_group)
        spon_label.setStyleSheet("color: #aaa; padding-left: 8px;")
        spon_sub.addWidget(spon_label)
        for slot in ("sponsors.png",):
            cb = QCheckBox(slot, slot_group)
            cb.setEnabled(False)
            cb.setStyleSheet("padding-left: 24px;")
            spon_sub.addWidget(cb)
            self.sponsor_slot_cbs[slot] = cb
        sl.addLayout(spon_sub)
        layout.addWidget(slot_group)

        # Livery name + display name + race number
        meta_group = QGroupBox("Livery metadata", self)
        mf = QFormLayout(meta_group)
        self.team_name = QLineEdit(self)
        self.team_name.setPlaceholderText("e.g. MyAwesomeLivery")
        self.team_name.setToolTip(
            "What to name the livery folder FD6 creates under "
            "Documents\\Assetto Corsa Competizione\\Customs\\Liveries\\. "
            "ACC reads each folder as one custom livery — pick anything that's "
            "meaningful to you (no spaces or special characters)."
        )
        mf.addRow("Livery name (folder):", self.team_name)
        self.display_name = QLineEdit(self)
        self.display_name.setPlaceholderText("(optional — defaults to livery name)")
        self.display_name.setToolTip(
            "Human-readable name shown inside ACC's livery editor. Leave blank "
            "to reuse the livery folder name above."
        )
        mf.addRow("Display name:", self.display_name)
        self.race_number = QSpinBox(self)
        self.race_number.setRange(0, 998)
        self.race_number.setValue(99)
        mf.addRow("Race number:", self.race_number)
        layout.addWidget(meta_group)

        # Paint & materials picker (ACC only — other AC titles use different schemas).
        self.paint_panel = ACPaintPanel(self)
        layout.addWidget(self.paint_panel)

        # Export button
        self.export_btn = QPushButton("Export to Assetto Corsa Competizione")
        self.export_btn.setMinimumHeight(40)
        self.export_btn.clicked.connect(self._on_export_clicked)
        layout.addWidget(self.export_btn)

        layout.addStretch()

        # Populate car list for the default target.
        self._refresh_car_list()

    # ── public ───────────────────────────────────────────────────────────

    def selected_profile(self) -> ACTitleProfile:
        idx = self.target_combo.currentIndex()
        return self._title_profiles[idx] if 0 <= idx < len(self._title_profiles) else self._title_profiles[0]

    # ── internal slots ───────────────────────────────────────────────────

    def _on_target_changed(self, idx: int) -> None:
        if not 0 <= idx < len(self._title_profiles):
            return
        prof = self._title_profiles[idx]
        # Beta/not-implemented titles disable the Export button.
        self.export_btn.setEnabled(prof.implemented)
        if not prof.implemented:
            self.export_btn.setText(f"{prof.label} — Coming Soon")
        else:
            clean = prof.label
            self.export_btn.setText(f"Export to {clean}")
        self._refresh_car_list()
        self.target_changed.emit(prof)

    def _on_auto_slot_changed(self, state: int) -> None:
        enabled = state != Qt.Checked
        for cb in self.main_slot_cbs.values():
            cb.setEnabled(enabled)
        for cb in self.sponsor_slot_cbs.values():
            cb.setEnabled(enabled)

    def _refresh_car_list(self) -> None:
        self.car_combo.clear()
        prof = self.selected_profile()
        cars = list_cars(prof)
        if not cars:
            self.car_combo.addItem("(no cars found — install game or open livery editor once)", "")
            self.car_combo.setEnabled(False)
            return
        self.car_combo.setEnabled(True)
        for c in cars:
            self.car_combo.addItem(f"[{c.category}]  {c.display_name}", c.car_model)

    def _gather_export_config(self) -> dict:
        prof = self.selected_profile()
        auto = self.auto_slot_cb.isChecked()
        manual_main = [s for s, cb in self.main_slot_cbs.items() if cb.isChecked()]
        manual_sponsors = [s for s, cb in self.sponsor_slot_cbs.items() if cb.isChecked()]
        return {
            "profile": prof,
            "car_model": str(self.car_combo.currentData() or ""),
            "team_name": self.team_name.text().strip(),
            "display_name": self.display_name.text().strip(),
            "race_number": int(self.race_number.value()),
            "resolution": int(self.resolution_combo.currentData() or 4096),
            "aspect": str(self.aspect_combo.currentData() or "auto"),
            "auto_slot": auto,
            "manual_main_slots": manual_main,
            "manual_sponsor_slots": manual_sponsors,
            "paint": self.paint_panel.gather(),
        }

    def _on_export_clicked(self) -> None:
        self.export_clicked.emit(self._gather_export_config())
