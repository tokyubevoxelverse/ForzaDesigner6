"""Preview pane variant for AC mode — shows the planned per-slot textures
with left/right buttons to cycle through them.

Replaces the Forza two-pane (Source / Generation Preview) layout when AC
suite is active. Layout:
  ┌──────────────────────────────────────────┐
  │           [Source image preview]         │
  ├──────────────────────────────────────────┤
  │ ◀  Preview 1/3 — decals.png           ▶  │
  │           [Texture slot preview]         │
  └──────────────────────────────────────────┘

Source view is left as-is (shows the uploaded image). Slot views below
let users see exactly what FD6 will write into each ACC slot before
clicking Export.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QProgressBar, QPushButton, QVBoxLayout, QWidget, QSplitter,
)

from fd6.gui.widgets import ImageView


class TexturePreviewPanel(QWidget):
    """AC-mode preview pane. Holds source + cycling slot previews."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        splitter = QSplitter(Qt.Horizontal, self)
        self.source_view = ImageView("Source", self)
        self.preview_view = ImageView("Slot preview", self)
        splitter.addWidget(self.source_view)
        splitter.addWidget(self.preview_view)
        splitter.setSizes([500, 500])
        layout.addWidget(splitter, stretch=1)

        # Cycle controls — sits between the two image panes and the status bar.
        cycle_row = QHBoxLayout()
        cycle_row.setSpacing(8)
        self.prev_btn = QPushButton("◀", self)
        self.prev_btn.setFixedWidth(40)
        self.prev_btn.clicked.connect(self._cycle_prev)
        self.cycle_label = QLabel("(no slots yet)", self)
        self.cycle_label.setAlignment(Qt.AlignCenter)
        self.cycle_label.setStyleSheet("color: #aaa;")
        self.next_btn = QPushButton("▶", self)
        self.next_btn.setFixedWidth(40)
        self.next_btn.clicked.connect(self._cycle_next)
        cycle_row.addWidget(self.prev_btn)
        cycle_row.addWidget(self.cycle_label, stretch=1)
        cycle_row.addWidget(self.next_btn)
        layout.addLayout(cycle_row)

        # Status / progress row — mirrors the Forza preview's bottom strip
        # so the rest of MainWindow (status_label / progress) can stay generic.
        info_row = QHBoxLayout()
        self.status_label = QLabel("Idle.", self)
        self.status_label.setStyleSheet("color: #aaa;")
        self.progress = QProgressBar(self)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        info_row.addWidget(self.status_label, stretch=1)
        info_row.addWidget(self.progress, stretch=2)
        layout.addLayout(info_row)

        # Slot store
        self._slots: list[tuple[str, np.ndarray]] = []
        self._cur = 0
        self._update_cycle_ui()

    # ── public API ───────────────────────────────────────────────────────

    def set_source(self, path: str | Path) -> None:
        """Show the user's uploaded image in the Source pane."""
        self.source_view.set_path(str(path))
        # Clear previous slot previews — user uploaded a new source.
        self.set_slots([])

    def set_slots(self, slots: list[tuple[str, np.ndarray]]) -> None:
        """Update the cycling slot preview list.

        Each entry: (filename, HxWx4 uint8 RGBA array). Caller invokes after
        the AC settings change so the user can see what'll be written.
        """
        self._slots = list(slots)
        self._cur = 0
        if self._slots:
            self._show_current()
        else:
            self.preview_view.clear_image()
        self._update_cycle_ui()

    def reset(self) -> None:
        self._slots = []
        self._cur = 0
        self.source_view.clear_image()
        self.preview_view.clear_image()
        self.status_label.setText("Idle.")
        self.progress.setValue(0)
        self._update_cycle_ui()

    # ── private ──────────────────────────────────────────────────────────

    def _cycle_prev(self) -> None:
        if not self._slots:
            return
        self._cur = (self._cur - 1) % len(self._slots)
        self._show_current()
        self._update_cycle_ui()

    def _cycle_next(self) -> None:
        if not self._slots:
            return
        self._cur = (self._cur + 1) % len(self._slots)
        self._show_current()
        self._update_cycle_ui()

    def _show_current(self) -> None:
        if not self._slots:
            return
        _name, rgba = self._slots[self._cur]
        self.preview_view.set_numpy(rgba)

    def _update_cycle_ui(self) -> None:
        n = len(self._slots)
        if n == 0:
            self.cycle_label.setText("(no slot previews yet)")
            self.prev_btn.setEnabled(False)
            self.next_btn.setEnabled(False)
            return
        name = self._slots[self._cur][0]
        self.cycle_label.setText(f"Slot {self._cur + 1} of {n}  —  {name}")
        # Always enable cycling buttons (we wrap-around).
        self.prev_btn.setEnabled(n > 1)
        self.next_btn.setEnabled(n > 1)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Left:
            self._cycle_prev()
        elif event.key() == Qt.Key_Right:
            self._cycle_next()
        else:
            super().keyPressEvent(event)
