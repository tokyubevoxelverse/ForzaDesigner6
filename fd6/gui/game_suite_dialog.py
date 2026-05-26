"""Modal popup: pick which game-family FD6 is currently targeting.

Shown:
  - On first-ever launch (no saved suite_mode in QSettings)
  - When the user picks 'Change Game Suite…' from the Customizations menu

Hidden:
  - On every subsequent image upload — the saved mode is reused silently.

Layout: 4 tiles in a 2x2 grid. Two tiles are enabled (Forza, AC); two are
"Coming Soon" placeholders (NFS, Crew). Clicking an enabled tile selects
that suite and closes the dialog. The disabled tiles are visually present
so users see the roadmap.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QDialog, QFrame, QGridLayout, QLabel, QPushButton, QVBoxLayout, QWidget,
)

from fd6.suite import SuiteMode, SUITE_DISPLAY


class SuiteTile(QFrame):
    """One of the four suite-picker tiles."""

    clicked = Signal(object)  # SuiteMode

    def __init__(self, mode: SuiteMode, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        meta = SUITE_DISPLAY[mode]
        self._mode = mode
        self._enabled = bool(meta["enabled"])
        self.setFixedSize(280, 130)
        self.setCursor(Qt.PointingHandCursor if self._enabled else Qt.ForbiddenCursor)
        self.setStyleSheet(self._stylesheet(self._enabled))
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(6)
        title = QLabel(meta["label"], self)
        tf = QFont(); tf.setBold(True); tf.setPointSize(13)
        title.setFont(tf)
        title.setStyleSheet("color: #f0f0f0;" if self._enabled else "color: #888;")
        sub = QLabel(meta["subtitle"], self)
        sub.setWordWrap(True)
        sub.setStyleSheet("color: #aaa;" if self._enabled else "color: #666;")
        layout.addWidget(title)
        layout.addWidget(sub, stretch=1)
        if not self._enabled:
            tag = QLabel("Coming soon", self)
            tag.setStyleSheet(
                "QLabel { background: #2a1f0a; border: 1px solid #b07a00;"
                " color: #f1c40f; padding: 2px 6px; border-radius: 4px;"
                " font-size: 10px; font-weight: bold; }"
            )
            tag.setAlignment(Qt.AlignLeft)
            tag.setFixedHeight(20)
            layout.addWidget(tag)

    @staticmethod
    def _stylesheet(enabled: bool) -> str:
        if enabled:
            return (
                "SuiteTile { background: #1f2228; border: 1px solid #444; border-radius: 8px; }"
                "SuiteTile:hover { background: #2a2f38; border-color: #888; }"
            )
        return (
            "SuiteTile { background: #15171a; border: 1px solid #2a2a2a; border-radius: 8px; }"
        )

    def mousePressEvent(self, event) -> None:
        if self._enabled and event.button() == Qt.LeftButton:
            self.clicked.emit(self._mode)
        super().mousePressEvent(event)


class GameSuiteDialog(QDialog):
    """Modal suite picker. Caller does dlg.exec(); reads dlg.selected after."""

    def __init__(self, parent=None, current: SuiteMode | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("FD6 — Choose Game Suite")
        self.setModal(True)
        self.setMinimumWidth(620)
        self.selected: SuiteMode | None = current

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 16)
        root.setSpacing(14)

        header = QLabel("Which game family are you working with?")
        hf = QFont(); hf.setBold(True); hf.setPointSize(13)
        header.setFont(hf)
        root.addWidget(header)

        sub = QLabel(
            "Pick your target game family. FD6 remembers your choice — you can "
            "switch later from View → Customizations → Change Game Suite."
        )
        sub.setWordWrap(True)
        sub.setStyleSheet("color: #999;")
        root.addWidget(sub)

        grid = QGridLayout()
        grid.setSpacing(12)
        # Two enabled tiles on top row, two coming-soon on bottom row.
        tiles_order = [SuiteMode.FORZA, SuiteMode.AC, SuiteMode.NFS, SuiteMode.CREW]
        for i, mode in enumerate(tiles_order):
            tile = SuiteTile(mode, self)
            tile.clicked.connect(self._on_tile_clicked)
            grid.addWidget(tile, i // 2, i % 2)
        root.addLayout(grid)

        # Cancel button — only if a `current` mode was passed (re-selection
        # from the menu). On first launch we hide it so users HAVE to pick.
        if current is not None:
            cancel_row = QFrame(self)
            cl = QVBoxLayout(cancel_row)
            cl.setContentsMargins(0, 0, 0, 0)
            cancel = QPushButton("Cancel", cancel_row)
            cancel.clicked.connect(self.reject)
            cl.addWidget(cancel, alignment=Qt.AlignRight)
            root.addWidget(cancel_row)

    def _on_tile_clicked(self, mode: SuiteMode) -> None:
        self.selected = mode
        self.accept()
