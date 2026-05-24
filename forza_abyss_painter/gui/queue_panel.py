from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPushButton, QVBoxLayout, QWidget
)


@dataclass
class QueueItem:
    path: Path
    status: str = "queued"  # queued | running | done | error


STATUS_ICON = {"queued": "⏳", "running": "▶", "done": "✓", "error": "✗"}


class QueueRow(QWidget):
    """One row of the queue: status-icon + filename + X (remove) button.

    Emits `remove_requested(path)` when the X is clicked. The X is auto-disabled
    while the item is `running` so the user can't pull the rug out from under
    the worker mid-generation.
    """

    remove_requested = Signal(object)  # Path

    def __init__(self, path: Path, parent=None) -> None:
        super().__init__(parent)
        self._path = path
        self._status = "queued"
        h = QHBoxLayout(self)
        h.setContentsMargins(6, 2, 4, 2)
        h.setSpacing(8)
        self.label = QLabel(f"{STATUS_ICON['queued']} {path.name}", self)
        # No bold etc — let the global QSS theme drive label styling
        h.addWidget(self.label, stretch=1)
        self.x_btn = QPushButton("✕", self)
        self.x_btn.setFixedSize(QSize(22, 22))
        self.x_btn.setToolTip("Remove from queue")
        self.x_btn.setCursor(Qt.PointingHandCursor)
        self.x_btn.setStyleSheet(
            "QPushButton { background: transparent; border: 1px solid #555;"
            " color: #ccc; border-radius: 11px; padding: 0; font-size: 12px; }"
            "QPushButton:hover { background: rgba(255, 80, 80, 0.18);"
            " border-color: #ff5555; color: #fff; }"
            "QPushButton:disabled { color: #666; border-color: #333; }"
        )
        self.x_btn.clicked.connect(lambda: self.remove_requested.emit(self._path))
        h.addWidget(self.x_btn)

    def set_status(self, status: str) -> None:
        self._status = status
        self.label.setText(f"{STATUS_ICON.get(status, '?')} {self._path.name}")
        # While running, don't let the user yank the row mid-write.
        self.x_btn.setEnabled(status != "running")


class QueuePanel(QWidget):
    cleared = Signal()
    item_removed = Signal(object)  # Path — emitted when user clicks the X

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        header = QHBoxLayout()
        header.addWidget(QLabel("Queue:"))
        header.addStretch()
        self.clear_btn = QPushButton("Clear done")
        self.clear_btn.clicked.connect(self._clear_done)
        header.addWidget(self.clear_btn)
        layout.addLayout(header)

        self.list = QListWidget(self)
        # Opt into the theme-glow styling defined in themes.py — gives the queue
        # a bright tint of the current theme color instead of near-black.
        self.list.setObjectName("ThemeGlow")
        layout.addWidget(self.list, stretch=1)

        self._items: list[QueueItem] = []

    # ------------------------------------------------------- helpers

    def _row_widget_at(self, idx: int) -> QueueRow | None:
        item = self.list.item(idx)
        if item is None:
            return None
        w = self.list.itemWidget(item)
        return w if isinstance(w, QueueRow) else None

    def _index_of(self, path: Path) -> int:
        for i, it in enumerate(self._items):
            if it.path == path:
                return i
        return -1

    # ------------------------------------------------------- public API

    def add(self, path: Path) -> None:
        item = QueueItem(path=path)
        self._items.append(item)
        li = QListWidgetItem()
        li.setData(Qt.UserRole, str(path))
        row = QueueRow(path, self.list)
        row.remove_requested.connect(self._on_row_remove)
        # Size hint must match the row's preferred height so the list lays out
        # widgets without clipping the X button.
        li.setSizeHint(row.sizeHint())
        self.list.addItem(li)
        self.list.setItemWidget(li, row)

    def set_status(self, path: Path, status: str) -> None:
        idx = self._index_of(path)
        if idx < 0:
            return
        self._items[idx].status = status
        row = self._row_widget_at(idx)
        if row is not None:
            row.set_status(status)

    def pop_next_queued(self) -> Path | None:
        for it in self._items:
            if it.status == "queued":
                return it.path
        return None

    def remove(self, path: Path) -> bool:
        """Remove an item by path. Refuses to remove a `running` item — caller
        should stop the worker first. Returns True if removed."""
        idx = self._index_of(path)
        if idx < 0:
            return False
        if self._items[idx].status == "running":
            return False
        self.list.takeItem(idx)
        self._items.pop(idx)
        return True

    # ------------------------------------------------------- internals

    def _on_row_remove(self, path: Path) -> None:
        if self.remove(path):
            self.item_removed.emit(path)

    def _clear_done(self) -> None:
        i = 0
        while i < len(self._items):
            if self._items[i].status in ("done", "error"):
                self.list.takeItem(i)
                self._items.pop(i)
            else:
                i += 1
        self.cleared.emit()
