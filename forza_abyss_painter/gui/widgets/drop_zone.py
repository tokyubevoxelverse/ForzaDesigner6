from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout


SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff", ".gif"}


class DropZone(QFrame):
    """Visible drop target. Emits files_dropped with a list of valid image paths."""

    files_dropped = Signal(list)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        # Use the shared "ThemeGlow" styling defined in themes.py — gives the
        # drop zone a theme-tinted background and dashed accent border.
        self.setObjectName("ThemeGlow")
        self.setFrameShape(QFrame.StyledPanel)
        self.setMinimumHeight(110)
        layout = QVBoxLayout(self)
        self._label = QLabel("Drop images here\n(or use the Upload button)", self)
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setStyleSheet("font-size: 12px;")
        layout.addWidget(self._label)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setProperty("dragOver", True)
            self.style().unpolish(self)
            self.style().polish(self)

    def dragLeaveEvent(self, event) -> None:
        self.setProperty("dragOver", False)
        self.style().unpolish(self)
        self.style().polish(self)

    def dropEvent(self, event: QDropEvent) -> None:
        self.setProperty("dragOver", False)
        self.style().unpolish(self)
        self.style().polish(self)
        paths: list[Path] = []
        for url in event.mimeData().urls():
            p = Path(url.toLocalFile())
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
                paths.append(p)
        if paths:
            self.files_dropped.emit(paths)
            event.acceptProposedAction()
