from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal, Qt, QTimer
from PySide6.QtWidgets import (
    QFileDialog, QHBoxLayout, QListWidget, QPushButton, QStackedWidget,
    QVBoxLayout, QWidget, QLabel,
)

from forza_abyss_painter.gui.widgets import DropZone
from forza_abyss_painter.gui.widgets.drop_zone import SUPPORTED_EXTS
# Note: forza_abyss_painter.gui.image_search is NOT imported at module load — that would force
# QWebEngine (Chromium, ~150MB renderer process) to spin up at startup even
# for users who never toggle the image searcher on. We lazy-import it inside
# `set_use_image_searcher()` the first time the toggle flips on.


class UploadPanel(QWidget):
    files_selected = Signal(list)        # list[Path] — image files chosen for generation
    json_loaded = Signal(Path)           # User uploaded a JSON: load + show preview (do NOT inject)
    download_json_requested = Signal()   # User wants to save the most-recent generated JSON

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._recent: list[Path] = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        self.upload_btn = QPushButton("Upload Image…")
        self.upload_btn.setMinimumHeight(40)
        self.upload_btn.clicked.connect(self._on_upload_clicked)
        layout.addWidget(self.upload_btn)

        self.drop = DropZone(self)
        self.drop.files_dropped.connect(self._on_files_dropped)
        layout.addWidget(self.drop)

        layout.addSpacing(6)
        # JSON Upload (for re-injecting) + Download (export generated)
        json_row = QHBoxLayout()
        self.upload_json_btn = QPushButton("Upload JSON…")
        self.upload_json_btn.setToolTip(
            "Load a previously-generated shapes JSON and preview it in the canvas. "
            "Click 'Inject into FH6' afterwards when you're ready to push it into the game."
        )
        self.upload_json_btn.clicked.connect(self._on_upload_json_clicked)
        self.download_json_btn = QPushButton("Download JSON")
        self.download_json_btn.setEnabled(False)
        self.download_json_btn.setToolTip("No generated JSON yet — finish generating an image first")
        self.download_json_btn.clicked.connect(self.download_json_requested.emit)
        self._download_default_style = self.download_json_btn.styleSheet()
        self._pulse_timer = QTimer(self)
        self._pulse_timer.setSingleShot(True)
        self._pulse_timer.timeout.connect(self._end_download_pulse)
        json_row.addWidget(self.upload_json_btn)
        json_row.addWidget(self.download_json_btn)
        layout.addLayout(json_row)

        layout.addSpacing(4)
        # Label tracks which panel is showing — flips when Customizations
        # toggle is flipped.
        self.section_label = QLabel("Recent:", self)
        layout.addWidget(self.section_label)
        # Stack the Recents list and Image Searcher; we swap by changing the
        # current index from the View → Customizations menu toggle.
        self.stack = QStackedWidget(self)
        self.recent_list = QListWidget(self.stack)
        self.recent_list.setObjectName("ThemeGlow")
        self.recent_list.itemDoubleClicked.connect(self._on_recent_dbl)
        self.stack.addWidget(self.recent_list)   # index 0 — recents
        # image_search panel is created lazily on first toggle (see below)
        self.image_search = None
        layout.addWidget(self.stack, stretch=1)

    def _ensure_image_search(self) -> None:
        """Construct ImageSearchPanel + Chromium renderer on first use."""
        if self.image_search is not None:
            return
        from forza_abyss_painter.gui.image_search import ImageSearchPanel
        self.image_search = ImageSearchPanel(self.stack)
        self.image_search.image_downloaded.connect(self._on_downloaded_image)
        self.stack.addWidget(self.image_search)   # index 1 — searcher

    def set_use_image_searcher(self, enabled: bool) -> None:
        """Wired to the View → Customizations toggle."""
        if enabled:
            self._ensure_image_search()
            self.stack.setCurrentIndex(1)
            self.section_label.setText("Image search:")
        else:
            self.stack.setCurrentIndex(0)
            self.section_label.setText("Recent:")

    def _on_downloaded_image(self, path: Path) -> None:
        # Feed the downloaded image through the same path as an Upload click
        self._emit([Path(path)])

    def _on_upload_clicked(self) -> None:
        exts = " ".join(f"*{e}" for e in sorted(SUPPORTED_EXTS))
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Pick image(s)", "", f"Images ({exts});;All files (*)"
        )
        if paths:
            self._emit([Path(p) for p in paths])

    def mark_json_ready(self, json_path: Path | None = None) -> None:
        """Called by MainWindow when a generation finishes. Enables the Download JSON button
        and briefly pulses it green so the user notices it's now actionable.
        """
        self.download_json_btn.setEnabled(True)
        tip = "Save the most-recent generated shapes JSON to a location of your choice"
        if json_path:
            tip = f"Save '{json_path.name}' to a location of your choice"
        self.download_json_btn.setToolTip(tip)
        # Pulse to draw attention
        self.download_json_btn.setStyleSheet(
            "QPushButton { background: #1f6f3a; color: white; font-weight: bold; "
            "border: 2px solid #2ecc71; border-radius: 4px; padding: 6px 10px; }"
            "QPushButton:hover { background: #258245; }"
        )
        self._pulse_timer.start(3000)  # revert styling after 3 sec

    def _end_download_pulse(self) -> None:
        self.download_json_btn.setStyleSheet(self._download_default_style)

    def _on_upload_json_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Pick shapes JSON to load for preview", "", "Forza Abyss Painter shapes (*.json);;All files (*)"
        )
        if path:
            self.json_loaded.emit(Path(path))

    def _on_files_dropped(self, paths: list[Path]) -> None:
        self._emit(paths)

    def _on_recent_dbl(self, item) -> None:
        p = Path(item.data(Qt.UserRole))
        if p.exists():
            self._emit([p])

    def _emit(self, paths: list[Path]) -> None:
        for p in paths:
            if p not in self._recent:
                self._recent.insert(0, p)
                self.recent_list.insertItem(0, p.name)
                self.recent_list.item(0).setData(Qt.UserRole, str(p))
                while self.recent_list.count() > 12:
                    self.recent_list.takeItem(12)
                self._recent = self._recent[:12]
        self.files_selected.emit(paths)
