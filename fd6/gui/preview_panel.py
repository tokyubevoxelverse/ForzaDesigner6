from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QProgressBar, QVBoxLayout, QWidget, QSplitter, QSizePolicy
)

from fd6.gui.widgets import ImageView


class _ElidedLabel(QLabel):
    """Single-line label that keeps the full text available in its tooltip."""

    def __init__(self, text: str = "", parent=None) -> None:
        super().__init__("", parent)
        self._full_text = ""
        self.setWordWrap(False)
        self.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.setText(text)

    def setText(self, text: str) -> None:  # noqa: N802 - Qt API name
        self._full_text = str(text)
        self.setToolTip(self._full_text)
        self._refresh_elided_text()

    def text(self) -> str:  # noqa: N802 - Qt API name
        return self._full_text

    def display_text(self) -> str:
        return super().text()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._refresh_elided_text()

    def _refresh_elided_text(self) -> None:
        width = max(0, self.width() - 4)
        if width <= 0:
            display = self._full_text
        else:
            display = self.fontMetrics().elidedText(
                self._full_text,
                Qt.TextElideMode.ElideRight,
                width,
            )
        super().setText(display)


class PreviewPanel(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._backend_label = ""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        splitter = QSplitter(Qt.Horizontal, self)
        self.source_view = ImageView("Source", self)
        self.preview_view = ImageView("Preview", self)
        splitter.addWidget(self.source_view)
        splitter.addWidget(self.preview_view)
        splitter.setSizes([500, 500])
        layout.addWidget(splitter, stretch=1)

        self.search_status_label = _ElidedLabel("Idle.", self)
        self.search_status_label.setStyleSheet("color: #c8c8c8;")
        self.search_status_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        layout.addWidget(self.search_status_label)

        info_row = QHBoxLayout()
        self.status_label = _ElidedLabel("Idle.", self)
        self.status_label.setStyleSheet("color: #aaa;")
        # The status text is set to long sentences on every generation start. A
        # plain QLabel reports that full text width/height as its sizeHint, which
        # pushed the whole window bigger on each new preview (the bottom dipped
        # off-screen; toggling fullscreen forced a relayout that "fixed" it).
        # Keep it a SINGLE line (constant height) and Ignore its width so it can
        # never dictate the window size — it just fills the row it's given and
        # elides if the text is too long.
        self.status_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.progress = QProgressBar(self)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setMinimumWidth(180)
        self.progress.setMaximumWidth(520)
        self.progress.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        info_row.addWidget(self.status_label, stretch=3)
        info_row.addWidget(self.progress, stretch=1)
        layout.addLayout(info_row)

    def set_source(self, path: str | Path) -> None:
        self.source_view.set_path(str(path))
        self.preview_view.clear_image()
        self.progress.setValue(0)
        self._backend_label = ""
        self.search_status_label.setText(f"Preparing generation for '{Path(path).name}'.")
        self.status_label.setText(
            "Idle — give the FD6 engine a moment to start. "
            "First-shape startup can take anywhere from a few seconds to several "
            "minutes depending on profile (random/mutated samples) and image size."
        )

    def on_backend(self, label: str) -> None:
        self._backend_label = label
        self.status_label.setText(f"Backend: {label}")
        if not self.search_status_label.text().startswith("Searching shape"):
            self.search_status_label.setText(f"Backend ready: {label}")

    def on_progress(self, count: int, total: int, rms: float) -> None:
        pct = int(round(100 * count / max(1, total)))
        self.progress.setValue(min(100, pct))
        self.status_label.setText(f"Shape {count}/{total}   RMS={rms:.2f}")

    def on_progress_details(
        self,
        count: int,
        total: int,
        rms: float,
        rate: float,
        eta_seconds: float,
    ) -> None:
        pct = int(round(100 * count / max(1, total)))
        self.progress.setValue(min(100, pct))
        eta = _format_eta(eta_seconds)
        backend = f"   {self._backend_label}" if self._backend_label else ""
        self.status_label.setText(
            f"Shape {count}/{total}   RMS={rms:.2f}   "
            f"{rate:.1f} shapes/s   ETA {eta}{backend}"
        )

    def on_search_progress(self, count: int, total: int, rms: float, message: str) -> None:
        pct = int(round(100 * count / max(1, total)))
        self.progress.setValue(min(100, pct))
        detail = message or f"Searching shape {count + 1}/{total}."
        self.search_status_label.setText(detail)

    def on_preview(self, arr) -> None:
        self.preview_view.set_numpy(arr)

    def reset(self) -> None:
        self.progress.setValue(0)
        self.search_status_label.setText("Idle.")
        self.status_label.setText("Idle.")
        self._backend_label = ""
        self.source_view.clear_image()
        self.preview_view.clear_image()


def _format_eta(seconds: float) -> str:
    if seconds <= 0:
        return "00:00"
    total = int(round(seconds))
    minutes, sec = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"
