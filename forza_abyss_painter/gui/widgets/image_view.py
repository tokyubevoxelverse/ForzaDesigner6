from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy


class ImageView(QLabel):
    """QLabel that scales its pixmap on resize while preserving aspect ratio."""

    def __init__(self, placeholder: str = "—", parent=None) -> None:
        super().__init__(placeholder, parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(160, 120)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("QLabel { background: #181818; color: #555; border: 1px solid #2a2a2a; }")
        self._pix: QPixmap | None = None

    def set_numpy(self, arr: np.ndarray) -> None:
        """Display an HxWx3 (RGB) or HxWx4 (RGBA) numpy array.

        Sticker-mode preview emits RGBA so the canvas's transparent regions
        render as the pane background instead of a solid grey rectangle —
        matching how the Source view displays the same PNG.
        """
        if arr.ndim != 3 or arr.shape[2] not in (3, 4):
            return
        h, w, c = arr.shape
        # QImage expects bytes contiguous; force a copy when not already.
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        if c == 4:
            img = QImage(arr.data, w, h, w * 4, QImage.Format_RGBA8888).copy()
        else:
            img = QImage(arr.data, w, h, w * 3, QImage.Format_RGB888).copy()
        self._pix = QPixmap.fromImage(img)
        self._rescale()

    def set_path(self, path: str) -> None:
        pm = QPixmap(path)
        if not pm.isNull():
            self._pix = pm
            self._rescale()

    def clear_image(self) -> None:
        self._pix = None
        self.setText("—")

    def _rescale(self) -> None:
        if self._pix is None:
            return
        scaled = self._pix.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._rescale()
