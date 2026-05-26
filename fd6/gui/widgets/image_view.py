from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy


def _pick_transform_mode(
    pix_width: int,
    pix_height: int,
    box_width: int,
    box_height: int,
    prefer_crisp_upscale: bool,
):
    if not prefer_crisp_upscale or pix_width <= 0 or pix_height <= 0 or box_width <= 0 or box_height <= 0:
        return Qt.SmoothTransformation
    scale = min(box_width / pix_width, box_height / pix_height)
    if scale > 1.0:
        return Qt.FastTransformation
    return Qt.SmoothTransformation


class ImageView(QLabel):
    """QLabel that scales its pixmap on resize while preserving aspect ratio."""

    def __init__(self, placeholder: str = "-", parent=None, prefer_crisp_upscale: bool = False) -> None:
        super().__init__(placeholder, parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(160, 120)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("QLabel { background: #181818; color: #555; border: 1px solid #2a2a2a; }")
        self._pix: QPixmap | None = None
        self._prefer_crisp_upscale = prefer_crisp_upscale

    def set_numpy(self, arr: np.ndarray) -> None:
        if arr.ndim != 3 or arr.shape[2] not in (3, 4):
            return
        h, w, c = arr.shape
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
        self.setText("-")

    def _rescale(self) -> None:
        if self._pix is None:
            return
        transform_mode = _pick_transform_mode(
            self._pix.width(),
            self._pix.height(),
            self.width(),
            self.height(),
            self._prefer_crisp_upscale,
        )
        scaled = self._pix.scaled(self.size(), Qt.KeepAspectRatio, transform_mode)
        self.setPixmap(scaled)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._rescale()
