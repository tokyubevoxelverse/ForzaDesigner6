"""Modal dialog for #86 Polish loaded JSON.

Two user-facing controls: polish iterations (50–500, default 150) and
lock alpha (default True). Output path defaults to
<input_json_stem>_polished.json next to the loaded JSON; user can
override via Choose output…
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QFileDialog, QFormLayout, QHBoxLayout, QLabel,
    QPushButton, QSpinBox, QVBoxLayout,
)


_DEFAULT_STEPS = 150
_STEPS_MIN = 50
_STEPS_MAX = 500
_STEPS_STEP = 10


class PolishDialog(QDialog):
    """Modal dialog that gathers polish parameters."""

    def __init__(
        self,
        parent=None,
        *,
        loaded_json_path: Path,
        source_image_path: Path,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Polish loaded JSON")
        self.setModal(True)
        self.setMinimumWidth(520)

        self._loaded_json_path = Path(loaded_json_path)
        self._source_image_path = Path(source_image_path)
        self._output_path = self._default_output_path()

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(10)

        intro = QLabel(
            "Refine the colors and opacity of the shapes already in the "
            "loaded JSON. Geometry (positions, sizes, angles) is NOT "
            "changed. Output is saved as a new file alongside the input."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #999;")
        root.addWidget(intro)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        form.addRow("Loaded JSON:", QLabel(self._loaded_json_path.name))
        form.addRow("Source image:", QLabel(self._source_image_path.name))

        self.steps_spinbox = QSpinBox(self)
        self.steps_spinbox.setRange(_STEPS_MIN, _STEPS_MAX)
        self.steps_spinbox.setSingleStep(_STEPS_STEP)
        self.steps_spinbox.setValue(_DEFAULT_STEPS)
        self.steps_spinbox.setToolTip(
            "Number of Adam optimization steps. Higher = slower but "
            "potentially better color match. Default 150 matches the "
            "engine's joint_polish budget for the Medium 1000 preset."
        )
        form.addRow("Polish iterations:", self.steps_spinbox)

        self.lock_alpha_cb = QCheckBox("Lock alpha to 255", self)
        self.lock_alpha_cb.setChecked(True)
        self.lock_alpha_cb.setToolTip(
            "Keep every shape fully opaque (required for FH6 injection). "
            "Uncheck only for diagnostic experiments."
        )
        form.addRow(self.lock_alpha_cb)

        out_row = QHBoxLayout()
        self.output_label = QLabel(self._output_path.name)
        self.output_label.setStyleSheet("color: #ccc;")
        out_row.addWidget(self.output_label, stretch=1)
        self.output_choose_btn = QPushButton("Choose output…", self)
        self.output_choose_btn.clicked.connect(self._on_choose_output)
        out_row.addWidget(self.output_choose_btn)
        form.addRow("Output:", out_row)

        root.addLayout(form)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.cancel_btn = QPushButton("Cancel", self)
        self.cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(self.cancel_btn)
        self.polish_btn = QPushButton("Polish", self)
        self.polish_btn.setDefault(True)
        self.polish_btn.clicked.connect(self.accept)
        btn_row.addWidget(self.polish_btn)
        root.addLayout(btn_row)

    def _default_output_path(self) -> Path:
        return self._loaded_json_path.parent / f"{self._loaded_json_path.stem}_polished.json"

    def set_output_path(self, path: Path) -> None:
        self._output_path = Path(path)
        self.output_label.setText(self._output_path.name)

    def _on_choose_output(self) -> None:
        path, _filter = QFileDialog.getSaveFileName(
            self, "Polish output JSON",
            str(self._output_path),
            "Forza Abyss Painter shapes (*.json);;All files (*)",
        )
        if path:
            self.set_output_path(Path(path))

    def values(self) -> dict:
        """Return the user-selected polish parameters as a plain dict.
        Caller hands these to gpu_gen_worker.build_polish_config()."""
        return {
            "steps": int(self.steps_spinbox.value()),
            "lock_alpha": bool(self.lock_alpha_cb.isChecked()),
            "output_path": self._output_path,
        }
