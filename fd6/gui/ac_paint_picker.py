"""Color + material pickers for ACC livery export.

One ColorSwatchPicker = a button that opens a small popup with all palette
colors laid out as a swatch grid. Clicking a swatch sets the picker's
current color and closes the popup. The button face shows the chosen
color as a flat fill so the user sees what they've picked.

MaterialDropdown = simple QComboBox over MATERIAL_TYPES.

The composite AC paint panel groups 3 skin color/material pairs plus 2 rim
color/material pairs into one collapsible group box for the AC settings
panel to embed.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QColor, QPainter, QPixmap, QIcon
from PySide6.QtWidgets import (
    QComboBox, QDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
    QPushButton, QVBoxLayout, QWidget,
)

from fd6.ac.paint_catalog import ACC_PALETTE, MATERIAL_TYPES, ColorEntry, color_for_id


_SWATCH_SIZE = 28
_SWATCHES_PER_ROW = 8


def _color_pixmap(rgb: tuple[int, int, int], size: int = _SWATCH_SIZE) -> QPixmap:
    pm = QPixmap(size, size)
    pm.fill(QColor(rgb[0], rgb[1], rgb[2]))
    # 1px dark border so very-light swatches stay visible against a light bg.
    p = QPainter(pm)
    p.setPen(QColor(40, 40, 40))
    p.drawRect(0, 0, size - 1, size - 1)
    p.end()
    return pm


class _SwatchGrid(QDialog):
    """Modal popup with a grid of color swatches. Emits picked color_id."""

    picked = Signal(int)  # color_id

    def __init__(self, parent: QWidget | None = None, current_id: int = 0) -> None:
        super().__init__(parent)
        self.setWindowTitle("Pick a color")
        self.setWindowModality(Qt.ApplicationModal)
        layout = QGridLayout(self)
        layout.setSpacing(2)
        layout.setContentsMargins(8, 8, 8, 8)
        for i, entry in enumerate(ACC_PALETTE):
            row, col = divmod(i, _SWATCHES_PER_ROW)
            btn = QPushButton(self)
            btn.setFixedSize(_SWATCH_SIZE + 6, _SWATCH_SIZE + 6)
            btn.setIcon(QIcon(_color_pixmap(entry.rgb)))
            btn.setIconSize(QSize(_SWATCH_SIZE, _SWATCH_SIZE))
            btn.setToolTip(f"{entry.name} (id {entry.color_id})")
            if entry.color_id == current_id:
                btn.setStyleSheet("border: 2px solid #ffcc00;")
            btn.clicked.connect(lambda _=False, cid=entry.color_id: self._on_pick(cid))
            layout.addWidget(btn, row, col)

    def _on_pick(self, color_id: int) -> None:
        self.picked.emit(color_id)
        self.accept()


class ColorSwatchPicker(QWidget):
    """A single color picker: button shows current swatch + name; clicking
    opens the grid popup."""

    color_changed = Signal(int)  # color_id

    def __init__(self, label: str, initial_id: int = 0, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._current_id = initial_id
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel(label, self)
        lbl.setMinimumWidth(110)
        row.addWidget(lbl)
        self.button = QPushButton(self)
        self.button.setMinimumWidth(140)
        self.button.setIconSize(QSize(20, 20))
        self.button.clicked.connect(self._open_picker)
        row.addWidget(self.button, stretch=1)
        self._refresh_button()

    def current_color_id(self) -> int:
        return self._current_id

    def set_color_id(self, color_id: int) -> None:
        if color_id == self._current_id:
            return
        self._current_id = int(color_id)
        self._refresh_button()
        self.color_changed.emit(self._current_id)

    def _refresh_button(self) -> None:
        entry = color_for_id(self._current_id)
        self.button.setIcon(QIcon(_color_pixmap(entry.rgb, 20)))
        self.button.setText(f"  {entry.name}")

    def _open_picker(self) -> None:
        dlg = _SwatchGrid(self, current_id=self._current_id)
        dlg.picked.connect(self.set_color_id)
        dlg.exec()


class MaterialDropdown(QComboBox):
    """ComboBox preloaded with ACC's material-type enum."""

    def __init__(self, initial_id: int = 0, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        for mid, name in MATERIAL_TYPES:
            self.addItem(name, mid)
        self.set_material_id(initial_id)

    def current_material_id(self) -> int:
        data = self.currentData()
        return int(data) if data is not None else 0

    def set_material_id(self, material_id: int) -> None:
        for i in range(self.count()):
            if int(self.itemData(i)) == material_id:
                self.setCurrentIndex(i)
                return


class ACPaintPanel(QGroupBox):
    """Composite panel: 3 skin color+material pairs + 2 rim color+material pairs.

    Embed this inside the AC settings panel. Call gather() to read the
    user's picks as a dict that the livery_writer accepts.
    """

    paint_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Paint & Materials", parent)
        self.setToolTip(
            "Pick the car's body and rim colors and finishes. These are "
            "written into the ACC team JSON and applied when the livery loads "
            "in-game. Color swatches are approximate previews of ACC's actual "
            "palette."
        )
        outer = QVBoxLayout(self)

        # Skin (body) — 3 paint slots
        skin_group = QGroupBox("Body", self)
        skin_layout = QGridLayout(skin_group)
        self.skin_colors: list[ColorSwatchPicker] = []
        self.skin_materials: list[MaterialDropdown] = []
        for i in range(3):
            cp = ColorSwatchPicker(f"Color {i + 1}", initial_id=0, parent=skin_group)
            md = MaterialDropdown(initial_id=0, parent=skin_group)
            cp.color_changed.connect(self._on_change)
            md.currentIndexChanged.connect(lambda _=0: self._on_change())
            skin_layout.addWidget(cp, i, 0)
            skin_layout.addWidget(QLabel("Finish:", skin_group), i, 1)
            skin_layout.addWidget(md, i, 2)
            self.skin_colors.append(cp)
            self.skin_materials.append(md)
        outer.addWidget(skin_group)

        # Rims — 2 paint slots
        rim_group = QGroupBox("Rims", self)
        rim_layout = QGridLayout(rim_group)
        self.rim_colors: list[ColorSwatchPicker] = []
        self.rim_materials: list[MaterialDropdown] = []
        for i in range(2):
            cp = ColorSwatchPicker(f"Color {i + 1}", initial_id=29, parent=rim_group)  # default silver
            md = MaterialDropdown(initial_id=3, parent=rim_group)  # default metallic
            cp.color_changed.connect(self._on_change)
            md.currentIndexChanged.connect(lambda _=0: self._on_change())
            rim_layout.addWidget(cp, i, 0)
            rim_layout.addWidget(QLabel("Finish:", rim_group), i, 1)
            rim_layout.addWidget(md, i, 2)
            self.rim_colors.append(cp)
            self.rim_materials.append(md)
        outer.addWidget(rim_group)

    def _on_change(self) -> None:
        self.paint_changed.emit()

    def gather(self) -> dict:
        """Return the user's picks as a flat dict matching ACC's team-JSON field names."""
        return {
            "skinColor1Id": self.skin_colors[0].current_color_id(),
            "skinColor2Id": self.skin_colors[1].current_color_id(),
            "skinColor3Id": self.skin_colors[2].current_color_id(),
            "skinMaterialType1": self.skin_materials[0].current_material_id(),
            "skinMaterialType2": self.skin_materials[1].current_material_id(),
            "skinMaterialType3": self.skin_materials[2].current_material_id(),
            "rimColor1Id": self.rim_colors[0].current_color_id(),
            "rimColor2Id": self.rim_colors[1].current_color_id(),
            "rimMaterialType1": self.rim_materials[0].current_material_id(),
            "rimMaterialType2": self.rim_materials[1].current_material_id(),
        }
