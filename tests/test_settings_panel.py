import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from fd6.gui.settings_panel import SettingsPanel


def _app():
    return QApplication.instance() or QApplication([])


def test_square_box_label_uses_rotated_rectangle_profile_code():
    _app()
    panel = SettingsPanel()

    assert "rectangle" not in panel._shape_checks
    square_box = panel._shape_checks["rotated_rectangle"]
    assert square_box.text() == "Square / Box (rotatable)"
    assert square_box.isEnabled()

    panel._shape_checks["rotated_ellipse"].setChecked(False)
    square_box.setChecked(True)

    assert panel.build_profile().shape_types == ["rotated_rectangle"]
