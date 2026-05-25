"""End-to-end test of TemplateSizePickerDialog using REAL PySide6.

The existing test_inject_template_size_constraint.py stubs PySide6 — that
caught signature/constant regressions but it missed a real bug that ate the
2026-05-25 release: `picker.Accepted` raises AttributeError on PySide6 6.x
subclasses because DialogCode enums don't propagate from QDialog to subclass
instances. Stubs that set `_QDialog.Accepted = 1` happily propagate (Python
class attributes always do), masking the bug entirely.

This test imports REAL PySide6 with a QApplication so the bug surfaces.
Skipped automatically if PySide6 isn't installed in the test env.
"""
from __future__ import annotations

import os
import sys
import pytest

PySide6 = pytest.importorskip("PySide6")

# Headless Qt — works on macOS / Linux / Windows CI without a display.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QDialog   # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv)


def test_qdialog_accepted_does_not_propagate_to_subclass_instances():
    """Pins the actual PySide6 6.x behavior that bit us: QDialog.Accepted is
    a DialogCode enum that's accessible on the CLASS but NOT on instances of
    subclasses. main_window must NEVER use `picker.Accepted` — use either
    `QDialog.Accepted` directly or `not picker.exec()` truthiness."""
    class _Sub(QDialog): pass
    d = _Sub()
    # Class access works
    assert QDialog.Accepted == 1
    # But instance access on subclass raises — this is THE bug
    with pytest.raises(AttributeError):
        _ = d.Accepted


def test_picker_accept_path_returns_selected_size():
    """User picks 1000 spheres, clicks OK → result() == 1, selected_template_size == 1000.
    Validates the entire combo → currentData → _on_accept → accept chain."""
    from forza_abyss_painter.gui.inject_template_picker import TemplateSizePickerDialog
    p = TemplateSizePickerDialog(None, json_shape_count=400)
    # Find and select "1000 spheres"
    for i in range(p.combo.count()):
        if p.combo.itemData(i) == 1000:
            p.combo.setCurrentIndex(i)
            break
    else:
        pytest.fail("1000 spheres not in combo")
    p._on_accept()
    assert p.result() == 1, "accept() should set result=1"
    assert p.selected_template_size == 1000
    # The fix in main_window: truthiness, not equality with picker.Accepted
    assert bool(p.result()) is True


def test_picker_cancel_path_returns_falsy():
    """User clicks Cancel → result() == 0 → `not picker.exec()` would be True
    → main_window returns early. selected_template_size left as None."""
    from forza_abyss_painter.gui.inject_template_picker import TemplateSizePickerDialog
    p = TemplateSizePickerDialog(None, json_shape_count=400)
    p.reject()
    assert p.result() == 0
    assert bool(p.result()) is False
    assert p.selected_template_size is None


def test_picker_auto_path_returns_none():
    """Default is Auto. User clicks OK without changing combo → returns None
    (signaling auto-search to the worker)."""
    from forza_abyss_painter.gui.inject_template_picker import TemplateSizePickerDialog
    p = TemplateSizePickerDialog(None, json_shape_count=400)
    # Index 0 is Auto
    assert p.combo.currentData() is None
    p._on_accept()
    assert p.selected_template_size is None


def test_picker_custom_value_path():
    """User picks Custom, types 2000, clicks OK → selected_template_size == 2000."""
    from forza_abyss_painter.gui.inject_template_picker import TemplateSizePickerDialog
    p = TemplateSizePickerDialog(None, json_shape_count=400)
    for i in range(p.combo.count()):
        if p.combo.itemData(i) == "custom":
            p.combo.setCurrentIndex(i)
            break
    p.custom_field.setText("2000")
    p._on_accept()
    assert p.selected_template_size == 2000


def test_picker_overflow_disables_ok_button():
    """User picks template size SMALLER than the JSON shape count → OK greyed
    out. They can't accidentally pick an undersized template."""
    from forza_abyss_painter.gui.inject_template_picker import TemplateSizePickerDialog
    from PySide6.QtWidgets import QDialogButtonBox
    p = TemplateSizePickerDialog(None, json_shape_count=1500)
    # Pick 100 — way too small for 1500 shapes
    for i in range(p.combo.count()):
        if p.combo.itemData(i) == 100:
            p.combo.setCurrentIndex(i)
            break
    ok_btn = p.buttons.button(QDialogButtonBox.StandardButton.Ok)
    assert ok_btn.isEnabled() is False, "OK must be disabled when template < JSON shapes"
    # And the warning text is set (visibility requires shown parent — verify state, not effective vis)
    assert "smaller than your JSON" in p.warn_label.text()
    assert p.warn_box.isHidden() is False, "warning panel should not be hidden"


def test_picker_overflow_recovers_when_user_picks_larger():
    """User picks 100 (too small) → OK disabled → user picks 1500 (fits) → OK enabled."""
    from forza_abyss_painter.gui.inject_template_picker import TemplateSizePickerDialog
    from PySide6.QtWidgets import QDialogButtonBox
    p = TemplateSizePickerDialog(None, json_shape_count=1500)
    ok_btn = p.buttons.button(QDialogButtonBox.StandardButton.Ok)
    # Pick 100 → disabled
    for i in range(p.combo.count()):
        if p.combo.itemData(i) == 100:
            p.combo.setCurrentIndex(i); break
    assert ok_btn.isEnabled() is False
    # Pick 1500 → recovers
    for i in range(p.combo.count()):
        if p.combo.itemData(i) == 1500:
            p.combo.setCurrentIndex(i); break
    assert ok_btn.isEnabled() is True
    assert p.warn_box.isHidden() is True


def test_picker_custom_invalid_disables_ok():
    """Custom picked but field empty / out of range → OK disabled."""
    from forza_abyss_painter.gui.inject_template_picker import TemplateSizePickerDialog
    from PySide6.QtWidgets import QDialogButtonBox
    p = TemplateSizePickerDialog(None, json_shape_count=100)
    for i in range(p.combo.count()):
        if p.combo.itemData(i) == "custom":
            p.combo.setCurrentIndex(i); break
    ok_btn = p.buttons.button(QDialogButtonBox.StandardButton.Ok)
    # Empty
    p.custom_field.setText("")
    p._update_ok_state()
    assert ok_btn.isEnabled() is False
    # Valid
    p.custom_field.setText("500")
    p._update_ok_state()
    assert ok_btn.isEnabled() is True


def test_main_window_truthiness_pattern_does_not_use_picker_Accepted():
    """Lock the fix in source: scan main_window.py and assert it doesn't
    use the `picker.Accepted` pattern that raises AttributeError on real
    PySide6. Allowed forms: `not picker.exec()`, `QDialog.Accepted`, etc.
    `picker.Accepted` (instance access on the picker variable) is banned."""
    import re
    from pathlib import Path
    # encoding="utf-8" required — main_window.py contains smart-quote
    # characters (eg in the About dialog text) that don't fit cp1252.
    # Without the explicit encoding, Windows default decoder raises
    # UnicodeDecodeError on the smart-quote bytes — CI build failure with
    # no relation to the actual test logic.
    src = (Path(__file__).resolve().parent.parent /
           "forza_abyss_painter" / "gui" / "main_window.py").read_text(encoding="utf-8")
    # Strict pattern: word-boundary `picker.Accepted` anywhere in the source.
    matches = re.findall(r"\bpicker\.Accepted\b", src)
    assert not matches, (
        "main_window.py uses `picker.Accepted` which raises AttributeError on "
        "real PySide6 6.x QDialog subclasses. Use `not picker.exec()` "
        "truthiness OR `QDialog.Accepted` (class access) instead."
    )
