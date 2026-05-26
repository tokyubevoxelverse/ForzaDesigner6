"""upload_panel exposes two new actions for loaded JSONs (#85 #86).
Visibility is gated on BOTH the feature flag AND the loaded-JSON state."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from forza_abyss_painter.gui import feature_flags  # noqa: E402
from forza_abyss_painter.gui.upload_panel import UploadPanel  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def both_flags_on(monkeypatch):
    monkeypatch.setattr(feature_flags, "RESHAPE_GEN_AVAILABLE", True)
    monkeypatch.setattr(feature_flags, "POLISH_LOADED_AVAILABLE", True)


def test_buttons_hidden_when_flags_off(qapp):
    # Defaults: both flags False
    panel = UploadPanel()
    assert panel.reshape_btn.isHidden()
    assert panel.polish_btn.isHidden()
    panel.deleteLater()


def test_buttons_hidden_when_flags_on_but_no_json(qapp, both_flags_on):
    panel = UploadPanel()
    assert panel.reshape_btn.isHidden()
    assert panel.polish_btn.isHidden()
    panel.deleteLater()


def test_buttons_visible_when_flags_on_and_json_loaded(qapp, both_flags_on, tmp_path):
    json_path = tmp_path / "shapes.json"
    json_path.write_text("{}")
    panel = UploadPanel()
    panel.set_json_loaded(json_path)
    # show() required so parent propagates visibility to children
    panel.show()
    assert panel.reshape_btn.isVisible() is True
    assert panel.polish_btn.isVisible() is True
    panel.hide()
    panel.deleteLater()


def test_buttons_hidden_again_when_json_cleared(qapp, both_flags_on, tmp_path):
    json_path = tmp_path / "shapes.json"
    json_path.write_text("{}")
    panel = UploadPanel()
    panel.set_json_loaded(json_path)
    panel.set_json_loaded(None)
    assert panel.reshape_btn.isHidden()
    assert panel.polish_btn.isHidden()
    panel.deleteLater()


def test_reshape_button_emits_signal(qapp, both_flags_on, tmp_path):
    json_path = tmp_path / "shapes.json"
    json_path.write_text("{}")
    panel = UploadPanel()
    panel.set_json_loaded(json_path)
    received: list[Path] = []
    panel.reshape_requested.connect(lambda p: received.append(p))
    panel.reshape_btn.click()
    assert received == [json_path]
    panel.deleteLater()


def test_polish_button_emits_signal(qapp, both_flags_on, tmp_path):
    json_path = tmp_path / "shapes.json"
    json_path.write_text("{}")
    panel = UploadPanel()
    panel.set_json_loaded(json_path)
    received: list[Path] = []
    panel.polish_requested.connect(lambda p: received.append(p))
    panel.polish_btn.click()
    assert received == [json_path]
    panel.deleteLater()
