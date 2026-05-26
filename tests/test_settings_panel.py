import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

import fd6.gui.settings_panel as settings_panel_module
from fd6.shapegen.profile import Profile
from fd6.gui.settings_panel import SettingsPanel


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_settings_panel_builds_line_guide_profile(qapp):
    panel = SettingsPanel()
    panel.line_guide_enabled.setChecked(True)
    panel.line_guide_image_path.setText("guide.png")
    panel.line_guide_model_path.setText("models/line_guide.onnx")
    panel.line_guide_strength.setValue(1.25)
    panel.line_guide_decay.setValue(0.35)
    panel.line_guide_agreement.setValue(0.8)
    panel.line_guide_candidate_ratio.setValue(0.4)
    panel.quality_batch_pixels.setValue(128000)
    panel.line_guide_max_resolution.setValue(640)

    profile = panel.build_profile()

    assert profile.line_guide_enabled is True
    assert profile.line_guide_image_path == "guide.png"
    assert profile.line_guide_model_path == "models/line_guide.onnx"
    assert profile.line_guide_strength == 1.25
    assert profile.line_guide_decay == 0.35
    assert profile.line_guide_agreement == 0.8
    assert profile.line_guide_candidate_ratio == 0.4
    assert profile.quality_batch_pixels == 128000
    assert profile.line_guide_max_resolution == 640


def test_settings_panel_restores_line_guide_profile_values(qapp, tmp_path, monkeypatch):
    path = tmp_path / "line.ini"
    path.write_text(
        Profile(
            stop_at=42,
            random_samples=11,
            mutated_samples=7,
            compute_backend="cpu",
            preview_every=3,
            shape_types=["rectangle"],
            line_guide_enabled=True,
            line_guide_image_path="guide.png",
            line_guide_model_path="models/line_guide.onnx",
            line_guide_strength=1.1,
            line_guide_decay=0.4,
            line_guide_agreement=0.85,
            line_guide_candidate_ratio=0.35,
            quality_batch_pixels=64000,
            line_guide_max_resolution=512,
        ).to_ini(),
        encoding="utf-8",
    )
    monkeypatch.setattr(settings_panel_module, "list_bundled_profiles", lambda: [path])

    panel = SettingsPanel()
    panel._on_profile_changed(0)

    assert panel.stop_at.value() == 42
    assert panel.compute_backend.currentData() == "cpu"
    assert panel.line_guide_enabled.isChecked() is True
    assert panel.line_guide_image_path.text() == "guide.png"
    assert panel.line_guide_model_path.text() == "models/line_guide.onnx"
    assert panel.line_guide_strength.value() == 1.1
    assert panel.line_guide_decay.value() == 0.4
    assert panel.line_guide_agreement.value() == 0.85
    assert panel.line_guide_candidate_ratio.value() == 0.35
    assert panel.quality_batch_pixels.value() == 64000
    assert panel.line_guide_max_resolution.value() == 512
    assert panel.build_profile().shape_types == ["rectangle"]


def test_settings_panel_updates_line_guide_status(qapp):
    panel = SettingsPanel()

    panel.set_line_guide_status("Line guide: unavailable.")

    assert panel.line_guide_status.text() == "Line guide: unavailable."
