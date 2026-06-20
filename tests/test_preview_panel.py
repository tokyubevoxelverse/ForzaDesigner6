import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from fd6.gui.preview_panel import PreviewPanel


def _app():
    return QApplication.instance() or QApplication([])


def test_preview_panel_reports_search_and_committed_shape_details():
    _app()
    panel = PreviewPanel()

    panel.on_backend("CPU")
    panel.on_search_progress(
        4,
        3000,
        123.45,
        "Searching shape 5/3000 on CPU with 8 CPU worker(s).",
    )

    assert panel.search_status_label.text() == (
        "Searching shape 5/3000 on CPU with 8 CPU worker(s)."
    )
    assert panel.status_label.text() == "Backend: CPU"

    panel.on_progress_details(
        5,
        3000,
        95.14,
        4.4,
        680,
    )

    text = panel.status_label.text()
    assert "Shape 5/3000" in text
    assert "RMS=95.14" in text
    assert "4.4 shapes/s" in text
    assert "ETA 11:20" in text
    assert "CPU" in text
    assert panel.search_status_label.text().startswith("Searching shape 5/3000")
