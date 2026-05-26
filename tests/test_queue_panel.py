import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from fd6.gui.main_window import MainWindow
from fd6.gui.queue_panel import QueuePanel


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_queue_panel_deduplicates_same_file_across_path_forms(qapp, tmp_path, monkeypatch):
    path = tmp_path / "sample.png"
    path.write_bytes(b"x")
    monkeypatch.chdir(tmp_path)
    panel = QueuePanel()

    assert panel.add(Path("sample.png")) is True
    assert panel.add(path.resolve()) is False
    assert panel.list.count() == 1
    assert len(panel._items) == 1


def test_queue_panel_requeues_completed_item_without_new_row(qapp, tmp_path):
    path = tmp_path / "sample.png"
    path.write_bytes(b"x")
    panel = QueuePanel()

    assert panel.add(path) is True
    panel.set_status(path, "done")
    assert panel.add(path) is True
    assert panel.list.count() == 1
    assert len(panel._items) == 1
    assert panel.pop_next_queued() == path.resolve()


def test_main_window_ignores_duplicate_selection_when_queue_rejects_all():
    class DummyQueue:
        def __init__(self):
            self.calls: list[Path] = []

        def add(self, path: Path) -> bool:
            self.calls.append(path)
            return False

    window = MainWindow.__new__(MainWindow)
    window.queue = DummyQueue()
    window._worker = None
    started: list[bool] = []
    window._start_next = lambda: started.append(True)

    MainWindow._on_files_selected(window, [Path("sample.png"), Path("sample.png")])

    assert started == []
