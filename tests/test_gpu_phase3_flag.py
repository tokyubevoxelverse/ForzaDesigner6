"""Pin the GPU_PHASE_3_AVAILABLE flag's current value + verify the Tools
menu rendering follows it.

Why a dedicated test: the flag is a SHIPPING-contract signal. Flipping
it should be a deliberate, reviewable change that happens in the SAME
commit as the underlying plumbing landing (real downloader + subprocess
runner). If someone toggles the flag without wiring the dialogs, this
test fails loud — and the test_main_window_tools_menu test catches the
inverse case (flag flipped without UI renderingit).
"""
from __future__ import annotations

import os
import sys

import pytest

PySide6 = pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QMessageBox   # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv)


@pytest.fixture(autouse=True)
def _no_block_messageboxes(monkeypatch):
    """MainWindow's constructor can fire info modals (missing music
    file, font fallback). Without this fixture an unrelated env issue
    would hang the test on a modal."""
    monkeypatch.setattr(QMessageBox, "critical",
                        staticmethod(lambda *a, **kw: QMessageBox.Ok))
    monkeypatch.setattr(QMessageBox, "warning",
                        staticmethod(lambda *a, **kw: QMessageBox.Ok))
    monkeypatch.setattr(QMessageBox, "information",
                        staticmethod(lambda *a, **kw: QMessageBox.Ok))
    yield


def test_gpu_phase3_flag_is_currently_true():
    """Phase 3 plumbing landed (tasks #93-#96 + #102-#104 all green
    locally + smoke-tested), so the flag is True and the Windows
    tester can REACH the install + generate UI surface. Real Windows
    validation produces diagnostic-bundle logs via Tools → Save
    diagnostics zip; we post-mortem from macOS using those.

    If a regression takes this back to False, ensure the Tools menu
    test (test_tools_menu_includes_generate_when_flag_enabled) is
    updated to match — the two tests are paired so a drift surfaces
    loudly.
    """
    from forza_abyss_painter.gui.feature_flags import GPU_PHASE_3_AVAILABLE
    assert GPU_PHASE_3_AVAILABLE is True, (
        "GPU_PHASE_3_AVAILABLE reverted to False — the Windows tester "
        "loses the GPU UI surface entirely. If this is intentional "
        "(known-broken state, rolling back to stub), also revert the "
        "Tools menu visibility test below."
    )


def test_tools_menu_includes_generate_when_flag_enabled():
    """With the flag True, the Tools menu MUST include the GPU Generate
    item — that's the user's entry point into the install + generate
    flow. Without it the EXE plumbing is unreachable from the UI."""
    from forza_abyss_painter.gui import feature_flags as ff
    assert ff.GPU_PHASE_3_AVAILABLE is True, (
        "fixture sanity — this test only meaningful when flag is True"
    )
    from forza_abyss_painter.gui.main_window import MainWindow
    window = MainWindow()
    try:
        mbar = window.menuBar()
        tools_menu = None
        for action in mbar.actions():
            menu = action.menu()
            if menu and action.text().replace("&", "") == "Tools":
                tools_menu = menu
                break
        assert tools_menu is not None, "Tools menu missing entirely"
        item_texts = [a.text().replace("&", "")
                      for a in tools_menu.actions() if not a.isSeparator()]
        # Both Generate and Clean items must be present in the rendered menu.
        assert any("Generate shapes locally" in t for t in item_texts), (
            f"GPU Generate item missing despite GPU_PHASE_3_AVAILABLE=True — "
            f"Tools menu items: {item_texts}. The Windows tester can't "
            f"reach the GPU UI surface from the rendered EXE."
        )
        assert "Clean current JSON…" in item_texts, (
            f"Clean menu item missing — Tools menu items: {item_texts}"
        )
        # Install runtime item should also be present so users can
        # re-install or pre-install before clicking Generate.
        assert any("Install GPU runtime" in t for t in item_texts), (
            f"Install GPU runtime item missing — without it, users can "
            f"only trigger install indirectly via Generate. Tools menu "
            f"items: {item_texts}"
        )
    finally:
        window.close()
        window.deleteLater()

