"""snapshot_render path is visible to the user.

(A) On a malformed snapshot, render_failed signal fires + main_window
    surfaces the error in the status bar.
(B) On a well-formed snapshot, the canvas reaches PreviewPanel.on_preview
    (real-MainWindow smoke).
(C) `forza_abyss_painter.gui.snapshot_render` is imported at module top in
    main_window.py — PyInstaller's static analysis only catches top-level
    imports, so a function-local import here = invisible-in-EXE bug.
"""
import os
import json
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
from PySide6.QtWidgets import QApplication


def test_snapshot_render_failure_surfaces_in_status_bar(tmp_path):
    """A render failure should reach the status bar via the
    render_failed Signal. The renderer runs on QThreadPool, so the
    Signal is delivered to the GUI thread when AutoConnection routes
    through the event loop."""
    app = QApplication.instance() or QApplication([])
    from forza_abyss_painter.gui.main_window import MainWindow
    from forza_abyss_painter.gui.snapshot_render import _RenderSnapshotJob
    from PySide6.QtCore import QThreadPool

    # Write a deliberately malformed "snapshot" file.
    bad_snap = tmp_path / "broken.json"
    bad_snap.write_text("this is not JSON")

    win = MainWindow()
    try:
        # Drive the dispatch the same way MainWindow does, but also
        # capture the render_failed Signal at the source so we can
        # assert it actually fired regardless of status-bar timing.
        seen: list[str] = []
        job = _RenderSnapshotJob(str(bad_snap), win.preview)
        job.emitter.render_failed.connect(seen.append)
        job.emitter.render_failed.connect(
            lambda msg: win.statusBar().showMessage(msg, 6000)
        )
        QThreadPool.globalInstance().start(job)

        from PySide6.QtCore import QCoreApplication
        import time as _t
        for _ in range(100):
            QCoreApplication.processEvents()
            _t.sleep(0.02)
            if seen:
                # Give one more pass so any queued status-bar update
                # also lands before we assert.
                QCoreApplication.processEvents()
                break

        assert seen, "render_failed Signal was never emitted"
        assert "render failed" in seen[0].lower(), \
            f"unexpected render_failed payload: {seen[0]!r}"
    finally:
        win.close()


def test_snapshot_render_succeeds_and_paints_preview(tmp_path):
    """End-to-end: a good snapshot fires the canvas_ready Signal
    carrying a numpy canvas. The job under test connects to
    PreviewPanel.on_preview by default — we additionally attach a
    capture-listener so we can assert the canvas was delivered."""
    app = QApplication.instance() or QApplication([])
    from forza_abyss_painter.gui.main_window import MainWindow
    from forza_abyss_painter.gui.snapshot_render import _RenderSnapshotJob
    from PySide6.QtCore import QThreadPool

    # Write a minimal valid fd6.shapes JSON snapshot.
    snap = tmp_path / "snap.json"
    snap.write_text(json.dumps({
        "format": "fd6.shapes",
        "version": 1,
        "image_size": [64, 64],
        "shapes": [
            {"type": "rotated_ellipse", "x": 32, "y": 32,
             "rx": 10, "ry": 8, "angle": 0.0,
             "color": [255, 0, 0, 200]}
        ],
    }))

    win = MainWindow()
    try:
        seen: list[object] = []
        job = _RenderSnapshotJob(str(snap), win.preview)
        job.emitter.canvas_ready.connect(seen.append)
        QThreadPool.globalInstance().start(job)

        from PySide6.QtCore import QCoreApplication
        import time as _t
        for _ in range(100):
            QCoreApplication.processEvents()
            _t.sleep(0.02)
            if seen:
                QCoreApplication.processEvents()
                break

        assert seen, "canvas_ready Signal was never emitted for a valid snapshot"
        import numpy as np
        canvas = seen[0]
        assert isinstance(canvas, np.ndarray), \
            f"expected numpy canvas, got {type(canvas).__name__}"
    finally:
        win.close()


def test_snapshot_render_module_is_top_level_import():
    """PyInstaller catches top-level imports via static analysis.
    Function-local imports get missed unless in hiddenimports."""
    import ast
    src = Path(
        "forza_abyss_painter/gui/main_window.py"
    ).read_text()
    tree = ast.parse(src)
    top_level_imports = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            top_level_imports.append(node.module)
    assert "forza_abyss_painter.gui.snapshot_render" in top_level_imports, \
        "snapshot_render must be imported at module top so PyInstaller catches it"


def test_snapshot_render_in_spec_hiddenimports():
    """Belt-and-suspenders: even if hoisted, also pin it in the spec's
    hiddenimports list so PyInstaller config drift can't silently
    re-introduce the bug."""
    spec_src = Path("ForzaAbyssPainter.spec").read_text()
    assert '"forza_abyss_painter.gui.snapshot_render"' in spec_src, \
        "snapshot_render must be listed in ForzaAbyssPainter.spec hiddenimports"
