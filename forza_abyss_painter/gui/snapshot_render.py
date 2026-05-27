"""Off-thread snapshot → canvas → PreviewPanel rendering.

Used by MainWindow when the GpuGenWorker emits a `snapshot` Signal
during a run. The QRunnable reads the snapshot JSON, renders via
`render_shapes` (pure CPU, no torch), and marshals the resulting
numpy canvas back to the GUI thread via a QObject Signal(object)
connected to `PreviewPanel.on_preview`.

Signal(object) handles the cross-thread Python-object marshal that
QMetaObject.invokeMethod cannot — PySide6 signals natively carry
Python objects across thread boundaries.

Throttling (single-slot queue) lives in MainWindow — this module
just renders one snapshot.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, QRunnable, Signal, Slot

if TYPE_CHECKING:
    from forza_abyss_painter.gui.preview_panel import PreviewPanel


class _CanvasEmitter(QObject):
    """Thin QObject whose only job is to carry the numpy canvas Signal.

    Lives on the same thread as the QRunnable body; the Signal's
    queued-connection semantics deliver the canvas to the GUI thread
    automatically when the connected slot belongs to a different thread.
    """

    canvas_ready = Signal(object)


class _RenderSnapshotJob(QRunnable):
    """Background render: snapshot JSON → numpy canvas → preview slot.

    Errors are swallowed silently:
      - Snapshot may be mid-write (next snapshot fires within ~1s on GPU).
      - Snapshot may have been deleted between event-fire and read.
      - render_shapes may raise on malformed shapes (unlikely; the
        runner-side validator catches most issues).

    All cases: log to stderr + return. The next snapshot event triggers
    another render.
    """

    def __init__(self, snapshot_path: "str | Path",
                 preview: "PreviewPanel") -> None:
        super().__init__()
        self._path = Path(snapshot_path)
        self._emitter = _CanvasEmitter()
        # connect(preview.on_preview) uses AutoConnection: Direct when
        # same-thread (tests), Queued when cross-thread (QThreadPool).
        self._emitter.canvas_ready.connect(preview.on_preview)

    def run(self) -> None:   # noqa: D401 — QRunnable contract
        try:
            from forza_abyss_painter.io.exporter import load_json
            from forza_abyss_painter.shapegen.render import render_shapes
            doc = load_json(str(self._path))
            shapes = doc.materialize_shapes()
            w, h = doc.image_size if doc.image_size else (1, 1)
            if w < 1 or h < 1:
                return
            transparent_bg = bool(getattr(doc, "sticker_mode", False))
            canvas = render_shapes(
                shapes, int(w), int(h),
                background=(255, 255, 255),
                transparent_bg=transparent_bg,
            )
        except Exception:
            # Best-effort: silently skip this render; the next snapshot
            # fires soon. Log to stderr for diagnostics; not via logger
            # to avoid pulling Qt-thread loggers into the worker.
            import sys
            import traceback
            print(
                f"snapshot_render: skipping {self._path.name}: "
                f"{traceback.format_exc(limit=2)}",
                file=sys.stderr,
            )
            return
        # Emit the canvas. AutoConnection routes: Direct (same thread in
        # tests), Queued (cross-thread from QThreadPool in production).
        self._emitter.canvas_ready.emit(canvas)
