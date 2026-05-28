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

    `render_failed` carries a short human-readable error summary that
    MainWindow surfaces in the status bar — without this users have no
    way to see why the preview isn't updating (stderr is invisible in
    the windowed EXE build).
    """

    canvas_ready = Signal(object)
    render_failed = Signal(str)


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
        # Public attribute (kept `_emitter` alias for back-compat with
        # any existing connections). MainWindow connects render_failed
        # via `job.emitter.render_failed`.
        self.emitter = _CanvasEmitter()
        self._emitter = self.emitter
        # connect(preview.on_preview) uses AutoConnection: Direct when
        # same-thread (tests), Queued when cross-thread (QThreadPool).
        self.emitter.canvas_ready.connect(preview.on_preview)

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
        except Exception as exc:
            # Best-effort: surface the failure so the user can see it
            # in the status bar (the EXE build can't tail stderr), then
            # log to stderr for dev-machine diagnostics. The next
            # snapshot fires soon and gets its own try.
            try:
                self.emitter.render_failed.emit(
                    f"Snapshot render failed: {type(exc).__name__}: {exc}"
                )
            except Exception:
                # The emitter may have been deleted by Qt during
                # teardown; best-effort only.
                pass
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
        self.emitter.canvas_ready.emit(canvas)
