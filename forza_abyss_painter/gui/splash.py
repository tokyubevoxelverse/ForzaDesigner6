from __future__ import annotations

import os
import sys
from pathlib import Path

from PySide6.QtCore import Qt, QUrl, Signal, QTimer
from PySide6.QtGui import QKeyEvent, QMouseEvent, QFont
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


def _splash_video_path() -> Path | None:
    """Find SplashScreen.mp4 — works both running from source and as PyInstaller-bundled EXE.

    When PyInstaller builds with --onefile, bundled resources land in sys._MEIPASS at runtime.
    Otherwise look in the FD6 project root.
    """
    candidates: list[Path] = []
    if hasattr(sys, "_MEIPASS"):
        candidates.append(Path(sys._MEIPASS) / "SplashScreen.mp4")
    # Project root: parent of fd6/ package
    here = Path(__file__).resolve().parent.parent.parent
    candidates.append(here / "SplashScreen.mp4")
    candidates.append(Path.cwd() / "SplashScreen.mp4")
    for p in candidates:
        if p.exists():
            return p
    return None


MAX_SPLASH_MS = 30_000   # absolute upper cap (used if duration never reports)
DURATION_BUFFER_MS = 250  # close this many ms after duration to ensure last frame shows


class SplashWindow(QWidget):
    """Borderless top-level window that plays the splash video, emits `finished` when done or skipped.

    End detection (most reliable first):
      A. As soon as `durationChanged` fires with a positive value, schedule a QTimer for
         `duration + 250ms`. Then we close on a clock — independent of any QMediaPlayer
         end-of-stream signal, which some builds never emit. This is the workhorse.
      B. mediaStatusChanged == EndOfMedia (fast path if it fires)
      C. positionChanged when pos >= duration - 50ms
      D. User click or Esc/Space/Enter
      E. Hard 30-second cap as final safety net
    """

    finished = Signal()

    def __init__(self, video_path: Path) -> None:
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setStyleSheet("background: #000;")
        self.resize(960, 540)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.video = QVideoWidget(self)
        layout.addWidget(self.video)

        # Blinking "skip" hint overlay (child of `self`, painted on top of the video widget)
        self.skip_hint = QLabel("Click or press Esc to skip", self)
        self.skip_hint.setStyleSheet(
            "color: #ff3030; background: rgba(0,0,0,160); padding: 6px 12px; border-radius: 6px;"
        )
        font = QFont()
        font.setBold(True)
        font.setPointSize(11)
        self.skip_hint.setFont(font)
        self.skip_hint.adjustSize()
        self.skip_hint.raise_()
        # Blink timer
        self._blink = QTimer(self)
        self._blink.setInterval(700)
        self._blink.timeout.connect(self._toggle_blink)
        self._blink.start()
        self._blink_on = True

        self.player = QMediaPlayer(self)
        self.audio = QAudioOutput(self)
        self.player.setAudioOutput(self.audio)
        self.player.setVideoOutput(self.video)
        self.player.mediaStatusChanged.connect(self._on_status)
        self.player.positionChanged.connect(self._on_position)
        self.player.durationChanged.connect(self._on_duration)
        self.player.errorOccurred.connect(self._on_error)
        self.player.setSource(QUrl.fromLocalFile(str(video_path)))

        self._done = False
        # Hard-cap fallback so splash NEVER hangs even if durationChanged never fires
        self._kill_timer = QTimer(self)
        self._kill_timer.setSingleShot(True)
        self._kill_timer.timeout.connect(self._emit_finished)
        self._kill_timer.start(MAX_SPLASH_MS)
        # Per-duration timer (set when we learn the actual video length)
        self._duration_timer = QTimer(self)
        self._duration_timer.setSingleShot(True)
        self._duration_timer.timeout.connect(self._emit_finished)
        self._duration_armed = False

    def start(self) -> None:
        self.show()
        self.raise_()
        self.activateWindow()
        self.player.play()
        self._reposition_skip_hint()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._reposition_skip_hint()

    def _reposition_skip_hint(self) -> None:
        if self.skip_hint:
            self.skip_hint.adjustSize()
            margin = 12
            x = self.width() - self.skip_hint.width() - margin
            y = self.height() - self.skip_hint.height() - margin
            self.skip_hint.move(max(0, x), max(0, y))
            self.skip_hint.raise_()

    def _toggle_blink(self) -> None:
        self._blink_on = not self._blink_on
        self.skip_hint.setVisible(self._blink_on)

    def _on_duration(self, dur_ms: int) -> None:
        # First positive duration we receive arms the deterministic close timer.
        if dur_ms > 0 and not self._duration_armed:
            self._duration_armed = True
            self._duration_timer.start(dur_ms + DURATION_BUFFER_MS)

    def _on_status(self, status) -> None:
        try:
            end_value = int(QMediaPlayer.MediaStatus.EndOfMedia)
        except AttributeError:
            end_value = int(QMediaPlayer.EndOfMedia)
        if int(status) == end_value:
            self._emit_finished()

    def _on_position(self, pos_ms: int) -> None:
        dur = self.player.duration()
        if dur > 0 and pos_ms >= dur - 50:
            self._emit_finished()

    def _on_error(self, *_args) -> None:
        self._emit_finished()

    def _emit_finished(self) -> None:
        if self._done:
            return
        self._done = True
        # Stop blink / safety timers first so they cannot re-fire mid-teardown
        for t in (self._blink, self._kill_timer, self._duration_timer):
            try:
                t.stop()
            except Exception:
                pass
        # Tear the player down deterministically: stop, drop video/audio
        # outputs, then clear the source. Without this, Qt may try to push
        # frames into a widget that's mid-destruction → crash.
        try:
            self.player.stop()
            self.player.setVideoOutput(None)
            self.player.setAudioOutput(None)
            self.player.setSource(QUrl())
        except Exception:
            pass
        # Disconnect every player signal so a late callback can't fire after
        # the SplashWindow is deleted.
        for sig in (self.player.mediaStatusChanged,
                    self.player.positionChanged,
                    self.player.durationChanged,
                    self.player.errorOccurred):
            try:
                sig.disconnect()
            except Exception:
                pass
        self.finished.emit()
        # Defer hide + delete to the next event-loop tick so the current
        # signal/event handler can fully unwind before Qt destroys this widget
        # (and the player/audio output owned by it).
        QTimer.singleShot(0, self.hide)
        QTimer.singleShot(0, self.deleteLater)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self._emit_finished()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in (Qt.Key_Escape, Qt.Key_Space, Qt.Key_Return, Qt.Key_Enter):
            self._emit_finished()
        else:
            super().keyPressEvent(event)


def maybe_show_splash(after_finished) -> SplashWindow | None:
    """Show splash if SplashScreen.mp4 is present. `after_finished` is invoked when splash ends (or immediately if no splash)."""
    if os.environ.get("FD6_NO_SPLASH"):
        after_finished()
        return None
    path = _splash_video_path()
    if path is None:
        after_finished()
        return None
    splash = SplashWindow(path)
    splash.finished.connect(after_finished)
    splash.start()
    return splash
