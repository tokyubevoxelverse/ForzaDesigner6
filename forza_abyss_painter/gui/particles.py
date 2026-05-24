"""Decorative particle overlay for FD6.

A transparent, click-through QWidget that floats above the MainWindow and
paints animated blossom-petal sprites tinted in the active theme's three
particle colors (`particle_1/2/3`).

Density and on/off state persist via QSettings (group: "particles"). Controlled
from View → Particles in the menu bar.

Visual rules (per user spec, v0.2.0):
- Sprite: bundled BlossomParticle.png, tinted with the per-theme particle color
  using the PNG's own alpha as a mask.
- Tiny random size, on the order of the previous flat-dot radii (a few pixels).
- Particles must NEVER be drawn over the preview area — we accept an
  "exclude rect" from MainWindow (mapped to overlay coordinates) and clip.
- The overlay sets `Qt.WA_TransparentForMouseEvents` so it never steals input.
"""

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QPointF, QRect, QSettings, Qt, QTimer
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap, QRegion
from PySide6.QtWidgets import QWidget


SETTINGS_GROUP = "particles"
DEFAULT_ENABLED = True
DEFAULT_COUNT = 120  # 2x the previous default per user request
COUNT_OPTIONS = (0, 60, 120, 240, 400, 800)

# Sprite size range (in pixels of the rendered blossom). Stays small — same
# order as the previous flat-dot radii so the field reads as "particles", not
# "icons floating around."
SPRITE_SIZE_MIN = 4
SPRITE_SIZE_MAX = 14


def _bundle_root() -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent.parent.parent  # repo root


def _sprite_path() -> Path | None:
    p = _bundle_root() / "BlossomParticle.png"
    return p if p.exists() else None


@dataclass
class _Particle:
    x: float
    y: float
    vx: float
    vy: float
    size: float          # rendered sprite edge length (px)
    color_idx: int
    phase: float         # alpha shimmer
    spin: float          # current rotation (rad)
    spin_v: float        # rotation velocity (rad/frame)


class ParticleOverlay(QWidget):
    """Theme-tinted blossom-sprite field. Constructed with MainWindow as parent."""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        # Pass mouse events through to widgets underneath
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setFocusPolicy(Qt.NoFocus)
        # Read persisted state.
        # One-time migration: in v0.1.x the default count was 60. Users who
        # never touched the menu have count=60 in QSettings. Bump them to the
        # new 120 default the first time v0.2 runs (gated by `schema` key).
        s = QSettings("ForzaAbyssPainter", "Forza Abyss Painter")
        s.beginGroup(SETTINGS_GROUP)
        schema = int(s.value("schema", 1))
        self._enabled: bool = s.value("enabled", DEFAULT_ENABLED, type=bool)
        self._count: int = int(s.value("count", DEFAULT_COUNT))
        if schema < 2 and self._count == 60:
            self._count = DEFAULT_COUNT  # bump from v0.1 default to v0.2 default
            s.setValue("count", self._count)
        if schema < 2:
            s.setValue("schema", 2)
        s.endGroup()
        # Theme colors — overridden by set_theme_colors
        self._colors: list[QColor] = [QColor("#a8d0ff"), QColor("#5295e0"), QColor("#1e4e90")]
        self._tinted: list[QPixmap | None] = [None, None, None]
        # Load sprite as mask source
        sp = _sprite_path()
        self._sprite_mask: QImage | None = None
        if sp:
            img = QImage(str(sp))
            if not img.isNull():
                # Force ARGB so the alpha channel survives later tint composites
                self._sprite_mask = img.convertToFormat(QImage.Format_ARGB32)
        # Exclusion rect (in overlay-local coords). Updated by MainWindow OR
        # computed live by `_exclude_provider` every paintEvent — providers
        # are more reliable than push-based caching because they react to
        # splitter drags too (which don't trigger MainWindow.resizeEvent).
        self._exclude_rect: QRect | None = None
        self._exclude_provider = None  # callable() -> QRect | None
        # Particle state
        self._particles: list[_Particle] = []
        self._rng = random.Random()
        # Animation timer
        self._timer = QTimer(self)
        self._timer.setInterval(33)  # ~30 fps
        self._timer.timeout.connect(self._tick)
        # Initial tint + particles
        self._rebuild_tinted()
        self._init_particles()
        if self._enabled and self._count > 0:
            self._timer.start()
            self.show()
        else:
            self.hide()

    # ------------------------------------------------------- public API

    def set_theme_colors(self, hex_1: str, hex_2: str, hex_3: str) -> None:
        self._colors = [QColor(hex_1), QColor(hex_2), QColor(hex_3)]
        self._rebuild_tinted()
        self.update()

    def set_enabled(self, on: bool) -> None:
        self._enabled = bool(on)
        self._persist("enabled", self._enabled)
        if self._enabled and self._count > 0:
            self._timer.start()
            self.raise_()
            self.show()
        else:
            self._timer.stop()
            self.hide()

    def set_count(self, n: int) -> None:
        self._count = max(0, int(n))
        self._persist("count", self._count)
        self._init_particles()
        if self._enabled and self._count > 0:
            self._timer.start()
            self.show()
        else:
            self._timer.stop()
            self.hide()
        self.update()

    def enabled(self) -> bool:
        return self._enabled

    def count(self) -> int:
        return self._count

    def reposition(self) -> None:
        """Match parent window's client size so particles cover the whole window."""
        p = self.parentWidget()
        if p is None:
            return
        self.setGeometry(0, 0, p.width(), p.height())
        self.raise_()

    def set_exclude_rect(self, rect: QRect | None) -> None:
        """Tell the overlay a fixed rectangle (in overlay-local coordinates)
        where particles must NOT be painted. Prefer `set_exclude_provider`
        for live tracking (splitter drags, panel resizes, etc.).
        """
        self._exclude_rect = rect
        self.update()

    def set_exclude_provider(self, provider) -> None:
        """Register a `callable() -> QRect | None` that returns the current
        exclude rect in overlay-local coords. Called every paintEvent so the
        exclude region stays correct without push notifications."""
        self._exclude_provider = provider
        self.update()

    # ------------------------------------------------------- internals

    def _rebuild_tinted(self) -> None:
        """For each theme color, build a pre-tinted QPixmap version of the
        sprite mask. We use the PNG's alpha channel as the shape, painted with
        the tint color via DestinationIn composition.
        """
        self._tinted = [None, None, None]
        if self._sprite_mask is None or self._sprite_mask.isNull():
            return
        w, h = self._sprite_mask.width(), self._sprite_mask.height()
        for i, color in enumerate(self._colors):
            tinted = QImage(w, h, QImage.Format_ARGB32_Premultiplied)
            tinted.fill(QColor(0, 0, 0, 0))
            qp = QPainter(tinted)
            try:
                qp.setRenderHint(QPainter.SmoothPixmapTransform, True)
                qp.fillRect(0, 0, w, h, color)
                qp.setCompositionMode(QPainter.CompositionMode_DestinationIn)
                qp.drawImage(0, 0, self._sprite_mask)
            finally:
                qp.end()
            self._tinted[i] = QPixmap.fromImage(tinted)

    def _init_particles(self) -> None:
        p = self.parentWidget()
        w = max(800, p.width() if p else 1280)
        h = max(600, p.height() if p else 760)
        self._particles = []
        for _ in range(self._count):
            self._particles.append(self._make_particle(w, h))

    def _make_particle(self, w: int, h: int) -> _Particle:
        return _Particle(
            x=self._rng.uniform(0, w),
            y=self._rng.uniform(0, h),
            vx=self._rng.uniform(-0.35, 0.35),
            vy=self._rng.uniform(-0.45, -0.05),  # gentle upward bias for petals
            size=self._rng.uniform(SPRITE_SIZE_MIN, SPRITE_SIZE_MAX),
            color_idx=self._rng.randint(0, 2),
            phase=self._rng.uniform(0, math.tau),
            spin=self._rng.uniform(0, math.tau),
            spin_v=self._rng.uniform(-0.04, 0.04),
        )

    def _tick(self) -> None:
        w, h = self.width(), self.height()
        if w <= 0 or h <= 0:
            return
        for pt in self._particles:
            pt.x += pt.vx
            pt.y += pt.vy
            pt.phase += 0.05
            pt.spin += pt.spin_v
            # Wrap edges
            if pt.x < -20:
                pt.x = w + 20
            elif pt.x > w + 20:
                pt.x = -20
            if pt.y < -20:
                pt.y = h + 20
            elif pt.y > h + 20:
                pt.y = -20
        self.update()

    def paintEvent(self, _event) -> None:
        if not self._enabled or not self._particles:
            return
        qp = QPainter(self)
        qp.setRenderHint(QPainter.Antialiasing, True)
        qp.setRenderHint(QPainter.SmoothPixmapTransform, True)
        # Don't draw into the exclude rect (the image preview area). We ask
        # the provider live so splitter drags and runtime layout changes are
        # always reflected — push-based caching missed those.
        excl = None
        if self._exclude_provider is not None:
            try:
                excl = self._exclude_provider()
            except Exception:
                excl = None
        if excl is None:
            excl = self._exclude_rect
        if excl is not None and not excl.isNull():
            full_region = QRegion(self.rect())
            keep = full_region.subtracted(QRegion(excl))
            qp.setClipRegion(keep)
        for pt in self._particles:
            pix = self._tinted[pt.color_idx % len(self._tinted)]
            if pix is None or pix.isNull():
                # Fallback: flat-dot when sprite is missing
                base = self._colors[pt.color_idx % len(self._colors)]
                alpha = int(118 + 62 * math.sin(pt.phase))
                qp.setBrush(QColor(base.red(), base.green(), base.blue(),
                                   max(40, min(220, alpha))))
                qp.setPen(Qt.NoPen)
                qp.drawEllipse(QPointF(pt.x, pt.y), pt.size / 2, pt.size / 2)
                continue
            # Render the tinted sprite, rotated + alpha-shimmered
            alpha = 0.55 + 0.30 * math.sin(pt.phase)  # 0.25 - 0.85
            qp.save()
            qp.translate(pt.x, pt.y)
            qp.rotate(math.degrees(pt.spin))
            qp.setOpacity(max(0.15, min(0.95, alpha)))
            half = pt.size / 2
            qp.drawPixmap(
                QPointF(-half, -half),
                pix.scaled(int(pt.size), int(pt.size),
                           Qt.KeepAspectRatio, Qt.SmoothTransformation),
            )
            qp.restore()
        qp.end()

    def _persist(self, key: str, value) -> None:
        s = QSettings("ForzaAbyssPainter", "Forza Abyss Painter")
        s.beginGroup(SETTINGS_GROUP)
        s.setValue(key, value)
        s.endGroup()
