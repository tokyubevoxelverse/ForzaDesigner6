"""Floating brand banner that lives in the bottom-left of the MainWindow.

Default: expanded panel showing the FD6 logo + "Forza Designer 6" title.
Click anywhere on the panel to collapse it. When collapsed, a small icon-only
button remains in the same corner; click it to re-expand.

Both states stay anchored to the bottom-left corner across window resizes.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt, QSize, QUrl
from PySide6.QtGui import QFont, QPixmap, QIcon, QMouseEvent, QDesktopServices
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget
)


TOKYUBE_URL = "https://tokyube.com"
TUTORIAL_URL = "https://youtu.be/8LGvE7O9aeg"
SUBSCRIBE_URL = "https://www.youtube.com/@DaMostPalone?sub_confirmation=1"
DISCORD_INVITE_URL = "https://discord.gg/PJFWdykGmS"
JBA_URL = "https://tokyubevoxelverse.github.io/gbajs4/"
RADIO_MAKER_URL = "https://github.com/tokyubevoxelverse/ForzaDesignerRadioMaker/releases/tag/0.0.1-Alpha"


def _bundle_root() -> Path:
    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent.parent.parent  # FD6/


def badge_path(filename: str) -> Path | None:
    """Return absolute path to a badge PNG, or None if missing. Bundle-aware."""
    root = _bundle_root()
    p = root / filename
    if p.exists():
        return p
    # Legacy fallbacks
    for cand in (root / "tools" / "fd6_128.png", root / "Logo.png"):
        if cand.exists():
            return cand
    return None


def _logo_path() -> Path | None:
    """Initial badge path — matches the currently saved theme so the brand
    banner shows the correct color immediately at startup (no flash of Pink
    before _set_theme runs)."""
    try:
        from fd6.gui.themes import badge_filename_for_theme, saved_theme_name
        p = badge_path(badge_filename_for_theme(saved_theme_name()))
        if p:
            return p
    except Exception:
        pass
    return badge_path("Pink.png") or badge_path("AppIconTransparent.png")


class BrandBanner(QWidget):
    """Brand banner that sits in the bottom-left corner. Click panel to collapse / click pill to expand."""

    MARGIN = 12
    # Height accommodates five CTA buttons stacked above the icon/title row:
    #   row 1: tokyube.com (rainbow)
    #   row 2: JBA Online GameBoy Emulator (tan -> brown)
    #   row 3: Forza Designer Radio Maker (neon purple)
    #   row 4: Tutorial / Trailer (YouTube red)
    #   row 5: Join the Imagineers (Discord orange)
    #   row 6: icon + Forza Designer 6 title
    BANNER_HEIGHT = 256
    BANNER_WIDTH = 260
    PILL_SIZE = 40
    CTA_HEIGHT = 30

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        # Make the banner widget itself transparent. Without this, the global
        # theme QSS paints a `bg` colored 40x40 square behind the round pill
        # button when collapsed, producing dark square corners around the icon.
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        # Explicit per-widget override so the global `QWidget { background: ... }`
        # in themes.py can't repaint behind us.
        self.setStyleSheet("BrandBanner { background: transparent; }")

        logo = _logo_path()
        self._pix: QPixmap | None = None
        if logo:
            pm = QPixmap(str(logo))
            if not pm.isNull():
                self._pix = pm

        # ---- expanded panel
        self.panel = QFrame(self)
        self.panel.setObjectName("brandPanel")
        self.panel.setStyleSheet(
            "#brandPanel { background: rgba(20, 20, 24, 230); border: 1px solid #333; border-radius: 8px; }"
            "#brandPanel:hover { background: rgba(30, 30, 36, 240); }"
        )
        self.panel.setCursor(Qt.PointingHandCursor)
        self.panel.setFixedSize(self.BANNER_WIDTH, self.BANNER_HEIGHT)

        # Vertical panel layout: [tokyube CTA] [discord CTA] [icon + title row]
        outer = QVBoxLayout(self.panel)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        # ── CTA 1: tokyube.com (rainbow gradient, white text) ────────────────
        self.tokyube_btn = QPushButton("tokyube.com", self.panel)
        self.tokyube_btn.setCursor(Qt.PointingHandCursor)
        self.tokyube_btn.setFixedHeight(self.CTA_HEIGHT)
        self.tokyube_btn.setStyleSheet(
            "QPushButton {"
            " background: qlineargradient(x1:0, y1:0, x2:1, y2:0,"
            "  stop:0 #ff0000, stop:0.166 #ff8800, stop:0.333 #ffff00,"
            "  stop:0.5 #00ff00, stop:0.666 #00ffff, stop:0.833 #8800ff,"
            "  stop:1 #ff00ff);"
            " color: #000000; font-weight: bold; letter-spacing: 1px;"
            " border: 1px solid #000; border-radius: 6px; padding: 0 10px; }"
            "QPushButton:hover { border-color: #fff; }"
        )
        self.tokyube_btn.setToolTip("Open tokyube.com")
        self.tokyube_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(TOKYUBE_URL)))
        outer.addWidget(self.tokyube_btn)

        # ── CTA 2: JBA Online GameBoy Emulator (tan -> brown, sand text) ─────
        self.jba_btn = QPushButton("  JBA Online GameBoy Emulator", self.panel)
        self.jba_btn.setCursor(Qt.PointingHandCursor)
        self.jba_btn.setFixedHeight(self.CTA_HEIGHT)
        jba_logo = badge_path("JavaBoyLogo.png")
        if jba_logo:
            jba_pix = QPixmap(str(jba_logo))
            if not jba_pix.isNull():
                self.jba_btn.setIcon(QIcon(jba_pix.scaled(20, 20, Qt.KeepAspectRatio, Qt.SmoothTransformation)))
                self.jba_btn.setIconSize(QSize(20, 20))
        self.jba_btn.setStyleSheet(
            "QPushButton {"
            " background: qlineargradient(x1:0, y1:0, x2:1, y2:1,"
            "  stop:0 #e6c79e, stop:0.5 #a47148, stop:1 #5b3a1e);"
            " color: #f4e4bc; font-weight: bold; letter-spacing: 0.5px;"
            " border: 1px solid #3d2412; border-radius: 6px; padding: 0 10px; text-align: left; }"
            "QPushButton:hover { border-color: #fff8e1; }"
        )
        self.jba_btn.setToolTip(f"Play GameBoy / GameBoy Advance games in your browser ({JBA_URL})")
        self.jba_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(JBA_URL)))
        outer.addWidget(self.jba_btn)

        # ── CTA 3: Forza Designer Radio Maker (neon purple gradient) ─────────
        self.radio_btn = QPushButton("🎵  Forza Designer Radio Maker", self.panel)
        self.radio_btn.setCursor(Qt.PointingHandCursor)
        self.radio_btn.setFixedHeight(self.CTA_HEIGHT)
        self.radio_btn.setStyleSheet(
            "QPushButton {"
            " background: qlineargradient(x1:0, y1:0, x2:1, y2:1,"
            "  stop:0 #c084fc, stop:0.5 #7c3aed, stop:1 #3d0fa0);"
            " color: #ffffff; font-weight: bold; letter-spacing: 0.5px;"
            " border: 1px solid #2a0a6e; border-radius: 6px; padding: 0 10px; }"
            "QPushButton:hover { border-color: #ffffff; }"
        )
        self.radio_btn.setToolTip(f"Open the Forza Designer Radio Maker release page ({RADIO_MAKER_URL})")
        self.radio_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(RADIO_MAKER_URL)))
        outer.addWidget(self.radio_btn)

        # ── CTA 4: Tutorial / Trailer (YouTube red, white play glyph) ────────
        self.tutorial_btn = QPushButton("▶  Tutorial / Trailer", self.panel)
        self.tutorial_btn.setCursor(Qt.PointingHandCursor)
        self.tutorial_btn.setFixedHeight(self.CTA_HEIGHT)
        self.tutorial_btn.setStyleSheet(
            "QPushButton {"
            " background: qlineargradient(x1:0, y1:0, x2:1, y2:1,"
            "  stop:0 #ff5252, stop:0.5 #ff0000, stop:1 #b30000);"
            " color: #ffffff; font-weight: bold; letter-spacing: 0.5px;"
            " border: 1px solid #000; border-radius: 6px; padding: 0 10px; }"
            "QPushButton:hover { border-color: #fff; }"
        )
        self.tutorial_btn.setToolTip(
            "Watch the FD6 trailer (opens the video, plus a subscribe prompt in a second tab)"
        )
        self.tutorial_btn.clicked.connect(self._on_tutorial_clicked)
        outer.addWidget(self.tutorial_btn)

        # ── CTA 3: Join the Imagineers (matches site .btn-discord orange) ────
        self.discord_btn = QPushButton("Join the Imagineers", self.panel)
        self.discord_btn.setCursor(Qt.PointingHandCursor)
        self.discord_btn.setFixedHeight(self.CTA_HEIGHT)
        self.discord_btn.setStyleSheet(
            "QPushButton {"
            " background: qlineargradient(x1:0, y1:0, x2:1, y2:1,"
            "  stop:0 #ffd9a8, stop:0.4 #ff8c1a, stop:1 #b35400);"
            " color: #000000; font-weight: bold; letter-spacing: 0.5px;"
            " border: 1px solid #6b3000; border-radius: 6px; padding: 0 10px; }"
            "QPushButton:hover { border-color: #fff; }"
        )
        self.discord_btn.setToolTip(f"Open Discord invite ({DISCORD_INVITE_URL})")
        self.discord_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(DISCORD_INVITE_URL)))
        outer.addWidget(self.discord_btn)

        # ── Row 3: icon + title (existing) ───────────────────────────────────
        bottom_row = QHBoxLayout()
        bottom_row.setContentsMargins(0, 0, 0, 0)
        bottom_row.setSpacing(10)
        self.icon_label = QLabel(self.panel)
        self.icon_label.setFixedSize(40, 40)
        self.icon_label.setAlignment(Qt.AlignCenter)
        if self._pix:
            self.icon_label.setPixmap(self._pix.scaled(40, 40, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        bottom_row.addWidget(self.icon_label)

        text_col = QVBoxLayout()
        text_col.setContentsMargins(0, 0, 0, 0)
        text_col.setSpacing(0)
        self.title_label = QLabel("Forza Designer 6+", self.panel)
        tf = QFont(); tf.setBold(True); tf.setPointSize(10)
        self.title_label.setFont(tf)
        self.title_label.setStyleSheet("color: #f0f0f0;")
        self.sub_label = QLabel("Click here to hide", self.panel)
        self.sub_label.setStyleSheet("color: #888; font-size: 10px;")
        text_col.addWidget(self.title_label)
        text_col.addWidget(self.sub_label)
        bottom_row.addLayout(text_col, stretch=1)
        outer.addLayout(bottom_row)

        # ---- collapsed pill (icon-only button)
        self.pill = QPushButton(self)
        self.pill.setFixedSize(self.PILL_SIZE, self.PILL_SIZE)
        self.pill.setCursor(Qt.PointingHandCursor)
        self.pill.setToolTip("Show FD6 banner")
        self.pill.setStyleSheet(
            "QPushButton { background: rgba(20, 20, 24, 230); border: 1px solid #333; border-radius: 20px; }"
            "QPushButton:hover { background: rgba(30, 30, 36, 240); border-color: #555; }"
        )
        if self._pix:
            self.pill.setIcon(QIcon(self._pix.scaled(28, 28, Qt.KeepAspectRatio, Qt.SmoothTransformation)))
            self.pill.setIconSize(QSize(28, 28))
        self.pill.clicked.connect(self.show_panel)

        # Make panel clickable to collapse
        self.panel.mousePressEvent = self._panel_clicked  # type: ignore

        # Start expanded
        self.pill.hide()
        self.panel.show()

        # Size of THIS widget covers the larger of the two states
        self.setFixedSize(self.BANNER_WIDTH, self.BANNER_HEIGHT)
        self.reposition()

    def set_badge(self, png_path: Path | str | None) -> None:
        """Swap the displayed badge — used when theme changes."""
        if not png_path:
            return
        pm = QPixmap(str(png_path))
        if pm.isNull():
            return
        self._pix = pm
        self.icon_label.setPixmap(pm.scaled(40, 40, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.pill.setIcon(QIcon(pm.scaled(28, 28, Qt.KeepAspectRatio, Qt.SmoothTransformation)))

    def _on_tutorial_clicked(self) -> None:
        """Open the trailer in tab 1 and the subscribe prompt in tab 2.

        Order matters: open the trailer first so it lands on the active tab —
        most browsers focus the first opened URL when receiving two openUrl
        calls back-to-back, and we want the user watching the video.
        """
        QDesktopServices.openUrl(QUrl(TUTORIAL_URL))
        QDesktopServices.openUrl(QUrl(SUBSCRIBE_URL))

    def _panel_clicked(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self.hide_panel()

    def hide_panel(self) -> None:
        self.panel.hide()
        self.setFixedSize(self.PILL_SIZE, self.PILL_SIZE)
        self.pill.show()
        self.pill.move(0, 0)
        self.reposition()

    def show_panel(self) -> None:
        self.pill.hide()
        self.setFixedSize(self.BANNER_WIDTH, self.BANNER_HEIGHT)
        self.panel.show()
        self.panel.move(0, 0)
        self.reposition()

    def reposition(self) -> None:
        """Anchor to bottom-left corner of parent widget with MARGIN."""
        parent = self.parentWidget()
        if parent is None:
            return
        x = self.MARGIN
        y = parent.height() - self.height() - self.MARGIN
        self.move(x, max(0, y))
        self.raise_()
