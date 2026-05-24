"""Color themes for Forza Abyss Painter.

Each theme is a dict of named colors plus a Qt stylesheet string built from them.
`apply_theme(app, name)` installs the stylesheet onto the QApplication and persists
the choice via QSettings so the next launch remembers it.

Themes:
  - "Abyss"             — pure black + deep purple + magenta (Forza Abyss Painter signature)
  - "Default"           — dark grey + blue accent (upstream FD6 original look)
  - "Japanese Blossoms" — cherry blossom + hot pink + reddish pink
  - "Purple Passion"    — lilac + indigo + deep blue
  - "Matrix Racing"     — lime + forest green + dark green
  - "Odaiba Bay"        — azure + powder blue
  - "Hokkaido Sunset"   — electric yellow + amber
  - "Cherry Soda Pop"   — crimson + maroon
"""

from __future__ import annotations

from typing import TypedDict

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication


SETTINGS_ORG = "ForzaAbyssPainter"
SETTINGS_APP = "Forza Abyss Painter"
SETTINGS_THEME_KEY = "ui/theme"


class ThemePalette(TypedDict):
    bg: str          # main window background
    surface: str     # panel / sidebar background
    surface_alt: str # secondary surface (input fields, list backgrounds)
    border: str      # 1px borders
    text: str        # primary text
    text_muted: str  # secondary / hint text
    accent: str      # primary accent (buttons, progress chunks, selected items)
    accent_hi: str   # brighter accent (hover, focus)
    success: str
    warning: str
    error: str
    # NEW (v0.2.0): bright theme-tinted background for the queue / recent / drop-zone
    # panels so they pop instead of being near-black. Should be brighter than `surface`,
    # darker than `accent`, in the same hue family.
    panel_glow: str
    # NEW (v0.2.0): three theme-related particle colors (light, mid, deep) used by the
    # particle overlay. Should all be readable against `bg`.
    particle_1: str
    particle_2: str
    particle_3: str


# ─── Theme palettes ─────────────────────────────────────────────────────────

DEFAULT: ThemePalette = {
    "bg":          "#161616",
    "surface":     "#1f1f1f",
    "surface_alt": "#181818",
    "border":      "#2a2a2a",
    "text":        "#dddddd",
    "text_muted":  "#888888",
    "accent":      "#3a7bd5",
    "accent_hi":   "#5295e0",
    "success":     "#2ecc71",
    "warning":     "#f1c40f",
    "error":       "#ff4d4d",
    "panel_glow":  "#1f3960",   # deep but readable blue tint
    "particle_1":  "#a8d0ff",
    "particle_2":  "#5295e0",
    "particle_3":  "#1e4e90",
}

JAPANESE_BLOSSOMS: ThemePalette = {
    "bg":          "#231116",
    "surface":     "#341920",
    "surface_alt": "#2a1419",
    "border":      "#52242f",
    "text":        "#ffe1e8",
    "text_muted":  "#c89aa6",
    "accent":      "#ff4d8d",   # hot pink
    "accent_hi":   "#ff86ad",   # cherry blossom
    "success":     "#7be3a0",
    "warning":     "#ffc35a",
    "error":       "#ff5577",
    "panel_glow":  "#4a1c30",   # bright cherry panel
    "particle_1":  "#ffb8d0",
    "particle_2":  "#ff86ad",
    "particle_3":  "#c03070",
}

LILAC: ThemePalette = {
    "bg":          "#161230",
    "surface":     "#221c44",
    "surface_alt": "#1b1638",
    "border":      "#3a2f64",
    "text":        "#e8ddff",
    "text_muted":  "#9b8acb",
    "accent":      "#b78aff",   # lilac
    "accent_hi":   "#d0b0ff",
    "success":     "#7be3a0",
    "warning":     "#ffc35a",
    "error":       "#ff6688",
    "panel_glow":  "#2e2360",   # deep indigo glow
    "particle_1":  "#d0b0ff",   # lilac
    "particle_2":  "#9568e0",   # purple
    "particle_3":  "#4b3796",   # indigo
}

MINT_FOREST: ThemePalette = {
    "bg":          "#0e1d16",
    "surface":     "#16301f",
    "surface_alt": "#11251a",
    "border":      "#274a32",
    "text":        "#d8ffe6",
    "text_muted":  "#8db89c",
    "accent":      "#3fe07a",   # lime green
    "accent_hi":   "#7af0a8",
    "success":     "#7be3a0",
    "warning":     "#ffc35a",
    "error":       "#ff6677",
    "panel_glow":  "#194a2d",   # bright forest green panel
    "particle_1":  "#a8f0c8",
    "particle_2":  "#3fe07a",
    "particle_3":  "#188a40",
}

SKY_AZURE: ThemePalette = {
    "bg":          "#0d1d2e",   # deep blue
    "surface":     "#15324d",
    "surface_alt": "#0f253b",
    "border":      "#284967",
    "text":        "#e6f4ff",   # snow blue
    "text_muted":  "#8fb6d8",
    "accent":      "#00b4ff",   # azure
    "accent_hi":   "#b8e0ff",   # powder blue
    "success":     "#7bdcb8",
    "warning":     "#ffd45a",
    "error":       "#ff6688",
    "panel_glow":  "#1a4a78",   # bay-blue panel
    "particle_1":  "#b8e0ff",   # powder blue
    "particle_2":  "#00b4ff",   # azure
    "particle_3":  "#1a608c",   # deep marine
}

PASTEL_YELLOW: ThemePalette = {
    "bg":          "#2a230d",   # dark goldenrod backdrop
    "surface":     "#3d3414",
    "surface_alt": "#322a10",
    "border":      "#5a4a1a",
    "text":        "#fff7c8",   # lemon-white
    "text_muted":  "#c8b870",
    "accent":      "#ffe600",   # electric yellow
    "accent_hi":   "#fff96b",   # lemon yellow
    "success":     "#8fe0a0",
    "warning":     "#daa520",   # goldenrod
    "error":       "#ff7766",
    "panel_glow":  "#5a4a1f",   # warm bright amber panel
    "particle_1":  "#fff080",   # lemon
    "particle_2":  "#ffd84a",   # sunset yellow
    "particle_3":  "#b07a00",   # deep amber
}

PASTEL_RED: ThemePalette = {
    "bg":          "#1f0a0c",   # dark red backdrop
    "surface":     "#3a1216",
    "surface_alt": "#2a0d10",
    "border":      "#5a1e24",
    "text":        "#ffd9dc",   # pastel red text
    "text_muted":  "#c08088",
    "accent":      "#dc143c",   # crimson
    "accent_hi":   "#ff5a6e",
    "success":     "#7be3a0",
    "warning":     "#ffc35a",
    "error":       "#800000",   # maroon
    "panel_glow":  "#5a1c24",   # bright cherry-red panel
    "particle_1":  "#ffaab0",   # pastel red
    "particle_2":  "#ff5a6e",   # bright red
    "particle_3":  "#a50d2a",   # crimson deep
}


# Abyss — Forza Abyss Painter signature theme.
# Palette: pure black base + deep purple panels + magenta/pink accent — the Forza Abyss Painter signature look.
# (pure black + deep purple panels + magenta/pink accent).
ABYSS: ThemePalette = {
    "bg":          "#0a0a0a",   # window background (pure black)
    "surface":     "#0f0f1a",   # panel background (deep purple-black)
    "surface_alt": "#13132a",   # secondary surface — slightly lifted
    "border":      "#3a2555",   # purple borders
    "text":        "#f5f1f8",   # primary text (off-white)
    "text_muted":  "#b8afc8",   # secondary / hint text
    "accent":      "#d94f90",   # primary pink/magenta accent
    "accent_hi":   "#e86ca8",   # hover state (accent-light)
    "success":     "#7be3a0",
    "warning":     "#ffc35a",
    "error":       "#ff5577",
    "panel_glow":  "#2a1844",   # bright purple glow panel
    "particle_1":  "#e86ca8",   # accent-light
    "particle_2":  "#8b5cf6",   # gradient-purple start
    "particle_3":  "#b61b70",   # accent-dark
}


THEMES: dict[str, ThemePalette] = {
    "Abyss":             ABYSS,
    "Default":           DEFAULT,
    "Japanese Blossoms": JAPANESE_BLOSSOMS,
    "Purple Passion":    LILAC,
    "Matrix Racing":     MINT_FOREST,
    "Odaiba Bay":        SKY_AZURE,
    "Hokkaido Sunset":   PASTEL_YELLOW,
    "Cherry Soda Pop":   PASTEL_RED,
}

DEFAULT_THEME_NAME = "Abyss"


# Per-theme app icon / badge filenames. Resolved by music/brand_banner logic
# relative to the bundle root (or repo root when running from source).
THEME_BADGES: dict[str, str] = {
    "Abyss":             "forza_abyss_painter_logo.png",
    "Default":           "Pink.png",
    "Japanese Blossoms": "Pink.png",          # cherry blossoms = pink, not yellow
    "Purple Passion":    "Purple.png",
    "Matrix Racing":     "Green.png",
    "Odaiba Bay":        "Blue.png",
    "Hokkaido Sunset":   "Orange.png",
    "Cherry Soda Pop":   "AppIconTransparent.png",  # the original red badge
}


def badge_filename_for_theme(name: str) -> str:
    return THEME_BADGES.get(name, "forza_abyss_painter_logo.png")


def _build_qss(p: ThemePalette) -> str:
    """Compose the QApplication-wide stylesheet from a theme palette."""
    return f"""
    /* === Global === */
    QMainWindow, QWidget, QDialog {{
        background: {p["bg"]};
        color: {p["text"]};
    }}
    QFrame {{ background: transparent; color: {p["text"]}; }}
    QLabel {{ background: transparent; color: {p["text"]}; }}
    QToolTip {{
        background: {p["surface"]}; color: {p["text"]};
        border: 1px solid {p["border"]}; padding: 4px;
    }}

    /* === Buttons === */
    QPushButton {{
        background: {p["surface"]};
        border: 1px solid {p["border"]};
        color: {p["text"]};
        padding: 6px 12px; border-radius: 4px;
    }}
    QPushButton:hover {{
        background: {p["surface_alt"]};
        border-color: {p["accent"]};
    }}
    QPushButton:pressed {{
        background: {p["accent"]};
        color: {p["bg"]};
    }}
    QPushButton:disabled {{
        color: {p["text_muted"]};
        border-color: {p["border"]};
    }}
    QPushButton:focus {{ outline: none; border-color: {p["accent_hi"]}; }}

    /* === Inputs === */
    QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit, QPlainTextEdit, QTextEdit {{
        background: {p["surface_alt"]};
        border: 1px solid {p["border"]};
        color: {p["text"]};
        padding: 4px 6px;
        border-radius: 3px;
        selection-background-color: {p["accent"]};
        selection-color: {p["bg"]};
    }}
    QComboBox::drop-down {{ border-left: 1px solid {p["border"]}; width: 18px; }}
    QComboBox QAbstractItemView {{
        background: {p["surface"]}; color: {p["text"]};
        border: 1px solid {p["border"]};
        selection-background-color: {p["accent"]};
        selection-color: {p["bg"]};
    }}
    QSpinBox::up-button, QSpinBox::down-button,
    QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
        background: {p["surface"]};
        border: none; width: 16px;
    }}

    /* === Lists === */
    QListWidget {{
        background: {p["surface_alt"]};
        border: 1px solid {p["border"]};
        color: {p["text"]};
    }}
    QListWidget::item:selected {{
        background: {p["accent"]}; color: {p["bg"]};
    }}
    QListWidget::item:hover {{ background: {p["surface"]}; }}

    /* === Theme-glow panels (queue, recent, drop zone) — bright theme tint ===
       Widgets opt in by setting objectName == "ThemeGlow". */
    QListWidget#ThemeGlow {{
        background: {p["panel_glow"]};
        border: 1px solid {p["accent"]};
        color: {p["text"]};
    }}
    QListWidget#ThemeGlow::item:selected {{
        background: {p["accent_hi"]}; color: {p["bg"]};
    }}
    QListWidget#ThemeGlow::item:hover {{ background: {p["accent"]}; color: {p["bg"]}; }}
    QFrame#ThemeGlow, QWidget#ThemeGlow {{
        background: {p["panel_glow"]};
        border: 1px dashed {p["accent"]};
        border-radius: 6px;
        color: {p["text"]};
    }}

    /* === Group boxes === */
    QGroupBox {{
        border: 1px solid {p["border"]};
        border-radius: 4px;
        margin-top: 12px;
        padding-top: 10px;
        color: {p["text"]};
    }}
    QGroupBox::title {{
        subcontrol-origin: margin; left: 8px; padding: 0 6px;
        color: {p["text_muted"]};
    }}

    /* === Checkbox === */
    QCheckBox {{ color: {p["text"]}; spacing: 6px; }}
    QCheckBox::indicator {{
        width: 14px; height: 14px;
        background: {p["surface_alt"]};
        border: 1px solid {p["border"]};
        border-radius: 2px;
    }}
    QCheckBox::indicator:checked {{
        background: {p["accent"]}; border-color: {p["accent_hi"]};
    }}

    /* === Progress bar === */
    QProgressBar {{
        background: {p["surface_alt"]};
        border: 1px solid {p["border"]};
        text-align: center;
        color: {p["text"]};
    }}
    QProgressBar::chunk {{ background: {p["accent"]}; }}

    /* === Menus === */
    QMenuBar {{ background: {p["bg"]}; color: {p["text"]}; }}
    QMenuBar::item:selected {{ background: {p["surface"]}; }}
    QMenu {{
        background: {p["surface"]}; color: {p["text"]};
        border: 1px solid {p["border"]};
    }}
    QMenu::item:selected {{ background: {p["accent"]}; color: {p["bg"]}; }}

    /* === Status bar (overridden in code for severity colors) === */
    QStatusBar {{ background: {p["surface"]}; color: {p["text"]}; }}

    /* === Splitter handle === */
    QSplitter::handle {{ background: {p["border"]}; }}
    QSplitter::handle:hover {{ background: {p["accent"]}; }}

    /* === Scrollbars === */
    QScrollBar:vertical, QScrollBar:horizontal {{
        background: {p["surface_alt"]}; border: none;
    }}
    QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
        background: {p["border"]}; border-radius: 3px; min-height: 20px;
    }}
    QScrollBar::handle:hover {{ background: {p["accent"]}; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ background: transparent; border: none; }}
    """


def apply_theme(app: QApplication, name: str) -> ThemePalette:
    """Install the named theme onto `app`. Falls back to Abyss if name unknown.
    Returns the palette that was applied so callers can read colors for in-code styling.
    """
    pal = THEMES.get(name, ABYSS)
    qss = _build_qss(pal)
    app.setStyleSheet(qss)
    # Persist for next launch
    s = QSettings(SETTINGS_ORG, SETTINGS_APP)
    s.setValue(SETTINGS_THEME_KEY, name if name in THEMES else DEFAULT_THEME_NAME)
    return pal


def saved_theme_name() -> str:
    """Return the persisted theme name, or 'Abyss' if none saved."""
    s = QSettings(SETTINGS_ORG, SETTINGS_APP)
    return s.value(SETTINGS_THEME_KEY, DEFAULT_THEME_NAME) or DEFAULT_THEME_NAME


def current_palette() -> ThemePalette:
    """Get the currently-saved theme's palette (used by widgets that need raw colors)."""
    return THEMES.get(saved_theme_name(), ABYSS)
