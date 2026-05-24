"""Font picker for FD6.

Loads every TTF/OTF in the bundled `fonts/` directory via QFontDatabase at
startup, exposes their family names, and lets the user pick one from
View → Fonts in the menu bar. The pick persists via QSettings.

Selecting "Default" reverts to the OS-preferred sans (Segoe UI Variable on
Windows 11 → Segoe UI on Windows 10 → sans fallback).
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QSettings
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication


SETTINGS_GROUP = "fonts"
DEFAULT_LABEL = "Default"
DEFAULT_POINT_SIZE = 10


def _bundle_root() -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent.parent.parent  # repo root


def _fonts_dir() -> Path:
    return _bundle_root() / "fonts"


# After load_bundled_fonts() runs:
#   _LOADED       maps real family name → loaded id (Qt internal)
#   _DISPLAY_MAP  maps cleaned display name → real family name (used by menu)
_LOADED: dict[str, int] = {}
_DISPLAY_MAP: dict[str, str] = {}


def _clean_display(family: str) -> str:
    """Cut the title at the first '-' and strip whitespace, so e.g.
    "AovelSansRounded-rdDL" → "AovelSansRounded" for display.
    """
    return family.split("-", 1)[0].strip() or family


def load_bundled_fonts() -> dict[str, int]:
    """Load every .ttf/.otf in fonts/ via QFontDatabase. Idempotent —
    re-loading the same file just returns its existing id without harm.
    Returns the {real_family_name: id} mapping.
    """
    global _LOADED, _DISPLAY_MAP
    if _LOADED:
        return _LOADED
    out: dict[str, int] = {}
    display_map: dict[str, str] = {}
    d = _fonts_dir()
    if not d.is_dir():
        _LOADED = out
        _DISPLAY_MAP = display_map
        return out
    for p in sorted(d.iterdir()):
        if p.suffix.lower() not in (".ttf", ".otf"):
            continue
        fid = QFontDatabase.addApplicationFont(str(p))
        if fid < 0:
            continue
        families = QFontDatabase.applicationFontFamilies(fid)
        for fam in families:
            out.setdefault(fam, fid)
            display = _clean_display(fam)
            # If two fonts collapse to the same display name (rare), suffix one
            n = 1
            unique = display
            while unique in display_map and display_map[unique] != fam:
                n += 1
                unique = f"{display} ({n})"
            display_map.setdefault(unique, fam)
    _LOADED = out
    _DISPLAY_MAP = display_map
    return out


def available_family_names() -> list[str]:
    """Sorted *display* names for the Fonts menu (Default first)."""
    return [DEFAULT_LABEL, *sorted(_DISPLAY_MAP.keys(), key=str.casefold)]


def _resolve_family(name: str) -> str | None:
    """Map a display name (or a real family name) back to the real family."""
    if name in _DISPLAY_MAP:
        return _DISPLAY_MAP[name]
    if name in _LOADED:
        return name
    return None


def saved_font_name() -> str:
    """The saved value is the *display* name. Migrates legacy real-family
    settings to their display equivalent on read."""
    s = QSettings("ForzaAbyssPainter", "Forza Abyss Painter")
    s.beginGroup(SETTINGS_GROUP)
    name = str(s.value("family", DEFAULT_LABEL))
    s.endGroup()
    if name == DEFAULT_LABEL:
        return name
    if name in _DISPLAY_MAP:
        return name
    if name in _LOADED:
        # Legacy: convert real family → display
        return _clean_display(name)
    return DEFAULT_LABEL


def apply_font(app: QApplication, name: str) -> None:
    """Apply the named font (display name) app-wide and persist the choice.
    `name == 'Default'` falls back to Segoe UI Variable → Segoe UI → sans.

    NOTE: `QApplication.setFont()` only changes the default for *newly created*
    widgets. To re-font an already-running UI we also walk every live widget
    and reset its font, then send a polish so styles re-cascade.
    """
    real = _resolve_family(name) if name != DEFAULT_LABEL else None
    if not real:
        f = QFont("Segoe UI Variable", DEFAULT_POINT_SIZE)
        f.setStyleStrategy(QFont.PreferAntialias)
        store_name = DEFAULT_LABEL
    else:
        f = QFont(real, DEFAULT_POINT_SIZE)
        f.setStyleStrategy(QFont.PreferAntialias)
        # Persist the display name so the menu re-check survives across launches
        store_name = next((d for d, r in _DISPLAY_MAP.items() if r == real), real)
    # 1. Set as the application default (affects future widgets)
    app.setFont(f)
    # 2. Walk every live widget and overwrite its font. This is the only way
    #    to push the change into widgets that already exist (Qt does not
    #    automatically re-broadcast app font changes).
    for w in app.allWidgets():
        try:
            w.setFont(f)
            # Polishing the widget triggers a style recompute so things like
            # menu items and tooltips pick up the metrics change immediately.
            st = w.style()
            if st is not None:
                st.unpolish(w)
                st.polish(w)
            w.update()
        except Exception:
            pass
    # 3. Repaint every top-level window so layouts recompute their geometries.
    for w in app.topLevelWidgets():
        try:
            w.updateGeometry()
            w.update()
        except Exception:
            pass
    # Persist
    s = QSettings("ForzaAbyssPainter", "Forza Abyss Painter")
    s.beginGroup(SETTINGS_GROUP)
    s.setValue("family", store_name)
    s.endGroup()
