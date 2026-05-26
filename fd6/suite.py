"""Game-suite dispatch — which family of games FD6 is currently targeting.

FD6's first three game families share the UI shell but diverge on:
  - which generation pipeline to run (Forza uses geometrize, AC uses raster only)
  - which target dropdown contents to show
  - which "Inject" button label / output behavior to use

The active suite is tracked here, persisted via QSettings between sessions,
and consumed by main_window.py to swap subpanels in/out. Strict isolation
rule: fd6.suite must NEVER import from fd6.inject, fd6.ac, fd6.shapegen,
fd6.nfs, or fd6.crew — those modules import FROM here, not the other way.
"""

from __future__ import annotations

from enum import Enum

from PySide6.QtCore import QSettings


class SuiteMode(str, Enum):
    """Top-level game-family selection. String-valued so QSettings round-trips cleanly."""

    FORZA = "forza"
    AC = "ac"
    NFS = "nfs"           # placeholder, not yet implemented
    CREW = "crew"         # placeholder, not yet implemented


# Display metadata for each suite — used by the suite-picker dialog and the
# Customizations submenu. Order matters: production-ready suites first.
SUITE_DISPLAY = {
    SuiteMode.FORZA: {
        "label": "Forza Titles",
        "subtitle": "FH3 / FH4 / FH5 / FH6",
        "enabled": True,
    },
    SuiteMode.AC: {
        "label": "Assetto Corsa Titles",
        "subtitle": "ACC fully supported; ACE / AC Rally / AC coming soon",
        "enabled": True,
    },
    SuiteMode.NFS: {
        "label": "Need for Speed Titles",
        "subtitle": "Heat / Unbound — coming in a future release",
        "enabled": False,
    },
    SuiteMode.CREW: {
        "label": "The Crew: Motorfest",
        "subtitle": "Coming in a future release",
        "enabled": False,
    },
}


_SETTINGS_GROUP = "suite_mode"
_KEY_LAST_SELECTED = "last_selected"


def saved_suite_mode() -> SuiteMode | None:
    """Read the user's last-selected suite from QSettings.

    Returns None on first-ever launch (no value saved yet); callers should
    treat that as "show the suite-picker popup on first image upload".
    """
    s = QSettings("FD6", "Forza Designer 6")
    s.beginGroup(_SETTINGS_GROUP)
    try:
        raw = s.value(_KEY_LAST_SELECTED, "")
    finally:
        s.endGroup()
    if not raw:
        return None
    try:
        return SuiteMode(str(raw))
    except ValueError:
        # Stored value no longer maps to a known suite (downgrade scenario).
        # Treat as unset so the popup re-asks.
        return None


def save_suite_mode(mode: SuiteMode) -> None:
    """Persist the user's selection so the next session opens in the same mode."""
    s = QSettings("FD6", "Forza Designer 6")
    s.beginGroup(_SETTINGS_GROUP)
    try:
        s.setValue(_KEY_LAST_SELECTED, mode.value)
    finally:
        s.endGroup()
