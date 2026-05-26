"""Decide which texture filenames (decals.png / decals_0.png / sponsors.png)
the user's source image should populate.

Two modes:
  - auto: FD6 picks the most appropriate slot for the source. For v0.3.5
    that's always the main `decals.png` for ACC — multi-slot auto-routing
    (splitting a complex image into base+layer pairs) is a future enhancement.
  - manual: the user has explicitly checked one or more slots in the
    settings panel; we honor exactly what they picked.

Returns a list of filenames (relative to the livery folder) the writer
should populate with the same source image. For sponsors.png we currently
treat it identically to decals.png — the user provides a single image, FD6
copies it into each requested slot. Splitting sponsor logos onto a separate
texture is a later v0.4.x feature.
"""

from __future__ import annotations


def plan_slots(
    auto: bool,
    manual_main: list[str] | None = None,
    manual_sponsors: list[str] | None = None,
) -> list[str]:
    """Resolve the user's slot configuration into a flat list of filenames."""
    if auto:
        return ["decals.png"]
    chosen: list[str] = []
    if manual_main:
        chosen.extend(manual_main)
    if manual_sponsors:
        chosen.extend(manual_sponsors)
    # Always guarantee at least the main decals.png — users who unchecked
    # everything would otherwise get an empty export.
    if not chosen:
        chosen = ["decals.png"]
    # Dedupe while preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for f in chosen:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out
