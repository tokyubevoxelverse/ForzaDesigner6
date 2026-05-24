"""ACC paint & material catalogs — colors and surface finishes the user can
pick from in the AC settings panel.

ACC stores chosen colors as integer IDs into a fixed in-game palette, not
RGB values. Same for material types (gloss / matte / metallic / etc). This
module bakes both as Python data so the GUI can render swatches and dropdowns
without re-reading anything from the ACC install at runtime.

Source of the palette: cross-referenced against an ACC-authored save and the
community-maintained color-id mapping that's been stable since ACC 1.0.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ColorEntry:
    """One ACC palette entry: integer ID + display name + RGB for the swatch."""

    color_id: int
    name: str
    rgb: tuple[int, int, int]


# ACC palette (color IDs the JSON writes into skinColor1Id / 2Id / 3Id /
# rimColor1Id / 2Id). RGB values are approximate — close enough for the
# GUI swatch to give the user the right visual cue. ACC's actual in-game
# rendering uses its own shader-stage colors, so these RGBs are picker
# affordances, not pixel-perfect previews.
ACC_PALETTE: tuple[ColorEntry, ...] = (
    ColorEntry( 0, "Pure White",          (240, 240, 240)),
    ColorEntry( 1, "Pure Black",          ( 18,  18,  18)),
    ColorEntry( 2, "Racing Red",          (180,  30,  30)),
    ColorEntry( 3, "Crimson",             (140,  20,  20)),
    ColorEntry( 4, "Burgundy",            ( 95,  20,  30)),
    ColorEntry( 5, "Orange",              (220, 100,  20)),
    ColorEntry( 6, "Tangerine",           (240, 140,  40)),
    ColorEntry( 7, "Yellow",              (240, 210,  30)),
    ColorEntry( 8, "Lime",                (160, 220,  40)),
    ColorEntry( 9, "Lawn Green",          ( 80, 180,  60)),
    ColorEntry(10, "Forest Green",        ( 30,  90,  40)),
    ColorEntry(11, "British Racing Green",( 25,  70,  45)),
    ColorEntry(12, "Teal",                ( 30, 130, 140)),
    ColorEntry(13, "Cyan",                ( 60, 180, 220)),
    ColorEntry(14, "Sky Blue",            (100, 170, 230)),
    ColorEntry(15, "Royal Blue",          ( 30,  70, 190)),
    ColorEntry(16, "Navy",                ( 20,  35,  90)),
    ColorEntry(17, "Indigo",              ( 60,  40, 140)),
    ColorEntry(18, "Violet",              (130,  60, 180)),
    ColorEntry(19, "Magenta",             (210,  50, 170)),
    ColorEntry(20, "Hot Pink",            (230, 100, 170)),
    ColorEntry(21, "Pastel Pink",         (240, 180, 200)),
    ColorEntry(22, "Salmon",              (230, 130, 110)),
    ColorEntry(23, "Bronze",              (160, 110,  60)),
    ColorEntry(24, "Brown",               (110,  70,  40)),
    ColorEntry(25, "Chocolate",           ( 70,  40,  20)),
    ColorEntry(26, "Tan",                 (200, 170, 120)),
    ColorEntry(27, "Beige",               (220, 200, 170)),
    ColorEntry(28, "Cream",               (240, 230, 200)),
    ColorEntry(29, "Silver",              (190, 190, 195)),
    ColorEntry(30, "Light Gray",          (170, 170, 175)),
    ColorEntry(31, "Medium Gray",         (130, 130, 135)),
    ColorEntry(32, "Dark Gray",           ( 80,  80,  85)),
    ColorEntry(33, "Charcoal",            ( 45,  45,  50)),
    ColorEntry(34, "Gunmetal",            ( 60,  65,  75)),
    ColorEntry(35, "Anthracite",          ( 40,  42,  48)),
    ColorEntry(36, "Carbon",              ( 25,  27,  30)),
    ColorEntry(37, "Champagne",           (220, 195, 150)),
    ColorEntry(38, "Gold",                (210, 175,  60)),
    ColorEntry(39, "Old Gold",            (170, 140,  40)),
    ColorEntry(40, "Copper",              (180, 100,  60)),
    ColorEntry(41, "Rose Gold",           (210, 160, 140)),
    ColorEntry(42, "Mint",                (160, 220, 190)),
    ColorEntry(43, "Seafoam",             (130, 200, 180)),
    ColorEntry(44, "Aqua",                ( 80, 200, 200)),
    ColorEntry(45, "Powder Blue",         (170, 200, 230)),
    ColorEntry(46, "Cobalt",              ( 30,  80, 180)),
    ColorEntry(47, "Electric Blue",       ( 50, 130, 230)),
    ColorEntry(48, "Ice Blue",            (200, 230, 240)),
    ColorEntry(49, "Lavender",            (190, 170, 220)),
    ColorEntry(50, "Plum",                (110,  60, 110)),
    ColorEntry(51, "Maroon",              (110,  30,  40)),
    ColorEntry(52, "Brick",               (160,  60,  50)),
    ColorEntry(53, "Coral",               (240, 130, 110)),
    ColorEntry(54, "Peach",               (250, 190, 150)),
    ColorEntry(55, "Apricot",             (240, 170, 110)),
    ColorEntry(56, "Mustard",             (210, 180,  50)),
    ColorEntry(57, "Olive",               (110, 110,  40)),
    ColorEntry(58, "Khaki",               (170, 160, 100)),
    ColorEntry(59, "Hunter Green",        ( 40,  80,  50)),
    ColorEntry(60, "Emerald",             ( 40, 160,  90)),
)


# ACC material-type enum (skinMaterialTypeN / rimMaterialTypeN integer field).
# Confirmed by reading the ACC-authored save and cross-checking against the
# editor's own dropdown labels.
MATERIAL_TYPES: tuple[tuple[int, str], ...] = (
    (0, "Gloss"),
    (1, "Matte"),
    (2, "Satin"),
    (3, "Metallic"),
    (4, "Chrome"),
)


def color_for_id(color_id: int) -> ColorEntry:
    """Look up the palette entry by ID. Falls back to entry 0 for unknown IDs."""
    for c in ACC_PALETTE:
        if c.color_id == color_id:
            return c
    return ACC_PALETTE[0]


def material_label(material_id: int) -> str:
    for mid, name in MATERIAL_TYPES:
        if mid == material_id:
            return name
    return "Gloss"
