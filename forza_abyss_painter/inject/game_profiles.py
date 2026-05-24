"""Per-game profiles for FD6 injection.

Each profile bundles everything that differs between Forza titles:
  - process name(s) to attach to
  - struct offsets within the LiveryGroup and Layer types
  - the MSVC RTTI class-name string used by the optional vtable locator
  - the scale divisors used when packing JSON scale fields into FH world units

Offsets and scale divisors for FH5 and FH6 are confirmed identical via the
public bvzrays/forza-painter-fh6 source (MIT). FH4 is provided as a beta
profile using the same layout under the working assumption that the Forge
engine carries the same CLiveryGroup struct across the FH4/FH5/FH6 lineage —
the user MUST verify before relying on it.

We do NOT load community-distributed "update codes". The single baseline
RTTI string `.?AVCLiveryGroup@@` is hardcoded; if a future game patch renames
the class we add the new string here in source rather than pulling external
pattern files at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# Baseline MSVC-mangled RTTI class name for the vinyl group object.
# This name has been stable across FH4/FH5/FH6 builds to date.
RTTI_CLIVERY_GROUP = b".?AVCLiveryGroup@@"


@dataclass(frozen=True)
class GameProfile:
    """Everything that varies between Forza titles for injection purposes."""

    key: str                              # "fh6" / "fh5" / "fh4"
    label: str                            # GUI display label
    process_names: tuple[str, ...]        # candidates passed to find_process_id
    rtti_class_name: bytes = RTTI_CLIVERY_GROUP

    # LiveryGroup struct offsets
    livery_count_offset: int = 0x5A       # u16 layer count
    layer_table_offset: int = 0x78        # u64 ptr to layer-pointer array

    # Layer struct offsets (within each Layer instance the table points to)
    layer_position_offset: int = 0x18     # 2 x f32: x, y
    layer_scale_offset: int = 0x28        # 2 x f32: scale_x, scale_y
    layer_rotation_offset: int = 0x50     # f32: rotation degrees
    layer_color_offset: int = 0x74        # 4 bytes: R, G, B, alpha
    layer_mask_offset: int = 0x78         # u8: mask flag (0 or 1)
    layer_shape_id_offset: int = 0x7A     # u8: shape type id (102 ellipse / 101 other)

    # JSON-space → game-space scale divisors
    scale_divisor_ellipse: float = 63.0
    scale_divisor_other: float = 127.0

    # Game's shape-id byte values
    shape_id_ellipse: int = 102
    shape_id_other: int = 101

    # Heuristic hint: is this profile validated against a known build?
    # Untrusted (beta) profiles should surface a warning in the GUI.
    beta: bool = False
    beta_note: str = ""


# Forza Horizon 6 — primary, fully validated against build 354.221.
FH6 = GameProfile(
    key="fh6",
    label="Forza Horizon 6",
    process_names=("forzahorizon6.exe", "ForzaHorizon6-Win64-Shipping.exe"),
)

# Forza Horizon 5 — same Forge-derived struct as FH6 per bvzrays. Marked beta
# until we verify a successful injection against a live FH5 install.
FH5 = GameProfile(
    key="fh5",
    label="Forza Horizon 5 (BETA)",
    process_names=("ForzaHorizon5.exe", "forzahorizon5.exe"),
    beta=True,
    beta_note=(
        "FH5 uses the same struct layout as FH6 according to publicly available "
        "research (bvzrays/forza-painter-fh6). Not independently validated by FD6 "
        "against a live FH5 install yet — first-time users should test with a "
        "throwaway vinyl group before injecting into anything they care about."
    ),
)

# Forza Horizon 4 — validated with a successful live injection. Same
# CLiveryGroup struct layout as FH5/FH6 (the Forge engine carries the
# livery system unchanged across the FH4 -> FH5 -> FH6 lineage).
FH4 = GameProfile(
    key="fh4",
    label="Forza Horizon 4",
    process_names=("ForzaHorizon4.exe", "forzahorizon4.exe"),
)

# Forza Horizon 3 — earliest title in the FH lineage covered by FD6. Same
# educated guess on struct layout as FH4; even less validated. Beta-flagged
# with the loudest warning of the bunch.
FH3 = GameProfile(
    key="fh3",
    label="Forza Horizon 3 (BETA)",
    process_names=("ForzaHorizon3.exe", "forzahorizon3.exe"),
    beta=True,
    beta_note=(
        "FH3 support is highly experimental. FH3 is the earliest title FD6 "
        "attempts to inject into; the CLiveryGroup struct layout is ASSUMED to "
        "match FH4/FH5/FH6 since they share the Forge engine lineage, but FD6 "
        "has not confirmed this against a live FH3 install and the gap from "
        "FH3 (2016) to FH6 (2026) is the widest of any supported title. Test "
        "on a throwaway vinyl group only. If injection produces garbage or "
        "crashes the game, the offsets and scale divisors likely need "
        "re-derivation — report findings to the FD6 maintainers before "
        "further use."
    ),
)


PROFILES: dict[str, GameProfile] = {
    FH6.key: FH6,
    FH5.key: FH5,
    FH4.key: FH4,
    FH3.key: FH3,
}


def get_profile(key: str) -> GameProfile:
    """Look up a profile by key. Raises ValueError on unknown keys."""
    normalized = (key or "").lower().strip()
    if normalized not in PROFILES:
        supported = ", ".join(PROFILES)
        raise ValueError(f"Unsupported game profile '{key}'. Known: {supported}")
    return PROFILES[normalized]


def default_profile() -> GameProfile:
    """The safe default for users who haven't picked a target."""
    return FH6


def list_profiles() -> list[GameProfile]:
    """All profiles in display order (production first, beta last)."""
    return [FH6, FH5, FH4, FH3]
