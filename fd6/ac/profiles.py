"""Per-title profiles for the Assetto Corsa family.

Mirrors the structure of fd6.inject.game_profiles.GameProfile but for the
file-export pipeline AC uses (no memory injection, no offsets). Each profile
knows where its user-folder livery directory lives, which texture filenames
the title expects, and what resolutions / aspect ratios it accepts.

ACC is the only title with full v0.3.5 support — the other three are
declared as profiles so the GUI can list them as 'Coming Soon' without
needing dead branches scattered through the code.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ACTitleProfile:
    """Everything that differs between Assetto Corsa titles for FD6 export."""

    key: str                                # "acc" / "ace" / "ac_rally" / "ac"
    label: str                              # GUI display label
    user_folder_subpath: tuple[str, ...]    # path relative to %USERPROFILE%\Documents

    # Texture filenames this title's livery folder is expected to contain.
    # FD6's livery_writer maps the user's source image onto one of these slots.
    main_decal_filenames: tuple[str, ...] = ()
    sponsor_decal_filenames: tuple[str, ...] = ()

    # Common output resolutions the title accepts. First entry is the default.
    accepted_resolutions: tuple[int, ...] = (4096, 2048, 1024)

    # Accepted aspect ratios for source images (UI exposes these as Auto / 1:1 / 1:2 / 1:4).
    accepted_aspects: tuple[tuple[int, int], ...] = ((1, 1), (1, 2), (1, 4))

    # When False, the GUI shows the title in the dropdown but disables it and
    # tags it "Coming Soon". Lets us surface the roadmap without writing
    # speculative export code.
    implemented: bool = False
    coming_soon_note: str = ""


# Assetto Corsa Competizione — the only title with full v0.3.5 export support.
# Custom liveries land in:
#   %USERPROFILE%\Documents\Assetto Corsa Competizione\Customs\Liveries\<team>\
# Each livery folder contains decals.json metadata plus PNG layers.
ACC = ACTitleProfile(
    key="acc",
    label="Assetto Corsa Competizione",
    user_folder_subpath=("Assetto Corsa Competizione", "Customs", "Liveries"),
    main_decal_filenames=("decals.png", "decals_0.png", "decals_1.png"),
    sponsor_decal_filenames=("sponsors.png",),
    accepted_resolutions=(4096, 2048, 1024),
    accepted_aspects=((1, 1), (1, 2), (1, 4)),
    implemented=True,
)

# Assetto Corsa Evo — Kunos's newer EA title. Livery system not yet
# publicly documented to the same standard as ACC. Surfaced in the dropdown
# as Coming Soon until we can verify file layout against a live install.
ACE = ACTitleProfile(
    key="ace",
    label="Assetto Corsa Evo",
    user_folder_subpath=("Assetto Corsa Evo", "Customs", "Liveries"),
    implemented=False,
    coming_soon_note=(
        "ACE livery folder layout has not been independently verified by FD6 "
        "against an installed copy yet. Coming in a future release once the "
        "PNG slot layout and decals.json schema are confirmed."
    ),
)

# Assetto Corsa Rally — Kunos rally-focused title. Treated separately from
# the main AC line because rally cars use different decal sheet conventions.
AC_RALLY = ACTitleProfile(
    key="ac_rally",
    label="Assetto Corsa Rally",
    user_folder_subpath=("Assetto Corsa Rally", "Customs", "Liveries"),
    implemented=False,
    coming_soon_note=(
        "AC Rally livery export support is planned. The file layout differs "
        "from ACC; verification against a live install is required first."
    ),
)

# Assetto Corsa (original 2014). Livery system is well-documented but lives
# in a totally different folder structure — content/cars/<car>/skins/<skin>/
# inside the install directory rather than Documents. Deferred until we
# can ship that path-resolution code cleanly.
AC_ORIGINAL = ACTitleProfile(
    key="ac",
    label="Assetto Corsa",
    user_folder_subpath=(),  # uses install-folder skins/, resolved elsewhere
    implemented=False,
    coming_soon_note=(
        "Original AC stores liveries (skins) inside the game's install "
        "directory rather than Documents. Coming in a future release once "
        "install-folder discovery is implemented."
    ),
)


PROFILES: dict[str, ACTitleProfile] = {
    ACC.key: ACC,
    ACE.key: ACE,
    AC_RALLY.key: AC_RALLY,
    AC_ORIGINAL.key: AC_ORIGINAL,
}


def get_profile(key: str) -> ACTitleProfile:
    normalized = (key or "").lower().strip()
    if normalized not in PROFILES:
        supported = ", ".join(PROFILES)
        raise ValueError(f"Unknown AC title profile '{key}'. Known: {supported}")
    return PROFILES[normalized]


def default_profile() -> ACTitleProfile:
    """ACC — the only implemented AC title in v0.3.5."""
    return ACC


def list_profiles() -> list[ACTitleProfile]:
    """All AC titles in dropdown display order — production-ready first."""
    return [ACC, ACE, AC_RALLY, AC_ORIGINAL]
