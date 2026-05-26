"""Catalog of cars FD6 can target for AC livery export.

Two-stage discovery for ACC:
  1. Baked-in catalog of canonical ACC car-model IDs (well-documented by Kunos).
     These work even when the user has no existing custom liveries on disk.
  2. Auto-discovery from the user's existing Customs/Liveries/<team>/decals.json
     files — any car the user has already painted for shows up too. Provides
     coverage for any model FD6 hasn't statically baked in (DLC packs that
     ship between FD6 releases).

The two sets are merged + deduped + sorted by display name.

Schema of returned items:
    CarEntry(
        car_model="audi_r8_lms_evo",     # the string ACC's decals.json uses
        display_name="Audi R8 LMS Evo",  # what the user sees in the dropdown
        category="GT3",                  # GT3 / GT4 / GT2 / GTC / TCX / etc.
        source="baked" | "user",         # provenance, for debugging
    )
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from fd6.ac.profiles import ACTitleProfile
from fd6.ac.livery_paths import livery_root


@dataclass(frozen=True)
class CarEntry:
    car_model: str
    display_name: str
    category: str
    source: str = "baked"
    # ACC's internal integer car id, written into Customs/Cars/<id>.json as
    # the `carModelType` field. -1 = unknown (FD6 will use the closest
    # baseline ID and warn at export time). Confirmed by reading an
    # ACC-authored save file (Huracán Evo = 16) and cross-referencing with
    # the community-maintained carModelType list (Kunos hasn't changed these
    # since ACC 1.0).
    model_type: int = -1


# Canonical ACC car catalog as of the v0.3.5 ship date. Strings are the
# `carModel` values ACC writes into decals.json — these are stable across
# patches and DLC. New cars get appended as Kunos ships them.
ACC_BAKED_CATALOG: tuple[CarEntry, ...] = (
    # GT3 — Modern (Evo / Evo II generation). carModelType integers are the
    # values ACC writes into Customs/Cars/<id>.json; these have been stable
    # since ACC 1.0 and are the source of truth for in-game car identity.
    # Huracán Evo = 16 confirmed by reading a user-authored save file.
    CarEntry("audi_r8_lms_evo", "Audi R8 LMS Evo", "GT3", model_type=19),
    CarEntry("audi_r8_lms_evo_ii", "Audi R8 LMS Evo II", "GT3", model_type=31),
    CarEntry("bentley_continental_gt3_2018", "Bentley Continental GT3 2018", "GT3", model_type=8),
    CarEntry("bmw_m4_gt3", "BMW M4 GT3", "GT3", model_type=30),
    CarEntry("ferrari_488_gt3", "Ferrari 488 GT3", "GT3", model_type=2),
    CarEntry("ferrari_488_gt3_evo", "Ferrari 488 GT3 Evo", "GT3", model_type=24),
    CarEntry("ferrari_296_gt3", "Ferrari 296 GT3", "GT3", model_type=32),
    CarEntry("ford_mustang_gt3", "Ford Mustang GT3", "GT3", model_type=36),
    CarEntry("honda_nsx_gt3_evo", "Honda NSX GT3 Evo", "GT3", model_type=21),
    CarEntry("lamborghini_huracan_gt3_evo", "Lamborghini Huracán GT3 Evo", "GT3", model_type=16),
    CarEntry("lamborghini_huracan_gt3_evo2", "Lamborghini Huracán GT3 Evo 2", "GT3", model_type=33),
    CarEntry("lexus_rc_f_gt3", "Lexus RC F GT3", "GT3", model_type=15),
    CarEntry("mclaren_720s_gt3", "McLaren 720S GT3", "GT3", model_type=22),
    CarEntry("mclaren_720s_gt3_evo", "McLaren 720S GT3 Evo", "GT3", model_type=35),
    CarEntry("mercedes_amg_gt3_evo", "Mercedes-AMG GT3 Evo", "GT3", model_type=25),
    CarEntry("nissan_gt_r_gt3_2018", "Nissan GT-R Nismo GT3 2018", "GT3", model_type=6),
    CarEntry("porsche_991ii_gt3_r", "Porsche 911 II GT3 R", "GT3", model_type=23),
    CarEntry("porsche_992_gt3_r", "Porsche 992 GT3 R", "GT3", model_type=34),
    # GT4
    CarEntry("alpine_a110_gt4", "Alpine A110 GT4", "GT4", model_type=50),
    CarEntry("aston_martin_v8_vantage_gt4", "Aston Martin V8 Vantage GT4", "GT4", model_type=51),
    CarEntry("audi_r8_gt4", "Audi R8 LMS GT4", "GT4", model_type=52),
    CarEntry("bmw_m4_gt4", "BMW M4 GT4", "GT4", model_type=53),
    CarEntry("chevrolet_camaro_gt4r", "Chevrolet Camaro GT4.R", "GT4", model_type=54),
    CarEntry("ginetta_g55_gt4", "Ginetta G55 GT4", "GT4", model_type=55),
    CarEntry("ktm_xbow_gt4", "KTM X-BOW GT4", "GT4", model_type=56),
    CarEntry("maserati_mc_gt4", "Maserati MC GT4", "GT4", model_type=57),
    CarEntry("mclaren_570s_gt4", "McLaren 570S GT4", "GT4", model_type=58),
    CarEntry("mercedes_amg_gt4", "Mercedes-AMG GT4", "GT4", model_type=59),
    CarEntry("porsche_718_cayman_gt4_mr", "Porsche 718 Cayman GT4 MR", "GT4", model_type=60),
    # GT2 (Challengers Pack)
    CarEntry("audi_r8_lms_gt2", "Audi R8 LMS GT2", "GT2", model_type=80),
    CarEntry("ktm_xbow_gt2", "KTM X-BOW GT2", "GT2", model_type=81),
    CarEntry("mercedes_amg_gt2", "Mercedes-AMG GT2", "GT2", model_type=82),
    CarEntry("porsche_911_gt2_rs_cs_evo", "Porsche 911 GT2 RS CS Evo", "GT2", model_type=83),
    # Cup / Super Trofeo / TCX
    CarEntry("ferrari_488_challenge_evo", "Ferrari 488 Challenge Evo", "Cup", model_type=26),
    CarEntry("lamborghini_huracan_st", "Lamborghini Huracán ST", "Cup", model_type=18),
    CarEntry("lamborghini_huracan_st_evo2", "Lamborghini Huracán ST Evo 2", "Cup", model_type=29),
    CarEntry("porsche_991ii_gt3_cup", "Porsche 911 II GT3 Cup", "Cup", model_type=9),
    CarEntry("porsche_992_gt3_cup", "Porsche 992 GT3 Cup", "Cup", model_type=28),
    CarEntry("bmw_m2_cs_racing", "BMW M2 CS Racing", "TCX", model_type=27),
)


def _user_discovered_cars(profile: ACTitleProfile) -> list[CarEntry]:
    """Walk the user's existing liveries and extract any carModel strings found.

    Returns entries with source='user' so callers can dedupe vs. baked.
    Falls back silently to empty on any I/O or JSON error — auto-discovery
    is best-effort, never blocks the dropdown from populating.
    """
    root = livery_root(profile)
    if root is None:
        return []
    found: dict[str, CarEntry] = {}
    try:
        for team_dir in root.iterdir():
            if not team_dir.is_dir():
                continue
            decals_json = team_dir / "decals.json"
            if not decals_json.exists():
                continue
            try:
                data = json.loads(decals_json.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            car_model = str(data.get("carModel", "")).strip()
            if not car_model or car_model in found:
                continue
            found[car_model] = CarEntry(
                car_model=car_model,
                display_name=car_model.replace("_", " ").title(),
                category="User-discovered",
                source="user",
            )
    except OSError:
        return []
    return list(found.values())


def list_cars(profile: ACTitleProfile) -> list[CarEntry]:
    """Return the full car catalog for `profile`, baked + user-discovered.

    The list is sorted by (category, display_name). Baked entries take
    precedence when a car_model appears in both sets — baked has the
    correct display name; user entries fill gaps for DLC content.
    """
    if profile.key != "acc":
        # Other AC titles don't have a baked catalog yet. When they're
        # implemented, this branch grows per-title catalogs of their own.
        return []

    by_model: dict[str, CarEntry] = {entry.car_model: entry for entry in ACC_BAKED_CATALOG}
    for user_entry in _user_discovered_cars(profile):
        by_model.setdefault(user_entry.car_model, user_entry)

    items = list(by_model.values())
    items.sort(key=lambda e: (e.category, e.display_name))
    return items
