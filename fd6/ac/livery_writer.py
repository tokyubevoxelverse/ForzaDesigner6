"""Write a complete ACC livery to the user's Documents folder.

ACC's livery picker rejects the naive single-folder layout. It actually
requires TWO artifacts:

  A) Customs/Cars/<filename>.json    — UTF-16 LE + BOM team-definition file.
                                       Links a chosen car (carModelType int)
                                       to a livery folder via customSkinName.
  B) Customs/Liveries/<customSkinName>/decals.png
                                     — the image. No JSON inside the folder
                                       is required by the picker when the
                                       Cars/<filename>.json above carries
                                       all team-side metadata.

Filename convention ACC itself uses for (A) is:
    <raceNumber>-<DDMMYY>-<HHMMSS>.json
Any unique name works; we mimic the convention so the file blends in.

Schema for (A) is taken from an ACC-authored save read on disk; every
field below was present in that file and the picker is known to reject
folders whose JSON omits required fields. UTF-16 is mandatory.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from fd6.ac.car_catalog import ACC_BAKED_CATALOG
from fd6.ac.livery_paths import _documents_root, livery_root
from fd6.ac.profiles import ACTitleProfile
from fd6.ac.texture_pipeline import save_texture_png


@dataclass
class LiveryWriteResult:
    success: bool
    team_folder: Path
    files_written: list[Path]
    message: str = ""


def _slugify_team_name(name: str) -> str:
    """Sanitize a free-form team name into a safe folder name."""
    name = name.strip()
    if not name:
        name = f"FD6_export_{int(time.time())}"
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name)[:64] or "FD6_export"


def _lookup_car_model_type(car_model: str) -> int:
    """Map the carModel string (e.g. 'lamborghini_huracan_gt3_evo') to ACC's
    carModelType integer. Returns -1 if the model isn't in the baked catalog
    (user-discovered DLC car); the caller decides whether to abort or fall
    through with a warning."""
    for entry in ACC_BAKED_CATALOG:
        if entry.car_model == car_model:
            return entry.model_type
    return -1


def _acc_filename_for(race_number: int) -> str:
    """Mimic ACC's own filename convention: <raceNumber>-<DDMMYY>-<HHMMSS>.json"""
    now = datetime.now()
    return f"{int(race_number)}-{now.strftime('%d%m%y')}-{now.strftime('%H%M%S')}.json"


def _new_guid() -> str:
    """Generate a GUID string in the form ACC uses for car/team identity."""
    return str(uuid.uuid4())


def _customs_root(profile: ACTitleProfile) -> Path | None:
    """Resolve Documents/Assetto Corsa Competizione/Customs (the parent of
    both Cars/ and Liveries/). Creates the path if it doesn't exist yet —
    ACC tolerates being given pre-created Customs/Cars and Customs/Liveries
    folders on first run."""
    existing = livery_root(profile)
    if existing is not None:
        return existing.parent
    docs = _documents_root()
    if docs is None:
        return None
    customs = docs.joinpath(*profile.user_folder_subpath[:-1]) if profile.user_folder_subpath else None
    if customs is None:
        return None
    customs.mkdir(parents=True, exist_ok=True)
    return customs


def _build_team_json(
    car_model_type: int,
    custom_skin_name: str,
    team_name: str,
    display_name: str,
    race_number: int,
    paint: dict | None = None,
) -> dict:
    """Construct the full team-definition payload ACC's picker expects.

    Field values mirror what ACC writes for a default save so the livery
    shows up without picking up stray visual elements the user didn't ask
    for. The optional `paint` dict overrides skin/rim color + material IDs;
    pass `ACPaintPanel.gather()` output here.
    """
    paint = paint or {}
    return {
        "carGuid": _new_guid(),
        "teamGuid": _new_guid(),
        "raceNumber": int(race_number),
        "raceNumberPadding": 0,
        "auxLightKey": 0,
        "auxLightColor": [255, 255, 255, 255],
        "skinTemplateKey": -1,
        "skinColor1Id": int(paint.get("skinColor1Id", 0)),
        "skinColor2Id": int(paint.get("skinColor2Id", 0)),
        "skinColor3Id": int(paint.get("skinColor3Id", 0)),
        "sponsorId": -1,
        "skinMaterialType1": int(paint.get("skinMaterialType1", 0)),
        "skinMaterialType2": int(paint.get("skinMaterialType2", 0)),
        "skinMaterialType3": int(paint.get("skinMaterialType3", 0)),
        "rimColor1Id": int(paint.get("rimColor1Id", 0)),
        "rimColor2Id": int(paint.get("rimColor2Id", 0)),
        "rimMaterialType1": int(paint.get("rimMaterialType1", 0)),
        "rimMaterialType2": int(paint.get("rimMaterialType2", 0)),
        "teamName": team_name,
        "nationality": 0,
        "displayName": display_name or team_name,
        "competitorName": "",
        "competitorNationality": 0,
        "teamTemplateKey": -1,
        "carModelType": int(car_model_type),
        "cupCategory": 0,
        "licenseType": 0,
        "useEnduranceKit": 0,
        "customSkinName": custom_skin_name,
        "bannerTemplateKey": -1,
    }


def write_acc_livery(
    profile: ACTitleProfile,
    car_model: str,
    team_name: str,
    rgba: np.ndarray,
    slot_filenames: list[str],
    display_name: str = "",
    race_number: int = 99,
    paint: dict | None = None,
) -> LiveryWriteResult:
    """Create the two-file ACC livery (Customs/Cars/<file>.json + Customs/
    Liveries/<folder>/decals.png) the picker actually accepts.

    `rgba` is reused across all `slot_filenames` requested by the user;
    multi-image-per-slot routing is a later feature.
    """
    if profile.key != "acc":
        return LiveryWriteResult(
            success=False, team_folder=Path(),
            files_written=[],
            message=(f"{profile.label} export is not implemented yet. "
                     "Only ACC is supported in v0.3.5."),
        )

    customs = _customs_root(profile)
    if customs is None:
        return LiveryWriteResult(
            success=False, team_folder=Path(),
            files_written=[],
            message=("Could not locate Documents\\Assetto Corsa Competizione. "
                     "Open ACC once to initialize the user folder, or set "
                     "%USERPROFILE% in your environment, then try again."),
        )

    car_model_type = _lookup_car_model_type(car_model)
    if car_model_type < 0:
        return LiveryWriteResult(
            success=False, team_folder=Path(),
            files_written=[],
            message=(f"Car model '{car_model}' has no known carModelType integer. "
                     "It's likely a DLC vehicle FD6 hasn't catalogued yet — "
                     "pick a different car or update car_catalog.py."),
        )

    skin_folder_name = _slugify_team_name(team_name)
    liveries_dir = customs / "Liveries"
    cars_dir = customs / "Cars"
    liveries_dir.mkdir(parents=True, exist_ok=True)
    cars_dir.mkdir(parents=True, exist_ok=True)

    team_folder = liveries_dir / skin_folder_name
    team_folder.mkdir(parents=True, exist_ok=True)

    files_written: list[Path] = []

    for slot in slot_filenames:
        out = team_folder / slot
        save_texture_png(rgba, out)
        files_written.append(out)

    team_payload = _build_team_json(
        car_model_type=car_model_type,
        custom_skin_name=skin_folder_name,
        team_name=team_name,
        display_name=display_name,
        race_number=race_number,
        paint=paint,
    )
    team_json_str = json.dumps(team_payload, indent=2, ensure_ascii=False)
    team_json_path = cars_dir / _acc_filename_for(race_number)
    # ACC requires UTF-16 LE with BOM. Python's "utf-16" codec emits LE+BOM
    # on Windows by default (utf-16-le omits the BOM, which the picker rejects).
    team_json_path.write_text(team_json_str, encoding="utf-16")
    files_written.append(team_json_path)

    return LiveryWriteResult(
        success=True,
        team_folder=team_folder,
        files_written=files_written,
        message=(
            f"Wrote {len(files_written)} file(s). Team JSON: {team_json_path.name}. "
            f"Skin folder: {team_folder.name}. Reload ACC's car-selection screen "
            f"to see the new livery under the matching car."
        ),
    )
