from __future__ import annotations

import configparser
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class Profile:
    name: str = "default"
    description: str = "Default profile"
    max_preview_size: int = 500
    max_resolution: int = 1200
    max_threads: int = 0  # 0 = auto (os.cpu_count())
    mutated_samples: int = 200
    posterize_levels: int = 256
    preview_every: int = 1
    random_samples: int = 1000
    redundant_check_every: int = 500
    save_at: list[int] = field(default_factory=lambda: [500, 1000, 1500, 2000, 2500, 3000])
    save_every: int = 100
    stop_at: int = 3000
    shape_types: list[str] = field(default_factory=lambda: ["rotated_ellipse"])

    def to_ini(self) -> str:
        cp = configparser.ConfigParser()
        cp["profile"] = {
            "description": self.description,
            "maxPreviewSize": str(self.max_preview_size),
            "maxResolution": str(self.max_resolution),
            "maxThreads": str(self.max_threads),
            "mutatedSamples": str(self.mutated_samples),
            "posterizeLevels": str(self.posterize_levels),
            "previewEvery": str(self.preview_every),
            "randomSamples": str(self.random_samples),
            "redundantCheckEvery": str(self.redundant_check_every),
            "saveAt": ",".join(str(s) for s in self.save_at),
            "saveEvery": str(self.save_every),
            "stopAt": str(self.stop_at),
            "shapeTypes": ",".join(self.shape_types),
        }
        from io import StringIO
        buf = StringIO()
        cp.write(buf)
        return buf.getvalue()


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_str_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def load_profile(name: str, text: str) -> Profile:
    cp = configparser.ConfigParser()
    # forza-painter .ini files don't use a section header. Try parsing as-is first;
    # on MissingSectionHeaderError, prepend a synthetic [profile] header and retry.
    try:
        cp.read_string(text)
    except configparser.MissingSectionHeaderError:
        cp = configparser.ConfigParser()
        cp.read_string("[profile]\n" + text)
    if cp.has_section("profile"):
        section = cp["profile"]
    else:
        cp = configparser.ConfigParser()
        cp.read_string("[profile]\n" + text)
        section = cp["profile"]

    p = Profile(name=name)
    getstr = lambda k, d: section.get(k, str(d))
    getint = lambda k, d: int(section.get(k, str(d)))

    p.description = getstr("description", p.description)
    p.max_preview_size = getint("maxPreviewSize", p.max_preview_size)
    p.max_resolution = getint("maxResolution", p.max_resolution)
    p.max_threads = getint("maxThreads", p.max_threads)
    p.mutated_samples = getint("mutatedSamples", p.mutated_samples)
    p.posterize_levels = getint("posterizeLevels", p.posterize_levels)
    p.preview_every = getint("previewEvery", p.preview_every)
    p.random_samples = getint("randomSamples", p.random_samples)
    p.redundant_check_every = getint("redundantCheckEvery", p.redundant_check_every)
    if "saveAt" in section:
        p.save_at = _parse_int_list(section["saveAt"])
    p.save_every = getint("saveEvery", p.save_every)
    p.stop_at = getint("stopAt", p.stop_at)
    if "shapeTypes" in section:
        p.shape_types = _parse_str_list(section["shapeTypes"])
    return p


def load_profile_from_file(path: str | Path) -> Profile:
    path = Path(path)
    return load_profile(path.stem, path.read_text(encoding="utf-8"))


def list_bundled_profiles() -> list[Path]:
    base = Path(__file__).resolve().parent.parent / "settings" / "profiles"
    if not base.exists():
        return []
    return sorted(base.glob("*.ini"))
