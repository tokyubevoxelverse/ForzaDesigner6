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
    compute_backend: str = "auto"
    mutated_samples: int = 200
    posterize_levels: int = 256
    preview_every: int = 10
    random_samples: int = 1000
    refine_passes: int = 0
    redundant_check_every: int = 500
    edge_weight_strength: float = 0.75
    gradient_weight: float = 0.12
    edge_candidate_ratio: float = 0.18
    edge_candidate_alpha: int = 224
    edge_rerank_top_k: int = 24
    quality_batch_pixels: int = 0
    line_guide_enabled: bool = False
    line_guide_image_path: str = ""
    line_guide_model_path: str = ""
    line_guide_strength: float = 0.75
    line_guide_decay: float = 0.55
    line_guide_agreement: float = 0.65
    line_guide_candidate_ratio: float = 0.22
    line_guide_max_resolution: int = 1024
    save_at: list[int] = field(default_factory=lambda: [500, 1000, 1500, 2000, 2500, 3000])
    save_every: int = 0
    stop_at: int = 3000
    shape_types: list[str] = field(default_factory=lambda: [
        "rotated_ellipse",
        "ellipse",
        "circle",
        "triangle",
        "rectangle",
        "rotated_rectangle",
    ])

    def to_ini(self) -> str:
        cp = configparser.ConfigParser()
        cp["profile"] = {
            "description": self.description,
            "maxPreviewSize": str(self.max_preview_size),
            "maxResolution": str(self.max_resolution),
            "maxThreads": str(self.max_threads),
            "computeBackend": self.compute_backend,
            "mutatedSamples": str(self.mutated_samples),
            "posterizeLevels": str(self.posterize_levels),
            "previewEvery": str(self.preview_every),
            "randomSamples": str(self.random_samples),
            "refinePasses": str(self.refine_passes),
            "redundantCheckEvery": str(self.redundant_check_every),
            "edgeWeightStrength": str(self.edge_weight_strength),
            "gradientWeight": str(self.gradient_weight),
            "edgeCandidateRatio": str(self.edge_candidate_ratio),
            "edgeCandidateAlpha": str(self.edge_candidate_alpha),
            "edgeRerankTopK": str(self.edge_rerank_top_k),
            "qualityBatchPixels": str(self.quality_batch_pixels),
            "lineGuideEnabled": str(self.line_guide_enabled),
            "lineGuideImagePath": self.line_guide_image_path,
            "lineGuideModelPath": self.line_guide_model_path,
            "lineGuideStrength": str(self.line_guide_strength),
            "lineGuideDecay": str(self.line_guide_decay),
            "lineGuideAgreement": str(self.line_guide_agreement),
            "lineGuideCandidateRatio": str(self.line_guide_candidate_ratio),
            "lineGuideMaxResolution": str(self.line_guide_max_resolution),
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
    getfloat = lambda k, d: float(section.get(k, str(d)))
    getbool = lambda k, d: section.getboolean(k, fallback=bool(d))

    p.description = getstr("description", p.description)
    p.max_preview_size = getint("maxPreviewSize", p.max_preview_size)
    p.max_resolution = getint("maxResolution", p.max_resolution)
    p.max_threads = getint("maxThreads", p.max_threads)
    p.compute_backend = getstr("computeBackend", p.compute_backend).strip().lower() or "auto"
    p.mutated_samples = getint("mutatedSamples", p.mutated_samples)
    p.posterize_levels = getint("posterizeLevels", p.posterize_levels)
    p.preview_every = getint("previewEvery", p.preview_every)
    p.random_samples = getint("randomSamples", p.random_samples)
    p.refine_passes = getint("refinePasses", p.refine_passes)
    p.redundant_check_every = getint("redundantCheckEvery", p.redundant_check_every)
    p.edge_weight_strength = getfloat("edgeWeightStrength", p.edge_weight_strength)
    p.gradient_weight = getfloat("gradientWeight", p.gradient_weight)
    p.edge_candidate_ratio = getfloat("edgeCandidateRatio", p.edge_candidate_ratio)
    p.edge_candidate_alpha = getint("edgeCandidateAlpha", p.edge_candidate_alpha)
    p.edge_rerank_top_k = getint("edgeRerankTopK", p.edge_rerank_top_k)
    p.quality_batch_pixels = getint("qualityBatchPixels", p.quality_batch_pixels)
    p.line_guide_enabled = getbool("lineGuideEnabled", p.line_guide_enabled)
    p.line_guide_image_path = getstr("lineGuideImagePath", p.line_guide_image_path)
    p.line_guide_model_path = getstr("lineGuideModelPath", p.line_guide_model_path)
    p.line_guide_strength = getfloat("lineGuideStrength", p.line_guide_strength)
    p.line_guide_decay = getfloat("lineGuideDecay", p.line_guide_decay)
    p.line_guide_agreement = getfloat("lineGuideAgreement", p.line_guide_agreement)
    p.line_guide_candidate_ratio = getfloat("lineGuideCandidateRatio", p.line_guide_candidate_ratio)
    p.line_guide_max_resolution = getint("lineGuideMaxResolution", p.line_guide_max_resolution)
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
