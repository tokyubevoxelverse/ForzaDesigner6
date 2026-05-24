from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Iterable

from forza_abyss_painter.shapegen.shapes import Shape, shape_from_json

FD6_FORMAT = "fd6.shapes"
FD6_VERSION = 1


@dataclass
class FD6Document:
    """v1 of the FD6 shape JSON document. See README for schema details."""

    format: str = FD6_FORMAT
    version: int = FD6_VERSION
    source_image: str = ""
    image_size: tuple[int, int] = (0, 0)  # (width, height)
    shape_count: int = 0
    generated_at: str = ""
    profile: str = ""
    # True when the JSON was generated with sticker mode (transparent backdrop —
    # "Add white background to transparent images" was UNCHECKED). Default False
    # for backwards compat with older JSONs that pre-date this field. Affects
    # how the GUI re-renders the preview on Upload JSON: sticker JSONs get a
    # transparent preview, non-sticker JSONs get a white canvas as before.
    sticker_mode: bool = False
    shapes: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["image_size"] = list(self.image_size)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "FD6Document":
        fmt = data.get("format")
        if fmt != FD6_FORMAT:
            raise ValueError(f"Unsupported document format: {fmt!r} (expected {FD6_FORMAT!r})")
        ver = data.get("version")
        if ver != FD6_VERSION:
            raise ValueError(f"Unsupported document version: {ver!r} (expected {FD6_VERSION})")
        size = data.get("image_size", [0, 0])
        return cls(
            format=fmt,
            version=ver,
            source_image=str(data.get("source_image", "")),
            image_size=(int(size[0]), int(size[1])),
            shape_count=int(data.get("shape_count", len(data.get("shapes", [])))),
            generated_at=str(data.get("generated_at", "")),
            profile=str(data.get("profile", "")),
            sticker_mode=bool(data.get("sticker_mode", False)),
            shapes=list(data.get("shapes", [])),
        )

    def materialize_shapes(self) -> list[Shape]:
        return [shape_from_json(s) for s in self.shapes]

    @classmethod
    def from_engine(
        cls,
        source_image: str,
        image_size: tuple[int, int],
        shapes: Iterable[Shape],
        profile_name: str = "",
        sticker_mode: bool = False,
    ) -> "FD6Document":
        shape_list = [s.to_json() for s in shapes]
        return cls(
            format=FD6_FORMAT,
            version=FD6_VERSION,
            source_image=source_image,
            image_size=image_size,
            shape_count=len(shape_list),
            generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            profile=profile_name,
            sticker_mode=sticker_mode,
            shapes=shape_list,
        )
