from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar
import random

import numpy as np

ShapeType = str

SHAPE_REGISTRY: dict[ShapeType, type["Shape"]] = {}


def _register(cls: type["Shape"]) -> type["Shape"]:
    SHAPE_REGISTRY[cls.type_name] = cls
    return cls


@dataclass
class Shape(ABC):
    """Abstract shape primitive. Color (RGBA 0-255) is tracked separately from geometry."""

    type_name: ClassVar[ShapeType] = "shape"

    color: tuple[int, int, int, int] = (0, 0, 0, 128)

    @abstractmethod
    def bbox(self, w: int, h: int) -> tuple[int, int, int, int]:
        """Return clipped (x0, y0, x1, y1) for rasterization. Coordinates are inclusive-exclusive."""

    @abstractmethod
    def rasterize_mask(self, w: int, h: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        """Return (alpha_mask_uint8, bbox). Mask is local to the bbox, not full-image."""

    @abstractmethod
    def mutate(self, rng: random.Random, w: int, h: int) -> "Shape":
        """Return a mutated copy. Color is mutated by the engine after geometry."""

    @abstractmethod
    def to_json(self) -> dict:
        ...

    @classmethod
    @abstractmethod
    def from_json(cls, data: dict) -> "Shape":
        ...

    @classmethod
    @abstractmethod
    def random(cls, rng: random.Random, w: int, h: int) -> "Shape":
        ...

    def with_color(self, color: tuple[int, int, int, int]) -> "Shape":
        from copy import copy as shallow_copy
        new = shallow_copy(self)
        new.color = color
        return new


def random_shape(rng: random.Random, w: int, h: int, allowed_types: list[ShapeType]) -> Shape:
    type_name = rng.choice(allowed_types)
    cls = SHAPE_REGISTRY[type_name]
    return cls.random(rng, w, h)


def shape_from_json(data: dict) -> Shape:
    type_name = data.get("type")
    if type_name not in SHAPE_REGISTRY:
        raise ValueError(f"Unknown shape type: {type_name!r}")
    return SHAPE_REGISTRY[type_name].from_json(data)


def _clip_bbox(x0: float, y0: float, x1: float, y1: float, w: int, h: int) -> tuple[int, int, int, int]:
    cx0 = max(0, int(np.floor(x0)))
    cy0 = max(0, int(np.floor(y0)))
    cx1 = min(w, int(np.ceil(x1)))
    cy1 = min(h, int(np.ceil(y1)))
    if cx1 <= cx0 or cy1 <= cy0:
        return (0, 0, 0, 0)
    return cx0, cy0, cx1, cy1


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))
