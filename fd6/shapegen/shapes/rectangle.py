from __future__ import annotations

from dataclasses import dataclass
import math
import random

import numpy as np

from fd6.shapegen.shapes.base import Shape, _clip_bbox, _clamp, _register


@dataclass
class Rectangle(Shape):
    type_name = "rectangle"

    x: float = 0.0
    y: float = 0.0
    hw: float = 1.0  # half-width
    hh: float = 1.0  # half-height

    def bbox(self, w: int, h: int) -> tuple[int, int, int, int]:
        return _clip_bbox(self.x - self.hw, self.y - self.hh, self.x + self.hw + 1, self.y + self.hh + 1, w, h)

    def rasterize_mask(self, w: int, h: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        bbox = self.bbox(w, h)
        x0, y0, x1, y1 = bbox
        if x1 <= x0 or y1 <= y0:
            return np.zeros((0, 0), dtype=np.uint8), bbox
        return np.full((y1 - y0, x1 - x0), 255, dtype=np.uint8), bbox

    def mutate(self, rng: random.Random, w: int, h: int) -> "Rectangle":
        new = self._copy_for_mutation()
        which = rng.randint(0, 1)
        if which == 0:
            new.x = _clamp(new.x + rng.gauss(0, 16), 0, w - 1)
            new.y = _clamp(new.y + rng.gauss(0, 16), 0, h - 1)
        else:
            new.hw = _clamp(new.hw + rng.gauss(0, 16), 1, w)
            new.hh = _clamp(new.hh + rng.gauss(0, 16), 1, h)
        return new

    def to_json(self) -> dict:
        return {
            "type": self.type_name,
            "x": round(self.x, 3), "y": round(self.y, 3),
            "hw": round(self.hw, 3), "hh": round(self.hh, 3),
            "color": list(self.color),
        }

    @classmethod
    def from_json(cls, data: dict) -> "Rectangle":
        return cls(
            color=tuple(data["color"]),
            x=float(data["x"]), y=float(data["y"]),
            hw=float(data["hw"]), hh=float(data["hh"]),
        )

    @classmethod
    def random(cls, rng: random.Random, w: int, h: int) -> "Rectangle":
        return cls(
            color=(rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255), 128),
            x=rng.uniform(0, w - 1), y=rng.uniform(0, h - 1),
            hw=rng.uniform(1, max(2, w / 8)), hh=rng.uniform(1, max(2, h / 8)),
        )


_register(Rectangle)


@dataclass
class RotatedRectangle(Shape):
    type_name = "rotated_rectangle"

    x: float = 0.0
    y: float = 0.0
    hw: float = 1.0
    hh: float = 1.0
    angle: float = 0.0  # degrees

    def bbox(self, w: int, h: int) -> tuple[int, int, int, int]:
        r = math.hypot(self.hw, self.hh)
        return _clip_bbox(self.x - r, self.y - r, self.x + r + 1, self.y + r + 1, w, h)

    def rasterize_mask(self, w: int, h: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        bbox = self.bbox(w, h)
        x0, y0, x1, y1 = bbox
        if x1 <= x0 or y1 <= y0:
            return np.zeros((0, 0), dtype=np.uint8), bbox
        rad = math.radians(self.angle)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        ys = np.arange(y0, y1, dtype=np.float32) - self.y
        xs = np.arange(x0, x1, dtype=np.float32) - self.x
        xr = cos_a * xs[None, :] + sin_a * ys[:, None]
        yr = -sin_a * xs[None, :] + cos_a * ys[:, None]
        mask = (np.abs(xr) <= self.hw) & (np.abs(yr) <= self.hh)
        return (mask.astype(np.uint8) * 255), bbox

    def mutate(self, rng: random.Random, w: int, h: int) -> "RotatedRectangle":
        new = self._copy_for_mutation()
        which = rng.randint(0, 2)
        if which == 0:
            new.x = _clamp(new.x + rng.gauss(0, 16), 0, w - 1)
            new.y = _clamp(new.y + rng.gauss(0, 16), 0, h - 1)
        elif which == 1:
            new.hw = _clamp(new.hw + rng.gauss(0, 16), 1, w)
            new.hh = _clamp(new.hh + rng.gauss(0, 16), 1, h)
        else:
            new.angle = (new.angle + rng.gauss(0, 25)) % 180.0
        return new

    def to_json(self) -> dict:
        return {
            "type": self.type_name,
            "x": round(self.x, 3), "y": round(self.y, 3),
            "hw": round(self.hw, 3), "hh": round(self.hh, 3),
            "angle": round(self.angle, 3),
            "color": list(self.color),
        }

    @classmethod
    def from_json(cls, data: dict) -> "RotatedRectangle":
        return cls(
            color=tuple(data["color"]),
            x=float(data["x"]), y=float(data["y"]),
            hw=float(data["hw"]), hh=float(data["hh"]),
            angle=float(data["angle"]),
        )

    @classmethod
    def random(cls, rng: random.Random, w: int, h: int) -> "RotatedRectangle":
        return cls(
            color=(rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255), 128),
            x=rng.uniform(0, w - 1), y=rng.uniform(0, h - 1),
            hw=rng.uniform(1, max(2, w / 8)), hh=rng.uniform(1, max(2, h / 8)),
            angle=rng.uniform(0, 180),
        )


_register(RotatedRectangle)
