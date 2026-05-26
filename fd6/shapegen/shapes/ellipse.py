from __future__ import annotations

from dataclasses import dataclass
import math
import random

import numpy as np

from fd6.shapegen.shapes.base import Shape, _clip_bbox, _clamp, _register


@dataclass
class Ellipse(Shape):
    type_name = "ellipse"

    x: float = 0.0
    y: float = 0.0
    rx: float = 1.0
    ry: float = 1.0

    def bbox(self, w: int, h: int) -> tuple[int, int, int, int]:
        return _clip_bbox(self.x - self.rx, self.y - self.ry, self.x + self.rx + 1, self.y + self.ry + 1, w, h)

    def rasterize_mask(self, w: int, h: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        bbox = self.bbox(w, h)
        x0, y0, x1, y1 = bbox
        if x1 <= x0 or y1 <= y0:
            return np.zeros((0, 0), dtype=np.uint8), bbox
        ys = np.arange(y0, y1, dtype=np.float32) - self.y
        xs = np.arange(x0, x1, dtype=np.float32) - self.x
        dx = xs / max(self.rx, 1e-6)
        dy = ys / max(self.ry, 1e-6)
        dx2 = dx[None, :] * dx[None, :]
        dy2 = dy[:, None] * dy[:, None]
        mask = (dx2 + dy2) <= 1.0
        return (mask.astype(np.uint8) * 255), bbox

    def mutate(self, rng: random.Random, w: int, h: int) -> "Ellipse":
        new = self._copy_for_mutation()
        which = rng.randint(0, 2)
        if which == 0:
            new.x = _clamp(new.x + rng.gauss(0, 16), 0, w - 1)
            new.y = _clamp(new.y + rng.gauss(0, 16), 0, h - 1)
        elif which == 1:
            new.rx = _clamp(new.rx + rng.gauss(0, 16), 1, w)
            new.ry = _clamp(new.ry + rng.gauss(0, 16), 1, h)
        else:
            new.x = _clamp(new.x + rng.gauss(0, 8), 0, w - 1)
            new.y = _clamp(new.y + rng.gauss(0, 8), 0, h - 1)
            new.rx = _clamp(new.rx + rng.gauss(0, 8), 1, w)
            new.ry = _clamp(new.ry + rng.gauss(0, 8), 1, h)
        return new

    def to_json(self) -> dict:
        return {
            "type": self.type_name,
            "x": round(self.x, 3), "y": round(self.y, 3),
            "rx": round(self.rx, 3), "ry": round(self.ry, 3),
            "color": list(self.color),
        }

    @classmethod
    def from_json(cls, data: dict) -> "Ellipse":
        return cls(
            color=tuple(data["color"]),
            x=float(data["x"]), y=float(data["y"]),
            rx=float(data["rx"]), ry=float(data["ry"]),
        )

    @classmethod
    def random(cls, rng: random.Random, w: int, h: int, max_size_frac: float | None = None) -> "Ellipse":
        if max_size_frac is None:
            rx_cap = max(2.0, w / 8.0)
            ry_cap = max(2.0, h / 8.0)
        else:
            rx_cap = max(2.0, (w * max_size_frac) / 2.0)
            ry_cap = max(2.0, (h * max_size_frac) / 2.0)
        return cls(
            color=(rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255), 128),
            x=rng.uniform(0, w - 1), y=rng.uniform(0, h - 1),
            rx=rng.uniform(1, rx_cap), ry=rng.uniform(1, ry_cap),
        )


_register(Ellipse)


@dataclass
class RotatedEllipse(Shape):
    type_name = "rotated_ellipse"

    x: float = 0.0
    y: float = 0.0
    rx: float = 1.0
    ry: float = 1.0
    angle: float = 0.0  # degrees

    def bbox(self, w: int, h: int) -> tuple[int, int, int, int]:
        rad = math.radians(self.angle)
        cos_a = abs(math.cos(rad))
        sin_a = abs(math.sin(rad))
        ext_x = math.sqrt((self.rx * cos_a) ** 2 + (self.ry * sin_a) ** 2)
        ext_y = math.sqrt((self.rx * sin_a) ** 2 + (self.ry * cos_a) ** 2)
        return _clip_bbox(self.x - ext_x, self.y - ext_y, self.x + ext_x + 1, self.y + ext_y + 1, w, h)

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
        dx = xr / max(self.rx, 1e-6)
        dy = yr / max(self.ry, 1e-6)
        mask = (dx * dx + dy * dy) <= 1.0
        return (mask.astype(np.uint8) * 255), bbox

    def mutate(self, rng: random.Random, w: int, h: int) -> "RotatedEllipse":
        new = self._copy_for_mutation()
        which = rng.randint(0, 3)
        if which == 0:
            new.x = _clamp(new.x + rng.gauss(0, 16), 0, w - 1)
            new.y = _clamp(new.y + rng.gauss(0, 16), 0, h - 1)
        elif which == 1:
            new.rx = _clamp(new.rx + rng.gauss(0, 16), 1, w)
            new.ry = _clamp(new.ry + rng.gauss(0, 16), 1, h)
        elif which == 2:
            new.angle = (new.angle + rng.gauss(0, 25)) % 180.0
        else:
            new.x = _clamp(new.x + rng.gauss(0, 8), 0, w - 1)
            new.y = _clamp(new.y + rng.gauss(0, 8), 0, h - 1)
            new.angle = (new.angle + rng.gauss(0, 15)) % 180.0
        return new

    def to_json(self) -> dict:
        return {
            "type": self.type_name,
            "x": round(self.x, 3), "y": round(self.y, 3),
            "rx": round(self.rx, 3), "ry": round(self.ry, 3),
            "angle": round(self.angle, 3),
            "color": list(self.color),
        }

    @classmethod
    def from_json(cls, data: dict) -> "RotatedEllipse":
        return cls(
            color=tuple(data["color"]),
            x=float(data["x"]), y=float(data["y"]),
            rx=float(data["rx"]), ry=float(data["ry"]),
            angle=float(data["angle"]),
        )

    @classmethod
    def random(cls, rng: random.Random, w: int, h: int, max_size_frac: float | None = None) -> "RotatedEllipse":
        if max_size_frac is None:
            rx_cap = max(2.0, w / 8.0)
            ry_cap = max(2.0, h / 8.0)
        else:
            rx_cap = max(2.0, (w * max_size_frac) / 2.0)
            ry_cap = max(2.0, (h * max_size_frac) / 2.0)
        return cls(
            color=(rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255), 128),
            x=rng.uniform(0, w - 1), y=rng.uniform(0, h - 1),
            rx=rng.uniform(1, rx_cap), ry=rng.uniform(1, ry_cap),
            angle=rng.uniform(0, 180),
        )


_register(RotatedEllipse)
