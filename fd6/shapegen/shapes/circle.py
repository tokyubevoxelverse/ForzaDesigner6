from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np

from fd6.shapegen.shapes.base import Shape, _clip_bbox, _clamp, _register


@dataclass
class Circle(Shape):
    type_name = "circle"

    x: float = 0.0
    y: float = 0.0
    r: float = 1.0

    def bbox(self, w: int, h: int) -> tuple[int, int, int, int]:
        return _clip_bbox(self.x - self.r, self.y - self.r, self.x + self.r + 1, self.y + self.r + 1, w, h)

    def rasterize_mask(self, w: int, h: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        bbox = self.bbox(w, h)
        x0, y0, x1, y1 = bbox
        if x1 <= x0 or y1 <= y0:
            return np.zeros((0, 0), dtype=np.uint8), bbox
        ys = np.arange(y0, y1, dtype=np.float32) - self.y
        xs = np.arange(x0, x1, dtype=np.float32) - self.x
        r2 = max(self.r, 1e-6) ** 2
        mask = (xs[None, :] ** 2 + ys[:, None] ** 2) <= r2
        return (mask.astype(np.uint8) * 255), bbox

    def mutate(self, rng: random.Random, w: int, h: int) -> "Circle":
        new = self._copy_for_mutation()
        which = rng.randint(0, 1)
        if which == 0:
            new.x = _clamp(new.x + rng.gauss(0, 16), 0, w - 1)
            new.y = _clamp(new.y + rng.gauss(0, 16), 0, h - 1)
        else:
            new.r = _clamp(new.r + rng.gauss(0, 16), 1, max(w, h))
        return new

    def to_json(self) -> dict:
        return {
            "type": self.type_name,
            "x": round(self.x, 3), "y": round(self.y, 3),
            "r": round(self.r, 3),
            "color": list(self.color),
        }

    @classmethod
    def from_json(cls, data: dict) -> "Circle":
        return cls(
            color=tuple(data["color"]),
            x=float(data["x"]), y=float(data["y"]),
            r=float(data["r"]),
        )

    @classmethod
    def random(cls, rng: random.Random, w: int, h: int, max_size_frac: float | None = None) -> "Circle":
        if max_size_frac is None:
            r_cap = max(2.0, min(w, h) / 8.0)
        else:
            r_cap = max(2.0, (min(w, h) * max_size_frac) / 2.0)
        return cls(
            color=(rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255), 128),
            x=rng.uniform(0, w - 1), y=rng.uniform(0, h - 1),
            r=rng.uniform(1, r_cap),
        )


_register(Circle)
