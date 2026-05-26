from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np

from fd6.shapegen.shapes.base import Shape, _clip_bbox, _clamp, _register


@dataclass
class Triangle(Shape):
    type_name = "triangle"

    x1: float = 0.0
    y1: float = 0.0
    x2: float = 0.0
    y2: float = 0.0
    x3: float = 0.0
    y3: float = 0.0

    def bbox(self, w: int, h: int) -> tuple[int, int, int, int]:
        xs = (self.x1, self.x2, self.x3)
        ys = (self.y1, self.y2, self.y3)
        return _clip_bbox(min(xs), min(ys), max(xs) + 1, max(ys) + 1, w, h)

    def rasterize_mask(self, w: int, h: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        bbox = self.bbox(w, h)
        x0, y0, x1, y1 = bbox
        if x1 <= x0 or y1 <= y0:
            return np.zeros((0, 0), dtype=np.uint8), bbox
        ys = np.arange(y0, y1, dtype=np.float32)[:, None]
        xs = np.arange(x0, x1, dtype=np.float32)[None, :]
        # Barycentric sign-of-edge test
        def edge(ax, ay, bx, by, px, py):
            return (bx - ax) * (py - ay) - (by - ay) * (px - ax)
        d1 = edge(self.x1, self.y1, self.x2, self.y2, xs, ys)
        d2 = edge(self.x2, self.y2, self.x3, self.y3, xs, ys)
        d3 = edge(self.x3, self.y3, self.x1, self.y1, xs, ys)
        has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
        has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
        mask = ~(has_neg & has_pos)
        return (mask.astype(np.uint8) * 255), bbox

    def mutate(self, rng: random.Random, w: int, h: int) -> "Triangle":
        new = self._copy_for_mutation()
        vertex = rng.randint(0, 2)
        dx, dy = rng.gauss(0, 16), rng.gauss(0, 16)
        if vertex == 0:
            new.x1 = _clamp(new.x1 + dx, 0, w - 1)
            new.y1 = _clamp(new.y1 + dy, 0, h - 1)
        elif vertex == 1:
            new.x2 = _clamp(new.x2 + dx, 0, w - 1)
            new.y2 = _clamp(new.y2 + dy, 0, h - 1)
        else:
            new.x3 = _clamp(new.x3 + dx, 0, w - 1)
            new.y3 = _clamp(new.y3 + dy, 0, h - 1)
        return new

    def to_json(self) -> dict:
        return {
            "type": self.type_name,
            "x1": round(self.x1, 3), "y1": round(self.y1, 3),
            "x2": round(self.x2, 3), "y2": round(self.y2, 3),
            "x3": round(self.x3, 3), "y3": round(self.y3, 3),
            "color": list(self.color),
        }

    @classmethod
    def from_json(cls, data: dict) -> "Triangle":
        return cls(
            color=tuple(data["color"]),
            x1=float(data["x1"]), y1=float(data["y1"]),
            x2=float(data["x2"]), y2=float(data["y2"]),
            x3=float(data["x3"]), y3=float(data["y3"]),
        )

    @classmethod
    def random(cls, rng: random.Random, w: int, h: int, max_size_frac: float | None = None) -> "Triangle":
        cx, cy = rng.uniform(0, w - 1), rng.uniform(0, h - 1)
        if max_size_frac is None:
            spread = max(4.0, min(w, h) / 8.0)
        else:
            # Gaussian spread ≈ half-diameter so the typical triangle dimension
            # stays within max_size_frac * canvas.
            spread = max(4.0, (min(w, h) * max_size_frac) / 2.0)
        return cls(
            color=(rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255), 128),
            x1=_clamp(cx + rng.gauss(0, spread), 0, w - 1),
            y1=_clamp(cy + rng.gauss(0, spread), 0, h - 1),
            x2=_clamp(cx + rng.gauss(0, spread), 0, w - 1),
            y2=_clamp(cy + rng.gauss(0, spread), 0, h - 1),
            x3=_clamp(cx + rng.gauss(0, spread), 0, w - 1),
            y3=_clamp(cy + rng.gauss(0, spread), 0, h - 1),
        )


_register(Triangle)
