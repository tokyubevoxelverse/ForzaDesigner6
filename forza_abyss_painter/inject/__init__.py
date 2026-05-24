"""FH6 memory injection.

This package defines the injector interface and the concrete FH6 implementation,
which uses the LiveryGroup + layer_table discovery strategy (see
`fh6_injector.py`). For a new FH6 build, layout offsets may need to be
re-derived; the in-app FH6 → Discovery Workflow dialog documents the steps.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class VinylGroupHandle:
    """Opaque handle returned by Injector.find_active_vinyl_group(). Fields are filled in by the concrete injector."""

    base_addr: int = 0
    layer_count: int = 0
    shape_array_addr: int = 0
    shape_stride: int = 0
    meta: dict[str, Any] | None = None


@dataclass
class InjectResult:
    success: bool
    shapes_written: int = 0
    message: str = ""


class Injector(ABC):
    """Abstract base for all per-game injectors."""

    game_label: str = "unknown"

    @abstractmethod
    def attach(self) -> None:
        """Locate and open the game process. Raise on failure."""

    @abstractmethod
    def find_active_vinyl_group(self) -> VinylGroupHandle:
        """Locate the currently-loaded vinyl group in memory. Caller must have already
        loaded a template group with N pre-allocated shapes inside the game.
        """

    @abstractmethod
    def inject(self, shapes: list, group: VinylGroupHandle) -> InjectResult:
        """Overwrite the shape slots in `group` with `shapes`. The number of slots
        in the group must be >= len(shapes) (per the 3000-sphere template workflow).
        """

    def detach(self) -> None:  # default no-op
        pass


from forza_abyss_painter.inject.fh6_injector import FH6Injector, patterns_are_populated  # noqa: E402

__all__ = ["Injector", "VinylGroupHandle", "InjectResult", "FH6Injector", "patterns_are_populated"]
