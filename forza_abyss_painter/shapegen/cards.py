"""GPU VRAM registry for the input-resizer (fitsize).

Keyed on USABLE GiB, not the vendor/Colab label (Colab mislabels some cards). Values are
conservative (slightly below nameplate); fitsize applies an additional budget headroom factor.
"""
from __future__ import annotations

CARDS: dict[str, int] = {
    "blackwell-96": 94,
    "a100-80": 80,
    "a100-40": 40,
    "l4-24": 22,
    "t4-16": 15,
    "v100-16": 15,
}


def usable_gib(card: str) -> int:
    """Usable VRAM (GiB) for a card key; ValueError listing valid names if unknown."""
    try:
        return CARDS[card]
    except KeyError:
        raise ValueError(
            f"unknown card {card!r}; valid: {', '.join(sorted(CARDS))}"
        ) from None
