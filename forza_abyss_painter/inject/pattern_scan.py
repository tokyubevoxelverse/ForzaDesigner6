"""Array-of-bytes (AOB) pattern scanner.

Patterns are strings like:
    "48 8B 05 ?? ?? ?? ?? 48 8D 0D"
where `??` is a wildcard byte. The scanner finds the first occurrence of the
pattern in a byte buffer.
"""

from __future__ import annotations

import re


def compile_pattern(pattern: str) -> tuple[bytes, bytes]:
    """Compile a textual AOB pattern into (mask, bytes). mask has 0xFF where the byte is fixed, 0x00 where wildcard."""
    tokens = pattern.strip().split()
    mask = bytearray()
    fixed = bytearray()
    for tok in tokens:
        if tok in ("??", "?"):
            mask.append(0)
            fixed.append(0)
        else:
            mask.append(0xFF)
            fixed.append(int(tok, 16))
    return bytes(mask), bytes(fixed)


def find_pattern(buf: bytes, pattern: str) -> int:
    """Return offset of first match in `buf`, or -1 if not found.

    Builds a regex equivalent of the pattern for speed (CPython's re is C-implemented).
    """
    tokens = pattern.strip().split()
    regex_parts = []
    for tok in tokens:
        if tok in ("??", "?"):
            regex_parts.append(b".")
        else:
            regex_parts.append(re.escape(bytes([int(tok, 16)])))
    rx = re.compile(b"".join(regex_parts), re.DOTALL)
    m = rx.search(buf)
    return m.start() if m else -1


def find_pattern_all(buf: bytes, pattern: str) -> list[int]:
    tokens = pattern.strip().split()
    regex_parts = []
    for tok in tokens:
        if tok in ("??", "?"):
            regex_parts.append(b".")
        else:
            regex_parts.append(re.escape(bytes([int(tok, 16)])))
    rx = re.compile(b"".join(regex_parts), re.DOTALL)
    return [m.start() for m in rx.finditer(buf)]
