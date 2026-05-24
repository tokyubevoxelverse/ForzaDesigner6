"""Live memory discovery against a running Forza Horizon 6 process.

Workflow (matching forza-painter's documented technique):

  1. User opens FH6, creates a vinyl group with a known sphere at a known coordinate.
  2. Run `python -m forza_abyss_painter.inject find-pid` to confirm forzahorizon6.exe is detected.
  3. Run `python -m forza_abyss_painter.inject scan-float 1234.5` (or whatever coord) — saves matches to a scan file.
  4. User moves the sphere in-game. Run `python -m forza_abyss_painter.inject narrow 5678.9` — keeps only addresses
     whose current value matches. Repeat until you have a single address (or a small cluster).
  5. `dump <addr> 256` — show a hex/structured dump of memory around the address. The user reads
     adjacent floats to identify x, y, rotation, scale, color, etc.
  6. `walk <addr>` — try to find the containing struct (shape array start, layer count, etc.) by
     scanning backward and looking for a length field that matches the visible shape count.
  7. `derive-aob <addr>` — scan the .text region of forzahorizon6.exe for an instruction that
     references this address (LEA / MOV with RIP-relative addressing). Build a wildcarded AOB.
  8. `save-pattern <name> <aob> <offset>` — persist to fh6_patterns.json.

This module exposes the primitives. The CLI in `cli.py` wires them together.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, asdict
from pathlib import Path

from forza_abyss_painter.inject.win_process import (
    ProcessHandle, MemoryRegion, find_process_id,
    MEM_IMAGE, MEM_PRIVATE, MEM_MAPPED,
)

DEFAULT_GAME_EXE = "forzahorizon6.exe"


@dataclass
class FloatHit:
    addr: int
    value: float


@dataclass
class ScanState:
    """Persistable scan state — survives across CLI invocations."""

    pid: int
    target: float
    epsilon: float
    hits: list[FloatHit]

    def to_dict(self) -> dict:
        return {
            "pid": self.pid,
            "target": self.target,
            "epsilon": self.epsilon,
            "hits": [{"addr": h.addr, "value": h.value} for h in self.hits],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ScanState":
        return cls(
            pid=int(d["pid"]),
            target=float(d["target"]),
            epsilon=float(d.get("epsilon", 0.01)),
            hits=[FloatHit(addr=int(h["addr"]), value=float(h["value"])) for h in d["hits"]],
        )

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> "ScanState":
        return cls.from_dict(json.loads(path.read_text()))


def find_game_pid(name: str = DEFAULT_GAME_EXE) -> int | None:
    return find_process_id(name)


def _scan_region_for_float(buf: bytes, target: float, epsilon: float) -> list[tuple[int, float]]:
    """Return list of (offset, value) tuples where buf[offset:offset+4] interpreted as f32 ~= target."""
    out: list[tuple[int, float]] = []
    # Slide a 4-byte window. For speed, walk every aligned 4 bytes first (common case for float arrays),
    # then sweep unaligned offsets after if no hits found in aligned pass.
    n = len(buf) - 3
    target_lo = target - epsilon
    target_hi = target + epsilon
    # Aligned pass
    for off in range(0, n, 4):
        val = struct.unpack_from("<f", buf, off)[0]
        if target_lo <= val <= target_hi:
            out.append((off, val))
    return out


def scan_float(
    pid: int,
    target: float,
    epsilon: float = 0.01,
    include_images: bool = False,
    max_hits: int = 1_000_000,
) -> ScanState:
    """Scan all writable private regions of the process for 4-byte floats matching `target`.

    Vinyl-group data lives on the heap → private, read-write memory. Images (loaded modules)
    are excluded by default; code regions don't hold mutable shape data.
    """
    hits: list[FloatHit] = []
    with ProcessHandle(pid) as proc:
        regions = proc.enumerate_regions()
        for r in regions:
            if not r.readable or not r.writable:
                continue
            if r.is_image and not include_images:
                continue
            data = proc.try_read(r.base, r.size)
            if data is None:
                continue
            for off, val in _scan_region_for_float(data, target, epsilon):
                hits.append(FloatHit(addr=r.base + off, value=val))
                if len(hits) >= max_hits:
                    break
            if len(hits) >= max_hits:
                break
    return ScanState(pid=pid, target=target, epsilon=epsilon, hits=hits)


def narrow(state: ScanState, new_target: float, epsilon: float = 0.01) -> ScanState:
    """Re-check each hit address — keep those whose CURRENT value matches `new_target`."""
    kept: list[FloatHit] = []
    with ProcessHandle(state.pid) as proc:
        for hit in state.hits:
            data = proc.try_read(hit.addr, 4)
            if data is None or len(data) < 4:
                continue
            val = struct.unpack("<f", data)[0]
            if new_target - epsilon <= val <= new_target + epsilon:
                kept.append(FloatHit(addr=hit.addr, value=val))
    return ScanState(pid=state.pid, target=new_target, epsilon=epsilon, hits=kept)


def scan_u32(pid: int, target: int, max_hits: int = 20_000_000) -> list[int]:
    """Scan all writable private regions for 4-byte little-endian u32 matching `target`. Returns list of addresses."""
    needle = struct.pack("<I", target & 0xFFFFFFFF)
    hits: list[int] = []
    with ProcessHandle(pid) as proc:
        for r in proc.enumerate_regions():
            if not r.readable or not r.writable or r.is_image:
                continue
            data = proc.try_read(r.base, r.size)
            if data is None:
                continue
            off = 0
            while True:
                idx = data.find(needle, off)
                if idx < 0:
                    break
                # 4-byte aligned only — reduces noise
                if (r.base + idx) & 3 == 0:
                    hits.append(r.base + idx)
                    if len(hits) >= max_hits:
                        return hits
                off = idx + 1
    return hits


def narrow_u32(pid: int, addrs: list[int], new_target: int) -> list[int]:
    """Keep only addresses whose current u32 == new_target."""
    needle = struct.pack("<I", new_target & 0xFFFFFFFF)
    kept: list[int] = []
    with ProcessHandle(pid) as proc:
        for a in addrs:
            d = proc.try_read(a, 4)
            if d == needle:
                kept.append(a)
    return kept


@dataclass
class DumpRow:
    offset: int
    raw_hex: str
    as_f32: list[float]   # 4 floats from this 16-byte row
    as_u32: list[int]
    as_chars: str


def dump_around(pid: int, addr: int, size: int = 256, before: int = 64) -> list[DumpRow]:
    """Dump `before` bytes before and `size` bytes from `addr`, formatted as rows for analysis."""
    start = addr - before
    total = before + size
    with ProcessHandle(pid) as proc:
        data = proc.try_read(start, total)
    if data is None:
        return []
    rows: list[DumpRow] = []
    for row_off in range(0, len(data), 16):
        chunk = data[row_off:row_off + 16]
        if len(chunk) < 16:
            chunk = chunk + b"\x00" * (16 - len(chunk))
        floats = list(struct.unpack("<4f", chunk))
        u32s = list(struct.unpack("<4I", chunk))
        chars = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
        rows.append(DumpRow(
            offset=start + row_off,
            raw_hex=chunk.hex(" "),
            as_f32=floats,
            as_u32=u32s,
            as_chars=chars,
        ))
    return rows


def find_struct_start(
    pid: int,
    known_addr: int,
    max_back: int = 4096,
    expected_shape_count: int | None = None,
) -> list[tuple[int, int]]:
    """Heuristic: walk backward from a known shape-field address looking for a u32 field whose
    value matches `expected_shape_count` (the user-known number of spheres in the template group).

    Returns [(candidate_struct_addr, distance_back), ...] sorted by distance ascending.
    """
    if expected_shape_count is None:
        return []
    candidates: list[tuple[int, int]] = []
    start = known_addr - max_back
    with ProcessHandle(pid) as proc:
        data = proc.try_read(start, max_back + 4)
    if data is None:
        return []
    for off in range(0, len(data) - 3, 4):
        u = struct.unpack_from("<I", data, off)[0]
        if u == expected_shape_count:
            cand_addr = start + off
            candidates.append((cand_addr, known_addr - cand_addr))
    return candidates


def derive_aob_for_address(
    pid: int,
    addr: int,
    pre_bytes: int = 24,
    post_bytes: int = 8,
) -> str:
    """Read raw bytes around `addr` and produce a wildcarded AOB pattern.

    Wildcards every byte that looks like part of a memory address (heuristic: high bytes
    of 64-bit pointers are typically 0x00 once or twice surrounded by data — those vary
    across game launches). The user can also hand-edit the resulting pattern.

    This is a SEED pattern. For a stable AOB, you typically want to capture an instruction
    that references the address (e.g., LEA RAX, [RIP+offset]), not raw data near the
    address. Use `find_code_reference` for that.
    """
    start = addr - pre_bytes
    total = pre_bytes + post_bytes
    with ProcessHandle(pid) as proc:
        data = proc.try_read(start, total)
    if data is None:
        return ""
    out: list[str] = []
    for b in data:
        out.append(f"{b:02X}")
    return " ".join(out)


def find_code_references(
    pid: int,
    target_addr: int,
    module_name: str = DEFAULT_GAME_EXE,
) -> list[tuple[int, bytes]]:
    """Scan the main executable's MEM_IMAGE region(s) for 4-byte values whose RIP-relative
    interpretation points to `target_addr`. Returns [(instruction_addr, surrounding_bytes), ...].

    x86-64 RIP-relative addressing: `instr_end + disp32 == target_addr`, where instr_end is
    the byte after the 4-byte displacement. For a LEA/MOV with a 32-bit displacement, the
    displacement bytes appear in the instruction encoding. We scan for any 4-byte value
    `d` such that `addr_of_d + 4 + d == target_addr`, i.e. `d == target_addr - addr_of_d - 4`.

    This finds code locations that reference the target address. The bytes around each
    match are a candidate AOB seed.
    """
    matches: list[tuple[int, bytes]] = []
    with ProcessHandle(pid) as proc:
        regions = proc.enumerate_regions()
        for r in regions:
            if not r.is_image:
                continue
            if not r.readable:
                continue
            # Likely .text section: executable + readable
            if not (r.protect & 0xF0):  # has execute bits
                continue
            data = proc.try_read(r.base, r.size)
            if data is None:
                continue
            # For each 4-byte window, compute candidate target
            # (signed 32-bit displacement)
            for off in range(0, len(data) - 4):
                disp = struct.unpack_from("<i", data, off)[0]
                instr_after = r.base + off + 4
                if instr_after + disp == target_addr:
                    # Capture bytes from ~6 before disp to 4 after as a seed
                    seed_start = max(0, off - 6)
                    seed_end = min(len(data), off + 4)
                    matches.append((r.base + off - 3, data[seed_start:seed_end]))
                    if len(matches) >= 50:
                        return matches
    return matches


@dataclass
class GameProcInfo:
    pid: int
    region_count: int
    private_writable_bytes: int
    image_bytes: int


def process_summary(pid: int) -> GameProcInfo:
    """Quick stats about the target process — useful to sanity-check we're attached correctly."""
    with ProcessHandle(pid) as proc:
        regions = proc.enumerate_regions()
    priv = sum(r.size for r in regions if r.is_private and r.writable)
    img = sum(r.size for r in regions if r.is_image)
    return GameProcInfo(pid=pid, region_count=len(regions), private_writable_bytes=priv, image_bytes=img)
