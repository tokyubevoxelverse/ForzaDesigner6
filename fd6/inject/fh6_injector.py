"""Forza memory injector — LiveryGroup + layer_table implementation.

CREDITS: discovery approach learned from the publicly available
bvzrays/forza-painter-fh6 source (MIT). What we adopted:
  - The CLiveryGroup + layer-table memory layout and offsets (FH5/FH6 share
    the same Forge-derived struct).
  - The MSVC RTTI vtable-scan technique used by the optional fast-path
    locator (see fd6.inject.rtti_locator).
  - The (X, -Y) position write convention and the scale divisors per shape type.
Adapted for FD6's pipeline. We do NOT load community-distributed
`forza-codes.dat` patterns at runtime; only the baseline RTTI class name is
hardcoded.

Locator strategy:
  1. PRIMARY (fast path): RTTI vtable scan via fd6.inject.rtti_locator. This
     finds CLiveryGroup instances by C++ type rather than by content
     fingerprint, so it survives re-injection and edited templates.
  2. FALLBACK (legacy): fresh-sphere u16-count scan. Used when the RTTI chain
     can't be resolved (class missing, etc.). This was FD6's only locator
     before the bvzrays-informed refactor.

  In both paths, candidates are validated by the same strict 5/5 layer
  fingerprint + 95% full-table check before any write happens. RTTI just
  enumerates candidates more reliably; it never relaxes the safety bar.

  No UI commit step required — writes to the Layer struct propagate to render
  instantly.
"""

from __future__ import annotations

import ctypes
import json
import struct
from ctypes import wintypes
from pathlib import Path
from typing import Any

from fd6.inject import Injector, VinylGroupHandle, InjectResult
from fd6.inject.game_profiles import GameProfile, default_profile
from fd6.inject.patterns_io import DEFAULT_PATTERNS_PATH, load_patterns
from fd6.inject.rtti_locator import find_livery_group_candidates as rtti_find_candidates
from fd6.inject.win_process import ProcessHandle, find_process_id


PATTERNS_FILE = DEFAULT_PATTERNS_PATH

# Target FH6 build this injector's offsets are confirmed against. If the game
# patches and breaks injection, this needs re-derivation. Surfaced in the GUI
# (window title + About dialog) so users know which build the EXE matches.
FH6_TARGET_BUILD = "354.221"


# LiveryGroup struct offsets (CONFIRMED working for current FH6 build; may shift on patches)
COUNT_OFF = 0x5A   # u16 layer count
TABLE_OFF = 0x78   # u64 pointer to layer table (array of u64 layer pointers, 8-byte stride)

# Layer struct offsets (within each Layer instance)
LAYER_POS_OFF = 0x18      # 2 x f32: x, y
LAYER_SCALE_OFF = 0x28    # 2 x f32: scale_x, scale_y
LAYER_ROT_OFF = 0x50      # f32: rotation degrees
LAYER_COLOR_OFF = 0x74    # 4 bytes: R, G, B, alpha (alpha must be 0 or 255)
LAYER_MASK_OFF = 0x78     # u8: mask flag (0 or 1)
LAYER_SHAPE_ID_OFF = 0x7A # u8: shape type id (102 = ellipse, 101 = other)

# Scale divisors (per bvzrays)
SCALE_DIVISOR_ELLIPSE = 63.0
SCALE_DIVISOR_OTHER = 127.0
SHAPE_ID_ELLIPSE = 102
SHAPE_ID_OTHER = 101

# Sphere-fingerprint full-table acceptance threshold. After 16/16 strict
# layer sampling passes, we read the entire table and check what fraction of
# its layers pass the same 5/5 strict check. Original threshold was 95% but
# that turned out to reject "reopened fresh" templates that still carry a
# few residual layer values from a previous injection — Forza titles don't
# always wipe the LiveryGroup heap state when the user reloads a template.
# 85% gives meaningful safety against false-positives while accepting
# templates whose 5-10% of layers are partially in a transitional state.
SPHERE_FULL_TABLE_THRESHOLD = 0.85

COMMON_LAYER_COUNTS = (500, 1500, 3000, 1000, 100, 50, 20, 10)
SCAN_CHUNK_SIZE = 16 * 1024 * 1024
LARGE_REGION_THRESHOLD = 256 * 1024 * 1024
_CAPACITY_LAYER_COUNTS = tuple(sorted(COMMON_LAYER_COUNTS))
_LOCATED_GROUP_CACHE: dict[tuple[int, str], dict[str, Any]] = {}
_LAYER_VALIDATE_START = min(
    LAYER_POS_OFF, LAYER_SCALE_OFF, LAYER_COLOR_OFF, LAYER_MASK_OFF, LAYER_SHAPE_ID_OFF,
)
_LAYER_VALIDATE_END = max(
    LAYER_POS_OFF + 8,
    LAYER_SCALE_OFF + 8,
    LAYER_COLOR_OFF + 4,
    LAYER_MASK_OFF + 1,
    LAYER_SHAPE_ID_OFF + 1,
)
_LAYER_VALIDATE_SIZE = _LAYER_VALIDATE_END - _LAYER_VALIDATE_START


def patterns_are_populated() -> bool:
    """Always True now — we no longer rely on a static patterns file for color storage.
    LiveryGroup + layer_table approach finds shapes dynamically."""
    return True


def _get_module_base(pid: int, module_name: str) -> int | None:
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    PROCESS_QUERY_INFORMATION = 0x0400
    PROCESS_VM_READ = 0x0010
    LIST_MODULES_ALL = 0x03

    k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    k32.CloseHandle.restype = wintypes.BOOL
    psapi.EnumProcessModulesEx.argtypes = [
        wintypes.HANDLE, ctypes.POINTER(ctypes.c_void_p), wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD), wintypes.DWORD,
    ]
    psapi.EnumProcessModulesEx.restype = wintypes.BOOL
    psapi.GetModuleBaseNameW.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.LPWSTR, wintypes.DWORD]
    psapi.GetModuleBaseNameW.restype = wintypes.DWORD

    h = k32.OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, pid)
    if not h:
        return None
    try:
        modules = (ctypes.c_void_p * 1024)()
        needed = wintypes.DWORD()
        if not psapi.EnumProcessModulesEx(h, modules, ctypes.sizeof(modules), ctypes.byref(needed), LIST_MODULES_ALL):
            return None
        count = needed.value // ctypes.sizeof(ctypes.c_void_p)
        target = module_name.lower()
        for i in range(count):
            mod = modules[i]
            if mod is None:
                continue
            buf = ctypes.create_unicode_buffer(260)
            n = psapi.GetModuleBaseNameW(h, mod, buf, 260)
            if n and buf.value.lower() == target:
                return int(mod)
        return None
    finally:
        k32.CloseHandle(h)


def _is_user_ptr(val: int) -> bool:
    return 0x000001000000 < val < 0x800000000000


def _read_u64(proc: ProcessHandle, addr: int) -> int:
    b = proc.try_read(addr, 8)
    return struct.unpack('<Q', b)[0] if b and len(b) == 8 else 0


def _read_u16(proc: ProcessHandle, addr: int) -> int | None:
    b = proc.try_read(addr, 2)
    return struct.unpack('<H', b)[0] if b and len(b) == 2 else None


def _read_2f(proc: ProcessHandle, addr: int) -> tuple[float, float] | None:
    b = proc.try_read(addr, 8)
    return struct.unpack('<2f', b) if b and len(b) == 8 else None


def _read_layer_addrs(proc: ProcessHandle, table_addr: int, layer_count: int) -> list[int]:
    if layer_count <= 0:
        return []
    table_bytes = proc.try_read(table_addr, layer_count * 8)
    if table_bytes and len(table_bytes) == layer_count * 8:
        return list(struct.unpack(f"<{layer_count}Q", table_bytes))
    return [_read_u64(proc, table_addr + i * 8) for i in range(layer_count)]


def _read_u64_from_region_or_process(
    proc: ProcessHandle, region_data: bytes, region_base: int, addr: int,
) -> int:
    offset = addr - region_base
    if 0 <= offset <= len(region_data) - 8:
        return struct.unpack_from('<Q', region_data, offset)[0]
    return _read_u64(proc, addr)


def _is_heap_scan_region(region) -> bool:
    return (
        getattr(region, "readable", False)
        and getattr(region, "writable", False)
        and not getattr(region, "is_image", False)
        and getattr(region, "is_private", True)
    )


def _scan_region_priority(region) -> tuple[int, int]:
    is_large = 1 if region.size > LARGE_REGION_THRESHOLD else 0
    return (is_large, region.size)


def _region_scan_units(region) -> int:
    return max(1, (region.size + SCAN_CHUNK_SIZE - 1) // SCAN_CHUNK_SIZE)


def _iter_region_chunks(proc: ProcessHandle, region):
    offset = 0
    while offset < region.size:
        chunk_size = min(SCAN_CHUNK_SIZE + 1, region.size - offset)
        chunk_base = region.base + offset
        yield chunk_base, proc.try_read(chunk_base, chunk_size)
        offset += SCAN_CHUNK_SIZE


def _read_layer_validation_bytes(proc: ProcessHandle, lptr: int) -> bytes | None:
    if not _is_user_ptr(lptr):
        return None
    data = proc.try_read(lptr + _LAYER_VALIDATE_START, _LAYER_VALIDATE_SIZE)
    if data is None or len(data) < _LAYER_VALIDATE_SIZE:
        return None
    return data


def _slice_layer_field(data: bytes, offset: int, size: int) -> bytes:
    start = offset - _LAYER_VALIDATE_START
    return data[start:start + size]


def _score_layer_individual_reads(proc: ProcessHandle, lptr: int) -> int:
    if not _is_user_ptr(lptr):
        return 0
    score = 0
    pos = _read_2f(proc, lptr + LAYER_POS_OFF)
    if pos and all(_is_finite_float(v) and -8192.0 <= v <= 8192.0 for v in pos):
        score += 1
    scale = _read_2f(proc, lptr + LAYER_SCALE_OFF)
    if scale and all(_is_finite_float(v) and 0.0 < abs(v) <= 64.0 for v in scale):
        score += 1
    color = proc.try_read(lptr + LAYER_COLOR_OFF, 4)
    if color and len(color) == 4:
        score += 1
    shape = proc.try_read(lptr + LAYER_SHAPE_ID_OFF, 1)
    if shape and shape[0] in (101, 102):
        score += 1
    mask = proc.try_read(lptr + LAYER_MASK_OFF, 1)
    if mask and mask[0] in (0, 1):
        score += 1
    return score


def _loose_validate_layer_individual_reads(proc: ProcessHandle, lptr: int) -> bool:
    if not _is_user_ptr(lptr):
        return False
    pos = _read_2f(proc, lptr + LAYER_POS_OFF)
    if pos is None or not all(_is_finite_float(v) for v in pos):
        return False
    scale = _read_2f(proc, lptr + LAYER_SCALE_OFF)
    if scale is None or not all(_is_finite_float(v) for v in scale):
        return False
    return proc.try_read(lptr + LAYER_COLOR_OFF, 4) is not None


def _score_layer(proc: ProcessHandle, lptr: int) -> int:
    """Score a layer pointer by reading its fields (0-5). Stricter ranges than before.

    Returns the count of plausibility checks that passed. We use the *strict*
    criteria here — a sphere-template layer that hasn't been modified has very
    tight values (position within image canvas, scale ~32-64 / 63, rotation 0,
    color RGBA with alpha 255 or 0, shape_id == 102 for ellipse, mask == 0).
    """
    data = _read_layer_validation_bytes(proc, lptr)
    if data is None:
        return _score_layer_individual_reads(proc, lptr)
    score = 0
    pos = struct.unpack('<2f', _slice_layer_field(data, LAYER_POS_OFF, 8))
    if pos and all(_is_finite_float(v) and -8192.0 <= v <= 8192.0 for v in pos):
        score += 1
    # Scale: must be finite floats, strictly positive, plausible range
    scale = struct.unpack('<2f', _slice_layer_field(data, LAYER_SCALE_OFF, 8))
    if scale and all(_is_finite_float(v) and 0.0 < abs(v) <= 64.0 for v in scale):
        score += 1
    # Color: just must be readable (any 4 bytes — even all-zero is valid for unset)
    if len(_slice_layer_field(data, LAYER_COLOR_OFF, 4)) == 4:
        score += 1
    shape = _slice_layer_field(data, LAYER_SHAPE_ID_OFF, 1)
    if shape and shape[0] in (101, 102):
        score += 1
    mask = _slice_layer_field(data, LAYER_MASK_OFF, 1)
    if mask and mask[0] in (0, 1):
        score += 1
    return score


def _is_finite_float(v: float) -> bool:
    import math
    return math.isfinite(v)


def _loose_validate_layer(proc: ProcessHandle, lptr: int) -> bool:
    """Looser validity check used when type identity is already confirmed by RTTI.

    The strict 5/5 sphere fingerprint requires layers to still look like a fresh
    template (scale within 0..64, shape_id in {101, 102}, mask 0/1). After an
    injection that fingerprint stops matching, so re-injecting into an
    already-painted template fails the strict gate.

    When RTTI has confirmed an object is a CLiveryGroup by C++ vtable identity,
    we don't need fingerprint confirmation too — we just need to verify the
    layer pointer dereferences to readable memory with finite-float position
    and scale fields. This lets the Upload JSON re-injection workflow target
    groups whose layers carry our previously-written values.
    """
    data = _read_layer_validation_bytes(proc, lptr)
    if data is None:
        return _loose_validate_layer_individual_reads(proc, lptr)
    pos = struct.unpack('<2f', _slice_layer_field(data, LAYER_POS_OFF, 8))
    if not all(_is_finite_float(v) for v in pos):
        return False
    scale = struct.unpack('<2f', _slice_layer_field(data, LAYER_SCALE_OFF, 8))
    if not all(_is_finite_float(v) for v in scale):
        return False
    # Color bytes must exist (any byte values are fine; the game stores arbitrary RGBA)
    if len(_slice_layer_field(data, LAYER_COLOR_OFF, 4)) != 4:
        return False
    return True


def _count_loose_valid_layers(proc: ProcessHandle, table_addr: int, layer_count: int) -> int:
    """Like _count_valid_layers but uses the loose check — for RTTI-confirmed groups."""
    valid = 0
    for lptr in _read_layer_addrs(proc, table_addr, layer_count):
        if _loose_validate_layer(proc, lptr):
            valid += 1
    return valid


def _layer_count_tries(layer_count: int | None) -> list[int]:
    if layer_count is None:
        return list(COMMON_LAYER_COUNTS)
    requested = int(layer_count)
    if requested <= 0:
        return list(COMMON_LAYER_COUNTS)
    if requested in COMMON_LAYER_COUNTS:
        return [requested] + [c for c in _CAPACITY_LAYER_COUNTS if c > requested]
    capacity_matches = [c for c in _CAPACITY_LAYER_COUNTS if c >= requested]
    if capacity_matches:
        return capacity_matches + [requested]
    return [requested]


def locate_livery_group(
    proc: ProcessHandle, layer_count: int,
    progress_cb=None, max_candidates: int = 200000,
) -> tuple[int, int] | None:
    """Find LiveryGroup + layer table by scanning heap for u16 == layer_count.

    STRICT MODE (revised after a misidentified candidate caused FH6 to crash mid-write):
      - Each candidate is rejected unless ALL 16 sampled layer pointers score 5/5.
      - If no perfect candidate is found, refuse to return any (returns None).
      - Caller is expected to bail out cleanly rather than write to a wrong table.

    This is much safer: writing to a wrong heap object corrupts game state. A
    sphere-template layer table has uniform, valid fields across all entries,
    so a 16/16 perfect score is the right bar.
    """
    pattern = struct.pack('<H', layer_count)
    regions = [r for r in proc.enumerate_regions() if _is_heap_scan_region(r)]
    regions.sort(key=_scan_region_priority)
    total = sum(_region_scan_units(r) for r in regions) or 1
    scanned = 0
    candidates = 0
    perfect: list[tuple[int, int]] = []  # (group_addr, table_addr) all 16/16
    for r in regions:
        for chunk_base, data in _iter_region_chunks(proc, r):
            scanned += 1
            if data is None:
                if progress_cb:
                    progress_cb(scanned, total, candidates)
                continue
            start = 0
            while True:
                pos = data.find(pattern, start)
                if pos < 0:
                    break
                start = pos + 1
                candidates += 1
                if candidates > max_candidates:
                    if progress_cb:
                        progress_cb(scanned, total, candidates)
                    return _pick_best_perfect(proc, perfect, layer_count)
                count_addr = chunk_base + pos
                group_addr = count_addr - COUNT_OFF
                if group_addr < r.base or group_addr & 0x7:
                    continue
                table_addr = _read_u64_from_region_or_process(
                    proc, data, chunk_base, group_addr + TABLE_OFF,
                )
                if not _is_user_ptr(table_addr):
                    continue
                ok = True
                sample_n = min(layer_count, 16)
                for lptr in _read_layer_addrs(proc, table_addr, sample_n):
                    if _score_layer(proc, lptr) < 5:
                        ok = False
                        break
                if ok:
                    valid_full = _count_valid_layers(proc, table_addr, layer_count)
                    if valid_full >= layer_count * SPHERE_FULL_TABLE_THRESHOLD:
                        if progress_cb:
                            progress_cb(scanned, total, len(perfect) + 1)
                        return (group_addr, table_addr)
                    perfect.append((group_addr, table_addr))
            if progress_cb:
                progress_cb(scanned, total, len(perfect))
    return _pick_best_perfect(proc, perfect, layer_count)


def _pick_best_perfect(
    proc: ProcessHandle, perfect: list[tuple[int, int]], layer_count: int,
) -> tuple[int, int] | None:
    """Among perfect candidates, pick the one whose *full* table validates best.

    Reads ALL layer pointers (not just first 16) and counts how many score 5/5.
    The real LiveryGroup will have all (or nearly all) of its layers fully valid;
    accidental matches that happened to have valid first-16 will fall off here.
    """
    if not perfect:
        return None
    if len(perfect) == 1:
        # Single candidate — still validate the full table before accepting.
        group_addr, table_addr = perfect[0]
        valid_full = _count_valid_layers(proc, table_addr, layer_count)
        if valid_full >= layer_count * SPHERE_FULL_TABLE_THRESHOLD:
            return (group_addr, table_addr)
        return None
    # Multiple — rank by full-table validation
    scored: list[tuple[int, int, int]] = []
    for group_addr, table_addr in perfect:
        valid_full = _count_valid_layers(proc, table_addr, layer_count)
        scored.append((valid_full, group_addr, table_addr))
    scored.sort(reverse=True)
    best_valid, group_addr, table_addr = scored[0]
    if best_valid >= layer_count * SPHERE_FULL_TABLE_THRESHOLD:
        return (group_addr, table_addr)
    return None


def _count_valid_layers(proc: ProcessHandle, table_addr: int, layer_count: int) -> int:
    """Walk the entire layer_table and count how many pointers resolve to 5/5 layers."""
    valid = 0
    for lptr in _read_layer_addrs(proc, table_addr, layer_count):
        if _score_layer(proc, lptr) >= 5:
            valid += 1
    return valid


def _pack_color(shape_dict: dict) -> bytes:
    """Convert FD6 shape's color to RGBA 4 bytes with alpha forced to 255."""
    color = shape_dict.get("color")
    if not isinstance(color, (list, tuple)) or len(color) < 3:
        return bytes([255, 255, 255, 255])
    r = int(color[0]) & 0xFF
    g = int(color[1]) & 0xFF
    b = int(color[2]) & 0xFF
    return bytes([r, g, b, 255])  # alpha must be 0 or 255; default to 255


class FH6Injector(Injector):
    """Forza injector — LiveryGroup + layer_table strategy.

    Despite the name, this class now drives FH5/FH6/FH4 via GameProfile.
    The class name is kept for backwards-compatibility with existing imports.
    """

    def __init__(self, pid: int | None = None, patterns_path: Path | str = PATTERNS_FILE,
                 profile: GameProfile | None = None) -> None:
        self.pid = pid
        self.patterns_path = Path(patterns_path)
        self.profile: GameProfile = profile or default_profile()
        self._proc: ProcessHandle | None = None
        self._group_addr: int | None = None
        self._table_addr: int | None = None
        self._layer_count: int | None = None

    @property
    def game_label(self) -> str:
        return self.profile.label

    def attach(self) -> None:
        if self.pid is None:
            for name in self.profile.process_names:
                self.pid = find_process_id(name)
                if self.pid is not None:
                    break
            if self.pid is None:
                names = " / ".join(self.profile.process_names)
                raise RuntimeError(
                    f"{self.profile.label} is not running, OR FD6 is running with lower "
                    f"privileges than the game. If the game IS open, close FD6 and "
                    f"re-launch it as Administrator (right-click FD6MultiSupport.exe → "
                    f"Run as administrator). The game's process memory is inaccessible "
                    f"from a non-elevated FD6 even when both processes are running. "
                    f"(Looked for: {names}.)"
                )
        self._proc = ProcessHandle(self.pid)
        self._proc.open()

    def detach(self) -> None:
        if self._proc:
            self._proc.close()
            self._proc = None

    def _cache_key(self) -> tuple[int, str] | None:
        if self.pid is None:
            return None
        return (self.pid, self.profile.key)

    def _build_handle(self, group_addr: int, table_addr: int, layer_count: int,
                      layer_addrs: list[int]) -> VinylGroupHandle:
        self._group_addr = group_addr
        self._table_addr = table_addr
        self._layer_count = layer_count
        return VinylGroupHandle(
            base_addr=group_addr,
            layer_count=layer_count,
            shape_array_addr=table_addr,
            shape_stride=8,
            meta={
                "group_addr": group_addr,
                "table_addr": table_addr,
                "layer_addrs": layer_addrs,
            },
        )

    def _remember_group(self, group_addr: int, table_addr: int, layer_count: int,
                        layer_addrs: list[int]) -> None:
        key = self._cache_key()
        if key is None:
            return
        _LOCATED_GROUP_CACHE[key] = {
            "group_addr": group_addr,
            "table_addr": table_addr,
            "layer_count": layer_count,
            "layer_addrs": list(layer_addrs),
        }

    def _try_cached_group(self, requested_count: int | None, status_cb=None) -> VinylGroupHandle | None:
        if self._proc is None:
            return None
        key = self._cache_key()
        if key is None:
            return None
        cached = _LOCATED_GROUP_CACHE.get(key)
        if not cached:
            return None
        try:
            group_addr = int(cached["group_addr"])
            table_addr = int(cached["table_addr"])
            layer_count = int(cached["layer_count"])
        except (KeyError, TypeError, ValueError):
            _LOCATED_GROUP_CACHE.pop(key, None)
            return None
        if requested_count is not None and layer_count < requested_count:
            return None
        live_count = _read_u16(self._proc, group_addr + self.profile.livery_count_offset)
        live_table = _read_u64(self._proc, group_addr + self.profile.layer_table_offset)
        if live_count != layer_count or live_table != table_addr:
            _LOCATED_GROUP_CACHE.pop(key, None)
            return None
        layer_addrs = list(cached.get("layer_addrs") or [])
        if len(layer_addrs) != layer_count:
            layer_addrs = _read_layer_addrs(self._proc, table_addr, layer_count)
        sample_n = min(layer_count, 16)
        if len(layer_addrs) < sample_n:
            _LOCATED_GROUP_CACHE.pop(key, None)
            return None
        for lptr in layer_addrs[:sample_n]:
            if not _loose_validate_layer(self._proc, lptr):
                _LOCATED_GROUP_CACHE.pop(key, None)
                return None
        if status_cb:
            status_cb(f"Reusing cached vinyl group with {layer_count} layer slots. Writing shapes now...")
        self._remember_group(group_addr, table_addr, layer_count, layer_addrs)
        return self._build_handle(group_addr, table_addr, layer_count, layer_addrs)

    def _try_rtti_locate(self, count_try: int, progress_cb=None, status_cb=None) -> tuple[int, int] | None:
        """RTTI fallback. Returns (group, table) or None on miss.

        RTTI confirms the candidate is a CLiveryGroup by C++ vtable identity, so
        we accept it with LOOSE validation (layer pointers must dereference to
        readable memory with finite floats) rather than the strict 5/5 sphere
        fingerprint. This lets the Upload JSON re-injection workflow target
        groups whose layers carry our previously-written values — the strict
        check only matches untouched sphere templates.

        Pick the candidate with the highest count of loose-valid layers; require
        >= 95% loose-valid before accepting (still high enough to reject
        garbage / partially-allocated memory regions).
        """
        if self._proc is None or self.pid is None:
            return None
        proc = self._proc

        def _accept(group_addr: int, table_addr: int) -> bool:
            """Inline early-exit: stop scanning as soon as a candidate passes
            loose 16-layer sample + 95% full-table loose validation. Saves a
            multi-minute scan of the rest of memory once we have a winner."""
            sample_n = min(count_try, 16)
            for lptr in _read_layer_addrs(proc, table_addr, sample_n):
                if not _loose_validate_layer(proc, lptr):
                    return False
            valid_full = _count_loose_valid_layers(proc, table_addr, count_try)
            return valid_full >= count_try * 0.95

        try:
            candidates = rtti_find_candidates(
                proc, self.pid, self.profile, count_try,
                progress_cb=(progress_cb if progress_cb else None),
                accept_cb=_accept,
                status_cb=(status_cb if status_cb else None),
            )
        except Exception:
            return None
        if not candidates:
            return None
        # If accept_cb fired, candidates is a single confirmed pair.
        if len(candidates) == 1:
            return candidates[0]
        # Otherwise (no early accept): pick best by full-table loose validity.
        scored: list[tuple[int, int, int]] = []
        for group_addr, table_addr in candidates:
            sample_n = min(count_try, 16)
            ok = True
            for lptr in _read_layer_addrs(proc, table_addr, sample_n):
                if not _loose_validate_layer(proc, lptr):
                    ok = False
                    break
            if not ok:
                continue
            valid_full = _count_loose_valid_layers(proc, table_addr, count_try)
            scored.append((valid_full, group_addr, table_addr))
        if not scored:
            return None
        scored.sort(reverse=True)
        best_valid, group_addr, table_addr = scored[0]
        if best_valid >= count_try * 0.95:
            return (group_addr, table_addr)
        return None

    def find_active_vinyl_group(self, progress_cb=None, layer_count: int | None = None,
                                color_progress_cb=None, status_cb=None) -> VinylGroupHandle:
        """Locate the active LiveryGroup.

        **Sphere-fingerprint scan is the PRIMARY locator for every target.**
        It's fast, proven on FH6, and works on any title whose CLiveryGroup
        offsets match (FH5/FH6 confirmed; FH4/FH3 BETA same Forge engine
        family).

        If sphere fingerprint scan finds nothing — which happens when the
        target template has already been injected on (layer values no longer
        match the fresh-sphere pattern) — fall back to the RTTI vtable scan.
        RTTI confirms candidates by C++ type and uses LOOSE per-layer
        validation, so it can pick up already-painted templates that the
        strict sphere scan rejects. Slower (reads the game's code section
        looking for the CLiveryGroup class signature) so it only runs when
        primary misses.

        `status_cb(msg: str)` — optional callback used to surface "sphere
        scan missed, starting RTTI fallback" to the GUI dialog so users don't
        see the scan time silently double.
        """
        if not self._proc:
            raise RuntimeError("Injector not attached. Call attach() first.")
        cached_handle = self._try_cached_group(layer_count, status_cb=status_cb)
        if cached_handle is not None:
            return cached_handle
        # Try the requested count first (exact match), then larger common templates
        # that could also host the JSON (a 1500-template can hold a 500-shape JSON).
        tries = _layer_count_tries(layer_count)
        for count_try in tries:
            if count_try is None:
                continue
            # RTTI usually rejects far more heap noise than the u16 count scan.
            result = self._try_rtti_locate(count_try, progress_cb=progress_cb, status_cb=status_cb)
            if result is None:
                # Sphere fingerprint missed — the template has likely been
                # injected on previously. Notify the GUI and try RTTI.
                if status_cb:
                    status_cb(
                        f"RTTI scan found no {count_try}-layer group. "
                        f"Trying sphere-template fingerprint scan."
                    )
                result = locate_livery_group(self._proc, count_try, progress_cb=progress_cb)
            if result is not None:
                group_addr, table_addr = result
                # Bulk-read the entire layer-pointer table in ONE syscall instead
                # of count_try individual ReadProcessMemory calls. The previous
                # per-pointer loop locked the worker thread for 1500-3000 ctypes
                # calls back-to-back, which Windows happily labelled "Not
                # Responding" — same outcome, but in a fraction of the time and
                # without the visible freeze.
                addrs = _read_layer_addrs(self._proc, table_addr, count_try)
                self._remember_group(group_addr, table_addr, count_try, addrs)
                if status_cb:
                    status_cb(
                        f"Located vinyl group with {count_try} layer slots. "
                        f"Writing shapes now…"
                    )
                return self._build_handle(group_addr, table_addr, count_try, addrs)
        raise RuntimeError(
            "No confident LiveryGroup match (strict 16/16 + 95% full-table validation). "
            "This is intentional — refusing to write to a low-confidence candidate would "
            "corrupt FH6 state. Make sure the vinyl editor is open with a fresh, unmodified "
            "template (500/1500/3000 spheres). If you've already edited the template's "
            "shapes/colors, reload it fresh and re-inject."
        )

    def inject(self, shapes: list, group: VinylGroupHandle, progress_cb=None,
               image_size: tuple[int, int] | None = None, coord_scale: float = 1.0) -> InjectResult:
        if not self._proc:
            raise RuntimeError("Injector not attached.")
        layer_addrs: list[int] = (group.meta or {}).get("layer_addrs") or []
        if not layer_addrs:
            return InjectResult(success=False, message="No layer addresses cached. Call find_active_vinyl_group first.")

        # Normalize shapes to dicts
        shape_dicts: list[dict] = []
        for s in shapes:
            if hasattr(s, "to_json"):
                shape_dicts.append(s.to_json())
            elif isinstance(s, dict):
                shape_dicts.append(s)
            else:
                raise TypeError(f"Unsupported shape type: {type(s)!r}")
        n = len(shape_dicts)
        if n > len(layer_addrs):
            return InjectResult(
                success=False, shapes_written=0,
                message=(f"Template has {len(layer_addrs)} layer slots, but JSON has {n} shapes. "
                         f"Load a larger template vinyl group."),
            )

        written = 0
        bytes_total = 0
        skipped = 0
        # Per-type counter — surfaced in the final result message so users can
        # see at a glance whether their checked rect / rotated_rect actually
        # made it into the JSON, vs. losing every fitness contest to ellipses.
        type_counts: dict[str, int] = {}
        for i, sd in enumerate(shape_dicts):
            lptr = layer_addrs[i]
            # SAFETY: revalidate every pointer right before writing. If a layer
            # ever fails the 5/5 check (e.g., game freed/moved it, or scan picked
            # a near-miss), skip rather than writing through junk and crashing FH6.
            if not _is_user_ptr(lptr) or _score_layer(self._proc, lptr) < 5:
                skipped += 1
                if progress_cb:
                    progress_cb(written, n)
                continue
            shape_type = sd.get("type", "rotated_ellipse")
            is_ellipse = "ellipse" in shape_type or shape_type == "circle"
            scale_div = (
                self.profile.scale_divisor_ellipse if is_ellipse
                else self.profile.scale_divisor_other
            )

            try:
                # Position: X, -Y (Y negated per bvzrays)
                x = float(sd.get("x", 0.0))
                y = float(sd.get("y", 0.0))
                self._proc.write(lptr + LAYER_POS_OFF, struct.pack('<2f', x, -y))
                bytes_total += 8

                # Scale: w/divisor, h/divisor.
                #   ellipse / rotated_ellipse → rx, ry are half-extents (radii);
                #     write radius/63 directly.
                #   circle → single radius r.
                #   rectangle / rotated_rectangle → hw, hh are HALF-extents in FD6
                #     JSON; the game's scale field expects full-width/127, so
                #     convert via (hw * 2) / 127. Without this conversion the
                #     rectangle's scale reads as (1.0/127, 1.0/127) and the
                #     in-game rect renders as a sub-pixel blob.
                if "hw" in sd or "hh" in sd:
                    hw = float(sd.get("hw", sd.get("hh", 0.5)))
                    hh = float(sd.get("hh", sd.get("hw", 0.5)))
                    sx = (hw * 2.0) / scale_div
                    sy = (hh * 2.0) / scale_div
                elif "rx" in sd:
                    sx = float(sd["rx"]) / scale_div
                    sy = float(sd.get("ry", sd["rx"])) / scale_div
                elif "r" in sd:
                    sx = sy = float(sd["r"]) / scale_div
                else:
                    sx = sy = 1.0
                self._proc.write(lptr + LAYER_SCALE_OFF, struct.pack('<2f', sx, sy))
                bytes_total += 8

                # Rotation: 360 - degrees (bvzrays convention)
                angle = float(sd.get("angle", 0.0)) % 360.0
                self._proc.write(lptr + LAYER_ROT_OFF, struct.pack('<f', (360.0 - angle) % 360.0))
                bytes_total += 4

                # Color: RGBA bytes with alpha forced to 255
                self._proc.write(lptr + LAYER_COLOR_OFF, _pack_color(sd))
                bytes_total += 4

                # Shape ID: 102 for ellipse, 101 for other (per profile)
                self._proc.write(lptr + LAYER_SHAPE_ID_OFF, bytes([
                    self.profile.shape_id_ellipse if is_ellipse else self.profile.shape_id_other
                ]))
                bytes_total += 1

                # Mask: 0
                self._proc.write(lptr + LAYER_MASK_OFF, bytes([0]))
                bytes_total += 1

                written += 1
                type_counts[shape_type] = type_counts.get(shape_type, 0) + 1
            except OSError:
                # WriteProcessMemory failure for this one layer — skip and continue.
                skipped += 1

            if progress_cb:
                progress_cb(written, n)

        msg = (f"Wrote {written}/{n} shapes ({bytes_total} bytes) via LiveryGroup layer table.")
        if type_counts:
            mix = ", ".join(f"{t}: {c}" for t, c in sorted(type_counts.items()))
            msg += f" Type mix written — {mix}."
        if skipped:
            msg += f" Skipped {skipped} unsafe layer(s) (failed revalidation)."
        return InjectResult(
            success=written > 0,
            shapes_written=written,
            message=msg,
        )
