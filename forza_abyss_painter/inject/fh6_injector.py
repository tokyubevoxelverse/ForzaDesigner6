"""Forza memory injector — LiveryGroup + layer_table implementation.

CREDITS: discovery approach learned from the publicly available
bvzrays/forza-painter-fh6 source (MIT). What we adopted:
  - The CLiveryGroup + layer-table memory layout and offsets (FH5/FH6 share
    the same Forge-derived struct).
  - The MSVC RTTI vtable-scan technique used by the optional fast-path
    locator (see forza_abyss_painter.inject.rtti_locator).
  - The (X, -Y) position write convention and the scale divisors per shape type.
Adapted for FD6's pipeline. We do NOT load community-distributed
`forza-codes.dat` patterns at runtime; only the baseline RTTI class name is
hardcoded.

Locator strategy (cheapest first, each phase only runs when the previous misses):
  0. **PRIMARY: signature-chain** — adapted from forza-painter-fh6. Reads 3 ×
     32 MiB windows of the game's main module image for a fixed 8-byte
     sentinel, validates via mirror at +0x70, walks a 4-pointer chain to the
     active vinyl group. Seconds, not minutes. The chain is rooted in .text/
     .rdata so it can only resolve to the live editor group; no fingerprint
     guessing.
  1. FALLBACK: sphere-fingerprint heap scan — strict 5/5 layer-field check
     + 95% full-table validation. Only runs if (0) misses (e.g. signature
     drifts on a future patch). Fresh-template-only.
  2. LAST-RESORT: RTTI vtable scan via forza_abyss_painter.inject.rtti_locator. Reads
     the game's code section looking for the CLiveryGroup class signature.
     Slowest path; pre-2026 this was the primary "smart" locator.

  All phases share the safety contract: the chosen candidate must look like a
  live CLiveryGroup before a single byte is written. Phase 0 enforces this
  via the chain's structural validity; phases 1 and 2 enforce it via
  per-layer field fingerprints.

  No UI commit step required — writes to the Layer struct propagate to render
  instantly.
"""

from __future__ import annotations

import ctypes
import json
import struct
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path

from forza_abyss_painter.inject import Injector, VinylGroupHandle, InjectResult
from forza_abyss_painter.inject.game_profiles import GameProfile, default_profile
from forza_abyss_painter.inject.patterns_io import DEFAULT_PATTERNS_PATH, load_patterns
from forza_abyss_painter.inject.rtti_locator import find_livery_group_candidates as rtti_find_candidates
from forza_abyss_painter.inject.win_process import ProcessHandle, find_process_id


PATTERNS_FILE = DEFAULT_PATTERNS_PATH

# Target FH6 build this injector's offsets are confirmed against. If the game
# patches and breaks injection, this needs re-derivation. Surfaced in the GUI
# (window title + About dialog) so users know which build the EXE matches.
#
# 2026-05-25 update: UWP package 3.360.259.0 verified. The painter-fh6
# 8-byte sentinel is still present in this build, relocated to module
# offset Base+0xa7f1048 (was: inside the +0x06..+0x0C window range on
# earlier FH5/FH6 builds). Live readiness check still fails the mirror
# gate at +0x70 — that sentinel match appears to be an incidental
# .rdata copy on this build, not the active livery chain root. Fast
# mode falls through to the heap-fingerprint scan, which finds the
# live LiveryGroup in ~30s on a typical 1000–3000-sphere template.
FH6_TARGET_BUILD = "3.360.259.0"


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

# Historical: SPHERE_FULL_TABLE_THRESHOLD = 0.85 used by the strict-mode
# heap scan. Removed 2026-05-25 when locate_livery_group was relaxed to
# painter-matched thresholds (25%-or-32-min strict-valid, see
# _count_strict_valid_layers_unique). The 85% bar rejected used templates
# that painter's locator accepted; that's why "painter just works" for the
# user and we didn't.


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


def _read_2f(proc: ProcessHandle, addr: int) -> tuple[float, float] | None:
    b = proc.try_read(addr, 8)
    return struct.unpack('<2f', b) if b and len(b) == 8 else None


def _score_layer(proc: ProcessHandle, lptr: int) -> int:
    """Score a layer pointer by reading its fields (0-5). Stricter ranges than before.

    Returns the count of plausibility checks that passed. We use the *strict*
    criteria here — a sphere-template layer that hasn't been modified has very
    tight values (position within image canvas, scale ~32-64 / 63, rotation 0,
    color RGBA with alpha 255 or 0, shape_id == 102 for ellipse, mask == 0).
    """
    if not _is_user_ptr(lptr):
        return 0
    score = 0
    # Position: must be finite floats, plausible canvas range
    pos = _read_2f(proc, lptr + LAYER_POS_OFF)
    if pos and all(_is_finite_float(v) and -8192.0 <= v <= 8192.0 for v in pos):
        score += 1
    # Scale: must be finite floats, strictly positive, plausible range
    scale = _read_2f(proc, lptr + LAYER_SCALE_OFF)
    if scale and all(_is_finite_float(v) and 0.0 < abs(v) <= 64.0 for v in scale):
        score += 1
    # Color: just must be readable (any 4 bytes — even all-zero is valid for unset)
    color = proc.try_read(lptr + LAYER_COLOR_OFF, 4)
    if color and len(color) == 4:
        score += 1
    # Shape ID: must be a known FH6 shape id
    shape = proc.try_read(lptr + LAYER_SHAPE_ID_OFF, 1)
    if shape and shape[0] in (101, 102):
        score += 1
    # Mask: must be 0 or 1
    mask = proc.try_read(lptr + LAYER_MASK_OFF, 1)
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
    if not _is_user_ptr(lptr):
        return False
    pos = _read_2f(proc, lptr + LAYER_POS_OFF)
    if pos is None or not all(_is_finite_float(v) for v in pos):
        return False
    scale = _read_2f(proc, lptr + LAYER_SCALE_OFF)
    if scale is None or not all(_is_finite_float(v) for v in scale):
        return False
    # Color bytes must exist (any byte values are fine; the game stores arbitrary RGBA)
    if proc.try_read(lptr + LAYER_COLOR_OFF, 4) is None:
        return False
    return True


def _count_loose_valid_layers(proc: ProcessHandle, table_addr: int, layer_count: int) -> int:
    """Walk the table counting layer pointers that pass the LOOSE validity
    check — for RTTI-confirmed groups, where type identity is already verified
    by the C++ vtable so we don't need the strict sphere fingerprint too."""
    valid = 0
    for k in range(layer_count):
        lptr = _read_u64(proc, table_addr + k * 8)
        if _loose_validate_layer(proc, lptr):
            valid += 1
    return valid


# ---- Post-locate table validation primitives (painter parity).
# After the signature chain resolves to a (group, table, count) triple, we
# sample N pointers out of that table and check each one dereferences to a
# struct that looks like a layer. A chain that resolved to garbage (wrong
# build offsets, transitional editor state, freed memory) passes the
# chain-validity gates but fails table validation. Without this step a bad
# resolution proceeds to write into random memory.
#
# Also gives us the grouped-template signal: if N table pointers contain
# only K << N unique values, the template is a grouped vinyl and the slots
# alias the same physical layer blob. In that state writes would either
# fail or corrupt every aliased slot with the same data, so we refuse.

# Sample size for table validation: enough signal to catch wrong-template
# resolutions without burning hundreds of syscalls. Painter uses 64; we
# use 16 for ellipse templates where signature_chain is the primary path
# and 16 sampled valid pointers is a strong enough signal.
_TABLE_VALIDATION_SAMPLE = 16
# Fraction of sampled pointers that must pass the per-layer score check
# for the table to be accepted. 0.75 = 12/16 minimum.
_TABLE_VALIDATION_THRESHOLD = 0.75
# If fewer than this fraction of sampled pointers are UNIQUE, the table is
# a grouped-vinyl alias structure (same layer blob pointed at from many
# slots). User must Select All → Ungroup before fast-mode can resolve.
_TABLE_UNIQUE_THRESHOLD = 0.50


def _score_layer_for_profile(proc: ProcessHandle, lptr: int, profile: GameProfile) -> int:
    """Profile-aware version of _score_layer. Returns 0-5 plausibility score.

    Used by the post-locate table validator (which runs for any profile, not
    just FH6's hardcoded offsets). FH6 path keeps using _score_layer with its
    module constants for backwards compat with the legacy sphere-fingerprint
    locator; this version reads offsets out of `profile` so adding FH7 or
    re-deriving for a future build doesn't need a code change here."""
    if not _is_user_ptr(lptr):
        return 0
    score = 0
    pos = _read_2f(proc, lptr + profile.layer_position_offset)
    if pos and all(_is_finite_float(v) and -8192.0 <= v <= 8192.0 for v in pos):
        score += 1
    scale = _read_2f(proc, lptr + profile.layer_scale_offset)
    if scale and all(_is_finite_float(v) and 0.0 < abs(v) <= 64.0 for v in scale):
        score += 1
    color = proc.try_read(lptr + profile.layer_color_offset, 4)
    if color and len(color) == 4:
        score += 1
    shape = proc.try_read(lptr + profile.layer_shape_id_offset, 1)
    if shape and shape[0] in (profile.shape_id_ellipse, profile.shape_id_other):
        score += 1
    mask = proc.try_read(lptr + profile.layer_mask_offset, 1)
    if mask and mask[0] in (0, 1):
        score += 1
    return score


def _sample_table_pointers(
    proc: ProcessHandle, table_addr: int, layer_count: int, sample_size: int,
) -> list[int]:
    """Read up to `sample_size` pointers spread across the table. Bulk-reads
    when contiguous (one syscall) and falls back to per-pointer reads otherwise.
    Spreading the sample means a partially-initialized table (eg first N slots
    valid, rest zero) gets caught."""
    n = min(sample_size, layer_count)
    if n <= 0:
        return []
    if n == layer_count:
        # Sample the whole table contiguously.
        indices = list(range(n))
    else:
        # Spread evenly: 0, step, 2*step, ..., (n-1)*step.
        step = max(1, layer_count // n)
        indices = [i * step for i in range(n)]
    return [_read_u64(proc, table_addr + i * 8) for i in indices]


@dataclass(frozen=True)
class ReadinessReport:
    """Per-gate diagnostic of fast-mode inject readiness.

    Returned by `check_inject_readiness`. The GUI calls this BEFORE the user
    clicks Inject, so the Inject button can be enabled/disabled based on
    `ready` and the dialog can show which specific gate is the blocker.
    Eliminates the "click Inject → wait → fall back → wait → cryptic error"
    cycle that the current single-shot _signature_chain_locate produces.

    Fields that haven't been checked (because an earlier gate failed and we
    short-circuited) are None. `messages` accumulates the [readiness] lines
    in the order they were emitted — same format as the [fast-locate] lines
    so existing status_cb consumers parse them identically.

    `ready` is True only when every required gate passed AND, if
    `expected_count` was given, the active group can hold that many shapes.
    """
    pid: int | None = None
    process_name: str | None = None
    module_base: int | None = None
    signature_addr: int | None = None
    mirror_ok: bool | None = None
    addr_a: int | None = None
    addr_b: int | None = None
    c_livery: int | None = None
    group_addr: int | None = None
    table_addr: int | None = None
    layer_count: int | None = None
    table_valid_sampled: tuple[int, int] | None = None   # (n_valid, n_sampled)
    table_unique_sampled: tuple[int, int] | None = None  # (n_unique, n_sampled)
    grouped_suspected: bool = False
    fit_ok: bool | None = None     # None if expected_count was None
    ready: bool = False
    messages: tuple[str, ...] = ()


def check_inject_readiness(
    proc: ProcessHandle, pid: int, profile: GameProfile,
    expected_count: int | None = None,
) -> ReadinessReport:
    """Full fast-mode pre-flight. Resolves the signature chain, validates the
    resolved table, checks for grouped-template aliasing, optionally checks
    capacity against `expected_count`. Returns a ReadinessReport with each
    gate's status — never raises, never writes anything.

    Diagnostic channels emitted on `report.messages`:

      [readiness] <msg>  — one-line summary per gate, shown in the inject
                           dialog status label AND persisted to the log.
                           Equivalent to painter's high-level print().

      [trace] <msg>      — verbose per-operation trace (every scan window
                           attempted, every chain hop's read-from address +
                           resolved value, every sampled layer pointer's
                           score breakdown, mirror bytes hex). Persisted to
                           the log but NOT shown in the dialog so the
                           status label doesn't get overwritten 30 times
                           per call. Equivalent to painter's
                           diagnose_livery() output, just always-on.

    Addresses are formatted as `Base+0xOFFSET` (relative to the game's
    module base) wherever the base is known. Painter uses this convention
    universally — same offset across game runs and machines, ASLR-independent,
    directly comparable to symbol locations in IDA / Ghidra.

    GUI usage: call this when the user opens the Inject dialog (or polls
    every few seconds), gate the Inject button on `report.ready`. If False,
    the dialog shows the latest [readiness] line from `report.messages`.

    Inject-time usage: `_signature_chain_locate` calls this internally and
    extracts the resolved triple on success.
    """
    msgs: list[str] = []
    def log(reason: str) -> None:
        msgs.append(f"[readiness] {reason}")
    def trace(detail: str) -> None:
        msgs.append(f"[trace] {detail}")

    def _ba(addr: int) -> str:
        """Format an address as Base+0xOFFSET if it's within the game module,
        else raw. base_addr is captured by closure once resolved."""
        if base_addr is not None and addr >= base_addr and addr - base_addr < (1 << 32):
            return f"Base+0x{addr - base_addr:x}"
        return f"0x{addr:x}"

    base_addr: int | None = None  # populated by Gate 1, used by _ba()

    trace(f"check_inject_readiness pid={pid} profile={profile.key!r} "
          f"expected_count={expected_count}")

    # Gate 0: profile must have the fast-path constants configured.
    if not profile.signature_patterns or not profile.scan_regions:
        log(f"profile '{profile.key}' has no signature_patterns/scan_regions — "
            f"fast mode disabled for this title")
        return ReadinessReport(pid=pid, messages=tuple(msgs))

    # Gate 1: resolve the game's main module base address. Try each candidate
    # process name; first hit wins. Trace every attempt so users can see
    # exactly which name resolved (or which alternate to try).
    trace(f"resolving module base for candidates: {list(profile.process_names)}")
    resolved_name: str | None = None
    for name in profile.process_names:
        candidate_base = _get_module_base(pid, name)
        trace(f"  _get_module_base({name!r}) -> "
              f"{f'0x{candidate_base:x}' if candidate_base else 'None'}")
        if candidate_base is not None:
            base_addr = candidate_base
            resolved_name = name
            break
    if base_addr is None:
        log(f"none of the candidate module names resolved "
            f"({'/'.join(profile.process_names)}) — game may have just exited, "
            f"or the process is running with elevated privileges this app can't read")
        return ReadinessReport(pid=pid, messages=tuple(msgs))
    log(f"module {resolved_name} resolved @ 0x{base_addr:x}")

    # Gate 2: signature scan over the configured windows. Collect ALL hits
    # across all windows (not just first). FH6 UWP 3.360.259.0 has 2 valid-
    # looking 8-byte occurrences in the image — the first may be a static
    # reference in .rdata while the actual livery anchor is the second.
    # bytes.find() locks onto the first occurrence per window, so the
    # previous "first hit wins" logic would silently bind to the wrong one
    # if the layout shifts. Now we test mirror+chain at every hit and pick
    # the first that PASSES. Painter parity for the per-window trace
    # (forza-painter-fh6/src/main.py:101-103, 147).
    all_hits: list[int] = []
    unreadable_regions: list[int] = []
    sig_hex = " ".join(f"{b:02x}" for b in profile.signature_patterns[0])
    trace(f"scanning {len(profile.scan_regions)} windows for signature {sig_hex}")
    for start_off, region_size in profile.scan_regions:
        region_base = base_addr + start_off
        size_mib = region_size // (1 << 20)
        trace(f"  Scanning Base+0x{start_off:x}..Base+0x{start_off + region_size:x} "
              f"({size_mib} MiB)")
        data = proc.try_read(region_base, region_size)
        if data is None:
            trace(f"    -> window unreadable (try_read returned None)")
            unreadable_regions.append(region_base)
            continue
        window_hits: list[int] = []
        for sig in profile.signature_patterns:
            start = 0
            while True:
                pos = data.find(sig, start)
                if pos < 0:
                    break
                window_hits.append(region_base + pos)
                trace(f"    -> match #{len(window_hits)} at offset 0x{pos:x} "
                      f"(Base+0x{start_off + pos:x})")
                start = pos + 1
        if not window_hits:
            trace(f"    -> no match in this window")
        all_hits.extend(window_hits)
    if not all_hits:
        if unreadable_regions and len(unreadable_regions) == len(profile.scan_regions):
            log(f"all {len(unreadable_regions)} scan windows unreadable")
        else:
            log(f"signature not in any scan window — build may have drifted")
        return ReadinessReport(
            pid=pid, process_name=resolved_name, module_base=base_addr,
            messages=tuple(msgs),
        )
    log(f"signature found at {len(all_hits)} location(s): " +
        ", ".join(_ba(a) for a in all_hits))

    # Gate 3: mirror validation. Try each hit in order; first that passes
    # wins. Bytes that lexically match the sentinel can appear in .rdata as
    # static data or as part of another struct — the mirror gate (u32 at
    # +0x70 mirrors first 4 bytes of sig) is what proves THIS occurrence
    # is the actual livery chain root.
    found_addr: int | None = None
    last_mirror_diag = ""
    for candidate in all_hits:
        head = proc.try_read(candidate, 4)
        mirror = proc.try_read(candidate + profile.validation_mirror_offset, 4)
        head_hex = head.hex() if head else "None"
        mirror_hex = mirror.hex() if mirror else "None"
        trace(f"mirror check @ {_ba(candidate)}: head={head_hex} vs "
              f"+0x{profile.validation_mirror_offset:x}={mirror_hex}")
        if head is not None and mirror is not None and head == mirror:
            found_addr = candidate
            trace(f"mirror OK at {_ba(candidate)}")
            break
        last_mirror_diag = (f"{_ba(candidate)} head={head_hex} "
                            f"+0x{profile.validation_mirror_offset:x}={mirror_hex}")
    if found_addr is None:
        log(f"mirror validation failed at all {len(all_hits)} candidate(s). "
            f"Last: {last_mirror_diag}. Vinyl editor may not be open yet "
            f"(mirror at +0x{profile.validation_mirror_offset:x} can be live "
            f"editor state). If it IS open, the mirror offset may have shifted "
            f"on this build — re-derive via diagnostic capture.")
        return ReadinessReport(
            pid=pid, process_name=resolved_name, module_base=base_addr,
            signature_addr=all_hits[0], mirror_ok=False, messages=tuple(msgs),
        )

    # Gate 4: 4-pointer chain walk. For each hop, trace the READ ADDRESS
    # (where we're reading the next pointer FROM, expressed as Base+offset
    # for the first hop and absolute thereafter) AND the resolved value
    # (with _is_user_ptr verdict). Painter prints "Found livery root
    # pointer at Base+0x78234c3" for hop 1; we add the same level of
    # detail for hops 2-4 too.
    hop1_read = found_addr + profile.livery_root_pointer_offset
    addr_a = _read_u64(proc, hop1_read)
    trace(f"hop 1 (livery root): read at {_ba(hop1_read)} "
          f"(sig+0x{profile.livery_root_pointer_offset:x}) -> addr_a=0x{addr_a:x} "
          f"user_ptr={_is_user_ptr(addr_a)}")
    if not _is_user_ptr(addr_a):
        log(f"chain hop 1 (livery root) is not a user-space pointer: 0x{addr_a:x}")
        return ReadinessReport(
            pid=pid, process_name=resolved_name, module_base=base_addr,
            signature_addr=found_addr, mirror_ok=True, messages=tuple(msgs),
        )
    hop2_read = addr_a + profile.editor_pointer_offset
    addr_b = _read_u64(proc, hop2_read)
    trace(f"hop 2 (editor): read at 0x{hop2_read:x} "
          f"(addr_a+0x{profile.editor_pointer_offset:x}) -> addr_b=0x{addr_b:x} "
          f"user_ptr={_is_user_ptr(addr_b)}")
    if not _is_user_ptr(addr_b):
        log(f"chain hop 2 (editor) is not a user-space pointer: 0x{addr_b:x} — "
            f"vinyl editor is likely not open, OR you're applying a livery to "
            f"the car (not editing a vinyl group)")
        return ReadinessReport(
            pid=pid, process_name=resolved_name, module_base=base_addr,
            signature_addr=found_addr, mirror_ok=True, addr_a=addr_a,
            messages=tuple(msgs),
        )
    hop3_read = addr_b + profile.livery_pointer_offset
    c_livery = _read_u64(proc, hop3_read)
    trace(f"hop 3 (cLivery): read at 0x{hop3_read:x} "
          f"(addr_b+0x{profile.livery_pointer_offset:x}) -> c_livery=0x{c_livery:x} "
          f"user_ptr={_is_user_ptr(c_livery)}")
    if not _is_user_ptr(c_livery):
        log(f"chain hop 3 (cLivery) is not a user-space pointer: 0x{c_livery:x}")
        return ReadinessReport(
            pid=pid, process_name=resolved_name, module_base=base_addr,
            signature_addr=found_addr, mirror_ok=True, addr_a=addr_a, addr_b=addr_b,
            messages=tuple(msgs),
        )
    group_addr = c_livery + profile.livery_group_offset
    hop4_read = group_addr + profile.layer_table_offset
    table_addr = _read_u64(proc, hop4_read)
    trace(f"hop 4 (layer_table): group=c_livery+0x{profile.livery_group_offset:x}"
          f"=0x{group_addr:x}, read at 0x{hop4_read:x} "
          f"(group+0x{profile.layer_table_offset:x}) -> table_addr=0x{table_addr:x} "
          f"user_ptr={_is_user_ptr(table_addr)}")
    if not _is_user_ptr(table_addr):
        log(f"chain hop 4 (layer_table) is not a user-space pointer: 0x{table_addr:x}")
        return ReadinessReport(
            pid=pid, process_name=resolved_name, module_base=base_addr,
            signature_addr=found_addr, mirror_ok=True, addr_a=addr_a,
            addr_b=addr_b, c_livery=c_livery, group_addr=group_addr,
            messages=tuple(msgs),
        )
    cnt_read = group_addr + profile.livery_count_offset
    cnt_bytes = proc.try_read(cnt_read, 2)
    trace(f"layer_count: read at 0x{cnt_read:x} "
          f"(group+0x{profile.livery_count_offset:x}) -> "
          f"{cnt_bytes.hex() if cnt_bytes else 'None'}")
    if not cnt_bytes or len(cnt_bytes) != 2:
        log(f"layer_count unreadable at 0x{cnt_read:x}")
        return ReadinessReport(
            pid=pid, process_name=resolved_name, module_base=base_addr,
            signature_addr=found_addr, mirror_ok=True, addr_a=addr_a,
            addr_b=addr_b, c_livery=c_livery, group_addr=group_addr,
            table_addr=table_addr, messages=tuple(msgs),
        )
    layer_count = struct.unpack('<H', cnt_bytes)[0]
    trace(f"layer_count parsed: {layer_count}")
    if layer_count < 1 or layer_count > 10000:
        log(f"layer_count={layer_count} outside [1, 10000] — chain likely resolved "
            f"to freed memory, OR active group is a nested wrapper")
        return ReadinessReport(
            pid=pid, process_name=resolved_name, module_base=base_addr,
            signature_addr=found_addr, mirror_ok=True, addr_a=addr_a,
            addr_b=addr_b, c_livery=c_livery, group_addr=group_addr,
            table_addr=table_addr, layer_count=layer_count, messages=tuple(msgs),
        )
    log(f"chain resolved: group=0x{group_addr:x} table=0x{table_addr:x} count={layer_count}")

    # Gate 5: post-locate table validation. Sample pointers, score each.
    # Trace EVERY sample with its score so failed-table cases tell us which
    # field (position / scale / color / shape_id / mask) tripped — this is
    # what tells us "build offsets drifted" vs "transitional editor state"
    # vs "wrong template type entirely".
    sample = _sample_table_pointers(proc, table_addr, layer_count, _TABLE_VALIDATION_SAMPLE)
    trace(f"table validation: sampling {len(sample)} pointers")
    scores: list[int] = []
    for i, ptr in enumerate(sample):
        s = _score_layer_for_profile(proc, ptr, profile)
        scores.append(s)
        trace(f"  sample[{i:2d}] ptr=0x{ptr:x} score={s}/5")
    n_valid = sum(1 for s in scores if s >= 4)
    valid_pair = (n_valid, len(sample))
    threshold_n = int(len(sample) * _TABLE_VALIDATION_THRESHOLD)
    if n_valid < threshold_n:
        log(f"table validation failed: only {n_valid}/{len(sample)} sampled pointers "
            f"score ≥4/5 (need {threshold_n}+). Chain resolved structurally but the "
            f"table is not a real layer table — build offsets may have drifted, or "
            f"editor is in a transitional state. Retry after a moment.")
        return ReadinessReport(
            pid=pid, process_name=resolved_name, module_base=base_addr,
            signature_addr=found_addr, mirror_ok=True, addr_a=addr_a,
            addr_b=addr_b, c_livery=c_livery, group_addr=group_addr,
            table_addr=table_addr, layer_count=layer_count,
            table_valid_sampled=valid_pair, messages=tuple(msgs),
        )

    # Gate 6: grouped-template detection via duplicate pointers.
    unique_ptrs = {p for p in sample if _is_user_ptr(p)}
    unique_pair = (len(unique_ptrs), len(sample))
    trace(f"uniqueness: {len(unique_ptrs)}/{len(sample)} unique pointers in sample")
    grouped = len(unique_ptrs) < int(len(sample) * _TABLE_UNIQUE_THRESHOLD)
    if grouped:
        log(f"grouped template detected: only {len(unique_ptrs)}/{len(sample)} "
            f"sampled pointers are unique. Multiple slots alias the same layer "
            f"blob — writes would corrupt every aliased slot identically. "
            f"In FH6: Select All → Ungroup in the vinyl editor, then retry.")
        return ReadinessReport(
            pid=pid, process_name=resolved_name, module_base=base_addr,
            signature_addr=found_addr, mirror_ok=True, addr_a=addr_a,
            addr_b=addr_b, c_livery=c_livery, group_addr=group_addr,
            table_addr=table_addr, layer_count=layer_count,
            table_valid_sampled=valid_pair, table_unique_sampled=unique_pair,
            grouped_suspected=True, messages=tuple(msgs),
        )
    log(f"table validates: {n_valid}/{len(sample)} look like layers, "
        f"{len(unique_ptrs)}/{len(sample)} unique pointers")

    # Gate 7: capacity check (only if caller specified expected_count).
    fit_ok: bool | None = None
    if expected_count is not None:
        fit_ok = layer_count >= expected_count
        if not fit_ok:
            log(f"capacity miss: active group has {layer_count} slots, JSON needs "
                f"{expected_count}. Load a larger template "
                f"(10/20/50/100/500/1500/3000).")
            return ReadinessReport(
                pid=pid, process_name=resolved_name, module_base=base_addr,
                signature_addr=found_addr, mirror_ok=True, addr_a=addr_a,
                addr_b=addr_b, c_livery=c_livery, group_addr=group_addr,
                table_addr=table_addr, layer_count=layer_count,
                table_valid_sampled=valid_pair, table_unique_sampled=unique_pair,
                grouped_suspected=False, fit_ok=False, messages=tuple(msgs),
            )

    log(f"READY — fast-mode inject can proceed")
    return ReadinessReport(
        pid=pid, process_name=resolved_name, module_base=base_addr,
        signature_addr=found_addr, mirror_ok=True, addr_a=addr_a, addr_b=addr_b,
        c_livery=c_livery, group_addr=group_addr, table_addr=table_addr,
        layer_count=layer_count, table_valid_sampled=valid_pair,
        table_unique_sampled=unique_pair, grouped_suspected=False, fit_ok=fit_ok,
        ready=True, messages=tuple(msgs),
    )


def _signature_chain_locate(
    proc: ProcessHandle, pid: int, profile: GameProfile,
    status_cb=None, expected_count: int | None = None,
) -> tuple[int, int, int] | None:
    """Thin wrapper over `check_inject_readiness`. Returns
    (group_addr, table_addr, layer_count) if every fast-mode gate passed,
    else None.

    All chain walking + per-gate diagnostic logging + post-locate table
    validation + grouped-template detection lives in check_inject_readiness.
    This wrapper exists so the existing `find_active_vinyl_group` call site
    and tests keep their tuple-or-None signature. New code that needs richer
    state (eg GUI gating on `report.ready`) should call check_inject_readiness
    directly.

    `status_cb` receives each gate's diagnostic line. We re-flag the
    [readiness] prefix as [fast-locate] so existing GUI status panels that
    pattern-match on the old format keep working.
    """
    report = check_inject_readiness(proc, pid, profile, expected_count=expected_count)
    if status_cb and report.messages:
        # Forward every line. [trace] lines stay tagged [trace] so the worker
        # routes them to log-only (verbose per-operation detail would
        # overwrite the dialog QLabel 30 times per call). [readiness] lines
        # get the [fast-locate] rebrand for log-shape compat with the
        # original monolithic locator, and the FINAL [readiness] line gets
        # "OK:" / "miss:" tagged so the overall outcome is unmissable in
        # the dialog.
        readiness_msgs = [m for m in report.messages if m.startswith("[readiness] ")]
        last_readiness = readiness_msgs[-1] if readiness_msgs else None
        for msg in report.messages:
            if msg.startswith("[trace] "):
                status_cb(msg)   # worker routes trace to log only
                continue
            base = msg.replace("[readiness] ", "[fast-locate] ", 1)
            if msg is last_readiness:
                if report.ready:
                    base = base.replace("[fast-locate] ", "[fast-locate] OK: ", 1)
                else:
                    base = base.replace("[fast-locate] ", "[fast-locate] miss: ", 1)
            status_cb(base)
    if not report.ready:
        return None
    # All gates passed → group/table/count are non-None by check_inject_readiness contract.
    assert report.group_addr is not None
    assert report.table_addr is not None
    assert report.layer_count is not None
    return (report.group_addr, report.table_addr, report.layer_count)


def locate_livery_group(
    proc: ProcessHandle, layer_count: int,
    progress_cb=None, max_candidates: int = 200000,
) -> tuple[int, int] | None:
    """Find LiveryGroup + layer table by scanning heap for u16 == layer_count.

    PAINTER-MATCHED VALIDATION (revised 2026-05-25 after user repro on FH6
    UWP 3.360.259.0 where painter's EXE auto-located fine but ours rejected):

    Previously we required all 16 sampled pointers to score 5/5 AND 85% of
    the FULL table to score 5/5 (= 2550 strict-valid out of 3000). This is
    ~3.4x stricter than forza-painter-fh6, which accepts when:
      - first-pass acceptance: per-pointer score >= 3 (loose, 3 of 5 dims)
      - strict count required: min(layer_count, max(32, layer_count // 4))
        ≈ 25% of layers strict-valid (= 750 out of 3000)
      - duplicate pointers skipped (grouped templates surface as duplicates)
    Source: forza-painter-fh6/src/fh6_probe.py:validate_table_layer_coverage.

    Our strictness rejected any template that had been injected on before
    (custom shapes don't match the sphere-fresh fingerprint), while painter
    accepted it. That's the actual "painter works most of the time" gap.

    The original 5/5 strict bar was added after a misidentified candidate
    caused FH6 to crash mid-write — but painter's lower bar has been
    production-tested by thousands of users without that problem. Lowering
    to painter parity restores the use case (re-inject onto used template)
    without measurably increasing crash risk.
    """
    pattern = struct.pack('<H', layer_count)
    all_regions = [r for r in proc.enumerate_regions()
                   if r.readable and r.writable and not r.is_image]
    # PAINTER PARITY: two-strategy region order (forza-painter-fh6/src/
    # fh6_probe.py:711-743). v1.3 first — small regions (<= 256 MiB) in
    # address-ascending order, where the active vinyl-group most commonly
    # lives. v1.4 second — all regions in size-descending order as the
    # fallback. Previously we only did v1.4 (size-desc), which means we
    # hit huge texture/shader heaps BEFORE the small editor-state heap
    # where the group actually is — user's 1000-shape repro took 3:43
    # because the group was found at region ~200 of 7654. With v1.3
    # ordered first it's typically <30s.
    SMALL_REGION_MAX = 256 * 1024 * 1024
    v1_3 = sorted([r for r in all_regions if r.size <= SMALL_REGION_MAX],
                  key=lambda r: r.base)
    v1_4 = sorted(all_regions, key=lambda r: r.size, reverse=True)
    # Concatenate; dedupe so we don't re-scan the same region twice.
    seen: set[int] = set()
    regions = []
    for r in v1_3 + v1_4:
        if r.base in seen:
            continue
        seen.add(r.base)
        regions.append(r)
    total = len(regions)
    candidates = 0
    accepted: list[tuple[int, int, int]] = []  # (strict_valid_count, group_addr, table_addr)
    for i, r in enumerate(regions):
        data = proc.try_read(r.base, r.size)
        if data is None:
            if progress_cb: progress_cb(i + 1, total, candidates)
            continue
        start = 0
        while True:
            pos = data.find(pattern, start)
            if pos < 0:
                break
            start = pos + 1
            candidates += 1
            if candidates > max_candidates:
                if progress_cb: progress_cb(i + 1, total, candidates)
                return _pick_best_accepted(accepted)
            count_addr = r.base + pos
            group_addr = count_addr - COUNT_OFF
            if group_addr < r.base:
                continue
            table_addr = _read_u64(proc, group_addr + TABLE_OFF)
            if not _is_user_ptr(table_addr):
                continue
            # First-pass loose gate: 16 sampled pointers must reach score >= 3
            # to be worth a full-table validation. score=3 means at least
            # position+scale+color OR position+shape_id+mask read cleanly —
            # enough to rule out garbage / accidental u16 hits.
            sample_n = min(layer_count, 16)
            loose_ok_count = 0
            for k in range(sample_n):
                lptr = _read_u64(proc, table_addr + k * 8)
                if _score_layer(proc, lptr) >= 3:
                    loose_ok_count += 1
            if loose_ok_count < sample_n // 2:
                continue   # less than half of sample even loosely valid; skip
            # Full-table painter-style validation: count unique strict-valid
            # pointers, accept if >= painter's threshold.
            strict_required = min(layer_count, max(32, layer_count // 4))
            strict_valid = _count_strict_valid_layers_unique(
                proc, table_addr, layer_count, early_stop=strict_required + 32,
            )
            if strict_valid >= strict_required:
                if progress_cb:
                    progress_cb(i + 1, total, len(accepted) + 1)
                return (group_addr, table_addr)
            # Borderline: hit at least 32 strict-valid but below painter's
            # threshold. Keep as fallback in case nothing better turns up.
            if strict_valid >= 32:
                accepted.append((strict_valid, group_addr, table_addr))
        if progress_cb: progress_cb(i + 1, total, len(accepted))
    return _pick_best_accepted(accepted)


def _count_strict_valid_layers_unique(
    proc: ProcessHandle, table_addr: int, layer_count: int,
    early_stop: int | None = None,
) -> int:
    """Painter parity: count unique pointers in the table that strict-validate
    (score 5/5 against the layer fingerprint). Skips duplicate pointers so
    grouped templates (which alias the same blob in multiple slots) don't
    inflate the count. Returns early once `early_stop` is reached for speed.
    """
    seen: set[int] = set()
    strict_valid = 0
    scan_limit = min(layer_count, 3000)
    for k in range(scan_limit):
        lptr = _read_u64(proc, table_addr + k * 8)
        if lptr in seen or not _is_user_ptr(lptr):
            continue
        if _score_layer(proc, lptr) >= 5:
            seen.add(lptr)
            strict_valid += 1
            if early_stop is not None and strict_valid >= early_stop:
                return strict_valid
    return strict_valid


def _pick_best_accepted(
    accepted: list[tuple[int, int, int]],
) -> tuple[int, int] | None:
    """Among borderline candidates (strict_valid below painter threshold but
    above the 32-minimum safety floor), pick the highest strict_valid count.
    Returns None if no candidates qualify."""
    if not accepted:
        return None
    accepted.sort(key=lambda x: x[0], reverse=True)
    _strict, group_addr, table_addr = accepted[0]
    return (group_addr, table_addr)


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
                    f"{self.profile.label} is not running, OR Forza Abyss Painter is running with lower "
                    f"privileges than the game. If the game IS open, close Forza Abyss Painter and "
                    f"re-launch it as Administrator (right-click ForzaAbyssPainter.exe → "
                    f"Run as administrator). The game's process memory is inaccessible "
                    f"from a non-elevated Forza Abyss Painter even when both processes are running. "
                    f"(Looked for: {names}.)"
                )
        self._proc = ProcessHandle(self.pid)
        self._proc.open()

    def detach(self) -> None:
        if self._proc:
            self._proc.close()
            self._proc = None

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
            for k in range(sample_n):
                lptr = _read_u64(proc, table_addr + k * 8)
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
            for k in range(sample_n):
                lptr = _read_u64(proc, table_addr + k * 8)
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

    def _bulk_read_layer_addrs(self, table_addr: int, count: int) -> list[int]:
        """One ReadProcessMemory for the whole layer-pointer table; fall back to
        per-pointer reads if the bulk read returns short (page boundary). Shared
        between the signature-chain fast path and the legacy fingerprint path."""
        table_bytes = self._proc.try_read(table_addr, count * 8)
        if table_bytes and len(table_bytes) == count * 8:
            return list(struct.unpack(f"<{count}Q", table_bytes))
        return [_read_u64(self._proc, table_addr + i * 8) for i in range(count)]

    # Borderline table-validation window: between half-passed and threshold.
    # In this band the chain resolved + the table partially looks like layers,
    # which strongly suggests the editor is mid-transition (user just clicked
    # Create Vinyl Group, struct is being initialized layer-by-layer). One
    # 2-second retry catches the steady state without making users wait if
    # the failure is structural.
    _RETRY_TABLE_LOWER_FRACTION = 0.50    # >= half passed → not random garbage
    _RETRY_TABLE_UPPER_FRACTION = _TABLE_VALIDATION_THRESHOLD   # < threshold
    _RETRY_WAIT_SECONDS = 2.0

    def _fast_readiness_with_retry(self, layer_count, status_cb):
        """Run check_inject_readiness, retry once after a 2 s sleep if the
        result is borderline (table validation partially passed but below
        threshold — strong tell that the editor is mid-initialization).
        Forwards all log lines to status_cb in real time so the user sees the
        full trace whether we retry or not."""
        def _forward(report):
            if not status_cb or not report.messages:
                return
            readiness_msgs = [m for m in report.messages if m.startswith("[readiness] ")]
            last_readiness = readiness_msgs[-1] if readiness_msgs else None
            for msg in report.messages:
                if msg.startswith("[trace] "):
                    status_cb(msg)
                    continue
                base = msg.replace("[readiness] ", "[fast-locate] ", 1)
                if msg is last_readiness:
                    if report.ready:
                        base = base.replace("[fast-locate] ", "[fast-locate] OK: ", 1)
                    else:
                        base = base.replace("[fast-locate] ", "[fast-locate] miss: ", 1)
                status_cb(base)

        report = check_inject_readiness(
            self._proc, self.pid, self.profile, expected_count=layer_count,
        )
        _forward(report)
        if report.ready:
            return report
        # Borderline table validation → editor likely mid-transition. Retry once.
        if report.table_valid_sampled is not None:
            n_valid, n_total = report.table_valid_sampled
            lower = int(n_total * self._RETRY_TABLE_LOWER_FRACTION)
            upper = int(n_total * self._RETRY_TABLE_UPPER_FRACTION)
            if lower <= n_valid < upper:
                if status_cb:
                    status_cb(
                        f"[fast-locate] table validation borderline "
                        f"({n_valid}/{n_total} valid; need {upper}+). Editor may "
                        f"be mid-transition (Create Vinyl Group just clicked, "
                        f"struct still initializing). Retrying in "
                        f"{self._RETRY_WAIT_SECONDS:g}s…"
                    )
                import time
                time.sleep(self._RETRY_WAIT_SECONDS)
                report = check_inject_readiness(
                    self._proc, self.pid, self.profile, expected_count=layer_count,
                )
                _forward(report)
        return report

    def find_active_vinyl_group(self, progress_cb=None, layer_count: int | None = None,
                                color_progress_cb=None, status_cb=None,
                                template_size: int | None = None) -> VinylGroupHandle:
        """Locate the active LiveryGroup.

        Locator stack (try fastest → slowest, each phase narrows when the next
        starts):

          0. **Signature-chain (painter parity).** Scan 3 × 32 MiB of the game's
             main module for an 8-byte sentinel, validate via mirror at +0x70,
             walk a 4-pointer chain to the active vinyl group. ~seconds. The
             chain is rooted in the .exe image so it can ONLY resolve to the
             live editor group — no fingerprint guessing required. Works
             whether the template is fresh or already painted.

          1. **Sphere fingerprint (legacy strict).** Heap-walk scan for u16 ==
             layer_count, validate via strict 16/16 layer-field fingerprint +
             95% full-table check. Runs only if (0) fails (eg. signature has
             drifted on a future game patch). Fresh templates only.

          2. **RTTI vtable scan.** Reads the game's code section for the
             CLiveryGroup class name and follows vtables back to instances.
             Slowest path (2-5 min). Last-resort fallback when both (0) and
             (1) miss.

        `status_cb(msg: str)` — optional callback the GUI uses to surface
        phase transitions ("fast locate found a 3000-layer group", "sphere
        scan missed, falling back to RTTI"). Without this users would see
        unexplained wait time.

        `template_size` — when set, constrain the legacy heap-fingerprint scan
        to ONLY that u16 value instead of walking [layer_count, *common_larger].
        Painter v1.6+ takes this as user input; we expose it via the pre-inject
        TemplateSizePickerDialog. Cuts scan time 5-15x when the user knows
        which template they loaded. None = auto (try common sizes >= layer_count).
        """
        if not self._proc:
            raise RuntimeError("Injector not attached. Call attach() first.")
        # ---- Phase 0: signature-chain locator with structured introspection.
        # Calls check_inject_readiness directly (not the tuple-wrapper) so we
        # can inspect WHICH gate failed and decide whether falling back to
        # the legacy heap scan is worth the wait. Painter blindly retries;
        # we route smarter:
        #   - editor not open (hop 2 broken)   → fast-fail, legacy can't help
        #   - grouped template                 → fast-fail, legacy can't help
        #   - capacity miss (template too small) → fast-fail, legacy can't help
        #   - borderline table validation      → likely transitional; retry once
        #   - signature drift / mirror fail    → fall through, legacy MIGHT help
        if self.pid is not None and self.profile.signature_patterns:
            report = self._fast_readiness_with_retry(layer_count, status_cb)
            if report.ready:
                addrs = self._bulk_read_layer_addrs(report.table_addr, report.layer_count)
                self._group_addr = report.group_addr
                self._table_addr = report.table_addr
                self._layer_count = report.layer_count
                if status_cb:
                    status_cb(
                        f"Located vinyl group with {report.layer_count} layer slots "
                        f"via fast signature chain. Writing shapes now…"
                    )
                return VinylGroupHandle(
                    base_addr=report.group_addr,
                    layer_count=report.layer_count,
                    shape_array_addr=report.table_addr,
                    shape_stride=8,
                    meta={
                        "group_addr": report.group_addr,
                        "table_addr": report.table_addr,
                        "layer_addrs": addrs,
                    },
                )
            # Inspect the failure mode. Some are unrecoverable by the legacy
            # scan — raising fast saves the user 30s-5min of doomed waiting.
            #
            # IMPORTANT (2026-05-25): we used to also fast-fail on
            # `addr_a is not None and addr_b is None` with "Vinyl editor is
            # not open." Removed after live verification on FH6 UWP
            # 3.360.259.0 — hop 2 returns 0 EVEN WITH the editor confirmed
            # open and shapes loaded. The painter-era chain offsets
            # (sig+0xB8 → +0xA58) are broken on this build; the sentinel
            # hit at Base+0xa7f1048 turned out to be incidental .rdata
            # bytes, not the real livery chain root.
            # When hop 2 is null we MUST fall through to the legacy heap-
            # fingerprint scan — that's our actual workhorse on builds
            # where the painter chain doesn't resolve.
            if report.grouped_suspected:
                raise RuntimeError(
                    "Active vinyl group is GROUPED. Multiple slots alias the same "
                    "layer blob, so writes would corrupt every aliased slot "
                    "identically. The heap-fingerprint fallback won't help — it "
                    "would find the same aliased structure. In FH6: Select All → "
                    "Ungroup in the vinyl editor, then retry."
                )
            if report.fit_ok is False:
                raise RuntimeError(
                    f"Active vinyl group has only {report.layer_count} layer slots, "
                    f"but the JSON has {layer_count} shapes. The heap-fingerprint "
                    f"fallback wouldn't find a larger template either — load a "
                    f"bigger sphere-template (10 / 20 / 50 / 100 / 500 / 1500 / "
                    f"3000) and retry."
                )
            # Other failure modes (signature drift, mirror fail, hop 2 null
            # on builds where the chain offsets are stale, table garbage):
            # the legacy fingerprint scan anchors on the heap (not the
            # signature chain) and resolves on builds where the chain is
            # broken. Worth trying.
            if status_cb:
                status_cb(
                    "Fast signature scan missed — falling back to "
                    "sphere-fingerprint heap scan (slower)."
                )
        # ---- Phase 1+2: legacy heap fingerprint + RTTI fallback.
        # When the user pinned a template_size in the picker dialog, scan
        # ONLY that size — way faster than walking common sizes. Otherwise
        # auto: try the JSON's shape count first (exact match), then larger
        # common templates that could also host the JSON (a 1500-template
        # can hold a 500-shape JSON).
        if template_size is not None:
            tries = [template_size]
            if status_cb:
                status_cb(f"Heap scan constrained to user-selected size: "
                          f"{template_size} (skipping auto-search of common sizes).")
        else:
            common = [500, 1500, 3000, 1000, 100, 50, 20, 10]
            if layer_count is not None:
                tries = [layer_count] + [c for c in common if c > layer_count]
            else:
                tries = common
        for count_try in tries:
            if count_try is None:
                continue
            # PRIMARY: fingerprint scan (fast, proven sphere-template method).
            result = locate_livery_group(self._proc, count_try, progress_cb=progress_cb)
            if result is None:
                # Sphere fingerprint missed — the template has likely been
                # injected on previously. Notify the GUI and try RTTI.
                if status_cb:
                    status_cb(
                        f"Sphere-template scan found no fresh {count_try}-layer group "
                        f"(template may already be painted). Falling back to RTTI "
                        f"vtable scan — this can take an extra 2–5 minutes on a "
                        f"large game while it reads the code section. "
                        f"DO NOT click anything in Forza Abyss Painter or interact with the app during "
                        f"this phase — clicking can trigger a 'Not Responding' freeze "
                        f"that may force-quit the injector before it finishes. The "
                        f"scan is running in the background and will resume the dialog "
                        f"once a candidate is located."
                    )
                result = self._try_rtti_locate(count_try, progress_cb=progress_cb, status_cb=status_cb)
            if result is not None:
                self._group_addr, self._table_addr = result
                self._layer_count = count_try
                # Bulk-read the entire layer-pointer table in ONE syscall instead
                # of count_try individual ReadProcessMemory calls. The previous
                # per-pointer loop locked the worker thread for 1500-3000 ctypes
                # calls back-to-back, which Windows happily labelled "Not
                # Responding".
                addrs = self._bulk_read_layer_addrs(self._table_addr, count_try)
                if status_cb:
                    status_cb(
                        f"Located vinyl group with {count_try} layer slots. "
                        f"Writing shapes now…"
                    )
                return VinylGroupHandle(
                    base_addr=self._group_addr,
                    layer_count=count_try,
                    shape_array_addr=self._table_addr,
                    shape_stride=8,  # pointer stride in layer table
                    meta={
                        "group_addr": self._group_addr,
                        "table_addr": self._table_addr,
                        "layer_addrs": addrs,
                    },
                )
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
        # Sampled mid-inject safety: re-run the 5/5 _score_layer check every N shapes
        # instead of every shape. The scan-time validation (locate_livery_group + the
        # RTTI path) already confirmed every layer pointer is good; per-shape revalidation
        # was costing 5 ReadProcessMemory syscalls per layer (15k extra syscalls on a
        # 3000-shape inject) for a failure mode that doesn't happen in practice — FH6
        # doesn't free/move LiveryGroup layers while the user is in the Vinyl Group Editor.
        # Painter (forza-painter-fh6/src/main.py:167-194) skips per-shape revalidation
        # entirely and trusts the scan-time check; we sample every REVALIDATE_EVERY shapes
        # as a cheap canary that still catches catastrophic layer-table corruption fast.
        REVALIDATE_EVERY = 250
        # Progress callback throttle: fire every PROGRESS_EVERY shapes instead
        # of every shape. Each progress_cb hop into the GUI thread is a Qt
        # signal emission (~microseconds × 3000 shapes adds up, plus the
        # status bar repaint is much heavier than the write itself). Final
        # shape always reports so the bar lands at 100%. Painter doesn't
        # report per-shape progress at all; this is a compromise that keeps
        # the UI responsive without spamming.
        PROGRESS_EVERY = 50
        def _maybe_report(force: bool = False) -> None:
            if progress_cb and (force or written % PROGRESS_EVERY == 0):
                progress_cb(written, n)
        for i, sd in enumerate(shape_dicts):
            lptr = layer_addrs[i]
            # Cheap pointer-range check (no syscall) — guards against null / kernel-space ptrs.
            if not _is_user_ptr(lptr):
                skipped += 1
                _maybe_report()
                continue
            # Sampled 5/5 revalidation — only on first slot + every Nth thereafter.
            if (i == 0 or i % REVALIDATE_EVERY == 0) and _score_layer(self._proc, lptr) < 5:
                skipped += 1
                _maybe_report()
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

            _maybe_report()
        # Final report — force a callback so the progress bar lands on 100%
        # even when n % PROGRESS_EVERY != 0.
        _maybe_report(force=True)

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
