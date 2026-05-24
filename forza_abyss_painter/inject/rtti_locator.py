"""RTTI-based LiveryGroup locator.

Discovery approach learned from the publicly available bvzrays/forza-painter-fh6
source (MIT). Adapted for FD6's pipeline. We intentionally do NOT load
community-distributed `forza-codes.dat` / `update-codes.dat` files at runtime —
the single baseline MSVC RTTI class name `.?AVCLiveryGroup@@` is the only
pattern used. If a future game patch renames the class we ship a code update
rather than a data update.

How it works:
  1. Find the MSVC RTTI TypeDescriptor string `.?AVCLiveryGroup@@` inside the
     game module's read-only data section (MEM_IMAGE region).
  2. From that string's address, derive the TypeDescriptor address
     (string starts at offset 0x10 inside the descriptor).
  3. Scan the module for u32 references to the TypeDescriptor's RVA. Each one
     that's preceded by an MSVC RTTI signature byte (0x01) is a
     CompleteObjectLocator (COL).
  4. Scan the module for u64 pointers to each COL address. Those pointers
     live at vtable[-1] (just before the first virtual function). The vtable
     address is therefore `match_address + 8`.
  5. Scan writable private heap for objects whose first qword equals one of
     the discovered vtables — those are guaranteed CLiveryGroup instances by
     type. Same approach bvzrays uses.

This is strictly more reliable than fresh-sphere fingerprint matching because:
  - Identity is by class type, not by content state, so re-injection works.
  - It will find a CLiveryGroup even after the user has edited/painted it.
  - Game-version-independent as long as the class is still named the same.

We still cross-check each candidate with the existing 5/5 layer fingerprint
scoring before accepting it — that filter is what protects FH6 today and we
do not loosen it. RTTI just enumerates the candidate set faster and more
honestly than scanning for `u16 == layer_count` ever did.

If RTTI returns no candidates, callers MUST fall back to the legacy
fresh-sphere u16-scan locator. Treat this module as an opportunistic
fast-path — never a replacement for the strict validator.
"""

from __future__ import annotations

import ctypes
import struct
from ctypes import wintypes

from forza_abyss_painter.inject.win_process import ProcessHandle, MEM_IMAGE


def _is_user_ptr(val: int) -> bool:
    return 0x000001000000 < val < 0x800000000000


def _read_u8(proc: ProcessHandle, addr: int) -> int | None:
    b = proc.try_read(addr, 1)
    return b[0] if b and len(b) == 1 else None


def _read_u32(proc: ProcessHandle, addr: int) -> int | None:
    b = proc.try_read(addr, 4)
    return struct.unpack('<I', b)[0] if b and len(b) == 4 else None


def _read_u64(proc: ProcessHandle, addr: int) -> int | None:
    b = proc.try_read(addr, 8)
    return struct.unpack('<Q', b)[0] if b and len(b) == 8 else None


def _get_main_module_base(pid: int) -> int | None:
    """Return the base address of the process's main module (the .exe itself).

    Callers use this only to compute TypeDescriptor RVAs; if we can't find it,
    RTTI scanning is skipped and we fall back to the legacy locator.
    """
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

    h = k32.OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, pid)
    if not h:
        return None
    try:
        modules = (ctypes.c_void_p * 1024)()
        needed = wintypes.DWORD()
        if not psapi.EnumProcessModulesEx(h, modules, ctypes.sizeof(modules),
                                          ctypes.byref(needed), LIST_MODULES_ALL):
            return None
        # The first entry returned by EnumProcessModulesEx is the executable itself.
        first = modules[0]
        if first is None:
            return None
        return int(first)
    finally:
        k32.CloseHandle(h)


def _find_pattern_in_image(proc: ProcessHandle, pattern: bytes,
                           alignment: int = 1, stop_after: int | None = None,
                           progress_cb=None, progress_label: str = "") -> list[int]:
    """Find every occurrence of `pattern` inside MEM_IMAGE regions.

    Emits progress on every region read so the inject dialog doesn't freeze
    during the multi-minute code-section scans.
    """
    matches: list[int] = []
    image_regions = [r for r in proc.enumerate_regions() if r.is_image and r.readable]
    total = len(image_regions) or 1
    for i, r in enumerate(image_regions):
        data = proc.try_read(r.base, r.size)
        if data is not None:
            start = 0
            while True:
                pos = data.find(pattern, start)
                if pos < 0:
                    break
                addr = r.base + pos
                if alignment <= 1 or addr % alignment == 0:
                    matches.append(addr)
                    if stop_after is not None and len(matches) >= stop_after:
                        if progress_cb:
                            progress_cb(i + 1, total, len(matches))
                        return matches
                start = pos + 1
        if progress_cb:
            progress_cb(i + 1, total, len(matches))
    return matches


def _find_clivery_group_vtables(proc: ProcessHandle, module_base: int,
                                rtti_class_name: bytes,
                                progress_cb=None, status_cb=None) -> list[int]:
    """Return all candidate CLiveryGroup vtable addresses inside the main module.

    Returns an empty list if anything along the chain fails — callers MUST
    treat empty as "fall back to legacy locator", never as "definitive miss".

    Each MEM_IMAGE sub-scan emits progress via `progress_cb(scanned, total, 0)`
    so the inject dialog doesn't sit frozen during this multi-minute phase.
    """
    # Step 1: locate the RTTI class-name string.
    if status_cb:
        status_cb("RTTI phase 1/3 — searching code section for CLiveryGroup class name (progress bar restarts per phase, this is normal)…")
    name_hits = _find_pattern_in_image(
        proc, rtti_class_name, stop_after=1, progress_cb=progress_cb,
    )
    if not name_hits:
        return []
    descriptor_string_addr = name_hits[0]
    type_descriptor_addr = descriptor_string_addr - 0x10
    if type_descriptor_addr < module_base or type_descriptor_addr > 0x7FFFFFFFFFFF:
        return []

    # Step 2: scan MEM_IMAGE for u32 == descriptor RVA.
    if status_cb:
        status_cb("RTTI phase 2/3 — searching for TypeDescriptor references in code section…")
    descriptor_rva = type_descriptor_addr - module_base
    if not 0 <= descriptor_rva <= 0xFFFFFFFF:
        return []
    rva_pattern = struct.pack('<I', descriptor_rva)
    col_addrs: list[int] = []
    for hit in _find_pattern_in_image(proc, rva_pattern, alignment=4,
                                      progress_cb=progress_cb):
        col_addr = hit - 0x0C
        sig = _read_u8(proc, col_addr)
        if sig == 1:
            col_addrs.append(col_addr)
    col_addrs = sorted(set(col_addrs))
    if not col_addrs:
        return []

    # Step 3: for each COL, find pointers to it in MEM_IMAGE.
    if status_cb:
        status_cb(f"RTTI phase 3/3 — searching for vtable references ({len(col_addrs)} COL slot(s))…")
    vtables: list[int] = []
    for col_addr in col_addrs:
        ptr_pattern = struct.pack('<Q', col_addr)
        for hit in _find_pattern_in_image(proc, ptr_pattern, alignment=8,
                                          progress_cb=progress_cb):
            vtables.append(hit + 8)
    return sorted(set(vtables))


def find_livery_group_candidates(
    proc: ProcessHandle, pid: int, profile,
    layer_count: int, progress_cb=None, accept_cb=None, status_cb=None,
) -> list[tuple[int, int]]:
    """Return [(group_addr, table_addr), ...] candidates discovered via RTTI.

    `accept_cb(group_addr, table_addr) -> bool` — optional early-exit hook.
    Called on each candidate as soon as it's found in the heap scan. If it
    returns True, this function returns immediately with that single
    candidate, skipping the rest of memory. Use this to cut scan time
    short once you have a verified winner.

    Each candidate has been filtered to those whose count field matches
    `layer_count` and whose layer-table pointer looks like a valid heap
    pointer. No deep validation is applied here — the caller's loose or
    strict layer check still runs unless accept_cb short-circuits.

    If the RTTI chain can't be resolved (e.g., class name not present,
    module base unknown, no vtables found), returns an empty list. Callers
    MUST then fall back to the legacy locator.
    """
    module_base = _get_main_module_base(pid)
    if module_base is None:
        return []

    if status_cb:
        status_cb("RTTI fallback: scanning game code for CLiveryGroup class signature…")
    # Plumb progress through every MEM_IMAGE phase so the dialog doesn't
    # freeze during the multi-minute code-section scans.
    vtables = _find_clivery_group_vtables(
        proc, module_base, profile.rtti_class_name,
        progress_cb=progress_cb, status_cb=status_cb,
    )
    if not vtables:
        return []
    if status_cb:
        status_cb(
            f"RTTI: located {len(vtables)} CLiveryGroup vtable(s). "
            f"Scanning game heap for live instances (progress bar will restart "
            f"— each RTTI sub-phase runs its own pass, this is normal)…"
        )

    # Heap scan — the slow part. Emit (region_index, total_regions, hits)
    # progress here so the dialog shows real region-by-region advancement
    # instead of sitting frozen on a coarse stage indicator.
    private_regions = [
        r for r in proc.enumerate_regions()
        if r.is_private and r.readable and r.writable
    ]
    private_regions.sort(key=lambda r: r.size, reverse=True)

    candidates: list[tuple[int, int]] = []
    seen_groups: set[int] = set()
    total = len(private_regions)
    for ri, region in enumerate(private_regions):
        data = proc.try_read(region.base, region.size)
        if data is None:
            if progress_cb:
                progress_cb(ri + 1, total, len(candidates))
            continue
        for vtable in vtables:
            pat = struct.pack('<Q', vtable)
            start = 0
            while True:
                pos = data.find(pat, start)
                if pos < 0:
                    break
                start = pos + 8
                group_addr = region.base + pos
                if group_addr in seen_groups:
                    continue
                # Read count + table pointer at the profile-defined offsets.
                count = _read_u32(proc, group_addr + profile.livery_count_offset)
                if count is None:
                    continue
                count_u16 = count & 0xFFFF
                if count_u16 != layer_count:
                    continue
                table_addr = _read_u64(proc, group_addr + profile.layer_table_offset)
                if not table_addr or not _is_user_ptr(table_addr):
                    continue
                candidates.append((group_addr, table_addr))
                seen_groups.add(group_addr)
                # Early-exit hook — if the caller validates this candidate as a
                # confident match, stop scanning the rest of memory.
                if accept_cb is not None and accept_cb(group_addr, table_addr):
                    if progress_cb:
                        progress_cb(ri + 1, total, len(candidates))
                    return [(group_addr, table_addr)]
        if progress_cb:
            progress_cb(ri + 1, total, len(candidates))
    return candidates
