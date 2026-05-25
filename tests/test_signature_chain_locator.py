"""Tests for the painter-parity signature-chain locator (`_signature_chain_locate`).

This is the fast path: scan a fixed 96 MiB window of the game's main module image
for a sentinel byte sequence, validate via mirror at +0x70, walk a 4-pointer
chain to the active vinyl-group base, read its layer count + table pointer.

The function calls Windows-only ctypes (`_get_module_base` uses psapi); we avoid
that by monkeypatching `_get_module_base` and feeding a `FakeProc` that returns
bytes from a flat dict-backed address space. This way the test runs on macOS/
Linux dev machines and exercises the actual locator logic, not just shape.
"""
from __future__ import annotations

import struct

import pytest

from forza_abyss_painter.inject import fh6_injector
from forza_abyss_painter.inject.game_profiles import FH6, KNOWN_LIVERY_SIGNATURE


class FakeProc:
    """Minimal ProcessHandle stand-in. `regions` maps base_addr → bytes; reads
    fall back to None for unmapped addresses (mirrors ProcessHandle.try_read
    behavior for unreadable pages)."""
    def __init__(self, regions: dict[int, bytes]) -> None:
        self.regions = regions

    def try_read(self, addr: int, size: int) -> bytes | None:
        for base, data in self.regions.items():
            if base <= addr < base + len(data):
                off = addr - base
                end = off + size
                if end > len(data):
                    return None
                return data[off:end]
        return None


def _make_layer_blob(profile=FH6) -> bytes:
    """One synthetic layer struct that passes _score_layer_for_profile at 5/5:
    finite position in canvas range, scale in (0, 64], readable color bytes,
    ellipse shape_id, mask=0."""
    blob = bytearray(0x100)
    struct.pack_into('<2f', blob, profile.layer_position_offset, 100.0, 100.0)
    struct.pack_into('<2f', blob, profile.layer_scale_offset, 1.0, 1.0)
    struct.pack_into('<f', blob, profile.layer_rotation_offset, 0.0)
    blob[profile.layer_color_offset:profile.layer_color_offset + 4] = bytes([200, 100, 50, 255])
    blob[profile.layer_shape_id_offset:profile.layer_shape_id_offset + 1] = bytes([profile.shape_id_ellipse])
    blob[profile.layer_mask_offset:profile.layer_mask_offset + 1] = bytes([0])
    return bytes(blob)


def _build_chain_layout(
    layer_count: int = 1500, grouped: bool = False,
) -> tuple[int, dict[int, bytes], int, int]:
    """Construct an in-memory representation of: module_base + signature window
    containing the sentinel at a known offset, plus a 4-link pointer chain
    pointing at a synthetic vinyl group with `layer_count` slots.

    Each table slot points at a real layer blob (built by _make_layer_blob) so
    the post-locate table validation gate passes. `grouped=True` aliases all
    table slots at a single layer blob — simulates a grouped-vinyl template
    that should trip the duplicate-pointer detector.

    Returns (module_base, regions_dict, expected_group_addr, expected_table_addr).
    Addresses are chosen well above the _is_user_ptr lower bound (0x10000000+).
    """
    MODULE_BASE = 0x140000000              # plausible exe image base
    # Place the signature inside the first scan region (Base + 0x06000000).
    SIG_OFFSET_IN_REGION = 0x12345         # arbitrary offset within the 32 MiB window
    sig_addr = MODULE_BASE + 0x06000000 + SIG_OFFSET_IN_REGION

    # Address layout for the chain. All in distinct, generously-spaced regions.
    ADDR_A     = 0x200000000
    ADDR_B     = 0x300000000
    C_LIVERY   = 0x400000000
    GROUP_ADDR = C_LIVERY + FH6.livery_group_offset   # cLivery + 0x20
    TABLE_ADDR = 0x500000000

    regions: dict[int, bytes] = {}

    # 1. Module signature region: 32 MiB of zero, with the sentinel placed at
    #    SIG_OFFSET_IN_REGION and the mirror u32 at +0x70. The 8 bytes after
    #    the sentinel + the pointer at +0xB8 also live here.
    region_size = FH6.scan_regions[0][1]
    region = bytearray(region_size)
    # Sentinel
    region[SIG_OFFSET_IN_REGION:SIG_OFFSET_IN_REGION + len(KNOWN_LIVERY_SIGNATURE)] = KNOWN_LIVERY_SIGNATURE
    # Mirror u32: first 4 bytes of the sentinel reflected at +0x70
    head_u32 = KNOWN_LIVERY_SIGNATURE[:4]
    region[SIG_OFFSET_IN_REGION + FH6.validation_mirror_offset:
           SIG_OFFSET_IN_REGION + FH6.validation_mirror_offset + 4] = head_u32
    # Chain root pointer at sig + 0xB8 → addr_a
    region[SIG_OFFSET_IN_REGION + FH6.livery_root_pointer_offset:
           SIG_OFFSET_IN_REGION + FH6.livery_root_pointer_offset + 8] = struct.pack('<Q', ADDR_A)
    regions[MODULE_BASE + 0x06000000] = bytes(region)
    # The 2nd and 3rd scan windows must also be readable; empty bytes suffice
    # because `find` returns -1 when the sentinel is absent.
    for off, size in FH6.scan_regions[1:]:
        regions[MODULE_BASE + off] = bytes(size)

    # 2. addr_a region: a small buffer with the editor pointer at +0xA58 → addr_b.
    a_buf = bytearray(FH6.editor_pointer_offset + 8)
    a_buf[FH6.editor_pointer_offset:FH6.editor_pointer_offset + 8] = struct.pack('<Q', ADDR_B)
    regions[ADDR_A] = bytes(a_buf)

    # 3. addr_b region: livery pointer at +0x8 → c_livery.
    b_buf = bytearray(FH6.livery_pointer_offset + 8)
    b_buf[FH6.livery_pointer_offset:FH6.livery_pointer_offset + 8] = struct.pack('<Q', C_LIVERY)
    regions[ADDR_B] = bytes(b_buf)

    # 4. c_livery region: contains the vinyl group base at +0x20. That group
    #    base needs the count (u16) at +0x5A and the table pointer (u64) at +0x78.
    group_size = max(FH6.layer_table_offset, FH6.livery_count_offset) + 16
    cl_size = FH6.livery_group_offset + group_size
    cl_buf = bytearray(cl_size)
    # group inside c_livery starts at +0x20
    group_off = FH6.livery_group_offset
    struct.pack_into('<H', cl_buf, group_off + FH6.livery_count_offset, layer_count)
    struct.pack_into('<Q', cl_buf, group_off + FH6.layer_table_offset, TABLE_ADDR)
    regions[C_LIVERY] = bytes(cl_buf)

    # 5. table region + per-slot layer blobs. Populated so the post-locate
    #    table validation gate passes (each sampled slot dereferences to a
    #    layer struct that scores 5/5). `grouped=True` aliases everything at
    #    LAYER_BASE_ADDR to trip the duplicate-pointer detector.
    LAYER_BASE_ADDR = 0x600000000
    LAYER_STRIDE = 0x200   # 512 B per blob — comfortably larger than 0x100
    layer_blob = _make_layer_blob(FH6)
    table_bytes = bytearray(layer_count * 8)
    for i in range(layer_count):
        ptr = LAYER_BASE_ADDR if grouped else LAYER_BASE_ADDR + i * LAYER_STRIDE
        struct.pack_into('<Q', table_bytes, i * 8, ptr)
        regions[ptr] = layer_blob
    regions[TABLE_ADDR] = bytes(table_bytes)

    return MODULE_BASE, regions, GROUP_ADDR, TABLE_ADDR


def test_signature_chain_locate_happy_path(monkeypatch):
    """End-to-end: signature found in window 1, mirror matches, chain walks
    cleanly, returns (group, table, count) with the constructed values."""
    module_base, regions, expected_group, expected_table = _build_chain_layout(layer_count=1500)
    monkeypatch.setattr(fh6_injector, "_get_module_base", lambda pid, name: module_base)
    proc = FakeProc(regions)
    result = fh6_injector._signature_chain_locate(proc, pid=1234, profile=FH6)
    assert result is not None, "fast locator should resolve the chain on a valid layout"
    group_addr, table_addr, layer_count = result
    assert group_addr == expected_group
    assert table_addr == expected_table
    assert layer_count == 1500


def test_signature_chain_missing_signature_returns_none(monkeypatch):
    """If none of the scan windows contain the sentinel, return None cleanly
    (caller falls back to legacy locator)."""
    MODULE_BASE = 0x140000000
    # All 3 scan windows present but zeroed — no sentinel anywhere.
    regions = {MODULE_BASE + off: bytes(size) for off, size in FH6.scan_regions}
    monkeypatch.setattr(fh6_injector, "_get_module_base", lambda pid, name: MODULE_BASE)
    proc = FakeProc(regions)
    assert fh6_injector._signature_chain_locate(proc, pid=1234, profile=FH6) is None


def test_signature_chain_multi_hit_prefers_first_mirror_pass(monkeypatch):
    """FH6 UWP 3.360.259.0 has TWO 8-byte sentinel occurrences in the module
    image. Painter's original first-hit-wins logic would lock onto whichever
    came lexically first and fail. Our refactored Gate 2/3 must iterate all
    hits and pick the first that mirror-validates.

    Fixture: build the standard chain layout, then plant an EXTRA sentinel
    earlier in the same scan window where the mirror at +0x70 is zeroed.
    The locator must skip that false positive and find the real sentinel
    further into the window."""
    module_base, regions, expected_group, expected_table = _build_chain_layout(layer_count=1500)
    sig_region_base = module_base + 0x06000000
    sig_region = bytearray(regions[sig_region_base])
    # Plant a false-positive sentinel at offset 0x1000 — earlier than the
    # real one at 0x12345 — with the mirror at +0x70 left as zeros.
    DECOY_OFF = 0x1000
    sig_region[DECOY_OFF:DECOY_OFF + len(KNOWN_LIVERY_SIGNATURE)] = KNOWN_LIVERY_SIGNATURE
    # Mirror at decoy+0x70 stays at zeros (region was zero-init'd); head is
    # the sentinel's first 4 bytes (0x12, 0x47, 0x9b, 0x13). Mirror mismatch.
    regions[sig_region_base] = bytes(sig_region)
    monkeypatch.setattr(fh6_injector, "_get_module_base", lambda pid, name: module_base)
    report = fh6_injector.check_inject_readiness(FakeProc(regions), pid=1, profile=FH6)
    assert report.ready is True, (
        "should have skipped the decoy and resolved on the real sentinel"
    )
    # Two hits should have been logged, and the second one (the real sentinel)
    # is what we resolved against.
    log_lines = "\n".join(report.messages)
    assert "2 location(s)" in log_lines or "match #2" in log_lines, (
        "expected multi-hit trace logging"
    )
    assert report.group_addr == expected_group


def test_signature_chain_multi_hit_all_fail_mirror(monkeypatch):
    """All N candidate sentinels fail mirror → report ready=False with a
    diagnostic naming the LAST candidate tried (so the user sees real
    address values for triage)."""
    module_base, regions, _, _ = _build_chain_layout(layer_count=1500)
    sig_region_base = module_base + 0x06000000
    sig_region = bytearray(regions[sig_region_base])
    # Zero the mirror at the real sentinel so it fails too.
    SIG_OFFSET = 0x12345
    sig_region[SIG_OFFSET + FH6.validation_mirror_offset:
               SIG_OFFSET + FH6.validation_mirror_offset + 4] = b'\x00' * 4
    regions[sig_region_base] = bytes(sig_region)
    monkeypatch.setattr(fh6_injector, "_get_module_base", lambda pid, name: module_base)
    report = fh6_injector.check_inject_readiness(FakeProc(regions), pid=1, profile=FH6)
    assert report.ready is False
    assert report.mirror_ok is False
    log_lines = "\n".join(m for m in report.messages if m.startswith("[readiness] "))
    assert "mirror validation failed at all" in log_lines
    # Diagnostic must mention editor-state caveat (matches real-world
    # capture observation: mirror can be live editor state).
    assert "editor" in log_lines.lower()


def test_signature_chain_mirror_mismatch_returns_none(monkeypatch):
    """Sentinel found but the u32 at +0x70 doesn't match the u32 at the
    sentinel head → return None. This protects against incidental byte
    matches in non-livery memory."""
    module_base, regions, _, _ = _build_chain_layout(layer_count=1500)
    # Corrupt the mirror in the sig region.
    sig_region_base = module_base + 0x06000000
    sig_region = bytearray(regions[sig_region_base])
    SIG_OFFSET = 0x12345
    sig_region[SIG_OFFSET + FH6.validation_mirror_offset:
               SIG_OFFSET + FH6.validation_mirror_offset + 4] = b'\x00\x00\x00\x00'
    regions[sig_region_base] = bytes(sig_region)
    monkeypatch.setattr(fh6_injector, "_get_module_base", lambda pid, name: module_base)
    proc = FakeProc(regions)
    assert fh6_injector._signature_chain_locate(proc, pid=1234, profile=FH6) is None


def test_signature_chain_broken_pointer_returns_none(monkeypatch):
    """Sentinel + mirror valid, but the first pointer in the chain is null →
    return None rather than dereferencing kernel-space junk."""
    module_base, regions, _, _ = _build_chain_layout(layer_count=1500)
    sig_region_base = module_base + 0x06000000
    sig_region = bytearray(regions[sig_region_base])
    SIG_OFFSET = 0x12345
    # Zero out the addr_a pointer at sig + 0xB8.
    sig_region[SIG_OFFSET + FH6.livery_root_pointer_offset:
               SIG_OFFSET + FH6.livery_root_pointer_offset + 8] = b'\x00' * 8
    regions[sig_region_base] = bytes(sig_region)
    monkeypatch.setattr(fh6_injector, "_get_module_base", lambda pid, name: module_base)
    proc = FakeProc(regions)
    assert fh6_injector._signature_chain_locate(proc, pid=1234, profile=FH6) is None


def test_signature_chain_bogus_layer_count_returns_none(monkeypatch):
    """Chain resolves but the read u16 layer_count is 0 (or absurdly large) →
    return None. Sanity guard against chains that walked into freed memory
    that happened to dereference but contains garbage at the count offset."""
    module_base, regions, _, _ = _build_chain_layout(layer_count=0)  # bogus
    monkeypatch.setattr(fh6_injector, "_get_module_base", lambda pid, name: module_base)
    proc = FakeProc(regions)
    assert fh6_injector._signature_chain_locate(proc, pid=1234, profile=FH6) is None


def test_signature_chain_disabled_when_no_patterns(monkeypatch):
    """Profile with signature_patterns=() skips the fast path entirely
    (returns None without calling _get_module_base)."""
    import dataclasses
    profile_no_sig = dataclasses.replace(FH6, signature_patterns=())
    calls: list[tuple] = []
    def fake_base(pid, name):
        calls.append((pid, name))
        return 0x140000000
    monkeypatch.setattr(fh6_injector, "_get_module_base", fake_base)
    proc = FakeProc({})
    assert fh6_injector._signature_chain_locate(proc, pid=1234, profile=profile_no_sig) is None
    assert calls == [], "should short-circuit without touching the module table"


def test_signature_chain_no_module_returns_none(monkeypatch):
    """If the process is alive but none of the candidate module names resolve
    (eg unsupported FH build naming), return None cleanly."""
    monkeypatch.setattr(fh6_injector, "_get_module_base", lambda pid, name: None)
    proc = FakeProc({})
    assert fh6_injector._signature_chain_locate(proc, pid=1234, profile=FH6) is None


# ---- Diagnostic logging: every miss must emit a [fast-locate] reason via status_cb.
# Without this the user sees "scan missed, falling back" with no explanation.

def _capture_cb():
    msgs: list[str] = []
    return msgs, msgs.append


def test_diagnostic_emitted_when_signature_missing(monkeypatch):
    MODULE_BASE = 0x140000000
    regions = {MODULE_BASE + off: bytes(size) for off, size in FH6.scan_regions}
    monkeypatch.setattr(fh6_injector, "_get_module_base", lambda pid, name: MODULE_BASE)
    msgs, cb = _capture_cb()
    fh6_injector._signature_chain_locate(FakeProc(regions), pid=1, profile=FH6, status_cb=cb)
    miss = [m for m in msgs if m.startswith("[fast-locate] miss:")]
    assert len(miss) == 1
    assert "signature not in any scan window" in miss[0]


def test_diagnostic_emitted_on_mirror_mismatch(monkeypatch):
    module_base, regions, _, _ = _build_chain_layout()
    sig_region_base = module_base + 0x06000000
    sig_region = bytearray(regions[sig_region_base])
    SIG_OFFSET = 0x12345
    sig_region[SIG_OFFSET + FH6.validation_mirror_offset:
               SIG_OFFSET + FH6.validation_mirror_offset + 4] = b'\x00' * 4
    regions[sig_region_base] = bytes(sig_region)
    monkeypatch.setattr(fh6_injector, "_get_module_base", lambda pid, name: module_base)
    msgs, cb = _capture_cb()
    fh6_injector._signature_chain_locate(FakeProc(regions), pid=1, profile=FH6, status_cb=cb)
    miss = [m for m in msgs if m.startswith("[fast-locate] miss:")]
    assert len(miss) == 1
    assert "mirror validation" in miss[0]


def test_diagnostic_emitted_on_broken_chain(monkeypatch):
    """First chain pointer null → log names which hop tripped."""
    module_base, regions, _, _ = _build_chain_layout()
    sig_region_base = module_base + 0x06000000
    sig_region = bytearray(regions[sig_region_base])
    SIG_OFFSET = 0x12345
    sig_region[SIG_OFFSET + FH6.livery_root_pointer_offset:
               SIG_OFFSET + FH6.livery_root_pointer_offset + 8] = b'\x00' * 8
    regions[sig_region_base] = bytes(sig_region)
    monkeypatch.setattr(fh6_injector, "_get_module_base", lambda pid, name: module_base)
    msgs, cb = _capture_cb()
    fh6_injector._signature_chain_locate(FakeProc(regions), pid=1, profile=FH6, status_cb=cb)
    miss = [m for m in msgs if m.startswith("[fast-locate] miss:")]
    assert len(miss) == 1
    assert "hop 1" in miss[0] and "livery root" in miss[0]


def test_diagnostic_emitted_on_bogus_count(monkeypatch):
    module_base, regions, _, _ = _build_chain_layout(layer_count=0)
    monkeypatch.setattr(fh6_injector, "_get_module_base", lambda pid, name: module_base)
    msgs, cb = _capture_cb()
    fh6_injector._signature_chain_locate(FakeProc(regions), pid=1, profile=FH6, status_cb=cb)
    miss = [m for m in msgs if m.startswith("[fast-locate] miss:")]
    assert len(miss) == 1
    assert "layer_count=0" in miss[0]
    # Hint about the grouped-shapes case must be present (this was the user-
    # reported failure mode that motivated the diagnostic).
    assert "nested wrapper" in miss[0] or "grouped" in miss[0]


def test_diagnostic_emitted_on_success(monkeypatch):
    """Successful locate emits a single OK line with the resolved addresses."""
    module_base, regions, _, _ = _build_chain_layout(layer_count=1500)
    monkeypatch.setattr(fh6_injector, "_get_module_base", lambda pid, name: module_base)
    msgs, cb = _capture_cb()
    result = fh6_injector._signature_chain_locate(FakeProc(regions), pid=1, profile=FH6, status_cb=cb)
    assert result is not None
    ok = [m for m in msgs if m.startswith("[fast-locate] OK:")]
    assert len(ok) == 1
    assert "READY" in ok[0]


# ---- check_inject_readiness / ReadinessReport (post-locate validation + grouped).

def test_readiness_report_happy_path_populates_every_gate(monkeypatch):
    """Ready=True path: every gate field is populated, fit_ok=None (no expected_count)."""
    module_base, regions, expected_group, expected_table = _build_chain_layout(layer_count=1500)
    monkeypatch.setattr(fh6_injector, "_get_module_base", lambda pid, name: module_base)
    report = fh6_injector.check_inject_readiness(FakeProc(regions), pid=42, profile=FH6)
    assert report.ready is True
    assert report.pid == 42
    assert report.module_base == module_base
    assert report.signature_addr is not None
    assert report.mirror_ok is True
    assert report.addr_a and report.addr_b and report.c_livery
    assert report.group_addr == expected_group
    assert report.table_addr == expected_table
    assert report.layer_count == 1500
    # Sample size is _TABLE_VALIDATION_SAMPLE=16.
    assert report.table_valid_sampled == (16, 16)
    assert report.table_unique_sampled == (16, 16)
    assert report.grouped_suspected is False
    assert report.fit_ok is None     # no expected_count given
    # READY message present.
    assert any("READY" in m for m in report.messages)


def test_readiness_report_capacity_fit_ok_when_enough_slots(monkeypatch):
    """expected_count <= layer_count → fit_ok=True, still ready."""
    module_base, regions, _, _ = _build_chain_layout(layer_count=1500)
    monkeypatch.setattr(fh6_injector, "_get_module_base", lambda pid, name: module_base)
    report = fh6_injector.check_inject_readiness(
        FakeProc(regions), pid=1, profile=FH6, expected_count=500,
    )
    assert report.ready is True
    assert report.fit_ok is True


def test_readiness_report_capacity_miss_when_too_few_slots(monkeypatch):
    """expected_count > layer_count → fit_ok=False, ready=False, capacity message."""
    module_base, regions, _, _ = _build_chain_layout(layer_count=500)
    monkeypatch.setattr(fh6_injector, "_get_module_base", lambda pid, name: module_base)
    report = fh6_injector.check_inject_readiness(
        FakeProc(regions), pid=1, profile=FH6, expected_count=3000,
    )
    assert report.ready is False
    assert report.fit_ok is False
    assert report.layer_count == 500   # full chain still resolved
    assert any("capacity miss" in m and "3000" in m for m in report.messages)


def test_readiness_report_rejects_table_with_garbage_pointers(monkeypatch):
    """Chain resolves but table is full of zero pointers → validation gate trips.
    This is the critical safety check that catches "chain resolved but resolved
    to garbage" — the gap that prompted the painter parity work."""
    module_base, regions, _, table_addr = _build_chain_layout(layer_count=1500)
    # Replace the populated table with a zero-filled one (so all pointers fail
    # _is_user_ptr → score 0 → 0/16 valid).
    regions[table_addr] = bytes(1500 * 8)
    monkeypatch.setattr(fh6_injector, "_get_module_base", lambda pid, name: module_base)
    report = fh6_injector.check_inject_readiness(FakeProc(regions), pid=1, profile=FH6)
    assert report.ready is False
    assert report.table_valid_sampled == (0, 16)
    assert any("table validation failed" in m for m in report.messages)


def test_readiness_report_detects_grouped_template(monkeypatch):
    """Grouped vinyl: all table slots alias the same layer blob → unique
    pointers < threshold → grouped_suspected=True, ready=False."""
    module_base, regions, _, _ = _build_chain_layout(layer_count=1500, grouped=True)
    monkeypatch.setattr(fh6_injector, "_get_module_base", lambda pid, name: module_base)
    report = fh6_injector.check_inject_readiness(FakeProc(regions), pid=1, profile=FH6)
    assert report.ready is False
    assert report.grouped_suspected is True
    # All 16 sampled pointers are the same — unique=1, sampled=16.
    assert report.table_unique_sampled == (1, 16)
    # Table validation passed (the single aliased blob IS a valid layer), but
    # the grouped detector rejects regardless.
    assert report.table_valid_sampled == (16, 16)
    assert any("grouped template detected" in m for m in report.messages)
    assert any("Ungroup" in m for m in report.messages)


def test_readiness_report_no_process_short_circuits_cleanly(monkeypatch):
    """Module name doesn't resolve → minimal report, no chain fields populated.
    Trace lines may also appear (resolution attempts per candidate name);
    we only assert ONE [readiness] summary line, which is the actionable
    failure reason."""
    monkeypatch.setattr(fh6_injector, "_get_module_base", lambda pid, name: None)
    report = fh6_injector.check_inject_readiness(FakeProc({}), pid=99, profile=FH6)
    assert report.ready is False
    assert report.pid == 99
    assert report.module_base is None
    assert report.signature_addr is None
    assert report.table_addr is None
    assert report.layer_count is None
    readiness_lines = [m for m in report.messages if m.startswith("[readiness] ")]
    assert len(readiness_lines) == 1
    assert "module names resolved" in readiness_lines[0]
    # Trace lines for each candidate name attempt MUST be present so users
    # can see exactly which names we tried.
    trace_lines = [m for m in report.messages if m.startswith("[trace] ")]
    assert any("_get_module_base" in m for m in trace_lines), (
        "expected per-candidate _get_module_base trace lines"
    )


# ---- Smart fallback routing tests. find_active_vinyl_group inspects the
# ReadinessReport on miss and decides whether falling back to the legacy
# heap scan is worth the wait. Some failure modes (grouped, capacity too
# small) are unrecoverable by the legacy path → must fast-fail. Others
# (chain hop 2 null on stale-offset builds, mirror fail, signature drift)
# DO fall through to the legacy heap-fingerprint scan because it anchors
# differently. The "hop 2 null = editor not open" shortcut was removed in
# the 3.360.259.0 fix — live capture proved hop 2 returns 0 even with editor
# confirmed open on that build, because the chain offsets themselves are
# stale, not because the editor is closed.


class _FakeInj:
    """Lightweight stand-in for FH6Injector exercising just the smart-routing
    logic in find_active_vinyl_group. We can't instantiate the real injector
    in dev tests because attach() calls Windows-only ctypes."""
    def __init__(self, pid, profile, proc):
        self.pid = pid
        self.profile = profile
        self._proc = proc
        self._group_addr = None
        self._table_addr = None
        self._layer_count = None
    # Reuse the real methods directly — they have no Windows-only dependencies
    # once self._proc / self.pid / self.profile are set.
    from forza_abyss_painter.inject.fh6_injector import FH6Injector
    _bulk_read_layer_addrs = FH6Injector._bulk_read_layer_addrs
    _fast_readiness_with_retry = FH6Injector._fast_readiness_with_retry
    _RETRY_TABLE_LOWER_FRACTION = FH6Injector._RETRY_TABLE_LOWER_FRACTION
    _RETRY_TABLE_UPPER_FRACTION = FH6Injector._RETRY_TABLE_UPPER_FRACTION
    _RETRY_WAIT_SECONDS = 0.0   # zero-wait for tests


def test_smart_fallback_retries_once_on_borderline_table(monkeypatch):
    """Borderline table validation (half passed but below threshold) → retry
    once after sleep. Simulates the editor-mid-transition case."""
    # First call: degrade fixture so table validation lands at 9/16 (between
    # 8 = 0.5 floor and 12 = 0.75 threshold). Second call: full fixture.
    module_base, full_regions, _, full_table = _build_chain_layout(layer_count=1500)
    # Degraded variant: zero out 7 of every 16 spaced pointers in the table
    # so the sampler hits exactly 9 valid.
    degraded_regions = dict(full_regions)
    table = bytearray(full_regions[full_table])
    step = max(1, 1500 // 16)
    for i in [0, 1, 2, 3, 4, 5, 6]:    # zero 7 sampled slots → 9 left valid
        struct.pack_into('<Q', table, i * step * 8, 0)
    degraded_regions[full_table] = bytes(table)
    monkeypatch.setattr(fh6_injector, "_get_module_base", lambda pid, name: module_base)

    call_count = {"n": 0}
    def fake_proc_factory():
        # Returns degraded on first call, full on second.
        class _Switching(FakeProc):
            def try_read(self, addr, size):
                if call_count["n"] == 0:
                    self.regions = degraded_regions
                else:
                    self.regions = full_regions
                return super().try_read(addr, size)
        return _Switching(degraded_regions)
    proc = fake_proc_factory()
    # Make the retry "sleep" decrement call_count so the second readiness call
    # sees the recovered fixture.
    def fake_sleep(_):
        call_count["n"] += 1
    monkeypatch.setattr("time.sleep", fake_sleep)

    inj = _FakeInj(pid=1, profile=FH6, proc=proc)
    msgs: list[str] = []
    report = inj._fast_readiness_with_retry(layer_count=None, status_cb=msgs.append)
    assert report.ready is True, "retry should have recovered on the second attempt"
    assert any("borderline" in m for m in msgs)
    assert any("Retrying" in m for m in msgs)


def test_smart_fallback_does_not_retry_when_zero_pointers_valid(monkeypatch):
    """0/16 valid pointers → structural failure, not transient. No retry."""
    module_base, regions, _, table_addr = _build_chain_layout(layer_count=1500)
    regions[table_addr] = bytes(1500 * 8)   # all zero pointers
    monkeypatch.setattr(fh6_injector, "_get_module_base", lambda pid, name: module_base)
    slept = []
    monkeypatch.setattr("time.sleep", lambda s: slept.append(s))
    inj = _FakeInj(pid=1, profile=FH6, proc=FakeProc(regions))
    report = inj._fast_readiness_with_retry(layer_count=None, status_cb=lambda m: None)
    assert report.ready is False
    assert slept == [], "should not retry when zero pointers are valid (not borderline)"


def test_smart_fallback_does_not_retry_when_table_passes_first_time(monkeypatch):
    """Happy path → ready immediately, no retry."""
    module_base, regions, _, _ = _build_chain_layout(layer_count=1500)
    monkeypatch.setattr(fh6_injector, "_get_module_base", lambda pid, name: module_base)
    slept = []
    monkeypatch.setattr("time.sleep", lambda s: slept.append(s))
    inj = _FakeInj(pid=1, profile=FH6, proc=FakeProc(regions))
    report = inj._fast_readiness_with_retry(layer_count=None, status_cb=lambda m: None)
    assert report.ready is True
    assert slept == [], "no retry on happy path"


def test_readiness_emits_full_trace_on_happy_path(monkeypatch):
    """Successful locate must trace: process search, per-window scan attempts
    with Base+offset addressing, mirror bytes, each chain hop's read address
    + resolved value + user_ptr verdict, per-sample table validation scores,
    uniqueness count. This is the painter-parity diagnostic contract."""
    module_base, regions, _, _ = _build_chain_layout(layer_count=1500)
    monkeypatch.setattr(fh6_injector, "_get_module_base", lambda pid, name: module_base)
    report = fh6_injector.check_inject_readiness(FakeProc(regions), pid=42, profile=FH6)
    assert report.ready is True
    trace_lines = [m for m in report.messages if m.startswith("[trace] ")]
    joined = "\n".join(trace_lines)
    # Process / module resolution
    assert "check_inject_readiness pid=42" in joined
    assert "resolving module base for candidates" in joined
    # Per-window scan attempts (3 windows, each logged with Base+0xOFFSET)
    assert "Scanning Base+0x6000000..Base+0x8000000" in joined
    # Match in first window (we constructed the fixture this way). The
    # multi-hit refactor numbers each hit as "match #1 at offset".
    assert "match #1 at offset" in joined
    # Mirror bytes hex — multi-hit refactor now tags with the candidate address.
    assert "mirror check @" in joined and "head=" in joined
    assert "mirror OK" in joined
    # Each chain hop with read address + resolved value + user_ptr verdict
    for hop in ("hop 1", "hop 2", "hop 3", "hop 4"):
        assert f"{hop}" in joined and "user_ptr=True" in joined.split(hop)[1].split("\n")[0], (
            f"hop trace missing or doesn't show user_ptr verdict for {hop}"
        )
    # Per-sample table validation scores (16 samples × score=N/5)
    sample_lines = [m for m in trace_lines if "sample[" in m and "score=" in m]
    assert len(sample_lines) == 16, f"expected 16 sample trace lines, got {len(sample_lines)}"
    # Uniqueness summary
    assert "uniqueness:" in joined
