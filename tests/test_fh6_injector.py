"""Tests for the FH6 injector's pure functions (process-independent parts).

The injector now uses the LiveryGroup + layer_table strategy (no static struct
layout). We test the helpers that don't require an attached process:
  - `_pack_color`: RGBA packing with forced alpha=255
  - `_is_user_ptr`: pointer-range validator
  - layer offset constants (smoke test — these are what the inject loop writes)
"""

import struct

from fd6.inject.fh6_injector import (
    _is_user_ptr,
    _pack_color,
    patterns_are_populated,
    _layer_count_tries,
    _score_layer,
    _loose_validate_layer,
    _LAYER_VALIDATE_START,
    _LAYER_VALIDATE_SIZE,
    _read_layer_addrs,
    _LOCATED_GROUP_CACHE,
    locate_livery_group,
    COUNT_OFF, TABLE_OFF,
    LAYER_POS_OFF, LAYER_SCALE_OFF, LAYER_ROT_OFF,
    LAYER_COLOR_OFF, LAYER_MASK_OFF, LAYER_SHAPE_ID_OFF,
    SCALE_DIVISOR_ELLIPSE, SCALE_DIVISOR_OTHER,
    SHAPE_ID_ELLIPSE, SHAPE_ID_OTHER,
    FH6Injector,
)
from fd6.inject.game_profiles import default_profile


class FakeRegion:
    def __init__(self, base: int, size: int, *, is_image: bool = False,
                 is_private: bool = True) -> None:
        self.base = base
        self.size = size
        self.readable = True
        self.writable = True
        self.is_image = is_image
        self.is_private = is_private


class FakeProc:
    def __init__(self, memory: dict[int, bytes], regions: list[FakeRegion] | None = None) -> None:
        self.memory = dict(memory)
        self.regions = list(regions or [])
        self.reads: list[tuple[int, int]] = []
        self.enumerations = 0

    def try_read(self, addr: int, size: int) -> bytes | None:
        self.reads.append((addr, size))
        for base, data in self.memory.items():
            end = base + len(data)
            if base <= addr and addr + size <= end:
                start = addr - base
                return data[start:start + size]
        return None

    def enumerate_regions(self) -> list[FakeRegion]:
        self.enumerations += 1
        return list(self.regions)


def _valid_layer_bytes() -> bytes:
    buf = bytearray(_LAYER_VALIDATE_SIZE)

    def put(offset: int, data: bytes) -> None:
        start = offset - _LAYER_VALIDATE_START
        buf[start:start + len(data)] = data

    put(LAYER_POS_OFF, struct.pack("<2f", 12.5, -8.25))
    put(LAYER_SCALE_OFF, struct.pack("<2f", 1.0, 1.5))
    put(LAYER_COLOR_OFF, bytes([10, 20, 30, 255]))
    put(LAYER_MASK_OFF, bytes([0]))
    put(LAYER_SHAPE_ID_OFF, bytes([SHAPE_ID_ELLIPSE]))
    return bytes(buf)


def test_patterns_are_populated_always_true():
    """LiveryGroup approach finds shapes dynamically; static patterns no longer required."""
    assert patterns_are_populated() is True


def test_pack_color_forces_alpha_255():
    shape = {"color": [255, 128, 64, 50]}  # alpha 50 should be overridden to 255
    out = _pack_color(shape)
    assert out == bytes([255, 128, 64, 255])


def test_pack_color_three_channel_input():
    shape = {"color": [10, 20, 30]}
    out = _pack_color(shape)
    assert out == bytes([10, 20, 30, 255])


def test_pack_color_missing_defaults_to_white():
    shape = {"x": 1, "y": 2}
    out = _pack_color(shape)
    assert out == bytes([255, 255, 255, 255])


def test_pack_color_clamps_to_byte_range():
    shape = {"color": [300, -5, 256, 0]}
    out = _pack_color(shape)
    # 300 & 0xFF = 44, -5 & 0xFF = 251, 256 & 0xFF = 0
    assert out == bytes([44, 251, 0, 255])


def test_is_user_ptr_validates_range():
    assert _is_user_ptr(0x1A2B3C4D5E6F) is True   # plausible heap addr
    assert _is_user_ptr(0) is False
    assert _is_user_ptr(0x1000) is False           # too low
    assert _is_user_ptr(0xFFFFFFFFFFFFFFFF) is False  # too high


def test_layer_offsets_distinct_and_sane():
    """Smoke test: the offsets the inject loop writes must not overlap unexpectedly."""
    offs = {
        "pos": LAYER_POS_OFF,
        "scale": LAYER_SCALE_OFF,
        "rot": LAYER_ROT_OFF,
        "color": LAYER_COLOR_OFF,
        "mask": LAYER_MASK_OFF,
        "shape": LAYER_SHAPE_ID_OFF,
    }
    # All non-negative, all under reasonable struct size
    for name, off in offs.items():
        assert 0 <= off < 0x200, f"{name} offset {off:#x} out of plausible range"
    # POS (8 bytes) must not overlap SCALE (8 bytes)
    assert LAYER_POS_OFF + 8 <= LAYER_SCALE_OFF
    # SCALE (8 bytes) must not overlap ROT (4 bytes)
    assert LAYER_SCALE_OFF + 8 <= LAYER_ROT_OFF
    # ROT (4 bytes) must not overlap COLOR (4 bytes)
    assert LAYER_ROT_OFF + 4 <= LAYER_COLOR_OFF


def test_group_offsets_distinct():
    # COUNT (u16, 2 bytes) must not overlap TABLE (u64, 8 bytes)
    assert COUNT_OFF + 2 <= TABLE_OFF


def test_scale_divisor_ellipse_is_63():
    """bvzrays-confirmed divisor; matches what works in-game for current FH6 build."""
    assert SCALE_DIVISOR_ELLIPSE == 63.0
    assert SCALE_DIVISOR_OTHER == 127.0


def test_shape_ids():
    assert SHAPE_ID_ELLIPSE == 102
    assert SHAPE_ID_OTHER == 101


def test_pack_color_returns_4_bytes():
    """Every input shape must produce exactly 4 bytes for write to LAYER_COLOR_OFF."""
    for shape in [{}, {"color": []}, {"color": [1, 2]}, {"color": [1, 2, 3]}, {"color": [1, 2, 3, 4]}]:
        assert len(_pack_color(shape)) == 4


def test_layer_count_tries_prefers_real_capacity_for_non_template_counts():
    assert _layer_count_tries(2400) == [3000, 2400]
    assert _layer_count_tries(501) == [1000, 1500, 3000, 501]
    assert _layer_count_tries(500) == [500, 1000, 1500, 3000]


def test_score_layer_reads_validation_window_once():
    layer_addr = 0x100000100000
    proc = FakeProc({layer_addr + _LAYER_VALIDATE_START: _valid_layer_bytes()})

    assert _score_layer(proc, layer_addr) == 5
    assert proc.reads == [(layer_addr + _LAYER_VALIDATE_START, _LAYER_VALIDATE_SIZE)]


def test_loose_validate_layer_reads_validation_window_once():
    layer_addr = 0x100000100000
    proc = FakeProc({layer_addr + _LAYER_VALIDATE_START: _valid_layer_bytes()})

    assert _loose_validate_layer(proc, layer_addr) is True
    assert proc.reads == [(layer_addr + _LAYER_VALIDATE_START, _LAYER_VALIDATE_SIZE)]


def test_read_layer_addrs_uses_bulk_table_read():
    table_addr = 0x100000200000
    addrs = [0x100000300000, 0x100000300080, 0x100000300100]
    proc = FakeProc({table_addr: struct.pack("<3Q", *addrs)})

    assert _read_layer_addrs(proc, table_addr, 3) == addrs
    assert proc.reads == [(table_addr, 24)]


def test_cached_group_skips_memory_scan_when_still_valid(monkeypatch):
    profile = default_profile()
    pid = 12345
    group_addr = 0x100000010000
    table_addr = 0x100000020000
    layer_addrs = [0x100000030000 + i * 0x100 for i in range(20)]
    group = bytearray(TABLE_OFF + 8)
    group[profile.livery_count_offset:profile.livery_count_offset + 2] = struct.pack("<H", 20)
    group[profile.layer_table_offset:profile.layer_table_offset + 8] = struct.pack("<Q", table_addr)
    memory: dict[int, bytes] = {
        group_addr: bytes(group),
        table_addr: struct.pack("<20Q", *layer_addrs),
    }
    for addr in layer_addrs:
        memory[addr + _LAYER_VALIDATE_START] = _valid_layer_bytes()
    proc = FakeProc(memory)
    _LOCATED_GROUP_CACHE.clear()
    _LOCATED_GROUP_CACHE[(pid, profile.key)] = {
        "group_addr": group_addr,
        "table_addr": table_addr,
        "layer_count": 20,
        "layer_addrs": layer_addrs,
    }
    injector = FH6Injector(pid=pid, profile=profile)
    injector._proc = proc

    def fail_scan(*_args, **_kwargs):
        raise AssertionError("memory scan should not run")

    monkeypatch.setattr("fd6.inject.fh6_injector.locate_livery_group", fail_scan)
    monkeypatch.setattr(injector, "_try_rtti_locate", fail_scan)

    handle = injector.find_active_vinyl_group(layer_count=10)

    assert handle.base_addr == group_addr
    assert handle.layer_count == 20
    assert handle.meta["layer_addrs"] == layer_addrs


def test_found_group_is_cached_for_next_lookup(monkeypatch):
    profile = default_profile()
    pid = 12346
    group_addr = 0x100000110000
    table_addr = 0x100000120000
    layer_addrs = [0x100000130000 + i * 0x100 for i in range(20)]
    group = bytearray(TABLE_OFF + 8)
    group[profile.livery_count_offset:profile.livery_count_offset + 2] = struct.pack("<H", 20)
    group[profile.layer_table_offset:profile.layer_table_offset + 8] = struct.pack("<Q", table_addr)
    memory: dict[int, bytes] = {
        group_addr: bytes(group),
        table_addr: struct.pack("<20Q", *layer_addrs),
    }
    for addr in layer_addrs:
        memory[addr + _LAYER_VALIDATE_START] = _valid_layer_bytes()
    proc = FakeProc(memory)
    _LOCATED_GROUP_CACHE.clear()
    injector = FH6Injector(pid=pid, profile=profile)
    injector._proc = proc

    monkeypatch.setattr("fd6.inject.fh6_injector.locate_livery_group", lambda *_args, **_kwargs: (group_addr, table_addr))

    handle = injector.find_active_vinyl_group(layer_count=20)

    assert handle.base_addr == group_addr
    assert _LOCATED_GROUP_CACHE[(pid, profile.key)]["group_addr"] == group_addr
    assert _LOCATED_GROUP_CACHE[(pid, profile.key)]["layer_addrs"] == layer_addrs


def test_cached_lookup_reduces_fake_scan_reads_by_more_than_10x():
    profile = default_profile()
    pid = 12347
    layer_count = 20
    region_size = 0x2000
    noise_regions = [
        FakeRegion(0x100000000000 + i * 0x10000, region_size)
        for i in range(220)
    ]
    target_base = 0x100000E00000
    target_region = FakeRegion(target_base, region_size)
    group_addr = target_base + 0x1000
    table_addr = 0x100001000000
    layer_addrs = [0x100002000000 + i * 0x100 for i in range(layer_count)]

    target = bytearray(b"A" * region_size)
    group_off = group_addr - target_base
    target[group_off + COUNT_OFF:group_off + COUNT_OFF + 2] = struct.pack("<H", layer_count)
    target[group_off + TABLE_OFF:group_off + TABLE_OFF + 8] = struct.pack("<Q", table_addr)

    memory: dict[int, bytes] = {
        region.base: b"A" * region_size
        for region in noise_regions
    }
    memory[target_base] = bytes(target)
    memory[table_addr] = struct.pack(f"<{layer_count}Q", *layer_addrs)
    for addr in layer_addrs:
        memory[addr + _LAYER_VALIDATE_START] = _valid_layer_bytes()

    full_proc = FakeProc(memory, regions=noise_regions + [target_region])
    assert locate_livery_group(full_proc, layer_count) == (group_addr, table_addr)
    full_reads = len(full_proc.reads)

    _LOCATED_GROUP_CACHE.clear()
    _LOCATED_GROUP_CACHE[(pid, profile.key)] = {
        "group_addr": group_addr,
        "table_addr": table_addr,
        "layer_count": layer_count,
        "layer_addrs": layer_addrs,
    }
    cached_proc = FakeProc(memory, regions=noise_regions + [target_region])
    injector = FH6Injector(pid=pid, profile=profile)
    injector._proc = cached_proc

    handle = injector.find_active_vinyl_group(layer_count=layer_count)
    cached_reads = len(cached_proc.reads)

    assert handle.base_addr == group_addr
    assert cached_proc.enumerations == 0
    assert full_reads >= cached_reads * 10


def test_locate_livery_group_scans_large_regions_in_chunks(monkeypatch):
    monkeypatch.setattr("fd6.inject.fh6_injector.SCAN_CHUNK_SIZE", 64)
    layer_count = 20
    region_base = 0x100000400000
    region_size = 0x200
    group_addr = region_base + 0x80
    table_addr = 0x100001400000
    layer_addrs = [0x100002400000 + i * 0x100 for i in range(layer_count)]

    region_data = bytearray(b"A" * region_size)
    group_off = group_addr - region_base
    region_data[group_off + COUNT_OFF:group_off + COUNT_OFF + 2] = struct.pack("<H", layer_count)
    region_data[group_off + TABLE_OFF:group_off + TABLE_OFF + 8] = struct.pack("<Q", table_addr)
    memory: dict[int, bytes] = {
        region_base: bytes(region_data),
        table_addr: struct.pack(f"<{layer_count}Q", *layer_addrs),
    }
    for addr in layer_addrs:
        memory[addr + _LAYER_VALIDATE_START] = _valid_layer_bytes()
    proc = FakeProc(memory, regions=[FakeRegion(region_base, region_size)])

    assert locate_livery_group(proc, layer_count) == (group_addr, table_addr)
    assert (region_base, region_size) not in proc.reads
    assert any(addr == region_base + 0xC0 and size <= 71 for addr, size in proc.reads)
