import struct

from fd6.inject import rtti_locator
from fd6.inject.game_profiles import default_profile


class FakeRegion:
    def __init__(self, base: int, data: bytes, *, is_image: bool, is_private: bool) -> None:
        self.base = base
        self.size = len(data)
        self.data = data
        self.readable = True
        self.writable = is_private
        self.is_image = is_image
        self.is_private = is_private


class FakeProc:
    def __init__(self, regions: list[FakeRegion]) -> None:
        self.regions = regions
        self.reads: list[tuple[int, int]] = []

    def enumerate_regions(self):
        return self.regions

    def try_read(self, addr: int, size: int) -> bytes | None:
        self.reads.append((addr, size))
        for region in self.regions:
            end = region.base + len(region.data)
            if region.base <= addr and addr + size <= end:
                start = addr - region.base
                return region.data[start:start + size]
        return None


def test_rtti_vtables_are_cached_between_candidate_scans(monkeypatch):
    profile = default_profile()
    pid = 24680
    module_base = 0x10000000
    image = FakeRegion(module_base, b"\0" * 256, is_image=True, is_private=False)
    heap_base = 0x100000000000
    table_addr = 0x100000001000
    vtable = 0x10000040
    group = bytearray(0x90)
    group[0:8] = struct.pack("<Q", vtable)
    group[profile.livery_count_offset:profile.livery_count_offset + 4] = struct.pack("<I", 500)
    group[profile.layer_table_offset:profile.layer_table_offset + 8] = struct.pack("<Q", table_addr)
    heap = FakeRegion(heap_base, bytes(group), is_image=False, is_private=True)
    proc = FakeProc([image, heap])
    calls = {"find": 0}
    rtti_locator._VTABLE_CACHE.clear()

    def fake_module_base(_pid):
        return module_base

    def fake_find(_proc, _module_base, _name, progress_cb=None, status_cb=None):
        calls["find"] += 1
        return [vtable]

    monkeypatch.setattr(rtti_locator, "_get_main_module_base", fake_module_base)
    monkeypatch.setattr(rtti_locator, "_find_clivery_group_vtables", fake_find)

    first = rtti_locator.find_livery_group_candidates(proc, pid, profile, 500)
    second = rtti_locator.find_livery_group_candidates(proc, pid, profile, 500)

    assert first == [(heap_base, table_addr)]
    assert second == [(heap_base, table_addr)]
    assert calls["find"] == 1


def test_iter_vtable_hit_offsets_handles_multiple_patterns():
    first = struct.pack("<Q", 0x1111111122222222)
    second = struct.pack("<Q", 0x3333333344444444)
    data = b"xxxx" + first + b"yyyyyyyy" + second + b"zzzz"

    assert list(rtti_locator._iter_vtable_hit_offsets(data, [first, second])) == [4, 20]
