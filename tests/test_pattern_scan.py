from forza_abyss_painter.inject.pattern_scan import find_pattern, find_pattern_all, compile_pattern


def test_find_exact():
    buf = bytes.fromhex("DEADBEEF00112233CAFEBABE")
    assert find_pattern(buf, "DE AD BE EF") == 0
    assert find_pattern(buf, "CA FE BA BE") == 8


def test_find_with_wildcards():
    buf = bytes.fromhex("48 8B 05 12 34 56 78 90 AB CD".replace(" ", ""))
    assert find_pattern(buf, "48 8B 05 ?? ?? ?? ?? 90") == 0


def test_find_returns_minus_one_on_miss():
    assert find_pattern(b"\x00\x01\x02", "DE AD BE EF") == -1


def test_find_all():
    buf = b"\xAA\xBB\xAA\xBB\xCC\xAA\xBB"
    matches = find_pattern_all(buf, "AA BB")
    assert matches == [0, 2, 5]


def test_compile_pattern_mask():
    mask, fixed = compile_pattern("DE ?? BE EF")
    assert mask == bytes([0xFF, 0x00, 0xFF, 0xFF])
    assert fixed == bytes([0xDE, 0x00, 0xBE, 0xEF])
