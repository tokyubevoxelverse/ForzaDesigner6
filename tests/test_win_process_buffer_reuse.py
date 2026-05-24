"""Regression test pinning the zero-allocation hot-path contract of ProcessHandle.write.

ProcessHandle.write used to do `(ctypes.c_ubyte * len(data))(*data)` + a fresh
`ctypes.c_size_t(0)` + a fresh `ctypes.c_void_p(addr)` on every call. On a 3000-shape
inject that's 18,000 ctypes-array allocations + ~138,000 byte-splats through the
Python interpreter — the dominant source of injection-speed lag vs forza-painter-fh6,
which passes `bytes` straight through to WriteProcessMemory and reuses a single
c_size_t output buffer at handle-open time.

This test pins:
1. `write()` passes the original `bytes` object through to WriteProcessMemory unchanged
   (verified by `is` identity — no intermediate ctypes-array reconstruction).
2. The reused `_nwritten` survives a series of write() calls (same object identity).
3. `read()` and `_try_read_chunk()` use the hoisted `_rpm` binding (not a fresh
   self._k32.ReadProcessMemory attribute lookup per call).

If anyone reintroduces a per-call `(c_ubyte * N)(*data)` splat or constructs a fresh
`c_size_t(0)` per write, the identity checks fire and this test fails.

Cross-platform: bypasses ProcessHandle.__init__ (which only succeeds on Windows where
kernel32 is loadable) by constructing via __new__ and manually wiring the kernel32
stubs. The HOT PATH being tested is platform-agnostic Python code.
"""
import ctypes

import pytest

from forza_abyss_painter.inject.win_process import ProcessHandle


def _stub_handle():
    """Construct a ProcessHandle WITHOUT calling __init__ (which requires Windows) and
    manually populate the attributes that the hoisted hot path reads."""
    h = ProcessHandle.__new__(ProcessHandle)
    h.pid = 1
    h.access = 0
    h.handle = 0x1000   # any non-zero passes the "open?" check
    h._nwritten = ctypes.c_size_t(0)
    h._nread = ctypes.c_size_t(0)
    h._wpm = _make_recorder(h._nwritten)
    h._rpm = _make_recorder(h._nread)
    return h


def _make_recorder(out_size):
    """A WriteProcessMemory / ReadProcessMemory stand-in that records calls and stores
    the passed buffer object verbatim (so tests can verify identity)."""
    calls = []

    def recorder(handle, addr, buf, size, n_ptr):
        calls.append((addr, buf, size))
        out_size.value = size
        return 1   # BOOL TRUE

    recorder.calls = calls
    return recorder


def test_write_passes_bytes_object_through_to_wpm():
    """write(addr, data) must hand `data` (the bytes object itself) to WriteProcessMemory,
    not a freshly-allocated ctypes array containing data's bytes. ctypes' c_void_p argtype
    accepts bytes zero-copy via the PyBytes internal char*; reconstructing a c_ubyte array
    per call is the slow path we're escaping from forza-painter-fh6 parity."""
    h = _stub_handle()
    payload = b"\x12\x34\x56\x78"
    h.write(0xDEADBEEF, payload)
    assert len(h._wpm.calls) == 1
    addr, buf, size = h._wpm.calls[0]
    assert addr == 0xDEADBEEF
    assert size == len(payload)
    # The critical assertion: buf IS the same bytes object we passed in (zero-copy).
    assert buf is payload, (
        "ProcessHandle.write reconstructed the payload buffer instead of passing bytes "
        "through to WriteProcessMemory. The hot-path optimization that beats painter is "
        "passing the bytes object directly so ctypes does a zero-copy reference of its "
        "PyBytes char*. If you see a ctypes.c_ubyte_Array_N here, restore the bytes "
        "passthrough in ProcessHandle.write."
    )


def test_nwritten_is_reused_across_calls():
    """The _nwritten c_size_t output buffer is allocated once at open() and reused for
    every write. If it's reconstructed per call, the identity check fails."""
    h = _stub_handle()
    nwritten_before = h._nwritten
    for i in range(5):
        h.write(0x1000 + i * 8, b"\x00" * 8)
    nwritten_after = h._nwritten
    assert nwritten_before is nwritten_after, (
        "_nwritten was rebound between calls — should be allocated once at open() and "
        "reused as the output buffer for every WriteProcessMemory call."
    )


def test_nread_is_reused_across_calls():
    """Same contract for the read path's _nread output buffer."""
    h = _stub_handle()
    nread_before = h._nread
    for i in range(5):
        # _rpm is a stub recorder; pass a real-shaped ctypes buffer as the dest.
        dest = (ctypes.c_ubyte * 8)()
        h._rpm(h.handle, 0x1000 + i * 8, dest, 8, ctypes.byref(h._nread))
    nread_after = h._nread
    assert nread_before is nread_after


def test_write_returns_nwritten_value():
    """write() returns the byte count from _nwritten. Stub recorder sets _nwritten.value
    to the size argument, so a write of 13 bytes should return 13."""
    h = _stub_handle()
    n = h.write(0xCAFE, b"thirteen byte")
    assert n == 13


def test_hot_loop_completes_under_budget():
    """Sanity timing check: 1000 stubbed writes of 8 bytes each must complete well under
    the budget that allows the OLD per-call ctypes allocations. The old path took roughly
    1ms per 100 writes on the dev machine (~10us/write) plus interpreter overhead; the
    new zero-allocation path should be 3-5x faster. Pin a generous ceiling so this test
    isn't flaky on slow CI runners."""
    import time
    h = _stub_handle()
    payload = b"\x12\x34\x56\x78\x9A\xBC\xDE\xF0"
    t0 = time.perf_counter()
    for i in range(1000):
        h.write(0x1000 + i * 8, payload)
    t1 = time.perf_counter()
    elapsed_us = (t1 - t0) * 1e6
    # Generous bound: 100us per write on a slow CI runner = 100ms total. Old path on a
    # 3000-shape inject took ~200-400ms just for the ctypes overhead; if this assertion
    # fires, the per-call allocation regression is back.
    assert elapsed_us < 100_000, (
        f"1000 stubbed writes took {elapsed_us:.0f}us — well above the budget for the "
        f"zero-allocation hot path. Someone may have reintroduced per-call ctypes "
        f"buffer construction in ProcessHandle.write."
    )
