"""Pin the partial-read contract on ProcessHandle._try_read_chunk.

The earlier version rejected partial reads (`if not ok or nread.value == 0`),
breaking fast-mode locate on FH6 UWP 3.360.x where scan window 3
(0x0a000000-0x0c000000) extends past the 178.93 MiB module end. Windows
returns ERROR_PARTIAL_COPY with ~11 MiB of valid bytes; we used to throw
them away. Painter's read_process_memory (native.py:62-75) accepts them.

This regression test mocks the kernel32 RPM call to simulate partial-read
semantics and asserts our helper returns the partial bytes.
"""
from __future__ import annotations

import ctypes
import sys
from unittest import mock

import pytest


def _make_fake_handle(partial_bytes: bytes, ok_status: bool):
    """Build a ProcessHandle-shaped object where _rpm simulates a kernel
    call that wrote `len(partial_bytes)` valid bytes into the buffer AND
    returned `ok_status` (True or False)."""
    # ProcessHandle's win_process module is Windows-only (raises ImportError
    # on dev). Stub the win_process module load to make the test runnable
    # cross-platform; we only exercise the helper method, not anything
    # Windows-specific.
    pytest.importorskip("forza_abyss_painter.inject.win_process",
                       reason="win_process loads on every platform — bytes module")

    from forza_abyss_painter.inject.win_process import ProcessHandle

    # Construct a real ProcessHandle without opening it (we'll stub _rpm).
    class _FakeHandle(ProcessHandle):
        def __init__(self):   # noqa: D401 — bypass real OpenProcess
            self.pid = 0
            self.handle = 0x12345
            self._nread = ctypes.c_size_t(0)
            self.access = 0
            # Stub _rpm to simulate the partial-read scenario.
            def _fake_rpm(handle, addr, buf, size, nread_ptr):
                n = min(len(partial_bytes), size)
                ctypes.memmove(buf, (ctypes.c_ubyte * n)(*partial_bytes[:n]), n)
                nread_ptr._obj.value = n   # write to the c_size_t target
                self._nread.value = n
                return ok_status
            self._rpm = _fake_rpm
    return _FakeHandle()


def test_try_read_chunk_accepts_partial_read_with_rpm_false():
    """Critical: RPM returns False (ERROR_PARTIAL_COPY) but nread > 0 →
    helper must return the partial bytes, NOT None. This is THE bug that
    blocked fast-mode locate on FH6 UWP 3.360.x — window 3 extends past
    the module end, partial bytes contain the sentinel."""
    partial = b"\x12\x47\x9b\x13\x29\xd9\xa2\xb1" + b"\x00" * 100   # sig + padding
    handle = _make_fake_handle(partial, ok_status=False)
    out = handle._try_read_chunk(addr=0x1000, size=4096)
    assert out is not None, (
        "partial-read rejected — fast-mode locate would miss the sentinel "
        "in scan windows that extend past the module end"
    )
    assert out == partial


def test_try_read_chunk_accepts_full_read_with_rpm_true():
    """Happy path: RPM returns True, nread = size → returns full bytes."""
    data = b"\x00" * 4096
    handle = _make_fake_handle(data, ok_status=True)
    out = handle._try_read_chunk(addr=0x1000, size=4096)
    assert out == data


def test_try_read_chunk_returns_none_only_when_nread_zero():
    """Genuinely unreadable: zero bytes returned, RPM False → return None.
    This is the only case where None is correct; ALL other cases are
    partials we want."""
    handle = _make_fake_handle(b"", ok_status=False)
    out = handle._try_read_chunk(addr=0x1000, size=4096)
    assert out is None
