"""Thin ctypes wrappers around the Win32 process-memory APIs.

These are the same primitives forza-painter uses (per its public README): OpenProcess,
ReadProcessMemory, WriteProcessMemory. Exported as a reusable helper so per-game
injectors don't each reinvent the syscalls.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import dataclass
import platform


PROCESS_VM_READ = 0x0010
PROCESS_VM_WRITE = 0x0020
PROCESS_VM_OPERATION = 0x0008
PROCESS_QUERY_INFORMATION = 0x0400
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

ALL_ACCESS = PROCESS_VM_READ | PROCESS_VM_WRITE | PROCESS_VM_OPERATION | PROCESS_QUERY_INFORMATION

# VirtualQueryEx constants
MEM_COMMIT = 0x1000
MEM_FREE = 0x10000
MEM_RESERVE = 0x2000
MEM_IMAGE = 0x1000000
MEM_MAPPED = 0x40000
MEM_PRIVATE = 0x20000

PAGE_NOACCESS = 0x01
PAGE_READONLY = 0x02
PAGE_READWRITE = 0x04
PAGE_WRITECOPY = 0x08
PAGE_EXECUTE = 0x10
PAGE_EXECUTE_READ = 0x20
PAGE_EXECUTE_READWRITE = 0x40
PAGE_EXECUTE_WRITECOPY = 0x80
PAGE_GUARD = 0x100

READABLE_FLAGS = (
    PAGE_READONLY | PAGE_READWRITE | PAGE_WRITECOPY
    | PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY
)
WRITABLE_FLAGS = PAGE_READWRITE | PAGE_WRITECOPY | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY


class MEMORY_BASIC_INFORMATION64(ctypes.Structure):
    _fields_ = [
        ("BaseAddress", ctypes.c_ulonglong),
        ("AllocationBase", ctypes.c_ulonglong),
        ("AllocationProtect", wintypes.DWORD),
        ("__alignment1", wintypes.DWORD),
        ("RegionSize", ctypes.c_ulonglong),
        ("State", wintypes.DWORD),
        ("Protect", wintypes.DWORD),
        ("Type", wintypes.DWORD),
        ("__alignment2", wintypes.DWORD),
    ]


@dataclass
class MemoryRegion:
    base: int
    size: int
    protect: int
    type: int  # MEM_IMAGE / MEM_MAPPED / MEM_PRIVATE

    @property
    def readable(self) -> bool:
        return bool(self.protect & READABLE_FLAGS) and not (self.protect & PAGE_GUARD)

    @property
    def writable(self) -> bool:
        return bool(self.protect & WRITABLE_FLAGS) and not (self.protect & PAGE_GUARD)

    @property
    def is_private(self) -> bool:
        return self.type == MEM_PRIVATE

    @property
    def is_image(self) -> bool:
        return self.type == MEM_IMAGE


def _is_windows() -> bool:
    return platform.system().lower().startswith("win")


def _kernel32():
    if not _is_windows():
        raise OSError("win_process is only available on Windows")
    return ctypes.WinDLL("kernel32", use_last_error=True)


def find_process_id(name: str) -> int | None:
    """Find the first PID matching the given executable name (case-insensitive). None if not found."""
    if not _is_windows():
        raise OSError("find_process_id is only available on Windows")
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    k32 = _kernel32()
    array_size = 4096
    pids = (wintypes.DWORD * array_size)()
    bytes_returned = wintypes.DWORD()
    if not psapi.EnumProcesses(ctypes.byref(pids), ctypes.sizeof(pids), ctypes.byref(bytes_returned)):
        raise ctypes.WinError(ctypes.get_last_error())
    count = bytes_returned.value // ctypes.sizeof(wintypes.DWORD)
    target = name.lower()
    for i in range(count):
        pid = pids[i]
        if pid == 0:
            continue
        h = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_VM_READ, False, pid)
        if not h:
            continue
        try:
            buf = ctypes.create_unicode_buffer(260)
            n = psapi.GetModuleFileNameExW(h, None, buf, 260)
            if n and target in buf.value.lower():
                return pid
        finally:
            k32.CloseHandle(h)
    return None


class ProcessHandle:
    """Context manager around OpenProcess + Read/Write memory."""

    def __init__(self, pid: int, access: int = ALL_ACCESS) -> None:
        self.pid = pid
        self.access = access
        self.handle: int | None = None
        self._k32 = _kernel32()

    def __enter__(self) -> "ProcessHandle":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def open(self) -> None:
        h = self._k32.OpenProcess(self.access, False, self.pid)
        if not h:
            raise ctypes.WinError(ctypes.get_last_error())
        self.handle = h
        # Pin argtypes for ReadProcessMemory / WriteProcessMemory so the size
        # argument is treated as SIZE_T (c_size_t) instead of ctypes' default
        # c_int. On systems where FH6 commits a single read-writable region
        # larger than 2 GiB (some texture/shader heaps), the unpinned default
        # overflows int32 and raises "argument 4: OverflowError: int too long
        # to convert" the moment the scanner touches that region.
        self._k32.ReadProcessMemory.argtypes = [
            wintypes.HANDLE, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t),
        ]
        self._k32.ReadProcessMemory.restype = wintypes.BOOL
        self._k32.WriteProcessMemory.argtypes = [
            wintypes.HANDLE, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t),
        ]
        self._k32.WriteProcessMemory.restype = wintypes.BOOL
        # Hoist bound functions + reusable output buffers out of the hot path. Every
        # WriteProcessMemory call previously paid for a fresh `ctypes.c_size_t(0)` and a
        # fresh `(c_ubyte * len)(*data)` construction; on a 3000-shape inject that's 18k
        # ctypes allocations + ~138k byte-splats through the Python interpreter — the
        # single biggest source of injection-speed lag vs forza-painter-fh6. ctypes
        # accepts a Python `bytes` object directly as the c_void_p buffer argument
        # (zero-copy via the internal PyBytes char*), so we just pass `data` through.
        # Pre-allocated _nwritten / _nread are mutated in place across calls — safe
        # because ProcessHandle is single-threaded (one per inject worker).
        self._wpm = self._k32.WriteProcessMemory
        self._rpm = self._k32.ReadProcessMemory
        self._nwritten = ctypes.c_size_t(0)
        self._nread = ctypes.c_size_t(0)

    def close(self) -> None:
        if self.handle:
            self._k32.CloseHandle(self.handle)
            self.handle = None

    def read(self, addr: int, size: int) -> bytes:
        if self.handle is None:
            raise RuntimeError("Process handle not open")
        buf = (ctypes.c_ubyte * size)()
        ok = self._rpm(self.handle, addr, buf, size, ctypes.byref(self._nread))
        if not ok:
            raise ctypes.WinError(ctypes.get_last_error())
        return bytes(buf[:self._nread.value])

    def write(self, addr: int, data: bytes) -> int:
        if self.handle is None:
            raise RuntimeError("Process handle not open")
        # `data` is a bytes object (from struct.pack in the inject loop). ctypes' c_void_p
        # argtype accepts bytes directly with zero copy. addr likewise — c_void_p accepts
        # Python int. No per-call allocations.
        ok = self._wpm(self.handle, addr, data, len(data), ctypes.byref(self._nwritten))
        if not ok:
            raise ctypes.WinError(ctypes.get_last_error())
        return self._nwritten.value

    # Per-call read cap. ReadProcessMemory itself supports SIZE_T sizes once
    # argtypes are pinned, but allocating a multi-gigabyte ctypes buffer up
    # front is wasteful and can OOM on 8 GB boxes. We slice the request into
    # 256 MiB chunks and stitch the results back together — callers see a
    # single bytes object covering the requested range.
    _TRY_READ_CHUNK = 256 * 1024 * 1024

    def try_read(self, addr: int, size: int) -> bytes | None:
        """Like read() but returns None on failure (used during memory scanning where some pages disappear)."""
        if self.handle is None:
            raise RuntimeError("Process handle not open")
        if size <= 0:
            return b""
        if size <= self._TRY_READ_CHUNK:
            return self._try_read_chunk(addr, size)
        out = bytearray()
        remaining = size
        cursor = addr
        while remaining > 0:
            take = min(remaining, self._TRY_READ_CHUNK)
            chunk = self._try_read_chunk(cursor, take)
            if chunk is None:
                # Partial read up to this chunk is still usable for the caller's
                # pattern scan; return what we have if any, else None.
                return bytes(out) if out else None
            out.extend(chunk)
            if len(chunk) < take:
                # Short read — page disappeared mid-region. Stop here.
                break
            cursor += take
            remaining -= take
        return bytes(out)

    def _try_read_chunk(self, addr: int, size: int) -> bytes | None:
        """Read up to `size` bytes from `addr`. Returns whatever was actually
        readable, even when ReadProcessMemory returns False with a non-zero
        partial nread count (ERROR_PARTIAL_COPY = 0x12B).

        PAINTER PARITY (2026-05-25): the previous version rejected partial
        reads (`if not ok or nread.value == 0: return None`). When a scan
        window extended past the end of the committed module image — eg
        our 0x06..0x0C signature scan windows on a 178 MiB FH6 module —
        Windows correctly returned ~11 MiB of valid bytes WITH RPM=False,
        and we threw the readable bytes away. Painter's read_process_memory
        (forza-painter-fh6/src/native.py:62-75) wraps RPM in try/except
        and uses `nread.value` regardless. This is what lets painter's
        fast-mode find the signature on the same build where ours can't:
        the sig at Base+0xa7f1048 falls inside window 3 (0x0a000000-
        0x0c000000) but window 3 extends past the module → our rejected
        partial → no match → fall through to slow mode.

        Returns None ONLY if zero bytes were readable (uncommitted region,
        cross-process protected page, etc). Otherwise returns the partial
        read — the caller's pattern-scan handles short buffers correctly
        (bytes.find returns -1 on no-match, so a partial that doesn't
        contain the needle just behaves the same as a full no-match).
        """
        buf = (ctypes.c_ubyte * size)()
        ok = self._rpm(self.handle, addr, buf, size, ctypes.byref(self._nread))
        # Accept partial reads. The kernel reports nread.value = bytes that
        # WERE valid even when the call returns False, which happens when
        # the requested range crosses an unreadable / uncommitted page.
        if self._nread.value == 0:
            return None
        return bytes(buf[:self._nread.value])

    def enumerate_regions(self) -> list[MemoryRegion]:
        """Walk the process's address space via VirtualQueryEx. Returns committed regions."""
        if self.handle is None:
            raise RuntimeError("Process handle not open")
        regions: list[MemoryRegion] = []
        mbi = MEMORY_BASIC_INFORMATION64()
        addr = 0
        self._k32.VirtualQueryEx.argtypes = [
            wintypes.HANDLE, ctypes.c_void_p, ctypes.POINTER(MEMORY_BASIC_INFORMATION64), ctypes.c_size_t,
        ]
        self._k32.VirtualQueryEx.restype = ctypes.c_size_t
        # Limit address ceiling to user-mode max on x64 Windows: 0x7FFF_FFFF_FFFF
        ceiling = 0x7FFFFFFFFFFF
        while addr < ceiling:
            n = self._k32.VirtualQueryEx(
                self.handle, ctypes.c_void_p(addr), ctypes.byref(mbi), ctypes.sizeof(mbi)
            )
            if n == 0:
                # End of address space or query failed; bump forward to skip past unqueriable region.
                addr += 0x1000
                if addr & 0xFFFFFFF == 0:  # safety: break after large skip
                    break
                continue
            if mbi.State == MEM_COMMIT:
                regions.append(MemoryRegion(
                    base=mbi.BaseAddress,
                    size=mbi.RegionSize,
                    protect=mbi.Protect,
                    type=mbi.Type,
                ))
            new_addr = mbi.BaseAddress + mbi.RegionSize
            if new_addr <= addr:
                break
            addr = new_addr
        return regions
