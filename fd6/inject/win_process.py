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

    def close(self) -> None:
        if self.handle:
            self._k32.CloseHandle(self.handle)
            self.handle = None

    def read(self, addr: int, size: int) -> bytes:
        if self.handle is None:
            raise RuntimeError("Process handle not open")
        buf = (ctypes.c_ubyte * size)()
        n = ctypes.c_size_t(0)
        ok = self._k32.ReadProcessMemory(self.handle, ctypes.c_void_p(addr), buf, size, ctypes.byref(n))
        if not ok:
            raise ctypes.WinError(ctypes.get_last_error())
        return bytes(buf[:n.value])

    def write(self, addr: int, data: bytes) -> int:
        if self.handle is None:
            raise RuntimeError("Process handle not open")
        buf = (ctypes.c_ubyte * len(data))(*data)
        n = ctypes.c_size_t(0)
        ok = self._k32.WriteProcessMemory(self.handle, ctypes.c_void_p(addr), buf, len(data), ctypes.byref(n))
        if not ok:
            raise ctypes.WinError(ctypes.get_last_error())
        return n.value

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
        buf = (ctypes.c_ubyte * size)()
        n = ctypes.c_size_t(0)
        ok = self._k32.ReadProcessMemory(
            self.handle, ctypes.c_void_p(addr), buf,
            ctypes.c_size_t(size), ctypes.byref(n),
        )
        if not ok or n.value == 0:
            return None
        return bytes(buf[:n.value])

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
