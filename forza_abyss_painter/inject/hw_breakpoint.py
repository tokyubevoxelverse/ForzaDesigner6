"""Hardware-breakpoint debugger for live RE.

Uses Win32 DebugActiveProcess + DR0-DR3 + WaitForDebugEvent to capture which
instruction(s) write to a target address. Equivalent of CE's "Find out what
writes to this address" feature.

Usage:
    python -m forza_abyss_painter.inject.hw_breakpoint <pid> <address_hex> [--seconds N]

Sets a hardware write breakpoint on `address` (4 bytes wide), waits for it to
fire, and prints the writing instruction's RIP + the bytes around it. Detaches
cleanly when the user Ctrl+Cs or after `seconds` elapses.

Notes / limitations:
  - DebugActiveProcess sets PEB.BeingDebugged on the target. Anti-cheat may detect.
  - Only catches writes that happen AFTER we attach. Have user trigger the
    suspected action (e.g., change color) while this is running.
  - DR0-DR3 give us 4 hardware breakpoints per thread. We use DR0 only.
"""

from __future__ import annotations

import argparse
import ctypes
import struct
import sys
import time
from ctypes import wintypes


# -- Win32 constants --
DEBUG_PROCESS = 0x00000001
DEBUG_ONLY_THIS_PROCESS = 0x00000002
INFINITE = 0xFFFFFFFF
EXCEPTION_DEBUG_EVENT = 1
CREATE_THREAD_DEBUG_EVENT = 2
CREATE_PROCESS_DEBUG_EVENT = 3
EXIT_THREAD_DEBUG_EVENT = 4
EXIT_PROCESS_DEBUG_EVENT = 5
LOAD_DLL_DEBUG_EVENT = 6
UNLOAD_DLL_DEBUG_EVENT = 7
OUTPUT_DEBUG_STRING_EVENT = 8

STATUS_SINGLE_STEP = 0x80000004
STATUS_BREAKPOINT = 0x80000003

DBG_CONTINUE = 0x00010002
DBG_EXCEPTION_NOT_HANDLED = 0x80010001

CONTEXT_AMD64 = 0x00100000
CONTEXT_CONTROL = CONTEXT_AMD64 | 0x00000001
CONTEXT_INTEGER = CONTEXT_AMD64 | 0x00000002
CONTEXT_SEGMENTS = CONTEXT_AMD64 | 0x00000004
CONTEXT_FLOATING_POINT = CONTEXT_AMD64 | 0x00000008
CONTEXT_DEBUG_REGISTERS = CONTEXT_AMD64 | 0x00000010
CONTEXT_FULL = CONTEXT_CONTROL | CONTEXT_INTEGER | CONTEXT_FLOATING_POINT

THREAD_ALL_ACCESS = 0x001F03FF
THREAD_GET_CONTEXT = 0x0008
THREAD_SET_CONTEXT = 0x0010
THREAD_SUSPEND_RESUME = 0x0002

TH32CS_SNAPTHREAD = 0x00000004


# -- Structures --
class CONTEXT(ctypes.Structure):
    _fields_ = [
        ("P1Home", ctypes.c_ulonglong), ("P2Home", ctypes.c_ulonglong),
        ("P3Home", ctypes.c_ulonglong), ("P4Home", ctypes.c_ulonglong),
        ("P5Home", ctypes.c_ulonglong), ("P6Home", ctypes.c_ulonglong),
        ("ContextFlags", wintypes.DWORD),
        ("MxCsr", wintypes.DWORD),
        ("SegCs", wintypes.WORD), ("SegDs", wintypes.WORD), ("SegEs", wintypes.WORD),
        ("SegFs", wintypes.WORD), ("SegGs", wintypes.WORD), ("SegSs", wintypes.WORD),
        ("EFlags", wintypes.DWORD),
        ("Dr0", ctypes.c_ulonglong), ("Dr1", ctypes.c_ulonglong),
        ("Dr2", ctypes.c_ulonglong), ("Dr3", ctypes.c_ulonglong),
        ("Dr6", ctypes.c_ulonglong), ("Dr7", ctypes.c_ulonglong),
        ("Rax", ctypes.c_ulonglong), ("Rcx", ctypes.c_ulonglong),
        ("Rdx", ctypes.c_ulonglong), ("Rbx", ctypes.c_ulonglong),
        ("Rsp", ctypes.c_ulonglong), ("Rbp", ctypes.c_ulonglong),
        ("Rsi", ctypes.c_ulonglong), ("Rdi", ctypes.c_ulonglong),
        ("R8", ctypes.c_ulonglong), ("R9", ctypes.c_ulonglong),
        ("R10", ctypes.c_ulonglong), ("R11", ctypes.c_ulonglong),
        ("R12", ctypes.c_ulonglong), ("R13", ctypes.c_ulonglong),
        ("R14", ctypes.c_ulonglong), ("R15", ctypes.c_ulonglong),
        ("Rip", ctypes.c_ulonglong),
        # FP regs + Vector regs follow — we don't need them; pad as needed
        ("_padding", ctypes.c_ubyte * 4096),
    ]
    _pack_ = 16


class THREADENTRY32(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD),
        ("cntUsage", wintypes.DWORD),
        ("th32ThreadID", wintypes.DWORD),
        ("th32OwnerProcessID", wintypes.DWORD),
        ("tpBasePri", wintypes.LONG),
        ("tpDeltaPri", wintypes.LONG),
        ("dwFlags", wintypes.DWORD),
    ]


class EXCEPTION_RECORD(ctypes.Structure):
    pass

EXCEPTION_RECORD._fields_ = [
    ("ExceptionCode", wintypes.DWORD),
    ("ExceptionFlags", wintypes.DWORD),
    ("ExceptionRecord", ctypes.POINTER(EXCEPTION_RECORD)),
    ("ExceptionAddress", ctypes.c_void_p),
    ("NumberParameters", wintypes.DWORD),
    ("ExceptionInformation", ctypes.c_ulonglong * 15),
]


class EXCEPTION_DEBUG_INFO(ctypes.Structure):
    _fields_ = [
        ("ExceptionRecord", EXCEPTION_RECORD),
        ("dwFirstChance", wintypes.DWORD),
    ]


class DEBUG_EVENT_UNION(ctypes.Union):
    _fields_ = [
        ("Exception", EXCEPTION_DEBUG_INFO),
        # Other event types omitted — we only care about exceptions
        ("_padding", ctypes.c_ubyte * 256),
    ]


class DEBUG_EVENT(ctypes.Structure):
    _fields_ = [
        ("dwDebugEventCode", wintypes.DWORD),
        ("dwProcessId", wintypes.DWORD),
        ("dwThreadId", wintypes.DWORD),
        ("u", DEBUG_EVENT_UNION),
    ]


k32 = ctypes.WinDLL("kernel32", use_last_error=True)
k32.DebugActiveProcess.argtypes = [wintypes.DWORD]
k32.DebugActiveProcess.restype = wintypes.BOOL
k32.DebugActiveProcessStop.argtypes = [wintypes.DWORD]
k32.DebugActiveProcessStop.restype = wintypes.BOOL
k32.DebugSetProcessKillOnExit.argtypes = [wintypes.BOOL]
k32.DebugSetProcessKillOnExit.restype = wintypes.BOOL
k32.WaitForDebugEvent.argtypes = [ctypes.POINTER(DEBUG_EVENT), wintypes.DWORD]
k32.WaitForDebugEvent.restype = wintypes.BOOL
k32.ContinueDebugEvent.argtypes = [wintypes.DWORD, wintypes.DWORD, wintypes.DWORD]
k32.ContinueDebugEvent.restype = wintypes.BOOL
k32.OpenThread.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
k32.OpenThread.restype = wintypes.HANDLE
k32.GetThreadContext.argtypes = [wintypes.HANDLE, ctypes.POINTER(CONTEXT)]
k32.GetThreadContext.restype = wintypes.BOOL
k32.SetThreadContext.argtypes = [wintypes.HANDLE, ctypes.POINTER(CONTEXT)]
k32.SetThreadContext.restype = wintypes.BOOL
k32.SuspendThread.argtypes = [wintypes.HANDLE]
k32.SuspendThread.restype = wintypes.DWORD
k32.ResumeThread.argtypes = [wintypes.HANDLE]
k32.ResumeThread.restype = wintypes.DWORD
k32.CloseHandle.argtypes = [wintypes.HANDLE]
k32.CloseHandle.restype = wintypes.BOOL
k32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
k32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
k32.Thread32First.argtypes = [wintypes.HANDLE, ctypes.POINTER(THREADENTRY32)]
k32.Thread32First.restype = wintypes.BOOL
k32.Thread32Next.argtypes = [wintypes.HANDLE, ctypes.POINTER(THREADENTRY32)]
k32.Thread32Next.restype = wintypes.BOOL
k32.ReadProcessMemory.argtypes = [
    wintypes.HANDLE, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)
]
k32.ReadProcessMemory.restype = wintypes.BOOL
k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
k32.OpenProcess.restype = wintypes.HANDLE


def list_threads(pid: int) -> list[int]:
    """Return thread IDs of all threads in process pid."""
    snap = k32.CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0)
    if not snap:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        te = THREADENTRY32()
        te.dwSize = ctypes.sizeof(THREADENTRY32)
        if not k32.Thread32First(snap, ctypes.byref(te)):
            return []
        tids = []
        while True:
            if te.th32OwnerProcessID == pid:
                tids.append(te.th32ThreadID)
            if not k32.Thread32Next(snap, ctypes.byref(te)):
                break
        return tids
    finally:
        k32.CloseHandle(snap)


def _build_dr7(slot: int, length_code: int = 0b11, type_code: int = 0b01) -> int:
    """Build DR7 value enabling one breakpoint in slot 0-3.
    type_code: 00=exec, 01=write, 10=io, 11=read/write
    length_code: 00=1 byte, 01=2 bytes, 10=8 bytes, 11=4 bytes
    """
    dr7 = 0
    dr7 |= (1 << (slot * 2 + 1))  # local enable bit for slot (use global, bit 2*i+1)
    # Conditions: bits 16-31, 4 bits per slot (2 type + 2 length)
    base = 16 + slot * 4
    dr7 |= (type_code & 0b11) << base
    dr7 |= (length_code & 0b11) << (base + 2)
    return dr7


def set_hw_breakpoint_on_all_threads(pid: int, target_addr: int) -> int:
    """Set DR0 = target_addr, DR7 = enable write-breakpoint on slot 0, on every thread.
    Returns count of threads we successfully programmed.
    """
    dr7 = _build_dr7(slot=0, length_code=0b11, type_code=0b01)
    tids = list_threads(pid)
    programmed = 0
    for tid in tids:
        h = k32.OpenThread(THREAD_GET_CONTEXT | THREAD_SET_CONTEXT | THREAD_SUSPEND_RESUME, False, tid)
        if not h:
            continue
        try:
            k32.SuspendThread(h)
            ctx = CONTEXT()
            ctx.ContextFlags = CONTEXT_DEBUG_REGISTERS
            if k32.GetThreadContext(h, ctypes.byref(ctx)):
                ctx.Dr0 = target_addr
                ctx.Dr7 = dr7
                if k32.SetThreadContext(h, ctypes.byref(ctx)):
                    programmed += 1
            k32.ResumeThread(h)
        finally:
            k32.CloseHandle(h)
    return programmed


def clear_hw_breakpoint_on_all_threads(pid: int) -> None:
    for tid in list_threads(pid):
        h = k32.OpenThread(THREAD_GET_CONTEXT | THREAD_SET_CONTEXT | THREAD_SUSPEND_RESUME, False, tid)
        if not h:
            continue
        try:
            k32.SuspendThread(h)
            ctx = CONTEXT()
            ctx.ContextFlags = CONTEXT_DEBUG_REGISTERS
            if k32.GetThreadContext(h, ctypes.byref(ctx)):
                ctx.Dr0 = 0; ctx.Dr7 = 0
                k32.SetThreadContext(h, ctypes.byref(ctx))
            k32.ResumeThread(h)
        finally:
            k32.CloseHandle(h)


def read_bytes(pid: int, addr: int, size: int) -> bytes | None:
    h = k32.OpenProcess(0x0410, False, pid)
    if not h: return None
    try:
        buf = (ctypes.c_ubyte * size)()
        n = ctypes.c_size_t(0)
        ok = k32.ReadProcessMemory(h, ctypes.c_void_p(addr), buf, size, ctypes.byref(n))
        if not ok: return None
        return bytes(buf[:n.value])
    finally:
        k32.CloseHandle(h)


def run(pid: int, target_addr: int, seconds: int = 30) -> list[dict]:
    """Attach as debugger, set hw breakpoint, capture write events, detach. Returns list of hits."""
    if not k32.DebugActiveProcess(pid):
        raise ctypes.WinError(ctypes.get_last_error())
    k32.DebugSetProcessKillOnExit(False)
    print(f"[hwbp] attached as debugger to PID {pid}")
    hits: list[dict] = []
    try:
        # Wait for initial debug event (LOAD_DLL etc.) and continue them
        # Then set our breakpoint
        n_programmed = 0
        end_time = time.time() + seconds
        while time.time() < end_time:
            ev = DEBUG_EVENT()
            remaining = max(100, int((end_time - time.time()) * 1000))
            if not k32.WaitForDebugEvent(ctypes.byref(ev), min(1000, remaining)):
                # No event in this interval — try again, also try setting bp if not yet
                if n_programmed == 0:
                    n_programmed = set_hw_breakpoint_on_all_threads(pid, target_addr)
                    if n_programmed:
                        print(f"[hwbp] set breakpoint on {n_programmed} thread(s)")
                continue
            code = ev.dwDebugEventCode
            if code == EXCEPTION_DEBUG_EVENT:
                exc_code = ev.u.Exception.ExceptionRecord.ExceptionCode
                exc_addr = ev.u.Exception.ExceptionRecord.ExceptionAddress
                if exc_code == STATUS_SINGLE_STEP:
                    # Our hw bp fired
                    # Read the writing instruction's RIP from the thread context
                    th = k32.OpenThread(THREAD_GET_CONTEXT | THREAD_SET_CONTEXT, False, ev.dwThreadId)
                    rip = exc_addr or 0
                    if th:
                        try:
                            ctx = CONTEXT()
                            ctx.ContextFlags = CONTEXT_FULL | CONTEXT_DEBUG_REGISTERS
                            if k32.GetThreadContext(th, ctypes.byref(ctx)):
                                rip = ctx.Rip
                                hit = {
                                    "tid": ev.dwThreadId,
                                    "rip": rip,
                                    "rax": ctx.Rax, "rcx": ctx.Rcx, "rdx": ctx.Rdx, "rbx": ctx.Rbx,
                                    "rsi": ctx.Rsi, "rdi": ctx.Rdi, "r8": ctx.R8, "r9": ctx.R9,
                                    "r10": ctx.R10, "r11": ctx.R11, "r12": ctx.R12, "r13": ctx.R13,
                                    "r14": ctx.R14, "r15": ctx.R15,
                                }
                                bytes_around = read_bytes(pid, max(0, rip - 16), 48)
                                if bytes_around:
                                    hit["bytes_around_rip"] = bytes_around.hex()
                                hits.append(hit)
                                print(f"[hwbp] WRITE caught — tid={ev.dwThreadId} RIP=0x{rip:016X}  RAX=0x{ctx.Rax:X} RCX=0x{ctx.Rcx:X} RDX=0x{ctx.Rdx:X}")
                                # Clear DR6 (status) so subsequent bps fire
                                ctx2 = CONTEXT()
                                ctx2.ContextFlags = CONTEXT_DEBUG_REGISTERS
                                if k32.GetThreadContext(th, ctypes.byref(ctx2)):
                                    ctx2.Dr6 = 0
                                    k32.SetThreadContext(th, ctypes.byref(ctx2))
                        finally:
                            k32.CloseHandle(th)
                    k32.ContinueDebugEvent(ev.dwProcessId, ev.dwThreadId, DBG_CONTINUE)
                    continue
                # Other exceptions — pass through
                k32.ContinueDebugEvent(ev.dwProcessId, ev.dwThreadId, DBG_EXCEPTION_NOT_HANDLED)
                continue
            if code == CREATE_PROCESS_DEBUG_EVENT or code == CREATE_THREAD_DEBUG_EVENT:
                # Re-program breakpoints (new thread)
                k32.ContinueDebugEvent(ev.dwProcessId, ev.dwThreadId, DBG_CONTINUE)
                set_hw_breakpoint_on_all_threads(pid, target_addr)
                continue
            if code == EXIT_PROCESS_DEBUG_EVENT:
                print("[hwbp] target process exited")
                return hits
            # Default: continue
            k32.ContinueDebugEvent(ev.dwProcessId, ev.dwThreadId, DBG_CONTINUE)
    finally:
        # Cleanup
        try:
            clear_hw_breakpoint_on_all_threads(pid)
        except Exception:
            pass
        if not k32.DebugActiveProcessStop(pid):
            print(f"[hwbp] warning: DebugActiveProcessStop failed ({ctypes.get_last_error()})")
        else:
            print("[hwbp] detached cleanly")
    return hits


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pid", type=int)
    ap.add_argument("address", help="target address (hex, e.g. 0x000002229EF2403C)")
    ap.add_argument("--seconds", type=int, default=60)
    args = ap.parse_args()
    addr = int(args.address, 0)
    print(f"[hwbp] arming breakpoint at 0x{addr:016X} on PID {args.pid} for {args.seconds}s")
    print(f"[hwbp] NOW change the color in FH6 — every write to that 4-byte slot will be caught")
    hits = run(args.pid, addr, args.seconds)
    print()
    print(f"[hwbp] DONE. Captured {len(hits)} write event(s).")
    for i, h in enumerate(hits):
        print(f"  [{i}] tid={h['tid']}  RIP=0x{h['rip']:016X}")
        if "bytes_around_rip" in h:
            print(f"      bytes around RIP-16: {h['bytes_around_rip']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
