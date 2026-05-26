"""GPU + runtime + process-management probe for QUASAR.

Run on Windows after a GPU run goes sideways (especially the
'unkillable subprocess' / 'PC restart required' failure). Produces a
single JSON report capturing host + GPU + runtime + orphan-process +
kill-ladder state so Mac-side post-mortem doesn't need access to the
box.

See docs/CURSOR_GPU_PROBE.md for the full schema + how-to-run + what
each section means.

Read-only by design — no installs, no writes, no impact on a future
EXE retry. Stdlib + nvidia-smi + (optionally) the embedded Python's
torch.

Usage (from the QUASAR build share):

    py -3 diagnostics\\probe_gpu_and_runtime.py \\
        --output diagnostics\\cursor_gpu_probe.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import re
import subprocess
import sys
import textwrap
import time
from pathlib import Path


PROBE_VERSION = "1.0"
EXE_NAME_PATTERNS = ("ForzaAbyssPainter", "forzaabysspainter")
RUNNER_CMDLINE_HINTS = (
    "forza_abyss_painter.runtime.torch_runner",
    "ForzaAbyssPainter",
)


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _run_capture(cmd: list[str], timeout: float = 10.0) -> dict:
    """Run a subprocess + capture stdout/stderr/returncode/elapsed.
    Returns a dict suitable for embedding in the report."""
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False,
        )
        return {
            "cmd": cmd, "returncode": proc.returncode,
            "stdout": proc.stdout[-4096:] if proc.stdout else "",
            "stderr": proc.stderr[-2048:] if proc.stderr else "",
            "elapsed_s": round(time.monotonic() - t0, 3),
        }
    except FileNotFoundError as exc:
        return {"cmd": cmd, "error": f"FileNotFoundError: {exc}"}
    except subprocess.TimeoutExpired:
        return {"cmd": cmd, "error": f"timed out after {timeout}s"}
    except OSError as exc:
        return {"cmd": cmd, "error": f"{type(exc).__name__}: {exc}"}


# ============================================================ Section A: host


def probe_host() -> dict:
    info = {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "python_version": sys.version.split()[0],
        "executable": sys.executable,
        "cwd": os.getcwd(),
    }
    # CPU info — Windows-specific via wmic (or via 'systeminfo' as fallback).
    if sys.platform == "win32":
        wmic = _run_capture(["wmic", "cpu", "get", "Name"], timeout=5.0)
        if wmic.get("stdout"):
            lines = [l.strip() for l in wmic["stdout"].splitlines() if l.strip()]
            if len(lines) > 1:
                info["cpu"] = lines[1]
        # Total RAM via wmic.
        ram_wmic = _run_capture(
            ["wmic", "computersystem", "get", "TotalPhysicalMemory"], timeout=5.0,
        )
        if ram_wmic.get("stdout"):
            for line in ram_wmic["stdout"].splitlines():
                line = line.strip()
                if line.isdigit():
                    info["ram_total_gb"] = round(int(line) / (1 << 30), 2)
                    break
    else:
        info["cpu"] = platform.processor() or "(unknown)"
    return info


# ============================================================ Section B: GPU


def probe_gpu() -> dict:
    """nvidia-smi probe. Returns whatever the binary reports + a
    structured list of GPUs. Missing nvidia-smi → empty devices list
    + the error captured for context."""
    result = {"nvidia_smi_available": False, "devices": [],
              "processes_using_gpu": []}

    smi = _run_capture(
        ["nvidia-smi",
         "--query-gpu=name,driver_version,compute_cap,memory.total,memory.used,memory.free",
         "--format=csv,noheader,nounits"],
        timeout=8.0,
    )
    if smi.get("returncode") != 0:
        result["smi_call"] = smi
        return result
    result["nvidia_smi_available"] = True
    for line in smi["stdout"].splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 6:
            try:
                result["devices"].append({
                    "name": parts[0],
                    "driver_version": parts[1],
                    "compute_capability": parts[2],
                    "memory_total_mb": int(parts[3]),
                    "memory_used_mb": int(parts[4]),
                    "memory_free_mb": int(parts[5]),
                })
                # Driver version goes on the top-level result too for
                # quick scanning.
                result.setdefault("driver_version", parts[1])
            except ValueError:
                # Malformed row; skip.
                pass

    # Processes using GPU — separate query (different schema).
    procs = _run_capture(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
         "--format=csv,noheader,nounits"],
        timeout=5.0,
    )
    if procs.get("returncode") == 0:
        for line in procs["stdout"].splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                try:
                    result["processes_using_gpu"].append({
                        "pid": int(parts[0]),
                        "name": parts[1],
                        "memory_mb": int(parts[2]),
                    })
                except ValueError:
                    pass
    return result


# ===================================================== Section C: embedded runtime


def runtime_root() -> Path:
    """Mirror the runtime path resolution from
    forza_abyss_painter.runtime.torch_installer.runtime_root(). Kept
    in sync manually since this script must run without imports from
    the package."""
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or
                    Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or
                    Path.home() / ".local" / "share")
    return base / "ForzaAbyssPainter" / "runtime"


def probe_embedded_runtime() -> dict:
    rt = runtime_root()
    result = {"runtime_root": str(rt), "exists": rt.exists()}
    if not rt.exists():
        return result

    # Marker file.
    marker = rt / "installed.json"
    if marker.exists():
        try:
            result["marker"] = json.loads(marker.read_text(encoding="utf-8"))
        except Exception as exc:
            result["marker_parse_error"] = str(exc)
    else:
        result["marker_present"] = False

    # Directory size + file count.
    total_size = 0
    file_count = 0
    for f in rt.rglob("*"):
        if f.is_file():
            try:
                total_size += f.stat().st_size
                file_count += 1
            except OSError:
                pass
    result["runtime_dir_size_mb"] = round(total_size / (1 << 20), 1)
    result["runtime_dir_file_count"] = file_count

    # Embedded python + torch presence.
    embed_dir = rt / "python311"
    exe_name = "python.exe" if sys.platform == "win32" else "python"
    embed_exe = embed_dir / exe_name
    result["embedded_python_exists"] = embed_exe.exists()

    torch_dist = embed_dir / "Lib" / "site-packages"
    torch_dist_info = list(torch_dist.glob("torch-*.dist-info")) if torch_dist.exists() else []
    result["torch_dist_info"] = [d.name for d in torch_dist_info]
    fap_pkg = torch_dist / "forza_abyss_painter"
    result["forza_abyss_painter_pkg_present"] = fap_pkg.is_dir()
    result["forza_abyss_painter_shapegen_present"] = (fap_pkg / "shapegen").is_dir()

    # Real-kernel probe — invoke the embedded python with
    # torch.zeros(1, device='cuda').add_(1). On success: cuda works
    # end-to-end. On failure: the exact error.
    if embed_exe.exists():
        probe_script = (
            "import json, torch\n"
            "info = {'cuda_available': torch.cuda.is_available()}\n"
            "if info['cuda_available']:\n"
            "    info['device_name'] = torch.cuda.get_device_name(0)\n"
            "    info['compute_capability'] = list(torch.cuda.get_device_capability(0))\n"
            "    try:\n"
            "        x = torch.zeros(1, device='cuda')\n"
            "        x.add_(1)\n"
            "        info['kernel_run_ok'] = True\n"
            "    except Exception as e:\n"
            "        info['kernel_run_ok'] = False\n"
            "        info['kernel_error'] = f'{type(e).__name__}: {e}'\n"
            "print(json.dumps(info))\n"
        )
        kernel = _run_capture([str(embed_exe), "-c", probe_script], timeout=30.0)
        if kernel.get("returncode") == 0:
            try:
                last_line = kernel["stdout"].strip().splitlines()[-1]
                result["real_kernel_probe"] = json.loads(last_line)
            except Exception as exc:
                result["real_kernel_probe"] = {"parse_error": str(exc),
                                                "raw": kernel["stdout"][-500:]}
        else:
            result["real_kernel_probe"] = kernel
    return result


# ============================================ Section D: orphan process detection


def probe_orphan_processes() -> dict:
    """Enumerate python.exe processes + check whether their parent
    EXE is alive. Surfaces the 'subprocess outlived parent' case."""
    result = {"python_processes": [], "orphans": []}
    if sys.platform != "win32":
        # Cross-platform fallback via psutil if available.
        try:
            import psutil
        except ImportError:
            result["error"] = "psutil not available + not on Windows"
            return result
        for p in psutil.process_iter(["pid", "name", "cmdline",
                                      "create_time", "ppid"]):
            try:
                info = p.info
                if info["name"] and "python" in info["name"].lower():
                    entry = {
                        "pid": info["pid"],
                        "name": info["name"],
                        "cmdline": " ".join(info["cmdline"] or [])[:200],
                        "age_seconds": int(time.time() - (info["create_time"] or 0)),
                        "ppid": info["ppid"],
                    }
                    result["python_processes"].append(entry)
            except Exception:
                continue
        return result

    # Windows: wmic process where name like 'python%' + parent check.
    wmic = _run_capture(
        ["wmic", "process", "where", "name like 'python%'",
         "get", "ProcessId,ParentProcessId,CommandLine,CreationDate",
         "/format:csv"],
        timeout=10.0,
    )
    if wmic.get("returncode") != 0:
        result["wmic_error"] = wmic
        return result

    lines = [l.strip() for l in wmic["stdout"].splitlines() if l.strip()]
    if len(lines) < 2:
        return result
    headers = [h.strip() for h in lines[0].split(",")]
    for line in lines[1:]:
        cells = [c.strip() for c in line.split(",")]
        if len(cells) < len(headers):
            continue
        row = dict(zip(headers, cells))
        cmdline = row.get("CommandLine", "")
        pid_s = row.get("ProcessId", "")
        ppid_s = row.get("ParentProcessId", "")
        if not pid_s.isdigit():
            continue
        entry = {
            "pid": int(pid_s),
            "ppid": int(ppid_s) if ppid_s.isdigit() else None,
            "cmdline": cmdline[:300],
            "creation_date": row.get("CreationDate", ""),
        }
        result["python_processes"].append(entry)
        # Orphan check: command line mentions our runner AND parent
        # is no longer alive.
        if any(hint in cmdline for hint in RUNNER_CMDLINE_HINTS):
            parent_alive = _process_alive_windows(entry["ppid"])
            if not parent_alive:
                entry["parent_alive"] = False
                result["orphans"].append(entry)
            else:
                entry["parent_alive"] = True
    return result


def _process_alive_windows(pid: int | None) -> bool:
    if pid is None:
        return False
    check = _run_capture(
        ["tasklist", "/FI", f"PID eq {pid}", "/NH"], timeout=5.0,
    )
    out = check.get("stdout", "")
    # tasklist prints the process row if it exists, or "No tasks..." if not.
    return bool(out and "No tasks" not in out and str(pid) in out)


# ============================================ Section E: kill-ladder self-test


_STUBBORN_CHILD_SOURCE = textwrap.dedent("""
    import signal, sys, time
    # Catch SIGINT (Unix) / SIGBREAK (Windows) and IGNORE it so the
    # ladder has to escalate. Real torch_runner DOES exit on these —
    # this child is deliberately broken to validate the kill ladder.
    def _ignore(*a):
        pass
    if sys.platform == 'win32':
        signal.signal(signal.SIGBREAK, _ignore)
    else:
        signal.signal(signal.SIGINT, _ignore)
        # Linux/macOS: also ignore SIGTERM so escalation reaches SIGKILL.
        signal.signal(signal.SIGTERM, _ignore)
    # Sleep up to 60s — long enough that the ladder escalates fully,
    # short enough that a forgotten zombie cleans up on its own.
    sys.stdout.write('stubborn-child-ready\\n')
    sys.stdout.flush()
    time.sleep(60)
""")


def probe_kill_ladder_self_test() -> dict:
    """Spawn a deliberately stuck child, run the same escalating ladder
    the EXE uses against it, report which stage actually reaped + total
    elapsed time. Validates that the EXE's process-management code path
    is actually working on this Windows build (Defender doesn't block
    Ctrl-Break, etc)."""
    result = {"started_at": _utc_now_iso()}
    # Spawn the stubborn child.
    creation_flags = 0
    if sys.platform == "win32":
        # CREATE_NEW_PROCESS_GROUP so we can send Ctrl-Break.
        creation_flags = 0x00000200
    try:
        proc = subprocess.Popen(
            [sys.executable, "-c", _STUBBORN_CHILD_SOURCE],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            creationflags=creation_flags if sys.platform == "win32" else 0,
            start_new_session=(sys.platform != "win32"),
        )
    except OSError as exc:
        result["spawn_error"] = str(exc)
        return result
    # Wait for child to confirm it's ready (otherwise signal can race
    # the handler install).
    try:
        first_line = proc.stdout.readline()
        if "ready" not in first_line:
            result["readiness_error"] = f"child first line: {first_line!r}"
    except Exception as exc:
        result["readiness_error"] = str(exc)
    result["child_pid"] = proc.pid
    t0 = time.monotonic()

    # Stage 1: graceful signal.
    import signal as _sig
    try:
        if sys.platform == "win32":
            proc.send_signal(_sig.CTRL_BREAK_EVENT)
        else:
            proc.send_signal(_sig.SIGINT)
    except Exception as exc:
        result["signal_error"] = str(exc)
    graceful_t0 = time.monotonic()
    GRACEFUL_TIMEOUT = 4.0
    try:
        proc.wait(timeout=GRACEFUL_TIMEOUT)
        result["stage_that_reaped"] = "graceful"
        result["graceful_seconds"] = round(time.monotonic() - graceful_t0, 2)
        result["total_seconds"] = round(time.monotonic() - t0, 2)
        return result
    except subprocess.TimeoutExpired:
        pass
    result["graceful_seconds"] = round(time.monotonic() - graceful_t0, 2)

    # Stage 2: terminate.
    term_t0 = time.monotonic()
    try:
        proc.terminate()
    except OSError as exc:
        result["terminate_error"] = str(exc)
    TERM_TIMEOUT = 3.0
    try:
        proc.wait(timeout=TERM_TIMEOUT)
        result["stage_that_reaped"] = "terminate"
        result["terminate_seconds"] = round(time.monotonic() - term_t0, 2)
        result["total_seconds"] = round(time.monotonic() - t0, 2)
        return result
    except subprocess.TimeoutExpired:
        pass
    result["terminate_seconds"] = round(time.monotonic() - term_t0, 2)

    # Stage 3: kill.
    kill_t0 = time.monotonic()
    try:
        proc.kill()
    except OSError as exc:
        result["kill_error"] = str(exc)
    KILL_TIMEOUT = 3.0
    try:
        proc.wait(timeout=KILL_TIMEOUT)
        result["stage_that_reaped"] = "kill"
        result["kill_seconds"] = round(time.monotonic() - kill_t0, 2)
        result["total_seconds"] = round(time.monotonic() - t0, 2)
        return result
    except subprocess.TimeoutExpired:
        pass
    result["kill_seconds"] = round(time.monotonic() - kill_t0, 2)

    # If we got here, the OS can't reap a process that ignores
    # signals AND survives kill. This SHOULDN'T be possible on a
    # working Windows install — surface it.
    result["stage_that_reaped"] = "never_reaped"
    result["total_seconds"] = round(time.monotonic() - t0, 2)
    return result


# ============================================ Section F: recent logs


def probe_log_files() -> dict:
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or
                    Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or
                    Path.home() / ".local" / "share")
    logs_dir = base / "ForzaAbyssPainter" / "logs"
    result = {"logs_dir": str(logs_dir), "exists": logs_dir.exists()}
    if not logs_dir.exists():
        return result
    files = []
    for f in sorted(logs_dir.glob("*.log")):
        try:
            stat = f.stat()
            files.append({
                "name": f.name,
                "size_kb": round(stat.st_size / 1024, 1),
                "mtime_utc": dt.datetime.utcfromtimestamp(stat.st_mtime)
                             .replace(tzinfo=dt.timezone.utc)
                             .isoformat(timespec="seconds"),
            })
        except OSError:
            pass
    result["log_files"] = files
    return result


# ============================================ main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="probe_gpu_and_runtime",
        description=(
            "Capture GPU + embedded-Python-runtime + orphan-process "
            "state into a single JSON report for Mac-side triage. See "
            "docs/CURSOR_GPU_PROBE.md for the schema + interpretation."
        ),
    )
    parser.add_argument(
        "-o", "--output", type=Path,
        default=Path("cursor_gpu_probe.json"),
        help="output JSON path (default: ./cursor_gpu_probe.json)",
    )
    parser.add_argument(
        "--skip-kill-ladder", action="store_true",
        help="skip the kill-ladder self-test (spawns a 60s-sleeping "
             "child; harmless but if you're in a hurry you can skip)",
    )
    args = parser.parse_args(argv)

    print("Probing host environment…", file=sys.stderr)
    host = probe_host()
    print("Probing GPU via nvidia-smi…", file=sys.stderr)
    gpu = probe_gpu()
    print("Probing embedded Python runtime…", file=sys.stderr)
    runtime = probe_embedded_runtime()
    print("Enumerating Python processes (orphan check)…", file=sys.stderr)
    orphans = probe_orphan_processes()
    if not args.skip_kill_ladder:
        print("Running kill-ladder self-test (spawns a sleeper)…", file=sys.stderr)
        kill_ladder = probe_kill_ladder_self_test()
    else:
        kill_ladder = {"skipped": True}
    print("Listing log files…", file=sys.stderr)
    logs = probe_log_files()

    report = {
        "captured_at_utc": _utc_now_iso(),
        "probe_version": PROBE_VERSION,
        "host": host,
        "gpu": gpu,
        "embedded_runtime": runtime,
        "orphan_processes": orphans,
        "kill_ladder_self_test": kill_ladder,
        "log_files": logs,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str),
                            encoding="utf-8")
    print(f"\nWrote {args.output} "
          f"({args.output.stat().st_size / 1024:.1f} KiB).",
          file=sys.stderr)
    # Brief stdout summary for the user.
    print("\n=== SUMMARY ===")
    print(f"  Driver:  {gpu.get('driver_version', '(none)')}")
    for dev in gpu.get("devices", []):
        used = dev.get("memory_used_mb", 0)
        total = dev.get("memory_total_mb", 0)
        print(f"  GPU:     {dev.get('name', '?')} "
              f"({dev.get('compute_capability', '?')}) — "
              f"{used} / {total} MiB used")
    rt = report["embedded_runtime"]
    print(f"  Runtime: {'installed' if rt.get('marker', {}).get('cuda_available') else 'missing or partial'}")
    n_orphans = len(orphans.get("orphans", []))
    print(f"  Orphans: {n_orphans} (subprocesses outliving parent)")
    print(f"  Kill ladder: stage_that_reaped="
          f"{kill_ladder.get('stage_that_reaped', '?')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
