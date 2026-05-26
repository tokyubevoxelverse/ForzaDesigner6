# Cursor diagnostic brief — GPU runtime + process-management probe

> **Audience:** Cursor agent (or any reverse-engineer) running on
> QUASAR after a GPU run goes sideways. Triggered when the EXE's
> `Tools → Save diagnostics zip…` flow isn't enough — for example,
> the user already restarted the PC (clearing all logs) or the
> failure is something the EXE itself couldn't capture (a
> stuck-process condition where the EXE crashed before writing
> the session_end event).
>
> **Goal:** produce a single JSON report Cursor can hand back to me
> (Claude, Mac side) that names exactly what state the machine + the
> installed runtime + the GPU were in. With it, I can post-mortem
> without needing to be on the box.

## Why this exists

The QUASAR test session on 2026-05-25 hit a process-management
failure mode: the GPU shape-gen subprocess ran into a CUDA kernel
that the driver wouldn't yield, terminate() didn't reap it, and the
user had to **restart their PC** to recover the GPU. The fix landed
in the EXE (escalating kill ladder + CTRL_BREAK_EVENT handler in
torch_runner) but if it still happens we need a way to see:

  1. What state was the GPU in when the failure occurred?
  2. What did the embedded Python runtime actually have installed?
  3. Were there orphan Python processes outliving the EXE?
  4. Did our escalation ladder behave on this specific Windows build?

The probe answers all four in one script run.

## How to run

On QUASAR with no other user-facing prerequisites:

```powershell
cd \\QUASAR\ContentCreation\ForzaAbyssPainter_build\diagnostics
py -3 probe_gpu_and_runtime.py --output cursor_gpu_probe.json
```

Output lands at `diagnostics\cursor_gpu_probe.json`. Email or paste
the file contents into the chat with me for triage.

The script is **read-only** — it shells out to `nvidia-smi`, walks
`%LOCALAPPDATA%\ForzaAbyssPainter\runtime\` (no writes), enumerates
processes by name, runs the embedded Python with a short
`torch.zeros(1, device='cuda')` probe. No changes to the runtime,
no extra installs, no impact on a future `Tools → Install GPU
runtime…` retry.

## What the probe checks

### Section A: Host environment

  - Windows version (build number, edition)
  - User locale + time zone (so we know what timestamps in logs mean)
  - CPU info, total RAM, free RAM
  - Whether FH6 is currently running (process name match)
  - Whether the EXE is currently running

### Section B: GPU state (nvidia-smi)

  - Driver version (must be ≥ 525.x for cu128 wheels per the install
    guide)
  - Per-GPU: name, compute_capability, memory_total, memory_used,
    memory_free
  - List of currently-running processes using the GPU + per-process
    VRAM usage (looking for orphan ForzaAbyssPainter\python.exe
    processes that survived the EXE crash)

### Section C: Embedded runtime state

  - `installed.json` contents (parsed RuntimeInfo)
  - `python311\` directory tree summary (file count, total size)
  - `site-packages\torch\` version pin + presence
  - `site-packages\forza_abyss_painter\` presence (was the
    copy_package step's output bundled correctly?)
  - Real-kernel probe: invoke the embedded python with
    `torch.zeros(1, device='cuda').add_(1)` and capture the result
    (success / `no kernel image available` / OOM / driver error)

### Section D: Orphan process detection

  - All `python.exe` / `python311.exe` processes on the system
  - For each: parent process (parent EXE alive?), command line
    (looking for `--config` pointing at runtime cfgs),
    creation_time (anything > 10 minutes old is likely orphan), and
    open file handles to the runtime dir
  - Identifies orphans that survived the parent EXE crash → these
    are the "unkillable" processes that hold GPU memory until
    PC restart

### Section E: Kill-ladder self-test

  - Spawns a deliberately stuck Python subprocess (catches SIGINT,
    ignores SIGTERM for 30s, etc.) inside a fresh CREATE_NEW_PROCESS_GROUP
  - Runs the same escalating ladder the EXE uses against it
  - Reports which stage actually reaped the process + total elapsed
    time
  - This validates that the EXE's process-management implementation
    isn't broken by a Windows update or Defender quirk

### Section F: Recent logs

  - List of `%LOCALAPPDATA%\ForzaAbyssPainter\logs\` files with sizes +
    mtimes. Doesn't dump the contents (those go in the diagnostics
    bundle separately), but tells me whether they exist + were
    written recently.

## Output schema

```json
{
  "captured_at_utc": "2026-05-26T03:12:54Z",
  "probe_version": "1.0",
  "platform": {
    "os_version": "Windows 10 22H2 build 19045.4170",
    "cpu": "Intel(R) Core(TM) i9-14900K",
    "ram_total_gb": 64.0,
    "ram_free_gb": 28.1,
    "fh6_running": false,
    "exe_running": false
  },
  "gpu": {
    "nvidia_smi_available": true,
    "driver_version": "555.42.02",
    "devices": [
      {
        "name": "NVIDIA GeForce RTX 5090",
        "compute_capability": "12.0",
        "memory_total_mb": 32768,
        "memory_used_mb": 8456,
        "memory_free_mb": 24312
      }
    ],
    "processes_using_gpu": [
      {"pid": 6789, "name": "python.exe", "memory_mb": 3200}
    ]
  },
  "embedded_runtime": {
    "marker_present": true,
    "marker": {"torch_version": "2.7.0+cu128", "cuda_available": true, ...},
    "runtime_dir_size_mb": 4012.3,
    "torch_dist_info_version": "2.7.0+cu128",
    "forza_abyss_painter_present": true,
    "real_kernel_probe": {"ok": true, "elapsed_ms": 124}
  },
  "orphan_processes": [
    {
      "pid": 6789, "parent_alive": false,
      "cmdline": "...\\python.exe -m forza_abyss_painter.runtime.torch_runner --config ...",
      "age_seconds": 1834
    }
  ],
  "kill_ladder_self_test": {
    "stage_that_reaped": "kill",
    "graceful_seconds": 8.1,
    "terminate_seconds": 4.0,
    "kill_seconds": 0.2,
    "total_seconds": 12.3
  },
  "log_files": [
    {"name": "gpu-2026-05-25-14h-main.log", "size_kb": 87, "mtime_utc": "..."}
  ]
}
```

## What I'll do with the report

1. If orphans are present + parent EXE is dead: the kill ladder
   landed but didn't actually reap. Surface as a Windows-version-
   specific bug to fix in the EXE.
2. If real_kernel_probe.ok=False with `no kernel image`: cu128 wheels
   weren't installed correctly. Fix is to re-run install.
3. If GPU memory_used >> what we'd expect for the EXE workload: some
   other app is hogging VRAM and chunked-K isn't enough.
4. If kill_ladder_self_test reaches "never reaped": process-management
   broken on this Windows build; surface upstream.

## Hands-off recovery (when in doubt)

If the diagnostic probe itself can't run (Python not available,
nvidia-smi missing, etc.), the user can still recover manually:

```powershell
# 1. Find + kill orphan ForzaAbyssPainter python.exe processes
Get-Process python | Where-Object {$_.MainModule.FileName -like "*ForzaAbyssPainter*"} | Stop-Process -Force

# 2. If GPU still feels stuck (nvidia-smi shows allocated memory with no process):
#    Restart the display driver (Windows reaches for this on Ctrl+Win+Shift+B)
#    OR restart nvidia-smi: nvidia-smi --gpu-reset (admin)
#    OR worst case: restart the EXE — that should NOT require PC restart anymore

# 3. Nuke the runtime if install state is suspect:
Remove-Item -Recurse -Force "$env:LOCALAPPDATA\ForzaAbyssPainter\runtime"
```

PC restart should ONLY be needed if all three above fail. If you hit
that case, the probe script is the priority — run it BEFORE the
restart so we capture the failure state.
