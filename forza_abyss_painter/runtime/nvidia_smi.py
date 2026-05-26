"""Pre-launch GPU free-VRAM probe via `nvidia-smi`.

## Why this exists

The settings panel's `vram_budget_gib` is what the USER says is safe to
spend on the run. But that's a static knob — it doesn't know whether
FH6 is open (eating ~6 GB), Chrome is hogging 2 GB, or another tool
left a process leaking VRAM. By the time the engine tries to allocate
a `(K, B, B, 3)` crop tensor for the bbox scorer, the user's "8 GB
budget" might only have 3 GB actually free, and the run OOMs.

This module reads the **actual free VRAM right now** so the pre-Start
guard can compare what the engine will need against what the GPU
actually has free, and warn the user before they launch a doomed run.
Same data nvidia-smi shows; just programmatic.

## Output

`ProbeResult` carries success/failure both ways:

  * `available=True, free_mib=…, total_mib=…, name=…` — typical case
  * `available=False, reason="nvidia-smi not on PATH"` — no NVIDIA driver
  * `available=False, reason="timeout"` — process didn't finish in 5s
  * `available=False, reason="parse error: …"` — output didn't match

Callers MUST handle `available=False` gracefully — non-NVIDIA boxes,
WSL without GPU passthrough, and machines without the CUDA driver
installed all hit that branch. The GUI surfaces it as "free VRAM:
unknown" instead of crashing.

## Caching

Probes are cached for `_CACHE_TTL_SECONDS` (default 5s) so the status-
bar refresh ticker can call `probe_free_vram` at 1 Hz without
spawning a subprocess every tick. Call `clear_cache()` for the
pre-Start guard which wants a fresh read.
"""
from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Optional


# nvidia-smi sometimes hangs on broken driver installs — bound the wait.
# 5 seconds is generous; a healthy probe returns in well under 200ms.
_PROBE_TIMEOUT_SECONDS = 5.0

# How long to trust a cached result before re-probing. Tuned for the
# status-bar refresh use case: we don't want to spawn nvidia-smi every
# tick, but stale-by-more-than-5s is misleading if FH6 just opened.
_CACHE_TTL_SECONDS = 5.0


@dataclass(frozen=True)
class ProbeResult:
    """One snapshot of GPU state. Immutable so callers can stash it
    in a status-bar widget without worrying about mutation."""

    available: bool

    # Free / total memory on the FIRST visible GPU. Multi-GPU boxes
    # report just GPU 0 — the engine pins to a single device anyway.
    # Both values are in MiB to match nvidia-smi's native unit.
    free_mib: Optional[int] = None
    total_mib: Optional[int] = None

    # Human-readable GPU name (e.g. "NVIDIA RTX PRO 6000 Blackwell"),
    # driver version (e.g. "555.85"). Both optional so non-NVIDIA
    # paths can still return a useful object.
    name: Optional[str] = None
    driver_version: Optional[str] = None

    # When available=False, this carries the user-readable reason
    # the status bar / log can display. Empty string when available.
    reason: str = ""

    # Wall-clock timestamp the probe completed at. Used for cache TTL
    # and for "last refreshed X seconds ago" status text.
    probed_at: float = 0.0

    @property
    def free_gib(self) -> Optional[float]:
        """Convenience: free VRAM in GiB. None when unavailable.
        Uses 1024-based division to match nvidia-smi's MiB."""
        if self.free_mib is None:
            return None
        return self.free_mib / 1024.0

    @property
    def total_gib(self) -> Optional[float]:
        if self.total_mib is None:
            return None
        return self.total_mib / 1024.0


# Module-level cache. Single result is plenty — we only ever probe
# GPU 0 — and the status bar reads it from one place.
_cached_result: Optional[ProbeResult] = None


def clear_cache() -> None:
    """Force the next probe_free_vram() call to actually run nvidia-smi.
    Use this from the pre-Start guard where stale data is a real risk
    (the user may have just closed FH6, freeing 6 GB)."""
    global _cached_result
    _cached_result = None


def probe_free_vram(*, force: bool = False) -> ProbeResult:
    """Return current GPU state, using a short-lived cache when possible.

    `force=True` bypasses the cache (used by the pre-Start guard).
    The status-bar tick should leave it False so we don't spawn a
    subprocess every second."""
    global _cached_result
    now = time.monotonic()
    if not force and _cached_result is not None:
        if now - _cached_result.probed_at < _CACHE_TTL_SECONDS:
            return _cached_result

    result = _run_probe(now)
    _cached_result = result
    return result


def _run_probe(timestamp: float) -> ProbeResult:
    """Actually invoke nvidia-smi. Returns a ProbeResult either way —
    the caller never has to handle exceptions from this function."""
    cmd = [
        "nvidia-smi",
        "--query-gpu=memory.free,memory.total,name,driver_version",
        "--format=csv,noheader,nounits",
        "--id=0",
    ]
    # On Windows, suppress the cmd console flash. Same flag we use in
    # torch_installer.py — the constant is defined inline rather than
    # imported to keep this module self-contained.
    creation_flags = 0x08000000 if sys.platform == "win32" else 0
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True, text=True,
            timeout=_PROBE_TIMEOUT_SECONDS,
            creationflags=creation_flags,
        )
    except FileNotFoundError:
        return ProbeResult(
            available=False,
            reason="nvidia-smi not on PATH "
                   "(no NVIDIA driver, or running without GPU passthrough)",
            probed_at=timestamp,
        )
    except subprocess.TimeoutExpired:
        return ProbeResult(
            available=False,
            reason=f"nvidia-smi timed out after "
                   f"{_PROBE_TIMEOUT_SECONDS:.0f}s "
                   "(driver may be in a bad state — try `nvidia-smi` manually)",
            probed_at=timestamp,
        )
    except OSError as exc:
        return ProbeResult(
            available=False,
            reason=f"failed to invoke nvidia-smi: {type(exc).__name__}: {exc}",
            probed_at=timestamp,
        )

    if proc.returncode != 0:
        return ProbeResult(
            available=False,
            reason=(
                f"nvidia-smi exited rc={proc.returncode}: "
                f"{(proc.stderr or proc.stdout or '').strip()[:200]}"
            ),
            probed_at=timestamp,
        )

    return _parse_csv_line(proc.stdout, timestamp)


def _parse_csv_line(stdout: str, timestamp: float) -> ProbeResult:
    """Parse the comma-separated single line nvidia-smi emits.

    Format (per the --query-gpu flags above):
        memory.free, memory.total, name, driver_version
    Example:
        '95234, 102400, NVIDIA RTX PRO 6000 Blackwell, 555.85'

    Values come without unit suffixes due to `--format=...,nounits`.
    Whitespace is tolerated. Returns a failure ProbeResult rather
    than raising — the caller wants a uniform return type."""
    first_line = stdout.strip().splitlines()[0] if stdout.strip() else ""
    if not first_line:
        return ProbeResult(
            available=False,
            reason="nvidia-smi returned empty output",
            probed_at=timestamp,
        )
    parts = [p.strip() for p in first_line.split(",")]
    if len(parts) < 4:
        return ProbeResult(
            available=False,
            reason=(
                f"unexpected nvidia-smi output (got {len(parts)} "
                f"fields, expected 4): {first_line[:200]!r}"
            ),
            probed_at=timestamp,
        )
    try:
        free_mib = int(parts[0])
        total_mib = int(parts[1])
    except ValueError as exc:
        return ProbeResult(
            available=False,
            reason=f"could not parse memory values from "
                   f"nvidia-smi output {first_line[:200]!r}: {exc}",
            probed_at=timestamp,
        )
    return ProbeResult(
        available=True,
        free_mib=free_mib,
        total_mib=total_mib,
        name=parts[2] or None,
        driver_version=parts[3] or None,
        probed_at=timestamp,
    )
