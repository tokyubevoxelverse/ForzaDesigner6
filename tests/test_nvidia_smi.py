"""Tests for the nvidia-smi free-VRAM probe.

Heavily mocked — we can't actually run nvidia-smi on the CI box (and
the dev Mac doesn't have it). Each test pins the subprocess.run call
to a specific behavior and verifies the ProbeResult shape.
"""
from __future__ import annotations

import subprocess
from unittest.mock import patch, MagicMock

import pytest

from forza_abyss_painter.runtime import nvidia_smi
from forza_abyss_painter.runtime.nvidia_smi import (
    ProbeResult, clear_cache, probe_free_vram,
)


@pytest.fixture(autouse=True)
def _isolate_cache():
    """Every test starts with a clean cache. Otherwise tests order
    would matter (the first probe would populate a 5s cache that
    later tests would see)."""
    clear_cache()
    yield
    clear_cache()


def _fake_completed(stdout: str = "", returncode: int = 0, stderr: str = ""):
    proc = MagicMock(spec=subprocess.CompletedProcess)
    proc.stdout = stdout
    proc.stderr = stderr
    proc.returncode = returncode
    return proc


# ========================================================== happy path


def test_typical_blackwell_response_parses():
    """Real-world output from the user's RTX PRO 6000 Blackwell: the
    102 GB card with ~95 GB free. Verify every field round-trips."""
    sample = "95234, 102400, NVIDIA RTX PRO 6000 Blackwell, 555.85\n"
    with patch.object(subprocess, "run", return_value=_fake_completed(sample)):
        r = probe_free_vram()
    assert r.available
    assert r.free_mib == 95234
    assert r.total_mib == 102400
    assert r.name == "NVIDIA RTX PRO 6000 Blackwell"
    assert r.driver_version == "555.85"
    # Convenience accessors
    assert abs(r.free_gib - 95234 / 1024) < 1e-6
    assert abs(r.total_gib - 102400 / 1024) < 1e-6


def test_handles_whitespace_in_csv_fields():
    """nvidia-smi pads values with spaces. Real output has them."""
    sample = "  7891 ,  16384 ,  NVIDIA GeForce RTX 4090 ,  556.12  \n"
    with patch.object(subprocess, "run", return_value=_fake_completed(sample)):
        r = probe_free_vram()
    assert r.available
    assert r.free_mib == 7891
    assert r.total_mib == 16384
    assert r.name == "NVIDIA GeForce RTX 4090"
    assert r.driver_version == "556.12"


# ========================================================== failure paths


def test_filenotfound_returns_unavailable_with_clear_reason():
    """The common 'no NVIDIA driver installed' case — must surface a
    user-readable reason so the status bar can show it."""
    with patch.object(subprocess, "run", side_effect=FileNotFoundError):
        r = probe_free_vram()
    assert not r.available
    assert "nvidia-smi not on PATH" in r.reason
    assert r.free_mib is None


def test_timeout_returns_unavailable():
    """Bad-driver hangs are real. Must time out gracefully."""
    with patch.object(subprocess, "run",
                       side_effect=subprocess.TimeoutExpired("nvidia-smi", 5)):
        r = probe_free_vram()
    assert not r.available
    assert "timed out" in r.reason


def test_nonzero_exit_surfaces_stderr():
    """Driver-installed-but-broken case: nvidia-smi exits 1 with a
    diagnostic in stderr. We want that diagnostic in the UI."""
    with patch.object(subprocess, "run",
                       return_value=_fake_completed("", returncode=1,
                                                    stderr="NVML: driver/library mismatch")):
        r = probe_free_vram()
    assert not r.available
    assert "rc=1" in r.reason
    assert "driver/library mismatch" in r.reason


def test_empty_output_returns_unavailable():
    """Some edge cases (early-startup nvidia-smi races) return empty.
    Don't crash — surface as parse failure."""
    with patch.object(subprocess, "run", return_value=_fake_completed("")):
        r = probe_free_vram()
    assert not r.available
    assert "empty output" in r.reason


def test_malformed_csv_returns_unavailable():
    """Defensive: nvidia-smi version drift could change column count.
    Don't index past the bounds — surface as parse failure."""
    with patch.object(subprocess, "run",
                       return_value=_fake_completed("only,two\n")):
        r = probe_free_vram()
    assert not r.available
    assert "unexpected nvidia-smi output" in r.reason


def test_non_integer_memory_returns_unavailable():
    """Defensive: if nvidia-smi ever emits something weird in the
    memory column, fail gracefully rather than ValueError-ing up to
    the GUI."""
    with patch.object(subprocess, "run",
                       return_value=_fake_completed(
                           "n/a, n/a, NVIDIA, 555\n")):
        r = probe_free_vram()
    assert not r.available
    assert "could not parse memory values" in r.reason


def test_os_error_returns_unavailable():
    """Defensive against PermissionError / OSError variants we don't
    explicitly catch — they should still surface gracefully."""
    with patch.object(subprocess, "run",
                       side_effect=PermissionError("denied")):
        r = probe_free_vram()
    assert not r.available
    assert "failed to invoke nvidia-smi" in r.reason


# ========================================================== caching


def test_repeated_probes_use_cache():
    """The status-bar refresh ticks every second; we must NOT spawn
    nvidia-smi every tick. Verify the second call doesn't run."""
    sample = "1000, 16384, GPU, 555.85\n"
    with patch.object(subprocess, "run",
                       return_value=_fake_completed(sample)) as mock_run:
        r1 = probe_free_vram()
        r2 = probe_free_vram()
        r3 = probe_free_vram()
    assert mock_run.call_count == 1, (
        f"expected 1 subprocess call, got {mock_run.call_count}"
    )
    assert r1.free_mib == r2.free_mib == r3.free_mib == 1000


def test_force_bypasses_cache():
    """The pre-Start guard wants a fresh read even if the status bar
    just polled — `force=True` must spawn nvidia-smi every time."""
    sample = "1000, 16384, GPU, 555.85\n"
    with patch.object(subprocess, "run",
                       return_value=_fake_completed(sample)) as mock_run:
        probe_free_vram()
        probe_free_vram(force=True)
        probe_free_vram(force=True)
    assert mock_run.call_count == 3


def test_clear_cache_forces_next_call():
    """Manual cache invalidation. Used when we know state changed
    (e.g. user closed FH6 and we want to re-probe)."""
    sample = "1000, 16384, GPU, 555.85\n"
    with patch.object(subprocess, "run",
                       return_value=_fake_completed(sample)) as mock_run:
        probe_free_vram()
        clear_cache()
        probe_free_vram()
    assert mock_run.call_count == 2


def test_cache_expires_after_ttl(monkeypatch):
    """The cache TTL must actually elapse — patch time.monotonic to
    simulate the 5s ageing without sleeping in CI."""
    sample = "1000, 16384, GPU, 555.85\n"
    # probe_free_vram() calls time.monotonic() exactly once per
    # invocation (single read at the top of the function).
    timeline = iter([
        100.0,   # first probe — populates cache at t=100
        102.0,   # second probe (cache hit, t=102 < 100+5)
        110.0,   # third probe (cache miss, t=110 > 100+5)
    ])
    monkeypatch.setattr(nvidia_smi.time, "monotonic", lambda: next(timeline))
    with patch.object(subprocess, "run",
                       return_value=_fake_completed(sample)) as mock_run:
        probe_free_vram()       # populates
        probe_free_vram()       # cache hit
        probe_free_vram()       # cache miss — re-probes
    assert mock_run.call_count == 2


# ========================================================== ProbeResult contract


def test_probe_result_is_frozen():
    """Status bar widgets stash this — accidental mutation would be
    a bug. Verify the dataclass actually rejects assignment."""
    r = ProbeResult(available=True, free_mib=1000, total_mib=2000)
    with pytest.raises(Exception):
        r.free_mib = 500   # type: ignore[misc]


def test_probe_result_free_gib_none_when_unavailable():
    """Convenience accessors must handle the failure path."""
    r = ProbeResult(available=False, reason="x")
    assert r.free_gib is None
    assert r.total_gib is None
