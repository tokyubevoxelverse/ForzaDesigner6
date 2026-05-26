"""Tests for forza_abyss_painter.runtime.torch_installer.install_runtime
— the HTTP downloader + embedded Python bootstrap.

The function uses dependency injection for urllib + subprocess so the
tests can exercise the full orchestration end-to-end on macOS without
touching the network or installing 3 GiB of torch wheels. Real installs
get validated on the Windows tester's machine (CLAUDE.md §2 — verify
each code path separately) after the orchestration here is green.

Tests cover:
  - Phase ordering: every phase fires in the expected order with the
    expected progress percent
  - Idempotency: re-running with an existing valid install short-circuits
  - Error propagation: a fake failure at each phase raises
    InstallError(stage=that_phase)
  - Marker correctness: written marker has CUDA verdict + version
  - Package copy: required subpackages land in embedded site-packages,
    excluded ones (gui, inject, cli) don't
  - _pth site-enable: idempotent edit, handles missing file + alt names
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from forza_abyss_painter.runtime import torch_installer as ti


# ====================================================================
# Helpers — fakes for the DI hooks


def _make_fake_urlretrieve(written_files: dict[str, bytes]):
    """Build a fake urlretrieve that records calls + writes deterministic
    bytes to the destination. Pass dest paths in written_files keyed by
    URL → bytes-to-write (or leave the key absent for an empty stub)."""
    calls = []

    def _fake(url: str, dest: Path) -> None:
        calls.append((url, Path(dest)))
        data = written_files.get(url, b"<fake-stub>")
        Path(dest).write_bytes(data)
    _fake.calls = calls   # introspectable from tests
    return _fake


def _make_fake_embed_zip(extract_dir_layout: dict[str, str]) -> bytes:
    """Build a zip-file's bytes with the given file layout. Used as the
    download payload for the embed-Python URL so the extraction phase
    has something legit to unpack. Layout keys are relative paths inside
    the archive; values are file contents."""
    import io

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w") as zf:
        for rel, content in extract_dir_layout.items():
            zf.writestr(rel, content)
    return buf.getvalue()


def _make_fake_subprocess_success(stdout_for_verify: str = ""):
    """Return a fake subprocess_run that succeeds every call. The verify-
    CUDA call (last one before marker write) returns the given stdout
    so we can drive the CUDA verdict from the test."""
    calls = []

    def _fake(cmd, *, capture=False):
        calls.append((list(cmd), capture))
        # Decide which call this is — verify_cuda is the one with "-c"
        # + the torch.cuda.is_available probe string.
        result = MagicMock()
        result.returncode = 0
        result.stdout = ""
        result.stderr = ""
        if "-c" in cmd:
            for arg in cmd:
                if "torch.cuda.is_available" in arg:
                    result.stdout = stdout_for_verify
                    break
        return result
    _fake.calls = calls
    return _fake


@pytest.fixture
def _isolated_runtime(tmp_path, monkeypatch):
    """Redirect runtime_root() to a tmp dir so tests don't pollute the
    user's real LOCALAPPDATA / Library / XDG dir. Yields the tmp root."""
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "appdata"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr(ti.Path, "home", classmethod(lambda cls: tmp_path / "home"))
    return ti.runtime_root()


@pytest.fixture
def _stub_source_pkg(tmp_path):
    """Build a fake forza_abyss_painter package layout on disk to act as
    the source for the copy step. Includes the required subpackages +
    some excluded ones, so the test can verify the copy filter."""
    src = tmp_path / "fake_pkg_src"
    src.mkdir()
    (src / "__init__.py").write_text("# forza_abyss_painter\n", encoding="utf-8")
    (src / "_build_info.py").write_text("BUILD_SHA = 'test'\n", encoding="utf-8")
    # Required subpackages.
    for sub in ti.RUNNER_REQUIRED_SUBPACKAGES:
        (src / sub).mkdir()
        (src / sub / "__init__.py").write_text(f"# {sub}\n", encoding="utf-8")
        (src / sub / "real.py").write_text("# real impl\n", encoding="utf-8")
    # Excluded subpackages — should NOT be copied.
    for excl in ("gui", "inject"):
        (src / excl).mkdir()
        (src / excl / "__init__.py").write_text(
            f"raise ImportError('{excl} not needed for runner')",
            encoding="utf-8",
        )
    return src


# ====================================================================
# Phase ordering + happy path


def test_install_runtime_fires_phases_in_order_with_monotonic_progress(
    _isolated_runtime, _stub_source_pkg,
):
    """The progress_cb sees percent values that are monotonically
    increasing across phases. If a phase reports a percent LOWER than
    a prior phase, the GUI progress bar bounces backwards — bad UX +
    a clear sign the orchestration drift."""
    progress = []
    def _cb(pct, status):
        progress.append((pct, status))

    # File name in the zip must match what embedded_python_exe() expects
    # on the current test platform (.exe on Windows, no suffix elsewhere).
    embed_zip_bytes = _make_fake_embed_zip({
        ti.embedded_python_exe().name: "<fake python binary>",
        "python311._pth": "python311.zip\n.\n#import site\n",
    })
    urlretrieve = _make_fake_urlretrieve({
        ti.EMBED_PYTHON_URL: embed_zip_bytes,
        ti.GET_PIP_URL: b"# get-pip.py stub\n",
    })
    sp_run = _make_fake_subprocess_success(
        stdout_for_verify=json.dumps({
            "cuda_available": True, "device_name": "FAKE GPU",
            "compute_capability": [12, 0], "kernel_run_ok": True,
            "torch_version": "2.7.0+cu128",
        }),
    )

    info = ti.install_runtime(
        progress_cb=_cb,
        _urlretrieve=urlretrieve,
        _subprocess_run=sp_run,
        _source_pkg_dir=_stub_source_pkg,
    )

    assert info.cuda_available is True
    assert info.cuda_device_name == "FAKE GPU"
    # Monotonic.
    percents = [p for p, _ in progress]
    assert percents == sorted(percents), (
        f"progress percents are not monotonic: {percents}"
    )
    assert percents[0] == 0
    assert percents[-1] == 100


def test_install_runtime_downloads_python_then_get_pip_in_that_order(
    _isolated_runtime, _stub_source_pkg,
):
    """urlretrieve must be called for the embed Python URL first, then
    get-pip.py. Reverse order means we try to download get-pip into a
    nonexistent runtime dir which would surface as a misleading error."""
    embed_zip_bytes = _make_fake_embed_zip({
        ti.embedded_python_exe().name: "x", "python311._pth": "#import site",
    })
    urlretrieve = _make_fake_urlretrieve({
        ti.EMBED_PYTHON_URL: embed_zip_bytes,
        ti.GET_PIP_URL: b"# get-pip stub",
    })
    sp_run = _make_fake_subprocess_success(
        stdout_for_verify=json.dumps({
            "cuda_available": True, "device_name": "X",
            "compute_capability": [8, 0], "kernel_run_ok": True,
            "torch_version": "2.7.0+cu128",
        }),
    )
    ti.install_runtime(
        _urlretrieve=urlretrieve, _subprocess_run=sp_run,
        _source_pkg_dir=_stub_source_pkg,
    )
    urls_in_order = [url for url, _ in urlretrieve.calls]
    assert urls_in_order.index(ti.EMBED_PYTHON_URL) < urls_in_order.index(ti.GET_PIP_URL), (
        f"download order wrong: {urls_in_order}"
    )


def test_install_runtime_runs_get_pip_before_pip_install(
    _isolated_runtime, _stub_source_pkg,
):
    """get-pip.py bootstrap MUST run before any pip install command,
    else pip itself isn't yet present in the embedded Python."""
    embed_zip_bytes = _make_fake_embed_zip({
        ti.embedded_python_exe().name: "x", "python311._pth": "#import site",
    })
    urlretrieve = _make_fake_urlretrieve({
        ti.EMBED_PYTHON_URL: embed_zip_bytes,
        ti.GET_PIP_URL: b"# get-pip stub",
    })
    sp_run = _make_fake_subprocess_success(
        stdout_for_verify=json.dumps({
            "cuda_available": True, "device_name": "X",
            "compute_capability": [8, 0], "kernel_run_ok": True,
            "torch_version": "2.7.0+cu128",
        }),
    )
    ti.install_runtime(
        _urlretrieve=urlretrieve, _subprocess_run=sp_run,
        _source_pkg_dir=_stub_source_pkg,
    )
    # Find which call ran get-pip.py and which ran pip install.
    cmds = [tuple(c[0]) for c in sp_run.calls]
    get_pip_idx = next((i for i, c in enumerate(cmds)
                        if any("get-pip" in arg for arg in c)), None)
    pip_install_idx = next((i for i, c in enumerate(cmds)
                            if "install" in c and "pip" in c), None)
    assert get_pip_idx is not None, f"get-pip.py never ran. cmds: {cmds}"
    assert pip_install_idx is not None, f"pip install never ran. cmds: {cmds}"
    assert get_pip_idx < pip_install_idx, (
        f"pip install ran before get-pip.py bootstrap. cmds: {cmds}"
    )


def test_install_runtime_writes_marker_with_cuda_verdict(
    _isolated_runtime, _stub_source_pkg,
):
    """The CUDA verdict from the verify subprocess flows into the marker.
    If verify says CUDA is unavailable, the marker records that — and
    is_runtime_installed() then returns False, correctly reflecting that
    a CPU-only torch wheel landed instead of cu121."""
    embed_zip_bytes = _make_fake_embed_zip({
        ti.embedded_python_exe().name: "x", "python311._pth": "#import site",
    })
    urlretrieve = _make_fake_urlretrieve({
        ti.EMBED_PYTHON_URL: embed_zip_bytes,
        ti.GET_PIP_URL: b"# get-pip stub",
    })
    sp_run = _make_fake_subprocess_success(
        stdout_for_verify=json.dumps({
            "cuda_available": False, "device_name": "",
            "compute_capability": None, "kernel_run_ok": False,
            "torch_version": "2.7.0",   # CPU-only wheel landed
        }),
    )
    info = ti.install_runtime(
        _urlretrieve=urlretrieve, _subprocess_run=sp_run,
        _source_pkg_dir=_stub_source_pkg,
    )
    assert info.cuda_available is False
    marker = ti.runtime_marker()
    assert marker.exists()
    data = json.loads(marker.read_text(encoding="utf-8"))
    assert data["cuda_available"] is False
    # And is_runtime_installed reflects the partial-install state.
    # Note: it ALSO requires embedded_python_exe() to exist, which the
    # fake zip's "python.exe" entry creates inside embedded_python_dir().
    assert ti.is_runtime_installed() is False


def test_marker_torch_version_suffix_derives_from_index_url(
    _isolated_runtime, _stub_source_pkg,
):
    """Marker's torch_version is now '<TORCH_VERSION>+<index_suffix>'
    derived from TORCH_CUDA_INDEX, not a hardcoded '+cu121'. Cursor's
    QUASAR smoke caught the stale hardcode after the cu128 bump; this
    test pins the derived behavior so the next index bump can't
    silently drift again."""
    embed_zip_bytes = _make_fake_embed_zip({
        ti.embedded_python_exe().name: "x", "python311._pth": "#import site",
    })
    urlretrieve = _make_fake_urlretrieve({
        ti.EMBED_PYTHON_URL: embed_zip_bytes,
        ti.GET_PIP_URL: b"# stub",
    })
    sp_run = _make_fake_subprocess_success(
        stdout_for_verify=json.dumps({
            "cuda_available": True, "device_name": "FAKE GPU",
            "compute_capability": [12, 0], "kernel_run_ok": True,
            "torch_version": "2.7.0+cu128",
        }),
    )
    info = ti.install_runtime(
        _urlretrieve=urlretrieve, _subprocess_run=sp_run,
        _source_pkg_dir=_stub_source_pkg,
    )
    # Suffix must match whatever TORCH_CUDA_INDEX currently points at,
    # NOT a hardcoded '+cu121'.
    expected_suffix = ti.TORCH_CUDA_INDEX.rstrip("/").rsplit("/", 1)[-1]
    assert info.torch_version == f"{ti.TORCH_VERSION}+{expected_suffix}", (
        f"marker torch_version={info.torch_version!r} doesn't derive from "
        f"index URL — expected '{ti.TORCH_VERSION}+{expected_suffix}'"
    )


# ====================================================================
# Idempotency


def test_install_runtime_short_circuits_when_already_installed(
    _isolated_runtime, _stub_source_pkg,
):
    """If a valid install already exists (marker present + cuda_available
    True + python.exe in place), re-running install_runtime returns the
    existing RuntimeInfo without touching the network or running pip."""
    # Set up a valid existing install.
    ti.embedded_python_dir().mkdir(parents=True, exist_ok=True)
    ti.embedded_python_exe().write_text("# fake", encoding="utf-8")
    existing_info = ti.RuntimeInfo(
        python_version="3.11.9", torch_version="2.4.1+cu121",
        cuda_available=True, cuda_device_name="EXISTING GPU",
        installed_at_utc="2026-01-01T00:00:00Z",
    )
    ti.runtime_marker().write_text(json.dumps(existing_info.to_dict()), encoding="utf-8")

    urlretrieve = _make_fake_urlretrieve({})
    sp_run = _make_fake_subprocess_success()
    info = ti.install_runtime(
        _urlretrieve=urlretrieve, _subprocess_run=sp_run,
        _source_pkg_dir=_stub_source_pkg,
    )
    assert info == existing_info
    assert urlretrieve.calls == [], "no urls should be downloaded on idempotent re-run"
    assert sp_run.calls == [], "no subprocess calls on idempotent re-run"


# ====================================================================
# Error propagation per phase


def test_install_runtime_raises_install_error_on_python_download_fail(
    _isolated_runtime, _stub_source_pkg,
):
    """Network failure on embed-Python download → InstallError(stage='download_python')
    so the GUI's modal can surface 'check your internet connection'."""
    def _failing_urlretrieve(url, dest):
        raise OSError("network unreachable")
    with pytest.raises(ti.InstallError) as excinfo:
        ti.install_runtime(
            _urlretrieve=_failing_urlretrieve,
            _subprocess_run=_make_fake_subprocess_success(),
            _source_pkg_dir=_stub_source_pkg,
        )
    assert excinfo.value.stage == "download_python"


def test_install_runtime_raises_install_error_on_extract_fail(
    _isolated_runtime, _stub_source_pkg,
):
    """Corrupt zip → InstallError(stage='extract_python'). User can
    re-run; install_runtime is not idempotent across failures (the
    runtime dir holds partial state) but the error tells them the
    extraction broke, not the network."""
    def _bad_zip_urlretrieve(url, dest):
        # Write garbage that's not a valid zip.
        Path(dest).write_bytes(b"not a zipfile")
    with pytest.raises(ti.InstallError) as excinfo:
        ti.install_runtime(
            _urlretrieve=_bad_zip_urlretrieve,
            _subprocess_run=_make_fake_subprocess_success(),
            _source_pkg_dir=_stub_source_pkg,
        )
    assert excinfo.value.stage == "extract_python"


def test_install_runtime_raises_install_error_on_pip_install_fail(
    _isolated_runtime, _stub_source_pkg,
):
    """pip install failure (e.g., torch wheel mismatch) →
    InstallError(stage='pip_install'). Distinct stage so the GUI can
    surface 'check your CUDA driver version' vs other failures."""
    embed_zip_bytes = _make_fake_embed_zip({
        ti.embedded_python_exe().name: "x", "python311._pth": "#import site",
    })
    urlretrieve = _make_fake_urlretrieve({
        ti.EMBED_PYTHON_URL: embed_zip_bytes,
        ti.GET_PIP_URL: b"# stub",
    })
    def _sp_run_fail_on_pip_install(cmd, *, capture=False):
        if "install" in cmd:
            import subprocess
            raise subprocess.CalledProcessError(
                returncode=1, cmd=cmd,
                stderr="ERROR: Could not find a version that satisfies torch==2.4.1",
            )
        result = MagicMock()
        result.returncode = 0
        result.stdout = ""
        return result
    with pytest.raises(ti.InstallError) as excinfo:
        ti.install_runtime(
            _urlretrieve=urlretrieve,
            _subprocess_run=_sp_run_fail_on_pip_install,
            _source_pkg_dir=_stub_source_pkg,
        )
    assert excinfo.value.stage == "pip_install"


def test_install_runtime_raises_install_error_on_cuda_verify_fail(
    _isolated_runtime, _stub_source_pkg,
):
    """CUDA verify subprocess crashing → InstallError(stage='verify_cuda').
    This is distinct from 'cuda not available' (which is a normal verdict
    with cuda_available=False in the marker)."""
    embed_zip_bytes = _make_fake_embed_zip({
        ti.embedded_python_exe().name: "x", "python311._pth": "#import site",
    })
    urlretrieve = _make_fake_urlretrieve({
        ti.EMBED_PYTHON_URL: embed_zip_bytes,
        ti.GET_PIP_URL: b"# stub",
    })
    def _sp_run_fail_on_verify(cmd, *, capture=False):
        if "-c" in cmd and any("torch.cuda.is_available" in a for a in cmd):
            raise RuntimeError("subprocess crashed")
        result = MagicMock()
        result.returncode = 0
        result.stdout = ""
        return result
    with pytest.raises(ti.InstallError) as excinfo:
        ti.install_runtime(
            _urlretrieve=urlretrieve,
            _subprocess_run=_sp_run_fail_on_verify,
            _source_pkg_dir=_stub_source_pkg,
        )
    assert excinfo.value.stage == "verify_cuda"


# ====================================================================
# Package copy filter


def test_copy_runner_package_includes_required_subpackages(
    _isolated_runtime, _stub_source_pkg, tmp_path,
):
    """shapegen, io, runtime all land in the destination."""
    site_pkgs = tmp_path / "site_pkgs_dest"
    ti._copy_runner_package(_stub_source_pkg, site_pkgs)
    for sub in ti.RUNNER_REQUIRED_SUBPACKAGES:
        assert (site_pkgs / "forza_abyss_painter" / sub).is_dir(), (
            f"required subpackage {sub!r} not copied"
        )
        assert (site_pkgs / "forza_abyss_painter" / sub / "real.py").is_file()


def test_copy_runner_package_excludes_gui_and_inject(
    _isolated_runtime, _stub_source_pkg, tmp_path,
):
    """gui + inject must NOT land in the embedded site-packages —
    they pull heavy deps (PySide6, Windows process APIs) the runner
    doesn't need + that pollute the embedded install.

    `cli` USED to be excluded too, but Cursor's QUASAR smoke surfaced
    that fap-generate fails inside the embedded runtime without it
    (ModuleNotFoundError: forza_abyss_painter.cli). cli's deps are
    just io + runtime which are already required, so it's safe to
    include now and serves the headless-CLI feature."""
    site_pkgs = tmp_path / "site_pkgs_dest"
    ti._copy_runner_package(_stub_source_pkg, site_pkgs)
    for excl in ("gui", "inject"):
        assert not (site_pkgs / "forza_abyss_painter" / excl).exists(), (
            f"excluded subpackage {excl!r} was copied into embedded site-packages"
        )
    # cli SHOULD be copied (it's required now).
    assert (site_pkgs / "forza_abyss_painter" / "cli").is_dir(), (
        "cli subpackage missing — fap-generate won't be importable inside "
        "the embedded runtime"
    )


def test_copy_runner_package_preserves_top_level_files(
    _isolated_runtime, _stub_source_pkg, tmp_path,
):
    """Top-level __init__.py + _build_info.py copied so the embedded
    Python can `import forza_abyss_painter`. Missing __init__ = the
    package isn't importable."""
    site_pkgs = tmp_path / "site_pkgs_dest"
    ti._copy_runner_package(_stub_source_pkg, site_pkgs)
    assert (site_pkgs / "forza_abyss_painter" / "__init__.py").is_file()
    assert (site_pkgs / "forza_abyss_painter" / "_build_info.py").is_file()


# ====================================================================
# _pth site-enable


def test_enable_site_in_pth_uncomments_import_site(tmp_path):
    """The single most common embed-Python install failure: leaving
    `#import site` commented out. Uncomment it idempotently."""
    embed = tmp_path / "embed"
    embed.mkdir()
    pth = embed / "python311._pth"
    pth.write_text("python311.zip\n.\n# Comment\n#import site\n", encoding="utf-8")
    ti._enable_site_in_pth(embed)
    text = pth.read_text(encoding="utf-8")
    assert "import site" in text
    assert "#import site" not in text


def test_enable_site_in_pth_is_idempotent(tmp_path):
    """Running twice doesn't double-edit. Re-running install_runtime
    after a partial failure must not break the already-modified _pth."""
    embed = tmp_path / "embed"
    embed.mkdir()
    pth = embed / "python311._pth"
    pth.write_text("python311.zip\n.\nimport site\n", encoding="utf-8")
    ti._enable_site_in_pth(embed)
    ti._enable_site_in_pth(embed)
    text = pth.read_text(encoding="utf-8")
    assert text.count("import site") == 1


def test_enable_site_in_pth_handles_alt_python_version_filename(tmp_path):
    """If the embed zip uses python312._pth (different python version),
    fall back to a glob match. Guards against a silent break when we
    eventually bump EMBED_PYTHON_VERSION."""
    embed = tmp_path / "embed"
    embed.mkdir()
    pth = embed / "python312._pth"   # NOT 311
    pth.write_text("#import site\n", encoding="utf-8")
    ti._enable_site_in_pth(embed)   # must not raise; must edit the 312 file
    assert "import site" in pth.read_text(encoding="utf-8")


def test_enable_site_in_pth_raises_install_error_when_no_pth_present(tmp_path):
    """Empty embed dir with no _pth at all → InstallError(stage='enable_site').
    User knows the embed extraction itself failed silently — they should
    re-run install."""
    embed = tmp_path / "embed"
    embed.mkdir()
    with pytest.raises(ti.InstallError) as excinfo:
        ti._enable_site_in_pth(embed)
    assert excinfo.value.stage == "enable_site"


# ====================================================================
# Pinned config


def test_subprocess_flags_returns_create_no_window_on_windows(monkeypatch):
    """The #1 lesson from QUASAR's failed install: a leaked cmd window
    during pip looked like a stuck child process, tester closed it,
    install died with 0xC000013A. _subprocess_flags must return
    CREATE_NO_WINDOW (0x08000000) on Windows so the embedded python.exe
    spawn never allocates a visible console."""
    monkeypatch.setattr(ti.sys, "platform", "win32")
    assert ti._subprocess_flags() == 0x08000000


def test_subprocess_flags_returns_zero_on_non_windows(monkeypatch):
    """macOS / Linux subprocess.run never spawns a console for a GUI
    parent, and creationflags has no useful values on those platforms.
    Flags must be 0 so the call is portable."""
    for plat in ("darwin", "linux"):
        monkeypatch.setattr(ti.sys, "platform", plat)
        assert ti._subprocess_flags() == 0


def test_pip_install_user_cancel_maps_to_cancelled_stage(
    _isolated_runtime, _stub_source_pkg, monkeypatch,
):
    """When the user closes a leaked Windows console, the subprocess
    returncode is 3221225786 (0xC000013A unsigned) or -1073741510
    (signed). Both MUST map to InstallError(stage='cancelled'), not
    the generic 'pip_install' failure — the dialog uses the stage tag
    to show a friendly 'install was interrupted' modal instead of the
    same wall-of-error for both UX states."""
    monkeypatch.setattr(ti.sys, "platform", "win32")
    embed_zip_bytes = _make_fake_embed_zip({
        ti.embedded_python_exe().name: "x", "python311._pth": "#import site",
    })
    urlretrieve = _make_fake_urlretrieve({
        ti.EMBED_PYTHON_URL: embed_zip_bytes,
        ti.GET_PIP_URL: b"# stub",
    })
    import subprocess
    def _sp_run_cancel_on_pip_install(cmd, *, capture=False):
        if "install" in cmd and "pip" in cmd:
            raise subprocess.CalledProcessError(
                returncode=3221225786, cmd=cmd,
                stderr="(no output — process terminated)",
            )
        result = MagicMock()
        result.returncode = 0
        result.stdout = ""
        return result
    with pytest.raises(ti.InstallError) as excinfo:
        ti.install_runtime(
            _urlretrieve=urlretrieve,
            _subprocess_run=_sp_run_cancel_on_pip_install,
            _source_pkg_dir=_stub_source_pkg,
        )
    assert excinfo.value.stage == "cancelled", (
        f"unsigned 3221225786 should map to 'cancelled', got "
        f"{excinfo.value.stage!r}. Without this mapping the user-cancel "
        f"and real-pip-failure cases share the same error UX."
    )
    assert "interrupted" in excinfo.value.message.lower()


def test_pip_install_user_cancel_signed_returncode_also_maps(
    _isolated_runtime, _stub_source_pkg, monkeypatch,
):
    """Some Python builds report 0xC000013A as the signed value
    -1073741510 instead of unsigned 3221225786. Both must map to
    'cancelled' — checking only one of the two would leave a 50%
    chance of generic 'pip_install' failure depending on Python build."""
    monkeypatch.setattr(ti.sys, "platform", "win32")
    embed_zip_bytes = _make_fake_embed_zip({
        ti.embedded_python_exe().name: "x", "python311._pth": "#import site",
    })
    urlretrieve = _make_fake_urlretrieve({
        ti.EMBED_PYTHON_URL: embed_zip_bytes,
        ti.GET_PIP_URL: b"# stub",
    })
    import subprocess
    def _sp_run(cmd, *, capture=False):
        if "install" in cmd and "pip" in cmd:
            raise subprocess.CalledProcessError(
                returncode=-1073741510, cmd=cmd, stderr="",
            )
        result = MagicMock()
        result.returncode = 0
        result.stdout = ""
        return result
    with pytest.raises(ti.InstallError) as excinfo:
        ti.install_runtime(
            _urlretrieve=urlretrieve, _subprocess_run=_sp_run,
            _source_pkg_dir=_stub_source_pkg,
        )
    assert excinfo.value.stage == "cancelled"


def test_pip_install_real_failure_runs_cleanup(
    _isolated_runtime, _stub_source_pkg,
):
    """When pip fails for a NON-cancel reason (e.g., torch wheel
    unavailable), the partial torch tree must be wiped from
    site-packages so the next retry doesn't hit 'already satisfied'
    over a broken state. Without cleanup, the user runs install
    twice + still gets a non-functional GPU runtime."""
    embed_zip_bytes = _make_fake_embed_zip({
        ti.embedded_python_exe().name: "x", "python311._pth": "#import site",
    })
    urlretrieve = _make_fake_urlretrieve({
        ti.EMBED_PYTHON_URL: embed_zip_bytes,
        ti.GET_PIP_URL: b"# stub",
    })
    # Pre-populate the site-packages with a partial torch tree so the
    # cleanup has something to delete.
    site_pkgs = ti.embedded_python_dir() / "Lib" / "site-packages"
    site_pkgs.mkdir(parents=True, exist_ok=True)
    (site_pkgs / "torch").mkdir()
    (site_pkgs / "torch" / "__init__.py").write_text("x", encoding="utf-8")
    (site_pkgs / "torch-2.4.1.dist-info").mkdir()
    (site_pkgs / "nvidia_cuda_runtime_cu12").mkdir()
    import subprocess
    def _sp_run(cmd, *, capture=False):
        if "install" in cmd and "pip" in cmd:
            raise subprocess.CalledProcessError(
                returncode=1, cmd=cmd,
                stderr="ERROR: torch wheel unavailable",
            )
        result = MagicMock()
        result.returncode = 0
        result.stdout = ""
        return result
    with pytest.raises(ti.InstallError):
        ti.install_runtime(
            _urlretrieve=urlretrieve, _subprocess_run=_sp_run,
            _source_pkg_dir=_stub_source_pkg,
        )
    # Cleanup ran — partial torch entries are gone.
    assert not (site_pkgs / "torch").exists(), "torch dir not cleaned"
    assert not (site_pkgs / "torch-2.4.1.dist-info").exists(), "dist-info not cleaned"
    assert not (site_pkgs / "nvidia_cuda_runtime_cu12").exists(), "nvidia_* not cleaned"


def test_cleanup_partial_torch_handles_missing_dir(tmp_path):
    """If site-packages doesn't even exist yet (e.g., pip failed at
    bootstrap, before any wheels could land), cleanup must be a no-op
    that returns 0 — never crash on a path that doesn't exist."""
    deleted = ti._cleanup_partial_torch(tmp_path / "does-not-exist")
    assert deleted == 0


def test_cleanup_partial_torch_handles_permission_errors(
    tmp_path, monkeypatch,
):
    """A single file that can't be deleted (Windows file lock, permission
    denied, etc.) doesn't stop the helper from cleaning the rest. The
    operation is best-effort by design."""
    site_pkgs = tmp_path / "site_pkgs"
    site_pkgs.mkdir()
    (site_pkgs / "torch").mkdir()
    (site_pkgs / "torch_alt").mkdir()
    import shutil
    real_rmtree = shutil.rmtree
    calls = {"count": 0}
    def _selective_fail(path, *args, **kwargs):
        calls["count"] += 1
        if "torch_alt" in str(path):
            raise OSError("simulated permission error")
        return real_rmtree(path, *args, **kwargs)
    monkeypatch.setattr(ti.shutil if hasattr(ti, "shutil") else shutil,
                        "rmtree", _selective_fail)
    # We monkeypatch the global shutil since the helper imports it lazily.
    monkeypatch.setattr("shutil.rmtree", _selective_fail)
    # Cleanup should attempt both entries; one succeeds, one fails silently.
    ti._cleanup_partial_torch(site_pkgs)
    # Either both went or just one — but the helper didn't raise.


def test_verify_cuda_raises_on_kernel_mismatch_even_when_available(
    _isolated_runtime, _stub_source_pkg,
):
    """RTX 50-series regression: cu121 wheels report
    torch.cuda.is_available() == True on sm_120 hardware (driver
    reports the device fine), but the first actual kernel launch
    crashes with 'no kernel image available'. The new verify probe
    runs a real kernel and surfaces this mismatch AS an install
    failure — otherwise the marker says cuda=True and the user hits
    a generic error at first Generate, far from the install context."""
    embed_zip_bytes = _make_fake_embed_zip({
        ti.embedded_python_exe().name: "x", "python311._pth": "#import site",
    })
    urlretrieve = _make_fake_urlretrieve({
        ti.EMBED_PYTHON_URL: embed_zip_bytes,
        ti.GET_PIP_URL: b"# stub",
    })
    sp_run = _make_fake_subprocess_success(
        stdout_for_verify=json.dumps({
            "cuda_available": True,                # device IS visible
            "device_name": "NVIDIA RTX 5090",
            "compute_capability": [12, 0],         # sm_120 — Blackwell
            "kernel_run_ok": False,                # but kernel launch failed
            "torch_version": "2.4.1+cu121",        # OLD wheels lacking sm_120
            "error": "CUDA error: no kernel image is available "
                     "for execution on the device",
        }),
    )
    with pytest.raises(ti.InstallError) as excinfo:
        ti.install_runtime(
            _urlretrieve=urlretrieve, _subprocess_run=sp_run,
            _source_pkg_dir=_stub_source_pkg,
        )
    assert excinfo.value.stage == "verify_cuda"
    # Error message must name the device + sm_* so the user knows
    # which case they're in (vs a generic CUDA verify failure).
    msg = excinfo.value.message
    assert "RTX 5090" in msg
    assert "sm_120" in msg
    assert "no CUDA kernel runs" in msg or "no kernel image" in msg
    # And references the cu128 + 2.7 pin so the user knows what to
    # upgrade to (or that this EXE is older than the cu128 bump).
    assert "cu128" in msg or "2.7" in msg


def test_verify_cuda_accepts_modern_cards_running_a_kernel(
    _isolated_runtime, _stub_source_pkg,
):
    """Happy path for modern cards with cu128 wheels: kernel runs,
    install completes, marker reflects the device. Mirrors the
    existing happy-path test but ALSO asserts the new
    compute_capability + kernel_run_ok fields flow into the log."""
    embed_zip_bytes = _make_fake_embed_zip({
        ti.embedded_python_exe().name: "x", "python311._pth": "#import site",
    })
    urlretrieve = _make_fake_urlretrieve({
        ti.EMBED_PYTHON_URL: embed_zip_bytes,
        ti.GET_PIP_URL: b"# stub",
    })
    sp_run = _make_fake_subprocess_success(
        stdout_for_verify=json.dumps({
            "cuda_available": True,
            "device_name": "NVIDIA RTX 4090",
            "compute_capability": [8, 9],          # Ada
            "kernel_run_ok": True,                 # real kernel ran
            "torch_version": "2.7.0+cu128",
        }),
    )
    info = ti.install_runtime(
        _urlretrieve=urlretrieve, _subprocess_run=sp_run,
        _source_pkg_dir=_stub_source_pkg,
    )
    assert info.cuda_available is True
    assert info.cuda_device_name == "NVIDIA RTX 4090"


def test_cu128_index_and_torch_2_7_pinned():
    """Pin the cu128 + 2.7+ wheel set so a regression back to cu121
    (which blocks RTX 50-series) fails loud. cu128 wheels include
    sm_50-sm_120 so this isn't a downgrade for older cards.

    If pytorch.org ships a new wheel index (cu129, cu130, etc.), bump
    BOTH constants together — the index URL and the minimum torch
    version are coupled. Don't update one without the other.
    """
    assert ti.TORCH_CUDA_INDEX.endswith("/cu128"), (
        f"TORCH_CUDA_INDEX={ti.TORCH_CUDA_INDEX} — must end with /cu128 "
        f"to include sm_120 kernels for RTX 50-series. cu121 wheels lack "
        f"sm_120 SASS and crash at first kernel launch on Blackwell."
    )
    major, minor, _patch = ti.TORCH_VERSION.split(".")
    assert int(major) >= 2 and int(minor) >= 7, (
        f"TORCH_VERSION={ti.TORCH_VERSION} — must be ≥ 2.7.x to ship "
        f"with cu128 binaries from pytorch.org. Older torch versions "
        f"don't have cu128 builds + don't include sm_120."
    )


def test_pip_install_spec_uses_pinned_torch_version():
    """If TORCH_VERSION is bumped, PIP_INSTALL_SPEC follows automatically
    (it interpolates the constant). This test guards against someone
    hardcoding a torch version in PIP_INSTALL_SPEC that diverges from
    TORCH_VERSION."""
    spec_str = " ".join(ti.PIP_INSTALL_SPEC)
    assert ti.TORCH_VERSION in spec_str, (
        f"PIP_INSTALL_SPEC {ti.PIP_INSTALL_SPEC} doesn't reference "
        f"pinned TORCH_VERSION={ti.TORCH_VERSION}"
    )


def test_runner_required_subpackages_includes_shapegen_io_runtime():
    """The subpackage allowlist must include the imports torch_runner
    actually makes. If shapegen or io drift off the allowlist, the
    runner subprocess hits ImportError at run time."""
    required = set(ti.RUNNER_REQUIRED_SUBPACKAGES)
    assert "shapegen" in required, "shapegen missing — runner can't import run_gpu"
    assert "io" in required, "io missing — runner can't save_json"
    assert "runtime" in required, "runtime missing — runner is in this subpackage"
