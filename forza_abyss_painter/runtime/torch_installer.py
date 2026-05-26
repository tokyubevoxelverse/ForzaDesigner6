"""On-demand PyTorch runtime installer for the EXE's local GPU shape-gen.

The main EXE ships at ~320 MB (no torch). When the user clicks "Generate
shapes locally" for the first time, this module:

  1. Downloads python.org's embeddable Python 3.11 zip (~25 MB) to
     %LOCALAPPDATA%/ForzaAbyssPainter/runtime/python311/
  2. Bootstraps pip into that embedded Python
  3. Uses that pip to install torch + numpy from the PyTorch CUDA wheel
     index (~2 GB) into the embedded Python's site-packages
  4. Writes a marker file with the installed version + verification
     (torch.cuda.is_available() must return True for the install to count
     as successful)

Subsequent runs reuse the cached install. Inject-only users never pay the
2 GB cost — they don't trigger this code path.

Why a subprocess-isolated embedded Python instead of in-process pip-install?
PyInstaller-bundled EXEs have a frozen import system that doesn't accept
new packages at runtime, and even if it did, torch + CUDA DLLs need careful
LD_LIBRARY_PATH / PATH manipulation that's hostile to the main app's
process state. Spawning the embedded Python as a worker subprocess (via
`torch_runner.py` — separate module) isolates the torch dependency
completely from the main EXE.

This module is the INSTALL side. `torch_runner.py` is the RUN side.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path


# Versions pinned for reproducibility. Bump deliberately when picking up new
# torch/CUDA releases — bundle size + CUDA driver compatibility both change.
EMBED_PYTHON_VERSION = "3.11.9"
EMBED_PYTHON_URL = (
    f"https://www.python.org/ftp/python/{EMBED_PYTHON_VERSION}/"
    f"python-{EMBED_PYTHON_VERSION}-embed-amd64.zip"
)
# Torch + CUDA wheel set. Bumped to cu128 / 2.7+ to cover RTX 50-series
# (Blackwell, sm_120) which cu121 wheels don't include kernels for.
# cu128 wheels include sm_50–sm_120 so older cards (RTX 20/30/40
# series) keep working — the bump is purely additive for them.
#
# Why this bump matters: RTX 5090 testers saw `install passes` (torch
# was imported, is_available() returned True, get_device_name() worked)
# but the first real tensor op crashed with
#     RuntimeError: CUDA error: no kernel image is available for execution on the device
# because the cu121 binaries we shipped have no sm_120 SASS. verify_cuda
# now actually executes a kernel (see _verify_cuda_runs_kernel) so this
# fails at install time on cu121-on-sm_120, not at first generate.
TORCH_VERSION = "2.7.0"
TORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu128"
GET_PIP_URL = "https://bootstrap.pypa.io/get-pip.py"


def runtime_root() -> Path:
    """Return the per-user runtime directory. Created if missing.

    Windows: %LOCALAPPDATA%/ForzaAbyssPainter/runtime/
    macOS:   ~/Library/Application Support/ForzaAbyssPainter/runtime/
    Linux:   $XDG_DATA_HOME/ForzaAbyssPainter/runtime/ (or ~/.local/share/...)

    On non-Windows platforms this exists for testing only — the EXE is
    Windows-only and the runtime download only fires on Windows in prod.
    """
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or
                    Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or
                    Path.home() / ".local" / "share")
    root = base / "ForzaAbyssPainter" / "runtime"
    root.mkdir(parents=True, exist_ok=True)
    return root


def embedded_python_dir() -> Path:
    """Path to the embedded Python directory (created by install_runtime).
    Doesn't necessarily exist yet — call is_runtime_installed() first."""
    return runtime_root() / "python311"


def embedded_python_exe() -> Path:
    """Path to the embedded Python interpreter executable. On Windows this
    is python.exe in the embed dir; on other platforms it's a placeholder
    that won't actually run (those platforms don't use this runtime)."""
    suffix = ".exe" if sys.platform == "win32" else ""
    return embedded_python_dir() / f"python{suffix}"


def runtime_marker() -> Path:
    """Path to the JSON marker file written when install completes.
    Presence + valid contents = runtime is ready to use."""
    return runtime_root() / "installed.json"


@dataclass(frozen=True)
class RuntimeInfo:
    """Captured state of an installed runtime. Written to runtime_marker()
    when install_runtime() completes successfully so subsequent launches
    can verify the install without re-running pip."""
    python_version: str
    torch_version: str
    cuda_available: bool
    cuda_device_name: str   # "" if cuda_available is False
    installed_at_utc: str   # ISO 8601

    def to_dict(self) -> dict:
        return {
            "python_version": self.python_version,
            "torch_version": self.torch_version,
            "cuda_available": self.cuda_available,
            "cuda_device_name": self.cuda_device_name,
            "installed_at_utc": self.installed_at_utc,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RuntimeInfo":
        return cls(
            python_version=str(d.get("python_version", "")),
            torch_version=str(d.get("torch_version", "")),
            cuda_available=bool(d.get("cuda_available", False)),
            cuda_device_name=str(d.get("cuda_device_name", "")),
            installed_at_utc=str(d.get("installed_at_utc", "")),
        )


def installed_runtime_info() -> RuntimeInfo | None:
    """Read the install marker, return parsed RuntimeInfo, or None if no
    valid install is present. Used by the GUI to decide whether to show
    "Install GPU runtime (~2 GB download)" vs "Generate locally" buttons."""
    marker = runtime_marker()
    if not marker.exists():
        return None
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
        info = RuntimeInfo.from_dict(data)
        # Also sanity-check the python executable still exists — user might
        # have deleted the runtime dir between sessions.
        if not embedded_python_exe().exists():
            return None
        return info
    except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError):
        return None


def is_runtime_installed() -> bool:
    """Cheap check: is a usable runtime present? False if marker is missing
    OR the embedded Python binary is gone OR the marker says CUDA isn't
    available (which means a partial / broken install)."""
    info = installed_runtime_info()
    if info is None:
        return False
    return info.cuda_available


def estimated_download_bytes() -> int:
    """Rough size of what install_runtime() will download. Surfaced in the
    GUI's "first-run download" confirmation prompt. Numbers are approximate:
        - Embedded Python 3.11.9 amd64: ~10 MiB
        - get-pip.py: ~2 MiB
        - torch 2.4.1+cu121: ~2.4 GiB
        - numpy + nvidia-* dep wheels: ~1.5 GiB cumulative
    Total fence-post: ~4 GiB. Confirmation prompt should round up.
    """
    return 4 * (1 << 30)   # 4 GiB


# ============================================================== install_runtime


class InstallError(RuntimeError):
    """Raised when install_runtime fails. Carries `stage` so the GUI's
    error modal can tell the user WHICH phase broke (download? pip?
    CUDA verify?) and the right remediation."""

    def __init__(self, stage: str, message: str) -> None:
        super().__init__(f"[{stage}] {message}")
        self.stage = stage
        self.message = message


# Pip install spec for the GPU shape-gen dependencies. Pinned to the
# torch version we tested against — never bump silently.
PIP_INSTALL_SPEC: list[str] = [
    f"torch=={TORCH_VERSION}",
    "numpy",
    "Pillow",
]

# Subpackages of forza_abyss_painter the embedded Python actually needs.
# `gui` is excluded because it imports PySide6 which the embedded Python
# doesn't have. `inject` is excluded because it does Windows process
# memory ops that aren't on the runner's path. `cli` excluded — runner
# doesn't shell out to fap-clean. Keep this list tight to avoid bloating
# the embedded site-packages with stuff users will never load.
RUNNER_REQUIRED_SUBPACKAGES: tuple[str, ...] = (
    "shapegen",
    "io",
    "runtime",
    # cli holds fap-generate (the headless CLI shim around
    # torch_runner). Without copying it, users invoking `fap-generate`
    # against the embedded Python hit ModuleNotFoundError on
    # `forza_abyss_painter.cli`. Cursor's QUASAR smoke fell back to
    # `python -m forza_abyss_painter.runtime.torch_runner --config ...`
    # which is fine for IPC test but defeats the CLI feature.
    "cli",
)


def _default_urlretrieve(url: str, dest: Path) -> None:
    """Production urlretrieve: downloads url to dest using stdlib urllib.
    Tests pass an injected fake instead so no real network happens."""
    import urllib.request
    urllib.request.urlretrieve(url, str(dest))


# Windows constant — same value as subprocess.CREATE_NO_WINDOW (0x08000000)
# but defined here so the function references work cross-platform (the
# subprocess module exports CREATE_NO_WINDOW only on win32, and we want
# `_subprocess_flags` importable everywhere for testability).
_CREATE_NO_WINDOW = 0x08000000


def _subprocess_flags() -> int:
    """Return Windows-specific subprocess creationflags so child
    processes (embedded python.exe running pip) DO NOT pop up a visible
    cmd window. Without this, a tester who saw the leaked console would
    close it thinking it was a stuck child — terminating the install
    mid-pip with NTSTATUS 0xC000013A (STATUS_CONTROL_C_EXIT).

    On non-Windows platforms this returns 0 (no flags), since macOS/
    Linux subprocess.run never spawns a console window for a GUI parent.
    """
    if sys.platform == "win32":
        return _CREATE_NO_WINDOW
    return 0


# Windows return codes that indicate the subprocess was terminated by
# the user closing its console window (NTSTATUS 0xC000013A —
# STATUS_CONTROL_C_EXIT). Some Python builds report this as the
# unsigned value 3221225786; others as the signed -1073741510. Map
# BOTH to a 'cancelled' stage so the dialog can show 'install was
# interrupted' rather than the generic 'pip failed' wall.
_CANCEL_RETURN_CODES = (3221225786, -1073741510)


def _default_subprocess_run(
    cmd: list[str], *, capture: bool = False,
) -> "subprocess.CompletedProcess":
    """Production subprocess.run: runs cmd with stdout/stderr captured
    or attached as appropriate. Tests pass an injected fake.

    On Windows, passes CREATE_NO_WINDOW via creationflags so child
    processes don't allocate a visible console window. See
    _subprocess_flags() docstring for why this matters.
    """
    import subprocess
    return subprocess.run(
        cmd,
        check=True,
        capture_output=capture,
        text=True,
        creationflags=_subprocess_flags() if sys.platform == "win32" else 0,
    )


def _cleanup_partial_torch(site_packages_dir: Path) -> int:
    """Wipe torch* and nvidia_* entries from the embedded
    site-packages directory. Called from the pip_install except branch
    when the user-cancel case isn't matched — without this, a retry
    hits pip's 'already satisfied' for a torch install that's
    structurally broken (no `installed.json`, package never copied)
    and the resume looks like a no-op success.

    Returns the count of top-level entries deleted. Best-effort: a
    single failing rmtree doesn't stop the rest.
    """
    import shutil
    if not site_packages_dir.is_dir():
        return 0
    deleted = 0
    patterns = ("torch", "torch-*", "torchgen*",
                "nvidia*", "fbgemm*", "sympy*")
    seen: set[Path] = set()
    for pattern in patterns:
        for entry in site_packages_dir.glob(pattern):
            if entry in seen:
                continue
            seen.add(entry)
            try:
                if entry.is_dir():
                    shutil.rmtree(entry, ignore_errors=True)
                else:
                    entry.unlink(missing_ok=True)
                deleted += 1
            except OSError:
                pass   # best-effort; partial cleanup is still progress
    return deleted


def _source_package_dir() -> Path:
    """Return the on-disk directory of the forza_abyss_painter package
    that's currently importing this module. Used by install_runtime to
    locate the source tree to copy into the embedded site-packages.

    PyInstaller onefile mode is the tricky case: imported modules live
    in the PYZ archive (not as on-disk dirs in _MEIPASS) unless they
    were added via `--add-data` / `datas=` in the spec. We bundle
    shapegen/io/runtime via the spec for exactly this reason — so the
    `Path(__file__).parent.parent` path resolves to a real on-disk
    directory tree in the _MEI extraction even when frozen.

    If that path doesn't actually contain the required subpackage dirs
    (the install would crash in _copy_runner_package), prefer
    sys._MEIPASS / 'forza_abyss_painter' as a fallback — covers the
    case where the spec was rebuilt without our datas entries.
    """
    candidate = Path(__file__).resolve().parent.parent
    # Frozen mode: __file__ points into _MEIPASS. If the candidate dir
    # doesn't contain shapegen/, fall back to the bundled-datas path
    # explicitly under sys._MEIPASS so the error message names the
    # right cause (missing --add-data, not missing source).
    if getattr(sys, "frozen", False) and not (candidate / "shapegen").is_dir():
        mei = Path(getattr(sys, "_MEIPASS", ""))
        if mei:
            fallback = mei / "forza_abyss_painter"
            if (fallback / "shapegen").is_dir():
                return fallback
    return candidate


def _copy_runner_package(
    source_pkg_dir: Path,
    site_packages_dir: Path,
    progress_cb=None,
) -> None:
    """Copy `forza_abyss_painter/__init__.py` + the runner-required
    subpackages into the embedded Python's site-packages so the runner
    subprocess can `from forza_abyss_painter.shapegen.gpu.engine import
    run_gpu`. Excludes gui/inject/cli to keep the embedded install lean
    (those subpackages have heavy deps the runner doesn't need)."""
    import shutil

    dest_pkg = site_packages_dir / "forza_abyss_painter"
    dest_pkg.mkdir(parents=True, exist_ok=True)

    # Top-level __init__.py + any other top-level *.py files (e.g.,
    # _build_info.py written by the CI pipeline).
    for top_file in source_pkg_dir.glob("*.py"):
        shutil.copy2(top_file, dest_pkg / top_file.name)

    # Each required subpackage as a recursive tree copy.
    for sub in RUNNER_REQUIRED_SUBPACKAGES:
        src = source_pkg_dir / sub
        if not src.is_dir():
            raise InstallError(
                stage="copy_package",
                message=f"required subpackage {sub!r} missing from source at {src}",
            )
        dst = dest_pkg / sub
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns(
            "__pycache__", "*.pyc", "*.pyo",
        ))


def _enable_site_in_pth(embedded_dir: Path) -> None:
    """The python-3.11.9-embed-amd64 zip ships with a `python311._pth`
    that has `#import site` commented out — without uncommenting that
    line, `python.exe -m pip` fails because the site-packages module
    never loads. This is the single most common embed-Python install
    failure if forgotten.

    Idempotent — running twice doesn't double-edit; just normalizes
    to the "site enabled" state.
    """
    pth = embedded_dir / "python311._pth"
    if not pth.exists():
        # Some embed builds use a different name; try a glob fallback so
        # a minor Python version bump doesn't silently break this step.
        candidates = list(embedded_dir.glob("python3*._pth"))
        if not candidates:
            raise InstallError(
                stage="enable_site",
                message=f"no python3*._pth file found in {embedded_dir} — "
                        f"embed zip extraction may have failed",
            )
        pth = candidates[0]
    text = pth.read_text(encoding="utf-8")
    # Replace the commented-out form OR insert if missing entirely.
    if "import site" in text:
        text = text.replace("#import site", "import site")
    else:
        text = text.rstrip() + "\nimport site\n"
    pth.write_text(text, encoding="utf-8")


def install_runtime(
    progress_cb=None,
    *,
    _urlretrieve=None,
    _subprocess_run=None,
    _source_pkg_dir=None,
    _extract_zip=None,
    _logger=None,
) -> RuntimeInfo:
    """Download embedded Python + torch (cu121) into the runtime dir,
    bootstrap pip, install dependencies, copy the forza_abyss_painter
    package, verify CUDA, write the marker. Returns the RuntimeInfo
    that was written.

    On success: subsequent `is_runtime_installed()` returns True; the
    EXE's Tools → Generate shapes locally menu (gated behind
    GPU_PHASE_3_AVAILABLE) becomes functional.

    On failure: raises InstallError(stage=..., message=...). The GUI's
    progress dialog surfaces both in its error modal so the user
    knows which phase needs attention.

    progress_cb signature: `(percent: int, status: str) -> None`. Called
    at every phase boundary. percent is monotonic 0→100; status is a
    human-readable label. Optional — pass None to silent-install.

    The `_urlretrieve`, `_subprocess_run`, `_source_pkg_dir`,
    `_extract_zip` kwargs are dependency-injection hooks for tests.
    Production callers use defaults (real urllib + real subprocess).
    Tests pass fakes so the orchestration runs without real network /
    real torch install.

    Idempotency: if the runtime is already installed (cuda_available
    True), returns the existing RuntimeInfo without reinstalling. To
    force a clean install, delete the runtime_root() directory first.
    """
    from datetime import datetime, timezone
    import zipfile

    urlretrieve = _urlretrieve or _default_urlretrieve
    sp_run = _subprocess_run or _default_subprocess_run
    src_pkg = _source_pkg_dir() if callable(_source_pkg_dir) else (
        _source_pkg_dir or _source_package_dir()
    )
    # Diagnostic logger: lazy-import so importing torch_installer doesn't
    # also import gpu_logger (which would create a log file the first
    # time `is_runtime_installed()` is checked at GUI startup — wasteful
    # if the user never triggers the install path).
    if _logger is None:
        from forza_abyss_painter.runtime.gpu_logger import get_gpu_logger
        _logger = get_gpu_logger()

    def _report(pct: int, status: str) -> None:
        _logger.log("install_progress", percent=pct, status=status)
        if progress_cb is not None:
            progress_cb(pct, status)

    _logger.log("install_runtime_called",
                runtime_root=str(runtime_root()),
                embed_python_version=EMBED_PYTHON_VERSION,
                torch_version=TORCH_VERSION,
                torch_cuda_index=TORCH_CUDA_INDEX)

    # Phase 0: skip if already installed.
    existing = installed_runtime_info()
    if existing is not None and existing.cuda_available:
        _logger.log("install_skip_already_installed",
                    torch_version=existing.torch_version,
                    cuda_device_name=existing.cuda_device_name,
                    installed_at_utc=existing.installed_at_utc)
        _report(100, f"Runtime already installed (torch {existing.torch_version})")
        return existing

    root = runtime_root()
    embed_dir = embedded_python_dir()
    embed_exe = embedded_python_exe()
    _report(0, "Preparing runtime directory")

    # Phase 1: download embed Python zip.
    embed_zip = root / f"python-{EMBED_PYTHON_VERSION}-embed-amd64.zip"
    _report(2, f"Downloading embedded Python {EMBED_PYTHON_VERSION}")
    try:
        with _logger.start_phase("download_python",
                                  url=EMBED_PYTHON_URL,
                                  dest=str(embed_zip)):
            urlretrieve(EMBED_PYTHON_URL, embed_zip)
            _logger.log("download_python_size",
                        size_bytes=embed_zip.stat().st_size if embed_zip.exists() else 0)
    except Exception as exc:
        raise InstallError(
            stage="download_python",
            message=f"failed to download {EMBED_PYTHON_URL}: {exc}",
        ) from exc

    # Phase 2: extract zip.
    _report(10, "Extracting embedded Python")
    embed_dir.mkdir(parents=True, exist_ok=True)
    try:
        with _logger.start_phase("extract_python", embed_dir=str(embed_dir)):
            if _extract_zip is not None:
                _extract_zip(embed_zip, embed_dir)
            else:
                with zipfile.ZipFile(embed_zip) as zf:
                    zf.extractall(embed_dir)
    except (zipfile.BadZipFile, OSError) as exc:
        raise InstallError(
            stage="extract_python",
            message=f"failed to extract {embed_zip}: {exc}",
        ) from exc

    if not embed_exe.exists():
        raise InstallError(
            stage="extract_python",
            message=f"embedded python exe not found at {embed_exe} "
                    f"after extraction — embed zip may be malformed",
        )

    # Phase 3: enable site in ._pth so pip works.
    _report(15, "Enabling site-packages in embedded Python")
    try:
        _enable_site_in_pth(embed_dir)
    except InstallError:
        raise   # already typed; propagate
    except Exception as exc:
        raise InstallError(
            stage="enable_site",
            message=f"unexpected failure: {exc}",
        ) from exc

    # Phase 4: download get-pip.py.
    get_pip = root / "get-pip.py"
    _report(20, "Downloading get-pip.py")
    try:
        urlretrieve(GET_PIP_URL, get_pip)
    except Exception as exc:
        raise InstallError(
            stage="download_pip",
            message=f"failed to download {GET_PIP_URL}: {exc}",
        ) from exc

    # Phase 5: bootstrap pip into the embedded Python.
    _report(25, "Bootstrapping pip")
    try:
        with _logger.start_phase("bootstrap_pip", get_pip=str(get_pip)):
            sp_run([str(embed_exe), str(get_pip)], capture=True)
    except Exception as exc:
        raise InstallError(
            stage="bootstrap_pip",
            message=f"get-pip.py failed: {exc}",
        ) from exc

    # Phase 6: pip install torch + numpy + Pillow. The longest step
    # in wall time (gigabytes of wheels). Progress jumps from 30 → 80
    # because we can't easily stream pip's own progress through this
    # callback — would need a `pip install --progress-bar json` parser
    # which doesn't exist. Just hold at the start-of-step value until
    # pip returns.
    _report(30, f"Installing torch {TORCH_VERSION} + deps "
                f"(~3 GiB; takes 5-15 min — DO NOT close any windows)")
    try:
        with _logger.start_phase("pip_install",
                                  index_url=TORCH_CUDA_INDEX,
                                  spec=PIP_INSTALL_SPEC):
            sp_run(
                [str(embed_exe), "-m", "pip", "install",
                 "--index-url", TORCH_CUDA_INDEX,
                 "--extra-index-url", "https://pypi.org/simple",
                 *PIP_INSTALL_SPEC],
                capture=True,
            )
    except Exception as exc:
        # Distinguish user-cancellation from real pip failures: the
        # tester closing a leaked Windows console produces NTSTATUS
        # 0xC000013A which surfaces as returncode 3221225786 (or the
        # signed equivalent -1073741510). Per Cursor's QUASAR
        # post-mortem, this is the #1 cause of perceived "install
        # failed" — we shouldn't show the same generic 'pip failed'
        # modal for a user-initiated cancel as for a real pip error.
        rc = getattr(exc, "returncode", None)
        if sys.platform == "win32" and rc in _CANCEL_RETURN_CODES:
            _logger.log("install_cancelled_by_user",
                        returncode=rc, ntstatus="0xC000013A")
            raise InstallError(
                stage="cancelled",
                message=(
                    "Install was interrupted (a window was closed or "
                    "the process was killed mid-download). Re-run "
                    "Install GPU runtime and let it complete — first "
                    "install takes 5-15 minutes."
                ),
            ) from exc
        # Real failure — try to clean up the partial torch install so
        # the retry doesn't hit 'already satisfied' for a broken state.
        site_pkgs = embed_dir / "Lib" / "site-packages"
        deleted = _cleanup_partial_torch(site_pkgs)
        _logger.log("partial_install_cleanup",
                    deleted_entries=deleted, site_pkgs=str(site_pkgs))
        raise InstallError(
            stage="pip_install",
            message=f"pip install failed: {exc}",
        ) from exc

    # Phase 7: copy forza_abyss_painter subpackages into embedded
    # site-packages so torch_runner can resolve its imports.
    _report(80, "Copying forza_abyss_painter package")
    site_pkgs = embed_dir / "Lib" / "site-packages"
    site_pkgs.mkdir(parents=True, exist_ok=True)
    try:
        _copy_runner_package(src_pkg, site_pkgs)
    except InstallError as exc:
        # The 'required subpackage missing' raise is generic. Catch and
        # rewrite the message if we're under _MEI* — that's the
        # PyInstaller-onefile-without-our-datas case, and the user can
        # only fix it by rebuilding the EXE. The generic 'permissions
        # or disk space' hint sent users on a wild goose chase.
        if (str(src_pkg).find("_MEI") >= 0 and
                getattr(sys, "frozen", False) and
                "missing from source" in exc.message):
            raise InstallError(
                stage="copy_package",
                message=(
                    f"{exc.message}\n\n"
                    f"This is a build issue, NOT a problem with your "
                    f"machine: the EXE was packaged without "
                    f"`--add-data` entries for shapegen/io/runtime, so "
                    f"those packages are only in the PyInstaller PYZ "
                    f"archive (not as on-disk folders the installer can "
                    f"copy). Rebuild the EXE with the current "
                    f"ForzaAbyssPainter.spec / build_exe.bat — those "
                    f"files include the required `--add-data` flags."
                ),
            ) from exc
        raise
    except Exception as exc:
        raise InstallError(
            stage="copy_package",
            message=f"failed to copy package: {exc}",
        ) from exc

    # Phase 8: verify CUDA availability via subprocess. We ask the
    # embedded Python to print torch.cuda.is_available() + device name
    # so the marker we write reflects what the user's GPU actually
    # supports — not what we hoped. If CUDA's not available after a
    # nominally-successful install, that's a partial install (e.g.,
    # CPU-only torch wheel landed instead of cu121) and the marker
    # records that — is_runtime_installed() then returns False.
    _report(85, "Verifying CUDA availability")
    cuda_available = False
    cuda_device_name = ""
    # Probe runs a REAL kernel — torch.zeros(1, device='cuda') — not
    # just is_available(). The old probe (just is_available() +
    # get_device_name()) passed on RTX 5090 with cu121 wheels because
    # the runtime reports the device fine; only the first kernel
    # launch fails with "no kernel image available". Running a kernel
    # at verify time means a cu121-on-sm_120 mismatch surfaces NOW
    # with stage='verify_cuda' rather than later as a generation
    # error the user doesn't know how to interpret.
    #
    # Also captures get_device_capability() so we can detect sm_120+
    # boards explicitly and emit a targeted error if the installed
    # torch wheel doesn't support them (currently anything < 2.7+cu128).
    probe_script = (
        "import json, sys, traceback, torch\n"
        "info = {'cuda_available': False, 'device_name': '', "
        "'compute_capability': None, 'kernel_run_ok': False, "
        "'torch_version': torch.__version__}\n"
        "try:\n"
        "    info['cuda_available'] = torch.cuda.is_available()\n"
        "    if info['cuda_available']:\n"
        "        info['device_name'] = torch.cuda.get_device_name(0)\n"
        "        cap = torch.cuda.get_device_capability(0)\n"
        "        info['compute_capability'] = list(cap)\n"
        "        # Real kernel — fails with 'no kernel image available' "
        "        # if torch wheel doesn't support this device's sm_*.\n"
        "        x = torch.zeros(1, device='cuda')\n"
        "        x.add_(1)\n"
        "        info['kernel_run_ok'] = True\n"
        "except Exception as exc:\n"
        "    info['error'] = f'{type(exc).__name__}: {exc}'\n"
        "print(json.dumps(info))\n"
    )
    try:
        with _logger.start_phase("verify_cuda"):
            proc = sp_run(
                [str(embed_exe), "-c", probe_script],
                capture=True,
            )
            out = proc.stdout.strip().splitlines()[-1]
            verdict = json.loads(out)
            cuda_available = bool(verdict.get("cuda_available", False))
            cuda_device_name = str(verdict.get("device_name", ""))
            kernel_ok = bool(verdict.get("kernel_run_ok", False))
            cc = verdict.get("compute_capability")  # [major, minor] or None
            _logger.log("cuda_verdict",
                        cuda_available=cuda_available,
                        cuda_device_name=cuda_device_name,
                        compute_capability=cc,
                        kernel_run_ok=kernel_ok,
                        probe_error=verdict.get("error"))
            # If torch reports CUDA available but the kernel won't run,
            # we have a wheel-vs-device mismatch (e.g., cu121 on sm_120).
            # Surface it AS an install failure — otherwise the marker
            # says cuda_available=True and the user hits the error
            # later at first generate, with no context.
            if cuda_available and not kernel_ok:
                sm = (f"sm_{cc[0]}{cc[1]}" if cc else "unknown")
                detail = verdict.get("error", "(no error captured)")
                raise InstallError(
                    stage="verify_cuda",
                    message=(
                        f"torch installed but no CUDA kernel runs on this "
                        f"GPU ({cuda_device_name}, {sm}). The installed "
                        f"wheel ({verdict.get('torch_version', '?')}) "
                        f"doesn't include kernels for your device.\n\n"
                        f"For RTX 50-series cards (sm_120+), you need "
                        f"torch ≥ 2.7 with cu128 wheels. The current "
                        f"installer pins TORCH_VERSION={TORCH_VERSION} "
                        f"+ {TORCH_CUDA_INDEX} — if this error appears, "
                        f"the EXE is older than the cu128 bump.\n\n"
                        f"Probe detail: {detail}"
                    ),
                )
    except InstallError:
        raise
    except Exception as exc:
        raise InstallError(
            stage="verify_cuda",
            message=f"CUDA verification subprocess failed: {exc}",
        ) from exc

    # Phase 9: write the marker.
    _report(95, "Writing install marker")
    # Marker's torch_version derives the +cu... suffix from the index
    # URL so a future cu12X bump stays in sync without code edits.
    # Cursor's QUASAR smoke caught the stale '+cu121' hardcode when we
    # bumped to cu128 — this prevents the next bump from drifting.
    _index_suffix = TORCH_CUDA_INDEX.rstrip("/").rsplit("/", 1)[-1]
    info = RuntimeInfo(
        python_version=EMBED_PYTHON_VERSION,
        torch_version=(f"{TORCH_VERSION}+{_index_suffix}"
                       if cuda_available else TORCH_VERSION),
        cuda_available=cuda_available,
        cuda_device_name=cuda_device_name,
        installed_at_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    runtime_marker().write_text(
        json.dumps(info.to_dict(), indent=2), encoding="utf-8",
    )
    _logger.log("install_runtime_done", outcome="ok",
                runtime_info=info.to_dict())
    _report(100, "Done")
    return info
