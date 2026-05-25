"""QObject worker that runs torch_installer.install_runtime() on a
background QThread + emits progress through Qt signals.

Separated from runtime_install_dialog.py so unit tests can exercise
the worker's signal routing without spinning the dialog GUI.

## Why a worker thread

install_runtime() takes 5-15 minutes (mostly downloading + pip
installing torch + numpy dep wheels, ~3 GiB total). Running it on the
GUI thread freezes the EXE for the duration — users would conclude
the app crashed and force-kill it. The worker thread keeps the UI
responsive: progress bar updates, status label refreshes, Cancel
button stays clickable.

## Flow

  1. RuntimeInstallDialog._on_install_clicked switches the dialog to
     install-phase UI (progress bar visible, status label active)
  2. Constructs GpuInstallWorker + QThread, moves worker to thread
  3. Connects signals (progress / done / error / finished) to dialog slots
  4. Starts thread → worker calls install_runtime with progress_cb
  5. Each install_runtime phase fires progress_cb(percent, status)
     which the worker forwards to the progress Qt signal
  6. On success → done(RuntimeInfo dict), dialog accepts
  7. On failure → error(stage, message), dialog shows modal + closes

## Cancel

Currently NOT supported mid-install. install_runtime is synchronous
and not designed to interrupt cleanly mid-download. The Cancel
button during install just disables itself until done. Future
improvement: structured cancellation via a shared flag the urlretrieve
hook checks. For now, the user can manually delete the runtime dir
after the dialog closes.
"""
from __future__ import annotations

from PySide6.QtCore import QObject, Signal, Slot


class GpuInstallWorker(QObject):
    """Wraps torch_installer.install_runtime() with Qt signal emission.

    Signals reflect the installer's contract: per-phase progress, a
    one-shot done with the resulting RuntimeInfo, or an error with
    stage tag. finished fires last regardless of outcome so the
    QThread teardown chain runs in order.
    """

    progress = Signal(int, str)   # (percent, status) — forwarded from progress_cb
    done = Signal(dict)           # RuntimeInfo.to_dict() on success
    error = Signal(str, str)      # (stage, message) on InstallError or unknown
    finished = Signal()           # always — clean termination signal

    def __init__(
        self,
        parent: QObject | None = None,
        *,
        _install_fn=None,
    ) -> None:
        """`_install_fn` is a DI hook for tests — callable matching
        torch_installer.install_runtime's signature. Production callers
        omit it (uses the real installer)."""
        super().__init__(parent)
        if _install_fn is None:
            from forza_abyss_painter.runtime.torch_installer import install_runtime
            _install_fn = install_runtime
        self._install_fn = _install_fn

    @Slot()
    def run(self) -> None:
        """Worker entry point — runs on the worker thread when connected
        to QThread.started. install_runtime is synchronous; we wrap it
        with a try/except matrix that maps each failure mode to a typed
        signal so the dialog can route to the right UX state."""
        from forza_abyss_painter.runtime.torch_installer import InstallError

        try:
            info = self._install_fn(progress_cb=self._on_install_progress)
            self.done.emit(info.to_dict())
        except InstallError as exc:
            # Typed installer error — stage carries enough context for
            # the dialog to surface a specific help message (download
            # failure vs pip failure vs CUDA verify failure).
            self.error.emit(exc.stage, exc.message)
        except Exception as exc:   # pragma: no cover — defensive
            # Anything else is a bug in the installer. Surface with
            # stage='unknown' so the dialog still shows SOMETHING; the
            # message captures the exception class for debugging.
            self.error.emit("unknown", f"{type(exc).__name__}: {exc}")
        finally:
            self.finished.emit()

    def _on_install_progress(self, percent: int, status: str) -> None:
        """install_runtime invokes this callback inline from its own
        thread context. Signal.emit() is thread-safe (queued connection
        delivers to slots on the receiver's thread) so this is fine
        even though we're not on the GUI thread."""
        self.progress.emit(int(percent), str(status))
