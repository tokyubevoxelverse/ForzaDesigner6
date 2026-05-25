"""Install-prompt + install-progress dialog for the on-demand GPU runtime.

Phase 2 of EXE GPU bundle (task #62) — the UI scaffolding. Phase 3 wires
the actual HTTP download + embedded-Python bootstrap + torch install
(`forza_abyss_painter.runtime.torch_installer.install_runtime`, not yet
implemented). For now the "Install" button reports "Phase 3 not yet
shipped — close this dialog and watch the EXE Releases for a build that
includes the runtime installer."

Two phases inside this single dialog:

  CONFIRM phase (initial state):
    - Explains what gets downloaded (~4 GiB), where (LOCALAPPDATA), why
      (one-time setup so consumer-GPU users can shape-gen in-app without
      needing Colab access)
    - Install / Cancel buttons

  INSTALL phase (after Install clicked):
    - Replaces the confirm text with a progress bar + status line
    - Cancel button still works (would terminate the download mid-stream
      in Phase 3; for now just closes the dialog)
"""
from __future__ import annotations

from PySide6.QtCore import Qt, QThread
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QMessageBox,
    QProgressBar, QPushButton, QVBoxLayout,
)

from forza_abyss_painter.gui.gpu_install_worker import GpuInstallWorker
from forza_abyss_painter.runtime import torch_installer


class RuntimeInstallDialog(QDialog):
    """Modal dialog: prompt to install the GPU runtime, then show install
    progress. Caller checks `was_installed` after exec() — True if the
    runtime is ready to use, False if the user cancelled or install
    failed.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Install GPU runtime")
        self.setModal(True)
        self.setMinimumWidth(560)
        self.was_installed: bool = False

        self._root = QVBoxLayout(self)
        self._root.setContentsMargins(20, 16, 20, 16)
        self._root.setSpacing(12)

        # Header
        hdr = QLabel("Install local GPU shape-generation runtime")
        hf = QFont(); hf.setBold(True); hf.setPointSize(13)
        hdr.setFont(hf)
        self._root.addWidget(hdr)

        # Confirm-phase content — what the user is agreeing to.
        gib = torch_installer.estimated_download_bytes() / (1 << 30)
        self.body = QLabel(
            f"To generate shapes locally on your GPU, Forza Abyss Painter "
            f"needs an isolated PyTorch + CUDA runtime. This is a "
            f"<b>one-time download of ~{gib:.0f} GiB</b> stored in your "
            f"local app data folder (<code>{torch_installer.runtime_root()}</code>) "
            f"— it doesn't affect your system Python or other applications.\n\n"
            f"Subsequent runs reuse the cached runtime instantly.\n\n"
            f"You can skip this entirely and use the Colab notebooks "
            f"(see the README) — those run on a dedicated cloud GPU and "
            f"don't touch your machine. The local runtime is for users who "
            f"prefer to keep everything offline."
        )
        self.body.setWordWrap(True)
        self.body.setTextFormat(Qt.RichText)
        self._root.addWidget(self.body)

        # Install-phase content — progress bar + status line, hidden until needed.
        self.progress = QProgressBar(self)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setVisible(False)
        self._root.addWidget(self.progress)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #888; font-size: 11px;")
        self.status_label.setVisible(False)
        self._root.addWidget(self.status_label)

        # Buttons — Install (primary) + Cancel.
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.cancel_btn = QPushButton("Cancel", self)
        self.cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(self.cancel_btn)
        self.install_btn = QPushButton("Install", self)
        self.install_btn.setDefault(True)
        self.install_btn.clicked.connect(self._on_install_clicked)
        btn_row.addWidget(self.install_btn)
        self._root.addLayout(btn_row)

    # --------------------------------------------------- install-phase machinery

    def _on_install_clicked(self) -> None:
        """User clicked Install — switch to install-phase UI + spawn the
        GpuInstallWorker on a QThread to actually run install_runtime().
        Progress signals update the progress bar + status label;
        done/error signals route to _on_install_done / _on_install_error
        which set was_installed + close the dialog or surface a modal."""
        self.body.setText(
            "Installing GPU runtime — this can take 5–15 minutes depending on "
            "your network speed. The dialog will close automatically when done. "
            "<br><br><i>Note: cancel during install isn't supported yet; the "
            "Cancel button is disabled until completion. If something goes "
            "wrong, manually delete the runtime directory and re-run.</i>"
        )
        self.progress.setVisible(True)
        self.status_label.setVisible(True)
        self.install_btn.setEnabled(False)
        # Cancel-during-install isn't safe (torch_installer's HTTP
        # downloads aren't interruptible mid-stream without orphaning
        # temp files). Disable until done; the dialog auto-closes on
        # success or surfaces an error modal on failure.
        self.cancel_btn.setEnabled(False)

        # Spawn the worker on a dedicated thread so the GUI stays
        # responsive while torch wheels download. Hold both as instance
        # attrs so the GC doesn't reap them mid-run.
        self._thread = QThread(self)
        self._worker = GpuInstallWorker()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.done.connect(self._on_install_done)
        self._worker.error.connect(self._on_install_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    def _on_progress(self, percent: int, status: str) -> None:
        """Called by the install worker (Phase 3) on each download chunk /
        install step. percent in [0, 100], status is a human-readable label
        like 'downloading torch-2.4.1+cu121-...whl (1.2 GiB / 2.4 GiB)'."""
        self.progress.setValue(max(0, min(100, percent)))
        self.status_label.setText(status)

    def _on_install_done(self, runtime_info_dict: dict) -> None:
        """Worker emitted done(RuntimeInfo). Surface a brief success
        state then auto-accept so the caller can immediately verify
        via is_runtime_installed() + proceed to the Generate dialog."""
        self.was_installed = True
        cuda = runtime_info_dict.get("cuda_available", False)
        device = runtime_info_dict.get("cuda_device_name", "")
        torch_v = runtime_info_dict.get("torch_version", "")
        if cuda:
            self.status_label.setText(
                f"Done — torch {torch_v} installed, CUDA ready on {device}"
            )
        else:
            # Partial install — torch landed but CUDA isn't reachable.
            # The marker records this and is_runtime_installed() will
            # return False, so the EXE doesn't try to GPU-generate.
            self.status_label.setText(
                f"Installed torch {torch_v} but CUDA isn't available — "
                f"check your Nvidia driver version and try again."
            )
            self.was_installed = False
        self.progress.setValue(100)
        from PySide6.QtCore import QTimer
        QTimer.singleShot(1500, self.accept)

    def _on_install_error(self, stage: str, message: str) -> None:
        """Worker emitted error(stage, message). Surface a modal with
        the stage tag so the user knows which phase to investigate.
        Leave the dialog open so they can dismiss after reading."""
        self.was_installed = False
        self.status_label.setText(f"Install failed at {stage}: {message}")
        self.cancel_btn.setEnabled(True)
        self.cancel_btn.setText("Close")
        QMessageBox.critical(
            self, f"GPU runtime install failed — {stage}",
            f"Stage: {stage}\n\n{message}\n\n"
            f"Try again, or use the Colab notebooks if your machine "
            f"can't host the local runtime.",
        )


def prompt_install_or_use_existing(parent=None) -> bool:
    """Convenience entry point used by the Tools menu. Returns True if the
    runtime is ready to use (either was already installed, or user just
    installed it). False if the user declined the install or it failed."""
    if torch_installer.is_runtime_installed():
        return True
    dlg = RuntimeInstallDialog(parent)
    dlg.exec()
    return dlg.was_installed
