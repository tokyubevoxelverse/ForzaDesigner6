"""Modal dialog shown during an active injection (FH6 / FH5 / FH4 / FH3).

Blocks the rest of the FD6 GUI, can't be closed by the user, and includes a
prominent professional warning that editing the game's vinyl group during the
operation will cause the injection to fail.

The dialog auto-closes when the worker emits its terminal status (success/warning/error
followed by `done`).
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QDialog, QFrame, QHBoxLayout, QLabel, QProgressBar, QPushButton, QVBoxLayout
)


def _short_game_label(label: str) -> str:
    """Compress 'Forza Horizon N' to 'FHN' for tight UI text. Other labels pass through."""
    if label.startswith("Forza Horizon "):
        suffix = label[len("Forza Horizon "):].strip()
        if suffix.isdigit():
            return f"FH{suffix}"
    return label


SEVERITY_COLORS = {
    "info":    ("#cccccc", "#1f1f1f"),
    "success": ("#2ecc71", "#0c2417"),
    "warning": ("#f1c40f", "#2a2410"),
    "error":   ("#ff4d4d", "#2a1414"),
}


class InjectionDialog(QDialog):
    """Modal injection-in-progress dialog. Caller wires our slots to InjectionWorker signals."""

    def __init__(self, parent=None, json_name: str = "", game_label: str = "Forza Horizon 6") -> None:
        super().__init__(parent)
        # Strip "(BETA)" suffix from the label for cleaner dialog text — beta status
        # is already surfaced as a yellow warning banner from the worker.
        self._clean_label = game_label.replace(" (BETA)", "")
        self._short_label = _short_game_label(self._clean_label)
        self.setWindowTitle(f"Forza Abyss Painter → {self._clean_label} Injection")
        # Block parent, no close button, no help button
        self.setModal(True)
        flags = self.windowFlags()
        flags &= ~Qt.WindowCloseButtonHint
        flags &= ~Qt.WindowContextHelpButtonHint
        flags |= Qt.WindowTitleHint
        self.setWindowFlags(flags)
        self.setMinimumWidth(520)

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 16)
        root.setSpacing(12)

        # Header
        header = QLabel(f"Injecting shapes into {self._clean_label}")
        hf = QFont(); hf.setBold(True); hf.setPointSize(13)
        header.setFont(hf)
        root.addWidget(header)
        if json_name:
            sub = QLabel(f"Source: {json_name}")
            sub.setStyleSheet("color: #888;")
            root.addWidget(sub)

        # Stage label + progress bar
        self.stage_label = QLabel("Preparing…")
        self.stage_label.setStyleSheet("color: #cccccc;")
        self.progress = QProgressBar(self)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        root.addWidget(self.stage_label)
        root.addWidget(self.progress)

        # Detail line (e.g., "324/1842 regions, 47 shape structs found")
        self.detail_label = QLabel("")
        self.detail_label.setStyleSheet("color: #999; font-size: 11px;")
        root.addWidget(self.detail_label)

        # Warning panel — prominent, never goes away during the op
        warn_box = QFrame(self)
        warn_box.setStyleSheet(
            "QFrame { background: #2a1f0a; border: 1px solid #b07a00; border-radius: 6px; }"
            "QLabel { color: #f1c40f; }"
        )
        wl = QVBoxLayout(warn_box)
        wl.setContentsMargins(14, 10, 14, 10)
        warn_title = QLabel(f"⚠  Do not modify {self._clean_label} during injection")
        wtf = QFont(); wtf.setBold(True); wtf.setPointSize(11)
        warn_title.setFont(wtf)
        wl.addWidget(warn_title)
        warn_body = QLabel(
            f"Editing, adding, deleting, or moving any vinyl shape in {self._short_label} while this "
            "operation is running will cause the in-game vinyl group's memory to be "
            "reallocated mid-write, which will fail the injection. Please leave the "
            "vinyl editor untouched until this dialog closes.\n\n"
            "After injection completes: the lowest layer indices in your vinyl group "
            "(typically layers 1–10) may contain placeholder geometry left over from "
            "the original template and can occasionally render in front of the "
            "injected artwork. If you notice unexpected shapes obscuring your design, "
            f"open the {self._short_label} layer panel and delete, hide, or reposition the affected "
            "low-index layers as needed."
        )
        warn_body.setWordWrap(True)
        wbf = QFont(); wbf.setPointSize(9)
        warn_body.setFont(wbf)
        wl.addWidget(warn_body)
        root.addWidget(warn_box)

        # Status line at bottom (colored per severity)
        self.status_label = QLabel("Starting…")
        slf = QFont(); slf.setBold(True)
        self.status_label.setFont(slf)
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)
        self._apply_severity_to_status("info")

        # Close button — disabled until injection ends
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.close_btn = QPushButton("Close")
        self.close_btn.setEnabled(False)
        self.close_btn.clicked.connect(self.accept)
        btn_row.addWidget(self.close_btn)
        root.addLayout(btn_row)

        self._final_severity: str | None = None

    # ------------------------------------------------------- Worker signal handlers

    def on_status(self, message: str, severity: str) -> None:
        self.status_label.setText(message)
        self._apply_severity_to_status(severity)
        if severity in ("success", "warning", "error"):
            self._final_severity = severity

    def on_scan_progress(self, scanned: int, total: int, hits: int) -> None:
        # Scan phase is treated as 0–50% of overall; write phase is 50–100%
        pct = int(round(50 * scanned / max(1, total)))
        self.progress.setValue(pct)
        self.stage_label.setText(f"Stage 1 of 2 — Scanning {self._short_label} memory")
        self.detail_label.setText(
            f"{scanned}/{total} regions  •  {hits} strict LiveryGroup candidate(s) found"
        )

    def on_write_progress(self, written: int, total: int) -> None:
        pct = 50 + int(round(50 * written / max(1, total)))
        self.progress.setValue(pct)
        self.stage_label.setText("Stage 2 of 2 — Writing shapes")
        self.detail_label.setText(f"{written}/{total} shapes written")

    def on_done(self) -> None:
        # Allow user to dismiss now
        self.close_btn.setEnabled(True)
        if self._final_severity == "success":
            self.progress.setValue(100)
        # Brief styling cue: pulse the Close button
        self.close_btn.setDefault(True)
        self.close_btn.setFocus()

    # ------------------------------------------------------- internals

    def _apply_severity_to_status(self, severity: str) -> None:
        fg, bg = SEVERITY_COLORS.get(severity, SEVERITY_COLORS["info"])
        self.status_label.setStyleSheet(
            f"QLabel {{ color: {fg}; background: {bg}; padding: 8px; border-radius: 4px; }}"
        )

    def keyPressEvent(self, event) -> None:
        # Block Esc-to-close while the operation is running
        if event.key() == Qt.Key_Escape and not self.close_btn.isEnabled():
            return
        super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        # Block window-close if the operation hasn't terminated
        if not self.close_btn.isEnabled():
            event.ignore()
            return
        super().closeEvent(event)
