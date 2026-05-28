"""Dialog labels reflect chunked-K behavior at the configured budget.

Task #80 (Chunked-K Task 3): the visible `vram_info` label uses the
chunk-aware estimator so users see the per-chunk peak that will actually
allocate (e.g. ~12 GiB on a 17 GiB budget with Hi-Res 3000) rather than
the misleading unchunked full-K number (~139 GiB).
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402


def _find_preset_index(dlg, needle: str) -> int:
    for i in range(dlg.preset_combo.count()):
        if needle in dlg.preset_combo.itemText(i):
            return i
    raise RuntimeError(f"No preset matching {needle!r}")


def test_hi_res_label_shows_chunked_peak_on_tight_budget():
    app = QApplication.instance() or QApplication([])  # noqa: F841
    from forza_abyss_painter.gui.generate_dialog import GenerateLocallyDialog

    dlg = GenerateLocallyDialog(parent=None, gpu_budget_gib=17.0)
    try:
        dlg.preset_combo.setCurrentIndex(_find_preset_index(dlg, "Hi-Res"))
        label_text = dlg.vram_info.text()
        # Hi-Res 3000 on 17 GiB engages chunked-K.
        assert ("chunked" in label_text.lower()
                or "batches" in label_text.lower()), \
            f"label missing chunked/batches: {label_text!r}"
        # The misleading full-K 139 GiB number must not appear.
        assert "139" not in label_text, \
            f"label still shows unchunked full-K peak: {label_text!r}"
        # Budget context should be present.
        assert "17" in label_text, \
            f"label missing budget context: {label_text!r}"
    finally:
        dlg.close()


def test_lineart_label_shows_no_chunking_needed():
    app = QApplication.instance() or QApplication([])  # noqa: F841
    from forza_abyss_painter.gui.generate_dialog import GenerateLocallyDialog

    dlg = GenerateLocallyDialog(parent=None, gpu_budget_gib=17.0)
    try:
        dlg.preset_combo.setCurrentIndex(_find_preset_index(dlg, "Lineart"))
        label_text = dlg.vram_info.text().lower()
        # Lineart fits easily; either "no chunking" or "1 batch" copy.
        assert ("no chunking" in label_text
                or "1 batch" in label_text
                or "chunked into 1" in label_text
                or "unchunked" in label_text), \
            f"label should indicate no chunking needed: {label_text!r}"
    finally:
        dlg.close()
