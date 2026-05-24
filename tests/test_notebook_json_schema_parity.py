"""Regression test for GPU/notebook JSON schema parity with FD6Document.

The notebook's `_doc(shapes)` helper builds the FD6 JSON dict manually inside the
inlined `CELL_RUN` cell (so the cell stays self-contained without importing
fd6.io.json_schema). That manual dict MUST carry every field FD6Document writes
when the CPU engine emits a JSON — otherwise sticker_mode (or any future field)
gets silently dropped and downstream tools (the desktop app preview, the
injector's color expectations) see the wrong content.

This test parses the `_doc` template in the notebook builder source and asserts
its dict keys are a superset of FD6Document's serialized dict keys.
"""
from __future__ import annotations

import re
from dataclasses import fields
from pathlib import Path

from forza_abyss_painter.io.json_schema import FD6Document

BUILDER = Path(__file__).resolve().parents[1] / "notebooks" / "build_colab_notebook.py"


def _doc_template_keys() -> set[str]:
    """Read CELL_RUN's `def _doc(shapes)` body and return the dict keys it writes."""
    src = BUILDER.read_text()
    # The helper is defined inside CELL_RUN as the only `def _doc(shapes):` literal.
    m = re.search(r"def _doc\(shapes\):.*?return\s*\{(.*?)\}", src, re.DOTALL)
    assert m, "could not locate `def _doc(shapes):` in build_colab_notebook.py"
    body = m.group(1)
    # Extract every double-quoted key on the left of a colon.
    return set(re.findall(r'"([a-zA-Z_]+)"\s*:', body))


def test_notebook_doc_carries_every_FD6Document_field():
    """The notebook's _doc(shapes) must write every field that FD6Document.to_dict() does."""
    notebook_keys = _doc_template_keys()
    schema_keys = {f.name for f in fields(FD6Document)}
    missing = schema_keys - notebook_keys
    assert not missing, (
        f"notebook's _doc(shapes) is missing fields that FD6Document writes: {sorted(missing)}. "
        f"If you add a new field to FD6Document, also add it to _doc(shapes) in CELL_RUN of "
        f"notebooks/build_colab_notebook.py. Dropping sticker_mode caused the FD6.exe "
        f"preview/inject mismatch in 2026-05-23."
    )


def test_notebook_doc_writes_sticker_mode_explicitly():
    """Regression: sticker_mode MUST be in the manual dict (was the bug)."""
    notebook_keys = _doc_template_keys()
    assert "sticker_mode" in notebook_keys, (
        "sticker_mode missing from CELL_RUN's _doc(shapes). The GPU engine writes shape "
        "colors optimized for a grey substrate when STICKER_MODE=True; downstream tools "
        "need this flag in the JSON to render against the matching backdrop."
    )
