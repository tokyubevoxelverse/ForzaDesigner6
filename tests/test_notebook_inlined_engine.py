"""Regression test for the inlined-engine cell in each generated notebook.

The notebook builder strips `from forza_abyss_painter.shapegen.gpu.*` imports from each engine
module's source and re-supplies dependencies by inlining the listed modules into
one big cell, plus a small preamble of standard-library imports.

Two classes of bug have shipped from this pipeline:
  1. Module in engine.py imports get stripped but the module itself isn't in the
     inline list → NameError on use (e.g. joint_polish).
  2. Stdlib import (e.g. `replace`) gets stripped but isn't in the preamble →
     NameError on use.

This test exec()s the setup-engine cell of every generated notebook and asserts
the public names a notebook actually uses are bound. If any of those names go
missing because of a future inlining miss, this fails immediately rather than
crashing the user mid-Colab.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

NOTEBOOKS = sorted((Path(__file__).resolve().parents[1] / "notebooks").glob("fap_gpu_colab_*.ipynb"))

# Names every generated preset notebook's run cell references — if any is missing
# from the inlined engine, the notebook crashes in Colab.
REQUIRED_NAMES = [
    "run_gpu",
    "GPUConfig",
    "joint_polish",
    "score_batch",
    "rasterize_rotated_ellipses",
    "get_device",
    "DTYPE",
    "replace",   # dataclasses.replace — engine.py uses it for the lock_alpha override
]


def _setup_engine_cell(nb_path: Path) -> str | None:
    """Return the source of the inlined-engine setup cell, or None if not found."""
    nb = json.loads(nb_path.read_text())
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if "Forza Abyss Painter GPU engine — inlined verbatim" in src:
            return src
    return None


@pytest.mark.parametrize("nb_path", NOTEBOOKS, ids=lambda p: p.name)
def test_setup_engine_cell_executes_cleanly(nb_path):
    """exec the inlined engine cell and assert every required name is bound."""
    src = _setup_engine_cell(nb_path)
    if src is None:
        # polish_reprocess notebook uses pip-install instead of inline; skip.
        pytest.skip(f"{nb_path.name}: no inlined-engine cell (likely pip-install notebook)")
    ns: dict = {}
    exec(compile(src, str(nb_path), "exec"), ns)
    missing = [n for n in REQUIRED_NAMES if n not in ns]
    assert not missing, (
        f"{nb_path.name}: inlined-engine cell is missing names: {missing}. "
        f"Either the inline list in build_colab_notebook.py is missing a module, "
        f"or the preamble is missing an import."
    )
