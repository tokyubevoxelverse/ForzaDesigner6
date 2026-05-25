"""Regression tests that catch the "engine cell calls a symbol that isn't
inlined" class of bug — exactly what bit us when build_colab_notebook.py
forgot to add refill.py to its inline list (user repro: NameError:
clean_and_refill after a 10-minute greedy run, in all 4 ellipse-only
production notebooks).

Strategy: for every production notebook, parse the inlined-engine cell with
ast, collect every name referenced AND every name defined, and assert the
referenced - defined - builtins - imports set is empty. Equivalent to running
pyflakes on the cell and asserting zero undefined-name errors.

The test runs without torch (we only ast.parse, never exec). It catches:
  - missing module inlines (the refill bug)
  - typos in function calls
  - references to package-only symbols the stripper didn't catch

It does NOT catch runtime-only failures (eg wrong call signatures, type
errors). For those we rely on the live-run smoke test on Colab.
"""
from __future__ import annotations

import ast
import builtins
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATHS = sorted(REPO_ROOT.glob("notebooks/fap_gpu_colab_*.ipynb"))

# Sanity check the glob found something — otherwise the parametrize gives an
# empty test set and the regression silently disappears.
assert NOTEBOOK_PATHS, f"no production notebooks found in {REPO_ROOT / 'notebooks'}"

# Symbols that come from the standard library / third-party imports we assume
# are present in the Colab runtime. Anything else referenced in the engine
# cell must be DEFINED in the same cell.
_RUNTIME_ALLOWLIST = {
    # numpy / torch / stdlib imports the engine cell uses
    "np", "torch", "time", "Image",
    # dataclass / typing decorators come in via `from dataclasses import ...`
    # which the engine cell emits via the preamble — but ast.walk sees them
    # as Name references in decorator position. Allowlist explicitly.
    "dataclass", "field", "replace", "Callable",
    # constants set up by CELL_SETUP_DEPS that the engine cell consumes
    "DTYPE", "DEVICE",
}


def _extract_engine_cell(nb_path: Path) -> str:
    """Find the cell that inlines the GPU package (identified by the
    'inlined verbatim from forza_abyss_painter/shapegen/gpu/' preamble)."""
    nb = json.loads(nb_path.read_text())
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if "inlined verbatim from forza_abyss_painter/shapegen/gpu/" in src:
            return src
    raise AssertionError(f"engine cell not found in {nb_path.name}")


def _collect_defined_and_referenced(source: str) -> tuple[set[str], set[str]]:
    """Walk the AST. Defined names = function/class defs at any scope, plus
    module-level Assign targets, plus import bindings. Referenced names =
    every Name in a Load context. Returns (defined, referenced)."""
    tree = ast.parse(source)
    defined: set[str] = set()
    referenced: set[str] = set()

    class _Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            defined.add(node.name)
            for arg in node.args.args + node.args.kwonlyargs + node.args.posonlyargs:
                defined.add(arg.arg)
            if node.args.vararg: defined.add(node.args.vararg.arg)
            if node.args.kwarg: defined.add(node.args.kwarg.arg)
            self.generic_visit(node)
        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_ClassDef(self, node):
            defined.add(node.name)
            self.generic_visit(node)

        def visit_Lambda(self, node):
            for arg in node.args.args + node.args.kwonlyargs:
                defined.add(arg.arg)
            self.generic_visit(node)

        def visit_Import(self, node):
            for alias in node.names:
                defined.add(alias.asname or alias.name.split(".")[0])

        def visit_ImportFrom(self, node):
            for alias in node.names:
                defined.add(alias.asname or alias.name)

        def visit_Assign(self, node):
            for target in node.targets:
                for n in ast.walk(target):
                    if isinstance(n, ast.Name):
                        defined.add(n.id)
            self.generic_visit(node)

        def visit_AnnAssign(self, node):
            if isinstance(node.target, ast.Name):
                defined.add(node.target.id)
            self.generic_visit(node)

        def visit_AugAssign(self, node):
            if isinstance(node.target, ast.Name):
                defined.add(node.target.id)
            self.generic_visit(node)

        def visit_For(self, node):
            for n in ast.walk(node.target):
                if isinstance(n, ast.Name):
                    defined.add(n.id)
            self.generic_visit(node)

        def visit_comprehension(self, node):
            for n in ast.walk(node.target):
                if isinstance(n, ast.Name):
                    defined.add(n.id)
            self.generic_visit(node)

        def visit_With(self, node):
            for item in node.items:
                if item.optional_vars:
                    for n in ast.walk(item.optional_vars):
                        if isinstance(n, ast.Name):
                            defined.add(n.id)
            self.generic_visit(node)

        def visit_Name(self, node):
            if isinstance(node.ctx, ast.Load):
                referenced.add(node.id)

        def visit_Global(self, node):
            for name in node.names:
                defined.add(name)

        def visit_Nonlocal(self, node):
            for name in node.names:
                defined.add(name)

    _Visitor().visit(tree)
    return defined, referenced


@pytest.mark.parametrize("nb_path", NOTEBOOK_PATHS, ids=lambda p: p.name)
def test_engine_cell_has_no_undefined_names(nb_path: Path):
    """Every name referenced in the engine cell must be defined in the cell,
    in the runtime allowlist, or a Python builtin. Catches missing module
    inlines (the refill.py bug) at build time instead of after the user
    waits 10 minutes for the greedy phase."""
    src = _extract_engine_cell(nb_path)
    defined, referenced = _collect_defined_and_referenced(src)
    builtin_names = set(dir(builtins))
    undefined = referenced - defined - builtin_names - _RUNTIME_ALLOWLIST
    assert not undefined, (
        f"{nb_path.name}: {len(undefined)} undefined names in engine cell — "
        f"likely a missing module inline in build_colab_notebook.py. "
        f"Undefined: {sorted(undefined)}"
    )


@pytest.mark.parametrize("nb_path", NOTEBOOK_PATHS, ids=lambda p: p.name)
def test_engine_cell_defines_clean_and_refill(nb_path: Path):
    """Explicit canary for the task #74 bug: every production notebook MUST
    define clean_and_refill because engine.run_gpu calls it unconditionally
    during the ellipse-only refill phase."""
    src = _extract_engine_cell(nb_path)
    assert "def clean_and_refill" in src, (
        f"{nb_path.name}: clean_and_refill not defined — the notebook will "
        f"NameError after 10+ min of greedy if cfg.refill_dead_shapes=True. "
        f"Add 'refill' to the module list in build_colab_notebook.py."
    )


@pytest.mark.parametrize("nb_path", NOTEBOOK_PATHS, ids=lambda p: p.name)
def test_engine_cell_defines_run_gpu(nb_path: Path):
    """Top-level smoke: run_gpu is the entry point the Run cell calls. If
    the engine inline broke entirely, this catches it."""
    src = _extract_engine_cell(nb_path)
    assert "def run_gpu" in src, f"{nb_path.name}: run_gpu missing"
