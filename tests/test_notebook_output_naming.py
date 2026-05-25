"""Regression test pinning the production notebooks' output-filename convention.

The output JSON + render PNG filenames must include the shape budget (NUM_SHAPES) so
users can tell at a glance which quality tier a file belongs to — eg
`my_image_3000.json` is the 3000-shape highres render vs `my_image_400.json` for
a 400-shape lineart render. Same for the render PNG and intermediate checkpoint files.

If anyone reverts to `<stem>.json` (no budget tag) this test fails. If the convention
changes (eg shape budget moved to a different position), update this test deliberately.
"""
import glob
import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_GLOB = "notebooks/fap_gpu_colab_*.ipynb"


def _all_code(nb_path: Path) -> str:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(c.get("source", []))
        for c in nb["cells"]
        if c["cell_type"] == "code"
    )


def _all_notebooks() -> list[Path]:
    nbs = sorted(REPO_ROOT.glob(NOTEBOOK_GLOB))
    assert nbs, f"no notebooks matched {NOTEBOOK_GLOB} under {REPO_ROOT}"
    return nbs


def test_every_production_notebook_includes_budget_tag_in_output_json():
    """JSON output filename must include NUM_SHAPES via _BUDGET_TAG."""
    for nb_path in _all_notebooks():
        src = _all_code(nb_path)
        # Match the line that writes the final JSON.
        m = re.search(r'json_path\s*=\s*[^\n]+', src)
        assert m, f"{nb_path.name}: no `json_path = ...` line found"
        line = m.group(0)
        assert "_BUDGET_TAG" in line, (
            f"{nb_path.name}: json_path doesn't reference _BUDGET_TAG.\n"
            f"  found: {line}\n"
            f"  expected pattern: json_path = out_dir / f\"{{stem}}_{{_BUDGET_TAG}}.json\""
        )


def test_every_production_notebook_includes_budget_tag_in_render_png():
    """Render PNG filename must include NUM_SHAPES via _BUDGET_TAG."""
    for nb_path in _all_notebooks():
        src = _all_code(nb_path)
        m = re.search(r'png_path\s*=\s*[^\n]+', src)
        assert m, f"{nb_path.name}: no `png_path = ...` line found"
        line = m.group(0)
        assert "_BUDGET_TAG" in line, (
            f"{nb_path.name}: png_path doesn't reference _BUDGET_TAG.\n  found: {line}"
        )


def test_checkpoint_files_include_budget_and_ckpt_marker():
    """Greedy-phase checkpoint files must include both NUM_SHAPES and a 'ckpt' marker so
    they're distinguishable from the final output (which has no ckpt marker)."""
    for nb_path in _all_notebooks():
        src = _all_code(nb_path)
        m = re.search(r'def _checkpoint[^\n]+\n(?:[ \t]+[^\n]+\n){1,4}', src)
        assert m, f"{nb_path.name}: no _checkpoint function found"
        fn_body = m.group(0)
        assert "_BUDGET_TAG" in fn_body, (
            f"{nb_path.name}: _checkpoint doesn't include _BUDGET_TAG"
        )
        assert "ckpt" in fn_body, (
            f"{nb_path.name}: _checkpoint doesn't include 'ckpt' marker — checkpoint files "
            f"would be indistinguishable from the final output"
        )


def test_budget_tag_value_is_num_shapes():
    """_BUDGET_TAG must derive from NUM_SHAPES (the user-edited knob in the Configure cell),
    not from some other source. Otherwise users editing NUM_SHAPES wouldn't see their
    change reflected in the output filename."""
    for nb_path in _all_notebooks():
        src = _all_code(nb_path)
        m = re.search(r'_BUDGET_TAG\s*=\s*[^\n]+', src)
        assert m, f"{nb_path.name}: _BUDGET_TAG not defined"
        line = m.group(0)
        assert "NUM_SHAPES" in line, (
            f"{nb_path.name}: _BUDGET_TAG derived from something other than NUM_SHAPES.\n"
            f"  found: {line}"
        )
