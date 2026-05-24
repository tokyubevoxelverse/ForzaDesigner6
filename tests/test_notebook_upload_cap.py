"""Regression test pinning the v0.1.5 input-size safety cap.

The notebook's `_load_image_bytes()` applies a hard ceiling on the input image's long
side BEFORE the engine sees it. This prevents VRAM OOM on large source images by
overriding the preset's max_resolution when it's larger than the safety cap (default 720).

Without this cap, large source images on high-resolution presets (eg highres_3000 with
max_resolution=1600) push the bbox-local scorer past 80+ GB of VRAM on the dev GPU.
The cap was added in v0.1.5 with UPLOAD_MAX_LONG_SIDE=720 as the default.

Tests exec only the `_load_image_bytes` function from a generated notebook (no notebook
globals required, so the test is cross-platform)."""
import ast
import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "fap_gpu_colab_highres_3000.ipynb"


def _extract_load_image_bytes(nb_path: Path):
    nb = json.loads(nb_path.read_text())
    for cell in nb["cells"]:
        if cell["cell_type"] != "code": continue
        src = "".join(cell["source"])
        if "def _load_image_bytes" not in src: continue
        tree = ast.parse(src)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == "_load_image_bytes":
                sandbox = {"Image": Image, "io": io, "np": np, "__builtins__": __builtins__}
                exec(compile(ast.unparse(node), "<test>", "exec"), sandbox)
                return sandbox["_load_image_bytes"]
    raise AssertionError(f"_load_image_bytes not found in {nb_path.name}")


def _img_bytes(w, h, fill=(60, 130, 200)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), fill).save(buf, format="PNG")
    return buf.getvalue()


def test_upload_cap_downscales_oversized_source():
    """Source larger than upload_cap → resized to cap regardless of preset's max_resolution."""
    fn = _extract_load_image_bytes(NOTEBOOK_PATH)
    # 2000x1200 source → cap at 720 long side. Source is larger than max_resolution too
    # (1600) so the resize WOULD happen anyway; this test pins that the smaller cap wins.
    raw = _img_bytes(2000, 1200)
    target_rgb, _ = fn("test.png", raw, max_resolution=1600, sticker=False, upload_cap=720)
    h, w = target_rgb.shape[:2]
    # Long side = 720, plus 8% padding per side.
    pad = max(8, int(round(720 * 0.08)))
    expected_long_side_padded = 720 + 2 * pad
    assert max(h, w) == expected_long_side_padded, (
        f"long side after cap+padding should be {expected_long_side_padded}, got {max(h,w)}"
    )


def test_upload_cap_zero_disables_safety_cap():
    """upload_cap=0 disables the safety cap — only max_resolution applies."""
    fn = _extract_load_image_bytes(NOTEBOOK_PATH)
    # 1200x800 source, max_resolution=1600. With cap=0, source stays at 1200 (no resize).
    raw = _img_bytes(1200, 800)
    target_rgb, _ = fn("test.png", raw, max_resolution=1600, sticker=False, upload_cap=0)
    h, w = target_rgb.shape[:2]
    pad = max(8, int(round(1200 * 0.08)))   # 96
    assert max(h, w) == 1200 + 2 * pad, (
        f"upload_cap=0 should let the source through unmodified; got long side {max(h,w)}"
    )


def test_upload_cap_above_source_no_resize():
    """If source is smaller than upload_cap, it stays at its native size."""
    fn = _extract_load_image_bytes(NOTEBOOK_PATH)
    raw = _img_bytes(500, 400)   # well under 720
    target_rgb, _ = fn("test.png", raw, max_resolution=1600, sticker=False, upload_cap=720)
    h, w = target_rgb.shape[:2]
    pad = max(8, int(round(500 * 0.08)))   # 40
    assert max(h, w) == 500 + 2 * pad, (
        f"source under cap should stay at native size; got long side {max(h,w)}"
    )


def test_upload_cap_smaller_than_max_resolution_wins():
    """upload_cap < max_resolution → cap is the effective ceiling."""
    fn = _extract_load_image_bytes(NOTEBOOK_PATH)
    raw = _img_bytes(1500, 900)
    target_rgb, _ = fn("test.png", raw, max_resolution=1600, sticker=False, upload_cap=720)
    pad = max(8, int(round(720 * 0.08)))
    assert max(target_rgb.shape[:2]) == 720 + 2 * pad


def test_upload_cap_default_is_720():
    """The default upload_cap (when caller doesn't pass it) is 720."""
    fn = _extract_load_image_bytes(NOTEBOOK_PATH)
    raw = _img_bytes(1500, 900)
    # NO upload_cap kwarg — should use the default.
    target_rgb, _ = fn("test.png", raw, max_resolution=1600, sticker=False)
    pad = max(8, int(round(720 * 0.08)))
    assert max(target_rgb.shape[:2]) == 720 + 2 * pad, (
        "default upload_cap should be 720"
    )


def test_every_production_notebook_sets_UPLOAD_MAX_LONG_SIDE_to_720():
    """All production notebooks should ship with the same 720 default so users on any
    preset get the same VRAM safety. (Users can edit per-run if they want.)"""
    import re
    for nb_path in sorted(REPO_ROOT.glob("notebooks/fap_gpu_colab_*.ipynb")):
        nb = json.loads(nb_path.read_text())
        src_all = "\n".join("".join(c.get("source", []))
                            for c in nb["cells"] if c["cell_type"] == "code")
        m = re.search(r"UPLOAD_MAX_LONG_SIDE\s*=\s*(\d+)", src_all)
        assert m, f"{nb_path.name}: UPLOAD_MAX_LONG_SIDE not defined"
        assert int(m.group(1)) == 720, (
            f"{nb_path.name}: UPLOAD_MAX_LONG_SIDE = {m.group(1)}, expected 720 default"
        )
