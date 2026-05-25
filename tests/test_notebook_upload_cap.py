"""Regression tests pinning the upload cap as a ceiling on the FINAL padded canvas.

The notebook's `_load_image_bytes()` enforces a hard ceiling on what gets handed
to run_gpu so users can't accidentally OOM the GPU by uploading a huge source.
The v0.1.5 cap was implemented as a ceiling on the INPUT image, then padding
(~8% per side) was added on top — which inflated the final canvas past the cap
and OOM'd users despite a "safe" cap value. This test suite pins the FIXED
contract: cap is a ceiling on the final padded canvas (= what the engine sees),
not on the input image.

Real-world repro that prompted the fix: 800px source, MAX_RESOLUTION=1000,
UPLOAD_MAX_LONG_SIDE=720. Pre-fix: final canvas was 836x836, VRAM peaked at
99 GB on a 102 GB GPU (vs probe's 65 GB prediction). Post-fix: final ≤ 720,
probe matches reality.

Tests exec only the `_load_image_bytes` function from a generated notebook (no
notebook globals required, so the test is cross-platform)."""
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


def test_upload_cap_is_ceiling_on_final_padded_canvas():
    """THE core contract: cap caps the FINAL padded canvas (= what hits VRAM),
    not the input image. This is the bug the user OOM'd on — pre-fix, padding
    leaked past the cap by ~16% (720 cap → 836 final on the user's repro)."""
    fn = _extract_load_image_bytes(NOTEBOOK_PATH)
    raw = _img_bytes(2000, 1200)
    target_rgb, _ = fn("test.png", raw, max_resolution=1600, sticker=False, upload_cap=720)
    h, w = target_rgb.shape[:2]
    assert max(h, w) <= 720, (
        f"final padded long side must be ≤ cap (720); got {max(h, w)}. "
        f"Padding is leaking past the cap."
    )
    # Sanity: cap should be ACTIVELY engaged (else the test passes vacuously).
    assert max(h, w) > 600, (
        f"cap should be engaged near 720; got tiny {max(h, w)} — fixture/cap mismatch"
    )


def test_user_repro_800px_source_max_1000_cap_720():
    """Exact reproduction of the user's OOM repro that drove this fix.
    800px source, MAX_RESOLUTION=1000, UPLOAD_MAX_LONG_SIDE=720."""
    fn = _extract_load_image_bytes(NOTEBOOK_PATH)
    raw = _img_bytes(800, 800)
    target_rgb, _ = fn("test.png", raw, max_resolution=1000, sticker=False, upload_cap=720)
    h, w = target_rgb.shape[:2]
    assert max(h, w) <= 720, (
        f"user-repro (800 src / max=1000 / cap=720): final={max(h, w)}, expected ≤720. "
        f"Pre-fix this was 836 and OOM'd a 102 GB GPU."
    )


def test_upload_cap_zero_disables_safety_cap():
    """upload_cap=0 disables the safety cap entirely. Source goes through at
    its native resolution (modulo max_resolution), then full padding added.
    Final canvas = source + 2*pad with no ceiling enforced."""
    fn = _extract_load_image_bytes(NOTEBOOK_PATH)
    raw = _img_bytes(1200, 800)
    target_rgb, _ = fn("test.png", raw, max_resolution=1600, sticker=False, upload_cap=0)
    h, w = target_rgb.shape[:2]
    pad = max(8, int(round(1200 * 0.08)))
    assert max(h, w) == 1200 + 2 * pad, (
        f"upload_cap=0 should let the source through unmodified; got long side {max(h,w)}"
    )


def test_upload_cap_above_source_no_resize():
    """If source is well under the cap, no input resize. Final = source + 2*pad
    (which itself stays under the cap by construction)."""
    fn = _extract_load_image_bytes(NOTEBOOK_PATH)
    raw = _img_bytes(500, 400)   # well under 720 even with padding
    target_rgb, _ = fn("test.png", raw, max_resolution=1600, sticker=False, upload_cap=720)
    h, w = target_rgb.shape[:2]
    pad = max(8, int(round(500 * 0.08)))
    assert max(h, w) == 500 + 2 * pad, (
        f"source under cap should stay at native size; got long side {max(h,w)}"
    )


def test_upload_cap_smaller_than_max_resolution_wins():
    """upload_cap < max_resolution → cap is the effective ceiling on FINAL canvas."""
    fn = _extract_load_image_bytes(NOTEBOOK_PATH)
    raw = _img_bytes(1500, 900)
    target_rgb, _ = fn("test.png", raw, max_resolution=1600, sticker=False, upload_cap=720)
    assert max(target_rgb.shape[:2]) <= 720


def test_upload_cap_default_is_720():
    """Default upload_cap (when caller doesn't pass it) is 720. Final ≤ 720."""
    fn = _extract_load_image_bytes(NOTEBOOK_PATH)
    raw = _img_bytes(1500, 900)
    target_rgb, _ = fn("test.png", raw, max_resolution=1600, sticker=False)
    assert max(target_rgb.shape[:2]) <= 720, (
        f"default cap should be 720; got final {max(target_rgb.shape[:2])}"
    )


def test_upload_cap_sticker_mode_also_respects_cap():
    """Sticker mode uses a different padding code path (RGB + alpha mask)
    — must respect the cap identically."""
    fn = _extract_load_image_bytes(NOTEBOOK_PATH)
    # Construct an RGBA source so sticker mode is valid.
    buf = io.BytesIO()
    img = Image.new("RGBA", (1500, 1500), (128, 128, 128, 255))
    img.save(buf, format="PNG")
    raw = buf.getvalue()
    target_rgb, alpha = fn("test.png", raw, max_resolution=1600, sticker=True, upload_cap=720)
    assert max(target_rgb.shape[:2]) <= 720
    assert alpha is not None
    # alpha mask must have the same dims as the RGB target.
    assert alpha.shape == target_rgb.shape[:2]


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
