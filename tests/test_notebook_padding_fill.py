"""Regression test pinning the opaque-mode notebook padding fill = source mean color.

The notebook's `_load_image_bytes()` helper pads the source by 8% on each side so shapes
that land on the outermost rows/cols of the content stay several pixels inside the actual
canvas edge (FH6's vinyl renderer treats edge-touching shapes as unbounded otherwise).

ORIGINAL bug: opaque mode padded with hardcoded `(255, 255, 255)` solid white. The engine
inits the opaque canvas to `mean(target)`, so an all-white margin has nonzero per-pixel
loss everywhere in the padding region and greedy/polish dutifully spent ~100-300 shapes
painting a phantom solid-white border before reaching the actual content. Confirmed by a
user-shipped 3000-shape inject where they had to manually delete the white border ellipses.

THE FIX: compute the source's mean color BEFORE padding, fill the padding with that
mean. Then the padded region matches the engine's canvas-init color exactly → zero per-
pixel loss outside the source → greedy never places candidates there → full shape budget
goes into the actual content.

This test extracts the generated `_load_image_bytes` function from a production preset
notebook, runs it on a deterministic synthetic image, and asserts the padding pixels
are the source's mean color (NOT (255, 255, 255)).
"""
import io
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "fap_gpu_colab_highres_3000.ipynb"


def _extract_load_image_bytes_from_notebook(nb_path: Path):
    """Find the `_load_image_bytes` function definition in the notebook's code cells and
    exec ONLY that def into a sandbox. The notebook cell that defines it also CALLS it
    afterward with notebook-level globals (SOURCE_IMAGE_NAME etc.) that aren't defined
    in our test context — so we ast-extract just the function and skip the call."""
    import ast
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if "def _load_image_bytes" not in src:
            continue
        # Parse and pull out ONLY the function def (skip the call after it).
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == "_load_image_bytes":
                fn_src = ast.unparse(node)
                sandbox: dict = {
                    "Image": Image, "io": io, "np": np,
                    "__builtins__": __builtins__,
                }
                exec(compile(fn_src, "<notebook-_load_image_bytes>", "exec"), sandbox)
                fn = sandbox["_load_image_bytes"]
                assert callable(fn)
                return fn
    raise AssertionError(f"_load_image_bytes not found in {nb_path.name}")


def _make_test_image_bytes(width=200, height=100, fill=(60, 130, 200)) -> bytes:
    """Solid-color test image whose mean is exactly `fill` — makes the padding-color
    assertion deterministic."""
    img = Image.new("RGB", (width, height), fill)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_opaque_padding_fills_with_source_mean_not_white():
    """The padding in OPAQUE mode must be the SOURCE's mean color. If it's (255,255,255)
    the regression has returned and greedy will waste shape budget painting the border."""
    _load_image_bytes = _extract_load_image_bytes_from_notebook(NOTEBOOK_PATH)
    # Solid blue-ish source → mean color is exactly (60, 130, 200).
    raw = _make_test_image_bytes(width=200, height=100, fill=(60, 130, 200))
    target_rgb, alpha_mask = _load_image_bytes("test.png", raw, max_resolution=1600, sticker=False)
    assert alpha_mask is None, "opaque mode should return alpha_mask=None"
    # The first row of the returned array IS padding (above the pasted source). Inspect a
    # padding pixel near the top-left corner.
    pad_pixel = tuple(int(c) for c in target_rgb[2, 2])
    assert pad_pixel == (60, 130, 200), (
        f"opaque padding pixel is {pad_pixel}, expected source-mean (60,130,200). "
        f"If you see (255,255,255) here, the white-padding regression is back — greedy "
        f"will waste shape budget painting a phantom solid-white margin instead of "
        f"placing shapes in the actual content."
    )


def test_opaque_padding_matches_engine_canvas_init():
    """The padding color must equal `mean(target_rgb)` so the engine's canvas-init
    (mean of target) covers the padding region with zero per-pixel loss. If the padding
    is even slightly off from the mean, the engine will spend SOME shapes correcting it."""
    _load_image_bytes = _extract_load_image_bytes_from_notebook(NOTEBOOK_PATH)
    raw = _make_test_image_bytes(width=200, height=100, fill=(77, 144, 33))
    target_rgb, _ = _load_image_bytes("test.png", raw, max_resolution=1600, sticker=False)
    # The engine computes canvas init as mean over the WHOLE target. With padding = source
    # mean = source-region color, the whole-target mean MUST equal that color too.
    whole_target_mean = tuple(int(round(c)) for c in target_rgb.reshape(-1, 3).mean(axis=0))
    pad_pixel = tuple(int(c) for c in target_rgb[2, 2])
    assert whole_target_mean == pad_pixel, (
        f"target-mean {whole_target_mean} != padding pixel {pad_pixel}. "
        f"If these drift apart the engine's canvas init will have nonzero loss in the "
        f"padded region, defeating the fix."
    )


def test_opaque_padding_does_not_contaminate_source_pixels():
    """The padding fix must NOT touch the source's interior pixels. A pixel deep inside
    the source region (well away from the padding margin) must equal the original color."""
    _load_image_bytes = _extract_load_image_bytes_from_notebook(NOTEBOOK_PATH)
    raw = _make_test_image_bytes(width=200, height=100, fill=(60, 130, 200))
    target_rgb, _ = _load_image_bytes("test.png", raw, max_resolution=1600, sticker=False)
    # Deep interior — well past any padding margin.
    h, w = target_rgb.shape[:2]
    center = tuple(int(c) for c in target_rgb[h // 2, w // 2])
    assert center == (60, 130, 200), (
        f"source-interior pixel is {center}, expected unchanged (60,130,200). "
        f"The padding fix must not modify the source content."
    )


def test_sticker_padding_unchanged_at_zero():
    """STICKER MODE padding stays at (0,0,0) — sticker's alpha-mask gate already rejects
    candidate shapes that don't overlap the silhouette, so the padding color is benign.
    Don't touch it; this test is here to catch over-eager refactoring that changes both
    branches at once."""
    _load_image_bytes = _extract_load_image_bytes_from_notebook(NOTEBOOK_PATH)
    # Sticker mode needs an RGBA source. Make one with a colored disc on transparent bg.
    rgba = np.zeros((100, 200, 4), dtype=np.uint8)
    rgba[20:80, 40:160, :3] = (60, 130, 200)
    rgba[20:80, 40:160, 3] = 255
    buf = io.BytesIO()
    Image.fromarray(rgba).save(buf, format="PNG")
    target_rgb, alpha_mask = _load_image_bytes("sticker.png", buf.getvalue(),
                                                max_resolution=1600, sticker=True)
    assert alpha_mask is not None, "sticker mode should return an alpha mask"
    pad_pixel = tuple(int(c) for c in target_rgb[2, 2])
    assert pad_pixel == (0, 0, 0), (
        f"sticker padding is {pad_pixel}, expected (0,0,0). The sticker padding is "
        f"a documented design choice — the alpha gate rejects candidate shapes outside "
        f"the silhouette, so its color is benign. Don't refactor both branches at once."
    )
