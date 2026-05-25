# GPU shape-gen notebook quickstart

The Colab notebooks turn any image into a vinyl-group JSON the Windows EXE can inject into Forza Horizon 6 (or any FH3-6 title). No local install — they run on Google's free T4 GPU.

This guide walks you from "I have an image" to "I have a JSON ready for `Upload JSON → Inject`" in about 10 minutes.

---

## Pick a notebook

Match your image's complexity to a shape budget. **More shapes ≠ better** beyond a point — pick the smallest budget that covers your detail level so the run finishes faster.

| Notebook | Shapes | What it's tuned for | Recommended VRAM |
|---|---|---|---|
| [`fap_gpu_colab_lineart_400.ipynb`](../notebooks/fap_gpu_colab_lineart_400.ipynb) | 400 | Lineart, kanji / logos / typography, silhouettes, high-contrast vector art | ≥ 8 GB (T4 OK) |
| [`fap_gpu_colab_headshots_700.ipynb`](../notebooks/fap_gpu_colab_headshots_700.ipynb) | 700 | Character headshots, portraits, single-subject illustrations | ≥ 12 GB (T4 OK) |
| [`fap_gpu_colab_medium_1000.ipynb`](../notebooks/fap_gpu_colab_medium_1000.ipynb) | 1000 | General-purpose mid-fidelity — most use cases | ≥ 16 GB (T4 OK with 720px max) |
| [`fap_gpu_colab_highres_3000.ipynb`](../notebooks/fap_gpu_colab_highres_3000.ipynb) | 3000 | Maximum-detail builds (FH6's per-vinyl-group ceiling). Full scenes, detailed characters | ≥ 24 GB (Pro V100 / L4 / A100) |
| `fap_gpu_colab_shapes_medium_1000.ipynb` | 1000 | **EVAL** — triangle + rotated_rectangle enabled alongside ellipse | ≥ 16 GB |
| `fap_gpu_colab_shapes_highres_3000.ipynb` | 3000 | **EVAL** — multi-shape at full budget | ≥ 24 GB |
| `fap_test_harness.ipynb` | varies | Internal stage-by-stage parity verification (engine, CPU, injector) — not for users | varies |

> **EVAL notebooks** ship the experimental multi-shape mode (ellipse + triangle + rectangle). Quality results pending — start with the ellipse-only mainline notebooks unless you're specifically validating multi-shape.

---

## Run it

### 1. Open in Colab

Click any notebook's `.ipynb` file in this repo, then click the **Open in Colab** badge at the top of the GitHub viewer. Or paste the GitHub URL into [colab.research.google.com](https://colab.research.google.com) → File → Open notebook → GitHub tab.

### 2. Enable the GPU

**Runtime → Change runtime type → Hardware accelerator: GPU.** Without this, the engine setup cell falls back to CPU and runs ~50× slower. T4 is the free tier and is fine for 400/700/1000-shape notebooks. Pro tiers (V100 / L4 / A100) are required for 3000-shape highres.

### 3. Mount Google Drive (recommended)

Section 4 of each notebook (the **Mount Google Drive** cell) prompts you to authorize Drive. Once authorized, every output (JSON + render PNG + intermediate checkpoints) saves directly to your Drive as soon as it's produced — so a Colab session reset or disconnect can't lose your work. Skipping this means outputs only live in the runtime filesystem and vanish on disconnect.

### 4. Upload an image

Section 5: click the **Choose Files** button that appears and pick a PNG or JPG.

**Sticker mode (transparent cutout):** if your image is RGBA with the background already cut out (eg a character on transparent), open the Configure cell (Section 7) and set `STICKER_MODE = True`. The engine then preserves the alpha and only puts shapes where the silhouette is.

### 5. Pick POSTERIZE_LEVELS

Section 6 (**Posterize preview**) shows your image at 4 posterize levels with a ★ recommendation. Posterize is a color-banding pre-pass that helps the engine commit to a smaller palette — too high washes out detail, too low banding looks crude.

Use the recommended ★ value unless your image has fine color detail (anime cels, lots of subtle gradients) that the recommendation washes out — in that case bump 1-2 levels higher and rerun.

### 6. Configure knobs (Section 7)

The preset defaults are tuned for the notebook's target use. Edit any of:
- `NUM_SHAPES` — total shape count (matches the notebook name; bumping rarely helps beyond the preset).
- `MAX_RESOLUTION` — max long-side pixels of the **final padded canvas**. Lower this if you OOM (see [VRAM tiers](#vram-tiers)).
- `POSTERIZE_LEVELS` — from Section 6.
- `STICKER_MODE` — `True` for transparent inputs.

### 7. Resolution planner (Section 8)

Hard-gates the resolution choice against your GPU's VRAM. Shows a table of MAX_RESOLUTION → final canvas → predicted peak VRAM → fit verdict. **If it says "STOP: MAX_RESOLUTION=N needs ~X GB but only ~Y GB is free,"** edit Configure (Section 7) to the recommended MAX_RESOLUTION it suggests and re-run.

### 8. Run (Section 9)

This is the heavy cell. Progress prints every 50 shapes. For a 1000-shape notebook at 720px MAX_RESOLUTION on a T4, expect ~5-10 minutes. On a V100 / L4 the same finishes in 1-3 minutes. The 3000-shape highres notebook takes 30-90 minutes depending on GPU.

Outputs land at `MyDrive/forza_abyss_painter/<filename_base>_<NUM_SHAPES>.json` and `..._render.png`.

### 9. Download

Section 10 displays the result inline so you can compare to the source. Section 11 (optional) pushes a browser download of the JSON + PNG — your Drive already has them.

---

## VRAM tiers

### Free Colab (T4, 16 GB)

| Notebook | Works at MAX_RESOLUTION ≤ | Notes |
|---|---|---|
| `lineart_400` | 1200 | Comfortable, fast (~3 min) |
| `headshots_700` | 1000 | Comfortable |
| `medium_1000` | **720** | Tight; use the auto-resize default. Bump down to 600 if you OOM. |
| `highres_3000` | ❌ | Won't fit on T4. Use Colab Pro V100/L4 or paid alternative. |
| `shapes_medium_1000` | 600 | Multi-shape adds memory overhead |

### Colab Pro tiers

| Tier | VRAM | Recommended for |
|---|---|---|
| Standard CPU/GPU | 16 GB (T4) | Same as Free tier |
| Premium V100 | 16 GB | Same as T4, but ~2× faster |
| L4 | 22 GB | 1000-shape at full 1200px res; 3000-shape at 800px |
| A100 | 40 GB | Anything, including 3000-shape at full 1600px res |

### Non-Colab alternatives

Any cloud GPU service with PyTorch 2+ and ≥ 16 GB VRAM works — the notebook installs the package from this repo, no Colab-specific dependencies. Tested informally on:

- **Modal** (`modal.com`) — fast cold start, pay-per-second
- **RunPod** — cheap A100 spot instances
- **vast.ai** — cheapest A100s but variable reliability
- **Databricks** (your university or org may have free GPU credits)

For non-Colab runs, replace the `from google.colab import files` / `from google.colab import drive` calls with local filesystem reads. The rest of the engine is platform-independent.

---

## OOM recovery

If the Run cell crashes with `CUDA out of memory`:

1. **Run the Cleanup cell (Section 3) first.** Frees Python's reference to CUDA tensors so the next attempt sees free memory.
2. **Lower `MAX_RESOLUTION` in Configure (Section 7).** Try the next step down in the planner table. Memory cost scales roughly with `MAX_RESOLUTION²`, so halving the resolution quarters the memory.
3. **Re-run from Configure (Section 7) onwards.** Don't re-run Setup unless you've restarted the runtime — the engine cell takes ~30s to redefine.
4. **If Cleanup can't get allocated under ~10 GB**, restart the runtime entirely (Runtime → Restart runtime) and re-run from Section 1. Drive stays mounted across restarts.

For persistent OOM at low MAX_RESOLUTION, try:
- Reducing `RANDOM_SAMPLES` in Configure (eg `24576 → 12288 → 6144`). This is the candidate batch size; smaller batches mean each shape commit takes longer but uses much less peak memory.
- Switching to a smaller-budget notebook (eg `medium_1000` → `headshots_700`).

---

## After the JSON is generated

The output is byte-compatible with the EXE's **Upload JSON** flow:

1. Copy the `..._render.png` from Drive — open it locally to confirm it looks like what you wanted.
2. Copy the matching `.json` to your Windows machine.
3. In `ForzaAbyssPainter.exe`: **Upload JSON**, pick the file, click **Inject into FH6**.
4. Pre-inject dialog asks for template size — pick the matching one (eg 1000 spheres if your JSON is a 1000-shape build).
5. Wait for the inject to complete (typically 30-60s including the heap scan). Your design appears in the FH6 vinyl editor.

See the main [README troubleshooting](../README.md#troubleshooting) section for inject-time failure modes.

---

## Questions / issues

File an issue at [github.com/whykusanagi/forza-abyss-painter/issues](https://github.com/whykusanagi/forza-abyss-painter/issues). Include:

- Which notebook + which preset
- Your GPU (Free T4 / Pro V100 / etc.)
- The full error traceback if any
- Your MAX_RESOLUTION / NUM_SHAPES / source image dimensions
