# Forza Designer 6+ (FD6)

<p align="center">
  <img src="tools/SplashScreen.gif" alt="Forza Designer 6 splash" width="600"/>
</p>

<p align="center">
  <a href="https://youtu.be/8LGvE7O9aeg">
    <img src="https://img.youtube.com/vi/8LGvE7O9aeg/maxresdefault.jpg" alt="Watch the FD6 trailer" width="600"/>
  </a>
  <br/>
  <sub><a href="https://youtu.be/8LGvE7O9aeg">▶ Trailer / tutorial on YouTube</a></sub>
  <br/>
  <sub><a href="https://github.com/tokyubevoxelverse/ForzaDesignerRadioMaker/releases/tag/0.0.1-Alpha">🎵 Forza Designer Radio Maker — v0.0.1-Alpha release</a></sub>
</p>

<p align="center">
  <img src="Pink.png" alt="FD6 badge" width="128"/>
</p>

> Convert any image into a vinyl group for **Forza Horizon 3, 4, 5, or 6**, or into a livery for **Assetto Corsa Competizione**.

**Repo:** https://github.com/NcaMoq/ForzaDesigner6 · **License:** MIT · **Windows 10/11 x64**

---

## Install

1. Download the `FD6MultiSupport` package from [Releases](https://github.com/NcaMoq/ForzaDesigner6/releases).
2. Double-click — no installer, no admin rights. Windows SmartScreen → "More info" → "Run anyway".

Source build: Python 3.10+, `pip install -r requirements.txt`. Microsoft Visual C++ Redistributable is normally already installed; if FD6 fails to launch, grab it [here](https://aka.ms/vs/17/release/vc_redist.x64.exe).

GPU generation is bundled in the packaged build. For source runs, install a CUDA-enabled `torch` build from the official PyTorch instructions, then pick `Compute = GPU (CUDA)`.

---

## How to use

**Generate:**

1. Launch `FD6MultiSupport.exe`. Click **Upload Image…** and pick a JPEG/PNG.
2. Right panel: pick a **Profile** (`balanced` recommended) and **Stop at shapes** (1500 or 3000 typical).
3. Optional: set **Compute** to `GPU (CUDA)` to run generation on the GPU.
4. Click **Start**. Watch the live preview rebuild your image. The JSON auto-saves next to your source image when done.

**Line guide (optional):**

- Enable **Use line guide** to steer early candidates toward outlines.
- Pick a guide image, or place an ONNX model at `models/line_guide.onnx` and install the optional `line-guide` or `line-guide-gpu` dependency when running from source.
- A small reference ONNX graph can be generated with `python tools\create_line_guide_sobel_onnx.py models\line_guide.onnx`.
- If the model needs BGR input, `-1.0` to `1.0` input, a fixed output channel, or inverted output, place `models/line_guide.json` next to it. See `models/line_guide.example.json`.
- Normal packages do not bundle model weights or ONNX Runtime. If ONNX is unavailable or the model fails to load, FD6 continues with the existing generation path.
- From source, install `onnxruntime` with `pip install -e ".[line-guide]"` or `onnxruntime-gpu` with `pip install -e ".[line-guide-gpu]"`.
- On Windows, DirectML can be used with `pip install -e ".[line-guide-directml]"` when CUDA provider dependencies are unavailable.
- Packaged builds can use the external guide image path immediately.
- The ONNX-enabled package is built with `.\build_exe_onnx.bat` as `FD6MultiSupport_onnx`. It bundles ONNX Runtime DirectML and CPU providers, plus the generated reference model under `_internal\models\line_guide.onnx`.
- If a copied packaged build reports `No module named 'onnxruntime'`, it is the normal package, not the ONNX-enabled package.
- Run `python tools/line_guide_benchmark.py --output docs/line_guide_measurements.json` to record speed, VRAM, RMS, edge F1, gradient error, and SSIM for line guide on/off. Add `--model models\line_guide.onnx --output docs\line_guide_measurements_onnx.json` to measure the ONNX extraction path.

**Inject:**

1. Launch your target game (**Forza Horizon 3, 4, 5, or 6**) and open the Vinyl Group editor.
2. Load a vinyl group with **at least N spheres**, where N is the shape count of your JSON. Keep saved 1500-sphere / 3000-sphere templates for reuse — fastest path is a fresh untouched template.
3. In FD6: pick your target from the **Target** dropdown, click **Upload JSON**, then **Inject into [game]**.
4. Watch the dialog progress. **Do not click anything in FD6 or the game during injection** — interacting can reallocate the in-game memory mid-write and fail the operation.
5. When the status turns 🟢 green, the vinyl group has been painted with your shapes.

**Re-injecting onto an already-painted template** works too — the locator falls back to an RTTI vtable scan when the fresh-sphere fingerprint misses. Expect an extra 2–5 minutes on the first re-injection scan per game session; subsequent ones are instant.

---

## Assetto Corsa Competizione

ACC support is **file-based** — no live memory injection. FD6 writes a two-file pair (livery `.json` + skin folder with the PNG textures) directly into ACC's user-data directory, and the in-game livery picker reads it on next launch.

1. Pick a target **car** from the ACC tab (FD6 ships a catalog of every base-game car). The catalog drives which texture slots are available.
2. **Upload Image…** to load the artwork you want printed on the car. FD6 currently writes the source PNG into the upper-left of the car's 4096×4096 livery sheet. UV-aware decal placement per car model is planned for v0.3.6.
3. Click **Export**. FD6 writes:
    - `Documents/Assetto Corsa Competizione/Customs/Cars/<car>/<livery>.json` — the metadata ACC's picker indexes.
    - `Documents/Assetto Corsa Competizione/Customs/Liveries/<livery>/decals.png` (plus `sponsors.png`) — the actual texture assets.
4. Launch ACC → Car Picker → Customs tab. The new livery appears immediately.

ACC export does not require ACC to be running and does not touch ACC's process memory — it's a pure file write to your Documents folder, so the ban-risk caveats that apply to Forza injection do **not** apply here.

---

## ✅ Do / ❌ Don't

- **Do** open the game's vinyl editor *before* clicking Inject.
- **Do** keep one or two sphere templates saved for fast re-use.
- **Do** wait for the green status — large-game scans can take minutes.
- **Don't** edit, add, delete, or move shapes in-game during an active injection.
- **Don't** click anything in FD6 during the RTTI fallback phase. Windows may label it "Not Responding"; it isn't — let it finish.
- **Don't** close the game mid-injection or run FD6 as admin.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| "No confident match" error | Vinyl editor not open, or template doesn't have enough spheres for the JSON's shape count. Load a bigger template. |
| Re-injection scan looks stalled | RTTI fallback phase can run silently for 2–5 min on a large game. The dialog resumes once a candidate is located. |
| Shapes offset or wrong scale | File an issue with the JSON, the source image, and a screenshot of the in-game result. |
| Game patched, all candidates rejected | The per-game struct offset may have shifted. Open an issue — the profile in `fd6/inject/game_profiles.py` needs updating. |

---

## Build from source

```powershell
git clone https://github.com/NcaMoq/ForzaDesigner6.git
cd ForzaDesigner6
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python -m fd6                # run from source
pytest                       # run tests
.\build_exe.bat              # → dist/FD6MultiSupport\
.\build_exe_onnx.bat         # → dist/FD6MultiSupport_onnx\
```

For editable development installs:

```powershell
pip install -e ".[dev]"
pip install -e ".[line-guide-directml]"   # optional ONNX DirectML support
```

Before publishing from this repository, run through `docs/PUBLISHING.md`.
The upstream MIT notice is listed in `NOTICE`. Bundled media, fonts, runtime
packages, and optional models are listed in `THIRD_PARTY_NOTICES.md` for license
review.

---

## Credits

Inspired by:

- **forza-painter** by `the_adawg` (FH4/FH5 — MIT)
- **geometrize-lib** by Sam Twidale (MIT)
- **Primitive** by Michael Fogleman (MIT)
- **bvzrays**' publicly available [forza-painter-fh6](https://github.com/bvzrays/forza-painter-fh6) (MIT). FD6 uses his published research for: the CLiveryGroup struct field offsets, the (X, −Y) position convention and 63 / 127 scale divisors, the confirmation that FH5 and FH6 share the same struct layout (which made FH3/FH4/FH5 support feasible), and the MSVC RTTI vtable-scan locator approach (FD6's `rtti_locator.py` looks up `.?AVCLiveryGroup@@` by C++ type). FD6 does **not** ship or load community-distributed `forza-codes.dat` runtime pattern files — only the single baseline RTTI class name is hardcoded.

The FH6 vinyl-group memory layout was independently reverse-engineered from scratch for this project against FH6 build 354.221 and later cross-validated against bvzrays' research; the two derivations matched on every field offset. The sphere-template injection workflow (load a fresh sphere group → fingerprint-locate it in memory → overwrite each sphere's bytes in place) and the strict 5/5 + 95% full-table validation gate are FD6 originals; both stay as the **primary** locator path, with RTTI vtable scan as the **secondary** fallback for re-injection onto already-painted templates.

---

## Disclaimer

**Use entirely at your own risk.** FD6 modifies the memory of a running Forza Horizon process to populate vinyl-group shapes. It does not patch the game executable, install drivers, modify save files, or attempt to bypass any anti-cheat or DRM system. However, **memory modification of a live game process may be interpreted by Microsoft, Xbox Live, or the game's publisher (Turn 10 / Playground Games) as a violation of the Microsoft Services Agreement, the Xbox Community Standards, or the relevant Forza title's terms of use. Doing so may result in temporary suspension or permanent ban of your Xbox / Microsoft account, loss of access to purchased games, online services, achievements, and any content created with FD6.**

The authors and contributors of Forza Designer 6 **accept no responsibility or liability whatsoever** for any consequences arising from the use of this software. By downloading, building, installing, or running FD6 you acknowledge these risks and accept them in full. This tool is provided as-is, MIT-licensed, with no warranties of any kind. Not affiliated with, endorsed by, or sponsored by Turn 10 Studios, Playground Games, Microsoft, Xbox, or any official Forza brand.

## License

[MIT](LICENSE). This repository is based on the MIT-licensed upstream project
listed in [NOTICE](NOTICE). Third-party and bundled-asset notes are listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
