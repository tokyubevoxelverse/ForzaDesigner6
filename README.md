# Forza Abyss Painter

<p align="center">
  <img src="assets/forza_abyss_painter_logo.png" alt="Forza Abyss Painter" width="200"/>
</p>

> Vinyl-design tool for **Forza Horizon 3, 4, 5, 6** and **Assetto Corsa Competizione**.
> Convert any image into a vinyl group ready to inject, or a livery ready to paint.

**Repo:** https://github.com/whykusanagi/forza-abyss-painter · **License:** MIT · **Windows 10/11 x64**

---

## What is Forza Abyss Painter

A clean, branded fork of [tokyubevoxelverse/ForzaDesigner6](https://github.com/tokyubevoxelverse/ForzaDesigner6) v0.3.5 with the iterative improvements from `whykusanagi/ForzaDesigner6` carried over as final products:

- **Injector performance fix** — sampled revalidation cuts ~1.83x syscall overhead on a 3000-shape inject (matches `forza-painter-fh6` throughput).
- **GPU shape-gen pipeline** — Colab notebooks turn your image into a 200-, 1000-, or 3000-shape JSON on a CUDA box; JSON output is byte-identical to upstream's CPU engine.
- **`polish_freeze_geometry` production mode** — verified 1000-shape parity with upstream while preserving sparkle/detail polish moves.
- **Dark theme reskin** — new `Abyss` theme (pure black + deep purple + magenta) is the default. All seven legacy themes (Default / Japanese Blossoms / Purple Passion / Matrix Racing / Odaiba Bay / Hokkaido Sunset / Cherry Soda Pop) remain available.
- **Rebranded EXE** — `ForzaAbyssPainter.exe`, new icon (cheek chibi).

JSON save-files remain byte-compatible with upstream Forza Designer 6, so existing JSONs load and inject without modification.

## Supported games

- **Forza Horizon 6** — full memory injection (vinyl groups: position, scale, rotation, color).
- **Forza Horizon 3 / 4 / 5** — memory injection via shared FH-engine profiles.
- **Assetto Corsa Competizione** — file-based PNG livery export to `Documents/Assetto Corsa Competizione/Customs/Liveries/...`.

## Install

Releases are not yet published (initial v0.1.0 build pending). For now, build from source:

```bash
git clone https://github.com/whykusanagi/forza-abyss-painter.git
cd forza-abyss-painter
pip install -r requirements.txt
pip install -e .
python -m forza_abyss_painter
```

Windows EXE build (PyInstaller):

```cmd
build_exe.bat
:: produces dist\ForzaAbyssPainter.exe
```

Once v0.1.0 ships, download `ForzaAbyssPainter.exe` from [Releases](https://github.com/whykusanagi/forza-abyss-painter/releases) — single-file, no installer.

## GPU shape-gen notebooks

Six production presets + one test harness, all in `notebooks/`. Run any of them in Google Colab on a free T4 GPU:

| Notebook | Shapes | Target use |
|---|---|---|
| `fap_gpu_colab_lineart_400.ipynb` | 400 | Lineart / typography / silhouettes |
| `fap_gpu_colab_headshots_700.ipynb` | 700 | Character headshots / portraits |
| `fap_gpu_colab_medium_1000.ipynb` | 1000 | General-purpose mid-fidelity |
| `fap_gpu_colab_highres_3000.ipynb` | 3000 | Maximum-detail builds (FH6 vinyl-group ceiling) |
| `fap_gpu_colab_shapes_medium_1000.ipynb` | 1000 | EVAL: triangle + rotated_rectangle enabled |
| `fap_gpu_colab_shapes_highres_3000.ipynb` | 3000 | EVAL: multi-shape at full budget |
| `fap_test_harness.ipynb` | varies | Stage-by-stage parity verification (engine, CPU, injector) |

Each notebook installs from this repo (`pip install git+https://github.com/whykusanagi/forza-abyss-painter.git@main`), inlines the GPU engine, and saves output JSON + PNG straight to Google Drive so a Colab session reset can't lose your work.

## Differences from upstream

| | Upstream FD6 v0.3.5 | Forza Abyss Painter v0.1.0 |
|---|---|---|
| Injector revalidation | per-shape (5/5 every shape) | sampled every 250 shapes (~1.83x syscall reduction) |
| GPU shape-gen | not available | full pipeline + 6 Colab notebooks |
| Polish geometry handling | float drift during polish | `polish_freeze_geometry=True` default; byte-parity at 1000 shapes |
| Default theme | "Default" (grey + blue) | "Abyss" (black + purple + magenta) |
| Available themes | 7 | 8 (Abyss + the original 7) |
| Binary name | `FD6MultiSupport.exe` | `ForzaAbyssPainter.exe` |
| JSON wire format | `"format": "fd6.shapes"` | unchanged (full save-file compatibility) |

## Credits

- **[tokyubevoxelverse/ForzaDesigner6](https://github.com/tokyubevoxelverse/ForzaDesigner6)** — upstream project. All multi-game suite scaffolding (FH3-6 + ACC), the original injector, the CPU shape generator, theme system, GUI, and most of what makes this work.
- **[bvzrays/forza-painter-fh6](https://github.com/bvzrays/forza-painter-fh6)** — injector performance research; the sampled-revalidation approach is adapted from their `forza-painter` per-shape loop.
- **forza-painter (the_adawg)** — original `forza-painter` tooling that inspired upstream FD6.
- **geometrize-lib (Sam Twidale)** + **Primitive (Michael Fogleman)** — the underlying greedy shape-fitting algorithm.

## License

MIT, inherited from upstream. See [LICENSE](LICENSE).
