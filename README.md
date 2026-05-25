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

**Easiest — download the prebuilt EXE.** Single-file Windows 10/11 x64 executable, no installer:

1. Go to [Releases](https://github.com/whykusanagi/forza-abyss-painter/releases) and download the latest `ForzaAbyssPainter.exe`.
2. Right-click the file → **Run as administrator** (required — see [Troubleshooting](#troubleshooting) for why).
3. On first run, Windows SmartScreen warns about an unsigned binary. Click **More info** → **Run anyway**. The release EXE is built by GitHub Actions directly from this repo's source (see the workflow run linked in each release for the exact build commit).

**Build from source (advanced):**

```bash
git clone https://github.com/whykusanagi/forza-abyss-painter.git
cd forza-abyss-painter
pip install -r requirements.txt
pip install -e .
python -m forza_abyss_painter
```

Windows EXE build:

```cmd
build_exe.bat
:: produces dist\ForzaAbyssPainter.exe
```

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

📖 **Full notebook walkthrough:** [docs/NOTEBOOK_QUICKSTART.md](docs/NOTEBOOK_QUICKSTART.md) — which preset to pick, upload→run→download flow, OOM recovery, Free vs Pro Colab VRAM tiers.

## Troubleshooting

### Inject can't find Forza Horizon 6

> *"Forza Horizon 6 is not running (looked for: forzahorizon6.exe, ForzaHorizon6-Win64-Shipping.exe). Start the game and try again."*

- **Most common cause: not running as administrator.** FH6 ships as a UWP package; non-elevated processes can't open a handle on it. Right-click `ForzaAbyssPainter.exe` → **Run as administrator** and retry. Inject will fail silently the same way if launched normally.
- Game actually not running? Open FH6 and wait for the main menu before clicking Inject.
- If you renamed the EXE or it's running through a wrapper, the process name might not match. The injector looks for `forzahorizon6.exe` (UWP build) and `ForzaHorizon6-Win64-Shipping.exe` (alternate). Open Task Manager → Details → confirm one of those names is alive.

### Inject scan takes 30+ seconds

That's expected on current FH6 builds. The fast-mode signature chain falls through to the heap-fingerprint scan because the chain anchor in the game image is currently an incidental `.rdata` match, not the live chain root (see the inject log for the full per-gate diagnostic). The heap scan typically completes in ~30s for a 1000-shape template, longer for larger templates. We're tracking a fix for the signature chain itself.

### "Active vinyl group is GROUPED" error

> *"Active vinyl group is GROUPED. Multiple slots alias the same layer blob… In FH6: Select All → Ungroup in the vinyl editor, then retry."*

In the vinyl editor, select every shape (Ctrl+A or the in-game select-all) and choose **Ungroup**. Grouped templates have shared underlying memory — writing to them would corrupt every slot identically. Re-inject after ungrouping.

### "Active vinyl group has only N slots, but the JSON has M shapes"

The template you have loaded is smaller than your JSON. Open the FH6 vinyl editor's Create Vinyl Group menu and pick a larger template (FH6 offers 10 / 20 / 50 / 100 / 500 / 1000 / 1500 / 3000-sphere templates). The template needs ≥ your JSON's shape count. Re-inject.

### Where's the inject log?

Persistent per-run log written to `%LOCALAPPDATA%\ForzaAbyssPainter\logs\inject-YYYYMMDD-HHMMSS.log`. The inject dialog also shows the path in its footer and has an **Open log folder** button after the run finishes. Every status line, per-gate `[fast-locate]` / `[readiness]` diagnostic, and progress milestone is captured — useful when reporting an issue.

### Notebook CUDA OOM

Colab's free T4 has 16 GB VRAM; Pro V100/A100/L4 have 16-40 GB. The notebooks default to settings tuned for ~24 GB+. If you OOM:

1. Run the **Cleanup** cell (Section 3 in every notebook) to free CUDA memory.
2. Lower `MAX_RESOLUTION` in the Configure cell (try 720 → 480 → 360).
3. Lower `RANDOM_SAMPLES` (try halving it).
4. Restart the runtime if the cleanup cell can't get allocated memory under ~10 GB.

See [docs/NOTEBOOK_QUICKSTART.md § VRAM tiers](docs/NOTEBOOK_QUICKSTART.md#vram-tiers) for per-Colab-tier recommended settings.

### Inject succeeds but shapes look wrong in-game

- **Sticker mode mismatch:** if you generated the JSON with `STICKER_MODE=True` (transparent background), the shapes are positioned for an alpha-cutout silhouette. Injecting into a non-sticker context will look offset. Regenerate with `STICKER_MODE=False`.
- **Coordinate convention:** FH6's vinyl editor uses an inverted Y axis. The injector handles this automatically — if shapes appear mirrored vertically, you're likely on a JSON exported from a tool that wasn't aware of the convention.
- **Pre-paint residue:** the template's pre-existing low-index shapes (slots 1-10) may show through. In the vinyl editor's layer panel, delete or move them to behind the injected design.

### Splash video doesn't play

The app needs `SplashScreen.mp4` next to the EXE. If you renamed/moved the file or extracted the EXE from an archive that dropped it, the app skips the splash silently and goes straight to the main window. Not fatal — just no intro.

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

GitHub only lets a repo declare one fork parent. We're nominally forked from `tokyubevoxelverse/ForzaDesigner6`, but the FH6 injection layer here is the result of cross-pollination with [`bvzrays/forza-painter-fh6`](https://github.com/bvzrays/forza-painter-fh6) — when our injector is fast and reliable on a current FH6 build, that's their research. Specific things we adopted (with locations in their source):

- **8-byte sentinel + scan-window strategy** — `src/game_profiles.py` (`KNOWN_LIVERY_SIGNATURE`, `COMMON_SCAN_REGIONS`). We use the same sentinel and the same three windows.
- **4-step pointer chain + mirror gate** — `src/main.py:calculate_CLivery`. Same offsets (sig+0xB8 → +0xA58 → +0x8 → +0x20), same `+0x70` mirror check.
- **Two-strategy heap region order** — `src/fh6_probe.py:locate_clivery_groups_by_layout_count`. We adopted both `v1.3` small-address-asc and `v1.4` large-size-desc strategies; without this our scan was 6.6× slower than painter's.
- **Painter-matched validation thresholds** — `src/fh6_probe.py:validate_table_layer_coverage`. The 25%-or-32-min strict-valid threshold and duplicate-pointer skip; before adopting this, our injector rejected templates that painter accepted.
- **Partial-read tolerance** — `src/native.py:read_process_memory`. Accepting `ERROR_PARTIAL_COPY` reads (rather than discarding them) — required for scan windows that extend past the module end.
- **Per-layer write loop + write convention** — position `(X, -Y)`, scale divisors (ellipse `/63`, other `/127`), rotation `360-deg`, color `RGBA` alpha forced to 255, shape-id bytes 101/102.

What's original here on top of painter's foundation:

- **GUI** — Qt-based dark-themed app with image upload, live preview, progress dialog, persistent inject log, dialog footer with log path.
- **Post-locate table validation as a hard gate** — sampled scoring + grouped-template detection via duplicate-pointer count, surfaced with actionable diagnostics in the inject dialog ("template is grouped — Select All → Ungroup").
- **Per-gate `[fast-locate]` / `[readiness]` trace logging** — every chain hop's read address + resolved value persisted to `%LOCALAPPDATA%\ForzaAbyssPainter\logs\inject-*.log` for post-mortem debugging.
- **Pre-inject template-size picker** — constrain heap scan to a user-specified size (skip the common-sizes walk).
- **GPU shape-gen pipeline** — Colab notebooks for 400 / 700 / 1000 / 3000-shape JSONs.

Other credits:

- **[tokyubevoxelverse/ForzaDesigner6](https://github.com/tokyubevoxelverse/ForzaDesigner6)** — upstream fork parent. Multi-game suite scaffolding (FH3-6 + ACC), the original FD6 GUI, the CPU shape generator, theme system, and JSON format.
- **forza-painter (the_adawg)** — original `forza-painter` tooling that inspired upstream FD6.
- **geometrize-lib (Sam Twidale)** + **Primitive (Michael Fogleman)** — the underlying greedy shape-fitting algorithm.

## License

MIT, inherited from upstream. See [LICENSE](LICENSE).
