# Third-Party Notices

This repository contains software, documentation, and bundled media used by
Forza Designer 6. Source code written for this project is distributed under the
MIT License in `LICENSE`.

## Upstream Base

This repository is based on `ForzaDesigner6`:

- URL: https://github.com/tokyubevoxelverse/ForzaDesigner6
- License: MIT
- Copyright (c) 2026 Forza Designer 6 contributors

Keep `LICENSE`, `NOTICE`, and this file in source archives and binary release
archives.

## Code Inspirations

FD6 credits the following projects and research sources in `README.md`:

- `forza-painter` by the_adawg, MIT, https://github.com/forza-painter/forza-painter
- `geometrize-lib` by Sam Twidale, MIT, https://samcodes.co.uk/
- `Primitive` by Michael Fogleman, MIT, https://github.com/fogleman/primitive
- Publicly available Forza vinyl research by bvzrays, MIT,
  https://github.com/bvzrays/forza-painter-fh6

## Runtime Dependencies

Source and binary releases may depend on third-party Python packages and native
runtime libraries. Confirm the installed package versions and include required
license texts when preparing a release archive.

The local packaging environment reviewed on 2026-05-26 reported the following
license metadata. Re-check these values for the exact package versions used in a
release:

- PySide6 and shiboken6: LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only
- NumPy: BSD-style license; binary wheels may include OpenBLAS, LAPACK, and GCC
  runtime components with their own notices
- Pillow: MIT-CMU
- PyTorch, when GPU generation support is distributed: BSD-3-Clause
- CuPy, when CUDA generation support is distributed: MIT
- ONNX Runtime and ONNX Runtime DirectML, when line-guide ONNX support is
  distributed: MIT
- PyInstaller, when distributing bundled executables: GPLv2-or-later with the
  PyInstaller exception

For PySide6 and Qt runtime files, confirm the selected license path and keep the
corresponding license files with the binary release. For dependency wheels that
ship `.dist-info` license directories, keep those directories in the PyInstaller
package.

## Bundled Assets

The repository currently includes image, audio, video, and font assets used by
the desktop application. Before publishing a release, confirm that each bundled
asset may be redistributed from a public repository and from binary releases.

Asset groups to verify:

- `AppIconTransparent.png`
- `BlossomParticle.png`
- `Pink.png`, `Yellow.png`, `Purple.png`, `Green.png`, `Blue.png`, `Orange.png`
- `Song1OpenSource.mp3`, `Song2OpenSource.mp3`, `Song3OpenSource.mp3`
- `SplashScreen.mp4`
- `fonts/*.ttf`
- `tools/SplashScreen.gif`, `tools/fd6.ico`, `tools/fd6_128.png`

Do not assume bundled assets are covered by the project MIT license unless their
license is explicitly confirmed. If an asset has a separate license, keep that
license text or attribution next to this file and mention it in release notes.
If redistribution cannot be confirmed, remove or replace the asset before
publishing the repository or a binary archive.

## Optional Models

`models/*.onnx`, `models/*.ort`, and `models/*.bin` are ignored because model
weights may have separate licenses. The small Sobel reference model can be
regenerated locally with:

```powershell
python tools\create_line_guide_sobel_onnx.py models\line_guide.onnx
```

Do not include external model weights in public releases until their upstream
license, attribution, and redistribution terms are documented here.
