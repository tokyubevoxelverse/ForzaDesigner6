# Publishing Checklist

Use this checklist before pushing the repository or preparing a GitHub release.

## Repository

- Run `git status --short` and review every modified or untracked file.
- Keep source files, tests, documentation, build scripts, specs, and model
  examples.
- Do not commit generated folders such as `build*/`, `dist*/`,
  `.pytest_cache/`, or `__pycache__/`.
- Do not commit local model weights: `models/*.onnx`, `models/*.ort`,
  `models/*.bin`.
- Do not commit memory scan outputs, local game dumps, or generated shape JSON
  files.

## Rights Review

- Include `LICENSE`, `NOTICE`, and `THIRD_PARTY_NOTICES.md` in source archives
  and binary release archives.
- Confirm the upstream MIT notice is preserved:
  `Copyright (c) 2026 Forza Designer 6 contributors`.
- Review package metadata and bundled license files for the runtime packages
  included in the release.
- Use this command as the local dependency-license check for the current build
  environment:

```powershell
pip show PySide6 shiboken6 numpy pillow torch cupy-cuda13x onnxruntime onnxruntime-directml pyinstaller
```

- Pay extra attention to PySide6 and Qt runtime files, because their package
  metadata can require LGPL or GPL compliance work depending on the chosen
  license path.
- Confirm every bundled image, audio, video, and font file listed in
  `THIRD_PARTY_NOTICES.md` may be redistributed publicly.
- If a bundled asset or model has no confirmed redistribution permission, remove
  or replace it before publishing.

## Verification

```powershell
pytest -q -p no:capture
git diff --check
```

Run the line-guide benchmark when changing guide scoring or ONNX behavior:

```powershell
python tools\line_guide_benchmark.py --output docs\line_guide_measurements.json
python tools\create_line_guide_sobel_onnx.py models\line_guide.onnx
python tools\line_guide_benchmark.py --model models\line_guide.onnx --output docs\line_guide_measurements_onnx.json
```

## Builds

Normal package:

```powershell
.\build_exe.bat
```

ONNX-enabled package:

```powershell
.\build_exe_onnx.bat
```

Confirm the built package contains:

```powershell
Test-Path dist\FD6MultiSupport\LICENSE
Test-Path dist\FD6MultiSupport\NOTICE
Test-Path dist\FD6MultiSupport\THIRD_PARTY_NOTICES.md
Test-Path dist\FD6MultiSupport_onnx\LICENSE
Test-Path dist\FD6MultiSupport_onnx\NOTICE
Test-Path dist\FD6MultiSupport_onnx\THIRD_PARTY_NOTICES.md
```

If PyInstaller cannot overwrite `dist\FD6MultiSupport_onnx`, close any running
copy of the application that was started from that folder and build again.

## Release Notes

Mention the following when publishing binaries:

- The package is based on the MIT-licensed upstream project listed in `NOTICE`.
- Target platform: Windows 10/11 x64.
- Normal package does not include ONNX Runtime.
- ONNX-enabled package includes DirectML and CPU ONNX providers.
- Optional line-guide model weights are not distributed unless their license is
  confirmed.
- The tool modifies live game memory for vinyl-group injection and may violate
  game or platform terms of use.

## Asset Review

Before the first public release, review `THIRD_PARTY_NOTICES.md` and confirm
that all bundled image, audio, video, and font assets are redistributable.
