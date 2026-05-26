# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Forza Abyss Painter (Windows EXE build).

Mirrors the per-flag build_exe.bat invocation so either path produces an
equivalent ForzaAbyssPainter.exe. Run with:

    pyinstaller --noconfirm ForzaAbyssPainter.spec

The .bat is still the canonical fast-iteration build script; this .spec is
useful when the build needs declarative overrides (e.g. CI, code signing,
custom version info resources) without bloating the CLI command.
"""

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, collect_dynamic_libs


block_cipher = None


# --- Hidden imports (every fap.gui / fap.inject / fap.ac module the dynamic
# loader can't find on its own — Qt plugin system + per-game-suite lazy loads) ---
hiddenimports = [
    "forza_abyss_painter.gui.music",
    "forza_abyss_painter.gui.particles",
    "forza_abyss_painter.gui.fonts",
    "forza_abyss_painter.gui.image_search",
    "forza_abyss_painter.gui.game_suite_dialog",
    "forza_abyss_painter.gui.ac_settings_panel",
    "forza_abyss_painter.gui.texture_preview_panel",
    "forza_abyss_painter.gui.inject_worker",
    "forza_abyss_painter.gui.inject_dialog",
    "forza_abyss_painter.gui.splash",
    "forza_abyss_painter.gui.brand_banner",
    "forza_abyss_painter.gui.themes",
    "forza_abyss_painter.inject.cli",
    "forza_abyss_painter.inject.discovery",
    "forza_abyss_painter.inject.patterns_io",
    "forza_abyss_painter.inject.win_process",
    "forza_abyss_painter.inject.fh6_injector",
    "forza_abyss_painter.inject.game_profiles",
    "forza_abyss_painter.inject.rtti_locator",
    "forza_abyss_painter.suite",
    "forza_abyss_painter.ac",
    "forza_abyss_painter.ac.profiles",
    "forza_abyss_painter.ac.livery_paths",
    "forza_abyss_painter.ac.car_catalog",
    "forza_abyss_painter.ac.texture_pipeline",
    "forza_abyss_painter.ac.slot_planner",
    "forza_abyss_painter.ac.livery_writer",
    "forza_abyss_painter.shapegen.render",
    "PySide6.QtWebEngineCore",
    "PySide6.QtWebEngineWidgets",
    "PySide6.QtWebChannel",
    "PySide6.QtWebEngineQuick",
    "PySide6.QtPrintSupport",
    "PySide6.QtMultimedia",
    "PySide6.QtMultimediaWidgets",
] + collect_submodules("PySide6.QtWebEngineCore")


# --- Bundled data + binaries (settings profiles, audio, fonts, theme badges,
# the splash mp4, and PySide6's Qt runtime). ---
datas = [
    ("forza_abyss_painter/settings/profiles", "forza_abyss_painter/settings/profiles"),
    ("forza_abyss_painter/inject/patterns", "forza_abyss_painter/inject/patterns"),
    # GPU runtime install copies these subpackages as directory trees into embedded
    # site-packages. PyInstaller onefile otherwise keeps .py only in PYZ (no shapegen/
    # io/runtime folders on disk) and torch_installer copy_package fails.
    ("forza_abyss_painter/shapegen", "forza_abyss_painter/shapegen"),
    ("forza_abyss_painter/io", "forza_abyss_painter/io"),
    ("forza_abyss_painter/runtime", "forza_abyss_painter/runtime"),
    ("forza_abyss_painter/cli", "forza_abyss_painter/cli"),
    ("SplashScreen.mp4", "."),
    ("Song1OpenSource.mp3", "."),
    ("Song2OpenSource.mp3", "."),
    ("Song3OpenSource.mp3", "."),
    ("AppIconTransparent.png", "."),
    ("BlossomParticle.png", "."),
    ("fonts", "fonts"),
    ("Pink.png", "."),
    ("Yellow.png", "."),
    ("Purple.png", "."),
    ("Green.png", "."),
    ("Blue.png", "."),
    ("Orange.png", "."),
    ("assets/forza_abyss_painter_logo.png", "assets"),
    ("assets/forza_abyss_painter.ico", "assets"),
] + collect_data_files("PySide6")

binaries = collect_dynamic_libs("PySide6")


a = Analysis(
    ["forza_abyss_painter/__main__.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)


exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="ForzaAbyssPainter",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # --windowed
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon="assets/forza_abyss_painter.ico",
)
