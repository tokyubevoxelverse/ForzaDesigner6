# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

from PyInstaller.utils.hooks import collect_all
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_dynamic_libs
from PyInstaller.utils.hooks import collect_submodules

def _keep_onnx_runtime_binary(entry):
    blocked = {'onnxruntime_providers_cuda.dll', 'onnxruntime_providers_tensorrt.dll'}
    return all(Path(str(value)).name.lower() not in blocked for value in entry[:2])

datas = [('fd6\\settings\\profiles', 'fd6\\settings\\profiles'), ('fd6\\inject\\patterns', 'fd6\\inject\\patterns'), ('LICENSE', '.'), ('NOTICE', '.'), ('THIRD_PARTY_NOTICES.md', '.'), ('SplashScreen.mp4', '.'), ('Song1OpenSource.mp3', '.'), ('Song2OpenSource.mp3', '.'), ('Song3OpenSource.mp3', '.'), ('tools\\fd6_128.png', 'tools'), ('AppIconTransparent.png', '.'), ('BlossomParticle.png', '.'), ('fonts', 'fonts'), ('Pink.png', '.'), ('Yellow.png', '.'), ('Purple.png', '.'), ('Green.png', '.'), ('Blue.png', '.'), ('Orange.png', '.'), ('models', 'models')]
binaries = []
hiddenimports = ['fd6.gui.music', 'fd6.gui.particles', 'fd6.gui.fonts', 'fd6.gui.image_search', 'PySide6.QtWebEngineCore', 'PySide6.QtWebEngineWidgets', 'PySide6.QtWebChannel', 'PySide6.QtWebEngineQuick', 'PySide6.QtPrintSupport', 'fd6.inject.cli', 'fd6.inject.discovery', 'fd6.inject.patterns_io', 'fd6.inject.win_process', 'fd6.inject.fh6_injector', 'fd6.inject.game_profiles', 'fd6.inject.rtti_locator', 'fd6.gui.inject_worker', 'fd6.gui.inject_dialog', 'fd6.gui.splash', 'fd6.gui.brand_banner', 'fd6.gui.themes', 'fd6.shapegen.render', 'PySide6.QtMultimedia', 'PySide6.QtMultimediaWidgets', 'torch', 'cupy']
onnx_datas, onnx_binaries, onnx_hiddenimports = collect_all('onnxruntime')
datas += onnx_datas
binaries += [entry for entry in onnx_binaries if _keep_onnx_runtime_binary(entry)]
hiddenimports += onnx_hiddenimports
datas += collect_data_files('PySide6')
datas += collect_data_files('torch')
datas += collect_data_files('cupy')
datas += collect_data_files('cupy_backends')
binaries += collect_dynamic_libs('PySide6')
binaries += collect_dynamic_libs('torch')
binaries += collect_dynamic_libs('cupy')
binaries += collect_dynamic_libs('cupy_backends')
hiddenimports += collect_submodules('PySide6.QtWebEngineCore')
hiddenimports += collect_submodules('torch')
hiddenimports += collect_submodules('cupy')
hiddenimports += collect_submodules('cupy_backends')


a = Analysis(
    ['fd6\\__main__.py'],
    pathex=['.'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
filtered_binaries = [entry for entry in a.binaries if _keep_onnx_runtime_binary(entry)]
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='FD64FH6354221_onnx',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['tools\\fd6.ico'],
)
coll = COLLECT(
    exe,
    filtered_binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='FD64FH6354221_onnx',
)
