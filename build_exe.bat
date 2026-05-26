@echo off
REM Build a directory-based FD6 package with PyInstaller.
REM Run from the FD6\ directory after `pip install -r requirements.txt`.

setlocal
cd /d "%~dp0"

pyinstaller ^
    --noconfirm ^
    --onedir ^
    --windowed ^
    --name "FD6MultiSupport" ^
    --icon "tools\fd6.ico" ^
    --add-data "fd6\settings\profiles;fd6\settings\profiles" ^
    --add-data "fd6\inject\patterns;fd6\inject\patterns" ^
    --add-data "LICENSE;." ^
    --add-data "NOTICE;." ^
    --add-data "THIRD_PARTY_NOTICES.md;." ^
    --add-data "Song1OpenSource.mp3;." ^
    --add-data "Song2OpenSource.mp3;." ^
    --add-data "Song3OpenSource.mp3;." ^
    --add-data "tools\fd6_128.png;tools" ^
    --add-data "AppIconTransparent.png;." ^
    --add-data "BlossomParticle.png;." ^
    --add-data "fonts;fonts" ^
    --add-data "Pink.png;." ^
    --add-data "Yellow.png;." ^
    --add-data "Purple.png;." ^
    --add-data "Green.png;." ^
    --add-data "Blue.png;." ^
    --add-data "Orange.png;." ^
    --hidden-import fd6.gui.music ^
    --hidden-import fd6.gui.particles ^
    --hidden-import fd6.gui.fonts ^
    --hidden-import fd6.gui.image_search ^
    --hidden-import PySide6.QtWebEngineCore ^
    --hidden-import PySide6.QtWebEngineWidgets ^
    --hidden-import PySide6.QtWebChannel ^
    --hidden-import PySide6.QtWebEngineQuick ^
    --hidden-import PySide6.QtPrintSupport ^
    --collect-submodules PySide6.QtWebEngineCore ^
    --collect-data PySide6 ^
    --collect-binaries PySide6 ^
    --hidden-import fd6.inject.cli ^
    --hidden-import fd6.inject.discovery ^
    --hidden-import fd6.inject.patterns_io ^
    --hidden-import fd6.inject.win_process ^
    --hidden-import fd6.inject.fh6_injector ^
    --hidden-import fd6.inject.game_profiles ^
    --hidden-import fd6.inject.rtti_locator ^
    --hidden-import fd6.suite ^
    --hidden-import fd6.ac ^
    --hidden-import fd6.ac.profiles ^
    --hidden-import fd6.ac.livery_paths ^
    --hidden-import fd6.ac.car_catalog ^
    --hidden-import fd6.ac.texture_pipeline ^
    --hidden-import fd6.ac.slot_planner ^
    --hidden-import fd6.ac.livery_writer ^
    --hidden-import fd6.gui.game_suite_dialog ^
    --hidden-import fd6.gui.ac_settings_panel ^
    --hidden-import fd6.gui.texture_preview_panel ^
    --hidden-import fd6.gui.inject_worker ^
    --hidden-import fd6.gui.inject_dialog ^
    --hidden-import fd6.gui.brand_banner ^
    --hidden-import fd6.gui.themes ^
    --hidden-import fd6.shapegen.render ^
    --hidden-import PySide6.QtMultimedia ^
    --hidden-import PySide6.QtMultimediaWidgets ^
    --hidden-import torch ^
    --hidden-import cupy ^
    --collect-submodules torch ^
    --collect-data torch ^
    --collect-binaries torch ^
    --collect-submodules cupy ^
    --collect-data cupy ^
    --collect-binaries cupy ^
    --collect-submodules cupy_backends ^
    --collect-data cupy_backends ^
    --collect-binaries cupy_backends ^
    --exclude-module onnxruntime ^
    -p . ^
    fd6\__main__.py

copy /Y "LICENSE" "dist\FD6MultiSupport\LICENSE" >nul
copy /Y "NOTICE" "dist\FD6MultiSupport\NOTICE" >nul
copy /Y "THIRD_PARTY_NOTICES.md" "dist\FD6MultiSupport\THIRD_PARTY_NOTICES.md" >nul

echo.
echo Built: dist\FD6MultiSupport\
endlocal
