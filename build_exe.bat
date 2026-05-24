@echo off
REM Build a single-file ForzaAbyssPainter.exe with PyInstaller.
REM Run from the repo root after `pip install -r requirements.txt`.

setlocal
cd /d "%~dp0"

pyinstaller ^
    --noconfirm ^
    --onefile ^
    --windowed ^
    --name "ForzaAbyssPainter" ^
    --icon "assets\forza_abyss_painter.ico" ^
    --add-data "forza_abyss_painter\settings\profiles;forza_abyss_painter\settings\profiles" ^
    --add-data "forza_abyss_painter\inject\patterns;forza_abyss_painter\inject\patterns" ^
    --add-data "SplashScreen.mp4;." ^
    --add-data "Song1OpenSource.mp3;." ^
    --add-data "Song2OpenSource.mp3;." ^
    --add-data "Song3OpenSource.mp3;." ^
    --add-data "AppIconTransparent.png;." ^
    --add-data "BlossomParticle.png;." ^
    --add-data "fonts;fonts" ^
    --add-data "Pink.png;." ^
    --add-data "Yellow.png;." ^
    --add-data "Purple.png;." ^
    --add-data "Green.png;." ^
    --add-data "Blue.png;." ^
    --add-data "Orange.png;." ^
    --add-data "assets\forza_abyss_painter_logo.png;assets" ^
    --add-data "assets\forza_abyss_painter.ico;assets" ^
    --hidden-import forza_abyss_painter.gui.music ^
    --hidden-import forza_abyss_painter.gui.particles ^
    --hidden-import forza_abyss_painter.gui.fonts ^
    --hidden-import forza_abyss_painter.gui.image_search ^
    --hidden-import PySide6.QtWebEngineCore ^
    --hidden-import PySide6.QtWebEngineWidgets ^
    --hidden-import PySide6.QtWebChannel ^
    --hidden-import PySide6.QtWebEngineQuick ^
    --hidden-import PySide6.QtPrintSupport ^
    --collect-submodules PySide6.QtWebEngineCore ^
    --collect-data PySide6 ^
    --collect-binaries PySide6 ^
    --hidden-import forza_abyss_painter.inject.cli ^
    --hidden-import forza_abyss_painter.inject.discovery ^
    --hidden-import forza_abyss_painter.inject.patterns_io ^
    --hidden-import forza_abyss_painter.inject.win_process ^
    --hidden-import forza_abyss_painter.inject.fh6_injector ^
    --hidden-import forza_abyss_painter.inject.game_profiles ^
    --hidden-import forza_abyss_painter.inject.rtti_locator ^
    --hidden-import forza_abyss_painter.suite ^
    --hidden-import forza_abyss_painter.ac ^
    --hidden-import forza_abyss_painter.ac.profiles ^
    --hidden-import forza_abyss_painter.ac.livery_paths ^
    --hidden-import forza_abyss_painter.ac.car_catalog ^
    --hidden-import forza_abyss_painter.ac.texture_pipeline ^
    --hidden-import forza_abyss_painter.ac.slot_planner ^
    --hidden-import forza_abyss_painter.ac.livery_writer ^
    --hidden-import forza_abyss_painter.gui.game_suite_dialog ^
    --hidden-import forza_abyss_painter.gui.ac_settings_panel ^
    --hidden-import forza_abyss_painter.gui.texture_preview_panel ^
    --hidden-import forza_abyss_painter.gui.inject_worker ^
    --hidden-import forza_abyss_painter.gui.inject_dialog ^
    --hidden-import forza_abyss_painter.gui.splash ^
    --hidden-import forza_abyss_painter.gui.brand_banner ^
    --hidden-import forza_abyss_painter.gui.themes ^
    --hidden-import forza_abyss_painter.shapegen.render ^
    --hidden-import PySide6.QtMultimedia ^
    --hidden-import PySide6.QtMultimediaWidgets ^
    -p . ^
    forza_abyss_painter\__main__.py

echo.
echo Built: dist\ForzaAbyssPainter.exe
endlocal
