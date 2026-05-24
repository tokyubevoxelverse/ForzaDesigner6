import sys
from pathlib import Path
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication
from forza_abyss_painter.gui.main_window import MainWindow
from forza_abyss_painter.gui.splash import maybe_show_splash


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Forza Abyss Painter")
    app.setOrganizationName("ForzaAbyssPainter")
    # Phase-1 window icon: ship the new branded logo at app boot so it shows on
    # the taskbar before MainWindow constructs. Theme-badge icon below may
    # override per the user's saved theme choice (legacy upstream behavior).
    _logo = Path(__file__).resolve().parent.parent / "assets" / "forza_abyss_painter_logo.png"
    if _logo.exists():
        app.setWindowIcon(QIcon(str(_logo)))
    # Load bundled TTFs and apply the user's saved font choice (or the
    # "Default" pseudo-family = Segoe UI Variable → Segoe UI → sans fallback).
    from forza_abyss_painter.gui.fonts import load_bundled_fonts, apply_font, saved_font_name
    load_bundled_fonts()
    apply_font(app, saved_font_name())
    # Window icon matches the current theme's badge (Default → Pink)
    from forza_abyss_painter.gui.brand_banner import badge_path
    from forza_abyss_painter.gui.themes import badge_filename_for_theme, saved_theme_name
    bp = badge_path(badge_filename_for_theme(saved_theme_name()))
    if bp:
        app.setWindowIcon(QIcon(str(bp)))

    # Apply persisted theme before constructing MainWindow so styling applies cleanly
    from forza_abyss_painter.gui.themes import apply_theme, saved_theme_name
    apply_theme(app, saved_theme_name())

    win = MainWindow()

    def show_main():
        win.show()
        # Defer music start by one event-loop tick so any splash teardown
        # finishes first (two simultaneous QMediaPlayer streams during splash
        # close was crashing the app).
        from PySide6.QtCore import QTimer
        QTimer.singleShot(150, win.start_music)

    # Show splash if SplashScreen.mp4 is present, then open main window when video ends or user clicks/keypress.
    # If no splash file, show main window immediately.
    splash = maybe_show_splash(show_main)
    if splash is None:
        # Already shown by callback above
        pass

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
