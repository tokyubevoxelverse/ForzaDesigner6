import sys

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Forza Designer 6")
    app.setOrganizationName("FD6")
    from PySide6.QtCore import QTimer

    from fd6.gui.fonts import load_bundled_fonts, apply_font, saved_font_name
    load_bundled_fonts()
    apply_font(app, saved_font_name())
    from fd6.gui.brand_banner import badge_path
    from fd6.gui.themes import apply_theme, badge_filename_for_theme, saved_theme_name
    theme_name = saved_theme_name()
    bp = badge_path(badge_filename_for_theme(theme_name))
    if bp:
        app.setWindowIcon(QIcon(str(bp)))
    apply_theme(app, theme_name)
    from fd6.gui.main_window import MainWindow
    win = MainWindow()
    win.show()
    QTimer.singleShot(0, win.start_music)

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
