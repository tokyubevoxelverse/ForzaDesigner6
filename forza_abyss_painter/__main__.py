import multiprocessing

from forza_abyss_painter.app import main

if __name__ == "__main__":
    # Required for ProcessPoolExecutor under PyInstaller --onefile on Windows.
    # Must run before any other multiprocessing usage. Safe no-op outside frozen exe.
    multiprocessing.freeze_support()
    raise SystemExit(main())
