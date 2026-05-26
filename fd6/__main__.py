import multiprocessing

if __name__ == "__main__":
    # Required for ProcessPoolExecutor under PyInstaller --onefile on Windows.
    # Must run before any other multiprocessing usage. Safe no-op outside frozen exe.
    multiprocessing.freeze_support()
    from fd6.app import main
    raise SystemExit(main())
