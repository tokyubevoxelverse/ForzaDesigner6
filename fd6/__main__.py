import multiprocessing

if __name__ == "__main__":
    # Required for ProcessPoolExecutor under PyInstaller --onefile on Windows.
    # MUST run before any project imports — otherwise worker subprocesses
    # re-enter this module before the worker dispatch hook has been installed,
    # which can trigger ImportError("attempted relative import with no known
    # parent package") inside PyInstaller's frozen bootstrap. Imports stay
    # inside the guard so the worker subprocesses don't replay them either.
    multiprocessing.freeze_support()
    from fd6.app import main
    raise SystemExit(main())
