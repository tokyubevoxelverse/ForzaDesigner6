"""Background worker that runs FH6 injection on a QThread and emits progress + colored status signals.

Severity codes for the `status` signal:
  "info"    — neutral (use default text color)
  "success" — green (operation completed OK)
  "warning" — yellow (completed but with caveats)
  "error"   — red (operation failed)
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, Signal


class InjectionWorker(QObject):
    scan_progress = Signal(int, int, int)   # scanned_regions, total_regions, hits_so_far
    write_progress = Signal(int, int)       # written_shapes, total_shapes
    status = Signal(str, str)               # message, severity ("info"|"success"|"warning"|"error")
    done = Signal()

    def __init__(self, json_path: Path, profile_key: str = "fh6") -> None:
        super().__init__()
        self.json_path = Path(json_path)
        self.profile_key = profile_key

    def run(self) -> None:
        from forza_abyss_painter.inject import FH6Injector, patterns_are_populated
        from forza_abyss_painter.inject.game_profiles import get_profile, default_profile
        from forza_abyss_painter.io.exporter import load_json

        if not patterns_are_populated():
            self.status.emit("Patterns file not populated. Re-derive via discovery workflow.", "error")
            self.done.emit()
            return

        try:
            profile = get_profile(self.profile_key)
        except ValueError:
            profile = default_profile()

        try:
            doc = load_json(str(self.json_path))
            shapes = doc.materialize_shapes()
        except Exception as exc:
            self.status.emit(f"Could not load JSON: {type(exc).__name__}: {exc}", "error")
            self.done.emit()
            return

        n_shapes = len(shapes)
        self.status.emit(f"Loaded {n_shapes} shapes from {self.json_path.name}.", "info")

        if profile.beta:
            self.status.emit(
                f"⚠ BETA target: {profile.label}. {profile.beta_note}",
                "warning",
            )

        inj = FH6Injector(profile=profile)
        try:
            # Upfront expectation-setting so users understand both the workflow
            # they should follow AND why a re-injection scan may take longer.
            self.status.emit(
                "Starting injection. For the fastest scan time, load a fresh, "
                "unmodified sphere-template vinyl group of the matching layer count "
                "before injecting. Re-injecting onto an already-painted template "
                "still works but the locator falls back to a slower memory scan "
                "(typically an extra 2–5 minutes on a large game).",
                "info",
            )
            self.status.emit(f"Attaching to {profile.label}...", "info")
            inj.attach()
            if profile.beta:
                self.status.emit(
                    f"Attached. Scanning memory for the {n_shapes}-layer sphere-template "
                    f"LiveryGroup (BETA fallback to RTTI will only run if sphere scan finds nothing)…",
                    "info",
                )
            else:
                self.status.emit(
                    f"Attached. Scanning memory for the {n_shapes}-layer LiveryGroup template…",
                    "info",
                )
            # Kick the dialog out of "Preparing" immediately so user sees activity
            # even before the worker emits real region progress.
            self.scan_progress.emit(0, 1, 0)
            # Callback the injector uses to tell us about phase transitions
            # (e.g., "sphere scan missed, starting RTTI fallback").
            def _on_phase_status(msg: str) -> None:
                self.status.emit(msg, "warning")
            # Pass n_shapes as preferred layer_count so we try the matching template first.
            handle = inj.find_active_vinyl_group(
                progress_cb=self._on_scan_progress,
                layer_count=n_shapes,
                status_cb=_on_phase_status,
            )
            slots = handle.layer_count
            if n_shapes > slots:
                self.status.emit(
                    f"Template has {slots} shape slots but JSON has {n_shapes}. "
                    f"Load a larger template (e.g., {n_shapes}-sphere vinyl group) and re-inject.",
                    "warning",
                )
                self.done.emit()
                return
            self.status.emit(f"Found {slots} shape slots. Writing {n_shapes} shapes...", "info")
            # Pass image_size so the injector can center coords + invert Y
            img_w, img_h = doc.image_size if doc.image_size else (0, 0)
            image_size = (img_w, img_h) if img_w > 0 and img_h > 0 else None
            result = inj.inject(
                shapes, handle, progress_cb=self._on_write_progress,
                image_size=image_size, coord_scale=1.0,
            )
            if result.success:
                self.status.emit(
                    f"Injected {result.shapes_written} shapes successfully. {result.message}",
                    "success",
                )
            else:
                self.status.emit(f"Injection failed: {result.message}", "error")
        except Exception as exc:
            self.status.emit(f"Injection error: {type(exc).__name__}: {exc}", "error")
        finally:
            try:
                inj.detach()
            except Exception:
                pass
            self.done.emit()

    def _on_scan_progress(self, scanned: int, total: int, hits: int) -> None:
        self.scan_progress.emit(scanned, total, hits)

    def _on_write_progress(self, written: int, total: int) -> None:
        self.write_progress.emit(written, total)
