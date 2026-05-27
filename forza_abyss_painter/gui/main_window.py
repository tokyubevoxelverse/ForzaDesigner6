from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Qt, QThread
from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QMainWindow, QMessageBox, QSplitter, QStackedWidget, QStatusBar, QVBoxLayout, QWidget
)

from forza_abyss_painter.gui.ac_settings_panel import ACSettingsPanel
from forza_abyss_painter.gui.brand_banner import BrandBanner, badge_path
from forza_abyss_painter.gui.game_suite_dialog import GameSuiteDialog
from forza_abyss_painter.gui.preview_panel import PreviewPanel
from forza_abyss_painter.gui.texture_preview_panel import TexturePreviewPanel
from forza_abyss_painter.gui.themes import THEMES, apply_theme, saved_theme_name, badge_filename_for_theme
from forza_abyss_painter.gui.queue_panel import QueuePanel
from forza_abyss_painter.gui.settings_panel import SettingsPanel
from forza_abyss_painter.gui.upload_panel import UploadPanel
from forza_abyss_painter.shapegen.profile import Profile
from forza_abyss_painter.shapegen.worker import GenerationWorker
from forza_abyss_painter.inject.fh6_injector import patterns_are_populated, FH6_TARGET_BUILD
from forza_abyss_painter.suite import SuiteMode, SUITE_DISPLAY, saved_suite_mode, save_suite_mode
from forza_abyss_painter.runtime.nvidia_smi import probe_free_vram, ProbeResult


def _vram_preflight_verdict(
    peak_gib: float, free_gib: float | None, budget_gib: float,
) -> tuple[str, str]:
    """Return (severity, recommendation) for a GPU pre-Start check.

    severity:
      - "ok"        — peak fits comfortably in free; no dialog needed.
      - "warn"      — peak > 0.85 * free OR free < 0.9 * budget; surface
                      a "proceed anyway?" dialog with chunked-K guidance.
      - "block"     — peak > free; the run will OOM regardless of chunking
                      (canvas/refill/joint_polish overhead is in the
                      estimate but not reduced by K-chunking).

    `free_gib=None` (probe unavailable, non-NVIDIA box) returns "ok" —
    we can't make a free-vs-peak comparison so we let the existing
    chunked-K fallback handle whatever happens at runtime.

    recommendation is a one-line user-facing summary the caller can
    paste into a dialog body.
    """
    if free_gib is None or free_gib <= 0:
        return "ok", ""
    if peak_gib > free_gib:
        return "block", (
            f"Estimated peak ~{peak_gib:.1f} GiB exceeds {free_gib:.1f} "
            f"GiB free. Close other GPU apps or lower the preset."
        )
    if peak_gib > free_gib * 0.85 or free_gib < budget_gib * 0.9:
        return "warn", (
            f"Estimated peak ~{peak_gib:.1f} GiB vs {free_gib:.1f} GiB "
            f"free (budget {budget_gib:.0f}). Chunked-K may help; may "
            f"still OOM if free shrinks."
        )
    return "ok", ""


def _resolve_source_image_path(json_path: Path, source_image_name: str) -> Path | None:
    """Resolve the source image for a loaded JSON via the same-folder
    heuristic. `source_image_name` is the JSON's `source_image` field
    (canonically a bare filename); if it accidentally contains path
    separators we use only the basename for safety. Returns the path
    if the sibling exists, otherwise None — caller falls back to a
    file picker.
    """
    if not source_image_name:
        return None
    bare = Path(source_image_name).name   # strip any embedded path
    candidate = json_path.parent / bare
    return candidate if candidate.is_file() else None


def _apply_autotune_to_preset(
    preset: dict,
    probe: "ProbeResult | None" = None,
) -> tuple[dict, str]:
    """Apply the #131 back-prop recommendation to a GPU preset dict.

    Returns a NEW dict (input is not mutated) with `max_resolution`
    bumped to the back-prop value when bigger than the baked preset
    value (never lower — the preset author's choice is the floor for
    that preset). The second element is a human-readable status line
    the caller can log to the status bar.

    `probe` lets the caller share an already-fetched ProbeResult to
    avoid spawning nvidia-smi twice (#131 review GAP 4). When None,
    the helper fetches one itself with `force=True`.
    """
    bumped = dict(preset)   # shallow copy; preset's leaf values are scalars
    baked_max_res = int(bumped["max_resolution"])
    K = int(bumped.get("random_samples", 0) or 0)
    if probe is None:
        probe = probe_free_vram(force=True)
    if probe.available and probe.free_gib is not None and K > 0:
        from forza_abyss_painter.shapegen.gpu.vram_planner import (
            recommend_max_resolution,
        )
        recommended = recommend_max_resolution(
            free_gib=probe.free_gib, K=K, bbox_local=True,
        )
        effective = max(baked_max_res, recommended)
        bumped["max_resolution"] = effective
        if effective > baked_max_res:
            message = (
                f"Auto-tuned max_res to {effective} px "
                f"({probe.name or 'GPU'}, {probe.free_gib:.1f} GiB free)"
            )
        else:
            message = (
                f"Using baked max_res {baked_max_res} px "
                f"({probe.free_gib:.1f} GiB free; back-prop says "
                f"{recommended} px — using floor)"
            )
    else:
        bumped["max_resolution"] = baked_max_res
        message = (
            f"Using baked max_res {baked_max_res} px "
            f"(VRAM probe unavailable; using safety floor)"
        )
    return bumped, message


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Forza Abyss Painter — Inject custom decals and vinyls into various racing titles")
        self.resize(1280, 760)
        self.setStatusBar(QStatusBar(self))
        # Permanent GPU status indicator on the right side of the status
        # bar — always visible, always reflects current install state.
        # Refreshed after the install dialog closes + on construction.
        # See _refresh_gpu_status_indicator for the state machine.
        self._gpu_status_label = QLabel("")
        self._gpu_status_label.setStyleSheet(
            "padding: 0 8px; color: #888; font-size: 11px;"
        )
        self.statusBar().addPermanentWidget(self._gpu_status_label)
        self._apply_dark_palette()

        # Suite mode — read persisted choice; default Forza on first launch.
        # The actual popup (if any) fires after window is shown, see show() override.
        self._suite_mode: SuiteMode = saved_suite_mode() or SuiteMode.FORZA
        self._suite_first_launch: bool = saved_suite_mode() is None

        # Panels — Forza panels existed in v0.3.0; AC panels are new for v0.3.5.
        self.upload = UploadPanel(self)
        self.preview = PreviewPanel(self)              # Forza preview (live shape gen)
        self.ac_preview = TexturePreviewPanel(self)    # AC preview (cycling slots)
        self.queue = QueuePanel(self)
        self.settings_panel = SettingsPanel(self)      # Forza settings (geometrize knobs)
        self.ac_settings = ACSettingsPanel(self)       # AC settings (car/slot/resolution)

        # Stacked widgets so suite switch is a one-call swap.
        self.preview_stack = QStackedWidget(self)
        self.preview_stack.addWidget(self.preview)     # index 0 — Forza
        self.preview_stack.addWidget(self.ac_preview)  # index 1 — AC
        self.settings_stack = QStackedWidget(self)
        self.settings_stack.addWidget(self.settings_panel)   # index 0 — Forza
        self.settings_stack.addWidget(self.ac_settings)      # index 1 — AC

        # Layout: [upload | center (preview-stack over queue) | settings-stack]
        center = QWidget(self)
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        vsplit = QSplitter(Qt.Vertical, center)
        vsplit.addWidget(self.preview_stack)
        vsplit.addWidget(self.queue)
        vsplit.setSizes([520, 220])
        center_layout.addWidget(vsplit)

        hsplit = QSplitter(Qt.Horizontal, self)
        hsplit.addWidget(self.upload)
        hsplit.addWidget(center)
        hsplit.addWidget(self.settings_stack)
        hsplit.setSizes([240, 760, 280])
        self.setCentralWidget(hsplit)

        # Wire signals — Forza paths (unchanged)
        self.upload.files_selected.connect(self._on_files_selected)
        self.upload.json_loaded.connect(self._on_json_loaded_for_preview)
        self.upload.reshape_requested.connect(self._on_reshape_requested)
        self.upload.polish_requested.connect(self._on_polish_requested)
        self.upload.resume_requested.connect(self._on_resume_requested)
        self.upload.download_json_requested.connect(self._on_download_json)
        self.settings_panel.start_clicked.connect(self._start_next)
        self.settings_panel.pause_clicked.connect(self._toggle_pause)
        self.settings_panel.stop_clicked.connect(self._stop_current)
        self.settings_panel.inject_clicked.connect(self._on_inject_clicked)
        # Backend selector: if user picks GPU but it's not installed,
        # auto-open the install dialog. After it closes (success or
        # cancel), refresh the dropdown label so the user sees the new
        # state immediately.
        self.settings_panel.gpu_install_requested.connect(
            self._on_gpu_install_requested
        )
        # AC path
        self.ac_settings.export_clicked.connect(self._on_ac_export_clicked)

        # Worker state
        self._worker: GenerationWorker | None = None
        self._thread: QThread | None = None
        self._current_path: Path | None = None
        self._current_profile: Profile | None = None
        self._last_finished_json: Path | None = None  # tracks most recent completed run for Download button
        self._loaded_json_path: Path | None = None    # JSON loaded via Upload JSON (ready to inject)
        # Single-slot render throttle for snapshot previews dispatched via
        # _on_gpu_snapshot. If a _RenderSnapshotJob is already in flight,
        # remember the latest snapshot path here; QTimer drain picks it up.
        self._snapshot_render_in_flight: bool = False
        self._snapshot_pending_path: str | None = None
        # Cached validation issues from the last auto-validate on Upload
        # JSON. Re-shown by Tools → Validate current JSON without
        # re-reading the file. None until the first successful load.
        self._loaded_json_issues: list | None = None
        self._inject_worker = None  # InjectionWorker (set when injecting)
        self._inject_thread: QThread | None = None

        # Menus / shortcuts
        self._build_menus()

        # Wire FH6 inject gating
        self._refresh_inject_button()

        # Floating brand banner in bottom-left (toggleable)
        self.brand_banner = BrandBanner(self)
        self.brand_banner.show()
        # Make sure the brand banner + window icon match the persisted theme
        # right at construction time (was previously only synced on theme change,
        # so launches would briefly show the Default Pink badge for non-Default
        # themes).
        from PySide6.QtGui import QIcon
        _saved_theme = saved_theme_name()
        _bp = badge_path(badge_filename_for_theme(_saved_theme))
        if _bp:
            self.setWindowIcon(QIcon(str(_bp)))
            self.brand_banner.set_badge(_bp)

        # Decorative particle overlay (theme-colored, transparent, click-through).
        # Constructed AFTER brand_banner so we can raise it above everything.
        from forza_abyss_painter.gui.particles import ParticleOverlay
        self.particles = ParticleOverlay(self)
        _pal = THEMES.get(_saved_theme, THEMES["Default"])
        self.particles.set_theme_colors(
            _pal["particle_1"], _pal["particle_2"], _pal["particle_3"],
        )
        self.particles.reposition()
        # Register a live exclude-rect provider so the rect stays correct even
        # when the user drags the splitter (which doesn't trigger MainWindow's
        # resizeEvent). Provider is called every particle paintEvent.
        self.particles.set_exclude_provider(self._compute_particle_exclude_rect)
        # Still call the push path once so the cached fallback is populated.
        from PySide6.QtCore import QTimer
        QTimer.singleShot(0, self._sync_particle_exclude_rect)
        # Initial state of menu actions (built in _build_menus before this point)
        if hasattr(self, "_particles_enabled_act"):
            self._particles_enabled_act.setChecked(self.particles.enabled())
            self._sync_particle_count_check(self.particles.count())

        # Apply the persisted suite mode now (after menus + panels exist).
        # The first-launch popup is deferred until showEvent fires — otherwise
        # it appears OVER the splash screen and blocks the user from skipping it.
        self._apply_suite_mode(self._suite_mode)
        self._suite_popup_shown_this_session = False

        # Background music: 3 looping OpenSource tracks. Construct now (cheap),
        # but DEFER starting until start_music() is called from app.py after the
        # splash finishes. Two simultaneous QMediaPlayer instances racing during
        # splash teardown was causing GUI crashes on skip / video end.
        from forza_abyss_painter.gui.music import MusicPlayer
        self.music = MusicPlayer(self)
        self.music.state_changed.connect(self._on_music_state)
        self.music.muted_changed.connect(self._on_music_muted)
        self.music.volume_changed.connect(self._on_music_volume)
        self.music.track_changed.connect(
            lambda name: self.statusBar().showMessage(f"♪ {name}", 4000)
        )
        if not self.music.has_tracks():
            for act in (self._music_play_act, self._music_mute_act):
                act.setEnabled(False)
            for act in self._music_vol_group.actions():
                act.setEnabled(False)

        # Initial population of the GPU status indicator. Reads
        # is_runtime_installed() + the install marker if present;
        # post-install refreshes from _on_install_gpu_runtime.
        self._refresh_gpu_status_indicator()

    def _refresh_gpu_status_indicator(self) -> None:
        """Update the permanent status-bar GPU label based on current
        install state. Called at construction time + every time the
        install dialog closes so the user sees state transitions
        without needing to restart the EXE.

        State machine:
          - GPU_PHASE_3_AVAILABLE=False → label hidden (no GPU UX yet)
          - Marker missing                → "GPU: not installed"
          - Marker present, cuda False    → "GPU: install incomplete"
          - Marker present, cuda True     → "GPU: ready ({device})"
        """
        from forza_abyss_painter.gui.feature_flags import GPU_PHASE_3_AVAILABLE
        if not GPU_PHASE_3_AVAILABLE:
            self._gpu_status_label.setVisible(False)
            return
        self._gpu_status_label.setVisible(True)
        from forza_abyss_painter.runtime.torch_installer import (
            installed_runtime_info, is_runtime_installed,
        )
        info = installed_runtime_info()
        if info is None:
            self._gpu_status_label.setText("GPU: not installed")
            self._gpu_status_label.setStyleSheet(
                "padding: 0 8px; color: #888; font-size: 11px;"
            )
        elif not info.cuda_available:
            self._gpu_status_label.setText("GPU: install incomplete (CPU-only torch)")
            self._gpu_status_label.setStyleSheet(
                "padding: 0 8px; color: #c97a4f; font-size: 11px;"
            )
        else:
            device = info.cuda_device_name or "CUDA device"
            # Append live free-VRAM if nvidia-smi works on this box.
            # Cheap call — cached for 5s so the status bar can refresh
            # frequently without spawning subprocesses. Falls back to
            # the plain "ready" string when nvidia-smi is unavailable.
            probe = probe_free_vram()
            if probe.available and probe.free_gib is not None and probe.total_gib is not None:
                vram_suffix = (
                    f"  •  {probe.free_gib:.1f} / {probe.total_gib:.1f} GiB free"
                )
            else:
                vram_suffix = ""
            self._gpu_status_label.setText(f"GPU: ready — {device}{vram_suffix}")
            self._gpu_status_label.setStyleSheet(
                "padding: 0 8px; color: #6fbf73; font-size: 11px;"
            )

    def start_music(self) -> None:
        """Begin background music. Call once, after the splash has finished, to
        avoid two QMediaPlayer audio streams colliding during splash teardown."""
        if not getattr(self, "music", None) or not self.music.has_tracks():
            return
        if getattr(self, "_music_started", False):
            return
        self._music_started = True
        self.music.start()
        self._music_play_act.setChecked(self.music.is_playing())
        self._music_mute_act.setChecked(self.music.muted())
        self._sync_volume_check(self.music.volume())

    def _apply_dark_palette(self) -> None:
        # Theme styling now lives in fd6/gui/themes.py and is applied at QApplication level.
        # We just trigger the saved theme here in case MainWindow is constructed before app-level apply.
        from PySide6.QtWidgets import QApplication
        apply_theme(QApplication.instance(), saved_theme_name())

    def _build_menus(self) -> None:
        mbar = self.menuBar()
        file_menu = mbar.addMenu("&File")

        open_act = QAction("&Upload Image…", self)
        open_act.setShortcut(QKeySequence("Ctrl+O"))
        open_act.triggered.connect(self.upload._on_upload_clicked)
        file_menu.addAction(open_act)

        quit_act = QAction("&Quit", self)
        quit_act.setShortcut(QKeySequence("Ctrl+Q"))
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)

        # ---- Tools menu ----
        #
        # GPU entries are gated behind GPU_PHASE_3_AVAILABLE. The flag
        # is True once the install + generate plumbing is real (tasks
        # #93-#96 + #102-#104 logging/diagnostics). See feature_flags.py.
        from forza_abyss_painter.gui.feature_flags import GPU_PHASE_3_AVAILABLE
        tools_menu = mbar.addMenu("&Tools")
        if GPU_PHASE_3_AVAILABLE:
            # Install runtime — direct entry point. Users can pre-install
            # without going through Generate, and re-install if their
            # marker shows partial (cuda_available=False) state.
            install_act = QAction("&Install GPU runtime…", self)
            install_act.setStatusTip(
                "Download and install the local GPU shape-gen runtime "
                "(~4 GiB; one-time)"
            )
            install_act.triggered.connect(self._on_install_gpu_runtime)
            tools_menu.addAction(install_act)
            # Generate — the workflow entry point. If runtime isn't yet
            # installed, this also prompts for install before showing
            # the Generate dialog.
            generate_act = QAction("&Generate shapes locally (GPU)…", self)
            generate_act.setStatusTip(
                "Run the GPU shape-generator on your local CUDA card "
                "(requires one-time ~4 GiB runtime download)"
            )
            generate_act.triggered.connect(self._on_generate_locally)
            tools_menu.addAction(generate_act)
            tools_menu.addSeparator()

        # One-click fap-clean: load a JSON, strip padding-whites + dead
        # weight, save the cleaned file. Same library function the CLI
        # uses; this is purely a UX surface for users who don't want to
        # drop to a terminal.
        clean_act = QAction("&Clean current JSON…", self)
        clean_act.setStatusTip(
            "Strip padding-white + fully-occluded shapes from a JSON "
            "(same cleanup as the fap-clean CLI)"
        )
        clean_act.triggered.connect(self._on_clean_json)
        tools_menu.addAction(clean_act)

        # Validate currently-loaded JSON against fd6.shapes v1 spec.
        # Surfaces issues that would silently break the inject — bad
        # extents, non-injector-safe shape types, invisible (alpha=0)
        # shapes without is_mask=true. Re-runs validation even if the
        # JSON was already auto-validated at load time, so users can
        # check after a hand-edit. Disabled until a JSON is loaded.
        self._validate_act = QAction("&Validate current JSON…", self)
        self._validate_act.setStatusTip(
            "Run fd6.shapes v1 schema validation on the currently-loaded "
            "JSON and show every finding (errors, warnings, info)"
        )
        self._validate_act.triggered.connect(self._on_validate_current_json)
        self._validate_act.setEnabled(False)   # nothing loaded yet
        tools_menu.addAction(self._validate_act)

        # Save diagnostics bundle: zip the GPU logs + runtime marker +
        # system info so testers can email a single attachment when
        # something goes wrong. Lives under Tools regardless of the
        # GPU flag because diagnostics are useful for non-GPU issues
        # too (the bundle includes inject logs from the existing
        # injector flow via the same logs directory).
        diag_act = QAction("Save &diagnostics zip…", self)
        diag_act.setStatusTip(
            "Bundle recent logs + runtime state + system info into a "
            "zip you can email for support"
        )
        diag_act.triggered.connect(self._on_save_diagnostics)
        tools_menu.addAction(diag_act)

        view_menu = mbar.addMenu("&View")
        theme_menu = view_menu.addMenu("&Theme")
        from PySide6.QtGui import QActionGroup
        self._theme_group = QActionGroup(self)
        self._theme_group.setExclusive(True)
        current_theme = saved_theme_name()
        for theme_name in THEMES.keys():
            act = QAction(theme_name, self, checkable=True)
            act.setChecked(theme_name == current_theme)
            act.triggered.connect(lambda _checked, n=theme_name: self._set_theme(n))
            self._theme_group.addAction(act)
            theme_menu.addAction(act)

        # --- Music submenu (3 looping OpenSource tracks, 0.3 vol default) ----
        view_menu.addSeparator()
        music_menu = view_menu.addMenu("&Music")
        self._music_play_act = QAction("&Play / Pause", self, checkable=True)
        self._music_play_act.setShortcut("Ctrl+M")
        self._music_play_act.triggered.connect(self._music_toggle_play)
        music_menu.addAction(self._music_play_act)
        self._music_mute_act = QAction("M&ute", self, checkable=True)
        self._music_mute_act.triggered.connect(self._music_toggle_mute)
        music_menu.addAction(self._music_mute_act)
        next_act = QAction("&Next track", self)
        next_act.setShortcut("Ctrl+Shift+M")
        next_act.triggered.connect(self._music_next)
        music_menu.addAction(next_act)
        music_menu.addSeparator()
        # Volume submenu (10 % steps)
        vol_menu = music_menu.addMenu("&Volume")
        self._music_vol_group = QActionGroup(self)
        self._music_vol_group.setExclusive(True)
        for pct in (0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100):
            a = QAction(f"{pct}%", self, checkable=True)
            a.triggered.connect(lambda _c, p=pct: self._music_set_volume(p / 100.0))
            self._music_vol_group.addAction(a)
            vol_menu.addAction(a)

        # --- Fonts submenu (Default + every bundled TTF/OTF) ------------------
        view_menu.addSeparator()
        fonts_menu = view_menu.addMenu("F&onts")
        from forza_abyss_painter.gui.fonts import (
            available_family_names as _font_names, saved_font_name as _saved_font,
        )
        self._font_group = QActionGroup(self)
        self._font_group.setExclusive(True)
        current_font = _saved_font()
        for fname in _font_names():
            a = QAction(fname, self, checkable=True)
            a.setChecked(fname == current_font)
            a.triggered.connect(lambda _c, name=fname: self._on_font_pick(name))
            self._font_group.addAction(a)
            fonts_menu.addAction(a)

        # --- Customizations submenu (panel-swap toggles, persisted) ----------
        view_menu.addSeparator()
        custom_menu = view_menu.addMenu("&Customizations")

        # Game-suite submenu — radio-group of FORZA / AC / NFS-coming / CREW-coming.
        # Switches the active suite without restarting; selection persists.
        suite_menu = custom_menu.addMenu("Change Game &Suite")
        self._suite_action_group = QActionGroup(self)
        self._suite_action_group.setExclusive(True)
        self._suite_actions: dict[SuiteMode, QAction] = {}
        for mode in (SuiteMode.FORZA, SuiteMode.AC, SuiteMode.NFS, SuiteMode.CREW):
            meta = SUITE_DISPLAY[mode]
            label = meta["label"] + ("" if meta["enabled"] else " (Coming Soon)")
            act = QAction(label, self, checkable=True)
            act.setEnabled(bool(meta["enabled"]))
            act.setChecked(mode == self._suite_mode)
            act.triggered.connect(lambda checked=False, m=mode: self._on_suite_menu_selected(m))
            self._suite_action_group.addAction(act)
            suite_menu.addAction(act)
            self._suite_actions[mode] = act
        custom_menu.addSeparator()

        self._swap_recents_act = QAction("&Swap recents with image searcher", self, checkable=True)
        self._swap_recents_act.setStatusTip(
            "Replace the Recent files list with a Google-style image search panel "
            "that downloads PNGs straight into the generation queue."
        )
        from PySide6.QtCore import QSettings as _QS
        _cs = _QS("ForzaAbyssPainter", "Forza Abyss Painter")
        _cs.beginGroup("customizations")
        _init_swap = _cs.value("swap_recents_with_image_searcher", False, type=bool)
        _cs.endGroup()
        self._swap_recents_act.setChecked(_init_swap)
        self._swap_recents_act.triggered.connect(self._on_swap_recents_toggled)
        custom_menu.addAction(self._swap_recents_act)
        # Apply the persisted state to the upload panel right away
        if hasattr(self, "upload"):
            self.upload.set_use_image_searcher(_init_swap)

        # --- Particles submenu (theme-colored animated overlay) --------------
        view_menu.addSeparator()
        particles_menu = view_menu.addMenu("&Particles")
        self._particles_enabled_act = QAction("&Show particles", self, checkable=True)
        self._particles_enabled_act.triggered.connect(self._on_particles_toggle)
        particles_menu.addAction(self._particles_enabled_act)
        particles_menu.addSeparator()
        density_menu = particles_menu.addMenu("&Density")
        self._particle_count_group = QActionGroup(self)
        self._particle_count_group.setExclusive(True)
        from forza_abyss_painter.gui.particles import COUNT_OPTIONS
        for n in COUNT_OPTIONS:
            label = "Off (0)" if n == 0 else f"{n} particles"
            a = QAction(label, self, checkable=True)
            a.triggered.connect(lambda _c, count=n: self._on_particles_count(count))
            self._particle_count_group.addAction(a)
            density_menu.addAction(a)

        fh6_menu = mbar.addMenu("F&H6")
        status_act = QAction("FH6 &Status…", self)
        status_act.triggered.connect(self._show_fh6_status)
        fh6_menu.addAction(status_act)
        discovery_act = QAction("&Discovery Workflow…", self)
        discovery_act.triggered.connect(self._show_discovery_help)
        fh6_menu.addAction(discovery_act)
        fh6_menu.addSeparator()
        reload_act = QAction("&Reload Patterns", self)
        reload_act.triggered.connect(self._refresh_inject_button)
        fh6_menu.addAction(reload_act)

        help_menu = mbar.addMenu("&Help")
        about_act = QAction("&About Forza Abyss Painter…", self)
        about_act.triggered.connect(self._show_about)
        help_menu.addAction(about_act)

    def _set_theme(self, theme_name: str) -> None:
        from PySide6.QtWidgets import QApplication
        from PySide6.QtGui import QIcon
        apply_theme(QApplication.instance(), theme_name)
        # Swap badge + window icon to match theme
        bp = badge_path(badge_filename_for_theme(theme_name))
        if bp:
            QApplication.instance().setWindowIcon(QIcon(str(bp)))
            self.setWindowIcon(QIcon(str(bp)))
            if hasattr(self, "brand_banner"):
                self.brand_banner.set_badge(bp)
        # Recolor particle overlay with the theme's particle palette
        if hasattr(self, "particles"):
            pal = THEMES.get(theme_name, THEMES["Default"])
            self.particles.set_theme_colors(
                pal["particle_1"], pal["particle_2"], pal["particle_3"],
            )
        self.statusBar().showMessage(f"Theme: {theme_name}", 3000)

    # -------- customizations --------
    def _on_swap_recents_toggled(self, checked: bool) -> None:
        from PySide6.QtCore import QSettings
        if hasattr(self, "upload"):
            self.upload.set_use_image_searcher(checked)
        s = QSettings("ForzaAbyssPainter", "Forza Abyss Painter")
        s.beginGroup("customizations")
        s.setValue("swap_recents_with_image_searcher", checked)
        s.endGroup()

    # -------- suite-mode dispatch --------
    def _apply_suite_mode(self, mode: SuiteMode) -> None:
        """Swap the visible settings/preview panels + tweak the upload panel for the new suite."""
        self._suite_mode = mode
        is_ac = (mode == SuiteMode.AC)
        # Stack swap
        self.preview_stack.setCurrentIndex(1 if is_ac else 0)
        self.settings_stack.setCurrentIndex(1 if is_ac else 0)
        # Hide Forza-only JSON buttons in AC mode (AC has no JSON pipeline).
        if hasattr(self.upload, "upload_json_btn"):
            self.upload.upload_json_btn.setVisible(not is_ac)
        if hasattr(self.upload, "download_json_btn"):
            self.upload.download_json_btn.setVisible(not is_ac)
        # Sync the radio group in the menu (in case suite changed via popup, not menu)
        if hasattr(self, "_suite_actions"):
            for m, act in self._suite_actions.items():
                act.setChecked(m == mode)
        # Status bar feedback
        meta = SUITE_DISPLAY[mode]
        self.statusBar().showMessage(f"Game suite: {meta['label']}", 4000)

    def _on_suite_menu_selected(self, mode: SuiteMode) -> None:
        """Customizations → Change Game Suite → <mode>."""
        meta = SUITE_DISPLAY[mode]
        if not meta["enabled"]:
            return
        if mode == self._suite_mode:
            return
        self._apply_suite_mode(mode)
        save_suite_mode(mode)

    def _prompt_gpu_install_on_first_launch(self) -> None:
        """First-launch CUDA detection + install offer (#99).

        Self-gates via `should_prompt()` — does nothing if the runtime
        is already installed, the user previously opted out, the
        GPU_PHASE_3_AVAILABLE flag is off, or no NVIDIA GPU is
        present. On 'Install now', chains into the existing
        RuntimeInstallDialog so users land in the same flow they'd
        get from Tools → Install GPU runtime.
        """
        from forza_abyss_painter.gui.gpu_first_launch import (
            GpuPromptDecision, maybe_prompt,
        )
        result = maybe_prompt(self)
        if result.decision is GpuPromptDecision.INSTALL_NOW:
            # Reuse the existing install dialog — single source of
            # truth for the install flow (download progress, error
            # handling, marker write-back).
            from forza_abyss_painter.gui.runtime_install_dialog import (
                RuntimeInstallDialog,
            )
            dlg = RuntimeInstallDialog(self)
            dlg.exec()
            # After the install finishes we refresh the status bar so
            # the GPU label flips from "not installed" to "ready" on
            # the spot, without the user needing to re-open menus.
            try:
                self._refresh_gpu_status_indicator()
            except Exception:
                pass   # status bar is decorative — don't break flow

    def _prompt_suite_on_first_launch(self) -> None:
        """Show the 4-tile suite picker if the user has never picked one.

        Called shortly after the window is shown so the splash teardown
        completes first. If a saved mode exists we skip the popup entirely.
        """
        if not self._suite_first_launch:
            return
        dlg = GameSuiteDialog(self, current=None)
        result = dlg.exec()
        if result and dlg.selected is not None:
            self._apply_suite_mode(dlg.selected)
            save_suite_mode(dlg.selected)
        else:
            # User dismissed without picking — default to Forza and save so we
            # don't keep popping the dialog.
            self._apply_suite_mode(SuiteMode.FORZA)
            save_suite_mode(SuiteMode.FORZA)
        self._suite_first_launch = False

    # -------- AC export handler --------
    def _on_ac_export_clicked(self, cfg: dict) -> None:
        """User clicked Export to ACC. cfg comes from ACSettingsPanel._gather_export_config."""
        from forza_abyss_painter.ac.livery_writer import write_acc_livery
        from forza_abyss_painter.ac.slot_planner import plan_slots
        from forza_abyss_painter.ac.texture_pipeline import build_decal_texture

        # We need a source image — use whatever file was last uploaded.
        if not getattr(self, "_current_path", None):
            QMessageBox.information(
                self, "No image",
                "Upload an image first via 'Upload Image…' before exporting an ACC livery.",
            )
            return
        if not cfg.get("car_model"):
            QMessageBox.information(
                self, "Pick a car",
                "Select an ACC car model from the dropdown before exporting.",
            )
            return
        try:
            rgba, applied_aspect = build_decal_texture(
                self._current_path,
                target_long_edge=int(cfg["resolution"]),
                aspect_choice=str(cfg["aspect"]),
            )
        except Exception as exc:
            QMessageBox.critical(self, "Texture build failed", f"{type(exc).__name__}: {exc}")
            return

        slot_filenames = plan_slots(
            auto=bool(cfg["auto_slot"]),
            manual_main=cfg.get("manual_main_slots"),
            manual_sponsors=cfg.get("manual_sponsor_slots"),
        )

        # Refresh the cycling preview so the user can verify before exporting again.
        self.ac_preview.set_slots([(s, rgba) for s in slot_filenames])

        # Write to disk
        result = write_acc_livery(
            profile=cfg["profile"],
            car_model=cfg["car_model"],
            team_name=cfg["team_name"] or f"FAP_{Path(self._current_path).stem}",
            rgba=rgba,
            slot_filenames=slot_filenames,
            display_name=cfg["display_name"],
            race_number=int(cfg["race_number"]),
            paint=cfg.get("paint"),
        )
        if result.success:
            # Progress bar to 100% so the bottom strip stops sitting at 0% after
            # a finished export — the export IS complete, the bar should reflect that.
            self.ac_preview.progress.setValue(100)
            self.ac_preview.status_label.setText(result.message)
            self.statusBar().showMessage(result.message, 8000)
            QMessageBox.information(
                self, "Livery exported",
                f"{result.message}\n\nFolder:\n{result.team_folder}",
            )
        else:
            self.ac_preview.progress.setValue(0)
            QMessageBox.critical(self, "Export failed", result.message)

    # -------- font handler --------
    def _on_font_pick(self, display_name: str) -> None:
        from PySide6.QtWidgets import QApplication
        from forza_abyss_painter.gui.fonts import apply_font
        apply_font(QApplication.instance(), display_name)
        self.statusBar().showMessage(f"Font: {display_name}", 3000)

    # -------- particle handlers --------
    def _on_particles_toggle(self, checked: bool) -> None:
        if hasattr(self, "particles"):
            self.particles.set_enabled(checked)

    def _on_particles_count(self, count: int) -> None:
        if hasattr(self, "particles"):
            self.particles.set_count(count)
            # Disabling via count==0 should also untick the Show particles action
            self._particles_enabled_act.setChecked(self.particles.enabled() and count > 0)

    def _sync_particle_count_check(self, n: int) -> None:
        for act in self._particle_count_group.actions():
            # Action labels are "Off (0)" or "N particles"
            t = act.text()
            digits = "".join(c for c in t.split()[0] if c.isdigit())
            if digits.isdigit() and int(digits) == n:
                act.setChecked(True)
                return

    # -------- music handlers --------
    def _music_toggle_play(self) -> None:
        playing = self.music.toggle_play()
        self._music_play_act.setChecked(playing)

    def _music_toggle_mute(self) -> None:
        muted = self.music.toggle_mute()
        self._music_mute_act.setChecked(muted)

    def _music_next(self) -> None:
        self.music.next_track()

    def _music_set_volume(self, vol: float) -> None:
        self.music.set_volume(vol)

    def _on_music_state(self, playing: bool) -> None:
        self._music_play_act.setChecked(playing)

    def _on_music_muted(self, muted: bool) -> None:
        self._music_mute_act.setChecked(muted)

    def _on_music_volume(self, vol: float) -> None:
        self._sync_volume_check(vol)

    def _sync_volume_check(self, vol: float) -> None:
        # Match closest 10%-step menu item
        nearest = round(vol * 10) * 10
        for act in self._music_vol_group.actions():
            label = act.text().rstrip("%")
            if label.isdigit() and int(label) == nearest:
                act.setChecked(True)
                break

    def _show_about(self) -> None:
        # Build provenance: BUILD_SHA / BUILD_TAG / BUILD_TIMESTAMP are written
        # by CI immediately before PyInstaller bundles the EXE (see
        # .github/workflows/release.yml). In a dev checkout the file ships
        # placeholders ("dev" / "" / "") which we render as "dev build".
        from forza_abyss_painter import _build_info
        if _build_info.BUILD_SHA == "dev":
            build_line = "<i>dev build (uncommitted local checkout)</i>"
        else:
            tag_part = f"{_build_info.BUILD_TAG} · " if _build_info.BUILD_TAG else ""
            build_line = (
                f"<i>{tag_part}commit "
                f"<code>{_build_info.BUILD_SHA[:7]}</code>"
                f"{' · ' + _build_info.BUILD_TIMESTAMP if _build_info.BUILD_TIMESTAMP else ''}"
                f"</i>"
            )
        QMessageBox.about(
            self,
            "About Forza Abyss Painter",
            f"<b>Forza Abyss Painter</b><br>v1.0.0<br>"
            f"{build_line}<br><br>"
            f"<i>For Forza Horizon 3 / 4 / 5 / 6 (FH6 build {FH6_TARGET_BUILD}) "
            f"and Assetto Corsa Competizione</i><br><br>"
            "Vinyl-design tool for Forza Horizon 3-6 + Assetto Corsa Competizione. "
            "Forza titles: live memory injection of vinyl-group shapes (position, "
            "scale, rotation, color). Assetto Corsa Competizione: file-based PNG "
            "livery export to the user's Documents folder.<br><br>"
            "A fork of <b>tokyubevoxelverse/ForzaDesigner6</b> with GPU shape-gen "
            "and injector performance improvements: sampled revalidation "
            "(~1.83x syscall reduction on 3000-shape injects), GPU shape-gen "
            "pipeline (Colab notebooks for 200/1000/3000-shape JSONs), and the "
            "polish_freeze_geometry production mode for byte-parity polish.<br><br>"
            "Inspired by forza-painter (the_adawg), built on the techniques of "
            "geometrize-lib (Sam Twidale) and Primitive (Michael Fogleman). "
            "LiveryGroup discovery approach adapted from bvzrays/forza-painter-fh6.<br><br>"
            "Splash music: <b>“HELLO.SPIRAL”</b> by <b>CelesteAI</b>.<br><br>"
            "Repo: <a href='https://github.com/whykusanagi/forza-abyss-painter'>"
            "github.com/whykusanagi/forza-abyss-painter</a><br><br>"
            "If FH6 patches and injection breaks, the LiveryGroup offsets in "
            "<code>forza_abyss_painter/inject/fh6_injector.py</code> need to be "
            "re-derived for the new build."
        )

    def _refresh_inject_button(self) -> None:
        ready = patterns_are_populated()
        self.settings_panel.inject_btn.setEnabled(ready)
        tip = (
            "Phase 2 feature. Requires FH6 memory patterns. See README §Phase 2."
            if not ready else
            "Inject the most recent shapes into a running FH6 vinyl group."
        )
        self.settings_panel.inject_btn.setToolTip(tip)

    def _on_install_gpu_runtime(self) -> None:
        """Tools menu → Install GPU runtime. Direct entry point — opens
        the install dialog regardless of current install state so users
        can install fresh or re-install over a partial state.

        After the dialog closes, refresh the status indicator so the
        user immediately sees the new GPU state."""
        from forza_abyss_painter.gui.runtime_install_dialog import (
            RuntimeInstallDialog,
        )
        dlg = RuntimeInstallDialog(self)
        dlg.exec()
        # Refresh BOTH GPU indicators (status bar + settings dropdown)
        # so the user sees the result of the install they just ran
        # without needing to restart the EXE.
        self._refresh_gpu_status_indicator()
        self.settings_panel.refresh_backend_state()
        if dlg.was_installed:
            self.statusBar().showMessage(
                "GPU runtime installed — ready to Generate.", 6000,
            )

    def _on_generate_locally(self) -> None:
        """Tools menu → Generate shapes locally. Lazy-import the dialog so
        the runtime modules don't get loaded on startup (kept off the hot
        path for inject-only users). If runtime isn't installed yet, the
        helper prompts the user; on success or already-installed it opens
        the generate-workflow dialog, then auto-loads the resulting JSON
        into the preview panel using the existing Upload JSON flow."""
        from forza_abyss_painter.gui.generate_dialog import (
            open_generate_dialog_if_runtime_ready,
        )
        out = open_generate_dialog_if_runtime_ready(self)
        if out is not None and out.exists():
            # Reuse the existing JSON-load preview path so the generated
            # output flows through the same preview + inject lineup as an
            # Upload JSON-sourced file.
            self._on_json_loaded_for_preview(out)
            self.statusBar().showMessage(
                f"Generated {out.name} — ready to inject.", 8000,
            )

    def _on_validate_current_json(self) -> None:
        """Tools menu → Validate current JSON. Re-runs validation on
        the JSON loaded via Upload JSON and shows every finding in a
        scrollable modal. If the user hasn't loaded a JSON yet, shows
        a hint instead of the dialog (the menu action is supposed to
        be disabled in that state, but defensive guard).

        Re-reads the file from disk rather than using the cached issue
        list — that way users get fresh validation after a hand-edit
        without having to Upload JSON again."""
        from forza_abyss_painter.gui.validation_dialog import show_validation_dialog
        if not self._loaded_json_path or not self._loaded_json_path.exists():
            QMessageBox.information(
                self, "Validate JSON",
                "Load a JSON via Upload JSON first, then re-run this "
                "to see schema findings.",
            )
            return
        # Re-read + re-validate from disk. Pulls in any external edits.
        import json
        from forza_abyss_painter.io.json_schema import FD6Document
        from forza_abyss_painter.io.validator import validate_document
        try:
            raw = json.loads(self._loaded_json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            QMessageBox.critical(
                self, "Re-validate failed",
                f"Could not re-read {self._loaded_json_path.name}: "
                f"{type(exc).__name__}: {exc}",
            )
            return
        # Run through FD6Document.from_dict so legacy JSONs get
        # normalized to v1 before validation — same path as load_json.
        try:
            doc = FD6Document.from_dict(raw)
            issues = validate_document(doc.to_dict())
        except Exception as exc:
            QMessageBox.critical(
                self, "Re-validate failed",
                f"Could not normalize document: {type(exc).__name__}: {exc}",
            )
            return
        # Update the cache so subsequent inject-time reads see the
        # fresh findings too.
        self._loaded_json_issues = issues
        show_validation_dialog(
            self, issues,
            title=f"Validation: {self._loaded_json_path.name}",
        )

    def _on_save_diagnostics(self) -> None:
        """Tools menu → Save diagnostics zip. Opens a file picker for
        the output path, calls the diagnostics-bundle library, then
        shows the result in the status bar so the user knows where the
        file landed. On error, surfaces a modal with the OS error
        message so the user knows what to retry."""
        from PySide6.QtWidgets import QFileDialog, QMessageBox
        from datetime import datetime, timezone
        from forza_abyss_painter.runtime.diagnostics import build_bundle

        default = (
            Path.home() /
            f"ForzaAbyssPainter-diag-"
            f"{datetime.now(timezone.utc).strftime('%Y-%m-%d-%H%M%S')}.zip"
        )
        path, _ = QFileDialog.getSaveFileName(
            self, "Save diagnostics zip",
            str(default), "Zip archive (*.zip)",
        )
        if not path:
            return
        try:
            result = build_bundle(Path(path))
        except OSError as exc:
            QMessageBox.critical(
                self, "Couldn't save diagnostics",
                f"Failed to write bundle:\n\n"
                f"{type(exc).__name__}: {exc}",
            )
            return
        size_mb = result.stat().st_size / (1 << 20)
        self.statusBar().showMessage(
            f"Diagnostics saved to {result.name} ({size_mb:.1f} MiB)",
            10000,
        )

    def _on_clean_json(self) -> None:
        """Tools menu → Clean current JSON. Lazy-import the dialog so
        the cleanup deps don't load until first use. On a successful
        save the cleaned JSON auto-loads into the preview panel via
        the same path Upload JSON uses — so the user immediately sees
        the cleaned result and can re-inject without an extra click."""
        from forza_abyss_painter.gui.clean_dialog import open_clean_json_dialog
        out = open_clean_json_dialog(self)
        if out is not None and out.exists():
            self._on_json_loaded_for_preview(out)
            self.statusBar().showMessage(
                f"Cleaned JSON saved to {out.name} — ready to inject.", 8000,
            )

    def _on_files_selected(self, paths: list[Path]) -> None:
        if self._suite_mode == SuiteMode.AC:
            # AC mode: no queue, no auto-start. Track the most-recently
            # uploaded file as the source for the next Export click, show it
            # in the source pane, AND build the slot previews immediately so
            # the user can see what'll be written before clicking Export.
            if paths:
                self._current_path = Path(paths[-1])
                self.ac_preview.set_source(self._current_path)
                self._refresh_ac_preview()
                self.statusBar().showMessage(
                    f"Loaded {self._current_path.name}. "
                    "Adjust settings then click Export to write the livery.",
                    6000,
                )
            return
        # Forza path — unchanged from v0.3.0 behavior.
        for p in paths:
            self.queue.add(p)
        if self._worker is None:
            self._start_next()

    def _refresh_ac_preview(self) -> None:
        """Rebuild the AC cycling-slot preview from the current source image
        and the settings-panel state. Called on upload and (in future) on
        settings changes. Cheap enough at default resolutions to run inline.
        """
        if not getattr(self, "_current_path", None):
            return
        try:
            from forza_abyss_painter.ac.slot_planner import plan_slots
            from forza_abyss_painter.ac.texture_pipeline import build_decal_texture
            cfg = self.ac_settings._gather_export_config()
            rgba, applied_aspect = build_decal_texture(
                self._current_path,
                target_long_edge=int(cfg["resolution"]),
                aspect_choice=str(cfg["aspect"]),
            )
            slot_filenames = plan_slots(
                auto=bool(cfg["auto_slot"]),
                manual_main=cfg.get("manual_main_slots"),
                manual_sponsors=cfg.get("manual_sponsor_slots"),
            )
            self.ac_preview.set_slots([(s, rgba) for s in slot_filenames])
            # Surface "preview is ready" so users don't think the pane is empty.
            h, w, _ = rgba.shape
            self.ac_preview.status_label.setText(
                f"Preview ready — {w}×{h}  •  aspect {applied_aspect}  •  "
                f"{len(slot_filenames)} slot(s) ready to write. "
                f"Click Export when satisfied."
            )
            self.ac_preview.progress.setValue(100)
        except Exception as exc:
            # Preview is best-effort; never block on a failure here.
            self.ac_preview.status_label.setText(
                f"Preview build failed: {type(exc).__name__}: {exc}"
            )
            self.ac_preview.progress.setValue(0)

    def _start_next(self) -> None:
        if self._worker is not None:
            return  # already running
        next_path = self.queue.pop_next_queued()
        if next_path is None:
            self.statusBar().showMessage("Nothing queued.")
            return
        profile = self.settings_panel.build_profile()
        self._current_path = next_path
        self._current_profile = profile
        self.preview.set_source(next_path)
        self.queue.set_status(next_path, "running")

        # Pull sticker-mode (transparent-background) toggle from settings panel.
        # When ON (default), we composite transparent areas onto white before generation.
        # When OFF, transparent areas remain transparent and don't get shapes.
        add_white_bg = self.settings_panel.sticker_mode_cb.isChecked()

        # Branch on user-selected backend. CPU path is the original
        # in-process GenerationWorker with live shape-by-shape preview.
        # GPU path subprocess'es torch_runner via GpuGenWorker — no
        # live preview (subprocess writes JSON at the end), but it's
        # 5-30× faster. See settings_panel for the selector.
        backend = self.settings_panel.selected_backend()
        if backend == "gpu":
            self._start_gpu(next_path, profile, sticker_mode=not add_white_bg)
            return

        self._worker = GenerationWorker(next_path, profile, sticker_mode=not add_white_bg)
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self.preview.on_progress)
        self._worker.preview.connect(self.preview.on_preview)
        self._worker.checkpoint_written.connect(lambda p: self.statusBar().showMessage(f"Checkpoint: {p}", 4000))
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._thread.start()
        self.settings_panel.set_running(True)
        self.statusBar().showMessage(f"Generating: {next_path.name}")

    def _start_gpu(self, image_path: Path, profile, sticker_mode: bool) -> None:
        """GPU shape-gen path. Writes an IPC config + spawns GpuGenWorker
        in a QThread. Progress shows in the status bar (no live preview
        — subprocess writes the final JSON at the end). On done: load
        the JSON into the preview pane via the same _on_json_loaded
        flow Upload JSON uses, so the user gets identical post-gen UX
        regardless of backend."""
        import json as _json
        from forza_abyss_painter.gui.gpu_gen_worker import GpuGenWorker
        from forza_abyss_painter.runtime.torch_installer import (
            embedded_python_exe, is_runtime_installed,
        )
        if not is_runtime_installed():
            # Defensive — settings panel should have prompted install,
            # but the user may have cancelled. Surface a clear modal.
            QMessageBox.warning(
                self, "GPU runtime not installed",
                "Pick 'GPU' from 'Generate using:' and accept the runtime "
                "install when prompted, then click Start again.",
            )
            self.queue.set_status(image_path, "queued")
            return

        # Build the baked preset dict (what the profile defaults to).
        preset_baked = {
            "label": profile.name,
            "num_shapes": profile.stop_at,
            "max_resolution": profile.max_resolution,
            "random_samples": profile.random_samples,
        }
        budget_gib = self.settings_panel.selected_vram_budget_gib()

        # Probe free VRAM ONCE — preflight + autotune share the result so
        # they see the same VRAM snapshot (no TOCTOU, no double nvidia-smi
        # subprocess spawn). (#131 review GAP 4)
        probe = probe_free_vram(force=True)
        free_gib = probe.free_gib if probe.available else None

        # Autotune the preset FIRST (#131 review BLOCKER 1). Subsequent
        # peak_gib estimate + preflight evaluate against the BUMPED
        # max_resolution — otherwise a baked-at-480 preset that autotunes
        # to 1319 would pass the block-guard at the baked peak but OOM at
        # the bumped peak.
        preset, autotune_message = _apply_autotune_to_preset(
            preset_baked, probe=probe,
        )

        # Compute peak_gib from the BUMPED preset, not the baked profile.
        # This is the painter-fh6 architectural trade-off generalized:
        # "set budget, let it cook" — but now the guard sees the real
        # post-autotune estimate, not the pre-autotune baked value.
        from forza_abyss_painter.shapegen.gpu.vram_planner import (
            estimate_peak_vram_gib,
        )
        peak_gib = estimate_peak_vram_gib(
            K=int(preset["random_samples"]),
            bbox_local=True,
            max_resolution=int(preset["max_resolution"]),
        )

        # Three-tier preflight (Item E): block / warn / ok.
        # Cursor's Run 4 showed chunked-K reduces K-batch peak but not
        # the total process footprint (canvas + refill + joint_polish).
        # The 15× safety multiplier in vram_planner is calibrated against
        # that — if peak > free, the run WILL OOM regardless of chunking.
        severity, summary = _vram_preflight_verdict(peak_gib, free_gib, budget_gib)

        if severity == "block":
            QMessageBox.critical(
                self,
                "Won't fit in free VRAM — cannot start",
                f"<b>{summary}</b><br><br>"
                f"This preset (after #131 auto-tune to "
                f"<b>{preset['max_resolution']} px</b>) estimates a peak "
                f"of <b>~{peak_gib:.1f} GiB</b> but only "
                f"<b>{free_gib:.1f} GiB</b> is currently free on "
                f"{probe.name or 'the GPU'}.<br><br>"
                f"<b>Chunked-K can't save this run</b> — the estimate "
                f"already includes non-K overhead (canvas, refill, "
                f"joint_polish). Run 4 evidence: 47.5 GiB allocated on "
                f"a 32 GiB GPU even with chunking active.<br><br>"
                f"<b>Fixes (in order of effort):</b><br>"
                f"  • Close FH6, Chrome, Discord, other GPU apps<br>"
                f"  • Lower <b>Random samples</b> (e.g. 8192 → 4096)<br>"
                f"  • Lower <b>Max resolution</b> (e.g. 720 → 600)<br>"
                f"  • Pick a smaller preset",
                QMessageBox.Ok,
            )
            self.queue.set_status(image_path, "queued")
            return

        if severity == "warn":
            answer = QMessageBox.question(
                self, "Less free VRAM than expected",
                f"<b>{summary}</b><br><br>"
                f"This preset estimates ~{peak_gib:.1f} GiB peak vs "
                f"{free_gib:.1f} GiB free on "
                f"{probe.name or 'the GPU'} (budget {budget_gib} GiB).<br><br>"
                f"Likely cause: another process (FH6, browser, another "
                f"GPU job) is holding memory.<br><br>"
                f"<b>Recommended:</b> close other GPU apps and re-Start. "
                f"Or proceed and let the engine chunk-K against the "
                f"smaller free pool — slower, may still OOM if free "
                f"shrinks further.<br><br>"
                f"Proceed anyway?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                self.queue.set_status(image_path, "queued")
                return

        # Chunked-K dialog uses the bumped peak_gib too (#131 review BLOCKER 2).
        chunk_warning = ""
        if budget_gib > 0 and peak_gib > budget_gib * 0.85:
            n_chunks = max(2, int(peak_gib / (budget_gib * 0.85)) + 1)
            answer = QMessageBox.question(
                self, "Chunked GPU run — longer wall-time",
                f"<b>Your settings would need ~{peak_gib:.1f} GiB</b> at "
                f"full-K, but your budget is <b>{budget_gib} GiB</b>.<br><br>"
                f"To stay under budget, the engine will auto-chunk the "
                f"candidate batch into <b>~{n_chunks} smaller chunks</b>.<br><br>"
                f"<b>Expected:</b><br>"
                f"  • Wall time roughly <b>{n_chunks}× longer</b> than a "
                f"single-chunk run<br>"
                f"  • Peak VRAM stays under {budget_gib} GiB — other apps "
                f"(FH6, Discord) keep their GPU memory<br>"
                f"  • Same quality output — chunking only changes "
                f"<i>when</i> candidates are scored, not <i>what</i> "
                f"candidates are tried<br><br>"
                f"<b>If you want it faster instead:</b><br>"
                f"  • Raise <b>GPU VRAM budget</b> in settings to a "
                f"higher tier<br>"
                f"  • Or lower <b>Random samples</b> / "
                f"<b>Max resolution</b> for a single-chunk run<br><br>"
                f"Proceed with chunked run?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,   # default Yes — chunking just works
            )
            if answer != QMessageBox.Yes:
                self.queue.set_status(image_path, "queued")
                return
            chunk_warning = (
                f" (auto-chunked into ~{n_chunks}, wall time ~{n_chunks}×)"
            )
        from forza_abyss_painter.gui.gpu_gen_worker import build_run_config
        output_path = image_path.parent / image_path.stem / f"{image_path.stem}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        config = build_run_config(image_path, output_path, preset,
                                    sticker_mode=sticker_mode,
                                    vram_budget_gib=float(budget_gib))
        config_path = image_path.parent / f".{image_path.stem}_gpu_config.json"
        try:
            config_path.write_text(_json.dumps(config, indent=2),
                                    encoding="utf-8")
        except OSError as exc:
            QMessageBox.critical(
                self, "Couldn't start GPU run",
                f"Failed to write GPU config: {exc}",
            )
            self.queue.set_status(image_path, "queued")
            return
        self._worker = GpuGenWorker(
            embedded_python_exe=embedded_python_exe(),
            config_path=config_path,
        )
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        # Display the source image immediately so the middle panel
        # stops sitting on "Idle". The CPU path does this around line
        # 1068 — the GPU path used to skip it (regression).
        self.preview.set_source(image_path)
        # Update both the middle PreviewPanel (progress bar + status
        # label) AND the bottom QMainWindow status bar via _on_gpu_progress
        # (which adds the rate + ETA context).
        self._worker.progress.connect(self.preview.on_progress)
        self._worker.checkpoint.connect(self.preview.on_progress)
        self._worker.progress.connect(self._on_gpu_progress)
        self._worker.checkpoint.connect(self._on_gpu_progress)
        self._worker.snapshot.connect(self._on_gpu_snapshot)
        self._worker.done.connect(self._on_gpu_done)
        self._worker.error.connect(self._on_gpu_error)
        self._worker.finished.connect(self._teardown_thread)
        # Capture start time so the GPU progress slot can compute a
        # live shapes/sec rate + ETA — same formula the colab pipeline
        # prints (engine.py:522-527 in the sister repo). Without this,
        # users staring at a frozen "shape N of T" indicator have no
        # way to judge whether the run takes 1 minute or 30.
        import time as _time
        self._gpu_run_start_t = _time.monotonic()
        self._thread.start()
        self.settings_panel.set_running(True)
        # Embed the autotune outcome in the "GPU generating" message
        # (#131 review GAP 3 fix). Previously the autotune showMessage(6000ms)
        # was synchronously overwritten by this line before the event loop
        # rendered a single frame.
        self.statusBar().showMessage(
            f"GPU generating: {image_path.name}{chunk_warning} — "
            f"{autotune_message} (live progress in preview panel)"
        )

    def _on_gpu_progress(self, shape_count: int, total: int) -> None:
        if total <= 0:
            return
        pct = max(0, min(100, int(100 * shape_count / total)))
        # Live rate + ETA from start-of-run timestamp. Mirrors the
        # colab engine's `print(f"{idx}/{total} ({rate}/s, ETA {eta}s, RMS …)")`
        # so users see a moving clock + speed signal regardless of
        # GPU model. Skip the ETA on the first event (rate not yet
        # meaningful with elapsed near zero).
        import time as _time
        elapsed = _time.monotonic() - getattr(self, "_gpu_run_start_t",
                                              _time.monotonic())
        if elapsed > 1.0 and shape_count > 0:
            rate = shape_count / elapsed
            eta_s = int((total - shape_count) / rate) if rate > 0 else 0
            mins, secs = divmod(eta_s, 60)
            eta_str = f"{mins}m {secs:02d}s" if mins else f"{secs}s"
            self.statusBar().showMessage(
                f"GPU: shape {shape_count} of {total} ({pct}%) — "
                f"{rate:.1f} shapes/s, ETA {eta_str}"
            )
        else:
            self.statusBar().showMessage(
                f"GPU: shape {shape_count} of {total} ({pct}%)"
            )

    def _on_gpu_done(self, output_path: str, shape_count: int) -> None:
        if self._current_path:
            self.queue.set_status(self._current_path, "done")
        out = Path(output_path)
        self._last_finished_json = out
        self.statusBar().showMessage(
            f"GPU done — {shape_count} shapes saved to {out.name}", 8000,
        )
        # Auto-load into preview via the same path Upload JSON uses,
        # so the user immediately sees the result + can inject.
        self._on_json_loaded_for_preview(out)
        self.upload.mark_json_ready(out)

    def _on_gpu_error(self, stage: str, message: str) -> None:
        if self._current_path:
            self.queue.set_status(self._current_path, "error")
        QMessageBox.critical(
            self, f"GPU generation failed — {stage}",
            f"Stage: {stage}\n\n{message}\n\n"
            f"For post-mortem: Tools → Save diagnostics zip…",
        )

    def _on_gpu_install_requested(self) -> None:
        """SettingsPanel emitted gpu_install_requested — user picked GPU
        from the backend dropdown but the runtime isn't installed. Open
        the install dialog; on close, refresh the dropdown label so the
        user sees the new state."""
        from forza_abyss_painter.gui.runtime_install_dialog import (
            RuntimeInstallDialog,
        )
        dlg = RuntimeInstallDialog(self)
        dlg.exec()
        self.settings_panel.refresh_backend_state()
        self._refresh_gpu_status_indicator()

    def _on_finished(self, out_path: str) -> None:
        if self._current_path:
            self.queue.set_status(self._current_path, "done")
        self._last_finished_json = Path(out_path)
        self.statusBar().showMessage(f"Saved: {out_path}", 8000)
        # Visual cue: green-pulse the Download JSON button so the user knows it's ready
        self.upload.mark_json_ready(self._last_finished_json)
        self._teardown_thread()
        # Auto-start next
        self._start_next()

    def _on_error(self, msg: str) -> None:
        if self._current_path:
            self.queue.set_status(self._current_path, "error")
        QMessageBox.critical(self, "Generation error", msg)
        self._teardown_thread()

    def _teardown_thread(self) -> None:
        if self._thread:
            self._thread.quit()
            self._thread.wait(3000)
        self._worker = None
        self._thread = None
        self._current_path = None
        self.settings_panel.set_running(False)
        self.settings_panel.pause_btn.setChecked(False)

    def _toggle_pause(self) -> None:
        if not self._worker:
            return
        paused = self.settings_panel.pause_btn.isChecked()
        self._worker.set_pause(paused)
        self.statusBar().showMessage("Paused." if paused else "Resumed.", 3000)

    def _stop_current(self) -> None:
        if self._worker:
            self._worker.stop()

    def _on_inject_clicked(self) -> None:
        from PySide6.QtWidgets import QFileDialog
        # If a JSON is already loaded into preview, inject that. Else prompt.
        if self._loaded_json_path and self._loaded_json_path.exists():
            self._on_inject_json_path(self._loaded_json_path)
            return
        json_path, _ = QFileDialog.getOpenFileName(
            self, "Pick shapes JSON to inject", "", "Forza Abyss Painter shapes (*.json);;All files (*)"
        )
        if json_path:
            self._on_inject_json_path(Path(json_path))

    def _prompt_source_image_picker(self, hint_filename: str) -> Path | None:
        """Fallback file picker when the sibling source image is missing.
        Returns the user-picked path or None on cancel."""
        from PySide6.QtWidgets import QFileDialog, QMessageBox
        QMessageBox.information(
            self, "Source image not found",
            f"The loaded JSON refers to source image '{hint_filename}', "
            f"but no file with that name exists next to the JSON. "
            f"Pick the source image manually.",
        )
        path, _ = QFileDialog.getOpenFileName(
            self, f"Pick source image (looking for '{hint_filename}')",
            "", "Images (*.png *.jpg *.jpeg *.webp);;All files (*)",
        )
        return Path(path) if path else None

    def _resolve_source_for_loaded_json(self, json_path: Path) -> Path | None:
        """Resolve the source image for a loaded JSON. Returns None when
        the user cancels the picker fallback — caller aborts silently."""
        from forza_abyss_painter.io.exporter import load_json
        try:
            doc = load_json(str(json_path))
        except Exception:
            # Validator already gated us on load; getting here means the
            # file changed between load and this call. Treat as fatal.
            return None
        sibling = _resolve_source_image_path(json_path, doc.source_image)
        if sibling is not None:
            return sibling
        return self._prompt_source_image_picker(doc.source_image or "<unknown>")

    def _on_reshape_requested(self, json_path: Path) -> None:
        """#85 — user clicked Re-shape-gen at higher budget. Resolve the
        source image, then open the existing GenerateLocallyDialog
        pre-populated with that source."""
        source = self._resolve_source_for_loaded_json(json_path)
        if source is None:
            return
        from forza_abyss_painter.gui.generate_dialog import GenerateLocallyDialog
        from forza_abyss_painter.gui.runtime_install_dialog import (
            prompt_install_or_use_existing,
        )
        if not prompt_install_or_use_existing(self):
            return
        dlg = GenerateLocallyDialog(self, initial_source_path=source)
        from PySide6.QtWidgets import QDialog as _QDialog
        if dlg.exec() == _QDialog.DialogCode.Accepted and dlg.output_path:
            self._on_json_loaded_for_preview(dlg.output_path)

    def _on_polish_requested(self, json_path: Path) -> None:
        """#86 — user clicked Polish loaded JSON. Resolve the source
        image, open PolishDialog, on accept spawn GpuGenWorker with the
        polish config."""
        from PySide6.QtWidgets import QMessageBox
        if getattr(self, "_polish_thread", None) and self._polish_thread.isRunning():
            from PySide6.QtWidgets import QMessageBox as _QMessageBox
            _QMessageBox.information(self, "Polish running",
                                       "A polish run is already in progress.")
            return
        source = self._resolve_source_for_loaded_json(json_path)
        if source is None:
            return
        from forza_abyss_painter.gui.polish_dialog import PolishDialog
        from forza_abyss_painter.gui.gpu_gen_worker import (
            GpuGenWorker, build_polish_config,
        )
        from forza_abyss_painter.gui.runtime_install_dialog import (
            prompt_install_or_use_existing,
        )
        from forza_abyss_painter.runtime.torch_installer import embedded_python_exe
        if not prompt_install_or_use_existing(self):
            return
        dlg = PolishDialog(self,
                            loaded_json_path=json_path,
                            source_image_path=source)
        from PySide6.QtWidgets import QDialog as _QDialog
        if dlg.exec() != _QDialog.DialogCode.Accepted:
            return
        values = dlg.values()

        # Mirror the fresh-gen sticker convention: sticker_mode = "no
        # white-bg composite" = the "Add white background" checkbox is
        # UNCHECKED. `not isChecked()` → True means transparent target
        # → sticker_mode=True.
        sticker = bool(getattr(self.settings_panel, "sticker_mode_cb", None) and
                        not self.settings_panel.sticker_mode_cb.isChecked())

        config = build_polish_config(
            source_image_path=source,
            input_shapes_path=json_path,
            output_path=values["output_path"],
            steps=values["steps"],
            lock_alpha=values["lock_alpha"],
            sticker_mode=sticker,
        )
        config_path = json_path.parent / f".{json_path.stem}_polish_config.json"
        try:
            config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        except OSError as exc:
            QMessageBox.critical(self, "Couldn't start polish",
                                  f"Failed to write polish config: {exc}")
            return
        py = embedded_python_exe()
        if not py.exists():
            QMessageBox.critical(
                self, "GPU runtime missing",
                f"Embedded Python not found at {py}. "
                f"Open Tools → Generate shapes locally to install the runtime.",
            )
            return

        # Spawn the worker on a QThread — same pattern GenerateLocallyDialog uses.
        self._polish_thread = QThread(self)
        self._polish_worker = GpuGenWorker(
            embedded_python_exe=py,
            config_path=config_path,
        )
        self._polish_worker.moveToThread(self._polish_thread)
        self._polish_thread.started.connect(self._polish_worker.run)
        self._polish_worker.started.connect(self._on_polish_started)
        self._polish_worker.progress.connect(self._on_polish_progress)
        self._polish_worker.checkpoint.connect(self._on_polish_progress)
        self._polish_worker.done.connect(self._on_polish_done)
        self._polish_worker.error.connect(self._on_polish_error)
        self._polish_worker.finished.connect(self._polish_thread.quit)
        self._polish_worker.finished.connect(self._polish_worker.deleteLater)
        self._polish_thread.finished.connect(self._polish_thread.deleteLater)
        self.statusBar().showMessage("Polishing — running optimizer on GPU…")
        self._polish_thread.start()

    # ------------------------------------------------------------------
    # Snapshot preview wiring
    # ------------------------------------------------------------------

    def _on_gpu_snapshot(self, count: int, total: int, snapshot_path: str) -> None:
        """Runner just wrote a snapshot; dispatch a render off-thread.

        Single-slot throttle: if a render is already in flight, just
        remember the latest path. When the in-flight job finishes, if
        a newer path is pending, start it. Drops intermediate renders
        if snapshots fire faster than render — for GPU at every-100,
        this rarely matters but it bounds memory + thread churn.
        """
        from pathlib import Path as _Path
        self.statusBar().showMessage(
            f"GPU: snapshot saved at {count}/{total} → "
            f"{_Path(snapshot_path).name}", 2000,
        )
        if self._snapshot_render_in_flight:
            self._snapshot_pending_path = snapshot_path
            return
        self._dispatch_snapshot_render(snapshot_path)

    def _dispatch_snapshot_render(self, snapshot_path: str) -> None:
        from PySide6.QtCore import QThreadPool, QTimer
        from forza_abyss_painter.gui.snapshot_render import _RenderSnapshotJob
        self._snapshot_render_in_flight = True
        job = _RenderSnapshotJob(snapshot_path, self.preview)
        QThreadPool.globalInstance().start(job)
        # QRunnable doesn't expose a finished signal; use a one-shot
        # timer (3s) to clear the flag + dispatch any pending render.
        # If a render actually takes longer, the next snapshot just
        # adds to the pending queue with no harm.
        QTimer.singleShot(3000, self._snapshot_render_drain)

    def _snapshot_render_drain(self) -> None:
        self._snapshot_render_in_flight = False
        if self._snapshot_pending_path is not None:
            pending = self._snapshot_pending_path
            self._snapshot_pending_path = None
            self._dispatch_snapshot_render(pending)

    # ------------------------------------------------------------------
    # Resume-from-snapshot wiring
    # ------------------------------------------------------------------

    def _on_resume_requested(self, snapshot_path: "Path") -> None:
        """User picked a snapshot to resume from. Resolve source image,
        open ResumeDialog, on accept spawn GpuGenWorker with the
        dialog's values dict."""
        from PySide6.QtWidgets import QMessageBox, QFileDialog
        from PySide6.QtWidgets import QDialog as _QDialog
        from pathlib import Path as _Path

        # Load + sanity-check snapshot.
        try:
            from forza_abyss_painter.io.exporter import load_json
            doc = load_json(str(snapshot_path))
        except Exception as exc:
            QMessageBox.critical(
                self, "Couldn't load snapshot",
                f"Snapshot {snapshot_path.name} could not be loaded:\n\n"
                f"{type(exc).__name__}: {exc}",
            )
            return

        # Resolve source image (same-folder heuristic + picker).
        sibling = _resolve_source_image_path(snapshot_path, doc.source_image)
        if sibling is not None:
            source = sibling
        else:
            QMessageBox.information(
                self, "Source image not found",
                f"Snapshot references '{doc.source_image}' but it's not "
                f"next to the snapshot file. Pick it manually.",
            )
            picked, _ = QFileDialog.getOpenFileName(
                self, f"Pick source image (looking for '{doc.source_image}')",
                "", "Images (*.png *.jpg *.jpeg *.webp);;All files (*)",
            )
            if not picked:
                return
            source = _Path(picked)

        # Runtime install check (resume uses the GPU runner).
        from forza_abyss_painter.gui.runtime_install_dialog import (
            prompt_install_or_use_existing,
        )
        if not prompt_install_or_use_existing(self):
            return

        # Dialog.
        from forza_abyss_painter.gui.resume_dialog import ResumeDialog
        dlg = ResumeDialog(
            parent=self,
            snapshot_path=snapshot_path,
            source_image_path=source,
        )
        if dlg.exec() != _QDialog.DialogCode.Accepted:
            return
        values = dlg.values()

        # Write the config + spawn GpuGenWorker.
        config_path = snapshot_path.parent / (
            f".{snapshot_path.stem}_resume_config.json"
        )
        try:
            config_path.write_text(
                json.dumps(values, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            QMessageBox.critical(
                self, "Couldn't start resume",
                f"Failed to write resume config: {exc}",
            )
            return

        from forza_abyss_painter.runtime.torch_installer import embedded_python_exe
        py = embedded_python_exe()
        if not py.exists():
            QMessageBox.critical(
                self, "GPU runtime missing",
                f"Embedded Python not found at {py}. "
                f"Open Tools → Generate shapes locally to install the runtime.",
            )
            return

        # Spawn — same pattern as _start_gpu.
        from forza_abyss_painter.gui.gpu_gen_worker import GpuGenWorker
        from PySide6.QtCore import QThread
        self._worker = GpuGenWorker(
            embedded_python_exe=py,
            config_path=config_path,
        )
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self.preview.on_progress)
        self._worker.checkpoint.connect(self.preview.on_progress)
        self._worker.progress.connect(self._on_gpu_progress)
        self._worker.checkpoint.connect(self._on_gpu_progress)
        self._worker.snapshot.connect(self._on_gpu_snapshot)
        self._worker.done.connect(self._on_gpu_done)
        self._worker.error.connect(self._on_gpu_error)
        self._worker.finished.connect(self._teardown_thread)
        import time as _time
        self._gpu_run_start_t = _time.monotonic()
        self.preview.set_source(source)
        self._thread.start()
        self.settings_panel.set_running(True)
        self.statusBar().showMessage(
            f"Resume started: {snapshot_path.name} → "
            f"continuing to {values['num_shapes']} shapes"
        )

    def _on_polish_started(self, summary: dict) -> None:
        self.statusBar().showMessage("Polish started — running joint_polish on GPU…")

    def _on_polish_progress(self, current: int, total: int) -> None:
        if total > 0:
            self.statusBar().showMessage(f"Polish — step {current}/{total}")

    def _on_polish_done(self, output_path: str, shape_count: int) -> None:
        path = Path(output_path)
        self.statusBar().showMessage(
            f"Polish done — {shape_count} shapes saved to {path.name}", 10000,
        )
        # Auto-load the polished JSON into the preview so the user can
        # compare visually + click Inject when ready.
        self._on_json_loaded_for_preview(path)
        self._polish_thread = None
        self._polish_worker = None

    def _on_polish_error(self, stage: str, message: str) -> None:
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.critical(self, f"Polish failed — {stage}",
                              f"Stage: {stage}\n\n{message}")
        self.statusBar().showMessage(f"Polish failed at {stage}.", 10000)
        self._polish_thread = None
        self._polish_worker = None

    def _on_json_loaded_for_preview(self, json_path: Path) -> None:
        """User clicked Upload JSON -> load the file, render shapes onto the preview pane.
        Does NOT inject. User must click Inject into FH6 after to actually push to game.

        Auto-validates the loaded JSON against fd6.shapes v1 schema. ERROR-severity
        findings block the preview (the JSON wouldn't inject anyway); WARNINGs flow
        through to a status-bar message + the cached issue list (re-viewable via
        Tools → Validate current JSON). See #100 / docs/JSON_SPEC.md.
        """
        from forza_abyss_painter.io.exporter import load_json
        from forza_abyss_painter.io.validator import Severity, validate_document
        from forza_abyss_painter.shapegen.render import render_shapes
        from forza_abyss_painter.gui.validation_dialog import (
            show_validation_dialog, summarize_for_status_bar,
        )
        try:
            doc = load_json(str(json_path))
            shapes = doc.materialize_shapes()
        except Exception as exc:
            QMessageBox.critical(self, "Load failed", f"{type(exc).__name__}: {exc}")
            self.upload.set_json_loaded(None)
            return

        # Validate the normalized document (legacy JSONs have already been
        # converted to native v1 by load_json → FD6Document.from_dict).
        validation_issues = validate_document(doc.to_dict())
        errors = [i for i in validation_issues if i.severity is Severity.ERROR]
        if errors:
            # Surface ALL findings (not just errors) so the user sees full
            # context — but block the preview render because a broken JSON
            # would either crash the renderer or paint nonsense.
            show_validation_dialog(
                self, validation_issues,
                title=f"Cannot load: {json_path.name} has validation errors",
            )
            self.statusBar().showMessage(
                f"Load aborted — {len(errors)} validation error(s) in {json_path.name}",
                8000,
            )
            self.upload.set_json_loaded(None)
            return
        # Cache the issues + path so the Tools menu can re-show them
        # without re-loading the file. Status-bar one-liner if there's
        # anything worth mentioning (warnings); silent on clean.
        self._loaded_json_issues = validation_issues
        summary = summarize_for_status_bar(validation_issues)
        if summary:
            self.statusBar().showMessage(
                f"{summary} — review via Tools → Validate current JSON", 6000,
            )
        w, h = doc.image_size if doc.image_size and doc.image_size[0] > 0 else (1200, 800)
        self.statusBar().showMessage(f"Rendering preview of {len(shapes)} shapes from {json_path.name}...")
        # Render with transparent backdrop when EITHER:
        #   - the JSON was generated in sticker mode (sticker_mode=True in file), OR
        #   - the current "Add white background to transparent images" toggle is
        #     UNCHECKED at upload time — the user's current intent overrides the
        #     stored flag, which also covers legacy JSONs that pre-date the field.
        white_bg_checked = self.settings_panel.sticker_mode_cb.isChecked()
        render_transparent = bool(getattr(doc, "sticker_mode", False)) or not white_bg_checked
        canvas = render_shapes(shapes, w, h, background=(255, 255, 255),
                               transparent_bg=render_transparent)
        self.preview.source_view.clear_image()
        self.preview.preview_view.set_numpy(canvas)
        self.preview.status_label.setText(
            f"Loaded {len(shapes)} shapes from '{json_path.name}'  •  {w}x{h}  •  ready to inject"
        )
        self.preview.progress.setValue(100)
        self._loaded_json_path = json_path
        self.upload.set_json_loaded(json_path)
        # Enable Tools → Validate current JSON now that we have something
        # to validate. Stays enabled across subsequent loads.
        if hasattr(self, "_validate_act"):
            self._validate_act.setEnabled(True)
        self.statusBar().showMessage(
            f"Preview ready. Click 'Inject into FH6' to push these shapes into the game.", 8000
        )

    def _on_inject_json_path(self, json_path: Path) -> None:
        """Inject the given shapes JSON into the running FH6 vinyl group.
        Opens a modal in-progress dialog (warns user not to touch FH6) and runs the
        injection in a background QThread. Status bar mirrors the same updates.

        Bulletproof error handling: every step writes a breadcrumb to a debug
        log at ~/Library/Logs/ForzaAbyssPainter/main_window_inject_debug.log
        (or %LOCALAPPDATA%/ForzaAbyssPainter/logs/ on Windows) so silent
        failures between user-clicks-OK and worker-thread-actually-starts
        leave a paper trail. Previously a silent exception (eg in the picker
        comparison or an import) would close the picker and do nothing
        visible — exactly what the user reported.
        """
        # ---- Breadcrumb logger (independent of the worker's log; the worker
        # may never run if we crash before thread.start()).
        from forza_abyss_painter.io.log_paths import log_root
        from datetime import datetime, timezone
        breadcrumb_path = log_root() / "main_window_inject_debug.log"
        def _crumb(msg: str) -> None:
            try:
                ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                with open(breadcrumb_path, "a", encoding="utf-8") as f:
                    f.write(f"{ts} {msg}\n")
            except Exception:
                pass
        _crumb(f"=== _on_inject_json_path entered: json={json_path} ===")

        try:
            from forza_abyss_painter.inject import patterns_are_populated
            from forza_abyss_painter.gui.inject_worker import InjectionWorker
            from forza_abyss_painter.gui.inject_dialog import InjectionDialog
            from forza_abyss_painter.gui.inject_template_picker import TemplateSizePickerDialog
            from forza_abyss_painter.io.exporter import load_json
            _crumb("imports OK")

            if not patterns_are_populated():
                _crumb("patterns_are_populated False → abort")
                QMessageBox.warning(
                    self, "FH6 Injection",
                    "Patterns file is incomplete. Use FH6 → Discovery Workflow… to populate it."
                )
                return

            if getattr(self, "_inject_thread", None) is not None:
                _crumb("inject_thread already exists → abort")
                QMessageBox.information(self, "Inject in progress", "An injection is already running. Wait for it to finish.")
                return

            # Read the JSON shape count up front so the template picker can
            # show the user "your JSON has N shapes" + reject sizes that
            # would overflow.
            try:
                doc_preview = load_json(str(json_path))
                json_shape_count = len(doc_preview.materialize_shapes())
                _crumb(f"json_shape_count = {json_shape_count}")
            except Exception as exc:
                _crumb(f"load_json failed: {type(exc).__name__}: {exc}")
                QMessageBox.warning(
                    self, "FH6 Injection",
                    f"Could not read JSON to count shapes: {type(exc).__name__}: {exc}"
                )
                return

            # Pre-inject template picker. exec() returns truthy (1) for OK,
            # falsy (0) for Cancel. The obvious-looking instance-access form
            # `picker[dot]Accepted` is BROKEN on PySide6 6.x — DialogCode
            # enums don't propagate to subclass instances, so it raises
            # AttributeError that Qt silently swallows in release builds
            # (the user reported "window closes and nothing happens" from
            # exactly that bug). Use truthiness OR QDialog.Accepted instead.
            picker = TemplateSizePickerDialog(self, json_shape_count=json_shape_count)
            _crumb("picker constructed; about to exec")
            exec_result = picker.exec()
            _crumb(f"picker.exec() returned {exec_result!r}")
            if not exec_result:
                _crumb("picker cancelled → return")
                self.statusBar().showMessage("Injection cancelled at template selection.", 4000)
                return
            template_size = picker.selected_template_size
            _crumb(f"picker.selected_template_size = {template_size!r}")

            target_key = self.settings_panel.selected_target_profile_key()
            _crumb(f"target_key = {target_key!r}")
            self._inject_worker = InjectionWorker(
                json_path, profile_key=target_key, template_size=template_size,
            )
            _crumb("InjectionWorker constructed")
            self._inject_thread = QThread(self)
            self._inject_worker.moveToThread(self._inject_thread)
            _crumb("worker moved to thread")

            from forza_abyss_painter.inject.game_profiles import get_profile, default_profile
            try:
                game_label = get_profile(target_key).label
            except ValueError:
                game_label = default_profile().label
            self._inject_dialog = InjectionDialog(self, json_name=json_path.name, game_label=game_label)
            _crumb("InjectionDialog constructed")

            # Wire worker → both dialog and status bar
            self._inject_worker.scan_progress.connect(self._inject_dialog.on_scan_progress)
            self._inject_worker.write_progress.connect(self._inject_dialog.on_write_progress)
            self._inject_worker.status.connect(self._inject_dialog.on_status)
            self._inject_worker.log_path.connect(self._inject_dialog.on_log_path)
            self._inject_worker.done.connect(self._inject_dialog.on_done)

            self._inject_worker.scan_progress.connect(self._on_inject_scan_progress)
            self._inject_worker.write_progress.connect(self._on_inject_write_progress)
            self._inject_worker.status.connect(self._on_inject_status)
            self._inject_worker.done.connect(self._on_inject_done)

            self._inject_thread.started.connect(self._inject_worker.run)
            self._set_inject_status("Starting injection…", "info")
            _crumb("about to thread.start()")
            self._inject_thread.start()
            _crumb("thread.start() returned; entering dialog.exec()")
            self._inject_dialog.exec()
            _crumb("dialog.exec() returned (user closed)")
        except Exception as exc:
            import traceback
            tb = traceback.format_exc()
            _crumb(f"!!! UNHANDLED EXCEPTION: {type(exc).__name__}: {exc}\n{tb}")
            QMessageBox.critical(
                self, "FH6 Injection — internal error",
                f"Unexpected error setting up injection:\n\n"
                f"{type(exc).__name__}: {exc}\n\n"
                f"A trace was written to:\n{breadcrumb_path}\n\n"
                f"Please share that file."
            )

    def _set_inject_status(self, message: str, severity: str = "info") -> None:
        """Color-coded persistent status line at the bottom of the main window.
        severity ∈ {info, success, warning, error}.
        """
        color = {
            "info":    "#cccccc",
            "success": "#2ecc71",
            "warning": "#f1c40f",
            "error":   "#ff4d4d",
        }.get(severity, "#cccccc")
        bg = {
            "info":    "#1f1f1f",
            "success": "#0c2417",
            "warning": "#2a2410",
            "error":   "#2a1414",
        }.get(severity, "#1f1f1f")
        # Use the QMainWindow's status bar — temporary text disappears after a timeout for success,
        # but for ongoing/error stays put.
        sb = self.statusBar()
        sb.setStyleSheet(f"QStatusBar {{ background: {bg}; color: {color}; font-weight: bold; }}")
        sb.showMessage(message, 0 if severity in ("info", "error", "warning") else 10000)

    def _on_inject_status(self, message: str, severity: str) -> None:
        self._set_inject_status(message, severity)

    def _on_inject_scan_progress(self, scanned: int, total: int, hits: int) -> None:
        pct = int(round(100 * scanned / max(1, total)))
        try:
            from forza_abyss_painter.inject.game_profiles import get_profile
            target_key = self.settings_panel.selected_target_profile_key()
            short = get_profile(target_key).label.replace(" (BETA)", "")
            # Compress "Forza Horizon N" → "FHN" for tight status bar text.
            if short.startswith("Forza Horizon ") and short[len("Forza Horizon "):].strip().isdigit():
                short = "FH" + short[len("Forza Horizon "):].strip()
        except Exception:
            short = "FH6"
        self._set_inject_status(
            f"Scanning {short} memory… {scanned}/{total} regions ({pct}%) — {hits} shape structs found so far",
            "info",
        )

    def _on_inject_write_progress(self, written: int, total: int) -> None:
        pct = int(round(100 * written / max(1, total)))
        self._set_inject_status(f"Writing shapes… {written}/{total} ({pct}%)", "info")

    def _on_inject_done(self) -> None:
        if self._inject_thread:
            self._inject_thread.quit()
            self._inject_thread.wait(3000)
        self._inject_worker = None
        self._inject_thread = None
        self._inject_dialog = None

    def _on_download_json(self) -> None:
        """Save (copy) the most-recent generated shapes JSON to a user-chosen location."""
        from PySide6.QtWidgets import QFileDialog
        import shutil
        if not self._last_finished_json or not self._last_finished_json.exists():
            QMessageBox.information(
                self, "No JSON yet",
                "No completed generation yet. Generate from an image first (or use Upload JSON to load an existing one)."
            )
            return
        dest, _ = QFileDialog.getSaveFileName(
            self, "Save shapes JSON as…", self._last_finished_json.name, "Forza Abyss Painter shapes (*.json);;All files (*)"
        )
        if not dest:
            return
        try:
            shutil.copy2(str(self._last_finished_json), dest)
            self.statusBar().showMessage(f"Exported to {dest}", 6000)
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", f"{type(exc).__name__}: {exc}")

    def _show_fh6_status(self) -> None:
        from forza_abyss_painter.inject import discovery as disc
        from forza_abyss_painter.inject.patterns_io import load_patterns, has_usable_patterns

        pid = disc.find_game_pid()
        if pid is None:
            game_line = "forzahorizon6.exe: <b>NOT RUNNING</b>"
        else:
            info = disc.process_summary(pid)
            game_line = (
                f"forzahorizon6.exe PID <b>{info.pid}</b><br>"
                f"&nbsp;&nbsp;committed regions: {info.region_count}<br>"
                f"&nbsp;&nbsp;private+writable bytes: {info.private_writable_bytes:,}<br>"
                f"&nbsp;&nbsp;image bytes: {info.image_bytes:,}"
            )
        pf = load_patterns()
        QMessageBox.information(
            self,
            "FH6 Status",
            f"{game_line}<br><br>"
            f"<b>Patterns file</b><br>"
            f"&nbsp;&nbsp;patterns: {len(pf.patterns)}<br>"
            f"&nbsp;&nbsp;shape_struct.stride: {pf.shape_struct.stride_bytes}<br>"
            f"&nbsp;&nbsp;shape_struct.fields: {len(pf.shape_struct.fields)}<br>"
            f"&nbsp;&nbsp;injector ready: <b>{has_usable_patterns(pf)}</b>"
        )

    def _show_discovery_help(self) -> None:
        QMessageBox.information(
            self,
            "FH6 Discovery Workflow",
            "<p>Discovery is done from the command line, run from the project root:</p>"
            "<pre>"
            "python -m forza_abyss_painter.inject status\n"
            "python -m forza_abyss_painter.inject scan-float &lt;known sphere coord&gt;\n"
            "python -m forza_abyss_painter.inject narrow &lt;moved coord&gt;   (repeat until ~1 hit)\n"
            "python -m forza_abyss_painter.inject dump &lt;addr&gt; 256\n"
            "python -m forza_abyss_painter.inject find-refs &lt;struct_addr&gt;\n"
            "python -m forza_abyss_painter.inject save-pattern shape_array_ref '&lt;AOB&gt;' --offset 3\n"
            "python -m forza_abyss_painter.inject test-injector\n"
            "</pre>"
            "<p>The interactive parts (initial float discovery, struct field identification) "
            "are done with an external memory-scanning tool of your choice — the app only consumes "
            "the resulting AOB pattern and offsets.</p>"
            "<p>Use FH6 → Reload Patterns once you've saved a usable pattern; the Inject "
            "button will then enable.</p>"
        )

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if hasattr(self, "brand_banner") and self.brand_banner is not None:
            self.brand_banner.reposition()

    def showEvent(self, event) -> None:
        """Fire the first-launch suite picker AFTER the splash hands off to us.

        Constructing MainWindow happens during the splash, but show() is only
        called once the splash teardown completes. Triggering the popup
        before that would render it on top of the splash and prevent the user
        from skipping the video. By waiting for showEvent we guarantee the
        main window is the active foreground surface.
        """
        super().showEvent(event)
        if (
            self._suite_first_launch
            and not self._suite_popup_shown_this_session
        ):
            self._suite_popup_shown_this_session = True
            # One event-loop tick of delay so the window has fully painted
            # before the modal blocks it — avoids a black-frame flash.
            QTimer.singleShot(0, self._prompt_suite_on_first_launch)
        # GPU install offer (#99). Fires AFTER the suite picker so the
        # user isn't hit with two modals at once on a brand-new session.
        # Self-gates internally (skips when already installed / opted
        # out / no NVIDIA GPU). Runs once per session via a flag so
        # window-show events on focus toggles don't re-trigger.
        if not getattr(self, "_gpu_first_launch_checked", False):
            self._gpu_first_launch_checked = True
            QTimer.singleShot(50, self._prompt_gpu_install_on_first_launch)
        if hasattr(self, "particles") and self.particles is not None:
            self.particles.reposition()
            self._sync_particle_exclude_rect()
            # Keep brand banner on top of the particle layer so it stays clickable
            if hasattr(self, "brand_banner"):
                self.brand_banner.raise_()

    def _compute_particle_exclude_rect(self):
        """Live rect provider: returns the preview panel's current rect in
        MainWindow client coords (== the particle overlay's coord space)."""
        if not hasattr(self, "preview") or self.preview is None:
            return None
        if self.preview.width() <= 0 or self.preview.height() <= 0:
            return None
        from PySide6.QtCore import QRect
        top_left = self.preview.mapTo(self, self.preview.rect().topLeft())
        return QRect(top_left, self.preview.size())

    def _sync_particle_exclude_rect(self) -> None:
        """Push the current exclude rect once (cached fallback path)."""
        if not hasattr(self, "particles") or self.particles is None:
            return
        excl = self._compute_particle_exclude_rect()
        if excl is not None:
            self.particles.set_exclude_rect(excl)

    def closeEvent(self, event) -> None:
        if self._worker:
            self._worker.stop()
            if self._thread:
                self._thread.quit()
                self._thread.wait(3000)
        # Stop any pending image-search webview cleanly (lazy-init: may be None)
        try:
            if (hasattr(self, "upload")
                    and getattr(self.upload, "image_search", None) is not None):
                self.upload.image_search.shutdown()
        except Exception:
            pass
        super().closeEvent(event)
