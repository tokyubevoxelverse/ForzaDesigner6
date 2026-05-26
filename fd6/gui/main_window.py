from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QThread, QTimer
from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtWidgets import (
    QHBoxLayout, QMainWindow, QMessageBox, QSplitter, QStackedWidget, QStatusBar, QVBoxLayout, QWidget
)

from fd6.gui.ac_settings_panel import ACSettingsPanel
from fd6.gui.brand_banner import BrandBanner, badge_path
from fd6.gui.game_suite_dialog import GameSuiteDialog
from fd6.gui.preview_panel import PreviewPanel
from fd6.gui.texture_preview_panel import TexturePreviewPanel
from fd6.gui.themes import THEMES, apply_theme, saved_theme_name, badge_filename_for_theme
from fd6.gui.queue_panel import QueuePanel
from fd6.gui.settings_panel import SettingsPanel
from fd6.gui.upload_panel import UploadPanel
from fd6.shapegen.profile import Profile
from fd6.suite import SuiteMode, SUITE_DISPLAY, saved_suite_mode, save_suite_mode


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Forza Designer 6+ — Inject custom decals and vinyls into various racing titles")
        self.resize(1280, 760)
        self.setStatusBar(QStatusBar(self))
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
        self.upload.download_json_requested.connect(self._on_download_json)
        self.settings_panel.start_clicked.connect(self._start_next)
        self.settings_panel.pause_clicked.connect(self._toggle_pause)
        self.settings_panel.stop_clicked.connect(self._stop_current)
        self.settings_panel.inject_clicked.connect(self._on_inject_clicked)
        # AC path
        self.ac_settings.export_clicked.connect(self._on_ac_export_clicked)

        # Worker state
        self._worker: GenerationWorker | None = None
        self._thread: QThread | None = None
        self._current_path: Path | None = None
        self._current_profile: Profile | None = None
        self._last_finished_json: Path | None = None  # tracks most recent completed run for Download button
        self._loaded_json_path: Path | None = None    # JSON loaded via Upload JSON (ready to inject)
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
        from fd6.gui.particles import ParticleOverlay
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
        # then start after the main window has been shown.
        from fd6.gui.music import MusicPlayer
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

    def start_music(self) -> None:
        """Begin background music after the main window has been shown."""
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
        from fd6.gui.fonts import (
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

        # --- Localization submenu (v0.4.0 scaffolding — translations land later) ---
        # English is the only working option right now; the rest are listed
        # disabled with a "(coming soon)" suffix so the menu telegraphs the
        # roadmap. Translation strings + Qt linguist .qm files will follow in a
        # subsequent release.
        loc_menu = view_menu.addMenu("&Localization")
        self._loc_group = QActionGroup(self)
        self._loc_group.setExclusive(True)
        loc_options = [
            ("English", True),
            ("Spanish (Español)", False),
            ("French (Français)", False),
            ("German (Deutsch)", False),
            ("Italian (Italiano)", False),
            ("Portuguese (Português)", False),
            ("Dutch (Nederlands)", False),
            ("Polish (Polski)", False),
            ("Russian (Русский)", False),
            ("Japanese (日本語)", False),
            ("Korean (한국어)", False),
            ("Simplified Chinese (简体中文)", False),
            ("Traditional Chinese (繁體中文)", False),
            ("Arabic (العربية)", False),
        ]
        for label, enabled in loc_options:
            act = QAction(label if enabled else f"{label} (coming soon)", self, checkable=True)
            act.setEnabled(enabled)
            act.setChecked(enabled)
            self._loc_group.addAction(act)
            loc_menu.addAction(act)

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
        _cs = _QS("FD6", "Forza Designer 6")
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
        from fd6.gui.particles import COUNT_OPTIONS
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
        about_act = QAction("&About FD6…", self)
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
        s = QSettings("FD6", "Forza Designer 6")
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
        from fd6.ac.livery_writer import write_acc_livery
        from fd6.ac.slot_planner import plan_slots
        from fd6.ac.texture_pipeline import build_decal_texture

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
            team_name=cfg["team_name"] or f"FD6_{Path(self._current_path).stem}",
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
        from fd6.gui.fonts import apply_font
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
        from fd6.inject.fh6_injector import FH6_TARGET_BUILD
        QMessageBox.about(
            self,
            "About Forza Designer 6+",
            f"<b>Forza Designer 6+</b><br>v0.4.0<br>"
            f"<i>For Forza Horizon 3 / 4 / 5 / 6 (FH6 build {FH6_TARGET_BUILD}) "
            f"and Assetto Corsa Competizione</i><br><br>"
            "Multi-game livery suite. Forza titles: live memory injection of "
            "vinyl-group shapes (position, scale, rotation, color). Assetto "
            "Corsa Competizione: file-based PNG livery export to the user's "
            "Documents folder.<br><br>"
            "v0.4.0 highlights: Assetto Corsa Competizione livery export, "
            "multi-game suite picker, expanded Forza target list "
            "(FH3 / FH4 / FH5 / FH6), 2 GiB memory-region read fix, "
            "transparent edge-buffer padding on all generations, JBA Online "
            "GameBoy Emulator banner CTA, and a per-type diagnostic line in "
            "the injection result dialog.<br><br>"
            "Inspired by forza-painter (the_adawg), built on the techniques of "
            "geometrize-lib (Sam Twidale) and Primitive (Michael Fogleman). "
            "LiveryGroup discovery approach adapted from bvzrays/forza-painter-fh6.<br><br>"
            "If FH6 patches and injection breaks, the LiveryGroup offsets in "
            "<code>fd6/inject/fh6_injector.py</code> need to be re-derived for the new build."
        )

    def _refresh_inject_button(self) -> None:
        from fd6.inject.fh6_injector import patterns_are_populated
        ready = patterns_are_populated()
        self.settings_panel.inject_btn.setEnabled(ready)
        tip = (
            "Phase 2 feature. Requires FH6 memory patterns. See README §Phase 2."
            if not ready else
            "Inject the most recent shapes into a running FH6 vinyl group."
        )
        self.settings_panel.inject_btn.setToolTip(tip)

    def _on_files_selected(self, paths: list[Path]) -> None:
        if getattr(self, "_suite_mode", SuiteMode.FORZA) == SuiteMode.AC:
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
        queued_any = False
        for p in paths:
            queued_any = self.queue.add(p) or queued_any
        if queued_any and self._worker is None:
            self._start_next()

    def _refresh_ac_preview(self) -> None:
        """Rebuild the AC cycling-slot preview from the current source image
        and the settings-panel state. Called on upload and (in future) on
        settings changes. Cheap enough at default resolutions to run inline.
        """
        if not getattr(self, "_current_path", None):
            return
        try:
            from fd6.ac.slot_planner import plan_slots
            from fd6.ac.texture_pipeline import build_decal_texture
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
        from fd6.shapegen.worker import GenerationWorker
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
        self._worker = GenerationWorker(next_path, profile, sticker_mode=not add_white_bg)
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.backend_ready.connect(lambda label: self.statusBar().showMessage(f"Compute: {label}", 8000))
        self._worker.line_guide_ready.connect(self.settings_panel.set_line_guide_status)
        self._worker.line_guide_ready.connect(lambda message: self.statusBar().showMessage(message, 8000))
        self._worker.progress.connect(self.preview.on_progress)
        self._worker.preview.connect(self.preview.on_preview)
        self._worker.checkpoint_written.connect(lambda p: self.statusBar().showMessage(f"Checkpoint: {p}", 4000))
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._thread.start()
        self.settings_panel.set_running(True)
        self.statusBar().showMessage(f"Generating: {next_path.name}")

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
            self, "Pick shapes JSON to inject", "", "FD6 shapes (*.json);;All files (*)"
        )
        if json_path:
            self._on_inject_json_path(Path(json_path))

    def _on_json_loaded_for_preview(self, json_path: Path) -> None:
        """User clicked Upload JSON -> load the file, render shapes onto the preview pane.
        Does NOT inject. User must click Inject into FH6 after to actually push to game.
        """
        from fd6.io.exporter import load_json
        from fd6.shapegen.render import render_shapes
        try:
            doc = load_json(str(json_path))
            shapes = doc.materialize_shapes()
        except Exception as exc:
            QMessageBox.critical(self, "Load failed", f"{type(exc).__name__}: {exc}")
            return
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
        self.statusBar().showMessage(
            f"Preview ready. Click 'Inject into FH6' to push these shapes into the game.", 8000
        )

    def _on_inject_json_path(self, json_path: Path) -> None:
        """Inject the given FD6 shapes JSON into the running FH6 vinyl group.
        Opens a modal in-progress dialog (warns user not to touch FH6) and runs the
        injection in a background QThread. Status bar mirrors the same updates.
        """
        from fd6.inject import patterns_are_populated
        from fd6.gui.inject_worker import InjectionWorker
        from fd6.gui.inject_dialog import InjectionDialog

        if not patterns_are_populated():
            QMessageBox.warning(
                self, "FH6 Injection",
                "Patterns file is incomplete. Use FH6 → Discovery Workflow… to populate it."
            )
            return

        if getattr(self, "_inject_thread", None) is not None:
            QMessageBox.information(self, "Inject in progress", "An injection is already running. Wait for it to finish.")
            return

        target_key = self.settings_panel.selected_target_profile_key()
        self._inject_worker = InjectionWorker(json_path, profile_key=target_key)
        self._inject_thread = QThread(self)
        self._inject_worker.moveToThread(self._inject_thread)

        # Modal blocking dialog — labelled with the selected target so users
        # see FH4 / FH5 / FH3 instead of always reading "FH6".
        from fd6.inject.game_profiles import get_profile, default_profile
        try:
            game_label = get_profile(target_key).label
        except ValueError:
            game_label = default_profile().label
        self._inject_dialog = InjectionDialog(self, json_name=json_path.name, game_label=game_label)

        # Wire worker → both dialog and status bar
        self._inject_worker.scan_progress.connect(self._inject_dialog.on_scan_progress)
        self._inject_worker.write_progress.connect(self._inject_dialog.on_write_progress)
        self._inject_worker.status.connect(self._inject_dialog.on_status)
        self._inject_worker.done.connect(self._inject_dialog.on_done)

        self._inject_worker.scan_progress.connect(self._on_inject_scan_progress)
        self._inject_worker.write_progress.connect(self._on_inject_write_progress)
        self._inject_worker.status.connect(self._on_inject_status)
        self._inject_worker.done.connect(self._on_inject_done)

        self._inject_thread.started.connect(self._inject_worker.run)
        self._set_inject_status("Starting injection…", "info")
        self._inject_thread.start()
        # Show modal — blocks until close button enabled + user clicks Close
        self._inject_dialog.exec()

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
            from fd6.inject.game_profiles import get_profile
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
            self, "Save shapes JSON as…", self._last_finished_json.name, "FD6 shapes (*.json);;All files (*)"
        )
        if not dest:
            return
        try:
            shutil.copy2(str(self._last_finished_json), dest)
            self.statusBar().showMessage(f"Exported to {dest}", 6000)
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", f"{type(exc).__name__}: {exc}")

    def _show_fh6_status(self) -> None:
        from fd6.inject import discovery as disc
        from fd6.inject.patterns_io import load_patterns, has_usable_patterns

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
            "<p>Discovery is done from the command line, run from the FD6 project root:</p>"
            "<pre>"
            "python -m fd6.inject status\n"
            "python -m fd6.inject scan-float &lt;known sphere coord&gt;\n"
            "python -m fd6.inject narrow &lt;moved coord&gt;   (repeat until ~1 hit)\n"
            "python -m fd6.inject dump &lt;addr&gt; 256\n"
            "python -m fd6.inject find-refs &lt;struct_addr&gt;\n"
            "python -m fd6.inject save-pattern shape_array_ref '&lt;AOB&gt;' --offset 3\n"
            "python -m fd6.inject test-injector\n"
            "</pre>"
            "<p>The interactive parts (initial float discovery, struct field identification) "
            "are done with an external memory-scanning tool of your choice — FD6 only consumes "
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
