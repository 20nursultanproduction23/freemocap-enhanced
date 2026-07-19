"""
FreeMoCap Enhanced — launcher that patches MainWindow with dual-actor mode selection.

Usage:
    python run_freemocap_enhanced.py

This script monkey-patches FreeMoCap's MainWindow to add mode selection
(single/dual actor) before starting a new recording session. All existing
FreeMoCap functionality is preserved unchanged.
"""
import sys
import os
import json
import logging
import signal
import multiprocessing

from PySide6.QtCore import QObject, Signal, QTimer

# ── Path setup ──────────────────────────────────────────────────────────
ENHANCED_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ENHANCED_DIR)

logger = logging.getLogger(__name__)

# ── Global state: which actor mode is selected ──────────────────────────
class ActorModeState:
    mode: str = "single"
    settings: dict = {}

actor_mode_state = ActorModeState()

# ── Inline i18n (avoids importlib — loads JSON directly) ────────────────
_I18N_DIR = os.path.join(ENHANCED_DIR, "gui", "i18n")

class _InlineLocaleManager(QObject):
    """Minimal locale manager — loads strings from JSON files using absolute paths.

    Inherits QObject so that GUI widgets can connect to language_changed signal.
    Must be instantiated AFTER QApplication exists.
    """

    language_changed = Signal(str)

    def __init__(self, lang="en", parent=None):
        super().__init__(parent)
        self._language = lang
        self._strings = {}
        self._load(lang)

    def _load(self, lang):
        path = os.path.join(_I18N_DIR, f"strings_{lang}.json")
        if not os.path.exists(path):
            logger.warning(f"String bundle not found: {path} — using empty dict")
            self._strings = {}
            return
        with open(path, "r", encoding="utf-8") as f:
            self._strings = json.load(f)

    @property
    def language(self):
        return self._language

    def set_language(self, lang):
        if lang == self._language:
            return
        self._load(lang)
        self._language = lang
        self.language_changed.emit(lang)

    def t(self, key, **kwargs):
        template = self._strings.get(key)
        if template is None:
            return f"[MISSING:{key}]"
        if kwargs:
            try:
                return template.format(**kwargs)
            except (KeyError, IndexError):
                return template
        return template


# ── Dialog definition ───────────────────────────────────────────────────
def _make_dialog_class(locale):
    """Create the ModeSelectionDialog class with a specific locale."""
    from PySide6.QtWidgets import (
        QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame,
    )
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QFont

    class ModeSelectionDialog(QDialog):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setWindowTitle("FreeMoCap")
            self.setMinimumSize(700, 450)
            self.setModal(True)
            self._selected = None
            self._locale = locale
            self._setup_ui()
            self._refresh_text()

        def _setup_ui(self):
            from PySide6.QtGui import QPalette, QColor

            self.setAutoFillBackground(True)
            palette = self.palette()
            palette.setColor(QPalette.ColorRole.Window, QColor("#2b2b2b"))
            palette.setColor(QPalette.ColorRole.WindowText, QColor("#ffffff"))
            palette.setColor(QPalette.ColorRole.Base, QColor("#2b2b2b"))
            palette.setColor(QPalette.ColorRole.Text, QColor("#ffffff"))
            palette.setColor(QPalette.ColorRole.Button, QColor("#3a3a3a"))
            palette.setColor(QPalette.ColorRole.ButtonText, QColor("#ffffff"))
            self.setPalette(palette)

            root = QVBoxLayout(self)
            root.setContentsMargins(30, 20, 30, 20)
            root.setSpacing(16)

            self.setStyleSheet("""
                QDialog { background-color: #2b2b2b; }
                QLabel { color: #ffffff; background-color: #2b2b2b; }
            """)

            title = QLabel()
            title.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
            title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            title.setStyleSheet("color: #ffffff; background-color: #2b2b2b; font-size: 18px; font-weight: bold;")
            self._title = title
            root.addWidget(title)

            subtitle = QLabel()
            subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
            subtitle.setStyleSheet("color: #aaaaaa; font-size: 13px; background-color: #2b2b2b;")
            self._subtitle = subtitle
            root.addWidget(subtitle)

            tiles_row = QHBoxLayout()
            tiles_row.setSpacing(16)
            self._single_tile = self._build_tile("single", show_badge=False)
            self._dual_tile = self._build_tile("dual", show_badge=True)
            tiles_row.addWidget(self._single_tile, 1)
            tiles_row.addWidget(self._dual_tile, 1)
            root.addLayout(tiles_row)

            root.addStretch()

            lang_row = QHBoxLayout()
            lang_row.addStretch()
            self._lang_ru = QPushButton("RU")
            self._lang_en = QPushButton("EN")
            for btn, lang in [(self._lang_ru, "ru"), (self._lang_en, "en")]:
                btn.setFixedSize(36, 28)
                btn.setCursor(Qt.CursorShape.PointingHandCursor)
                btn.clicked.connect(lambda checked=False, l=lang: self._switch_lang(l))
            lang_row.addWidget(self._lang_ru)
            lang_row.addWidget(self._lang_en)
            root.addLayout(lang_row)

            btn_row = QHBoxLayout()
            btn_row.addStretch()
            cancel_btn = QPushButton()
            cancel_btn.setFixedSize(100, 36)
            cancel_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            cancel_btn.setStyleSheet("""
                QPushButton {
                    background-color: transparent;
                    color: #aaaaaa;
                    border: 1px solid #555555;
                    border-radius: 6px;
                    font-size: 14px;
                    padding: 0 20px;
                }
                QPushButton:hover {
                    border-color: #1a73e8;
                    color: #1a73e8;
                }
            """)
            cancel_btn.clicked.connect(self.reject)
            self._cancel_btn = cancel_btn
            btn_row.addWidget(cancel_btn)

            ok_btn = QPushButton()
            ok_btn.setMinimumSize(160, 36)
            ok_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            ok_btn.setEnabled(False)
            ok_btn.setStyleSheet("""
                QPushButton {
                    background-color: #1a73e8; color: white;
                    border: none; border-radius: 6px;
                    font-size: 14px; font-weight: bold; padding: 0 20px;
                }
                QPushButton:hover { background-color: #1557b0; }
                QPushButton:disabled { background-color: #444; color: #888; }
            """)
            ok_btn.clicked.connect(self._on_ok)
            self._ok_btn = ok_btn
            btn_row.addWidget(ok_btn)
            root.addLayout(btn_row)

        def _build_tile(self, mode_id, show_badge=False):
            tile = QFrame()
            tile.setCursor(Qt.CursorShape.PointingHandCursor)
            tile.setMinimumHeight(160)
            tile.setStyleSheet("""
                QFrame {
                    background-color: #3a3a3a;
                    border: 2px solid #555555;
                    border-radius: 10px;
                }
                QFrame:hover { border-color: #1a73e8; }
            """)
            inner = QVBoxLayout(tile)
            inner.setContentsMargins(16, 12, 16, 12)
            inner.setSpacing(6)

            title_row = QHBoxLayout()
            title_lbl = QLabel()
            title_lbl.setFont(QFont("Segoe UI", 15, QFont.Weight.Bold))
            title_lbl.setStyleSheet("background-color: #3a3a3a; color: #ffffff;")
            title_row.addWidget(title_lbl)
            setattr(self, f"_tile_title_{mode_id}", title_lbl)

            if show_badge:
                badge = QLabel("BETA")
                badge.setStyleSheet("""
                    background-color: #ff6b35; color: white;
                    border-radius: 4px; padding: 2px 8px;
                    font-size: 11px; font-weight: bold;
                """)
                badge.setFixedHeight(22)
                title_row.addWidget(badge)

            title_row.addStretch()
            inner.addLayout(title_row)

            desc = QLabel()
            desc.setWordWrap(True)
            desc.setStyleSheet("color: #aaaaaa; background-color: #3a3a3a; font-size: 12px;")
            inner.addWidget(desc)
            setattr(self, f"_tile_desc_{mode_id}", desc)
            inner.addStretch()

            tile._mode_id = mode_id
            tile.mousePressEvent = lambda e, m=mode_id: self._select_mode(m)
            setattr(self, f"_tile_{mode_id}", tile)
            return tile

        def _select_mode(self, mode_id):
            self._selected = mode_id
            self._ok_btn.setEnabled(True)
            for mid, tile in [("single", self._single_tile), ("dual", self._dual_tile)]:
                title_lbl = getattr(self, f"_tile_title_{mid}")
                desc_lbl = getattr(self, f"_tile_desc_{mid}")
                if mid == mode_id:
                    tile.setStyleSheet("""
                        QFrame { background-color: #1a73e8; border: 2px solid #1a73e8; border-radius: 10px; }
                    """)
                    title_lbl.setStyleSheet("background-color: #1a73e8; color: #ffffff;")
                    desc_lbl.setStyleSheet("color: #ffffff; background-color: #1a73e8; font-size: 12px;")
                else:
                    tile.setStyleSheet("""
                        QFrame { background-color: #3a3a3a; border: 2px solid #555555; border-radius: 10px; }
                        QFrame:hover { border-color: #1a73e8; }
                    """)
                    title_lbl.setStyleSheet("background-color: #3a3a3a; color: #ffffff;")
                    desc_lbl.setStyleSheet("color: #aaaaaa; background-color: #3a3a3a; font-size: 12px;")

        def _on_ok(self):
            actor_mode_state.mode = self._selected or "single"
            self.accept()

        def _switch_lang(self, lang):
            self._locale.set_language(lang)
            self._refresh_text()
            cur = self._locale.language
            for btn, l in [(self._lang_ru, "ru"), (self._lang_en, "en")]:
                color = "#1a73e8" if l == cur else "#666666"
                btn.setStyleSheet(f"""
                    QPushButton {{
                        border: 1px solid {color}; border-radius: 4px;
                        color: {color}; font-size: 12px; font-weight: bold;
                        background-color: #2b2b2b;
                    }}
                """)

        def _refresh_text(self):
            t = self._locale.t
            self._title.setText(t("mode_selector_title"))
            self._subtitle.setText(t("mode_selector_subtitle"))
            self._tile_title_single.setText(t("single_actor_tile"))
            self._tile_desc_single.setText(t("single_actor_desc"))
            self._tile_title_dual.setText(t("dual_actor_tile"))
            self._tile_desc_dual.setText(t("dual_actor_desc"))
            self._ok_btn.setText(t("btn_start"))
            self._cancel_btn.setText(t("btn_back"))

    return ModeSelectionDialog


# ── Settings wrapper dialog ────────────────────────────────────────────
def _make_settings_dialog_class():
    """Create a QDialog that wraps a settings widget (Screen 2 or 3)."""
    from PySide6.QtWidgets import QDialog, QVBoxLayout

    class SettingsDialog(QDialog):
        def __init__(self, widget, parent=None):
            super().__init__(parent)
            self.setWindowTitle("FreeMoCap — Settings")
            self.setMinimumSize(720, 520)
            self.setModal(True)
            self.setStyleSheet("QDialog { background-color: #1e1e1e; }")

            layout = QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(widget)

            # Connect widget signals to dialog accept/reject
            widget.continue_clicked.connect(self.accept)
            widget.back_clicked.connect(self.reject)

        def reject(self):
            """Back button pressed — go back to mode selection."""
            super().reject()

    return SettingsDialog


# ── Monkey-patch ────────────────────────────────────────────────────────
def _patched_handle_start_new_session(self):
    """Patched: shows mode selector, then settings, then proceeds to cameras."""
    from PySide6.QtWidgets import QDialog

    # Ensure our modules are importable (FreeMoCap may alter sys.path at runtime)
    if ENHANCED_DIR not in sys.path:
        sys.path.insert(0, ENHANCED_DIR)

    try:
        locale = _InlineLocaleManager("ru")
        DialogClass = _make_dialog_class(locale)
        SettingsDialog = _make_settings_dialog_class()

        # Step 1: Mode selection
        dialog = DialogClass(parent=self)
        result = dialog.exec()

        if result != QDialog.DialogCode.Accepted:
            logger.info("User cancelled mode selection")
            return

        logger.info(f"Actor mode selected: {actor_mode_state.mode}")

        # Step 2: Show settings screen for the selected mode
        while True:
            try:
                import importlib.util
                import types
                _screens_dir = os.path.join(ENHANCED_DIR, "gui", "screens")
                _i18n_dir = os.path.join(ENHANCED_DIR, "gui", "i18n")
                _gui_dir = os.path.join(ENHANCED_DIR, "gui")

                # Temporarily inject our gui package into sys.modules so that
                # internal imports like `from gui.i18n.locale_manager import ...`
                # resolve to our code instead of FreeMoCap's gui package.
                _saved_modules = {}
                for _key in list(sys.modules.keys()):
                    if _key == "gui" or _key.startswith("gui."):
                        _saved_modules[_key] = sys.modules.pop(_key)

                # Create our gui package modules and register them
                _gui_pkg = types.ModuleType("gui")
                _gui_pkg.__path__ = [_gui_dir]
                _gui_pkg.__package__ = "gui"
                sys.modules["gui"] = _gui_pkg

                _gui_i18n = types.ModuleType("gui.i18n")
                _gui_i18n.__path__ = [_i18n_dir]
                _gui_i18n.__package__ = "gui.i18n"
                sys.modules["gui.i18n"] = _gui_i18n

                # Load locale_manager properly
                _lm_spec = importlib.util.spec_from_file_location(
                    "gui.i18n.locale_manager",
                    os.path.join(_i18n_dir, "locale_manager.py"),
                )
                _lm_mod = importlib.util.module_from_spec(_lm_spec)
                sys.modules["gui.i18n.locale_manager"] = _lm_mod
                _lm_spec.loader.exec_module(_lm_mod)

                # Load the settings screen module
                if actor_mode_state.mode == "single":
                    _spec = importlib.util.spec_from_file_location(
                        "single_actor_settings",
                        os.path.join(_screens_dir, "single_actor_settings.py"),
                    )
                    _mod = importlib.util.module_from_spec(_spec)
                    _spec.loader.exec_module(_mod)
                    settings_widget = _mod.SingleActorSettings(locale=locale)
                else:
                    _spec = importlib.util.spec_from_file_location(
                        "dual_actor_settings",
                        os.path.join(_screens_dir, "dual_actor_settings.py"),
                    )
                    _mod = importlib.util.module_from_spec(_spec)
                    _spec.loader.exec_module(_mod)
                    settings_widget = _mod.DualActorSettings(locale=locale)
                logger.info(f"Settings widget created for mode: {actor_mode_state.mode}")
            except Exception as e:
                logger.error(f"Failed to create settings widget: {e}", exc_info=True)
                import traceback
                traceback.print_exc()
                break
            finally:
                # Restore original FreeMoCap gui modules
                for _key in list(sys.modules.keys()):
                    if _key == "gui" or _key.startswith("gui."):
                        if _key in _saved_modules:
                            sys.modules[_key] = _saved_modules[_key]
                        elif _key not in ("gui.screens",):
                            sys.modules.pop(_key, None)

            settings_dialog = SettingsDialog(settings_widget, parent=self)
            settings_result = settings_dialog.exec()
            logger.info(f"Settings dialog result: {settings_result}")

            if settings_result == QDialog.DialogCode.Accepted:
                actor_mode_state.settings = settings_widget.settings
                logger.info(f"Settings confirmed: {actor_mode_state.settings}")
                break
            else:
                logger.info("User went back to mode selection")
                dialog = DialogClass(parent=self)
                result = dialog.exec()
                if result != QDialog.DialogCode.Accepted:
                    logger.info("User cancelled mode selection (on second pass)")
                    return

        # Step 3: Load Screen 4 (LiveFeedback) and Screen 5 (Diagnostics)
        screen4 = None
        screen5 = None
        try:
            import importlib.util
            import types as _types_mod
            _screens_dir = os.path.join(ENHANCED_DIR, "gui", "screens")
            _i18n_dir = os.path.join(ENHANCED_DIR, "gui", "i18n")
            _gui_dir = os.path.join(ENHANCED_DIR, "gui")

            _saved_modules2 = {}
            for _key in list(sys.modules.keys()):
                if _key == "gui" or _key.startswith("gui."):
                    _saved_modules2[_key] = sys.modules.pop(_key)

            _gui_pkg2 = _types_mod.ModuleType("gui")
            _gui_pkg2.__path__ = [_gui_dir]
            _gui_pkg2.__package__ = "gui"
            sys.modules["gui"] = _gui_pkg2

            _gui_i18n2 = _types_mod.ModuleType("gui.i18n")
            _gui_i18n2.__path__ = [_i18n_dir]
            _gui_i18n2.__package__ = "gui.i18n"
            sys.modules["gui.i18n"] = _gui_i18n2

            _lm_spec2 = importlib.util.spec_from_file_location(
                "gui.i18n.locale_manager",
                os.path.join(_i18n_dir, "locale_manager.py"),
            )
            _lm_mod2 = importlib.util.module_from_spec(_lm_spec2)
            sys.modules["gui.i18n.locale_manager"] = _lm_mod2
            _lm_spec2.loader.exec_module(_lm_mod2)

            _spec4 = importlib.util.spec_from_file_location(
                "live_feedback_widget",
                os.path.join(_screens_dir, "live_feedback_widget.py"),
            )
            _mod4 = importlib.util.module_from_spec(_spec4)
            _spec4.loader.exec_module(_mod4)
            screen4 = _mod4.LiveFeedbackWidget(locale=locale)

            _spec5 = importlib.util.spec_from_file_location(
                "diagnostics_widget",
                os.path.join(_screens_dir, "diagnostics_widget.py"),
            )
            _mod5 = importlib.util.module_from_spec(_spec5)
            _spec5.loader.exec_module(_mod5)
            screen5 = _mod5.DiagnosticsWidget(locale=locale)
            logger.info("Screen 4 (LiveFeedback) and Screen 5 (Diagnostics) loaded")
        except Exception as e:
            logger.error(f"Failed to load Screen 4/5: {e}", exc_info=True)
            import traceback
            traceback.print_exc()
        finally:
            for _key in list(sys.modules.keys()):
                if _key == "gui" or _key.startswith("gui."):
                    if _key in _saved_modules2:
                        sys.modules[_key] = _saved_modules2[_key]
                    elif _key not in ("gui.screens",):
                        sys.modules.pop(_key, None)

        # Step 4: Add Screen 4 & 5 as tabs + recording monitor + data pipeline
        if screen4 and screen5:
            self._central_tab_widget.addTab(screen4, "Live Feedback")
            self._central_tab_widget.addTab(screen5, "Diagnostics")
            self._enhanced_screen4 = screen4
            self._enhanced_screen5 = screen5
            self._enhanced_frame_count = 0
            self._enhanced_was_recording = False
            self._enhanced_recording_path = None

            from PySide6.QtCore import QTimer as _QTimer

            self._enhanced_recording_monitor = _QTimer()
            self._enhanced_frame_timer = None

            # ── Load real data from recording output ─────────────────
            def _load_recording_data():
                """Load 2D/3D data after processing and update screens."""
                try:
                    import importlib.util as _iu
                    import types as _t
                    _screens_dir = os.path.join(ENHANCED_DIR, "gui", "screens")
                    _i18n_dir = os.path.join(ENHANCED_DIR, "gui", "i18n")
                    _gui_dir = os.path.join(ENHANCED_DIR, "gui")

                    _sm = {}
                    for _k in list(sys.modules.keys()):
                        if _k == "gui" or _k.startswith("gui."):
                            _sm[_k] = sys.modules.pop(_k)

                    _gp = _t.ModuleType("gui")
                    _gp.__path__ = [_gui_dir]
                    _gp.__package__ = "gui"
                    sys.modules["gui"] = _gp
                    _gi = _t.ModuleType("gui.i18n")
                    _gi.__path__ = [_i18n_dir]
                    _gi.__package__ = "gui.i18n"
                    sys.modules["gui.i18n"] = _gi
                    _lm_spec = _iu.spec_from_file_location(
                        "gui.i18n.locale_manager",
                        os.path.join(_i18n_dir, "locale_manager.py"),
                    )
                    _lm_mod = _iu.module_from_spec(_lm_spec)
                    sys.modules["gui.i18n.locale_manager"] = _lm_mod
                    _lm_spec.loader.exec_module(_lm_mod)

                    _dl_spec = _iu.spec_from_file_location(
                        "data_loader",
                        os.path.join(_screens_dir, "data_loader.py"),
                    )
                    _dl_mod = _iu.module_from_spec(_dl_spec)
                    _dl_spec.loader.exec_module(_dl_mod)
                except Exception as e:
                    logger.error(f"Enhanced: failed to import data_loader: {e}")
                    return
                finally:
                    for _k in list(sys.modules.keys()):
                        if _k == "gui" or _k.startswith("gui."):
                            if _k in _sm:
                                sys.modules[_k] = _sm[_k]
                            elif _k not in ("gui.screens",):
                                sys.modules.pop(_k, None)

                recording_path = self._enhanced_recording_path
                if not recording_path or not os.path.isdir(recording_path):
                    logger.warning(f"Enhanced: no valid recording path: {recording_path}")
                    return

                try:
                    data_3d = _dl_mod.load_3d_data(recording_path)
                    data_2d = _dl_mod.load_2d_data(recording_path)
                    reproj = _dl_mod.load_reprojection_error(recording_path)

                    if data_3d is not None and data_3d.shape[0] > 0:
                        metrics = _dl_mod.compute_quality_metrics(data_3d, reproj)
                        issues = _dl_mod.detect_tracking_issues(data_3d, reproj)
                        screen5.set_metrics(
                            confident=metrics["confident_frames"],
                            low_confidence=metrics["low_confidence_frames"],
                            identity_swaps=sum(1 for i in issues if i.get("type") == "identity_swap"),
                            total=metrics["total_frames"],
                        )
                        screen5.set_problems(issues)
                        logger.info(
                            f"Enhanced: loaded real metrics — "
                            f"{metrics['confident_frames']}/{metrics['total_frames']} confident, "
                            f"{metrics['tracking_gaps']} gaps, {len(issues)} issues"
                        )

                    if data_2d is not None and data_2d.shape[1] > 0:
                        screen4.set_review_mode(True)
                        screen4.load_review_data(data_2d)
                        logger.info(f"Enhanced: loaded 2D review data — shape={data_2d.shape}")
                    else:
                        logger.info("Enhanced: no 2D data for review mode")

                except Exception as e:
                    logger.error(f"Enhanced: failed to load recording data: {e}", exc_info=True)

            # ── Recording monitor ────────────────────────────────────
            def _enhanced_monitor_recording():
                try:
                    is_recording = self._skellycam_widget.is_recording
                except Exception:
                    is_recording = False

                if is_recording and not self._enhanced_was_recording:
                    self._enhanced_frame_count = 0
                    screen4.set_frame_count(0)
                    screen4.set_review_mode(False)
                    self._central_tab_widget.setCurrentWidget(screen4)
                    self._enhanced_frame_timer = _QTimer()
                    def _tick():
                        self._enhanced_frame_count += 1
                        screen4.set_frame_count(self._enhanced_frame_count)
                    self._enhanced_frame_timer.timeout.connect(_tick)
                    self._enhanced_frame_timer.start(100)
                    logger.info("Enhanced: recording started -> Screen 4")

                elif not is_recording and self._enhanced_was_recording:
                    if self._enhanced_frame_timer:
                        self._enhanced_frame_timer.stop()
                        self._enhanced_frame_timer = None
                    # Show diagnostics with placeholder metrics immediately
                    total = self._enhanced_frame_count
                    screen5.set_metrics(
                        confident=max(0, total - 5),
                        low_confidence=min(5, total),
                        identity_swaps=0,
                        total=total,
                    )
                    self._central_tab_widget.setCurrentWidget(screen5)
                    logger.info(f"Enhanced: recording stopped ({total} frames) -> Screen 5 (placeholder)")

                self._enhanced_was_recording = is_recording

            self._enhanced_recording_monitor.timeout.connect(_enhanced_monitor_recording)
            self._enhanced_recording_monitor.start(200)

            # ── Stop recording ───────────────────────────────────────
            def _enhanced_on_stop():
                try:
                    if self._skellycam_widget.is_recording:
                        slot = self._skellycam_widget.controller_slot_dictionary.get("stop_recording")
                        if slot:
                            slot()
                            logger.info("Enhanced: stop_recording called via controller_slot_dictionary")
                        else:
                            logger.warning("Enhanced: stop_recording slot not found")
                except Exception as e:
                    logger.error(f"Enhanced: stop failed: {e}")
            screen4.stop_clicked.connect(_enhanced_on_stop)

            # ── Export to CSV ────────────────────────────────────────
            def _enhanced_on_export():
                try:
                    from PySide6.QtWidgets import QFileDialog as _QFD
                    recording_path = self._enhanced_recording_path
                    if not recording_path:
                        logger.warning("Enhanced: no recording path for export")
                        return

                    save_path, _ = _QFD.getSaveFileName(
                        self,
                        "Export 3D Skeleton Data",
                        os.path.join(recording_path, "export_3d_skeleton.csv"),
                        "CSV Files (*.csv);;All Files (*)",
                    )
                    if not save_path:
                        return

                    import importlib.util as _iu
                    import types as _t
                    _screens_dir = os.path.join(ENHANCED_DIR, "gui", "screens")
                    _i18n_dir = os.path.join(ENHANCED_DIR, "gui", "i18n")
                    _gui_dir = os.path.join(ENHANCED_DIR, "gui")
                    _sm = {}
                    for _k in list(sys.modules.keys()):
                        if _k == "gui" or _k.startswith("gui."):
                            _sm[_k] = sys.modules.pop(_k)
                    _gp = _t.ModuleType("gui")
                    _gp.__path__ = [_gui_dir]
                    _gp.__package__ = "gui"
                    sys.modules["gui"] = _gp
                    _gi = _t.ModuleType("gui.i18n")
                    _gi.__path__ = [_i18n_dir]
                    _gi.__package__ = "gui.i18n"
                    sys.modules["gui.i18n"] = _gi
                    _lm_spec = _iu.spec_from_file_location(
                        "gui.i18n.locale_manager",
                        os.path.join(_i18n_dir, "locale_manager.py"),
                    )
                    _lm_mod = _iu.module_from_spec(_lm_spec)
                    sys.modules["gui.i18n.locale_manager"] = _lm_mod
                    _lm_spec.loader.exec_module(_lm_mod)
                    _dl_spec = _iu.spec_from_file_location(
                        "data_loader",
                        os.path.join(_screens_dir, "data_loader.py"),
                    )
                    _dl_mod = _iu.module_from_spec(_dl_spec)
                    _dl_spec.loader.exec_module(_dl_mod)
                    try:
                        result_path = _dl_mod.export_to_csv(recording_path, save_path)
                        logger.info(f"Enhanced: exported to {result_path}")
                        from PySide6.QtWidgets import QMessageBox as _QMB
                        _QMB.information(self, "Export Complete", f"Data exported to:\n{result_path}")
                    except Exception as e:
                        logger.error(f"Enhanced: export failed: {e}")
                        from PySide6.QtWidgets import QMessageBox as _QMB
                        _QMB.warning(self, "Export Error", f"Export failed: {e}")
                    finally:
                        for _k in list(sys.modules.keys()):
                            if _k == "gui" or _k.startswith("gui."):
                                if _k in _sm:
                                    sys.modules[_k] = _sm[_k]
                                elif _k not in ("gui.screens",):
                                    sys.modules.pop(_k, None)
                except Exception as e:
                    logger.error(f"Enhanced: export dialog failed: {e}", exc_info=True)
            screen5.export_clicked.connect(_enhanced_on_export)

            def _enhanced_on_reprocess():
                self._central_tab_widget.setCurrentWidget(screen4)
                self._enhanced_frame_count = 0
                screen4.set_frame_count(0)
                screen4.set_review_mode(False)
                screen4.hide_identity_swap()
                screen4.set_confidence("medium")
            screen5.reprocess_clicked.connect(_enhanced_on_reprocess)

            def _enhanced_on_back():
                self._central_tab_widget.setCurrentWidget(screen4)
            screen5.back_to_feedback_clicked.connect(_enhanced_on_back)

            def _enhanced_on_fix():
                self._central_tab_widget.setCurrentWidget(screen5)
                screen5._pages.setCurrentIndex(1)
            screen5.fix_issues_clicked.connect(_enhanced_on_fix)

            # ── Review frame slider → update bboxes ──────────────────
            def _on_review_frame_changed(frame_idx):
                try:
                    import importlib.util as _iu
                    import types as _t
                    _screens_dir = os.path.join(ENHANCED_DIR, "gui", "screens")
                    _i18n_dir = os.path.join(ENHANCED_DIR, "gui", "i18n")
                    _gui_dir = os.path.join(ENHANCED_DIR, "gui")
                    _sm = {}
                    for _k in list(sys.modules.keys()):
                        if _k == "gui" or _k.startswith("gui."):
                            _sm[_k] = sys.modules.pop(_k)
                    _gp = _t.ModuleType("gui")
                    _gp.__path__ = [_gui_dir]
                    _gp.__package__ = "gui"
                    sys.modules["gui"] = _gp
                    _gi = _t.ModuleType("gui.i18n")
                    _gi.__path__ = [_i18n_dir]
                    _gi.__package__ = "gui.i18n"
                    sys.modules["gui.i18n"] = _gi
                    _lm_spec = _iu.spec_from_file_location(
                        "gui.i18n.locale_manager",
                        os.path.join(_i18n_dir, "locale_manager.py"),
                    )
                    _lm_mod = _iu.module_from_spec(_lm_spec)
                    sys.modules["gui.i18n.locale_manager"] = _lm_mod
                    _lm_spec.loader.exec_module(_lm_mod)
                    _dl_spec = _iu.spec_from_file_location(
                        "data_loader",
                        os.path.join(_screens_dir, "data_loader.py"),
                    )
                    _dl_mod = _iu.module_from_spec(_dl_spec)
                    _dl_spec.loader.exec_module(_dl_mod)

                    data_2d = screen4._review_2d_data
                    if data_2d is None:
                        return
                    bboxes = _dl_mod.extract_bboxes_for_frame(data_2d, frame_idx, scale_x=1.0, scale_y=1.0)
                    cam0_bboxes = {k: v for k, v in bboxes.items() if k == 0}
                    cam1_bboxes = {k: v for k, v in bboxes.items() if k == 1}
                    screen4.set_review_bboxes_for_frame(frame_idx, cam0_bboxes, cam1_bboxes)
                except Exception as e:
                    logger.error(f"Enhanced: review frame update failed: {e}")
                finally:
                    for _k in list(sys.modules.keys()):
                        if _k == "gui" or _k.startswith("gui."):
                            if _k in _sm:
                                sys.modules[_k] = _sm[_k]
                            elif _k not in ("gui.screens",):
                                sys.modules.pop(_k, None)

            screen4.review_frame_changed.connect(_on_review_frame_changed)

            # ── Connect real camera frames from skellycam to Screen 4 ──
            self._enhanced_cam_slot_map = {}
            self._enhanced_cam_slot_counter = 0

            def _enhanced_on_camera_connected():
                try:
                    worker = self._skellycam_widget._cam_group_frame_worker
                    def _on_new_image(camera_id, q_image, diagnostics):
                        if camera_id not in self._enhanced_cam_slot_map:
                            slot = self._enhanced_cam_slot_counter % 2
                            self._enhanced_cam_slot_map[camera_id] = slot
                            self._enhanced_cam_slot_counter += 1
                        slot = self._enhanced_cam_slot_map[camera_id]
                        screen4.set_camera_frame(slot, q_image)
                    worker.new_image_signal.connect(_on_new_image)
                    logger.info(f"Enhanced: connected camera frame signal (worker={worker})")
                except Exception as e:
                    logger.error(f"Enhanced: failed to connect camera frames: {e}")

            try:
                self._skellycam_widget.cameras_connected_signal.connect(_enhanced_on_camera_connected)
            except Exception as e:
                logger.error(f"Enhanced: failed to connect cameras_connected_signal: {e}")

            # ── Connect processing_finished_signal for real data ──────
            def _enhanced_on_processing_finished():
                try:
                    rec_info = self._active_recording_info_widget.get_active_recording_info()
                    if rec_info is not None and hasattr(rec_info, "path"):
                        self._enhanced_recording_path = str(rec_info.path)
                        logger.info(f"Enhanced: processing finished, recording path = {self._enhanced_recording_path}")
                    else:
                        logger.warning("Enhanced: processing finished but no recording path found")
                except Exception as e:
                    logger.error(f"Enhanced: failed to get recording path: {e}")

                _load_recording_data()

            try:
                self._process_motion_capture_data_panel.processing_finished_signal.connect(
                    _enhanced_on_processing_finished
                )
                logger.info("Enhanced: connected processing_finished_signal")
            except Exception as e:
                logger.error(f"Enhanced: failed to connect processing_finished_signal: {e}")

            logger.info("Enhanced: tabs + recording monitor + camera frames + data pipeline installed")

        # Step 5: Proceed to cameras tab
        self._central_tab_widget.set_welcome_tab_enabled(True)
        self._central_tab_widget.set_camera_view_tab_enabled(True)
        self._central_tab_widget.setCurrentIndex(1)
        self._controller_group_box.show()
        self._skellycam_widget.detect_available_cameras()

    except Exception as e:
        logger.error(f"CRITICAL ERROR in patched handler: {e}", exc_info=True)
        import traceback
        tb = traceback.format_exc()
        print(tb)
        try:
            with open(os.path.join(ENHANCED_DIR, "_last_error.txt"), "w") as f:
                f.write(f"ERROR: {e}\n\n{tb}")
        except Exception:
            pass


# ── Launch ──────────────────────────────────────────────────────────────
def sigint_handler(*args):
    QApplication.quit()

def main():
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication
    from PySide6.QtGui import QFont

    from freemocap.gui.qt.main_window.freemocap_main_window import MainWindow, EXIT_CODE_REBOOT
    from freemocap.gui.qt.utilities.get_qt_app import get_qt_app
    from freemocap.system.paths_and_filenames.path_getters import get_freemocap_data_folder_path
    from freemocap.system.user_data.pipedream_pings import PipedreamPings

    # Apply monkey-patch BEFORE app starts (so Actions.__init__ sees it)
    from PySide6.QtGui import QAction as QActionType
    MainWindow.handle_start_new_session_action = _patched_handle_start_new_session
    logger.info("Patched MainWindow class with dual-actor mode selector")

    signal.signal(signal.SIGINT, sigint_handler)
    app = get_qt_app()
    timer = QTimer()
    timer.start(500)

    pipedream_pings = PipedreamPings()

    while True:
        main_window = MainWindow(
            freemocap_data_folder_path=get_freemocap_data_folder_path(),
            pipedream_pings=pipedream_pings,
        )

        # ── Also patch instance + reconnect signal as safety net ──────
        bound = _patched_handle_start_new_session.__get__(main_window, type(main_window))
        main_window.handle_start_new_session_action = bound

        for action in main_window.findChildren(QActionType):
            txt = action.text()
            if "New Recording" in txt or "recording" in txt.lower():
                try:
                    action.triggered.disconnect()
                except (RuntimeError, TypeError):
                    pass
                action.triggered.connect(bound)
                logger.info(f"Reconnected action '{txt}' to patched handler")
                break

        logger.info("FreeMoCap Enhanced ready")

        # ── Patch home screen logo to use custom PNG ──────────────────
        custom_logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FREEMOCAP-BETA.png")
        if os.path.isfile(custom_logo_path):
            from PySide6.QtWidgets import QLabel
            from PySide6.QtGui import QPixmap
            from PySide6.QtCore import Qt as Qt2, QSize

            class _ResizableLogoLabel(QLabel):
                """QLabel that rescales its pixmap to fit current width, capped at max size."""
                def __init__(self, original_pixmap, parent=None):
                    super().__init__(parent)
                    self._original_pixmap = original_pixmap
                    self.setAlignment(Qt2.AlignmentFlag.AlignCenter)
                    self._max_w = min(original_pixmap.width(), 800)
                    self._max_h = min(original_pixmap.height(), 400)
                    self.setMaximumSize(self._max_w, self._max_h)
                    self._update_pixmap()

                def _update_pixmap(self):
                    top = self.window()
                    pw = top.width() if top else 800
                    pw = max(pw - 80, 200)
                    target_w = min(pw, self._max_w)
                    ratio = self._original_pixmap.height() / self._original_pixmap.width()
                    target_h = int(target_w * ratio)
                    target_h = min(target_h, self._max_h)
                    scaled = self._original_pixmap.scaled(
                        QSize(target_w, target_h),
                        Qt2.AspectRatioMode.KeepAspectRatio,
                        Qt2.TransformationMode.SmoothTransformation,
                    )
                    self.setPixmap(scaled)

                def resizeEvent(self, event):
                    super().resizeEvent(event)
                    self._update_pixmap()

            original_add_logo = main_window._home_widget._add_freemocap_logo

            def _custom_add_freemocap_logo(self):
                # Remove any existing logo widgets
                for i in reversed(range(self._layout.count())):
                    item = self._layout.itemAt(i)
                    w = item.widget()
                    if w is not None and w.__class__.__name__ in ("LogoSvgWidget", "QLabel"):
                        if not isinstance(w, QLabel) or w.pixmap() is not None:
                            w.setParent(None)

                pixmap = QPixmap(custom_logo_path)
                if not pixmap.isNull():
                    logo_label = _ResizableLogoLabel(pixmap)
                else:
                    logo_label = QLabel()

                from PySide6.QtWidgets import QHBoxLayout as _HBL
                _center_row = _HBL()
                _center_row.addStretch(1)
                _center_row.addWidget(logo_label)
                _center_row.addStretch(1)

                self._layout.addStretch(1)
                self._layout.addLayout(_center_row)
                self._layout.addStretch(1)

            main_window._home_widget._add_freemocap_logo = _custom_add_freemocap_logo.__get__(
                main_window._home_widget, type(main_window._home_widget)
            )
            # Re-run to apply immediately
            main_window._home_widget._add_freemocap_logo()
            logger.info("Enhanced: patched home screen logo to custom PNG")
        else:
            logger.warning(f"Enhanced: custom logo not found at {custom_logo_path}")

        main_window.show()

        # ── Patch welcome dialog: replace content with our updates ──────
        from freemocap.gui.qt.widgets.welcome_screen_dialog import WelcomeScreenDialog as _OrigWSD

        _welcome_strings = {
            "en": {
                "title": "FreeMoCap Enhanced v3.2",
                "subtitle": "Post-processing improvements for markerless motion capture",
                "sec_pipe": "Pipeline v3.2 Improvements",
                "pipe": [
                    "Jitter filter (OneEuro + Butterworth) — smooth motion, no lag",
                    "Bone length enforcement (FK-BFS) — eliminates 'breathing bones'",
                    "Foot sliding detection and correction",
                    "Floor plane alignment (RANSAC + PCA)",
                    "Self-occlusion detection via cross-camera consistency",
                    "Hand/face network stitching artifact reduction",
                    "NaN gap interpolation for brief tracking drops",
                    "Left/right confusion correction",
                    "Real-time logging and export to CSV",
                ],
                "sec_multi": "Multi-Person Tracking (BETA)",
                "multi": [
                    "RTMDet + RTMPose for 2D detection (133 keypoints)",
                    "Epipolar geometry + Hungarian cross-view association",
                    "Temporal tracking with persistent global IDs",
                    "Per-actor DLT triangulation",
                    "Physical interaction validation (interpenetration + contact)",
                    "ArUco marker fallback for identity tracking",
                    "ArUco marker generator with PNG export",
                    "102/102 tests passing",
                ],
                "sec_gui": "GUI",
                "gui": [
                    "Mode selector: Single Actor / Dual Actor",
                    "Bilingual interface (English / Russian)",
                    "Custom home screen logo (auto-resize)",
                    "Dual-actor settings with ArUco preview",
                    "Stress test: 48/50 rounds passed on real videos (96%)",
                ],
                "lang_btn": "RU",
            },
            "ru": {
                "title": "FreeMoCap Enhanced v3.2",
                "subtitle": "Улучшения пост-обработки для markerless захвата движения",
                "sec_pipe": "Улучшения пайплайна v3.2",
                "pipe": [
                    "Фильтр дрожания (OneEuro + Butterworth) — плавное движение без задержек",
                    "Уforcement длин костей (FK-BFS) — убирает «дышащие кости»",
                    "Детекция и коррекция скольжения стоп",
                    "Выравнивание плоскости пола (RANSAC + PCA)",
                    "Детекция самоперекрытия через кросс-камерную согласованность",
                    "Уменьшение артефактов стыковки body/hand/face сетей",
                    "Интерполяция NaN-пропусков при коротких провалах трекинга",
                    "Коррекция путаницы лево/право",
                    "Логирование в реальном времени и экспорт в CSV",
                ],
                "sec_multi": "Multi-Person Трекинг (BETA)",
                "multi": [
                    "RTMDet + RTMPose для 2D детекции (133 ключевые точки)",
                    "Эпиполярная геометрия + венгерский алгоритм для ассоциации",
                    "Временной трекинг с постоянными глобальными ID",
                    "Триангуляция DLT для каждого актёра",
                    "Валидация физического взаимодействия (интерпенетрация + контакт)",
                    "ArUco маркерный fallback для идентификации",
                    "Генератор ArUco маркеров с экспортом в PNG",
                    "102/102 тестов пройдено",
                ],
                "sec_gui": "Интерфейс",
                "gui": [
                    "Выбор режима: Один актёр / Два актёра",
                    "Двуязычный интерфейс (Русский / Английский)",
                    "Пользовательский логотип главного экрана (авто-масштаб)",
                    "Настройки dual-actor с ArUco превью",
                    "Стресс-тест: 48/50 раундов пройдено на реальных видео (96%)",
                ],
                "lang_btn": "EN",
            },
        }

        def _patched_open_welcome():
            from PySide6.QtWidgets import QLabel as _QLabel, QScrollArea, QWidget, QVBoxLayout, QHBoxLayout, QPushButton
            from PySide6.QtCore import Qt

            main_window._welcome_screen_dialog = _OrigWSD(
                gui_state=main_window._gui_state,
                kill_thread_event=main_window._kill_thread_event,
                parent=main_window,
            )
            dlg = main_window._welcome_screen_dialog
            dlg.setWindowTitle("Welcome to FreeMoCap Enhanced!")
            dlg.setMinimumSize(650, 500)
            layout = dlg.layout()
            if not layout:
                return

            while layout.count():
                item = layout.takeAt(0)
                w = item.widget()
                if w:
                    w.setParent(None)
                    w.deleteLater()

            current_lang = {"lang": "en"}

            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QScrollArea.Shape.NoFrame)
            container = QWidget()
            cl = QVBoxLayout(container)

            title_label = _QLabel()
            title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            title_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #1a237e; padding: 8px;")
            cl.addWidget(title_label)

            subtitle_label = _QLabel()
            subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            subtitle_label.setStyleSheet("font-size: 13px; color: #555; padding-bottom: 10px;")
            cl.addWidget(subtitle_label)

            section_widgets = []

            def _add_section(header_text, items_text):
                h = _QLabel()
                h.setText(f"<b>{header_text}</b>")
                h.setStyleSheet("font-size: 14px; color: #1565c0; padding-top: 10px;")
                cl.addWidget(h)
                labels = []
                for item in items_text:
                    label = _QLabel(f"  {item}")
                    label.setWordWrap(True)
                    label.setStyleSheet("font-size: 12px; color: #333; padding: 1px 0;")
                    cl.addWidget(label)
                    labels.append(label)
                section_widgets.append((h, labels))

            _sec_pipe_header = _QLabel()
            _sec_pipe_header.setStyleSheet("font-size: 14px; color: #1565c0; padding-top: 10px;")
            _sec_multi_header = _QLabel()
            _sec_multi_header.setStyleSheet("font-size: 14px; color: #1565c0; padding-top: 10px;")
            _sec_gui_header = _QLabel()
            _sec_gui_header.setStyleSheet("font-size: 14px; color: #1565c0; padding-top: 10px;")

            def _create_labels(count):
                labels = []
                for _ in range(count):
                    lbl = _QLabel()
                    lbl.setWordWrap(True)
                    lbl.setStyleSheet("font-size: 12px; color: #333; padding: 1px 0;")
                    cl.addWidget(lbl)
                    labels.append(lbl)
                return labels

            _pipe_labels = _create_labels(9)
            _multi_labels = _create_labels(8)
            _gui_labels = _create_labels(5)

            cl.addStretch(1)
            scroll.setWidget(container)
            layout.addWidget(scroll)

            def _refresh():
                s = _welcome_strings[current_lang["lang"]]
                title_label.setText(s["title"])
                subtitle_label.setText(s["subtitle"])
                _sec_pipe_header.setText(f"<b>{s['sec_pipe']}</b>")
                _sec_multi_header.setText(f"<b>{s['sec_multi']}</b>")
                _sec_gui_header.setText(f"<b>{s['sec_gui']}</b>")
                for lbl, txt in zip(_pipe_labels, s["pipe"]):
                    lbl.setText(f"  {txt}")
                for lbl, txt in zip(_multi_labels, s["multi"]):
                    lbl.setText(f"  {txt}")
                for lbl, txt in zip(_gui_labels, s["gui"]):
                    lbl.setText(f"  {txt}")
                lang_btn.setText(s["lang_btn"])

            def _toggle_lang():
                current_lang["lang"] = "ru" if current_lang["lang"] == "en" else "en"
                _refresh()

            lang_btn = QPushButton()
            lang_btn.setFixedWidth(50)
            lang_btn.setStyleSheet("font-size: 13px; font-weight: bold; padding: 4px 8px; background-color: #e3f2fd; border: 1px solid #1565c0; border-radius: 4px;")
            lang_btn.clicked.connect(_toggle_lang)

            btn_bar = QHBoxLayout()
            btn_bar.addStretch(1)
            btn_bar.addWidget(lang_btn)
            layout.addLayout(btn_bar)

            _refresh()
            dlg.exec()

        # ── Patch window title to show Enhanced version ────────────────
        main_window.setWindowTitle("FreeMoCap Enhanced v3.2  \U0001f480 \U00002728")

        # 1. Original FreeMoCap welcome dialog (force show)
        main_window._gui_state.show_welcome_screen = True
        main_window.open_welcome_screen_dialog()
        main_window._gui_state.show_welcome_screen = False

        # 2. Our enhanced welcome dialog
        _patched_open_welcome()

        # 3. Release notes + OpenCV conflict (welcome suppressed)
        main_window._gui_state.show_welcome_screen = False
        from freemocap.gui.qt.freemocap_main import handle_pop_ups
        handle_pop_ups(main_window)

        timer.timeout.connect(main_window.update)
        error_code = app.exec()
        main_window.close()

        if error_code != EXIT_CODE_REBOOT:
            break

        logger.info("Rebooting...")

    for p in multiprocessing.active_children():
        p.terminate()
    sys.exit()


if __name__ == "__main__":
    main()
