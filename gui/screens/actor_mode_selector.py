"""
Screen 1: ActorModeSelector — capture mode entry point.

Two-tile selector choosing between single-actor and dual-actor mode.
Includes language switcher (RU/EN) visible on first screen.

This is the first screen the user sees before starting a new recording session.
Mode selection is not final until recording starts, but locked during recording.
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QGraphicsDropShadowEffect,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont

from gui.i18n.locale_manager import LocaleManager


# Color constants — language-independent
COLOR_SELECTED_BG = "#1a73e8"
COLOR_UNSELECTED_BG = "#2d2d2d"
COLOR_UNSELECTED_BORDER = "#555555"
COLOR_TEXT_PRIMARY = "#ffffff"
COLOR_TEXT_SECONDARY = "#aaaaaa"
COLOR_BADGE_BETA = "#ff6b35"
COLOR_INFO_ICON = "#888888"
COLOR_LANG_ACTIVE = "#1a73e8"
COLOR_LANG_INACTIVE = "#666666"


class ActorModeSelector(QWidget):
    """Two-tile mode selector with language switcher.

    Signals:
        mode_selected(str): emitted when user clicks Start Recording.
            Payload is "single" or "dual".
        language_changed(str): emitted when user switches language.
    """

    mode_selected = Signal(str)
    language_changed = Signal(str)

    def __init__(self, locale: LocaleManager, parent=None):
        super().__init__(parent)
        self._locale = locale
        self._selected_mode = None  # None, "single", or "dual"
        self._setup_ui()
        self._connect_signals()
        self._refresh_text()

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(40, 30, 40, 30)
        root.setSpacing(20)

        # --- Top row: title + language switcher ---
        top_row = QHBoxLayout()
        top_row.setSpacing(12)

        self._title_label = QLabel()
        self._title_label.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        self._title_label.setStyleSheet(f"color: {COLOR_TEXT_PRIMARY};")

        self._lang_widget = self._build_lang_switcher()

        top_row.addWidget(self._title_label, 1)
        top_row.addLayout(self._lang_widget)
        root.addLayout(top_row)

        # --- Subtitle ---
        self._subtitle_label = QLabel()
        self._subtitle_label.setStyleSheet(f"color: {COLOR_TEXT_SECONDARY}; font-size: 14px;")
        root.addWidget(self._subtitle_label)

        root.addSpacing(10)

        # --- Tile container ---
        tiles_row = QHBoxLayout()
        tiles_row.setSpacing(20)

        self._single_tile = self._build_tile(
            key_prefix="single_actor",
            mode_id="single",
        )
        self._dual_tile = self._build_tile(
            key_prefix="dual_actor",
            mode_id="dual",
            show_badge=True,
            show_info=True,
        )

        tiles_row.addWidget(self._single_tile, 1)
        tiles_row.addWidget(self._dual_tile, 1)
        root.addLayout(tiles_row)

        root.addStretch(1)

        # --- Bottom: Start button ---
        self._start_btn = QPushButton()
        self._start_btn.setEnabled(False)
        self._start_btn.setMinimumHeight(48)
        self._start_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._start_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLOR_SELECTED_BG};
                color: {COLOR_TEXT_PRIMARY};
                border: none;
                border-radius: 6px;
                font-size: 16px;
                font-weight: bold;
                padding: 0 32px;
            }}
            QPushButton:hover {{
                background-color: #1557b0;
            }}
            QPushButton:disabled {{
                background-color: #444;
                color: #888;
            }}
        """)
        self._start_btn.clicked.connect(self._on_start_clicked)
        root.addWidget(self._start_btn)

    def _build_lang_switcher(self) -> QHBoxLayout:
        """Build RU / EN toggle buttons."""
        layout = QHBoxLayout()
        layout.setSpacing(4)

        self._lang_ru = QPushButton("RU")
        self._lang_en = QPushButton("EN")

        for btn in (self._lang_ru, self._lang_en):
            btn.setFixedSize(36, 28)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setStyleSheet(f"""
                QPushButton {{
                    border: 1px solid {COLOR_LANG_INACTIVE};
                    border-radius: 4px;
                    color: {COLOR_LANG_INACTIVE};
                    font-size: 12px;
                    font-weight: bold;
                    background: transparent;
                }}
            """)

        self._lang_ru.clicked.connect(lambda: self._switch_language("ru"))
        self._lang_en.clicked.connect(lambda: self._switch_language("en"))

        layout.addWidget(self._lang_ru)
        layout.addWidget(self._lang_en)
        return layout

    def _build_tile(self, key_prefix: str, mode_id: str,
                    show_badge: bool = False, show_info: bool = False) -> QFrame:
        """Build a mode selection tile."""
        tile = QFrame()
        tile.setObjectName(f"tile_{mode_id}")
        tile.setCursor(Qt.CursorShape.PointingHandCursor)
        tile.setMinimumHeight(200)
        tile.setStyleSheet(f"""
            QFrame {{
                background-color: {COLOR_UNSELECTED_BG};
                border: 2px solid {COLOR_UNSELECTED_BORDER};
                border-radius: 12px;
            }}
            QFrame:hover {{
                border-color: {COLOR_SELECTED_BG};
            }}
        """)

        outer = QVBoxLayout(tile)
        outer.setContentsMargins(20, 20, 20, 20)
        outer.setSpacing(8)

        # Title row (with optional badge)
        title_row = QHBoxLayout()
        title_row.setSpacing(8)

        title_lbl = QLabel()
        title_lbl.setObjectName(f"title_{mode_id}")
        title_lbl.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        title_lbl.setStyleSheet(f"color: {COLOR_TEXT_PRIMARY}; background: transparent;")
        title_row.addWidget(title_lbl)

        if show_badge:
            badge_lbl = QLabel("BETA")
            badge_lbl.setStyleSheet(f"""
                background-color: {COLOR_BADGE_BETA};
                color: white;
                border-radius: 4px;
                padding: 2px 8px;
                font-size: 11px;
                font-weight: bold;
            """)
            badge_lbl.setFixedHeight(22)
            title_row.addWidget(badge_lbl)

        title_row.addStretch()
        outer.addLayout(title_row)

        # Description
        desc_lbl = QLabel()
        desc_lbl.setObjectName(f"desc_{mode_id}")
        desc_lbl.setWordWrap(True)
        desc_lbl.setStyleSheet(f"color: {COLOR_TEXT_SECONDARY}; background: transparent; font-size: 13px;")
        outer.addWidget(desc_lbl)

        outer.addStretch()

        # Info icon (for dual-actor only)
        if show_info:
            info_row = QHBoxLayout()
            info_row.setSpacing(4)
            info_icon = QLabel("i")
            info_icon.setFixedSize(18, 18)
            info_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
            info_icon.setStyleSheet(f"""
                color: {COLOR_INFO_ICON};
                border: 1px solid {COLOR_INFO_ICON};
                border-radius: 9px;
                font-size: 11px;
                font-style: italic;
                background: transparent;
            """)
            info_lbl = QLabel()
            info_lbl.setObjectName(f"info_{mode_id}")
            info_lbl.setWordWrap(True)
            info_lbl.setStyleSheet(f"color: {COLOR_TEXT_SECONDARY}; background: transparent; font-size: 11px;")
            info_row.addWidget(info_icon)
            info_row.addWidget(info_lbl, 1)
            outer.addLayout(info_row)

        # Store references for click handling
        tile._mode_id = mode_id
        tile.mousePressEvent = lambda e, m=mode_id: self._on_tile_clicked(m)

        return tile

    def _connect_signals(self):
        self._locale.language_changed.connect(self._on_language_changed)

    def _on_language_changed(self, lang: str):
        self._refresh_text()
        self.language_changed.emit(lang)

    def _on_tile_clicked(self, mode_id: str):
        self._selected_mode = mode_id
        self._start_btn.setEnabled(True)
        self._update_tile_styles()

    def _on_start_clicked(self):
        if self._selected_mode:
            self.mode_selected.emit(self._selected_mode)

    def _switch_language(self, lang: str):
        self._locale.set_language(lang)

    def _update_tile_styles(self):
        for mode_id, tile in [("single", self._single_tile), ("dual", self._dual_tile)]:
            if mode_id == self._selected_mode:
                tile.setStyleSheet(f"""
                    QFrame {{
                        background-color: {COLOR_SELECTED_BG};
                        border: 2px solid {COLOR_SELECTED_BG};
                        border-radius: 12px;
                    }}
                """)
            else:
                tile.setStyleSheet(f"""
                    QFrame {{
                        background-color: {COLOR_UNSELECTED_BG};
                        border: 2px solid {COLOR_UNSELECTED_BORDER};
                        border-radius: 12px;
                    }}
                    QFrame:hover {{
                        border-color: {COLOR_SELECTED_BG};
                    }}
                """)

    def _refresh_text(self):
        t = self._locale.t
        self._title_label.setText(t("mode_selector_title"))
        self._subtitle_label.setText(t("mode_selector_subtitle"))

        # Single tile
        self._find_child(self._single_tile, "title_single").setText(t("single_actor_tile"))
        self._find_child(self._single_tile, "desc_single").setText(t("single_actor_desc"))

        # Dual tile
        self._find_child(self._dual_tile, "title_dual").setText(t("dual_actor_tile"))
        self._find_child(self._dual_tile, "desc_dual").setText(t("dual_actor_desc"))
        info_lbl = self._find_child(self._dual_tile, "info_dual")
        if info_lbl:
            info_lbl.setText(t("dual_actor_info"))

        # Start button
        self._start_btn.setText(t("btn_start"))

        # Language buttons highlight
        current_lang = self._locale.language
        for btn, lang in [(self._lang_ru, "ru"), (self._lang_en, "en")]:
            is_active = lang == current_lang
            border = COLOR_LANG_ACTIVE if is_active else COLOR_LANG_INACTIVE
            color = COLOR_LANG_ACTIVE if is_active else COLOR_LANG_INACTIVE
            btn.setStyleSheet(f"""
                QPushButton {{
                    border: 1px solid {border};
                    border-radius: 4px;
                    color: {color};
                    font-size: 12px;
                    font-weight: bold;
                    background: transparent;
                }}
            """)

    def _find_child(self, parent: QWidget, name: str) -> QLabel:
        """Find child QLabel by objectName."""
        for child in parent.findChildren(QLabel):
            if child.objectName() == name:
                return child
        return None
