"""
Screen 2: SingleActorSettings — wrapper for single-actor recording.

Minimal screen: shows description + Continue button.
Does not modify existing FreeMoCap settings (stable 13/13 tests untouched).
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont

from gui.i18n.locale_manager import LocaleManager

COLOR_TEXT_PRIMARY = "#ffffff"
COLOR_TEXT_SECONDARY = "#aaaaaa"
COLOR_BG = "#1e1e1e"
COLOR_BTN_PRIMARY = "#1a73e8"
COLOR_BTN_PRIMARY_HOVER = "#1557b0"
COLOR_BTN_DISABLED = "#444444"


class SingleActorSettings(QWidget):
    """Single-actor settings wrapper.

    Signals:
        continue_clicked: emitted when user clicks Continue.
        back_clicked: emitted when user clicks Back.
    """

    continue_clicked = Signal()
    back_clicked = Signal()

    def __init__(self, locale: LocaleManager, parent=None):
        super().__init__(parent)
        self._locale = locale
        self._setup_ui()
        self._connect_signals()
        self._refresh_text()

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(40, 30, 40, 30)
        root.setSpacing(20)

        # --- Top row: back button + title ---
        top_row = QHBoxLayout()
        top_row.setSpacing(12)

        self._back_btn = QPushButton()
        self._back_btn.setFixedSize(36, 28)
        self._back_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._back_btn.setStyleSheet(f"""
            QPushButton {{
                border: 1px solid #555555;
                border-radius: 4px;
                color: #aaaaaa;
                font-size: 12px;
                background: transparent;
            }}
            QPushButton:hover {{
                border-color: {COLOR_BTN_PRIMARY};
                color: {COLOR_BTN_PRIMARY};
            }}
        """)
        self._back_btn.clicked.connect(self.back_clicked.emit)
        top_row.addWidget(self._back_btn)

        self._title_label = QLabel()
        self._title_label.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        self._title_label.setStyleSheet(f"color: {COLOR_TEXT_PRIMARY};")
        top_row.addWidget(self._title_label, 1)
        root.addLayout(top_row)

        # --- Description card ---
        card = QFrame()
        card.setStyleSheet(f"""
            QFrame {{
                background-color: #2d2d2d;
                border: 1px solid #444444;
                border-radius: 8px;
            }}
        """)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(24, 24, 24, 24)
        card_layout.setSpacing(12)

        self._desc_label = QLabel()
        self._desc_label.setWordWrap(True)
        self._desc_label.setStyleSheet(f"color: {COLOR_TEXT_SECONDARY}; font-size: 14px; background: transparent;")
        card_layout.addWidget(self._desc_label)

        root.addWidget(card)
        root.addStretch(1)

        # --- Bottom: Continue button ---
        self._continue_btn = QPushButton()
        self._continue_btn.setMinimumHeight(48)
        self._continue_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._continue_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLOR_BTN_PRIMARY};
                color: {COLOR_TEXT_PRIMARY};
                border: none;
                border-radius: 6px;
                font-size: 16px;
                font-weight: bold;
                padding: 0 32px;
            }}
            QPushButton:hover {{
                background-color: {COLOR_BTN_PRIMARY_HOVER};
            }}
        """)
        self._continue_btn.clicked.connect(self.continue_clicked.emit)
        root.addWidget(self._continue_btn)

    def _connect_signals(self):
        self._locale.language_changed.connect(self._refresh_text)

    def _refresh_text(self):
        t = self._locale.t
        self._title_label.setText(t("single_actor_settings_title"))
        self._desc_label.setText(t("single_actor_settings_desc"))
        self._back_btn.setText(t("btn_back"))
        self._continue_btn.setText(t("btn_continue"))
