"""
Screen 3: DualActorSettings — configuration for two-actor recording.

Features:
  - Actor count selector (2 now, 3 "Coming soon" disabled)
  - "Physical contact expected" checkbox with warning
  - Live indicator: cameras seeing both actors
  - Advanced settings (collapsed): association threshold, tracking threshold, shared floor
  - Restore defaults button
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame,
    QCheckBox, QSlider, QSizePolicy, QStackedWidget,
)
from PySide6.QtCore import Qt, Signal, Property
from PySide6.QtGui import QFont

from gui.i18n.locale_manager import LocaleManager

COLOR_TEXT_PRIMARY = "#ffffff"
COLOR_TEXT_SECONDARY = "#aaaaaa"
COLOR_TEXT_DISABLED = "#666666"
COLOR_BG = "#1e1e1e"
COLOR_CARD_BG = "#2d2d2d"
COLOR_BORDER = "#444444"
COLOR_BORDER_HOVER = "#1a73e8"
COLOR_BTN_PRIMARY = "#1a73e8"
COLOR_BTN_PRIMARY_HOVER = "#1557b0"
COLOR_BTN_SECONDARY = "#555555"
COLOR_ACCENT = "#1a73e8"
COLOR_WARNING = "#ff9800"
COLOR_CHECKBOX = "#1a73e8"


class DualActorSettings(QWidget):
    """Two-actor settings with actor count, contact, advanced thresholds.

    Signals:
        continue_clicked: emitted when user clicks Continue to Recording.
        back_clicked: emitted when user clicks Back.
        settings_changed(dict): emitted when any setting changes.
            Payload is dict with keys: actor_count, contact_expected,
            association_threshold, tracking_threshold, shared_floor.
    """

    continue_clicked = Signal()
    back_clicked = Signal()
    settings_changed = Signal(dict)

    # Default values
    DEFAULT_ASSOCIATION_THRESHOLD = 0.5  # 0.0 (looser) .. 1.0 (stricter)
    DEFAULT_TRACKING_THRESHOLD = 0.5
    DEFAULT_SHARED_FLOOR = True
    DEFAULT_CONTACT_EXPECTED = False
    DEFAULT_ACTOR_COUNT = 2
    TOTAL_CAMERAS = 6

    def __init__(self, locale: LocaleManager, parent=None):
        super().__init__(parent)
        self._locale = locale
        self._association_threshold = self.DEFAULT_ASSOCIATION_THRESHOLD
        self._tracking_threshold = self.DEFAULT_TRACKING_THRESHOLD
        self._shared_floor = self.DEFAULT_SHARED_FLOOR
        self._contact_expected = self.DEFAULT_CONTACT_EXPECTED
        self._actor_count = self.DEFAULT_ACTOR_COUNT
        self._cameras_seeing_both = 0
        self._advanced_visible = False
        self._setup_ui()
        self._connect_signals()
        self._refresh_text()

    # ---- Properties for settings ----
    @property
    def settings(self) -> dict:
        return {
            "actor_count": self._actor_count,
            "contact_expected": self._contact_expected,
            "association_threshold": self._association_threshold,
            "tracking_threshold": self._tracking_threshold,
            "shared_floor": self._shared_floor,
        }

    def set_cameras_seeing_both(self, count: int):
        """Update the live indicator for how many cameras see both actors."""
        self._cameras_seeing_both = count
        self._update_live_indicator()

    # ---- UI Setup ----
    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(40, 30, 40, 30)
        root.setSpacing(16)

        # --- Top row: back + title ---
        top_row = QHBoxLayout()
        top_row.setSpacing(12)

        self._back_btn = QPushButton()
        self._back_btn.setFixedSize(36, 28)
        self._back_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._back_btn.setStyleSheet(f"""
            QPushButton {{
                border: 1px solid {COLOR_BTN_SECONDARY};
                border-radius: 4px;
                color: {COLOR_TEXT_SECONDARY};
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

        # --- Actor count selector ---
        count_card = self._build_card()
        count_layout = QVBoxLayout(count_card)
        count_layout.setContentsMargins(20, 16, 20, 16)
        count_layout.setSpacing(8)

        self._count_label = QLabel()
        self._count_label.setStyleSheet(f"color: {COLOR_TEXT_PRIMARY}; font-size: 14px; font-weight: bold; background: transparent;")
        count_layout.addWidget(self._count_label)

        count_row = QHBoxLayout()
        count_row.setSpacing(8)

        self._count_2_btn = QPushButton("2")
        self._count_2_btn.setFixedSize(48, 36)
        self._count_2_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._count_3_btn = QPushButton("3")
        self._count_3_btn.setFixedSize(48, 36)
        self._count_3_btn.setEnabled(False)

        self._count_2_btn.clicked.connect(lambda: self._set_actor_count(2))

        count_row.addWidget(self._count_2_btn)
        count_row.addWidget(self._count_3_btn)

        # "Coming soon" label for count 3
        self._soon_label = QLabel()
        self._soon_label.setStyleSheet(f"color: {COLOR_TEXT_DISABLED}; font-size: 11px; background: transparent;")
        count_row.addWidget(self._soon_label)
        count_row.addStretch()

        count_layout.addLayout(count_row)
        root.addWidget(count_card)

        # --- Contact expected checkbox ---
        contact_card = self._build_card()
        contact_layout = QVBoxLayout(contact_card)
        contact_layout.setContentsMargins(20, 16, 20, 16)
        contact_layout.setSpacing(6)

        self._contact_cb = QCheckBox()
        self._contact_cb.setStyleSheet(f"""
            QCheckBox {{
                color: {COLOR_TEXT_PRIMARY};
                font-size: 14px;
                spacing: 8px;
                background: transparent;
            }}
            QCheckBox::indicator {{
                width: 18px;
                height: 18px;
                border: 2px solid {COLOR_BORDER};
                border-radius: 4px;
                background: transparent;
            }}
            QCheckBox::indicator:checked {{
                background-color: {COLOR_CHECKBOX};
                border-color: {COLOR_CHECKBOX};
            }}
        """)
        self._contact_cb.stateChanged.connect(self._on_contact_changed)
        contact_layout.addWidget(self._contact_cb)

        self._contact_desc = QLabel()
        self._contact_desc.setWordWrap(True)
        self._contact_desc.setStyleSheet(f"color: {COLOR_TEXT_SECONDARY}; font-size: 12px; background: transparent;")
        contact_layout.addWidget(self._contact_desc)

        # Warning label (hidden by default)
        self._contact_warning = QLabel()
        self._contact_warning.setWordWrap(True)
        self._contact_warning.setStyleSheet(f"""
            color: {COLOR_WARNING};
            font-size: 12px;
            background: transparent;
            padding: 8px;
            border: 1px solid {COLOR_WARNING};
            border-radius: 4px;
        """)
        self._contact_warning.setVisible(False)
        contact_layout.addWidget(self._contact_warning)

        root.addWidget(contact_card)

        # --- Live indicator ---
        live_card = self._build_card()
        live_layout = QVBoxLayout(live_card)
        live_layout.setContentsMargins(20, 16, 20, 16)
        live_layout.setSpacing(4)

        self._live_label = QLabel()
        self._live_label.setStyleSheet(f"color: {COLOR_TEXT_SECONDARY}; font-size: 13px; background: transparent;")
        live_layout.addWidget(self._live_label)

        self._live_bar = QFrame()
        self._live_bar.setFixedHeight(4)
        self._live_bar.setStyleSheet(f"background-color: {COLOR_BORDER}; border-radius: 2px;")
        live_layout.addWidget(self._live_bar)

        root.addWidget(live_card)

        # --- Advanced settings (collapsed by default) ---
        self._advanced_toggle = QPushButton()
        self._advanced_toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self._advanced_toggle.setStyleSheet(f"""
            QPushButton {{
                color: {COLOR_TEXT_SECONDARY};
                font-size: 13px;
                border: none;
                background: transparent;
                text-align: left;
                padding: 4px 0;
            }}
            QPushButton:hover {{
                color: {COLOR_BTN_PRIMARY};
            }}
        """)
        self._advanced_toggle.clicked.connect(self._toggle_advanced)
        root.addWidget(self._advanced_toggle)

        self._advanced_container = QWidget()
        self._advanced_container.setVisible(False)
        adv_layout = QVBoxLayout(self._advanced_container)
        adv_layout.setContentsMargins(0, 0, 0, 0)
        adv_layout.setSpacing(12)

        # Association threshold
        assoc_card = self._build_card()
        assoc_layout = QVBoxLayout(assoc_card)
        assoc_layout.setContentsMargins(20, 16, 20, 16)
        assoc_layout.setSpacing(6)

        self._assoc_label = QLabel()
        self._assoc_label.setStyleSheet(f"color: {COLOR_TEXT_PRIMARY}; font-size: 13px; font-weight: bold; background: transparent;")
        assoc_layout.addWidget(self._assoc_label)

        assoc_slider_row = QHBoxLayout()
        assoc_slider_row.setSpacing(8)

        self._assoc_stricter = QLabel()
        self._assoc_stricter.setStyleSheet(f"color: {COLOR_TEXT_SECONDARY}; font-size: 11px; background: transparent;")
        self._assoc_slider = QSlider(Qt.Orientation.Horizontal)
        self._assoc_slider.setRange(0, 100)
        self._assoc_slider.setValue(int(self._association_threshold * 100))
        self._assoc_slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                height: 4px;
                background: {COLOR_BORDER};
                border-radius: 2px;
            }}
            QSlider::handle:horizontal {{
                background: {COLOR_BTN_PRIMARY};
                width: 16px;
                height: 16px;
                margin: -6px 0;
                border-radius: 8px;
            }}
        """)
        self._assoc_slider.valueChanged.connect(self._on_assoc_slider_changed)
        self._assoc_looser = QLabel()
        self._assoc_looser.setStyleSheet(f"color: {COLOR_TEXT_SECONDARY}; font-size: 11px; background: transparent;")

        assoc_slider_row.addWidget(self._assoc_stricter)
        assoc_slider_row.addWidget(self._assoc_slider, 1)
        assoc_slider_row.addWidget(self._assoc_looser)
        assoc_layout.addLayout(assoc_slider_row)

        self._assoc_desc = QLabel()
        self._assoc_desc.setWordWrap(True)
        self._assoc_desc.setStyleSheet(f"color: {COLOR_TEXT_SECONDARY}; font-size: 11px; background: transparent;")
        assoc_layout.addWidget(self._assoc_desc)

        adv_layout.addWidget(assoc_card)

        # Tracking threshold
        track_card = self._build_card()
        track_layout = QVBoxLayout(track_card)
        track_layout.setContentsMargins(20, 16, 20, 16)
        track_layout.setSpacing(6)

        self._track_label = QLabel()
        self._track_label.setStyleSheet(f"color: {COLOR_TEXT_PRIMARY}; font-size: 13px; font-weight: bold; background: transparent;")
        track_layout.addWidget(self._track_label)

        track_slider_row = QHBoxLayout()
        track_slider_row.setSpacing(8)

        self._track_stricter = QLabel()
        self._track_stricter.setStyleSheet(f"color: {COLOR_TEXT_SECONDARY}; font-size: 11px; background: transparent;")
        self._track_slider = QSlider(Qt.Orientation.Horizontal)
        self._track_slider.setRange(0, 100)
        self._track_slider.setValue(int(self._tracking_threshold * 100))
        self._track_slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                height: 4px;
                background: {COLOR_BORDER};
                border-radius: 2px;
            }}
            QSlider::handle:horizontal {{
                background: {COLOR_BTN_PRIMARY};
                width: 16px;
                height: 16px;
                margin: -6px 0;
                border-radius: 8px;
            }}
        """)
        self._track_slider.valueChanged.connect(self._on_track_slider_changed)
        self._track_looser = QLabel()
        self._track_looser.setStyleSheet(f"color: {COLOR_TEXT_SECONDARY}; font-size: 11px; background: transparent;")

        track_slider_row.addWidget(self._track_stricter)
        track_slider_row.addWidget(self._track_slider, 1)
        track_slider_row.addWidget(self._track_looser)
        track_layout.addLayout(track_slider_row)

        self._track_desc = QLabel()
        self._track_desc.setWordWrap(True)
        self._track_desc.setStyleSheet(f"color: {COLOR_TEXT_SECONDARY}; font-size: 11px; background: transparent;")
        track_layout.addWidget(self._track_desc)

        adv_layout.addWidget(track_card)

        # Shared floor checkbox
        floor_card = self._build_card()
        floor_layout = QVBoxLayout(floor_card)
        floor_layout.setContentsMargins(20, 16, 20, 16)
        floor_layout.setSpacing(4)

        self._floor_cb = QCheckBox()
        self._floor_cb.setChecked(self._shared_floor)
        self._floor_cb.setStyleSheet(f"""
            QCheckBox {{
                color: {COLOR_TEXT_PRIMARY};
                font-size: 13px;
                spacing: 8px;
                background: transparent;
            }}
            QCheckBox::indicator {{
                width: 18px;
                height: 18px;
                border: 2px solid {COLOR_BORDER};
                border-radius: 4px;
                background: transparent;
            }}
            QCheckBox::indicator:checked {{
                background-color: {COLOR_CHECKBOX};
                border-color: {COLOR_CHECKBOX};
            }}
        """)
        self._floor_cb.stateChanged.connect(self._on_floor_changed)
        floor_layout.addWidget(self._floor_cb)

        self._floor_desc = QLabel()
        self._floor_desc.setWordWrap(True)
        self._floor_desc.setStyleSheet(f"color: {COLOR_TEXT_SECONDARY}; font-size: 11px; background: transparent;")
        floor_layout.addWidget(self._floor_desc)

        adv_layout.addWidget(floor_card)

        root.addWidget(self._advanced_container)

        # --- Restore defaults ---
        self._restore_btn = QPushButton()
        self._restore_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._restore_btn.setStyleSheet(f"""
            QPushButton {{
                color: {COLOR_TEXT_SECONDARY};
                font-size: 12px;
                border: 1px solid {COLOR_BORDER};
                border-radius: 4px;
                padding: 6px 16px;
                background: transparent;
            }}
            QPushButton:hover {{
                border-color: {COLOR_BTN_PRIMARY};
                color: {COLOR_BTN_PRIMARY};
            }}
        """)
        self._restore_btn.clicked.connect(self._restore_defaults)
        root.addWidget(self._restore_btn, alignment=Qt.AlignmentFlag.AlignLeft)

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

    # ---- Helper ----
    def _build_card(self) -> QFrame:
        card = QFrame()
        card.setStyleSheet(f"""
            QFrame {{
                background-color: {COLOR_CARD_BG};
                border: 1px solid {COLOR_BORDER};
                border-radius: 8px;
            }}
        """)
        return card

    # ---- Signals ----
    def _connect_signals(self):
        self._locale.language_changed.connect(self._refresh_text)

    # ---- Event handlers ----
    def _set_actor_count(self, count: int):
        self._actor_count = count
        self._update_count_style()
        self.settings_changed.emit(self.settings)

    def _on_contact_changed(self, state: int):
        self._contact_expected = state == Qt.CheckState.Checked.value
        self._contact_warning.setVisible(self._contact_expected)
        self.settings_changed.emit(self.settings)

    def _on_assoc_slider_changed(self, value: int):
        self._association_threshold = value / 100.0
        self.settings_changed.emit(self.settings)

    def _on_track_slider_changed(self, value: int):
        self._tracking_threshold = value / 100.0
        self.settings_changed.emit(self.settings)

    def _on_floor_changed(self, state: int):
        self._shared_floor = state == Qt.CheckState.Checked.value
        self.settings_changed.emit(self.settings)

    def _toggle_advanced(self):
        self._advanced_visible = not self._advanced_visible
        self._advanced_container.setVisible(self._advanced_visible)
        self._refresh_text()

    def _restore_defaults(self):
        self._association_threshold = self.DEFAULT_ASSOCIATION_THRESHOLD
        self._tracking_threshold = self.DEFAULT_TRACKING_THRESHOLD
        self._shared_floor = self.DEFAULT_SHARED_FLOOR
        self._contact_expected = self.DEFAULT_CONTACT_EXPECTED
        self._actor_count = self.DEFAULT_ACTOR_COUNT

        self._assoc_slider.setValue(int(self._association_threshold * 100))
        self._track_slider.setValue(int(self._tracking_threshold * 100))
        self._floor_cb.setChecked(self._shared_floor)
        self._contact_cb.setChecked(self._contact_expected)
        self._contact_warning.setVisible(False)
        self._update_count_style()
        self.settings_changed.emit(self.settings)

    # ---- UI Updates ----
    def _update_count_style(self):
        for count, btn in [(2, self._count_2_btn), (3, self._count_3_btn)]:
            if count == self._actor_count:
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: {COLOR_BTN_PRIMARY};
                        color: {COLOR_TEXT_PRIMARY};
                        border: none;
                        border-radius: 6px;
                        font-size: 16px;
                        font-weight: bold;
                    }}
                """)
            else:
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: transparent;
                        color: {COLOR_TEXT_SECONDARY};
                        border: 1px solid {COLOR_BORDER};
                        border-radius: 6px;
                        font-size: 16px;
                        font-weight: bold;
                    }}
                    QPushButton:disabled {{
                        color: {COLOR_TEXT_DISABLED};
                        border-color: {COLOR_TEXT_DISABLED};
                    }}
                """)

    def _update_live_indicator(self):
        t = self._locale.t
        self._live_label.setText(t("live_indicator_label",
                                  seen=self._cameras_seeing_both,
                                  total=self.TOTAL_CAMERAS))
        # Update bar color
        ratio = self._cameras_seeing_both / self.TOTAL_CAMERAS
        if ratio >= 0.67:
            bar_color = "#4caf50"  # green
        elif ratio >= 0.33:
            bar_color = COLOR_WARNING
        else:
            bar_color = "#f44336"  # red
        self._live_bar.setStyleSheet(f"background-color: {bar_color}; border-radius: 2px;")

    def _refresh_text(self):
        t = self._locale.t
        self._title_label.setText(t("dual_actor_settings_title"))
        self._back_btn.setText(t("btn_back"))
        self._count_label.setText(t("actor_count_label"))
        self._soon_label.setText(t("actor_count_soon"))
        self._contact_cb.setText(t("contact_expected_label"))
        self._contact_desc.setText(t("contact_expected_desc"))
        self._contact_warning.setText(t("contact_expected_warning"))
        self._update_live_indicator()

        # Advanced toggle
        toggle_key = "advanced_settings_hide" if self._advanced_visible else "advanced_settings_show"
        self._advanced_toggle.setText(t(toggle_key))

        # Association threshold
        self._assoc_label.setText(t("association_threshold_label"))
        self._assoc_stricter.setText(t("association_threshold_stricter"))
        self._assoc_looser.setText(t("association_threshold_looser"))
        self._assoc_desc.setText(t("association_threshold_desc"))

        # Tracking threshold
        self._track_label.setText(t("tracking_threshold_label"))
        self._track_stricter.setText(t("tracking_threshold_stricter"))
        self._track_looser.setText(t("tracking_threshold_looser"))
        self._track_desc.setText(t("tracking_threshold_desc"))

        # Shared floor
        self._floor_cb.setText(t("shared_floor_label"))
        self._floor_desc.setText(t("shared_floor_desc"))

        self._restore_btn.setText(t("btn_restore_defaults"))
        self._continue_btn.setText(t("btn_record"))

        self._update_count_style()
