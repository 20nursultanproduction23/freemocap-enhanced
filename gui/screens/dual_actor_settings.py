"""
Screen 3: DualActorSettings — configuration for two-actor recording.

Features:
  - Actor count selector (2 now, 3 "Coming soon" disabled)
  - "Physical contact expected" checkbox with warning
  - "Use ArUco markers" toggle with marker generation section
  - Live indicator: cameras seeing both actors
  - Advanced settings (collapsed): association threshold, tracking threshold, shared floor
  - Restore defaults button
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame,
    QCheckBox, QSlider, QSizePolicy, QStackedWidget, QScrollArea,
    QFileDialog, QSizePolicy,
)
from PySide6.QtCore import Qt, Signal, Property
from PySide6.QtGui import QFont, QPixmap, QPainter, QPdfWriter, QPageLayout
from PySide6.QtPrintSupport import QPrintDialog, QPrinter

from gui.i18n.locale_manager import LocaleManager

import os

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
    DEFAULT_ARUCO_FALLBACK = False

    def __init__(self, locale: LocaleManager, parent=None):
        super().__init__(parent)
        self._locale = locale
        self._association_threshold = self.DEFAULT_ASSOCIATION_THRESHOLD
        self._tracking_threshold = self.DEFAULT_TRACKING_THRESHOLD
        self._shared_floor = self.DEFAULT_SHARED_FLOOR
        self._contact_expected = self.DEFAULT_CONTACT_EXPECTED
        self._actor_count = self.DEFAULT_ACTOR_COUNT
        self._aruco_fallback_enabled = self.DEFAULT_ARUCO_FALLBACK

        self._marker_pixmaps = {}  # actor_name -> QPixmap for preview
        self._marker_paths = {}    # actor_name -> filepath after generation

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
            "aruco_fallback_enabled": self._aruco_fallback_enabled,
        }

    # ---- UI Setup ----
    def _setup_ui(self):
        self.setStyleSheet(f"background-color: {COLOR_BG};")
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # --- Top row: back + title (fixed, outside scroll) ---
        top_row = QHBoxLayout()
        top_row.setContentsMargins(40, 30, 40, 10)
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

        # --- Scrollable content area ---
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setStyleSheet(f"QScrollArea {{ background-color: {COLOR_BG}; border: none; }}")

        scroll_content = QWidget()
        scroll_content.setStyleSheet(f"background-color: {COLOR_BG};")
        content_layout = QVBoxLayout(scroll_content)
        content_layout.setContentsMargins(40, 10, 40, 10)
        content_layout.setSpacing(16)

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

        self._soon_label = QLabel()
        self._soon_label.setStyleSheet(f"color: {COLOR_TEXT_DISABLED}; font-size: 11px; background: transparent;")
        count_row.addWidget(self._soon_label)
        count_row.addStretch()

        count_layout.addLayout(count_row)
        content_layout.addWidget(count_card)

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

        content_layout.addWidget(contact_card)

        # --- ArUco marker fallback toggle ---
        aruco_card = self._build_card()
        aruco_layout = QVBoxLayout(aruco_card)
        aruco_layout.setContentsMargins(20, 16, 20, 16)
        aruco_layout.setSpacing(6)

        self._aruco_cb = QCheckBox()
        self._aruco_cb.setStyleSheet(f"""
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
        self._aruco_cb.stateChanged.connect(self._on_aruco_changed)
        aruco_layout.addWidget(self._aruco_cb)

        self._aruco_desc = QLabel()
        self._aruco_desc.setWordWrap(True)
        self._aruco_desc.setStyleSheet(f"color: {COLOR_TEXT_SECONDARY}; font-size: 12px; background: transparent;")
        aruco_layout.addWidget(self._aruco_desc)

        # --- Marker generation section (hidden by default) ---
        self._marker_container = QWidget()
        self._marker_container.setVisible(False)
        marker_layout = QVBoxLayout(self._marker_container)
        marker_layout.setContentsMargins(0, 8, 0, 0)
        marker_layout.setSpacing(10)

        # Generate button
        self._generate_btn = QPushButton()
        self._generate_btn.setMinimumHeight(40)
        self._generate_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._generate_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLOR_BTN_SECONDARY};
                color: {COLOR_TEXT_PRIMARY};
                border: 1px solid {COLOR_BORDER};
                border-radius: 6px;
                font-size: 13px;
                padding: 0 20px;
            }}
            QPushButton:hover {{
                border-color: {COLOR_BTN_PRIMARY};
                color: {COLOR_BTN_PRIMARY};
            }}
        """)
        self._generate_btn.clicked.connect(self._on_generate_markers)
        marker_layout.addWidget(self._generate_btn)

        # Preview area: two labels side by side
        preview_row = QHBoxLayout()
        preview_row.setSpacing(16)

        self._preview_actor0 = self._build_preview_label("actor_0")
        self._preview_actor1 = self._build_preview_label("actor_1")
        preview_row.addWidget(self._preview_actor0["container"], 1)
        preview_row.addWidget(self._preview_actor1["container"], 1)
        marker_layout.addLayout(preview_row)

        # Download + Print buttons
        action_row = QHBoxLayout()
        action_row.setSpacing(10)

        self._download_btn = QPushButton()
        self._download_btn.setMinimumHeight(36)
        self._download_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._download_btn.setEnabled(False)
        self._download_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {COLOR_TEXT_SECONDARY};
                border: 1px solid {COLOR_BORDER};
                border-radius: 6px;
                font-size: 12px;
                padding: 0 16px;
            }}
            QPushButton:hover {{
                border-color: {COLOR_BTN_PRIMARY};
                color: {COLOR_BTN_PRIMARY};
            }}
            QPushButton:disabled {{
                color: {COLOR_TEXT_DISABLED};
                border-color: {COLOR_TEXT_DISABLED};
            }}
        """)
        self._download_btn.clicked.connect(self._on_download_markers)
        action_row.addWidget(self._download_btn)

        self._print_btn = QPushButton()
        self._print_btn.setMinimumHeight(36)
        self._print_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._print_btn.setEnabled(False)
        self._print_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {COLOR_TEXT_SECONDARY};
                border: 1px solid {COLOR_BORDER};
                border-radius: 6px;
                font-size: 12px;
                padding: 0 16px;
            }}
            QPushButton:hover {{
                border-color: {COLOR_BTN_PRIMARY};
                color: {COLOR_BTN_PRIMARY};
            }}
            QPushButton:disabled {{
                color: {COLOR_TEXT_DISABLED};
                border-color: {COLOR_TEXT_DISABLED};
            }}
        """)
        self._print_btn.clicked.connect(self._on_print_markers)
        action_row.addWidget(self._print_btn)

        marker_layout.addLayout(action_row)

        self._download_notification = QLabel()
        self._download_notification.setStyleSheet(f"color: #4caf50; font-size: 12px; background: transparent;")
        self._download_notification.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._download_notification.setVisible(False)
        marker_layout.addWidget(self._download_notification)

        aruco_layout.addWidget(self._marker_container)
        content_layout.addWidget(aruco_card)

        # --- Advanced settings toggle ---
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
        content_layout.addWidget(self._advanced_toggle)

        # --- Advanced settings container (collapsed) ---
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

        content_layout.addWidget(self._advanced_container)

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
        content_layout.addWidget(self._restore_btn, alignment=Qt.AlignmentFlag.AlignLeft)

        content_layout.addStretch(1)

        scroll_area.setWidget(scroll_content)
        root.addWidget(scroll_area, 1)

        # --- Bottom: Continue button (fixed, outside scroll) ---
        btn_bar = QWidget()
        btn_bar.setStyleSheet("background: transparent;")
        btn_layout = QHBoxLayout(btn_bar)
        btn_layout.setContentsMargins(40, 10, 40, 20)

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
        btn_layout.addWidget(self._continue_btn)

        root.addWidget(btn_bar)

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

    def _build_preview_label(self, actor_name: str) -> dict:
        """Build a preview widget for one actor's marker."""
        container = QFrame()
        container.setStyleSheet(f"""
            QFrame {{
                background-color: {COLOR_CARD_BG};
                border: 1px solid {COLOR_BORDER};
                border-radius: 6px;
            }}
        """)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        image_label = QLabel()
        image_label.setFixedSize(180, 180)
        image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        image_label.setStyleSheet(f"background: {COLOR_BG}; border-radius: 4px;")
        image_label.setText("---")
        layout.addWidget(image_label, alignment=Qt.AlignmentFlag.AlignCenter)

        info_label = QLabel()
        info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info_label.setStyleSheet(f"color: {COLOR_TEXT_SECONDARY}; font-size: 11px; background: transparent;")
        info_label.setText(actor_name)
        layout.addWidget(info_label)

        return {"container": container, "image": image_label, "info": info_label}

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
        self._aruco_fallback_enabled = self.DEFAULT_ARUCO_FALLBACK

        self._assoc_slider.setValue(int(self._association_threshold * 100))
        self._track_slider.setValue(int(self._tracking_threshold * 100))
        self._floor_cb.setChecked(self._shared_floor)
        self._contact_cb.setChecked(self._contact_expected)
        self._contact_warning.setVisible(False)
        self._aruco_cb.setChecked(self._aruco_fallback_enabled)
        self._marker_container.setVisible(False)
        self._update_count_style()
        self.settings_changed.emit(self.settings)

    # ---- ArUco marker handlers ----
    def _on_aruco_changed(self, state: int):
        self._aruco_fallback_enabled = state == Qt.CheckState.Checked.value
        self._marker_container.setVisible(self._aruco_fallback_enabled)
        self.settings_changed.emit(self.settings)

    def _on_generate_markers(self):
        """Generate ArUco markers and show previews."""
        try:
            from tools.generate_actor_markers import generate_all_markers, load_config

            config_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "tools", "actor_marker_map.yaml"
            )
            config = load_config(config_path)

            output_dir = os.path.join(os.path.dirname(__file__), "..", "..", "markers")
            generated = generate_all_markers(config, output_dir)

            for actor_name, info in generated.items():
                filepath = info["filepath"]
                self._marker_paths[actor_name] = filepath

                pixmap = QPixmap(filepath)
                if not pixmap.isNull():
                    scaled = pixmap.scaled(
                        160, 160,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )

                    widget_info = self._preview_actor0 if actor_name == "actor_0" else self._preview_actor1
                    widget_info["image"].setPixmap(scaled)
                    widget_info["image"].setText("")
                    widget_info["info"].setText(
                        f"{actor_name}\nID: {info['marker_id']}  |  {info['size_mm']}mm"
                    )

            self._download_btn.setEnabled(True)
            self._print_btn.setEnabled(True)

        except Exception as e:
            pass

    def _on_download_markers(self):
        """Save marker PNGs to user's Downloads folder."""
        try:
            from PySide6.QtCore import QTimer
            downloads = os.path.join(os.path.expanduser("~"), "Downloads")
            for actor_name, filepath in self._marker_paths.items():
                if os.path.exists(filepath):
                    dest = os.path.join(downloads, os.path.basename(filepath))
                    import shutil
                    shutil.copy2(filepath, dest)
            self._download_notification.setText(self._locale.t("aruco_download_saved"))
            self._download_notification.setVisible(True)
            QTimer.singleShot(3000, lambda: self._download_notification.setVisible(False))
        except Exception:
            pass

    def _on_print_markers(self):
        """Open system print dialog with the first marker image."""
        if not self._marker_paths:
            return

        first_path = next(iter(self._marker_paths.values()))
        if not os.path.exists(first_path):
            return

        printer = QPrinter(QPrinter.PrinterMode.HighResolution)
        dialog = QPrintDialog(printer)
        if dialog.exec() == QPrintDialog.DialogCode.Accepted:
            pixmap = QPixmap(first_path)
            if not pixmap.isNull():
                painter = QPainter(printer)
                rect = printer.pageRect(QPrinter.Unit.DevicePixel)
                scaled = pixmap.scaled(
                    rect.width(), rect.height(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                x = (rect.width() - scaled.width()) / 2
                y = (rect.height() - scaled.height()) / 2
                painter.drawPixmap(int(x), int(y), scaled)
                painter.end()

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

    def _refresh_text(self):
        t = self._locale.t
        self._title_label.setText(t("dual_actor_settings_title"))
        self._back_btn.setText(t("btn_back"))
        self._count_label.setText(t("actor_count_label"))
        self._soon_label.setText(t("actor_count_soon"))
        self._contact_cb.setText(t("contact_expected_label"))
        self._contact_desc.setText(t("contact_expected_desc"))
        self._contact_warning.setText(t("contact_expected_warning"))

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

        # ArUco fallback
        self._aruco_cb.setText(t("aruco_fallback_label"))
        self._aruco_desc.setText(t("aruco_fallback_desc"))
        self._generate_btn.setText(t("aruco_generate_btn"))
        self._download_btn.setText(t("aruco_download_btn"))
        self._print_btn.setText(t("aruco_print_btn"))

        self._restore_btn.setText(t("btn_restore_defaults"))
        self._continue_btn.setText(t("btn_record"))

        self._update_count_style()
