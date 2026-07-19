"""
Screen 4: LiveFeedbackWidget — real-time recording feedback.

Features:
  - Bounding boxes on camera preview with "Actor 1"/"Actor 2" labels
  - Cross-camera matching confidence indicator (High/Medium/Low)
  - Identity swap warning with detail text
  - Frame counter and recording indicator
  - Stop Recording button -> navigates to Diagnostics (Screen 5)
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame,
    QGridLayout, QSizePolicy, QStackedWidget, QSlider,
)
from PySide6.QtCore import Qt, Signal, QRectF
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QBrush, QPalette, QPixmap, QImage

from gui.i18n.locale_manager import LocaleManager

COLOR_TEXT_PRIMARY = "#ffffff"
COLOR_TEXT_SECONDARY = "#aaaaaa"
COLOR_BG = "#1e1e1e"
COLOR_CARD_BG = "#2d2d2d"
COLOR_BORDER = "#444444"
COLOR_ACTOR_1 = "#1a73e8"
COLOR_ACTOR_2 = "#e81a73"
COLOR_CONFIDENCE_HIGH = "#4caf50"
COLOR_CONFIDENCE_MEDIUM = "#ff9800"
COLOR_CONFIDENCE_LOW = "#f44336"
COLOR_SWAP_WARNING = "#f44336"
COLOR_BTN_PRIMARY = "#1a73e8"
COLOR_BTN_PRIMARY_HOVER = "#1557b0"
COLOR_BTN_STOP = "#c62828"
COLOR_BTN_STOP_HOVER = "#b71c1c"
COLOR_RECORDING_DOT = "#f44336"


class BBoxOverlay(QWidget):
    """Transparent overlay for drawing bounding boxes on camera preview."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._bboxes = []

    def set_bboxes(self, bboxes: list):
        self._bboxes = bboxes
        self.update()

    def paintEvent(self, event):
        if not self._bboxes:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        for x1, y1, x2, y2, label, color_hex in self._bboxes:
            pen = QPen(QColor(color_hex), 2)
            painter.setPen(pen)
            painter.setBrush(QBrush(QColor(0, 0, 0, 0)))
            painter.drawRect(int(x1), int(y1), int(x2 - x1), int(y2 - y1))

            font = QFont("Segoe UI", 10, QFont.Weight.Bold)
            painter.setFont(font)
            text_rect = painter.boundingRect(
                QRectF(x1, y1 - 22, 220, 22),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                label,
            )
            painter.fillRect(text_rect, QColor(color_hex))
            painter.setPen(QPen(QColor("#ffffff")))
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, label)

        painter.end()


class LiveFeedbackWidget(QWidget):
    """Real-time feedback during dual-actor recording.

    Signals:
        stop_clicked: emitted when user clicks Stop Recording.
        swap_dismissed: emitted when user acknowledges identity swap warning.
    """

    stop_clicked = Signal()
    swap_dismissed = Signal()
    review_frame_changed = Signal(int)

    ACTOR_COLORS = {
        1: COLOR_ACTOR_1,
        2: COLOR_ACTOR_2,
    }

    def __init__(self, locale: LocaleManager, parent=None):
        super().__init__(parent)
        self._locale = locale
        self._confidence_level = "medium"
        self._identity_swap_detected = False
        self._frame_count = 0
        self._review_mode = False
        self._review_2d_data = None
        self._review_total_frames = 0
        self._review_current_frame = 0
        self._setup_ui()
        self._connect_signals()
        self._refresh_text()

    # ---- Public API ----
    def set_bboxes(self, actor_id: int, bboxes: list):
        """Update bounding boxes for a specific actor.

        Args:
            actor_id: 1 or 2
            bboxes: list of (x1, y1, x2, y2) tuples
        """
        color = self.ACTOR_COLORS.get(actor_id, COLOR_TEXT_SECONDARY)
        label = self._locale.t("actor_label", id=actor_id)
        overlay_bboxes = [(x1, y1, x2, y2, label, color) for x1, y1, x2, y2 in bboxes]

        overlay_attr = f"_overlay_actor{actor_id}"
        if hasattr(self, overlay_attr):
            getattr(self, overlay_attr).set_bboxes(overlay_bboxes)

    def set_confidence(self, level: str):
        """Set cross-camera matching confidence.

        Args:
            level: "high", "medium", or "low"
        """
        if level not in ("high", "medium", "low"):
            raise ValueError(f"Invalid confidence level: {level}")
        self._confidence_level = level
        self._update_confidence_display()

    def show_identity_swap(self):
        """Show the identity swap warning banner."""
        self._identity_swap_detected = True
        self._swap_warning_frame.setVisible(True)

    def hide_identity_swap(self):
        """Hide the identity swap warning banner."""
        self._identity_swap_detected = False
        self._swap_warning_frame.setVisible(False)

    def set_frame_count(self, count: int):
        """Update the current frame counter."""
        self._frame_count = count
        self._frame_label.setText(self._locale.t("frame_counter", frame=count))

    def set_total_frames(self, total: int):
        """Set total frames for progress display (optional)."""
        self._total_frames = total

    def set_camera_frame(self, slot: int, q_image):
        """Display a camera frame in a preview slot.

        Args:
            slot: 0 for first preview, 1 for second preview
            q_image: QImage from skellycam's new_image_signal
        """
        feed_key = f"_feed_actor{slot + 1}"
        feed_label = getattr(self, feed_key, None)
        if feed_label is None:
            return

        pixmap = QPixmap.fromImage(q_image)
        if pixmap.isNull():
            return

        scaled = pixmap.scaled(
            feed_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        feed_label.setPixmap(scaled)

    # ---- Review Mode ----
    def set_review_mode(self, enabled: bool):
        """Switch between live recording mode and review mode."""
        self._review_mode = enabled
        self._recording_dot.setVisible(not enabled)
        self._recording_text.setVisible(not enabled)
        self._stop_btn.setVisible(not enabled)
        self._review_controls.setVisible(enabled)
        self._confidence_card.setVisible(not enabled)
        self._swap_warning_frame.setVisible(False)

    def load_review_data(self, two_d_data):
        """Load 2D detection data for review playback.

        Args:
            two_d_data: numpy array of shape (num_cameras, num_frames, num_keypoints, 2)
        """
        self._review_2d_data = two_d_data
        if two_d_data is not None and two_d_data.shape[1] > 0:
            total = two_d_data.shape[1]
            self._review_total_frames = total
            self._review_slider.setMaximum(max(0, total - 1))
            self._review_slider.setValue(0)
            self._review_current_frame = 0
            self._update_review_frame_display()
        else:
            self._review_total_frames = 0
            self._review_slider.setMaximum(0)

    def set_review_bboxes_for_frame(self, frame_idx: int, bboxes_cam0: dict, bboxes_cam1: dict):
        """Set bounding boxes for a specific review frame on both camera previews.

        Args:
            frame_idx: frame index
            bboxes_cam0: dict {camera_index: (x1, y1, x2, y2)} for camera 0
            bboxes_cam1: dict {camera_index: (x1, y1, x2, y2)} for camera 1
        """
        for cam_idx, (x1, y1, x2, y2) in bboxes_cam0.items():
            self.set_bboxes(1, [(x1, y1, x2, y2)])
        for cam_idx, (x1, y1, x2, y2) in bboxes_cam1.items():
            self.set_bboxes(2, [(x1, y1, x2, y2)])

    def _on_review_slider_changed(self, value: int):
        self._review_current_frame = value
        self._update_review_frame_display()
        self.review_frame_changed.emit(value)

    def _on_prev_frame(self):
        val = max(0, self._review_current_frame - 1)
        self._review_slider.setValue(val)

    def _on_next_frame(self):
        val = min(self._review_total_frames - 1, self._review_current_frame + 1)
        self._review_slider.setValue(val)

    def _update_review_frame_display(self):
        t = self._locale.t
        self._review_frame_label.setText(
            t("review_frame_of", frame=self._review_current_frame + 1, total=self._review_total_frames)
        )

    # ---- UI Setup ----
    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # --- Top bar: title + recording indicator + frame counter ---
        top_bar = QWidget()
        top_bar.setStyleSheet(f"background-color: {COLOR_CARD_BG}; border-bottom: 1px solid {COLOR_BORDER};")
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(24, 12, 24, 12)
        top_layout.setSpacing(12)

        self._recording_dot = QLabel()
        self._recording_dot.setFixedSize(10, 10)
        self._recording_dot.setStyleSheet(
            f"background-color: {COLOR_RECORDING_DOT}; border-radius: 5px;"
        )
        top_layout.addWidget(self._recording_dot, 0, Qt.AlignmentFlag.AlignVCenter)

        self._recording_text = QLabel()
        self._recording_text.setStyleSheet(
            f"color: {COLOR_RECORDING_DOT}; font-size: 13px; font-weight: bold; background: transparent;"
        )
        top_layout.addWidget(self._recording_text, 0, Qt.AlignmentFlag.AlignVCenter)

        self._title_label = QLabel()
        self._title_label.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        self._title_label.setStyleSheet(f"color: {COLOR_TEXT_PRIMARY}; background: transparent;")
        top_layout.addWidget(self._title_label, 1, Qt.AlignmentFlag.AlignVCenter)

        self._frame_label = QLabel()
        self._frame_label.setStyleSheet(
            f"color: {COLOR_TEXT_SECONDARY}; font-size: 13px; background: transparent;"
        )
        top_layout.addWidget(self._frame_label, 0, Qt.AlignmentFlag.AlignVCenter)

        root.addWidget(top_bar)

        # --- Main content area (scrollable) ---
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(24, 16, 24, 16)
        content_layout.setSpacing(16)

        # --- Camera previews grid (2x1) ---
        previews_grid = QHBoxLayout()
        previews_grid.setSpacing(12)

        self._preview_actor1 = self._build_preview("actor1", 1)
        self._preview_actor2 = self._build_preview("actor2", 2)

        previews_grid.addWidget(self._preview_actor1, 1)
        previews_grid.addWidget(self._preview_actor2, 1)
        content_layout.addLayout(previews_grid)

        # --- Confidence indicator ---
        conf_card = self._build_card()
        conf_layout = QHBoxLayout(conf_card)
        conf_layout.setContentsMargins(16, 12, 16, 12)
        conf_layout.setSpacing(10)

        self._conf_label = QLabel()
        self._conf_label.setStyleSheet(
            f"color: {COLOR_TEXT_SECONDARY}; font-size: 13px; background: transparent;"
        )
        conf_layout.addWidget(self._conf_label)

        self._conf_indicator = QFrame()
        self._conf_indicator.setFixedSize(12, 12)
        self._update_confidence_indicator_color()
        conf_layout.addWidget(self._conf_indicator, 0, Qt.AlignmentFlag.AlignVCenter)

        self._conf_text = QLabel()
        self._conf_text.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        self._update_confidence_display()
        conf_layout.addWidget(self._conf_text, 1, Qt.AlignmentFlag.AlignVCenter)

        content_layout.addWidget(conf_card)

        # --- Identity swap warning ---
        self._swap_warning_frame = QFrame()
        self._swap_warning_frame.setStyleSheet(f"""
            QFrame {{
                background-color: #3d1111;
                border: 1px solid {COLOR_SWAP_WARNING};
                border-radius: 8px;
            }}
        """)
        swap_layout = QVBoxLayout(self._swap_warning_frame)
        swap_layout.setContentsMargins(16, 12, 16, 12)
        swap_layout.setSpacing(6)

        swap_header = QHBoxLayout()
        swap_header.setSpacing(10)

        swap_icon = QLabel("!")
        swap_icon.setFixedSize(28, 28)
        swap_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        swap_icon.setStyleSheet(f"""
            background-color: {COLOR_SWAP_WARNING};
            color: white;
            border-radius: 14px;
            font-size: 16px;
            font-weight: bold;
        """)
        swap_header.addWidget(swap_icon, 0, Qt.AlignmentFlag.AlignTop)

        self._swap_text = QLabel()
        self._swap_text.setWordWrap(True)
        self._swap_text.setStyleSheet(
            f"color: {COLOR_SWAP_WARNING}; font-size: 13px; font-weight: bold; background: transparent;"
        )
        swap_header.addWidget(self._swap_text, 1)
        swap_layout.addLayout(swap_header)

        self._swap_detail = QLabel()
        self._swap_detail.setWordWrap(True)
        self._swap_detail.setStyleSheet(
            f"color: {COLOR_TEXT_SECONDARY}; font-size: 12px; background: transparent;"
        )
        swap_layout.addWidget(self._swap_detail)

        self._swap_warning_frame.setVisible(False)
        content_layout.addWidget(self._swap_warning_frame)

        # --- Review controls (hidden during recording, shown after processing) ---
        self._review_controls = QWidget()
        self._review_controls.setVisible(False)
        review_layout = QVBoxLayout(self._review_controls)
        review_layout.setContentsMargins(0, 8, 0, 8)
        review_layout.setSpacing(8)

        self._review_frame_label = QLabel()
        self._review_frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._review_frame_label.setStyleSheet(
            f"color: {COLOR_TEXT_SECONDARY}; font-size: 13px; background: transparent;"
        )
        review_layout.addWidget(self._review_frame_label)

        slider_row = QHBoxLayout()
        slider_row.setSpacing(12)

        self._review_prev_btn = QPushButton()
        self._review_prev_btn.setFixedHeight(32)
        self._review_prev_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._review_prev_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {COLOR_TEXT_SECONDARY};
                border: 1px solid {COLOR_BORDER};
                border-radius: 4px;
                font-size: 12px;
                padding: 0 12px;
            }}
            QPushButton:hover {{
                border-color: {COLOR_BTN_PRIMARY};
                color: {COLOR_BTN_PRIMARY};
            }}
        """)
        self._review_prev_btn.clicked.connect(self._on_prev_frame)
        slider_row.addWidget(self._review_prev_btn)

        self._review_slider = QSlider(Qt.Orientation.Horizontal)
        self._review_slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                background: {COLOR_BORDER};
                height: 6px;
                border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                background: {COLOR_BTN_PRIMARY};
                width: 16px;
                height: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }}
            QSlider::handle:horizontal:hover {{
                background: {COLOR_BTN_PRIMARY_HOVER};
            }}
        """)
        self._review_slider.valueChanged.connect(self._on_review_slider_changed)
        slider_row.addWidget(self._review_slider, 1)

        self._review_next_btn = QPushButton()
        self._review_next_btn.setFixedHeight(32)
        self._review_next_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._review_next_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {COLOR_TEXT_SECONDARY};
                border: 1px solid {COLOR_BORDER};
                border-radius: 4px;
                font-size: 12px;
                padding: 0 12px;
            }}
            QPushButton:hover {{
                border-color: {COLOR_BTN_PRIMARY};
                color: {COLOR_BTN_PRIMARY};
            }}
        """)
        self._review_next_btn.clicked.connect(self._on_next_frame)
        slider_row.addWidget(self._review_next_btn)

        review_layout.addLayout(slider_row)
        content_layout.addWidget(self._review_controls)

        # --- Confidence card (reference for show/hide) ---
        self._confidence_card = conf_card

        content_layout.addStretch(1)
        root.addWidget(content, 1)

        # --- Bottom bar: Stop Recording button ---
        bottom_bar = QWidget()
        bottom_bar.setStyleSheet(f"background-color: {COLOR_CARD_BG}; border-top: 1px solid {COLOR_BORDER};")
        bottom_layout = QHBoxLayout(bottom_bar)
        bottom_layout.setContentsMargins(24, 12, 24, 12)

        bottom_layout.addStretch(1)

        self._stop_btn = QPushButton()
        self._stop_btn.setMinimumHeight(48)
        self._stop_btn.setMinimumWidth(200)
        self._stop_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._stop_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLOR_BTN_STOP};
                color: {COLOR_TEXT_PRIMARY};
                border: none;
                border-radius: 6px;
                font-size: 15px;
                font-weight: bold;
                padding: 0 32px;
            }}
            QPushButton:hover {{
                background-color: {COLOR_BTN_STOP_HOVER};
            }}
        """)
        self._stop_btn.clicked.connect(self.stop_clicked.emit)
        bottom_layout.addWidget(self._stop_btn, 0, Qt.AlignmentFlag.AlignVCenter)

        root.addWidget(bottom_bar)

    def _build_preview(self, actor_key: str, actor_id: int) -> QFrame:
        """Build a camera preview with feed label and overlay."""
        frame = QFrame()
        frame.setMinimumHeight(200)
        frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        frame.setStyleSheet(f"""
            QFrame {{
                background-color: {COLOR_CARD_BG};
                border: 1px solid {COLOR_BORDER};
                border-radius: 8px;
            }}
        """)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        actor_color = self.ACTOR_COLORS.get(actor_id, COLOR_TEXT_SECONDARY)
        label_text = self._locale.t("actor_label", id=actor_id)

        header = QLabel(label_text)
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet(
            f"color: {actor_color}; font-size: 13px; font-weight: bold; background: transparent; padding: 4px;"
        )
        layout.addWidget(header, 0, Qt.AlignmentFlag.AlignTop)

        feed_label = QLabel()
        feed_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        feed_label.setStyleSheet("background: transparent;")
        feed_label.setMinimumSize(160, 120)
        layout.addWidget(feed_label, 1, Qt.AlignmentFlag.AlignCenter)

        overlay = BBoxOverlay(frame)
        overlay.setStyleSheet("background: transparent;")
        layout.addWidget(overlay, 0)

        setattr(self, f"_overlay_{actor_key}", overlay)
        setattr(self, f"_preview_header_{actor_key}", header)
        setattr(self, f"_feed_{actor_key}", feed_label)
        return frame

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

    # ---- UI Updates ----
    def _update_confidence_display(self):
        t = self._locale.t
        level_map = {
            "high": ("confidence_high", COLOR_CONFIDENCE_HIGH),
            "medium": ("confidence_medium", COLOR_CONFIDENCE_MEDIUM),
            "low": ("confidence_low", COLOR_CONFIDENCE_LOW),
        }
        key, color = level_map.get(self._confidence_level, ("confidence_medium", COLOR_CONFIDENCE_MEDIUM))
        self._conf_text.setText(t(key))
        self._conf_text.setStyleSheet(
            f"color: {color}; font-size: 14px; font-weight: bold; background: transparent;"
        )
        self._update_confidence_indicator_color()

    def _update_confidence_indicator_color(self):
        color_map = {
            "high": COLOR_CONFIDENCE_HIGH,
            "medium": COLOR_CONFIDENCE_MEDIUM,
            "low": COLOR_CONFIDENCE_LOW,
        }
        color = color_map.get(self._confidence_level, COLOR_CONFIDENCE_MEDIUM)
        self._conf_indicator.setStyleSheet(
            f"background-color: {color}; border-radius: 6px;"
        )

    def _refresh_text(self):
        t = self._locale.t
        self._title_label.setText(t("live_feedback_title"))
        self._recording_text.setText(t("recording_indicator"))
        self._frame_label.setText(t("frame_counter", frame=self._frame_count))
        self._conf_label.setText(t("matching_confidence_label"))
        self._swap_text.setText(t("identity_swap_warning"))
        self._swap_detail.setText(t("swap_warning_detail"))
        self._stop_btn.setText(t("btn_stop_recording"))
        self._review_prev_btn.setText(t("review_prev"))
        self._review_next_btn.setText(t("review_next"))
        self._update_review_frame_display()

        for actor_id, key in [(1, "actor1"), (2, "actor2")]:
            header = getattr(self, f"_preview_header_{key}", None)
            if header:
                header.setText(t("actor_label", id=actor_id))

        self._update_confidence_display()
