"""
Screen 4: LiveFeedbackWidget — real-time recording feedback.

Features:
  - Bounding boxes on camera preview with "Actor 1"/"Actor 2" labels
    (color is language-independent, text translates)
  - Cross-camera matching confidence indicator (High/Medium/Low)
  - Identity swap warning (icon universal, text translates)
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QGraphicsDropShadowEffect,
)
from PySide6.QtCore import Qt, Signal, QRect, QPointF
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QBrush

from gui.i18n.locale_manager import LocaleManager

COLOR_TEXT_PRIMARY = "#ffffff"
COLOR_TEXT_SECONDARY = "#aaaaaa"
COLOR_BG = "#1e1e1e"
COLOR_CARD_BG = "#2d2d2d"
COLOR_BORDER = "#444444"
COLOR_ACTOR_1 = "#1a73e8"   # blue
COLOR_ACTOR_2 = "#e81a73"   # pink/magenta
COLOR_CONFIDENCE_HIGH = "#4caf50"
COLOR_CONFIDENCE_MEDIUM = "#ff9800"
COLOR_CONFIDENCE_LOW = "#f44336"
COLOR_SWAP_WARNING = "#f44336"


class BBoxOverlay(QWidget):
    """Transparent overlay for drawing bounding boxes on camera preview.

    Attributes:
        bboxes: list of tuples (x1, y1, x2, y2, label, color_hex)
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._bboxes = []

    def set_bboxes(self, bboxes: list):
        """Set bounding boxes to draw.

        Args:
            bboxes: list of (x1, y1, x2, y2, label, color_hex) tuples.
        """
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
            painter.setBrush(QBrush(QColor(0, 0, 0, 0)))  # transparent fill
            painter.drawRect(int(x1), int(y1), int(x2 - x1), int(y2 - y1))

            # Label background
            font = QFont("Segoe UI", 10, QFont.Weight.Bold)
            painter.setFont(font)
            text_rect = painter.boundingRect(QRectF(x1, y1 - 20, 200, 20),
                                              Qt.AlignmentFlag.AlignLeft, label)
            painter.fillRect(text_rect, QColor(color_hex))
            painter.setPen(QPen(QColor("#ffffff")))
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, label)

        painter.end()


class LiveFeedbackWidget(QWidget):
    """Real-time feedback during dual-actor recording.

    Signals:
        swap_dismissed: emitted when user acknowledges identity swap warning.
    """

    swap_dismissed = Signal()

    # Actor colors — language-independent
    ACTOR_COLORS = {
        1: COLOR_ACTOR_1,
        2: COLOR_ACTOR_2,
    }

    def __init__(self, locale: LocaleManager, parent=None):
        super().__init__(parent)
        self._locale = locale
        self._confidence_level = "medium"  # "high", "medium", "low"
        self._identity_swap_detected = False
        self._setup_ui()
        self._connect_signals()
        self._refresh_text()

    def set_bboxes(self, actor_id: int, bboxes: list):
        """Update bounding boxes for a specific actor.

        Args:
            actor_id: 1 or 2
            bboxes: list of (x1, y1, x2, y2) tuples
        """
        color = self.ACTOR_COLORS.get(actor_id, COLOR_TEXT_SECONDARY)
        label_key = "actor_label"
        label = self._locale.t(label_key, id=actor_id)

        overlay_bboxes = [(x1, y1, x2, y2, label, color)
                          for x1, y1, x2, y2 in bboxes]

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

    # ---- UI Setup ----
    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(16)

        # --- Title ---
        self._title_label = QLabel()
        self._title_label.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        self._title_label.setStyleSheet(f"color: {COLOR_TEXT_PRIMARY};")
        root.addWidget(self._title_label)

        # --- Camera preview area with overlay placeholders ---
        preview_row = QHBoxLayout()
        preview_row.setSpacing(12)

        self._preview_actor1 = self._build_preview("actor1")
        self._preview_actor2 = self._build_preview("actor2")

        preview_row.addWidget(self._preview_actor1, 1)
        preview_row.addWidget(self._preview_actor2, 1)
        root.addLayout(preview_row)

        # --- Confidence indicator ---
        conf_card = self._build_card()
        conf_layout = QVBoxLayout(conf_card)
        conf_layout.setContentsMargins(16, 12, 16, 12)
        conf_layout.setSpacing(4)

        self._conf_label = QLabel()
        self._conf_label.setStyleSheet(f"color: {COLOR_TEXT_SECONDARY}; font-size: 13px; background: transparent;")
        conf_layout.addWidget(self._conf_label)

        conf_value_row = QHBoxLayout()
        conf_value_row.setSpacing(8)

        self._conf_indicator = QFrame()
        self._conf_indicator.setFixedSize(12, 12)
        self._update_confidence_indicator_color()
        conf_value_row.addWidget(self._conf_indicator)

        self._conf_text = QLabel()
        self._conf_text.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        self._update_confidence_display()
        conf_value_row.addWidget(self._conf_text)
        conf_value_row.addStretch()

        conf_layout.addLayout(conf_value_row)
        root.addWidget(conf_card)

        # --- Identity swap warning ---
        self._swap_warning_frame = QFrame()
        self._swap_warning_frame.setStyleSheet(f"""
            QFrame {{
                background-color: #3d1111;
                border: 1px solid {COLOR_SWAP_WARNING};
                border-radius: 8px;
            }}
        """)
        swap_layout = QHBoxLayout(self._swap_warning_frame)
        swap_layout.setContentsMargins(16, 12, 16, 12)
        swap_layout.setSpacing(12)

        # Universal warning icon (triangle)
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
        swap_layout.addWidget(swap_icon)

        self._swap_text = QLabel()
        self._swap_text.setWordWrap(True)
        self._swap_text.setStyleSheet(f"color: {COLOR_SWAP_WARNING}; font-size: 13px; background: transparent;")
        swap_layout.addWidget(self._swap_text, 1)

        self._swap_warning_frame.setVisible(False)
        root.addWidget(self._swap_warning_frame)

        root.addStretch(1)

    def _build_preview(self, actor_key: str) -> QFrame:
        """Build a camera preview placeholder with overlay."""
        frame = QFrame()
        frame.setMinimumHeight(180)
        frame.setStyleSheet(f"""
            QFrame {{
                background-color: {COLOR_CARD_BG};
                border: 1px solid {COLOR_BORDER};
                border-radius: 8px;
            }}
        """)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(4, 4, 4, 4)

        overlay = BBoxOverlay(frame)
        overlay.setStyleSheet("background: transparent;")
        layout.addWidget(overlay)

        setattr(self, f"_overlay_{actor_key}", overlay)
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
        self._conf_text.setStyleSheet(f"color: {color}; font-size: 14px; font-weight: bold; background: transparent;")
        self._update_confidence_indicator_color()

    def _update_confidence_indicator_color(self):
        color_map = {
            "high": COLOR_CONFIDENCE_HIGH,
            "medium": COLOR_CONFIDENCE_MEDIUM,
            "low": COLOR_CONFIDENCE_LOW,
        }
        color = color_map.get(self._confidence_level, COLOR_CONFIDENCE_MEDIUM)
        self._conf_indicator.setStyleSheet(f"""
            background-color: {color};
            border-radius: 6px;
        """)

    def _refresh_text(self):
        t = self._locale.t
        self._title_label.setText(t("live_feedback_title"))
        self._conf_label.setText(t("matching_confidence_label"))
        self._swap_text.setText(t("identity_swap_warning"))
        self._update_confidence_display()
