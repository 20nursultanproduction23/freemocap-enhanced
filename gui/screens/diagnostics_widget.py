"""
Screen 5: DiagnosticsWidget — post-recording quality review.

Features:
  - Summary card with metrics (confident frames / low confidence / identity swaps)
  - Problem timestamp list with frame preview on click
  - Buttons: "Looks good -> Export" and "Issues found -> What can I do"
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame,
    QListWidget, QListWidgetItem, QStackedWidget,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont

from gui.i18n.locale_manager import LocaleManager

COLOR_TEXT_PRIMARY = "#ffffff"
COLOR_TEXT_SECONDARY = "#aaaaaa"
COLOR_BG = "#1e1e1e"
COLOR_CARD_BG = "#2d2d2d"
COLOR_BORDER = "#444444"
COLOR_BTN_PRIMARY = "#1a73e8"
COLOR_BTN_PRIMARY_HOVER = "#1557b0"
COLOR_BTN_SUCCESS = "#4caf50"
COLOR_BTN_SUCCESS_HOVER = "#388e3c"
COLOR_BTN_WARNING = "#ff9800"
COLOR_BTN_WARNING_HOVER = "#f57c00"
COLOR_METRIC_GOOD = "#4caf50"
COLOR_METRIC_WARN = "#ff9800"
COLOR_METRIC_BAD = "#f44336"
COLOR_NO_PROBLEMS = "#4caf50"


class DiagnosticsWidget(QWidget):
    """Post-recording diagnostics with metrics, problems, and action buttons.

    Signals:
        export_clicked: emitted when user clicks "Looks good -> Export".
        fix_issues_clicked: emitted when user clicks "Issues found -> What can I do".
        reprocess_clicked: emitted when user clicks "Reprocess with different settings".
    """

    export_clicked = Signal()
    fix_issues_clicked = Signal()
    reprocess_clicked = Signal()

    def __init__(self, locale: LocaleManager, parent=None):
        super().__init__(parent)
        self._locale = locale
        self._metrics = {
            "confident_frames": 0,
            "low_confidence_frames": 0,
            "identity_swaps": 0,
            "total_frames": 0,
        }
        self._problems = []  # list of dicts: {frame, timestamp, description, severity}
        self._setup_ui()
        self._connect_signals()
        self._refresh_text()

    def set_metrics(self, confident: int, low_confidence: int, identity_swaps: int, total: int):
        """Update the metrics display."""
        self._metrics = {
            "confident_frames": confident,
            "low_confidence_frames": low_confidence,
            "identity_swaps": identity_swaps,
            "total_frames": total,
        }
        self._update_metrics_display()

    def set_problems(self, problems: list):
        """Set the list of problem timestamps.

        Args:
            problems: list of dicts with keys:
                frame (int): frame number
                timestamp (str): human-readable timestamp
                description (str): problem description
                severity (str): "warning" or "error"
        """
        self._problems = problems
        self._update_problems_list()

    # ---- UI Setup ----
    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(40, 30, 40, 30)
        root.setSpacing(16)

        # --- Title ---
        self._title_label = QLabel()
        self._title_label.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        self._title_label.setStyleSheet(f"color: {COLOR_TEXT_PRIMARY};")
        root.addWidget(self._title_label)

        # --- Metrics cards ---
        metrics_row = QHBoxLayout()
        metrics_row.setSpacing(12)

        self._metric_confident = self._build_metric_card("confident")
        self._metric_low = self._build_metric_card("low")
        self._metric_swaps = self._build_metric_card("swaps")
        self._metric_total = self._build_metric_card("total")

        metrics_row.addWidget(self._metric_confident, 1)
        metrics_row.addWidget(self._metric_low, 1)
        metrics_row.addWidget(self._metric_swaps, 1)
        metrics_row.addWidget(self._metric_total, 1)
        root.addLayout(metrics_row)

        # --- Problems section ---
        problems_card = QFrame()
        problems_card.setStyleSheet(f"""
            QFrame {{
                background-color: {COLOR_CARD_BG};
                border: 1px solid {COLOR_BORDER};
                border-radius: 8px;
            }}
        """)
        problems_layout = QVBoxLayout(problems_card)
        problems_layout.setContentsMargins(20, 16, 20, 16)
        problems_layout.setSpacing(8)

        self._problems_title = QLabel()
        self._problems_title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        self._problems_title.setStyleSheet(f"color: {COLOR_TEXT_PRIMARY}; background: transparent;")
        problems_layout.addWidget(self._problems_title)

        self._problems_list = QListWidget()
        self._problems_list.setStyleSheet(f"""
            QListWidget {{
                background-color: {COLOR_BG};
                border: 1px solid {COLOR_BORDER};
                border-radius: 4px;
                color: {COLOR_TEXT_PRIMARY};
                font-size: 13px;
            }}
            QListWidget::item {{
                padding: 8px;
                border-bottom: 1px solid {COLOR_BORDER};
            }}
            QListWidget::item:selected {{
                background-color: #1a73e833;
            }}
        """)
        self._problems_list.setMinimumHeight(120)
        problems_layout.addWidget(self._problems_list)

        self._problems_empty_label = QLabel()
        self._problems_empty_label.setStyleSheet(f"color: {COLOR_NO_PROBLEMS}; font-size: 13px; background: transparent;")
        self._problems_empty_label.setVisible(False)
        problems_layout.addWidget(self._problems_empty_label)

        root.addWidget(problems_card)

        # --- Action buttons ---
        buttons_row = QHBoxLayout()
        buttons_row.setSpacing(12)

        self._reprocess_btn = QPushButton()
        self._reprocess_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._reprocess_btn.setMinimumHeight(44)
        self._reprocess_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {COLOR_TEXT_SECONDARY};
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
        self._reprocess_btn.clicked.connect(self.reprocess_clicked.emit)
        buttons_row.addWidget(self._reprocess_btn)

        buttons_row.addStretch()

        self._fix_btn = QPushButton()
        self._fix_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._fix_btn.setMinimumHeight(44)
        self._fix_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLOR_BTN_WARNING};
                color: {COLOR_TEXT_PRIMARY};
                border: none;
                border-radius: 6px;
                font-size: 14px;
                font-weight: bold;
                padding: 0 24px;
            }}
            QPushButton:hover {{
                background-color: {COLOR_BTN_WARNING_HOVER};
            }}
        """)
        self._fix_btn.clicked.connect(self.fix_issues_clicked.emit)
        buttons_row.addWidget(self._fix_btn)

        self._export_btn = QPushButton()
        self._export_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._export_btn.setMinimumHeight(44)
        self._export_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLOR_BTN_SUCCESS};
                color: {COLOR_TEXT_PRIMARY};
                border: none;
                border-radius: 6px;
                font-size: 14px;
                font-weight: bold;
                padding: 0 24px;
            }}
            QPushButton:hover {{
                background-color: {COLOR_BTN_SUCCESS_HOVER};
            }}
        """)
        self._export_btn.clicked.connect(self.export_clicked.emit)
        buttons_row.addWidget(self._export_btn)

        root.addLayout(buttons_row)

        root.addStretch(1)

    def _build_metric_card(self, metric_type: str) -> QFrame:
        """Build a metric card widget."""
        card = QFrame()
        card.setStyleSheet(f"""
            QFrame {{
                background-color: {COLOR_CARD_BG};
                border: 1px solid {COLOR_BORDER};
                border-radius: 8px;
            }}
        """)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(4)

        value_label = QLabel("0")
        value_label.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
        value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        value_label.setObjectName(f"value_{metric_type}")
        layout.addWidget(value_label)

        name_label = QLabel()
        name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        name_label.setObjectName(f"name_{metric_type}")
        name_label.setStyleSheet(f"color: {COLOR_TEXT_SECONDARY}; font-size: 11px; background: transparent;")
        layout.addWidget(name_label)

        return card

    # ---- Signals ----
    def _connect_signals(self):
        self._locale.language_changed.connect(self._refresh_text)

    # ---- UI Updates ----
    def _update_metrics_display(self):
        t = self._locale.t
        m = self._metrics

        # Confident frames
        confident_value = self._metric_confident.findChild(QLabel, "value_confident")
        confident_name = self._metric_confident.findChild(QLabel, "name_confident")
        if confident_value:
            confident_value.setText(str(m["confident_frames"]))
            confident_value.setStyleSheet(f"color: {COLOR_METRIC_GOOD}; font-size: 24px; font-weight: bold; background: transparent;")
        if confident_name:
            confident_name.setText(t("metric_confident_frames"))

        # Low confidence frames
        low_value = self._metric_low.findChild(QLabel, "value_low")
        low_name = self._metric_low.findChild(QLabel, "name_low")
        if low_value:
            low_value.setText(str(m["low_confidence_frames"]))
            low_color = COLOR_METRIC_WARN if m["low_confidence_frames"] > 0 else COLOR_METRIC_GOOD
            low_value.setStyleSheet(f"color: {low_color}; font-size: 24px; font-weight: bold; background: transparent;")
        if low_name:
            low_name.setText(t("metric_low_confidence_frames"))

        # Identity swaps
        swaps_value = self._metric_swaps.findChild(QLabel, "value_swaps")
        swaps_name = self._metric_swaps.findChild(QLabel, "name_swaps")
        if swaps_value:
            swaps_value.setText(str(m["identity_swaps"]))
            swaps_color = COLOR_METRIC_BAD if m["identity_swaps"] > 0 else COLOR_METRIC_GOOD
            swaps_value.setStyleSheet(f"color: {swaps_color}; font-size: 24px; font-weight: bold; background: transparent;")
        if swaps_name:
            swaps_name.setText(t("metric_identity_swaps"))

        # Total frames
        total_value = self._metric_total.findChild(QLabel, "value_total")
        total_name = self._metric_total.findChild(QLabel, "name_total")
        if total_value:
            total_value.setText(str(m["total_frames"]))
            total_value.setStyleSheet(f"color: {COLOR_TEXT_PRIMARY}; font-size: 24px; font-weight: bold; background: transparent;")
        if total_name:
            total_name.setText(t("metric_total_frames"))

    def _update_problems_list(self):
        t = self._locale.t
        self._problems_list.clear()

        if not self._problems:
            self._problems_list.setVisible(False)
            self._problems_empty_label.setVisible(True)
            self._problems_empty_label.setText(t("problems_empty"))
            return

        self._problems_list.setVisible(True)
        self._problems_empty_label.setVisible(False)

        for problem in self._problems:
            severity_icon = "!" if problem.get("severity") == "error" else "~"
            text = f"[{severity_icon}] Frame {problem['frame']} ({problem['timestamp']}): {problem['description']}"
            item = QListWidgetItem(text)
            if problem.get("severity") == "error":
                item.setForeground(QColor(COLOR_METRIC_BAD))
            else:
                item.setForeground(QColor(COLOR_METRIC_WARN))
            self._problems_list.addItem(item)

    def _refresh_text(self):
        t = self._locale.t
        self._title_label.setText(t("diagnostics_title"))
        self._problems_title.setText(t("problems_title"))
        self._export_btn.setText(t("btn_export"))
        self._fix_btn.setText(t("btn_fix_issues"))
        self._reprocess_btn.setText(t("btn_reprocess"))
        self._update_metrics_display()
        self._update_problems_list()
