"""
Screen 5: DiagnosticsWidget — post-recording quality review.

Features:
  - Summary card with metrics + quality rating
  - Problem timestamp list with severity
  - Action buttons: Export / What can I do / Reprocess / Back
  - "What can I do" sub-screen with fix suggestions
  - Export confirmation sub-screen
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame,
    QListWidget, QListWidgetItem, QStackedWidget, QTextBrowser,
    QScrollArea, QSizePolicy,
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

PAGE_METRICS = 0
PAGE_FIXES = 1
PAGE_EXPORT = 2


class DiagnosticsWidget(QWidget):
    """Post-recording diagnostics with metrics, problems, and action buttons.

    Signals:
        export_clicked: emitted when user clicks "Export" (after confirmation).
        fix_issues_clicked: emitted when user clicks "Issues found -> What can I do".
        reprocess_clicked: emitted when user clicks "Reprocess with different settings".
        back_to_feedback_clicked: emitted when user clicks "Back to Recording".
    """

    export_clicked = Signal()
    fix_issues_clicked = Signal()
    reprocess_clicked = Signal()
    back_to_feedback_clicked = Signal()

    def __init__(self, locale: LocaleManager, parent=None):
        super().__init__(parent)
        self._locale = locale
        self._metrics = {
            "confident_frames": 0,
            "low_confidence_frames": 0,
            "identity_swaps": 0,
            "total_frames": 0,
        }
        self._problems = []
        self._setup_ui()
        self._connect_signals()
        self._refresh_text()

    # ---- Public API ----
    def set_metrics(self, confident: int, low_confidence: int, identity_swaps: int, total: int):
        self._metrics = {
            "confident_frames": confident,
            "low_confidence_frames": low_confidence,
            "identity_swaps": identity_swaps,
            "total_frames": total,
        }
        self._update_metrics_display()
        self._update_fix_suggestions()

    def set_problems(self, problems: list):
        """Set the list of problem timestamps.

        Args:
            problems: list of dicts with keys:
                frame (int): frame number
                timestamp (str): human-readable timestamp
                description (str): problem description
                severity (str): "warning" or "error"
                type (str): "identity_swap", "low_confidence", "tracking_gap"
        """
        self._problems = problems
        self._update_problems_list()
        self._update_fix_suggestions()

    def reset(self):
        """Reset to metrics page."""
        self._pages.setCurrentIndex(PAGE_METRICS)

    # ---- UI Setup ----
    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # --- Top bar ---
        top_bar = QWidget()
        top_bar.setStyleSheet(f"background-color: {COLOR_CARD_BG}; border-bottom: 1px solid {COLOR_BORDER};")
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(24, 14, 24, 14)

        self._title_label = QLabel()
        self._title_label.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        self._title_label.setStyleSheet(f"color: {COLOR_TEXT_PRIMARY}; background: transparent;")
        top_layout.addWidget(self._title_label, 1)

        self._quality_badge = QLabel()
        self._quality_badge.setStyleSheet(f"""
            color: {COLOR_TEXT_PRIMARY};
            font-size: 13px; font-weight: bold;
            padding: 4px 12px;
            border-radius: 4px;
            background-color: {COLOR_METRIC_GOOD};
        """)
        top_layout.addWidget(self._quality_badge, 0, Qt.AlignmentFlag.AlignVCenter)

        root.addWidget(top_bar)

        # --- Stacked pages: metrics | fixes | export ---
        self._pages = QStackedWidget()

        self._pages.addWidget(self._build_metrics_page())
        self._pages.addWidget(self._build_fixes_page())
        self._pages.addWidget(self._build_export_page())

        root.addWidget(self._pages, 1)

    # ---- Page 1: Metrics + Problems + Buttons ----
    def _build_metrics_page(self) -> QWidget:
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")

        scroll_content = QWidget()
        scroll_content.setStyleSheet("background: transparent;")
        content = QVBoxLayout(scroll_content)
        content.setContentsMargins(24, 16, 24, 16)
        content.setSpacing(14)

        # Metrics row
        metrics_row = QHBoxLayout()
        metrics_row.setSpacing(10)

        self._metric_confident = self._build_metric_card("confident")
        self._metric_low = self._build_metric_card("low")
        self._metric_swaps = self._build_metric_card("swaps")
        self._metric_total = self._build_metric_card("total")

        metrics_row.addWidget(self._metric_confident, 1)
        metrics_row.addWidget(self._metric_low, 1)
        metrics_row.addWidget(self._metric_swaps, 1)
        metrics_row.addWidget(self._metric_total, 1)
        content.addLayout(metrics_row)

        # Problems section
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
        self._problems_list.setMinimumHeight(100)
        self._problems_list.setMaximumHeight(200)
        problems_layout.addWidget(self._problems_list)

        self._problems_empty_label = QLabel()
        self._problems_empty_label.setStyleSheet(
            f"color: {COLOR_NO_PROBLEMS}; font-size: 13px; background: transparent;"
        )
        self._problems_empty_label.setVisible(False)
        problems_layout.addWidget(self._problems_empty_label)

        content.addWidget(problems_card)
        content.addStretch(1)

        scroll.setWidget(scroll_content)
        page_layout.addWidget(scroll, 1)

        # Bottom buttons bar (fixed)
        btn_bar = QWidget()
        btn_bar.setStyleSheet(f"background-color: {COLOR_CARD_BG}; border-top: 1px solid {COLOR_BORDER};")
        btn_layout = QHBoxLayout(btn_bar)
        btn_layout.setContentsMargins(24, 12, 24, 12)
        btn_layout.setSpacing(10)

        self._back_btn = QPushButton()
        self._back_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._back_btn.setMinimumHeight(44)
        self._back_btn.setStyleSheet(f"""
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
        self._back_btn.clicked.connect(self.back_to_feedback_clicked.emit)
        btn_layout.addWidget(self._back_btn)

        btn_layout.addStretch(1)

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
        btn_layout.addWidget(self._reprocess_btn)

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
        self._fix_btn.clicked.connect(self._show_fixes_page)
        btn_layout.addWidget(self._fix_btn)

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
        self._export_btn.clicked.connect(self._show_export_page)
        btn_layout.addWidget(self._export_btn)

        page_layout.addWidget(btn_bar)
        return page

    # ---- Page 2: Fix suggestions ----
    def _build_fixes_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Top bar with back
        top = QWidget()
        top.setStyleSheet(f"background-color: {COLOR_CARD_BG}; border-bottom: 1px solid {COLOR_BORDER};")
        top_layout = QHBoxLayout(top)
        top_layout.setContentsMargins(24, 14, 24, 14)

        self._fix_back_btn = QPushButton()
        self._fix_back_btn.setFixedSize(36, 28)
        self._fix_back_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._fix_back_btn.setStyleSheet(f"""
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
        self._fix_back_btn.clicked.connect(lambda: self._pages.setCurrentIndex(PAGE_METRICS))
        top_layout.addWidget(self._fix_back_btn, 0, Qt.AlignmentFlag.AlignVCenter)

        self._fix_title = QLabel()
        self._fix_title.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        self._fix_title.setStyleSheet(f"color: {COLOR_TEXT_PRIMARY}; background: transparent;")
        top_layout.addWidget(self._fix_title, 1, Qt.AlignmentFlag.AlignVCenter)

        layout.addWidget(top)

        # Scrollable suggestions
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")

        scroll_content = QWidget()
        scroll_content.setStyleSheet("background: transparent;")
        self._fix_content = QVBoxLayout(scroll_content)
        self._fix_content.setContentsMargins(24, 16, 24, 16)
        self._fix_content.setSpacing(12)

        # Suggestion cards — built dynamically
        self._fix_swap_card = self._build_fix_card("fix_identity_swap")
        self._fix_low_card = self._build_fix_card("fix_low_confidence")
        self._fix_gap_card = self._build_fix_card("fix_tracking_gap")
        self._fix_general_card = self._build_fix_card("fix_general")

        self._fix_content.addWidget(self._fix_swap_card)
        self._fix_content.addWidget(self._fix_low_card)
        self._fix_content.addWidget(self._fix_gap_card)
        self._fix_content.addWidget(self._fix_general_card)
        self._fix_content.addStretch(1)

        scroll.setWidget(scroll_content)
        layout.addWidget(scroll, 1)

        return page

    def _build_fix_card(self, text_key: str) -> QFrame:
        card = QFrame()
        card.setStyleSheet(f"""
            QFrame {{
                background-color: {COLOR_CARD_BG};
                border: 1px solid {COLOR_BORDER};
                border-radius: 8px;
            }}
        """)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(20, 16, 20, 16)
        card_layout.setSpacing(6)

        text = QLabel()
        text.setWordWrap(True)
        text.setStyleSheet(
            f"color: {COLOR_TEXT_SECONDARY}; font-size: 13px; background: transparent; line-height: 1.5;"
        )
        card_layout.addWidget(text)

        setattr(card, "_text_key", text_key)
        setattr(card, "_text_label", text)
        return card

    # ---- Page 3: Export confirmation ----
    def _build_export_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 40, 24, 40)
        layout.setSpacing(20)

        # Success icon
        icon = QLabel("\u2714")
        icon.setFixedSize(72, 72)
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon.setStyleSheet(f"""
            background-color: {COLOR_BTN_SUCCESS};
            color: white;
            border-radius: 36px;
            font-size: 36px;
            font-weight: bold;
        """)
        layout.addWidget(icon, 0, Qt.AlignmentFlag.AlignHCenter)

        self._export_title = QLabel()
        self._export_title.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        self._export_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._export_title.setStyleSheet(f"color: {COLOR_TEXT_PRIMARY}; background: transparent;")
        layout.addWidget(self._export_title)

        self._export_desc = QLabel()
        self._export_desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._export_desc.setWordWrap(True)
        self._export_desc.setMaximumWidth(500)
        self._export_desc.setStyleSheet(
            f"color: {COLOR_TEXT_SECONDARY}; font-size: 14px; background: transparent;"
        )
        layout.addWidget(self._export_desc, 0, Qt.AlignmentFlag.AlignHCenter)

        layout.addStretch(1)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)

        self._export_back_btn = QPushButton()
        self._export_back_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._export_back_btn.setMinimumHeight(44)
        self._export_back_btn.setStyleSheet(f"""
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
        self._export_back_btn.clicked.connect(lambda: self._pages.setCurrentIndex(PAGE_METRICS))
        btn_row.addWidget(self._export_back_btn)

        self._export_confirm_btn = QPushButton()
        self._export_confirm_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._export_confirm_btn.setMinimumHeight(44)
        self._export_confirm_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLOR_BTN_SUCCESS};
                color: {COLOR_TEXT_PRIMARY};
                border: none;
                border-radius: 6px;
                font-size: 14px;
                font-weight: bold;
                padding: 0 32px;
            }}
            QPushButton:hover {{
                background-color: {COLOR_BTN_SUCCESS_HOVER};
            }}
        """)
        self._export_confirm_btn.clicked.connect(self.export_clicked.emit)
        btn_row.addWidget(self._export_confirm_btn)

        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        return page

    def _build_metric_card(self, metric_type: str) -> QFrame:
        card = QFrame()
        card.setStyleSheet(f"""
            QFrame {{
                background-color: {COLOR_CARD_BG};
                border: 1px solid {COLOR_BORDER};
                border-radius: 8px;
            }}
        """)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(4)

        value_label = QLabel("0")
        value_label.setFont(QFont("Segoe UI", 22, QFont.Weight.Bold))
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

    # ---- Navigation ----
    def _show_fixes_page(self):
        self._pages.setCurrentIndex(PAGE_FIXES)
        self.fix_issues_clicked.emit()

    def _show_export_page(self):
        self._pages.setCurrentIndex(PAGE_EXPORT)

    # ---- UI Updates ----
    def _update_metrics_display(self):
        t = self._locale.t
        m = self._metrics

        confident_value = self._metric_confident.findChild(QLabel, "value_confident")
        confident_name = self._metric_confident.findChild(QLabel, "name_confident")
        if confident_value:
            confident_value.setText(str(m["confident_frames"]))
            confident_value.setStyleSheet(f"color: {COLOR_METRIC_GOOD}; font-size: 22px; font-weight: bold; background: transparent;")
        if confident_name:
            confident_name.setText(t("metric_confident_frames"))

        low_value = self._metric_low.findChild(QLabel, "value_low")
        low_name = self._metric_low.findChild(QLabel, "name_low")
        if low_value:
            low_value.setText(str(m["low_confidence_frames"]))
            low_color = COLOR_METRIC_WARN if m["low_confidence_frames"] > 0 else COLOR_METRIC_GOOD
            low_value.setStyleSheet(f"color: {low_color}; font-size: 22px; font-weight: bold; background: transparent;")
        if low_name:
            low_name.setText(t("metric_low_confidence_frames"))

        swaps_value = self._metric_swaps.findChild(QLabel, "value_swaps")
        swaps_name = self._metric_swaps.findChild(QLabel, "name_swaps")
        if swaps_value:
            swaps_value.setText(str(m["identity_swaps"]))
            swaps_color = COLOR_METRIC_BAD if m["identity_swaps"] > 0 else COLOR_METRIC_GOOD
            swaps_value.setStyleSheet(f"color: {swaps_color}; font-size: 22px; font-weight: bold; background: transparent;")
        if swaps_name:
            swaps_name.setText(t("metric_identity_swaps"))

        total_value = self._metric_total.findChild(QLabel, "value_total")
        total_name = self._metric_total.findChild(QLabel, "name_total")
        if total_value:
            total_value.setText(str(m["total_frames"]))
            total_value.setStyleSheet(f"color: {COLOR_TEXT_PRIMARY}; font-size: 22px; font-weight: bold; background: transparent;")
        if total_name:
            total_name.setText(t("metric_total_frames"))

        self._update_quality_badge()

    def _update_quality_badge(self):
        t = self._locale.t
        m = self._metrics
        total = m["total_frames"]
        if total == 0:
            quality = "fair"
            label = t("quality_fair")
            color = COLOR_METRIC_WARN
        else:
            swap_rate = m["identity_swaps"] / total
            low_rate = m["low_confidence_frames"] / total
            if swap_rate == 0 and low_rate < 0.05:
                quality = "excellent"
                label = t("quality_excellent")
                color = COLOR_METRIC_GOOD
            elif swap_rate < 0.01 and low_rate < 0.15:
                quality = "good"
                label = t("quality_good")
                color = COLOR_METRIC_GOOD
            elif swap_rate < 0.05:
                quality = "fair"
                label = t("quality_fair")
                color = COLOR_METRIC_WARN
            else:
                quality = "poor"
                label = t("quality_poor")
                color = COLOR_METRIC_BAD

        self._quality_badge.setText(f"{t('quality_label')}: {label}")
        self._quality_badge.setStyleSheet(f"""
            color: {COLOR_TEXT_PRIMARY};
            font-size: 13px; font-weight: bold;
            padding: 4px 12px;
            border-radius: 4px;
            background-color: {color};
        """)

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
            text = f"[{severity_icon}] Frame {problem['frame']} ({problem.get('timestamp', '')}): {problem['description']}"
            item = QListWidgetItem(text)
            if problem.get("severity") == "error":
                item.setForeground(QColor(COLOR_METRIC_BAD))
            else:
                item.setForeground(QColor(COLOR_METRIC_WARN))
            self._problems_list.addItem(item)

    def _update_fix_suggestions(self):
        t = self._locale.t
        m = self._metrics
        total = m["total_frames"]

        has_swaps = m["identity_swaps"] > 0
        has_low = m["low_confidence_frames"] > 0

        self._fix_swap_card.setVisible(has_swaps)
        self._fix_low_card.setVisible(has_low)
        self._fix_gap_card.setVisible(False)
        self._fix_general_card.setVisible(True)

        for card in [self._fix_swap_card, self._fix_low_card, self._fix_gap_card, self._fix_general_card]:
            text_key = getattr(card, "_text_key", None)
            text_label = getattr(card, "_text_label", None)
            if text_key and text_label:
                text_label.setText(t(text_key))

    def _refresh_text(self):
        t = self._locale.t
        self._title_label.setText(t("diagnostics_title"))
        self._problems_title.setText(t("problems_title"))
        self._back_btn.setText(t("btn_back_to_feedback"))
        self._reprocess_btn.setText(t("btn_reprocess"))
        self._fix_btn.setText(t("btn_fix_issues"))
        self._export_btn.setText(t("btn_export"))
        self._fix_back_btn.setText(t("btn_back"))
        self._fix_title.setText(t("fix_title"))
        self._export_title.setText(t("btn_export"))
        self._export_desc.setText(t("tooltip_export"))
        self._export_back_btn.setText(t("btn_back"))
        self._export_confirm_btn.setText(t("btn_export"))
        self._update_metrics_display()
        self._update_problems_list()
        self._update_fix_suggestions()
