"""Quick visual test for Screen 4 and Screen 5 widgets."""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget
from PySide6.QtCore import QTimer

from gui.i18n.locale_manager import LocaleManager
from gui.screens.live_feedback_widget import LiveFeedbackWidget
from gui.screens.diagnostics_widget import DiagnosticsWidget


def main():
    app = QApplication(sys.argv)

    locale = LocaleManager("ru")

    window = QMainWindow()
    window.setWindowTitle("Screen 4-5 Visual Test")
    window.setMinimumSize(900, 600)

    tabs = QTabWidget()

    # Screen 4
    feedback = LiveFeedbackWidget(locale)

    # Screen 5
    diagnostics = DiagnosticsWidget(locale)
    diagnostics.set_metrics(confident=85, low_confidence=10, identity_swaps=3, total=100)
    diagnostics.set_problems([
        {"frame": 12, "timestamp": "00:00.5", "description": "Identity swap Actor 1 <-> Actor 2", "severity": "error", "type": "identity_swap"},
        {"frame": 45, "timestamp": "00:01.8", "description": "Low confidence association", "severity": "warning", "type": "low_confidence"},
        {"frame": 78, "timestamp": "00:03.1", "description": "Tracking gap — 12 frames", "severity": "warning", "type": "tracking_gap"},
    ])

    tabs.addTab(feedback, "Screen 4: Live Feedback")
    tabs.addTab(diagnostics, "Screen 5: Diagnostics")

    window.setCentralWidget(tabs)
    window.show()

    # Simulate some activity after 2 seconds
    def simulate():
        feedback.set_frame_count(42)
        feedback.set_confidence("low")
        feedback.show_identity_swap()
        feedback.set_bboxes(1, [(50, 30, 200, 250)])
        feedback.set_bboxes(2, [(250, 40, 420, 260)])

    QTimer.singleShot(2000, simulate)

    # Tab switching test: connect diagnostics signals
    def on_export():
        print("EXPORT CLICKED")
    def on_fix():
        tabs.setCurrentIndex(1)  # stay on diagnostics
    def on_reprocess():
        print("REPROCESS CLICKED")
    def on_back():
        tabs.setCurrentIndex(0)

    diagnostics.export_clicked.connect(on_export)
    diagnostics.fix_issues_clicked.connect(on_fix)
    diagnostics.reprocess_clicked.connect(on_reprocess)
    diagnostics.back_to_feedback_clicked.connect(on_back)
    feedback.stop_clicked.connect(lambda: tabs.setCurrentIndex(1))

    app.exec()


if __name__ == "__main__":
    main()
