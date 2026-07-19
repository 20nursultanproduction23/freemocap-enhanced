"""
Tests for GUI screens: i18n, Screen 1-5.

Tests verify:
  - i18n: language loading, translation, language switching, signal emission
  - Screen 1: mode selection, language switching, start button state
  - Screen 2: back/continue signals, text refresh
  - Screen 3: actor count, contact checkbox, advanced toggle, restore defaults, settings dict
  - Screen 4: confidence display, identity swap warning, bbox overlay
  - Screen 5: metrics display, problems list, empty state, button signals
"""
import sys
import os

import pytest

# Ensure gui package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from gui.i18n.locale_manager import LocaleManager, AVAILABLE_LANGUAGES

# Import all screens
from gui.screens.actor_mode_selector import ActorModeSelector
from gui.screens.single_actor_settings import SingleActorSettings
from gui.screens.dual_actor_settings import DualActorSettings
from gui.screens.live_feedback_widget import LiveFeedbackWidget
from gui.screens.diagnostics_widget import DiagnosticsWidget


# ---- Fixtures ----

@pytest.fixture(scope="session")
def qapp():
    """Create QApplication for all tests."""
    if not QApplication.instance():
        return QApplication(sys.argv)
    return QApplication.instance()


@pytest.fixture
def locale_en():
    return LocaleManager("en")


@pytest.fixture
def locale_ru():
    return LocaleManager("ru")


# ==========================================
# i18n Tests
# ==========================================

class TestLocaleManager:

    def test_load_english(self, locale_en):
        assert locale_en.language == "en"
        assert locale_en.t("single_actor_tile") == "Single Actor"

    def test_load_russian(self, locale_ru):
        assert locale_ru.language == "ru"
        assert locale_ru.t("single_actor_tile") == "Один актёр"

    def test_missing_key_returns_placeholder(self, locale_en):
        assert locale_en.t("nonexistent_key_xyz") == "[MISSING:nonexistent_key_xyz]"

    def test_interpolation(self, locale_en):
        result = locale_en.t("actor_label", id=1)
        assert result == "Actor 1"

    def test_interpolation_ru(self, locale_ru):
        result = locale_ru.t("actor_label", id=2)
        assert result == "Актёр 2"

    def test_switch_language(self, locale_en):
        locale_en.set_language("ru")
        assert locale_en.language == "ru"
        assert locale_en.t("single_actor_tile") == "Один актёр"
        locale_en.set_language("en")  # restore

    def test_switch_same_language_noop(self, locale_en):
        locale_en.set_language("en")
        assert locale_en.language == "en"

    def test_invalid_language_raises(self, locale_en):
        with pytest.raises(ValueError, match="Unsupported language"):
            locale_en.set_language("fr")

    def test_available_languages(self, locale_en):
        langs = locale_en.available_languages()
        assert "en" in langs
        assert "ru" in langs

    def test_all_string_keys_both_languages(self):
        en = LocaleManager("en")
        ru = LocaleManager("ru")
        en_keys = set(en._strings.keys())
        ru_keys = set(ru._strings.keys())
        assert en_keys == ru_keys, f"Mismatch: {en_keys.symmetric_difference(ru_keys)}"

    def test_no_empty_values_ru(self):
        ru = LocaleManager("ru")
        for key, val in ru._strings.items():
            if key in ("single_actor_badge", "single_actor_info"):
                continue  # intentionally empty
            assert val, f"Empty value for key '{key}' in Russian strings"


# ==========================================
# Screen 1: ActorModeSelector
# ==========================================

class TestActorModeSelector:

    def _make(self, locale_en):
        return ActorModeSelector(locale_en)

    def test_instantiation(self, qapp, locale_en):
        w = self._make(locale_en)
        assert w is not None

    def test_initial_state_no_selection(self, qapp, locale_en):
        w = self._make(locale_en)
        assert w._selected_mode is None

    def test_start_button_disabled_initially(self, qapp, locale_en):
        w = self._make(locale_en)
        assert not w._start_btn.isEnabled()

    def test_select_single_mode(self, qapp, locale_en):
        w = self._make(locale_en)
        w._on_tile_clicked("single")
        assert w._selected_mode == "single"
        assert w._start_btn.isEnabled()

    def test_select_dual_mode(self, qapp, locale_en):
        w = self._make(locale_en)
        w._on_tile_clicked("dual")
        assert w._selected_mode == "dual"

    def test_mode_selected_signal(self, qapp, locale_en):
        w = self._make(locale_en)
        received = []
        w.mode_selected.connect(lambda m: received.append(m))
        w._on_tile_clicked("single")
        w._on_start_clicked()
        assert received == ["single"]

    def test_start_no_selection_no_signal(self, qapp, locale_en):
        w = self._make(locale_en)
        received = []
        w.mode_selected.connect(lambda m: received.append(m))
        w._on_start_clicked()
        assert received == []

    def test_language_switch_refreshes_text(self, qapp, locale_en):
        w = self._make(locale_en)
        assert w._title_label.text() == "Capture Mode"
        w._switch_language("ru")
        assert w._title_label.text() == "Режим захвата"
        w._switch_language("en")  # restore

    def test_language_changed_signal(self, qapp, locale_en):
        w = self._make(locale_en)
        received = []
        w.language_changed.connect(lambda lang: received.append(lang))
        w._switch_language("ru")
        assert received == ["ru"]
        w._switch_language("en")  # restore

    def test_dual_tile_has_beta_badge(self, qapp, locale_en):
        w = self._make(locale_en)
        badges = w._dual_tile.findChildren(type(w._title_label))
        # Badge is a QLabel with "BETA" text
        found_badge = False
        for lbl in w._dual_tile.findChildren(type(w._title_label)):
            if lbl.text() == "BETA":
                found_badge = True
                break
        assert found_badge, "BETA badge not found on dual tile"


# ==========================================
# Screen 2: SingleActorSettings
# ==========================================

class TestSingleActorSettings:

    def _make(self, locale_en):
        return SingleActorSettings(locale_en)

    def test_instantiation(self, qapp, locale_en):
        w = self._make(locale_en)
        assert w is not None

    def test_title_text(self, qapp, locale_en):
        w = self._make(locale_en)
        assert w._title_label.text() == "Single Actor Settings"

    def test_continue_signal(self, qapp, locale_en):
        w = self._make(locale_en)
        received = []
        w.continue_clicked.connect(lambda: received.append(True))
        w.continue_clicked.emit()
        assert len(received) == 1

    def test_back_signal(self, qapp, locale_en):
        w = self._make(locale_en)
        received = []
        w.back_clicked.connect(lambda: received.append(True))
        w.back_clicked.emit()
        assert len(received) >= 1

    def test_settings_property(self, qapp, locale_en):
        w = self._make(locale_en)
        s = w.settings
        assert s == {"mode": "single"}

    def test_language_switch(self, qapp, locale_en):
        w = self._make(locale_en)
        assert "Settings" in w._title_label.text()
        w._locale.set_language("ru")
        assert "Настройки" in w._title_label.text()
        w._locale.set_language("en")  # restore


# ==========================================
# Screen 3: DualActorSettings
# ==========================================

class TestDualActorSettings:

    def _make(self, locale_en):
        return DualActorSettings(locale_en)

    def test_instantiation(self, qapp, locale_en):
        w = self._make(locale_en)
        assert w is not None

    def test_default_settings(self, qapp, locale_en):
        w = self._make(locale_en)
        s = w.settings
        assert s["actor_count"] == 2
        assert s["contact_expected"] is False
        assert s["shared_floor"] is True
        assert 0.0 <= s["association_threshold"] <= 1.0
        assert 0.0 <= s["tracking_threshold"] <= 1.0

    def test_set_actor_count(self, qapp, locale_en):
        w = self._make(locale_en)
        received = []
        w.settings_changed.connect(lambda s: received.append(s))
        w._set_actor_count(2)
        assert w._actor_count == 2
        assert len(received) >= 1

    def test_contact_checkbox(self, qapp, locale_en):
        w = self._make(locale_en)
        assert w._contact_expected is False
        assert w._contact_warning.isHidden()
        # Simulate checking via the internal handler
        w._on_contact_changed(Qt.CheckState.Checked.value)
        assert w._contact_expected is True
        assert not w._contact_warning.isHidden()

    def test_contact_uncheck_hides_warning(self, qapp, locale_en):
        w = self._make(locale_en)
        w._on_contact_changed(Qt.CheckState.Checked.value)
        assert not w._contact_warning.isHidden()
        w._on_contact_changed(Qt.CheckState.Unchecked.value)
        assert w._contact_warning.isHidden()

    def test_advanced_toggle(self, qapp, locale_en):
        w = self._make(locale_en)
        assert not w._advanced_visible
        assert w._advanced_container.isHidden()
        w._toggle_advanced()
        assert w._advanced_visible
        assert not w._advanced_container.isHidden()
        w._toggle_advanced()
        assert not w._advanced_visible
        assert w._advanced_container.isHidden()

    def test_association_slider(self, qapp, locale_en):
        w = self._make(locale_en)
        w._assoc_slider.setValue(75)
        assert abs(w._association_threshold - 0.75) < 0.01

    def test_tracking_slider(self, qapp, locale_en):
        w = self._make(locale_en)
        w._track_slider.setValue(30)
        assert abs(w._tracking_threshold - 0.30) < 0.01

    def test_shared_floor_checkbox(self, qapp, locale_en):
        w = self._make(locale_en)
        assert w._shared_floor is True
        w._floor_cb.setChecked(False)
        assert w._shared_floor is False

    def test_restore_defaults(self, qapp, locale_en):
        w = self._make(locale_en)
        w._assoc_slider.setValue(90)
        w._track_slider.setValue(10)
        w._floor_cb.setChecked(False)
        w._contact_cb.setChecked(True)
        w._restore_defaults()
        assert w.settings["actor_count"] == 2
        assert w.settings["contact_expected"] is False
        assert w.settings["shared_floor"] is True
        assert abs(w.settings["association_threshold"] - 0.5) < 0.01
        assert abs(w.settings["tracking_threshold"] - 0.5) < 0.01

    def test_settings_changed_signal(self, qapp, locale_en):
        w = self._make(locale_en)
        received = []
        w.settings_changed.connect(lambda s: received.append(s))
        w._set_actor_count(2)
        assert len(received) == 1
        assert received[0]["actor_count"] == 2

    def test_language_switch(self, qapp, locale_en):
        w = self._make(locale_en)
        assert "Settings" in w._title_label.text() or "Настройки" in w._title_label.text()
        w._locale.set_language("ru")
        assert "Настройки" in w._title_label.text()
        w._locale.set_language("en")  # restore


# ==========================================
# Screen 4: LiveFeedbackWidget
# ==========================================

class TestLiveFeedbackWidget:

    def _make(self, locale_en):
        return LiveFeedbackWidget(locale_en)

    def test_instantiation(self, qapp, locale_en):
        w = self._make(locale_en)
        assert w is not None

    def test_default_confidence(self, qapp, locale_en):
        w = self._make(locale_en)
        assert w._confidence_level == "medium"

    def test_set_confidence_high(self, qapp, locale_en):
        w = self._make(locale_en)
        w.set_confidence("high")
        assert w._confidence_level == "high"
        assert "High" in w._conf_text.text()

    def test_set_confidence_low(self, qapp, locale_en):
        w = self._make(locale_en)
        w.set_confidence("low")
        assert "Low" in w._conf_text.text()

    def test_set_confidence_invalid_raises(self, qapp, locale_en):
        w = self._make(locale_en)
        with pytest.raises(ValueError, match="Invalid confidence"):
            w.set_confidence("ultra")

    def test_identity_swap_show_hide(self, qapp, locale_en):
        w = self._make(locale_en)
        assert not w._identity_swap_detected
        assert w._swap_warning_frame.isHidden()
        w.show_identity_swap()
        assert w._identity_swap_detected
        assert not w._swap_warning_frame.isHidden()
        w.hide_identity_swap()
        assert not w._identity_swap_detected
        assert w._swap_warning_frame.isHidden()

    def test_set_bboxes(self, qapp, locale_en):
        w = self._make(locale_en)
        w.set_bboxes(1, [(10, 20, 100, 200)])
        w.set_bboxes(2, [(30, 40, 150, 250)])

    def test_set_frame_count(self, qapp, locale_en):
        w = self._make(locale_en)
        w.set_frame_count(42)
        assert w._frame_count == 42
        assert "42" in w._frame_label.text()

    def test_stop_signal(self, qapp, locale_en):
        w = self._make(locale_en)
        received = []
        w.stop_clicked.connect(lambda: received.append(True))
        w.stop_clicked.emit()
        assert len(received) == 1

    def test_recording_indicator_visible(self, qapp, locale_en):
        w = self._make(locale_en)
        assert not w._recording_text.isHidden()

    def test_stop_button_visible(self, qapp, locale_en):
        w = self._make(locale_en)
        assert not w._stop_btn.isHidden()
        assert w._stop_btn.isEnabled()

    def test_swap_detail_hidden_initially(self, qapp, locale_en):
        w = self._make(locale_en)
        assert w._swap_warning_frame.isHidden()

    def test_swap_detail_visible_after_show(self, qapp, locale_en):
        w = self._make(locale_en)
        w.show_identity_swap()
        assert not w._swap_warning_frame.isHidden()

    def test_language_switch(self, qapp, locale_en):
        w = self._make(locale_en)
        en_title = w._title_label.text()
        w._locale.set_language("ru")
        ru_title = w._title_label.text()
        assert en_title != ru_title
        assert ru_title == "Запись в процессе"
        w._locale.set_language("en")  # restore

    def test_actor_labels_in_previews(self, qapp, locale_en):
        w = self._make(locale_en)
        assert "Actor 1" in w._preview_header_actor1.text()
        assert "Actor 2" in w._preview_header_actor2.text()


# ==========================================
# Screen 5: DiagnosticsWidget
# ==========================================

class TestDiagnosticsWidget:

    def _make(self, locale_en):
        return DiagnosticsWidget(locale_en)

    def test_instantiation(self, qapp, locale_en):
        w = self._make(locale_en)
        assert w is not None

    def test_default_metrics(self, qapp, locale_en):
        w = self._make(locale_en)
        m = w._metrics
        assert m["confident_frames"] == 0
        assert m["low_confidence_frames"] == 0
        assert m["identity_swaps"] == 0
        assert m["total_frames"] == 0

    def test_set_metrics(self, qapp, locale_en):
        w = self._make(locale_en)
        w.set_metrics(confident=85, low_confidence=10, identity_swaps=3, total=100)
        assert w._metrics["confident_frames"] == 85
        assert w._metrics["low_confidence_frames"] == 10
        assert w._metrics["identity_swaps"] == 3
        assert w._metrics["total_frames"] == 100

    def test_set_problems(self, qapp, locale_en):
        w = self._make(locale_en)
        problems = [
            {"frame": 5, "timestamp": "00:00.2", "description": "Low confidence", "severity": "warning", "type": "low_confidence"},
            {"frame": 12, "timestamp": "00:00.5", "description": "Identity swap", "severity": "error", "type": "identity_swap"},
        ]
        w.set_problems(problems)
        assert len(w._problems) == 2
        assert w._problems_list.count() == 2

    def test_empty_problems_shows_message(self, qapp, locale_en):
        w = self._make(locale_en)
        w.set_problems([])
        assert w._problems_list.count() == 0
        assert not w._problems_empty_label.isHidden()
        assert "No problems" in w._problems_empty_label.text()

    def test_export_signal(self, qapp, locale_en):
        w = self._make(locale_en)
        received = []
        w.export_clicked.connect(lambda: received.append(True))
        w.export_clicked.emit()
        assert len(received) >= 1

    def test_fix_issues_signal(self, qapp, locale_en):
        w = self._make(locale_en)
        received = []
        w.fix_issues_clicked.connect(lambda: received.append(True))
        w.fix_issues_clicked.emit()
        assert len(received) >= 1

    def test_reprocess_signal(self, qapp, locale_en):
        w = self._make(locale_en)
        received = []
        w.reprocess_clicked.connect(lambda: received.append(True))
        w.reprocess_clicked.emit()
        assert len(received) >= 1

    def test_back_to_feedback_signal(self, qapp, locale_en):
        w = self._make(locale_en)
        received = []
        w.back_to_feedback_clicked.connect(lambda: received.append(True))
        w.back_to_feedback_clicked.emit()
        assert len(received) == 1

    def test_language_switch(self, qapp, locale_en):
        w = self._make(locale_en)
        assert "Diagnostics" in w._title_label.text()
        w._locale.set_language("ru")
        assert "Диагностика" in w._title_label.text()
        w._locale.set_language("en")  # restore

    def test_metrics_display_update(self, qapp, locale_en):
        w = self._make(locale_en)
        w.set_metrics(confident=50, low_confidence=5, identity_swaps=1, total=60)
        confident_val = w._metric_confident.findChild(type(w._title_label), "value_confident")
        assert confident_val.text() == "50"

    def test_quality_badge_shows_excellent(self, qapp, locale_en):
        w = self._make(locale_en)
        w.set_metrics(confident=100, low_confidence=0, identity_swaps=0, total=100)
        assert "Excellent" in w._quality_badge.text()

    def test_quality_badge_shows_poor(self, qapp, locale_en):
        w = self._make(locale_en)
        w.set_metrics(confident=20, low_confidence=40, identity_swaps=40, total=100)
        assert "Poor" in w._quality_badge.text()

    def test_fix_page_navigation(self, qapp, locale_en):
        w = self._make(locale_en)
        w._show_fixes_page()
        assert w._pages.currentIndex() == 1

    def test_export_page_navigation(self, qapp, locale_en):
        w = self._make(locale_en)
        w._show_export_page()
        assert w._pages.currentIndex() == 2

    def test_reset_returns_to_metrics(self, qapp, locale_en):
        w = self._make(locale_en)
        w._show_fixes_page()
        assert w._pages.currentIndex() == 1
        w.reset()
        assert w._pages.currentIndex() == 0

    def test_fix_swap_card_hidden_when_no_swaps(self, qapp, locale_en):
        w = self._make(locale_en)
        w.set_problems([])
        assert w._fix_swap_card.isHidden()

    def test_fix_swap_card_visible_when_swaps(self, qapp, locale_en):
        w = self._make(locale_en)
        w.set_metrics(confident=50, low_confidence=0, identity_swaps=5, total=100)
        assert not w._fix_swap_card.isHidden()


# ==========================================
# Summary
# ==========================================

def print_summary():
    print("\n" + "=" * 60)
    print("GUI SCREENS TEST SUITE — SUMMARY")
    print("=" * 60)
    print("  i18n (LocaleManager): 11 tests")
    print("  Screen 1 (ActorModeSelector): 10 tests")
    print("  Screen 2 (SingleActorSettings): 5 tests")
    print("  Screen 3 (DualActorSettings): 12 tests")
    print("  Screen 4 (LiveFeedbackWidget): 15 tests")
    print("  Screen 5 (DiagnosticsWidget): 19 tests")
    print("  " + "-" * 50)
    print("  Total: 72 tests")
    print("=" * 60)


if __name__ == "__main__":
    print_summary()
