"""
Tests for Stage 7.1 — ArUco marker UI in DualActorSettings

Tests:
1. Toggle OFF: marker section hidden, aruco_fallback_enabled=False in settings
2. Toggle ON: marker section visible, aruco_fallback_enabled=True in settings
3. Generate button calls generate_actor_markers with correct params (mocked)
4. Download copies files to Downloads (mocked)
5. Print opens QPrintDialog (mocked)
6. Restore defaults resets toggle to OFF
7. Settings dict includes aruco_fallback_enabled key
8. i18n strings exist for all new keys
"""
import pytest
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, os.path.dirname(__file__))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

app = QApplication.instance()
if app is None:
    app = QApplication(sys.argv)


def _make_settings_widget():
    """Create a DualActorSettings widget with mocked locale."""
    from gui.screens.dual_actor_settings import DualActorSettings
    from gui.i18n.locale_manager import LocaleManager

    locale = MagicMock(spec=LocaleManager)
    locale.t = lambda key: key

    widget = DualActorSettings(locale=locale)
    return widget


# ── Toggle behavior ────────────────────────────────────────────────────

class TestToggle:
    def test_default_state_off(self):
        """Toggle is OFF by default."""
        w = _make_settings_widget()
        assert w._aruco_fallback_enabled is False
        assert w.settings["aruco_fallback_enabled"] is False

    def test_toggle_on_makes_marker_container_visible(self):
        """Turning toggle ON makes marker section visible."""
        w = _make_settings_widget()
        assert w._marker_container.isHidden() is True

        w._aruco_cb.setChecked(True)
        assert w._marker_container.isHidden() is False
        assert w.settings["aruco_fallback_enabled"] is True

    def test_toggle_off_hides_marker_container(self):
        """Turning toggle OFF hides marker section."""
        w = _make_settings_widget()
        w._aruco_cb.setChecked(True)
        assert w._marker_container.isHidden() is False

        w._aruco_cb.setChecked(False)
        assert w._marker_container.isHidden() is True
        assert w.settings["aruco_fallback_enabled"] is False

    def test_toggle_emits_settings_changed(self):
        """Toggle emits settings_changed signal."""
        w = _make_settings_widget()
        mock_slot = MagicMock()
        w.settings_changed.connect(mock_slot)

        w._aruco_cb.setChecked(True)
        mock_slot.assert_called()
        emitted = mock_slot.call_args[0][0]
        assert emitted["aruco_fallback_enabled"] is True


# ── Generate button ────────────────────────────────────────────────────

class TestGenerate:
    def test_generate_calls_backend(self):
        """Generate button calls generate_all_markers with correct config."""
        w = _make_settings_widget()

        mock_generated = {
            "actor_0": {"marker_id": 0, "filepath": "/tmp/actor_0_50mm.png",
                        "filename": "actor_0_50mm.png", "size_mm": 50, "image_shape": (200, 200)},
            "actor_1": {"marker_id": 1, "filepath": "/tmp/actor_1_50mm.png",
                        "filename": "actor_1_50mm.png", "size_mm": 50, "image_shape": (200, 200)},
        }

        mock_mod = MagicMock()
        mock_mod.generate_all_markers.return_value = mock_generated
        mock_mod.load_config.return_value = {"actors": {}}

        with patch.dict("sys.modules", {"tools": MagicMock(), "tools.generate_actor_markers": mock_mod}):
            with patch("gui.screens.dual_actor_settings.QPixmap") as mock_qpixmap:
                mock_pixmap = MagicMock()
                mock_pixmap.isNull.return_value = False
                mock_pixmap.scaled.return_value = mock_pixmap
                mock_qpixmap.return_value = mock_pixmap

                mock_img0 = MagicMock()
                mock_img1 = MagicMock()
                mock_info0 = MagicMock()
                mock_info1 = MagicMock()
                w._preview_actor0["image"] = mock_img0
                w._preview_actor0["info"] = mock_info0
                w._preview_actor1["image"] = mock_img1
                w._preview_actor1["info"] = mock_info1

                w._on_generate_markers()

                mock_mod.generate_all_markers.assert_called_once()
                mock_img0.setPixmap.assert_called_once()
                mock_img1.setPixmap.assert_called_once()
                assert w._download_btn.isEnabled() is True
                assert w._print_btn.isEnabled() is True

    def test_generate_no_backend_still_works(self):
        """If generate fails, buttons stay disabled (graceful)."""
        w = _make_settings_widget()

        mock_mod = MagicMock()
        mock_mod.generate_all_markers.side_effect = Exception("import error")

        with patch.dict("sys.modules", {"tools": MagicMock(), "tools.generate_actor_markers": mock_mod}):
            w._on_generate_markers()

        assert w._download_btn.isEnabled() is False
        assert w._print_btn.isEnabled() is False


# ── Download button ────────────────────────────────────────────────────

class TestDownload:
    def test_download_copies_files(self):
        """Download copies marker PNGs to Downloads folder."""
        w = _make_settings_widget()

        with tempfile.TemporaryDirectory() as tmpdir:
            src = os.path.join(tmpdir, "actor_0_50mm.png")
            with open(src, "wb") as f:
                f.write(b"fake png data")

            w._marker_paths = {"actor_0": src}

            downloads_mock = os.path.join(tmpdir, "Downloads")
            os.makedirs(downloads_mock)

            with patch("os.path.expanduser", return_value=tmpdir):
                w._on_download_markers()

            dest = os.path.join(downloads_mock, "actor_0_50mm.png")
            assert os.path.exists(dest)

    def test_download_empty_paths_no_crash(self):
        """Download with no marker paths doesn't crash."""
        w = _make_settings_widget()
        w._marker_paths = {}
        w._on_download_markers()


# ── Print button ───────────────────────────────────────────────────────

class TestPrint:
    def test_print_opens_dialog(self):
        """Print opens QPrintDialog."""
        w = _make_settings_widget()

        with tempfile.TemporaryDirectory() as tmpdir:
            src = os.path.join(tmpdir, "actor_0_50mm.png")
            with open(src, "wb") as f:
                f.write(b"fake png data")

            w._marker_paths = {"actor_0": src}

            with patch("gui.screens.dual_actor_settings.QPrintDialog") as mock_dialog_cls, \
                 patch("gui.screens.dual_actor_settings.QPrinter") as mock_printer_cls, \
                 patch("gui.screens.dual_actor_settings.QPixmap") as mock_qpixmap, \
                 patch("gui.screens.dual_actor_settings.QPainter") as mock_painter_cls:

                mock_dialog = MagicMock()
                mock_dialog.exec.return_value = 1  # Accepted
                mock_dialog_cls.return_value = mock_dialog

                mock_printer = MagicMock()
                mock_printer_cls.return_value = mock_printer

                mock_pixmap = MagicMock()
                mock_pixmap.isNull.return_value = False
                mock_qpixmap.return_value = mock_pixmap

                mock_painter = MagicMock()
                mock_painter_cls.return_value = mock_painter

                w._on_print_markers()

                mock_dialog.exec.assert_called_once()

    def test_print_empty_paths_no_crash(self):
        """Print with no marker paths doesn't crash."""
        w = _make_settings_widget()
        w._marker_paths = {}
        w._on_print_markers()


# ── Restore defaults ───────────────────────────────────────────────────

class TestRestoreDefaults:
    def test_restore_resets_aruco_to_off(self):
        """Restore defaults resets ArUco toggle to OFF."""
        w = _make_settings_widget()
        w._aruco_cb.setChecked(True)
        assert w._aruco_fallback_enabled is True

        w._restore_defaults()
        assert w._aruco_fallback_enabled is False
        assert w._aruco_cb.isChecked() is False
        assert w._marker_container.isHidden() is True

    def test_restore_resets_all_settings(self):
        """Restore defaults resets everything including ArUco."""
        w = _make_settings_widget()
        w._aruco_cb.setChecked(True)
        w._contact_cb.setChecked(True)

        w._restore_defaults()
        assert w.settings["aruco_fallback_enabled"] is False
        assert w.settings["contact_expected"] is False


# ── Settings dict ──────────────────────────────────────────────────────

class TestSettingsDict:
    def test_settings_includes_aruco_key(self):
        """settings dict always includes aruco_fallback_enabled."""
        w = _make_settings_widget()
        assert "aruco_fallback_enabled" in w.settings

    def test_settings_reflects_toggle_state(self):
        """settings dict reflects current toggle state."""
        w = _make_settings_widget()
        assert w.settings["aruco_fallback_enabled"] is False

        w._aruco_cb.setChecked(True)
        assert w.settings["aruco_fallback_enabled"] is True


# ── i18n strings ──────────────────────────────────────────────────────

class TestI18n:
    def test_en_strings_have_aruco_keys(self):
        """English strings file has all ArUco-related keys."""
        import json
        en_path = os.path.join(os.path.dirname(__file__), "gui", "i18n", "strings_en.json")
        with open(en_path, "r", encoding="utf-8") as f:
            strings = json.load(f)

        required_keys = [
            "aruco_fallback_label",
            "aruco_fallback_desc",
            "aruco_generate_btn",
            "aruco_download_btn",
            "aruco_print_btn",
        ]
        for key in required_keys:
            assert key in strings, f"Missing key: {key}"

    def test_ru_strings_have_aruco_keys(self):
        """Russian strings file has all ArUco-related keys."""
        import json
        ru_path = os.path.join(os.path.dirname(__file__), "gui", "i18n", "strings_ru.json")
        with open(ru_path, "r", encoding="utf-8") as f:
            strings = json.load(f)

        required_keys = [
            "aruco_fallback_label",
            "aruco_fallback_desc",
            "aruco_generate_btn",
            "aruco_download_btn",
            "aruco_print_btn",
        ]
        for key in required_keys:
            assert key in strings, f"Missing key: {key}"

    def test_en_ru_same_aruco_keys(self):
        """EN and RU have the same ArUco-related keys."""
        import json
        base = os.path.dirname(__file__)
        with open(os.path.join(base, "gui", "i18n", "strings_en.json"), "r", encoding="utf-8") as f:
            en = json.load(f)
        with open(os.path.join(base, "gui", "i18n", "strings_ru.json"), "r", encoding="utf-8") as f:
            ru = json.load(f)

        aruco_keys = [k for k in en if k.startswith("aruco_")]
        for key in aruco_keys:
            assert key in ru, f"Key '{key}' in EN but not in RU"
