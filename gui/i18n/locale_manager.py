"""
LocaleManager — i18n manager for dual-actor GUI screens.

Loads string bundles from JSON files, provides t(key) translation,
emits signal on language change so widgets can refresh.

Architecture (per decision: option A — new screens only):
  - Only new dual-actor screens use this manager
  - Existing FreeMoCap UI remains English-only (no changes)
  - JSON files: gui/i18n/strings_en.json, gui/i18n/strings_ru.json

Usage:
    locale = LocaleManager()
    print(locale.t("single_actor_tile"))  # "Single Actor" or "Один актёр"
    locale.set_language("ru")
    print(locale.t("single_actor_tile"))  # "Один актёр"
"""
import json
import os
from typing import Optional
from PySide6.QtCore import QObject, Signal


AVAILABLE_LANGUAGES = ["en", "ru"]
DEFAULT_LANGUAGE = "en"

_i18n_dir = os.path.dirname(os.path.abspath(__file__))


class LocaleManager(QObject):
    """Manages UI string localization for dual-actor screens.

    Signals:
        language_changed(str): emitted when language is switched.
            Connected widgets should call t() for their strings and update.
    """

    language_changed = Signal(str)

    def __init__(self, language: str = DEFAULT_LANGUAGE, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._strings = {}
        self._language = language
        self._load(language)

    def _load(self, lang: str):
        """Load string bundle for given language."""
        path = os.path.join(_i18n_dir, f"strings_{lang}.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"String bundle not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            self._strings = json.load(f)

    @property
    def language(self) -> str:
        return self._language

    def set_language(self, lang: str):
        """Switch language and emit signal for widgets to refresh."""
        if lang not in AVAILABLE_LANGUAGES:
            raise ValueError(f"Unsupported language: {lang}. Available: {AVAILABLE_LANGUAGES}")
        if lang == self._language:
            return
        self._load(lang)
        self._language = lang
        self.language_changed.emit(lang)

    def t(self, key: str, **kwargs) -> str:
        """Translate a string key with optional interpolation.

        Args:
            key: string key from JSON bundle (e.g. "single_actor_tile")
            **kwargs: format parameters (e.g. t("actor_label", id=1) -> "Actor 1")

        Returns:
            Translated string. If key not found, returns the key itself.
        """
        template = self._strings.get(key)
        if template is None:
            return f"[MISSING:{key}]"
        if kwargs:
            try:
                return template.format(**kwargs)
            except (KeyError, IndexError):
                return template
        return template

    def available_languages(self) -> list:
        return AVAILABLE_LANGUAGES.copy()
