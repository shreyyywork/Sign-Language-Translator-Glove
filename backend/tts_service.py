"""
Text-to-Speech (TTS) Service for ISL Glove.
Provides modular multilingual speech synthesis:
  - English (default)
  - Hindi (हिंदी)
Includes phrase builder for concatenating consecutive sign gestures into fluent sentences.
"""

import os
import json
from typing import Dict, Any, Optional, List


class TTSService:
    def __init__(self, gestures_config_path: str = "config/gestures.json"):
        self.gestures_config_path = gestures_config_path
        self.gesture_map: Dict[str, Dict[str, str]] = {}
        self.sentence_buffer: List[str] = []
        self._load_translations()

    def _load_translations(self):
        """Load vocabulary text and Hindi mappings from config."""
        if os.path.exists(self.gestures_config_path):
            with open(self.gestures_config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for g in data.get("gestures", []):
                    self.gesture_map[g["id"]] = {
                        "en": g.get("name", g["id"]),
                        "hi": g.get("hindi", g["id"]),
                        "description": g.get("description", "")
                    }

    def get_speech_text(self, gesture_id: str, language: str = "en") -> str:
        """Get translated utterance string for a gesture."""
        lang = language.lower()
        if gesture_id in self.gesture_map:
            entry = self.gesture_map[gesture_id]
            if lang.startswith("hi"):
                return entry.get("hi", entry.get("en", gesture_id))
            return entry.get("en", gesture_id)
        # Fallback to readable string
        return gesture_id.replace("_", " ").title()

    def add_to_sentence(self, gesture_id: str) -> List[str]:
        """Add recognized gesture to current sentence buffer."""
        text = self.get_speech_text(gesture_id, "en")
        # Avoid duplicate consecutive additions
        if not self.sentence_buffer or self.sentence_buffer[-1] != text:
            self.sentence_buffer.append(text)
        return self.sentence_buffer

    def clear_sentence(self) -> List[str]:
        """Clear sentence buffer."""
        self.sentence_buffer.clear()
        return self.sentence_buffer

    def get_full_sentence(self, language: str = "en") -> str:
        """Construct a natural sentence from buffered gestures."""
        if not self.sentence_buffer:
            return ""
        return " ".join(self.sentence_buffer)
