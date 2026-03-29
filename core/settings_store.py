"""
This file is used to store the settings for the Jarvis assistant.

Saves all preferences and settings to ~/.jarvis/settings.json

Why we do this?
    When the user changes their city or picks a different model
    We need to save the settings so that the assistant can use them next time.

How it works?
    On startup, the settings are loaded from settings.json
    When you call settings.get("weather.city"): returns "Brampton, ON"
    When you call settings.set("weather.city", "Toronto"): saves to disk immediately
    Thread-safe: multiple threads can read/write without corrupting data

Thread safety is used to just make sure that the settings are not corrupted when multiple threads are reading/writing to the file.
"""

import json
import threading
from pathlib import Path
from typing import Any, Dict

#Default Settings

DEFAULT_SETTINGS = {
    "theme": "dark",
    "ollama_url": "http://localhost:11434/api", #Ollama URL for local host LLM inferencing
    "models":{
        "chat": "qwen3:1.7b",
        "web_agent": "qwen3-vl:4b",
    },

    "tts":{
        "voice": "bf_emma",
        "language": "b",
        "enabled": False,
    },

    "general":{
        max_history": 20,
    },

    "weather":{
        "latitude": 43.7315,
        "longitude": -79.7624,
        "city": "Brampton, ON",
    },
}

class SettingsStore:
    """
    Settings Manager that sends info to JSON
    We use dot notation for nested access like "settings.get"
    """
     def __init__(self):
        # RLock = "reentrant lock" — the same thread can lock it multiple
        # times without deadlocking. Regular Lock would freeze if the same
        # thread tried to lock it twice.
        self._lock = threading.RLock()
        
        # The actual settings dictionary (loaded from disk)
        self._settings: Dict[str, Any] = {}
        
        # Where the file lives: ~/.jarvis/settings.json
        # Path.home() returns your user folder (C:\Users\YourName on Windows)
        self._dir = Path.home() / ".jarvis"
        self._file = self._dir / "settings.json"
        
        # Load settings from disk (or create defaults)
        self._load()
    
    def _load(self):
        """Read settings from disk. If file doesn't exist, use defaults."""
        with self._lock:
            if self._file.exists():
                try:
                    with open(self._file, "r") as f:
                        loaded = json.load(f)
                    # Merge: start with defaults, then overlay what's on disk.
                    # This way, if we add new settings in an update, they get
                    # their default values automatically.
                    self._settings = self._merge(DEFAULT_SETTINGS.copy(), loaded)
                except Exception:
                    # File is corrupted? Just use defaults.
                    self._settings = DEFAULT_SETTINGS.copy()
            else:
                self._settings = DEFAULT_SETTINGS.copy()
                self._save()  # Create the file for the first time
    
    def _save(self):
        """Write current settings to disk."""
        with self._lock:
            # Create ~/.jarvis/ folder if it doesn't exist
            # parents=True means "create parent folders too"
            # exist_ok=True means "don't crash if folder already exists"
            self._dir.mkdir(parents=True, exist_ok=True)
            with open(self._file, "w") as f:
                # indent=2 makes the JSON human-readable (not one giant line)
                json.dump(self._settings, f, indent=2)
    
    def _merge(self, base: dict, override: dict) -> dict:
        """
        Deep merge two dictionaries.
        
        Example:
            base     = {"weather": {"city": "NYC", "units": "F"}}
            override = {"weather": {"city": "Toronto"}}
            result   = {"weather": {"city": "Toronto", "units": "F"}}
        
        The override replaces matching keys but doesn't delete keys
        that only exist in base. This is how new default settings
        survive when an old settings file is loaded.
        """
        result = base.copy()
        for key, value in override.items():
            # If both base and override have a dict for this key,
            # merge them recursively (go deeper)
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge(result[key], value)
            else:
                # Otherwise, override wins
                result[key] = value
        return result
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get a setting using dot notation.
        
        settings.get("weather.city")     → "Brampton, ON"
        settings.get("models.chat")      → "qwen3:1.7b"
        settings.get("missing.key", 42)  → 42 (returns default if not found)
        
        HOW DOT NOTATION WORKS:
            "weather.city" gets split into ["weather", "city"]
            Then we drill into the dict: settings["weather"]["city"]
        """
        with self._lock:
            keys = key_path.split(".")
            value = self._settings
            try:
                for key in keys:
                    value = value[key]
                return value
            except (KeyError, TypeError):
                return default
    
    def set(self, key_path: str, value: Any):
        """
        Set a setting using dot notation. Saves to disk immediately.
        
        settings.set("weather.city", "Toronto")
        """
        with self._lock:
            keys = key_path.split(".")
            
            # Walk to the parent of the final key
            # For "weather.city", we walk to settings["weather"]
            # then set ["city"] = value
            target = self._settings
            for key in keys[:-1]:       # All keys except the last one
                if key not in target:
                    target[key] = {}    # Create nested dict if missing
                target = target[key]
            
            # Set the final key
            target[keys[-1]] = value
            
            # Save to disk immediately
            self._save()
    
    def get_all(self) -> Dict:
        """Return a copy of all settings."""
        with self._lock:
            return self._settings.copy()
    
    def reset(self):
        """Reset everything to defaults."""
        with self._lock:
            self._settings = DEFAULT_SETTINGS.copy()
            self._save()
 
 
# ─────────────────────────────────────────────
# Global Instance
# ─────────────────────────────────────────────
# Same pattern as the registry — one instance for the whole app.
# Every file that does "from core.settings_store import settings"
# gets the same object.
settings = SettingsStore()
