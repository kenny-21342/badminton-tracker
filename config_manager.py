import json
import os
from pathlib import Path


class ConfigManager:
    """
    Manages persistent user settings for the badminton tracker.
    Settings are saved to ~/.badminton-tracker/config.json
    """

    def __init__(self, config_path=None):
        if config_path is None:
            self.config_dir = Path.home() / ".badminton-tracker"
            self.config_path = self.config_dir / "config.json"
        else:
            self.config_path = Path(config_path)
            self.config_dir = self.config_path.parent

        # Create config directory if it doesn't exist
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Default settings
        self.defaults = {
            "detector_mode": "tracknet",
            "smoothing": 0.6,
            "theme": "dark",
            "trail_opacity": 0.8,
            "show_minimap": True,
            "show_hud": True,
            "tracknet_resolution": "384x216",
            "brightness_threshold": 200,
            "confidence_threshold": 0.3,
            "hit_threshold": 25,
            "rally_cooldown": 12,
        }

        # Load existing config or use defaults
        self.config = self._load()

    def _load(self):
        """Load config from disk, or return defaults if not found."""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    loaded = json.load(f)
                    # Merge with defaults (in case new settings were added)
                    return {**self.defaults, **loaded}
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load config from {self.config_path}: {e}")
                return self.defaults.copy()
        else:
            return self.defaults.copy()

    def save(self):
        """Save current config to disk."""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        except IOError as e:
            print(f"Warning: Could not save config to {self.config_path}: {e}")

    def get(self, key, default=None):
        """Get a config value."""
        return self.config.get(key, default)

    def set(self, key, value):
        """Set a config value."""
        self.config[key] = value

    def update(self, **kwargs):
        """Update multiple config values at once."""
        self.config.update(kwargs)

    def reset(self):
        """Reset all settings to defaults."""
        self.config = self.defaults.copy()
