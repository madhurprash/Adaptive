"""
Simple configuration manager for Adaptive CLI.
Inspired by Claude Code's approach - stores config locally.
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional


logger = logging.getLogger(__name__)


# Configuration file location
CONFIG_DIR = Path.home() / ".adaptive"
CONFIG_FILE = CONFIG_DIR / "config.json"


class ConfigManager:
    """Manages Adaptive configuration and API keys."""

    def __init__(self):
        """Initialize configuration manager."""
        self._ensure_config_dir()
        self._config = self._load_config()

    def _ensure_config_dir(self) -> None:
        """Ensure configuration directory exists."""
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        # Set directory permissions to user-only
        CONFIG_DIR.chmod(0o700)

    def _load_config(self) -> dict[str, Any]:
        """Load configuration from file."""
        if not CONFIG_FILE.exists():
            return self._get_default_config()

        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return self._get_default_config()

    def _save_config(self) -> None:
        """Save configuration to file."""
        try:
            with open(CONFIG_FILE, "w") as f:
                json.dump(self._config, f, indent=2)
            # Set file permissions to user-only
            CONFIG_FILE.chmod(0o600)
            logger.info(f"Configuration saved to {CONFIG_FILE}")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            raise

    def _get_default_config(self) -> dict[str, Any]:
        """Get default configuration."""
        return {
            "model_id": "us.anthropic.claude-3-5-sonnet-20240620-v1:0",
            "platform": "langsmith",
            "temperature": "0.1",
            "max_tokens": "2048",
            "top_p": "0.92",
            "api_keys": {},
            "memory_enabled": True,
            "memory_name": "AdaptiveMemory",
            "memory_region": "us-west-2",
        }

    def get(
        self,
        key: str,
        default: Any = None,
    ) -> Any:
        """
        Get a configuration value.

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value
        """
        return self._config.get(key, default)

    def set(
        self,
        key: str,
        value: Any,
    ) -> None:
        """
        Set a configuration value.

        Args:
            key: Configuration key
            value: Configuration value
        """
        self._config[key] = value
        self._save_config()
        logger.info(f"Set {key} = {value}")

    def get_api_key(
        self,
        platform: str,
    ) -> Optional[str]:
        """
        Get API key for a platform.

        Args:
            platform: Platform name

        Returns:
            API key or None
        """
        return self._config.get("api_keys", {}).get(platform)

    def set_api_key(
        self,
        platform: str,
        api_key: str,
    ) -> None:
        """
        Set API key for a platform.

        Args:
            platform: Platform name
            api_key: API key value
        """
        if "api_keys" not in self._config:
            self._config["api_keys"] = {}

        self._config["api_keys"][platform] = api_key
        self._save_config()
        logger.info(f"Stored API key for {platform}")

    def list_api_keys(self) -> list[str]:
        """
        List platforms with stored API keys.

        Returns:
            List of platform names
        """
        return list(self._config.get("api_keys", {}).keys())

    def delete_api_key(
        self,
        platform: str,
    ) -> bool:
        """
        Delete API key for a platform.

        Args:
            platform: Platform name

        Returns:
            True if deleted, False if not found
        """
        api_keys = self._config.get("api_keys", {})
        if platform in api_keys:
            del api_keys[platform]
            self._save_config()
            logger.info(f"Deleted API key for {platform}")
            return True
        return False

    def show(self) -> None:
        """Display current configuration."""
        print("\n" + "=" * 70)
        print("ADAPTIVE CONFIGURATION")
        print("=" * 70)
        print(f"Model ID:        {self._config.get('model_id')}")
        print(f"Temperature:     {self._config.get('temperature')}")
        print(f"Max Tokens:      {self._config.get('max_tokens')}")
        print(f"Top-P:           {self._config.get('top_p')}")
        print(f"Platform:        {self._config.get('platform')}")
        print(f"\nMemory Enabled:  {self._config.get('memory_enabled')}")
        if self._config.get('memory_enabled'):
            print(f"Memory Name:     {self._config.get('memory_name')}")
            print(f"Memory Region:   {self._config.get('memory_region')}")

        # Show API keys
        api_keys = self.list_api_keys()
        print(f"\nStored API Keys: {', '.join(api_keys) if api_keys else 'None'}")
        print("=" * 70)
        print(f"\nConfig file: {CONFIG_FILE}")
        print()

    def get_all(self) -> dict[str, Any]:
        """
        Get all configuration.

        Returns:
            Complete configuration dictionary
        """
        return self._config.copy()
