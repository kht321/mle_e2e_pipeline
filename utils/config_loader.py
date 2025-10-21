"""
Configuration Loader

Handles loading and validation of YAML configuration files.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
import logging

logger = logging.getLogger(__name__)


class ConfigLoader:
    """
    Load and manage pipeline configuration from YAML files.

    This class implements the Singleton pattern to ensure consistent
    configuration across the entire pipeline.

    Attributes:
        config_path (Path): Path to the configuration file
        config (Dict): Loaded configuration dictionary

    Example:
        >>> config = ConfigLoader("config/pipeline_config.yaml")
        >>> model_params = config.get("model.algorithms")
    """

    _instance: Optional['ConfigLoader'] = None

    def __new__(cls, config_path: str = "config/pipeline_config.yaml"):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config_path: str = "config/pipeline_config.yaml"):
        """
        Initialize configuration loader.

        Args:
            config_path: Path to YAML configuration file

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is malformed
        """
        if self._initialized:
            return

        self.config_path = Path(config_path)

        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        logger.info(f"Configuration loaded from {config_path}")
        self._initialized = True

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            key: Configuration key in dot notation (e.g., "model.algorithms")
            default: Default value if key not found

        Returns:
            Configuration value or default

        Example:
            >>> config.get("model.label_definition.mob_months")
            6
        """
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value

    def get_path(self, key: str) -> Path:
        """
        Get configuration value as Path object.

        Args:
            key: Configuration key for path

        Returns:
            Path object

        Example:
            >>> config.get_path("paths.bronze_dir")
            PosixPath('datamart/bronze')
        """
        value = self.get(key)
        return Path(value) if value else None

    def validate(self) -> bool:
        """
        Validate that required configuration keys exist.

        Returns:
            True if valid, raises ValueError otherwise

        Raises:
            ValueError: If required keys are missing
        """
        required_keys = [
            "paths.data_dir",
            "paths.bronze_dir",
            "paths.silver_dir",
            "paths.gold_dir",
            "paths.models_dir",
            "model.label_definition.mob_months",
            "model.label_definition.dpd_threshold",
        ]

        missing_keys = []
        for key in required_keys:
            if self.get(key) is None:
                missing_keys.append(key)

        if missing_keys:
            raise ValueError(f"Missing required config keys: {missing_keys}")

        logger.info("Configuration validation passed")
        return True

    def __str__(self) -> str:
        """String representation of config."""
        return f"ConfigLoader(path={self.config_path}, keys={len(self.config)})"

    def __repr__(self) -> str:
        """Detailed representation."""
        return self.__str__()
