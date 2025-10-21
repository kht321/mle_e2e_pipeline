"""
Unit tests for Configuration Loader
"""

import pytest
from pathlib import Path
from utils.config_loader import ConfigLoader


class TestConfigLoader:
    """Test suite for ConfigLoader class."""

    def test_singleton_pattern(self):
        """Test that ConfigLoader implements singleton pattern."""
        config1 = ConfigLoader()
        config2 = ConfigLoader()
        assert config1 is config2

    def test_get_existing_key(self):
        """Test retrieving existing configuration key."""
        config = ConfigLoader()
        result = config.get("project.name")
        assert result is not None

    def test_get_nonexistent_key(self):
        """Test retrieving non-existent key returns default."""
        config = ConfigLoader()
        result = config.get("nonexistent.key", default="default_value")
        assert result == "default_value"

    def test_get_path(self):
        """Test get_path returns Path object."""
        config = ConfigLoader()
        result = config.get_path("paths.data_dir")
        assert isinstance(result, Path)

    def test_validate(self):
        """Test configuration validation."""
        config = ConfigLoader()
        assert config.validate() is True


class TestConfigLoaderEdgeCases:
    """Test edge cases for ConfigLoader."""

    def test_nested_key_access(self):
        """Test accessing deeply nested keys."""
        config = ConfigLoader()
        result = config.get("model.label_definition.mob_months")
        assert result is not None
        assert isinstance(result, int)

    def test_missing_nested_key(self):
        """Test missing nested key returns default."""
        config = ConfigLoader()
        result = config.get("missing.nested.key", default=None)
        assert result is None
