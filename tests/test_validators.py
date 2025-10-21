"""
Unit tests for Data Validators
"""

import pytest
import pandas as pd
import numpy as np
from utils.validators import DataValidator, ValidationLevel, ValidationResult


class TestDataValidator:
    """Test suite for DataValidator class."""

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for testing."""
        return pd.DataFrame({
            "Customer_ID": [1, 2, 3, 4, 5],
            "Age": [25, 30, 150, 40, 45],  # 150 is outlier
            "Income": [50000, 60000, 70000, None, 90000],
            "SSN": ["123-45-6789", "987-65-4321", "invalid", "111-22-3333", "444-55-6666"],
            "Score": [0.5, 0.7, 0.9, 0.2, 0.4]
        })

    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return DataValidator()

    def test_validate_schema_pass(self, validator, sample_df):
        """Test schema validation with all columns present."""
        expected_cols = ["Customer_ID", "Age", "Income"]
        result = validator.validate_schema(sample_df, expected_cols)
        assert result.passed is True

    def test_validate_schema_fail(self, validator, sample_df):
        """Test schema validation with missing columns."""
        expected_cols = ["Customer_ID", "Age", "Missing_Column"]
        result = validator.validate_schema(sample_df, expected_cols)
        assert result.passed is False

    def test_validate_missing_values(self, validator, sample_df):
        """Test missing value validation."""
        result = validator.validate_missing_values(
            sample_df,
            columns=["Income"],
            max_missing_pct=0.0
        )
        assert result.passed is False  # Has 1 missing value

    def test_validate_duplicates_no_dups(self, validator, sample_df):
        """Test duplicate check with no duplicates."""
        result = validator.validate_duplicates(sample_df)
        assert result.passed is True

    def test_validate_duplicates_with_dups(self, validator):
        """Test duplicate check with duplicates."""
        df_with_dups = pd.DataFrame({
            "A": [1, 2, 2, 3],
            "B": [4, 5, 5, 6]
        })
        result = validator.validate_duplicates(df_with_dups)
        assert result.passed is False

    def test_validate_range_pass(self, validator, sample_df):
        """Test range validation passing."""
        result = validator.validate_range(
            sample_df,
            column="Score",
            min_value=0.0,
            max_value=1.0
        )
        assert result.passed is True

    def test_validate_range_fail(self, validator, sample_df):
        """Test range validation failing."""
        result = validator.validate_range(
            sample_df,
            column="Age",
            min_value=18,
            max_value=75
        )
        assert result.passed is False  # 150 is out of range

    def test_validate_pattern_pass(self, validator):
        """Test pattern validation passing."""
        df = pd.DataFrame({"SSN": ["123-45-6789", "987-65-4321"]})
        result = validator.validate_pattern(
            df,
            column="SSN",
            pattern=r"^\d{3}-\d{2}-\d{4}$"
        )
        assert result.passed is True

    def test_validate_pattern_fail(self, validator, sample_df):
        """Test pattern validation failing."""
        result = validator.validate_pattern(
            sample_df,
            column="SSN",
            pattern=r"^\d{3}-\d{2}-\d{4}$"
        )
        assert result.passed is False  # "invalid" doesn't match

    def test_validate_row_count_pass(self, validator, sample_df):
        """Test row count validation passing."""
        result = validator.validate_row_count(sample_df, min_rows=1, max_rows=10)
        assert result.passed is True

    def test_validate_row_count_fail_min(self, validator, sample_df):
        """Test row count validation failing on min."""
        result = validator.validate_row_count(sample_df, min_rows=10)
        assert result.passed is False

    def test_has_critical_failures(self, validator, sample_df):
        """Test detection of critical failures."""
        validator.validate_schema(
            sample_df,
            ["Missing_Column"],
            level=ValidationLevel.CRITICAL
        )
        assert validator.has_critical_failures() is True

    def test_get_summary(self, validator, sample_df):
        """Test getting validation summary."""
        validator.validate_schema(sample_df, ["Customer_ID"])
        validator.validate_row_count(sample_df, min_rows=1)

        summary = validator.get_summary()
        assert "total_checks" in summary
        assert "passed" in summary
        assert "failed" in summary
        assert summary["total_checks"] == 2
