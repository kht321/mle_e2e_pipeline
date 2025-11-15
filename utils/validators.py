"""
Data Validation Framework

Provides data quality checks and validation for pipeline stages.
"""

import re
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Severity levels for validation failures."""
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """
    Result of a validation check.

    Attributes:
        check_name: Name of the validation check
        passed: Whether check passed
        level: Severity level
        message: Description of result
        details: Additional details (e.g., failed row count)
    """
    check_name: str
    passed: bool
    level: ValidationLevel
    message: str
    details: Optional[Dict] = None

    def __str__(self) -> str:
        status = "✓ PASS" if self.passed else "✗ FAIL"
        return f"{status} [{self.level.value.upper()}] {self.check_name}: {self.message}"


class DataValidator:
    """
    Validates data quality across different pipeline stages.

    Implements comprehensive data quality checks including:
    - Schema validation
    - Missing value checks
    - Duplicate detection
    - Range validation
    - Pattern matching
    - Statistical validation

    Example:
        >>> validator = DataValidator()
        >>> results = validator.validate_schema(df, expected_columns)
        >>> validator.print_results(results)
    """

    def __init__(self):
        """Initialize validator."""
        self.results: List[ValidationResult] = []

    def validate_schema(
        self,
        df: pd.DataFrame,
        expected_columns: List[str],
        level: ValidationLevel = ValidationLevel.ERROR
    ) -> ValidationResult:
        """
        Validate that DataFrame has expected columns.

        Args:
            df: DataFrame to validate
            expected_columns: List of required column names
            level: Severity level if validation fails

        Returns:
            ValidationResult
        """
        actual_columns = set(df.columns)
        expected_set = set(expected_columns)

        missing = expected_set - actual_columns
        extra = actual_columns - expected_set

        if missing:
            result = ValidationResult(
                check_name="schema_validation",
                passed=False,
                level=level,
                message=f"Missing columns: {missing}",
                details={"missing": list(missing), "extra": list(extra)}
            )
        else:
            result = ValidationResult(
                check_name="schema_validation",
                passed=True,
                level=ValidationLevel.WARNING,
                message=f"All {len(expected_columns)} expected columns present",
                details={"extra": list(extra) if extra else None}
            )

        self.results.append(result)
        logger.info(str(result))
        return result

    def validate_missing_values(
        self,
        df: pd.DataFrame,
        columns: List[str],
        max_missing_pct: float = 0.0,
        level: ValidationLevel = ValidationLevel.WARNING
    ) -> ValidationResult:
        """
        Validate missing value percentage for specified columns.

        Args:
            df: DataFrame to validate
            columns: Columns to check
            max_missing_pct: Maximum allowed missing percentage (0.0 to 1.0)
            level: Severity level if validation fails

        Returns:
            ValidationResult
        """
        total_rows = len(df)
        missing_info = {}

        for col in columns:
            if col in df.columns:
                missing_count = df[col].isna().sum()
                missing_pct = missing_count / total_rows
                missing_info[col] = {
                    "count": int(missing_count),
                    "percentage": float(missing_pct)
                }

        failed_cols = {
            col: info for col, info in missing_info.items()
            if info["percentage"] > max_missing_pct
        }

        if failed_cols:
            result = ValidationResult(
                check_name="missing_values_check",
                passed=False,
                level=level,
                message=f"{len(failed_cols)} columns exceed {max_missing_pct*100}% missing threshold",
                details=failed_cols
            )
        else:
            result = ValidationResult(
                check_name="missing_values_check",
                passed=True,
                level=ValidationLevel.WARNING,
                message=f"All columns have ≤{max_missing_pct*100}% missing values",
                details=missing_info
            )

        self.results.append(result)
        logger.info(str(result))
        return result

    def validate_duplicates(
        self,
        df: pd.DataFrame,
        subset: Optional[List[str]] = None,
        level: ValidationLevel = ValidationLevel.WARNING
    ) -> ValidationResult:
        """
        Check for duplicate rows.

        Args:
            df: DataFrame to validate
            subset: Columns to check for duplicates (None = all columns)
            level: Severity level if validation fails

        Returns:
            ValidationResult
        """
        dup_count = df.duplicated(subset=subset).sum()
        total_rows = len(df)
        dup_pct = dup_count / total_rows if total_rows > 0 else 0

        if dup_count > 0:
            result = ValidationResult(
                check_name="duplicate_check",
                passed=False,
                level=level,
                message=f"Found {dup_count} duplicate rows ({dup_pct*100:.2f}%)",
                details={
                    "duplicate_count": int(dup_count),
                    "duplicate_percentage": float(dup_pct),
                    "subset": subset
                }
            )
        else:
            result = ValidationResult(
                check_name="duplicate_check",
                passed=True,
                level=ValidationLevel.WARNING,
                message="No duplicate rows found",
                details={"subset": subset}
            )

        self.results.append(result)
        logger.info(str(result))
        return result

    def validate_range(
        self,
        df: pd.DataFrame,
        column: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        level: ValidationLevel = ValidationLevel.ERROR
    ) -> ValidationResult:
        """
        Validate that values in column are within specified range.

        Args:
            df: DataFrame to validate
            column: Column to check
            min_value: Minimum allowed value (inclusive)
            max_value: Maximum allowed value (inclusive)
            level: Severity level if validation fails

        Returns:
            ValidationResult
        """
        if column not in df.columns:
            result = ValidationResult(
                check_name=f"range_check_{column}",
                passed=False,
                level=ValidationLevel.CRITICAL,
                message=f"Column '{column}' not found",
                details=None
            )
            self.results.append(result)
            return result

        series = df[column].dropna()

        # Coerce non-numeric entries to numeric and treat them as violations
        numeric_series = pd.to_numeric(series, errors="coerce")
        non_numeric_mask = numeric_series.isna()
        non_numeric_count = int(non_numeric_mask.sum())
        numeric_series = numeric_series.dropna()

        violations = non_numeric_count
        if min_value is not None:
            violations += int((numeric_series < min_value).sum())
        if max_value is not None:
            violations += int((numeric_series > max_value).sum())

        total_valid = len(numeric_series)
        total_checked = total_valid + non_numeric_count
        violation_pct = violations / total_checked if total_checked > 0 else 0

        if violations > 0:
            result = ValidationResult(
                check_name=f"range_check_{column}",
                passed=False,
                level=level,
                message=f"{violations} values outside range [{min_value}, {max_value}]",
                details={
                    "column": column,
                    "violations": int(violations),
                    "violation_percentage": float(violation_pct),
                    "non_numeric_values": non_numeric_count,
                    "min_value": min_value,
                    "max_value": max_value,
                    "actual_min": float(numeric_series.min()) if total_valid else None,
                    "actual_max": float(numeric_series.max()) if total_valid else None
                }
            )
        else:
            result = ValidationResult(
                check_name=f"range_check_{column}",
                passed=True,
                level=ValidationLevel.WARNING,
                message=f"All values within range [{min_value}, {max_value}]",
                details={
                    "column": column,
                    "actual_min": float(numeric_series.min()) if total_valid else None,
                    "actual_max": float(numeric_series.max()) if total_valid else None
                }
            )

        self.results.append(result)
        logger.info(str(result))
        return result

    def validate_pattern(
        self,
        df: pd.DataFrame,
        column: str,
        pattern: str,
        level: ValidationLevel = ValidationLevel.ERROR
    ) -> ValidationResult:
        """
        Validate that values match a regex pattern.

        Args:
            df: DataFrame to validate
            column: Column to check
            pattern: Regex pattern to match
            level: Severity level if validation fails

        Returns:
            ValidationResult
        """
        if column not in df.columns:
            result = ValidationResult(
                check_name=f"pattern_check_{column}",
                passed=False,
                level=ValidationLevel.CRITICAL,
                message=f"Column '{column}' not found",
                details=None
            )
            self.results.append(result)
            return result

        series = df[column].dropna().astype(str)
        matches = series.str.match(pattern)
        violations = (~matches).sum()
        total_valid = len(series)
        violation_pct = violations / total_valid if total_valid > 0 else 0

        if violations > 0:
            result = ValidationResult(
                check_name=f"pattern_check_{column}",
                passed=False,
                level=level,
                message=f"{violations} values don't match pattern '{pattern}'",
                details={
                    "column": column,
                    "pattern": pattern,
                    "violations": int(violations),
                    "violation_percentage": float(violation_pct)
                }
            )
        else:
            result = ValidationResult(
                check_name=f"pattern_check_{column}",
                passed=True,
                level=ValidationLevel.WARNING,
                message=f"All values match pattern '{pattern}'",
                details={"column": column, "pattern": pattern}
            )

        self.results.append(result)
        logger.info(str(result))
        return result

    def validate_row_count(
        self,
        df: pd.DataFrame,
        min_rows: int = 1,
        max_rows: Optional[int] = None,
        level: ValidationLevel = ValidationLevel.CRITICAL
    ) -> ValidationResult:
        """
        Validate that DataFrame has expected row count.

        Args:
            df: DataFrame to validate
            min_rows: Minimum required rows
            max_rows: Maximum allowed rows (optional)
            level: Severity level if validation fails

        Returns:
            ValidationResult
        """
        row_count = len(df)

        if row_count < min_rows:
            result = ValidationResult(
                check_name="row_count_check",
                passed=False,
                level=level,
                message=f"Row count {row_count} below minimum {min_rows}",
                details={"row_count": row_count, "min_rows": min_rows}
            )
        elif max_rows is not None and row_count > max_rows:
            result = ValidationResult(
                check_name="row_count_check",
                passed=False,
                level=level,
                message=f"Row count {row_count} exceeds maximum {max_rows}",
                details={"row_count": row_count, "max_rows": max_rows}
            )
        else:
            result = ValidationResult(
                check_name="row_count_check",
                passed=True,
                level=ValidationLevel.WARNING,
                message=f"Row count {row_count} within expected range",
                details={"row_count": row_count}
            )

        self.results.append(result)
        logger.info(str(result))
        return result

    def print_results(self, results: Optional[List[ValidationResult]] = None):
        """
        Print validation results in a formatted way.

        Args:
            results: List of results to print (defaults to all stored results)
        """
        if results is None:
            results = self.results

        if not results:
            print("No validation results to display")
            return

        print("\n" + "="*80)
        print("VALIDATION RESULTS")
        print("="*80)

        passed = sum(1 for r in results if r.passed)
        failed = len(results) - passed

        for result in results:
            print(f"\n{result}")
            if result.details and not result.passed:
                print(f"  Details: {result.details}")

        print("\n" + "="*80)
        print(f"SUMMARY: {passed} passed, {failed} failed out of {len(results)} checks")
        print("="*80 + "\n")

    def clear_results(self):
        """Clear stored validation results."""
        self.results = []

    def has_critical_failures(self) -> bool:
        """
        Check if any critical validations failed.

        Returns:
            True if critical failures exist
        """
        return any(
            not r.passed and r.level == ValidationLevel.CRITICAL
            for r in self.results
        )

    def get_summary(self) -> Dict:
        """
        Get summary statistics of validation results.

        Returns:
            Dictionary with summary statistics
        """
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed

        by_level = {}
        for level in ValidationLevel:
            by_level[level.value] = sum(
                1 for r in self.results
                if not r.passed and r.level == level
            )

        return {
            "total_checks": total,
            "passed": passed,
            "failed": failed,
            "failures_by_level": by_level,
            "has_critical_failures": self.has_critical_failures()
        }
