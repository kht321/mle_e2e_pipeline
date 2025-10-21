"""
Silver Layer Processing

Cleans and validates data from Bronze layer.

This layer implements data quality enforcement:
- Schema validation and type conversion
- Missing value handling (identification, not imputation)
- Outlier detection (identification, not capping)
- Business rule validation
- MOB and DPD calculation for loans
- NO feature engineering (deferred to Gold layer)
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import re

from .config_loader import ConfigLoader
from .logger import get_logger
from .validators import DataValidator, ValidationLevel
from .bronze_processing import BronzeProcessor

logger = get_logger(__name__)


class SilverProcessor:
    """
    Processes Bronze data into Silver layer with data quality enforcement.

    Responsibilities:
    - Validate and enforce schemas
    - Clean data (handle nulls, outliers, invalid values)
    - Apply business rules
    - Calculate derived columns (MOB, DPD)
    - Save validated data to Silver layer

    Example:
        >>> processor = SilverProcessor()
        >>> processor.process_all_sources()
    """

    def __init__(self, config_path: str = "config/pipeline_config.yaml"):
        """
        Initialize Silver processor.

        Args:
            config_path: Path to pipeline configuration file
        """
        self.config = ConfigLoader(config_path)
        self.bronze_processor = BronzeProcessor(config_path)

        self.silver_dir = Path(self.config.get("paths.silver_dir"))
        self.silver_dir.mkdir(parents=True, exist_ok=True)

        self.validator = DataValidator()
        self.validation_config = self.config.get("validation", {})

        logger.info("SilverProcessor initialized")

    def clean_attributes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean customer attributes data.

        Validations:
        - Age bounds (18-75)
        - SSN format validation
        - Remove null Customer_IDs
        - Standardize occupation values

        Args:
            df: Raw attributes DataFrame from Bronze

        Returns:
            Cleaned DataFrame
        """
        logger.info("Cleaning attributes data")
        initial_rows = len(df)

        # Make a copy to avoid modifying original
        df = df.copy()

        # Remove rows with null Customer_ID
        df = df.dropna(subset=["Customer_ID"])

        # Convert snapshot_date to datetime
        if "snapshot_date" in df.columns:
            df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])

        # Validate and clean Age
        age_config = self.validation_config.get("attributes", {})
        age_min = age_config.get("age_min", 18)
        age_max = age_config.get("age_max", 75)

        self.validator.validate_range(
            df, "Age",
            min_value=age_min,
            max_value=age_max,
            level=ValidationLevel.WARNING
        )

        # Filter to valid age range
        df = df[(df["Age"] >= age_min) & (df["Age"] <= age_max)]

        # Validate SSN format (XXX-XX-XXXX)
        ssn_pattern = age_config.get("ssn_pattern", r"^\d{3}-\d{2}-\d{4}$")

        if "SSN" in df.columns:
            self.validator.validate_pattern(
                df, "SSN",
                pattern=ssn_pattern,
                level=ValidationLevel.WARNING
            )

            # Keep only valid SSNs (or null)
            valid_ssn = df["SSN"].isna() | df["SSN"].str.match(ssn_pattern, na=False)
            df = df[valid_ssn]

        # Clean Occupation
        if "Occupation" in df.columns:
            # Standardize occupation values
            df["Occupation"] = df["Occupation"].str.strip()
            df["Occupation"] = df["Occupation"].replace(
                ["_______", "N/A", "na", "NA", ""],
                np.nan
            )

        # Clean Name
        if "Name" in df.columns:
            df["Name"] = df["Name"].str.strip()
            df["Name"] = df["Name"].replace(["_______", ""], np.nan)

        logger.info(
            f"Attributes cleaned: {initial_rows:,} → {len(df):,} rows "
            f"({initial_rows - len(df):,} removed)"
        )

        return df

    def clean_financials(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean financial features data.

        Validations:
        - Remove invalid string values ("_", "NA", etc.)
        - Convert numeric columns to proper types
        - Validate ranges for key financial metrics
        - Clean categorical values

        Args:
            df: Raw financials DataFrame from Bronze

        Returns:
            Cleaned DataFrame
        """
        logger.info("Cleaning financials data")
        initial_rows = len(df)

        df = df.copy()

        # Remove rows with null Customer_ID
        df = df.dropna(subset=["Customer_ID"])

        # Convert snapshot_date to datetime
        if "snapshot_date" in df.columns:
            df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])

        # Replace placeholder values with NaN
        null_placeholders = ["_", "NA", "na", "N/A", "!@9#%8", "", " "]

        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].replace(null_placeholders, np.nan)

        # Clean numeric columns
        numeric_cols = [
            "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts",
            "Num_Credit_Card", "Interest_Rate", "Num_of_Loan",
            "Delay_from_due_date", "Num_of_Delayed_Payment",
            "Num_Credit_Inquiries", "Outstanding_Debt",
            "Credit_Utilization_Ratio", "Total_EMI_per_month",
            "Amount_invested_monthly", "Monthly_Balance"
        ]

        for col in numeric_cols:
            if col in df.columns:
                # Remove non-numeric characters (except - and .)
                if df[col].dtype == "object":
                    df[col] = df[col].str.replace(r"[^0-9.-]", "", regex=True)
                    df[col] = pd.to_numeric(df[col], errors="coerce")

        # Validate Annual_Income range
        fin_config = self.validation_config.get("financials", {})
        income_min = fin_config.get("annual_income_min", 0)
        income_max = fin_config.get("annual_income_max", 10000000)

        if "Annual_Income" in df.columns:
            self.validator.validate_range(
                df, "Annual_Income",
                min_value=income_min,
                max_value=income_max,
                level=ValidationLevel.WARNING
            )

            # Filter to valid range
            df = df[
                (df["Annual_Income"].isna()) |
                ((df["Annual_Income"] >= income_min) & (df["Annual_Income"] <= income_max))
            ]

        # Clean categorical columns
        categorical_cols = [
            "Type_of_Loan", "Credit_Mix", "Payment_of_Min_Amount",
            "Payment_Behaviour"
        ]

        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].str.strip() if df[col].dtype == "object" else df[col]

        # Parse Credit_History_Age (format: "X Years and Y Months")
        if "Credit_History_Age" in df.columns:
            df["Credit_History_Age_Months"] = df["Credit_History_Age"].apply(
                self._parse_credit_history_age
            )

        logger.info(
            f"Financials cleaned: {initial_rows:,} → {len(df):,} rows "
            f"({initial_rows - len(df):,} removed)"
        )

        return df

    def clean_clickstream(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean clickstream behavioral data.

        Validations:
        - Convert feature columns to numeric
        - Identify outliers (but don't cap - deferred to Gold)
        - Remove negative values if configured
        - Validate snapshot_date

        Args:
            df: Raw clickstream DataFrame from Bronze

        Returns:
            Cleaned DataFrame
        """
        logger.info("Cleaning clickstream data")
        initial_rows = len(df)

        df = df.copy()

        # Remove rows with null Customer_ID
        df = df.dropna(subset=["Customer_ID"])

        # Convert snapshot_date to datetime
        if "snapshot_date" in df.columns:
            df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])

        # Get feature columns (fe_1 to fe_20)
        feature_cols = [col for col in df.columns if col.startswith("fe_")]

        # Convert to numeric
        for col in feature_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Validate ranges (identify but don't filter yet)
        click_config = self.validation_config.get("clickstream", {})
        feat_min = click_config.get("feature_min", -1000)
        feat_max = click_config.get("feature_max", 10000)

        # Sample validate first feature column
        if len(feature_cols) > 0:
            self.validator.validate_range(
                df, feature_cols[0],
                min_value=feat_min,
                max_value=feat_max,
                level=ValidationLevel.WARNING
            )

        logger.info(
            f"Clickstream cleaned: {initial_rows:,} → {len(df):,} rows "
            f"({initial_rows - len(df):,} removed)"
        )

        return df

    def clean_loans(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean loans data and calculate MOB, DPD.

        Calculations:
        - MOB (Months on Book) = installment_num
        - DPD (Days Past Due) = days between snapshot_date and first missed payment
        - installments_missed = count of missed installments

        Args:
            df: Raw loans DataFrame from Bronze

        Returns:
            Cleaned DataFrame with MOB and DPD columns
        """
        logger.info("Cleaning loans data and calculating MOB/DPD")
        initial_rows = len(df)

        df = df.copy()

        # Remove rows with null loan_id or Customer_ID
        df = df.dropna(subset=["loan_id", "Customer_ID"])

        # Convert dates
        date_cols = ["loan_start_date", "snapshot_date"]
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])

        # Convert numeric columns
        numeric_cols = [
            "installment_num", "loan_amt", "due_amt", "paid_amt",
            "overdue_amt", "balance"
        ]

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Calculate MOB (Months on Book)
        if "installment_num" in df.columns:
            df["mob"] = df["installment_num"]
        else:
            # Fallback: calculate from loan_start_date and snapshot_date
            if "loan_start_date" in df.columns and "snapshot_date" in df.columns:
                df["mob"] = (
                    (df["snapshot_date"].dt.year - df["loan_start_date"].dt.year) * 12 +
                    (df["snapshot_date"].dt.month - df["loan_start_date"].dt.month)
                )

        # Calculate DPD (Days Past Due)
        # DPD is when overdue_amt > 0, we need to identify first missed payment
        df = df.sort_values(["loan_id", "snapshot_date"])

        # For each loan, find first snapshot where overdue_amt > 0
        df["has_overdue"] = df["overdue_amt"] > 0

        # Calculate cumulative overdue flag
        df["first_missed_date"] = df.groupby("loan_id")["snapshot_date"].transform(
            lambda x: x.iloc[0] if (df.loc[x.index, "has_overdue"].any()) else pd.NaT
        )

        # For loans with overdue, find first occurrence
        overdue_first = df[df["has_overdue"]].groupby("loan_id")["snapshot_date"].min()
        df["first_missed_date"] = df["loan_id"].map(overdue_first)

        # DPD = days since first missed payment
        df["dpd"] = (
            (df["snapshot_date"] - df["first_missed_date"]).dt.days
        ).fillna(0).astype(int)

        # Ensure DPD is non-negative
        df["dpd"] = df["dpd"].clip(lower=0)

        # Drop intermediate columns
        df = df.drop(columns=["has_overdue", "first_missed_date"], errors="ignore")

        logger.info(
            f"Loans cleaned: {initial_rows:,} → {len(df):,} rows "
            f"({initial_rows - len(df):,} removed)"
        )

        # Log DPD and MOB stats
        logger.info(f"MOB range: {df['mob'].min()} to {df['mob'].max()}")
        logger.info(f"DPD range: {df['dpd'].min()} to {df['dpd'].max()}")
        logger.info(f"Loans with DPD > 0: {(df['dpd'] > 0).sum():,}")

        return df

    @staticmethod
    def _parse_credit_history_age(value: str) -> Optional[int]:
        """
        Parse credit history age string to months.

        Args:
            value: String like "5 Years and 3 Months"

        Returns:
            Total months, or None if parsing fails

        Example:
            >>> _parse_credit_history_age("5 Years and 3 Months")
            63
        """
        if pd.isna(value) or not isinstance(value, str):
            return None

        try:
            years = 0
            months = 0

            year_match = re.search(r"(\d+)\s+Years?", value, re.IGNORECASE)
            if year_match:
                years = int(year_match.group(1))

            month_match = re.search(r"(\d+)\s+Months?", value, re.IGNORECASE)
            if month_match:
                months = int(month_match.group(1))

            return years * 12 + months

        except Exception:
            return None

    def process_source(
        self,
        source_name: str,
        validate: bool = True
    ) -> pd.DataFrame:
        """
        Process a single source from Bronze to Silver.

        Args:
            source_name: Name of source (clickstream, attributes, financials, loans)
            validate: Whether to perform validation

        Returns:
            Cleaned DataFrame

        Raises:
            ValueError: If source is unknown or validation fails
        """
        logger.info(f"Processing Silver layer for: {source_name}")

        # Read from Bronze
        df_bronze = self.bronze_processor.read_bronze_source(source_name)

        # Apply source-specific cleaning
        cleaning_methods = {
            "attributes": self.clean_attributes,
            "financials": self.clean_financials,
            "clickstream": self.clean_clickstream,
            "loans": self.clean_loans,
        }

        clean_method = cleaning_methods.get(source_name)
        if not clean_method:
            raise ValueError(f"Unknown source: {source_name}")

        df_silver = clean_method(df_bronze)

        # Validate if requested
        if validate:
            self._validate_silver_data(df_silver, source_name)

            if self.validator.has_critical_failures():
                self.validator.print_results()
                raise ValueError(f"Critical validation failures for {source_name}")

        # Save to Silver layer
        self._save_to_silver(df_silver, source_name)

        return df_silver

    def _validate_silver_data(self, df: pd.DataFrame, source_name: str):
        """
        Validate Silver layer data quality.

        Args:
            df: DataFrame to validate
            source_name: Name of data source
        """
        logger.info(f"Validating Silver data: {source_name}")

        # Row count check
        self.validator.validate_row_count(
            df,
            min_rows=10,
            level=ValidationLevel.WARNING
        )

        # Check for excessive nulls in key columns
        if source_name == "attributes":
            self.validator.validate_missing_values(
                df,
                ["Customer_ID", "Age"],
                max_missing_pct=0.0,
                level=ValidationLevel.CRITICAL
            )

        elif source_name == "financials":
            self.validator.validate_missing_values(
                df,
                ["Customer_ID"],
                max_missing_pct=0.0,
                level=ValidationLevel.CRITICAL
            )

        elif source_name == "clickstream":
            self.validator.validate_missing_values(
                df,
                ["Customer_ID", "snapshot_date"],
                max_missing_pct=0.0,
                level=ValidationLevel.CRITICAL
            )

        elif source_name == "loans":
            self.validator.validate_missing_values(
                df,
                ["Customer_ID", "loan_id", "mob", "dpd"],
                max_missing_pct=0.0,
                level=ValidationLevel.CRITICAL
            )

    def _save_to_silver(self, df: pd.DataFrame, source_name: str):
        """
        Save DataFrame to Silver layer.

        Args:
            df: DataFrame to save
            source_name: Name of data source
        """
        output_path = self.silver_dir / f"{source_name}_clean"

        # Partition temporal data
        if "snapshot_date" in df.columns:
            df["year_month"] = df["snapshot_date"].dt.to_period("M").astype(str)

            for year_month, group in df.groupby("year_month"):
                partition_path = output_path / f"year_month={year_month}"
                partition_path.mkdir(parents=True, exist_ok=True)

                group_to_save = group.drop(columns=["year_month"])
                parquet_file = partition_path / "data.parquet"

                group_to_save.to_parquet(
                    parquet_file,
                    engine="pyarrow",
                    compression="snappy",
                    index=False
                )

            logger.info(f"Saved {source_name} to Silver (partitioned)")

        else:
            output_path.mkdir(parents=True, exist_ok=True)
            parquet_file = output_path / "data.parquet"

            df.to_parquet(
                parquet_file,
                engine="pyarrow",
                compression="snappy",
                index=False
            )

            logger.info(f"Saved {source_name} to Silver")

    def process_all_sources(self) -> Dict[str, pd.DataFrame]:
        """
        Process all sources from Bronze to Silver.

        Returns:
            Dictionary mapping source names to cleaned DataFrames

        Raises:
            ValueError: If any critical validations fail
        """
        logger.info("Processing all sources to Silver layer")
        start_time = datetime.now()

        results = {}
        source_names = ["attributes", "financials", "clickstream", "loans"]

        for source_name in source_names:
            try:
                self.validator.clear_results()
                df = self.process_source(source_name, validate=True)
                results[source_name] = df
                logger.info(f"✓ Successfully processed {source_name}")

            except Exception as e:
                logger.error(f"✗ Failed to process {source_name}: {e}")
                raise

        duration = (datetime.now() - start_time).total_seconds()

        logger.info("\n" + "="*80)
        logger.info("SILVER LAYER PROCESSING SUMMARY")
        logger.info("="*80)
        logger.info(f"Duration: {duration:.2f} seconds")
        logger.info(f"Successfully processed: {len(results)} sources")
        logger.info("="*80 + "\n")

        return results

    def read_silver_source(
        self,
        source_name: str,
        date_filter: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Read data from Silver layer.

        Args:
            source_name: Name of source to read
            date_filter: Optional year-month filter (e.g., "2023-01")

        Returns:
            DataFrame from Silver layer
        """
        source_path = self.silver_dir / f"{source_name}_clean"

        if not source_path.exists():
            raise FileNotFoundError(
                f"Silver source not found: {source_name}. "
                "Run process_all_sources() first."
            )

        # Check if partitioned
        partitions = list(source_path.glob("year_month=*"))

        if partitions and date_filter:
            partition_path = source_path / f"year_month={date_filter}"
            df = pd.read_parquet(partition_path / "data.parquet")

        elif partitions:
            dfs = [
                pd.read_parquet(p / "data.parquet")
                for p in sorted(partitions)
            ]
            df = pd.concat(dfs, ignore_index=True)

        else:
            df = pd.read_parquet(source_path / "data.parquet")

        logger.info(f"Read {len(df):,} rows from Silver: {source_name}")
        return df


# Convenience function
def process_silver_tables(config_path: str = "config/pipeline_config.yaml") -> Dict[str, pd.DataFrame]:
    """
    Convenience function to process all Silver tables.

    Args:
        config_path: Path to pipeline configuration

    Returns:
        Dictionary of source name to cleaned DataFrame
    """
    processor = SilverProcessor(config_path)
    return processor.process_all_sources()
