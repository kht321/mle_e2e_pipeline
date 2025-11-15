"""
Gold Layer Processing

Creates ML-ready feature store and label store with sophisticated feature engineering.

This layer implements:
- Multi-snapshot temporal join (prevents data leakage)
- Label creation (default = DPD >= 30 at MOB = 6)
- Advanced feature engineering (ratios, log transforms, interactions)
- Outlier capping (quantile-based)
- Categorical encoding preparation
- Feature store and label store creation
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import StandardScaler
import logging
from pandas.api.types import is_categorical_dtype

from .config_loader import ConfigLoader
from .logger import get_logger
from .silver_processing import SilverProcessor

logger = get_logger(__name__)


class GoldProcessor:
    """
    Processes Silver data into Gold layer with ML-ready features.

    This is the most critical layer for preventing data leakage.
    Uses explicit temporal alignment: features from time T, labels from T+MOB_months.

    Example:
        >>> processor = GoldProcessor()
        >>> feature_store, label_store = processor.create_feature_and_label_stores()
    """

    def __init__(self, config_path: str = "config/pipeline_config.yaml"):
        """
        Initialize Gold processor.

        Args:
            config_path: Path to pipeline configuration file
        """
        self.config = ConfigLoader(config_path)
        self.silver_processor = SilverProcessor(config_path)

        self.gold_dir = Path(self.config.get("paths.gold_dir"))
        self.gold_dir.mkdir(parents=True, exist_ok=True)

        # Label definition
        self.mob_months = self.config.get("model.label_definition.mob_months", 6)
        self.dpd_threshold = self.config.get("model.label_definition.dpd_threshold", 30)

        # Feature engineering config
        self.feat_config = self.config.get("feature_engineering", {})

        logger.info(
            f"GoldProcessor initialized (MOB={self.mob_months}, "
            f"DPD>={self.dpd_threshold})"
        )

    def multi_snapshot_join(
        self,
        features_df: pd.DataFrame,
        loans_df: pd.DataFrame,
        mob_months: Optional[int] = None,
        dpd_threshold: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Join features at time T with labels at time T+mob_months.

        This function prevents data leakage by explicitly ensuring:
        - Features are from snapshot_date
        - Labels are from snapshot_date + mob_months
        - Both dates exist before joining
        - Temporal semantics are clear with renamed columns

        Args:
            features_df: DataFrame with features and snapshot_date
            loans_df: DataFrame with loans, mob, dpd, snapshot_date
            mob_months: Months on book for label (default from config)
            dpd_threshold: DPD threshold for default label (default from config)

        Returns:
            DataFrame with explicit temporal alignment

        Example:
            If features are from 2023-01-01, labels will be from 2023-07-01 (MOB=6)
        """
        if mob_months is None:
            mob_months = self.mob_months
        if dpd_threshold is None:
            dpd_threshold = self.dpd_threshold

        logger.info(
            f"Performing multi-snapshot join (MOB={mob_months}, DPD>={dpd_threshold})"
        )

        # Get unique snapshot dates from both datasets
        feat_dates = sorted(features_df["snapshot_date"].unique())
        loan_dates = sorted(loans_df["snapshot_date"].unique())

        logger.info(f"Feature dates: {len(feat_dates)} snapshots")
        logger.info(f"Loan dates: {len(loan_dates)} snapshots")

        joined_dfs = []
        successful_joins = 0
        skipped_joins = 0

        for feature_date in feat_dates:
            # Calculate label date (feature_date + mob_months)
            label_date = feature_date + relativedelta(months=mob_months)

            # Check if label date exists in loans data
            if label_date not in loan_dates:
                logger.debug(
                    f"Skipping {feature_date}: label date {label_date} not available"
                )
                skipped_joins += 1
                continue

            # Extract features at feature_date
            features_snapshot = features_df[
                features_df["snapshot_date"] == feature_date
            ].copy()

            # Extract loans at label_date with MOB = mob_months
            loans_snapshot = loans_df[
                (loans_df["snapshot_date"] == label_date) &
                (loans_df["mob"] == mob_months)
            ].copy()

            # Create label: default = 1 if DPD >= threshold
            loans_snapshot["label"] = (
                loans_snapshot["dpd"] >= dpd_threshold
            ).astype(int)

            # Select relevant columns from loans
            loans_snapshot = loans_snapshot[[
                "Customer_ID", "label", "dpd", "overdue_amt"
            ]].copy()

            # Rename snapshot_date to feature_snapshot_date
            features_snapshot = features_snapshot.rename(
                columns={"snapshot_date": "feature_snapshot_date"}
            )

            # Add label_snapshot_date
            loans_snapshot["label_snapshot_date"] = label_date

            # Join on Customer_ID
            joined = features_snapshot.merge(
                loans_snapshot,
                on="Customer_ID",
                how="inner"
            )

            if len(joined) > 0:
                joined_dfs.append(joined)
                successful_joins += 1
                logger.debug(
                    f"Joined {feature_date} -> {label_date}: {len(joined):,} rows"
                )

        if not joined_dfs:
            raise ValueError(
                "No valid temporal joins found. Check that feature dates + "
                f"{mob_months} months exist in loans data."
            )

        # Concatenate all joins
        result = pd.concat(joined_dfs, ignore_index=True)

        logger.info(
            f"Multi-snapshot join complete: {successful_joins} successful, "
            f"{skipped_joins} skipped"
        )
        logger.info(f"Total rows: {len(result):,}")
        logger.info(
            f"Label distribution: {result['label'].value_counts().to_dict()}"
        )
        logger.info(
            f"Default rate: {result['label'].mean()*100:.2f}%"
        )

        return result

    def engineer_financial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features from financial data.

        Creates:
        - Log transformations for income
        - Financial ratios (debt-to-income, EMI-to-salary, etc.)
        - Total financial products
        - Utilization metrics

        Args:
            df: DataFrame with financial features

        Returns:
            DataFrame with engineered features
        """
        logger.info("Engineering financial features")

        df = df.copy()

        # Log transformation for income (handles skewness)
        if "Annual_Income" in df.columns:
            df["log_annual_income"] = np.log1p(df["Annual_Income"].fillna(0))

        if "Monthly_Inhand_Salary" in df.columns:
            df["log_monthly_salary"] = np.log1p(df["Monthly_Inhand_Salary"].fillna(0))

        # Financial ratios (with safety: add small value to denominators)
        epsilon = 1.0

        if "Outstanding_Debt" in df.columns and "Annual_Income" in df.columns:
            df["debt_to_income"] = (
                df["Outstanding_Debt"] / (df["Annual_Income"] + epsilon)
            )

        if "Total_EMI_per_month" in df.columns and "Monthly_Inhand_Salary" in df.columns:
            df["emi_to_salary"] = (
                df["Total_EMI_per_month"] / (df["Monthly_Inhand_Salary"] + epsilon)
            )

        if "Monthly_Balance" in df.columns and "Outstanding_Debt" in df.columns:
            df["balance_to_debt"] = (
                (df["Monthly_Balance"] + epsilon) / (df["Outstanding_Debt"] + epsilon)
            )

        if "Amount_invested_monthly" in df.columns and "Monthly_Inhand_Salary" in df.columns:
            df["investment_to_income"] = (
                df["Amount_invested_monthly"] / (df["Monthly_Inhand_Salary"] + epsilon)
            )

        # Total financial products
        if all(col in df.columns for col in ["Num_Bank_Accounts", "Num_Credit_Card", "Num_of_Loan"]):
            df["total_financial_products"] = (
                df["Num_Bank_Accounts"].fillna(0) +
                df["Num_Credit_Card"].fillna(0) +
                df["Num_of_Loan"].fillna(0)
            )

            # Loans per credit product
            df["loans_per_credit_product"] = (
                df["Num_of_Loan"] /
                (df["Num_Bank_Accounts"] + df["Num_Credit_Card"] + epsilon)
            )

        # Credit inquiry intensity
        if "Num_Credit_Inquiries" in df.columns and "Credit_History_Age_Months" in df.columns:
            df["inquiries_per_month"] = (
                df["Num_Credit_Inquiries"] /
                (df["Credit_History_Age_Months"] + epsilon)
            )

        # Payment behavior indicators
        if "Num_of_Delayed_Payment" in df.columns and "Credit_History_Age_Months" in df.columns:
            df["delayed_payments_per_month"] = (
                df["Num_of_Delayed_Payment"] /
                (df["Credit_History_Age_Months"] + epsilon)
            )

        logger.info(f"Engineered {len([c for c in df.columns if c.startswith(('log_', 'debt_', 'emi_', 'balance_', 'investment_', 'total_', 'loans_', 'inquiries_', 'delayed_'))])} financial features")

        return df

    def engineer_attribute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features from attribute data.

        Creates:
        - Age bins
        - Occupation categories

        Args:
            df: DataFrame with attribute features

        Returns:
            DataFrame with engineered features
        """
        logger.info("Engineering attribute features")

        df = df.copy()

        # Age bins
        if "Age" in df.columns:
            df["age_group"] = pd.cut(
                df["Age"],
                bins=[0, 25, 35, 45, 55, 100],
                labels=["18-25", "26-35", "36-45", "46-55", "56+"]
            )

        # Occupation grouping (can be customized based on domain knowledge)
        if "Occupation" in df.columns:
            # Fill missing occupation
            df["Occupation"] = df["Occupation"].fillna("Unknown")

        return df

    def engineer_clickstream_features(
        self,
        df: pd.DataFrame,
        clickstream_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Engineer features from clickstream data.

        Aggregates clickstream features over time windows.

        Args:
            df: Main DataFrame with Customer_ID and feature_snapshot_date
            clickstream_df: Full clickstream data from Silver

        Returns:
            DataFrame with clickstream aggregations
        """
        logger.info("Engineering clickstream features")

        # Get feature columns
        feat_cols = [col for col in clickstream_df.columns if col.startswith("fe_")]

        if not feat_cols:
            logger.warning("No clickstream features found")
            return df

        # For each customer and feature_snapshot_date, aggregate clickstream from past 6 months
        result_rows = []

        for idx, row in df[["Customer_ID", "feature_snapshot_date"]].drop_duplicates().iterrows():
            customer_id = row["Customer_ID"]
            snapshot_date = row["feature_snapshot_date"]

            # Get clickstream data for this customer up to snapshot_date
            customer_clicks = clickstream_df[
                (clickstream_df["Customer_ID"] == customer_id) &
                (clickstream_df["snapshot_date"] <= snapshot_date)
            ].copy()

            if len(customer_clicks) == 0:
                # No clickstream data for this customer
                agg_row = {
                    "Customer_ID": customer_id,
                    "feature_snapshot_date": snapshot_date
                }
                for col in feat_cols:
                    agg_row[f"{col}_mean"] = 0
                    agg_row[f"{col}_max"] = 0
                    agg_row[f"{col}_sum"] = 0
                result_rows.append(agg_row)
                continue

            # Sort by snapshot_date descending
            customer_clicks = customer_clicks.sort_values("snapshot_date", ascending=False)

            # Take last 6 months of data
            customer_clicks = customer_clicks.head(6)

            # Aggregate
            agg_row = {
                "Customer_ID": customer_id,
                "feature_snapshot_date": snapshot_date
            }

            for col in feat_cols:
                agg_row[f"{col}_mean"] = customer_clicks[col].mean()
                agg_row[f"{col}_max"] = customer_clicks[col].max()
                agg_row[f"{col}_sum"] = customer_clicks[col].sum()

            result_rows.append(agg_row)

        clickstream_agg = pd.DataFrame(result_rows)

        # Merge with main DataFrame
        df = df.merge(
            clickstream_agg,
            on=["Customer_ID", "feature_snapshot_date"],
            how="left"
        )

        logger.info(f"Added {len([c for c in clickstream_agg.columns if c not in ['Customer_ID', 'feature_snapshot_date']])} clickstream aggregation features")

        return df

    def cap_outliers(
        self,
        df: pd.DataFrame,
        columns: List[str],
        method: str = "quantile",
        lower_q: float = 0.01,
        upper_q: float = 0.95
    ) -> pd.DataFrame:
        """
        Cap outliers using quantile-based approach.

        Args:
            df: DataFrame with features
            columns: Columns to cap
            method: Method to use ("quantile", "iqr", "zscore")
            lower_q: Lower quantile for capping
            upper_q: Upper quantile for capping

        Returns:
            DataFrame with capped values
        """
        logger.info(f"Capping outliers using {method} method")

        df = df.copy()
        capped_count = 0

        for col in columns:
            if col not in df.columns:
                continue

            if method == "quantile":
                lower_val = df[col].quantile(lower_q)
                upper_val = df[col].quantile(upper_q)

                original_min = df[col].min()
                original_max = df[col].max()

                df[col] = df[col].clip(lower=lower_val, upper=upper_val)

                if original_min < lower_val or original_max > upper_val:
                    logger.debug(
                        f"  {col}: [{original_min:.2f}, {original_max:.2f}] -> "
                        f"[{lower_val:.2f}, {upper_val:.2f}]"
                    )
                    capped_count += 1

        logger.info(f"Capped outliers in {capped_count} columns")

        return df

    def create_feature_and_label_stores(
        self,
        train_start: Optional[str] = None,
        train_end: Optional[str] = None,
        test_start: Optional[str] = None,
        test_end: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create feature store and label store for ML.

        This is the main entry point for Gold layer processing.

        Args:
            train_start: Training start date (YYYY-MM-DD)
            train_end: Training end date (YYYY-MM-DD)
            test_start: Test start date (YYYY-MM-DD)
            test_end: Test end date (YYYY-MM-DD)

        Returns:
            Tuple of (feature_store, label_store) DataFrames

        Example:
            >>> feature_store, label_store = processor.create_feature_and_label_stores()
        """
        logger.info("="*80)
        logger.info("CREATING FEATURE AND LABEL STORES")
        logger.info("="*80)

        start_time = datetime.now()

        # Read Silver data
        logger.info("Reading Silver layer data...")
        financials = self.silver_processor.read_silver_source("financials")
        attributes = self.silver_processor.read_silver_source("attributes")
        clickstream = self.silver_processor.read_silver_source("clickstream")
        loans = self.silver_processor.read_silver_source("loans")

        # Step 1: Create base feature set from financials
        logger.info("\nStep 1: Multi-snapshot join for financials...")
        financials_with_labels = self.multi_snapshot_join(
            financials, loans,
            self.mob_months, self.dpd_threshold
        )

        # Step 2: Engineer financial features
        logger.info("\nStep 2: Engineering financial features...")
        financials_with_labels = self.engineer_financial_features(financials_with_labels)

        # Step 3: Join attributes
        logger.info("\nStep 3: Joining attributes...")
        attributes_aligned = self.multi_snapshot_join(
            attributes, loans,
            self.mob_months, self.dpd_threshold
        )

        attributes_aligned = self.engineer_attribute_features(attributes_aligned)

        # Merge with financials
        feature_store = financials_with_labels.merge(
            attributes_aligned[[
                "Customer_ID", "feature_snapshot_date", "label_snapshot_date",
                "Age", "Occupation", "age_group"
            ]],
            on=["Customer_ID", "feature_snapshot_date", "label_snapshot_date"],
            how="inner"
        )

        logger.info(f"After attributes join: {len(feature_store):,} rows")

        # Step 4: Engineer clickstream features
        logger.info("\nStep 4: Engineering clickstream features...")
        feature_store = self.engineer_clickstream_features(
            feature_store, clickstream
        )

        # Step 5: Cap outliers on numeric columns
        logger.info("\nStep 5: Capping outliers...")
        numeric_cols = feature_store.select_dtypes(include=[np.number]).columns.tolist()

        # Exclude ID, label, and date columns
        exclude_cols = [
            "Customer_ID", "label", "dpd", "overdue_amt",
            "Age", "Num_Bank_Accounts", "Num_Credit_Card", "Num_of_Loan"
        ]
        cols_to_cap = [c for c in numeric_cols if c not in exclude_cols]

        outlier_config = self.feat_config.get("outlier_handling", {})
        lower_q = outlier_config.get("lower_quantile", 0.01)
        upper_q = outlier_config.get("upper_quantile", 0.95)

        feature_store = self.cap_outliers(
            feature_store,
            cols_to_cap,
            method="quantile",
            lower_q=lower_q,
            upper_q=upper_q
        )

        # Step 6: Create label store
        logger.info("\nStep 6: Creating label store...")
        label_store = feature_store[[
            "Customer_ID", "feature_snapshot_date", "label_snapshot_date",
            "label", "dpd", "overdue_amt"
        ]].copy()

        # Step 7: Drop PII and intermediate columns from feature store
        logger.info("\nStep 7: Cleaning feature store...")
        pii_cols = ["Name", "SSN"]
        intermediate_cols = ["dpd", "overdue_amt"]
        label_cols = ["label"]

        cols_to_drop = [
            c for c in pii_cols + intermediate_cols + label_cols
            if c in feature_store.columns
        ]
        feature_store = feature_store.drop(columns=cols_to_drop)

        # Step 8: Handle missing values
        logger.info("\nStep 8: Handling missing values...")

        # Categorical columns: fill with "Unknown"
        categorical_cols = feature_store.select_dtypes(include=["object", "category"]).columns.tolist()
        for col in categorical_cols:
            if col in ["Customer_ID"]:
                continue
            series = feature_store[col]
            if is_categorical_dtype(series):
                if "Unknown" not in series.cat.categories:
                    series = series.cat.add_categories(["Unknown"])
                feature_store[col] = series.fillna("Unknown")
            else:
                feature_store[col] = series.fillna("Unknown")

        # Numeric columns: fill with median
        for col in numeric_cols:
            if col in feature_store.columns and col != "label":
                median_val = feature_store[col].median()
                feature_store[col] = feature_store[col].fillna(median_val)

        # Step 9: Save to Gold layer
        logger.info("\nStep 9: Saving to Gold layer...")
        self._save_feature_store(feature_store)
        self._save_label_store(label_store)

        duration = (datetime.now() - start_time).total_seconds()

        logger.info("\n" + "="*80)
        logger.info("FEATURE AND LABEL STORES CREATED")
        logger.info("="*80)
        logger.info(f"Duration: {duration:.2f} seconds")
        logger.info(f"Feature store shape: {feature_store.shape}")
        logger.info(f"Label store shape: {label_store.shape}")
        logger.info(f"Default rate: {label_store['label'].mean()*100:.2f}%")
        logger.info(f"Features: {len([c for c in feature_store.columns if c not in ['Customer_ID', 'feature_snapshot_date', 'label_snapshot_date', 'label']])}")
        logger.info("="*80 + "\n")

        return feature_store, label_store

    def _save_feature_store(self, df: pd.DataFrame):
        """Save feature store to Gold layer."""
        output_path = self.gold_dir / "feature_store"
        output_path.mkdir(parents=True, exist_ok=True)

        parquet_file = output_path / "features.parquet"
        df.to_parquet(
            parquet_file,
            engine="pyarrow",
            compression="snappy",
            index=False
        )

        logger.info(f"Saved feature store: {len(df):,} rows")

    def _save_label_store(self, df: pd.DataFrame):
        """Save label store to Gold layer."""
        output_path = self.gold_dir / "label_store"
        output_path.mkdir(parents=True, exist_ok=True)

        parquet_file = output_path / "labels.parquet"
        df.to_parquet(
            parquet_file,
            engine="pyarrow",
            compression="snappy",
            index=False
        )

        logger.info(f"Saved label store: {len(df):,} rows")

    def read_feature_store(self) -> pd.DataFrame:
        """Read feature store from Gold layer."""
        parquet_file = self.gold_dir / "feature_store" / "features.parquet"

        if not parquet_file.exists():
            raise FileNotFoundError(
                "Feature store not found. Run create_feature_and_label_stores() first."
            )

        return pd.read_parquet(parquet_file)

    def read_label_store(self) -> pd.DataFrame:
        """Read label store from Gold layer."""
        parquet_file = self.gold_dir / "label_store" / "labels.parquet"

        if not parquet_file.exists():
            raise FileNotFoundError(
                "Label store not found. Run create_feature_and_label_stores() first."
            )

        return pd.read_parquet(parquet_file)


# Convenience function
def create_gold_tables(config_path: str = "config/pipeline_config.yaml") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to create feature and label stores.

    Args:
        config_path: Path to pipeline configuration

    Returns:
        Tuple of (feature_store, label_store)
    """
    processor = GoldProcessor(config_path)
    return processor.create_feature_and_label_stores()
