"""
Bronze Layer Processing

Ingests raw CSV data and saves to Parquet format with basic validation.

This layer implements the first stage of the Medallion Architecture:
- Reads raw CSV files
- Performs minimal validation (schema, row counts)
- Saves to Parquet for efficient downstream processing
- Preserves all data (no filtering or transformation)
"""

from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime

from .config_loader import ConfigLoader
from .logger import get_logger
from .validators import DataValidator, ValidationLevel

logger = get_logger(__name__)


class BronzeProcessor:
    """
    Processes raw data into Bronze layer.

    Responsibilities:
    - Ingest CSV files
    - Validate basic data quality (schema, row counts, duplicates)
    - Convert to Parquet format
    - Partition temporal data by snapshot_date

    Example:
        >>> processor = BronzeProcessor()
        >>> processor.ingest_all_sources()
    """

    def __init__(self, config_path: str = "config/pipeline_config.yaml"):
        """
        Initialize Bronze processor.

        Args:
            config_path: Path to pipeline configuration file
        """
        self.config = ConfigLoader(config_path)
        self.config.validate()

        self.data_dir = Path(self.config.get("paths.data_dir"))
        self.bronze_dir = Path(self.config.get("paths.bronze_dir"))
        self.bronze_dir.mkdir(parents=True, exist_ok=True)

        self.sources = self.config.get("data_sources")
        self.validator = DataValidator()

        logger.info("BronzeProcessor initialized")

    def ingest_source(
        self,
        source_name: str,
        validate: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Ingest a single data source.

        Args:
            source_name: Name of source (e.g., "clickstream", "loans")
            validate: Whether to perform validation checks

        Returns:
            DataFrame if successful, None otherwise

        Raises:
            FileNotFoundError: If source file doesn't exist
            ValueError: If critical validations fail
        """
        logger.info(f"Ingesting source: {source_name}")

        # Get source configuration
        source_config = self.sources.get(source_name)
        if not source_config:
            raise ValueError(f"Unknown source: {source_name}")

        # Read CSV file
        file_path = self.data_dir / source_config["file"]
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        logger.info(f"Reading CSV: {file_path}")
        df = pd.read_csv(file_path)

        logger.info(
            f"Loaded {len(df):,} rows, {len(df.columns)} columns from {source_name}"
        )

        # Validation
        if validate:
            self._validate_bronze_data(df, source_name, source_config)

            if self.validator.has_critical_failures():
                self.validator.print_results()
                raise ValueError(
                    f"Critical validation failures for {source_name}. "
                    "Data quality issues must be resolved."
                )

        # Save to Bronze layer
        self._save_to_bronze(df, source_name)

        return df

    def _validate_bronze_data(
        self,
        df: pd.DataFrame,
        source_name: str,
        source_config: Dict
    ):
        """
        Validate Bronze layer data quality.

        Args:
            df: DataFrame to validate
            source_name: Name of data source
            source_config: Source configuration dictionary
        """
        logger.info(f"Validating {source_name} data quality")

        # Row count validation
        self.validator.validate_row_count(
            df,
            min_rows=100,
            level=ValidationLevel.CRITICAL
        )

        # Primary key validation
        primary_keys = source_config.get("primary_keys", [])
        if primary_keys:
            # Check that primary key columns exist
            self.validator.validate_schema(
                df,
                primary_keys,
                level=ValidationLevel.CRITICAL
            )

            # Check for nulls in primary keys
            self.validator.validate_missing_values(
                df,
                primary_keys,
                max_missing_pct=0.0,
                level=ValidationLevel.CRITICAL
            )

            # Check for duplicates on primary keys
            self.validator.validate_duplicates(
                df,
                subset=primary_keys,
                level=ValidationLevel.WARNING
            )

        # Snapshot date validation (for temporal sources)
        if "snapshot_date" in df.columns:
            # Validate no missing snapshot_dates
            self.validator.validate_missing_values(
                df,
                ["snapshot_date"],
                max_missing_pct=0.0,
                level=ValidationLevel.CRITICAL
            )

            # Convert to datetime and validate format
            try:
                df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
                logger.info(
                    f"Date range: {df['snapshot_date'].min()} to "
                    f"{df['snapshot_date'].max()}"
                )
            except Exception as e:
                logger.error(f"Failed to parse snapshot_date: {e}")
                raise

    def _save_to_bronze(
        self,
        df: pd.DataFrame,
        source_name: str
    ):
        """
        Save DataFrame to Bronze layer in Parquet format.

        Args:
            df: DataFrame to save
            source_name: Name of data source
        """
        output_path = self.bronze_dir / source_name

        # Convert snapshot_date to datetime if present
        if "snapshot_date" in df.columns:
            df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])

            # Save with partitioning by snapshot_date
            logger.info(f"Saving {source_name} to {output_path} (partitioned)")

            # Partition by year-month for efficiency
            df["year_month"] = df["snapshot_date"].dt.to_period("M").astype(str)

            for year_month, group in df.groupby("year_month"):
                partition_path = output_path / f"year_month={year_month}"
                partition_path.mkdir(parents=True, exist_ok=True)

                # Drop partition column before saving
                group_to_save = group.drop(columns=["year_month"])
                parquet_file = partition_path / "data.parquet"

                group_to_save.to_parquet(
                    parquet_file,
                    engine="pyarrow",
                    compression="snappy",
                    index=False
                )

                logger.info(
                    f"  Saved partition {year_month}: {len(group):,} rows"
                )

        else:
            # Save without partitioning for non-temporal sources
            logger.info(f"Saving {source_name} to {output_path}")
            output_path.mkdir(parents=True, exist_ok=True)

            parquet_file = output_path / "data.parquet"
            df.to_parquet(
                parquet_file,
                engine="pyarrow",
                compression="snappy",
                index=False
            )

        logger.info(f"Successfully saved {source_name} to Bronze layer")

    def ingest_all_sources(self) -> Dict[str, pd.DataFrame]:
        """
        Ingest all configured data sources.

        Returns:
            Dictionary mapping source names to DataFrames

        Raises:
            ValueError: If any critical validations fail
        """
        logger.info("Starting ingestion of all sources")
        start_time = datetime.now()

        results = {}
        failed_sources = []

        for source_name in self.sources.keys():
            try:
                self.validator.clear_results()
                df = self.ingest_source(source_name, validate=True)
                results[source_name] = df
                logger.info(f"✓ Successfully ingested {source_name}")

            except Exception as e:
                logger.error(f"✗ Failed to ingest {source_name}: {e}")
                failed_sources.append(source_name)

        duration = (datetime.now() - start_time).total_seconds()

        # Summary
        logger.info("\n" + "="*80)
        logger.info("BRONZE LAYER INGESTION SUMMARY")
        logger.info("="*80)
        logger.info(f"Duration: {duration:.2f} seconds")
        logger.info(f"Successfully ingested: {len(results)}/{len(self.sources)} sources")

        if failed_sources:
            logger.error(f"Failed sources: {failed_sources}")
            raise ValueError(f"Failed to ingest sources: {failed_sources}")

        logger.info("="*80 + "\n")

        return results

    def read_bronze_source(
        self,
        source_name: str,
        date_filter: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Read data from Bronze layer.

        Args:
            source_name: Name of source to read
            date_filter: Optional year-month filter (e.g., "2023-01")

        Returns:
            DataFrame from Bronze layer

        Example:
            >>> df = processor.read_bronze_source("loans", date_filter="2023-01")
        """
        source_path = self.bronze_dir / source_name

        if not source_path.exists():
            raise FileNotFoundError(
                f"Bronze source not found: {source_name}. "
                "Run ingest_all_sources() first."
            )

        # Check if partitioned
        partitions = list(source_path.glob("year_month=*"))

        if partitions and date_filter:
            # Read specific partition
            partition_path = source_path / f"year_month={date_filter}"
            if not partition_path.exists():
                raise ValueError(f"Partition not found: {date_filter}")

            df = pd.read_parquet(partition_path / "data.parquet")
            logger.info(
                f"Read {len(df):,} rows from {source_name} "
                f"(partition: {date_filter})"
            )

        elif partitions:
            # Read all partitions
            dfs = []
            for partition in sorted(partitions):
                parquet_file = partition / "data.parquet"
                if parquet_file.exists():
                    dfs.append(pd.read_parquet(parquet_file))

            df = pd.concat(dfs, ignore_index=True)
            logger.info(
                f"Read {len(df):,} rows from {source_name} "
                f"({len(partitions)} partitions)"
            )

        else:
            # Read non-partitioned source
            df = pd.read_parquet(source_path / "data.parquet")
            logger.info(f"Read {len(df):,} rows from {source_name}")

        return df


# Convenience functions for direct usage
def ingest_bronze_tables(config_path: str = "config/pipeline_config.yaml") -> Dict[str, pd.DataFrame]:
    """
    Convenience function to ingest all Bronze tables.

    Args:
        config_path: Path to pipeline configuration

    Returns:
        Dictionary of source name to DataFrame

    Example:
        >>> tables = ingest_bronze_tables()
        >>> loans_df = tables["loans"]
    """
    processor = BronzeProcessor(config_path)
    return processor.ingest_all_sources()
