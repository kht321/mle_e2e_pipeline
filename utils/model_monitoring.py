"""
Model Monitoring Module

Monitors model performance and stability using Evidently AI.

Features:
- Performance monitoring (accuracy, precision, recall, F1, AUC)
- Data drift detection (PSI, KS statistic)
- Prediction drift monitoring
- Comprehensive HTML reports with visualizations
- Metric tracking over time
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import json

# Evidently AI imports
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import (
    DataDriftPreset,
    TargetDriftPreset,
    ClassificationPreset
)
from evidently.metrics import (
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
    ClassificationQualityMetric,
    ClassificationClassBalance,
    ClassificationConfusionMatrix
)

from .config_loader import ConfigLoader
from .logger import get_logger
from .gold_processing import GoldProcessor
from .model_inference import ModelInference

logger = get_logger(__name__)


class ModelMonitor:
    """
    Monitors ML model performance and stability over time using Evidently AI.

    Tracks:
    - Model performance metrics (accuracy, precision, recall, F1, AUC)
    - Data drift (feature distribution changes)
    - Prediction drift (prediction distribution changes)
    - Target drift (label distribution changes)

    Example:
        >>> monitor = ModelMonitor()
        >>> monitor.generate_monitoring_reports()
    """

    def __init__(self, config_path: str = "config/pipeline_config.yaml"):
        """
        Initialize Model Monitor.

        Args:
            config_path: Path to pipeline configuration file
        """
        self.config = ConfigLoader(config_path)
        self.gold_processor = GoldProcessor(config_path)
        self.inference = ModelInference(config_path)

        self.monitoring_dir = Path(self.config.get("paths.monitoring_dir"))
        self.monitoring_dir.mkdir(parents=True, exist_ok=True)

        self.plots_dir = Path(self.config.get("paths.plots_dir"))
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        # Monitoring configuration
        self.performance_metrics = self.config.get("monitoring.performance_metrics", [])
        self.stability_metrics = self.config.get("monitoring.stability_metrics", [])

        logger.info("ModelMonitor initialized")

    def calculate_performance_metrics(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
        y_pred_proba: Optional[pd.Series] = None,
        snapshot_date: Optional[pd.Timestamp] = None
    ) -> Dict:
        """
        Calculate performance metrics for a given time window.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities (optional)
            snapshot_date: Snapshot date for this window

        Returns:
            Dictionary of performance metrics
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, average_precision_score
        )

        metrics = {
            "snapshot_date": snapshot_date,
            "sample_count": len(y_true),
            "actual_default_rate": y_true.mean(),
            "predicted_default_rate": y_pred.mean()
        }

        # Classification metrics
        if "accuracy" in self.performance_metrics:
            metrics["accuracy"] = accuracy_score(y_true, y_pred)

        if "precision" in self.performance_metrics:
            metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)

        if "recall" in self.performance_metrics:
            metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)

        if "f1" in self.performance_metrics:
            metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)

        # AUC metrics (require probabilities)
        if y_pred_proba is not None:
            if "roc_auc" in self.performance_metrics:
                try:
                    metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba)
                except:
                    metrics["roc_auc"] = np.nan

            if "pr_auc" in self.performance_metrics:
                try:
                    metrics["pr_auc"] = average_precision_score(y_true, y_pred_proba)
                except:
                    metrics["pr_auc"] = np.nan

        return metrics

    def calculate_stability_metrics(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        snapshot_date: Optional[pd.Timestamp] = None
    ) -> Dict:
        """
        Calculate stability metrics (drift) between reference and current data.

        Args:
            reference_data: Reference dataset (e.g., training data)
            current_data: Current dataset (e.g., new production data)
            snapshot_date: Snapshot date for current data

        Returns:
            Dictionary of stability metrics
        """
        metrics = {
            "snapshot_date": snapshot_date,
            "reference_sample_count": len(reference_data),
            "current_sample_count": len(current_data)
        }

        # PSI (Population Stability Index) for numeric features
        if "psi" in self.stability_metrics:
            psi_scores = self._calculate_psi(reference_data, current_data)
            metrics["mean_psi"] = np.mean(list(psi_scores.values()))
            metrics["max_psi"] = np.max(list(psi_scores.values()))
            metrics["psi_scores"] = psi_scores

        # KS statistic
        if "ks_statistic" in self.stability_metrics:
            ks_scores = self._calculate_ks(reference_data, current_data)
            metrics["mean_ks"] = np.mean(list(ks_scores.values()))
            metrics["max_ks"] = np.max(list(ks_scores.values()))
            metrics["ks_scores"] = ks_scores

        # Prediction drift
        if "prediction_drift" in self.stability_metrics:
            if "prediction_proba_1" in current_data.columns and "prediction_proba_1" in reference_data.columns:
                ref_mean = reference_data["prediction_proba_1"].mean()
                curr_mean = current_data["prediction_proba_1"].mean()
                metrics["prediction_drift"] = abs(curr_mean - ref_mean)

        # Label drift
        if "label_drift" in self.stability_metrics:
            if "label" in current_data.columns and "label" in reference_data.columns:
                ref_rate = reference_data["label"].mean()
                curr_rate = current_data["label"].mean()
                metrics["label_drift"] = abs(curr_rate - ref_rate)

        return metrics

    def _calculate_psi(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        bins: int = 10
    ) -> Dict[str, float]:
        """
        Calculate Population Stability Index (PSI) for numeric features.

        PSI measures how much a variable has shifted between two datasets.
        PSI < 0.1: No significant change
        PSI 0.1-0.25: Some change
        PSI > 0.25: Significant change

        Args:
            reference_data: Reference dataset
            current_data: Current dataset
            bins: Number of bins for PSI calculation

        Returns:
            Dictionary of PSI scores per feature
        """
        numeric_cols = reference_data.select_dtypes(include=[np.number]).columns.tolist()

        # Exclude ID and label columns
        exclude_cols = ["Customer_ID", "label", "prediction", "prediction_proba_0", "prediction_proba_1"]
        numeric_cols = [c for c in numeric_cols if c not in exclude_cols]

        psi_scores = {}

        for col in numeric_cols:
            if col not in current_data.columns:
                continue

            try:
                # Get values
                ref_vals = reference_data[col].dropna()
                curr_vals = current_data[col].dropna()

                if len(ref_vals) == 0 or len(curr_vals) == 0:
                    continue

                # Create bins based on reference data
                _, bin_edges = np.histogram(ref_vals, bins=bins)

                # Count samples in each bin
                ref_counts, _ = np.histogram(ref_vals, bins=bin_edges)
                curr_counts, _ = np.histogram(curr_vals, bins=bin_edges)

                # Calculate proportions (avoid division by zero)
                ref_props = (ref_counts + 1) / (len(ref_vals) + bins)
                curr_props = (curr_counts + 1) / (len(curr_vals) + bins)

                # Calculate PSI
                psi = np.sum((curr_props - ref_props) * np.log(curr_props / ref_props))
                psi_scores[col] = float(psi)

            except Exception as e:
                logger.debug(f"Failed to calculate PSI for {col}: {e}")
                continue

        return psi_scores

    def _calculate_ks(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate Kolmogorov-Smirnov statistic for numeric features.

        KS statistic measures the maximum distance between cumulative distributions.

        Args:
            reference_data: Reference dataset
            current_data: Current dataset

        Returns:
            Dictionary of KS scores per feature
        """
        from scipy.stats import ks_2samp

        numeric_cols = reference_data.select_dtypes(include=[np.number]).columns.tolist()

        # Exclude ID and label columns
        exclude_cols = ["Customer_ID", "label", "prediction", "prediction_proba_0", "prediction_proba_1"]
        numeric_cols = [c for c in numeric_cols if c not in exclude_cols]

        ks_scores = {}

        for col in numeric_cols:
            if col not in current_data.columns:
                continue

            try:
                ref_vals = reference_data[col].dropna()
                curr_vals = current_data[col].dropna()

                if len(ref_vals) == 0 or len(curr_vals) == 0:
                    continue

                # Calculate KS statistic
                ks_stat, p_value = ks_2samp(ref_vals, curr_vals)
                ks_scores[col] = float(ks_stat)

            except Exception as e:
                logger.debug(f"Failed to calculate KS for {col}: {e}")
                continue

        return ks_scores

    def generate_evidently_report(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        report_name: str = "model_monitoring_report"
    ) -> str:
        """
        Generate comprehensive Evidently AI monitoring report.

        Args:
            reference_data: Reference dataset (training data)
            current_data: Current dataset (production data)
            report_name: Name for the report file

        Returns:
            Path to generated HTML report
        """
        logger.info("Generating Evidently AI monitoring report...")

        # Identify numeric and categorical features
        numeric_features = reference_data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = reference_data.select_dtypes(include=["object", "category"]).columns.tolist()

        # Exclude metadata columns
        exclude_cols = [
            "Customer_ID", "feature_snapshot_date", "label_snapshot_date",
            "prediction_timestamp", "model_name", "model_version"
        ]

        numeric_features = [c for c in numeric_features if c not in exclude_cols]
        categorical_features = [c for c in categorical_features if c not in exclude_cols]

        # Create column mapping
        column_mapping = ColumnMapping()

        if "label" in current_data.columns:
            column_mapping.target = "label"
            column_mapping.prediction = "prediction" if "prediction" in current_data.columns else None

        column_mapping.numerical_features = numeric_features
        column_mapping.categorical_features = categorical_features

        # Create report with multiple presets and metrics
        report = Report(metrics=[
            DataDriftPreset(),
            TargetDriftPreset(),
            ClassificationPreset() if "label" in current_data.columns else DatasetDriftMetric(),
            DatasetMissingValuesMetric(),
            ClassificationQualityMetric() if "label" in current_data.columns and "prediction" in current_data.columns else None,
        ])

        # Run report
        report.run(
            reference_data=reference_data,
            current_data=current_data,
            column_mapping=column_mapping
        )

        # Save HTML report
        report_path = self.monitoring_dir / f"{report_name}.html"
        report.save_html(str(report_path))

        logger.info(f"Saved Evidently report to: {report_path}")

        return str(report_path)

    def monitor_over_time(
        self,
        window_size: str = "M"  # Monthly windows
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Monitor model performance and stability over time windows.

        Args:
            window_size: Time window size ("W"=weekly, "M"=monthly)

        Returns:
            Tuple of (performance_df, stability_df)
        """
        logger.info("Monitoring model over time...")

        # Load feature store, label store, and predictions
        feature_store = self.gold_processor.read_feature_store()
        label_store = self.gold_processor.read_label_store()
        predictions = self.inference.read_predictions()

        # Merge everything
        data = feature_store.merge(
            label_store[["Customer_ID", "feature_snapshot_date", "label_snapshot_date", "label"]],
            on=["Customer_ID", "feature_snapshot_date", "label_snapshot_date"],
            how="inner"
        )

        data = data.merge(
            predictions[["Customer_ID", "feature_snapshot_date", "prediction", "prediction_proba_1"]],
            on=["Customer_ID", "feature_snapshot_date"],
            how="inner"
        )

        # Sort by date
        data = data.sort_values("feature_snapshot_date")

        # Define reference period (first window)
        reference_end = data["feature_snapshot_date"].min() + pd.DateOffset(months=3)
        reference_data = data[data["feature_snapshot_date"] <= reference_end]

        logger.info(f"Reference period: {len(reference_data):,} samples")

        # Calculate metrics for each time window
        performance_results = []
        stability_results = []

        # Group by time window
        data["time_window"] = data["feature_snapshot_date"].dt.to_period(window_size)

        for window, window_data in data.groupby("time_window"):
            snapshot_date = window_data["feature_snapshot_date"].iloc[0]

            # Performance metrics
            perf_metrics = self.calculate_performance_metrics(
                y_true=window_data["label"],
                y_pred=window_data["prediction"],
                y_pred_proba=window_data["prediction_proba_1"],
                snapshot_date=snapshot_date
            )
            performance_results.append(perf_metrics)

            # Stability metrics (compare to reference)
            stab_metrics = self.calculate_stability_metrics(
                reference_data=reference_data,
                current_data=window_data,
                snapshot_date=snapshot_date
            )
            stability_results.append(stab_metrics)

        # Convert to DataFrames
        performance_df = pd.DataFrame(performance_results)
        stability_df = pd.DataFrame(stability_results)

        # Save results
        performance_df.to_csv(
            self.monitoring_dir / "performance_metrics.csv",
            index=False
        )

        stability_df.to_csv(
            self.monitoring_dir / "stability_metrics.csv",
            index=False
        )

        logger.info(f"Performance metrics: {len(performance_df)} windows")
        logger.info(f"Stability metrics: {len(stability_df)} windows")

        return performance_df, stability_df

    def generate_monitoring_reports(self) -> Dict[str, str]:
        """
        Generate all monitoring reports and metrics.

        Returns:
            Dictionary of report paths
        """
        logger.info("="*80)
        logger.info("GENERATING MONITORING REPORTS")
        logger.info("="*80)

        start_time = datetime.now()

        # Monitor over time
        performance_df, stability_df = self.monitor_over_time()

        # Load data for Evidently report
        feature_store = self.gold_processor.read_feature_store()
        label_store = self.gold_processor.read_label_store()
        predictions = self.inference.read_predictions()

        # Merge for report
        data = feature_store.merge(
            label_store[["Customer_ID", "feature_snapshot_date", "label_snapshot_date", "label"]],
            on=["Customer_ID", "feature_snapshot_date", "label_snapshot_date"],
            how="inner"
        )

        data = data.merge(
            predictions[["Customer_ID", "feature_snapshot_date", "prediction", "prediction_proba_1"]],
            on=["Customer_ID", "feature_snapshot_date"],
            how="inner"
        )

        # Split into reference and current
        split_date = data["feature_snapshot_date"].quantile(0.5)
        reference_data = data[data["feature_snapshot_date"] <= split_date]
        current_data = data[data["feature_snapshot_date"] > split_date]

        # Generate Evidently report
        report_path = self.generate_evidently_report(
            reference_data=reference_data,
            current_data=current_data,
            report_name="model_monitoring_report"
        )

        duration = (datetime.now() - start_time).total_seconds()

        logger.info("\n" + "="*80)
        logger.info("MONITORING REPORTS GENERATED")
        logger.info("="*80)
        logger.info(f"Duration: {duration:.2f} seconds")
        logger.info(f"Performance metrics: {self.monitoring_dir / 'performance_metrics.csv'}")
        logger.info(f"Stability metrics: {self.monitoring_dir / 'stability_metrics.csv'}")
        logger.info(f"Evidently report: {report_path}")
        logger.info("="*80 + "\n")

        return {
            "performance_metrics": str(self.monitoring_dir / "performance_metrics.csv"),
            "stability_metrics": str(self.monitoring_dir / "stability_metrics.csv"),
            "evidently_report": report_path
        }


# Convenience function
def generate_monitoring_reports(config_path: str = "config/pipeline_config.yaml") -> Dict[str, str]:
    """
    Convenience function to generate monitoring reports.

    Args:
        config_path: Path to pipeline configuration

    Returns:
        Dictionary of report paths
    """
    monitor = ModelMonitor(config_path)
    return monitor.generate_monitoring_reports()
