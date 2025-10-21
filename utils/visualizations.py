"""
Visualization Module

Creates professional charts for monitoring and presentation.

Features:
- Performance metrics over time
- Stability metrics visualization
- Feature importance charts
- Prediction distribution analysis
- Model comparison charts
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from .config_loader import ConfigLoader
from .logger import get_logger

logger = get_logger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class Visualizer:
    """
    Creates visualizations for ML pipeline monitoring.

    Generates publication-quality charts for:
    - Model performance tracking
    - Data drift monitoring
    - Feature analysis
    - Model comparison

    Example:
        >>> viz = Visualizer()
        >>> viz.create_all_visualizations()
    """

    def __init__(self, config_path: str = "config/pipeline_config.yaml"):
        """
        Initialize Visualizer.

        Args:
            config_path: Path to pipeline configuration file
        """
        self.config = ConfigLoader(config_path)

        self.monitoring_dir = Path(self.config.get("paths.monitoring_dir"))
        self.plots_dir = Path(self.config.get("paths.plots_dir"))
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Visualizer initialized")

    def plot_performance_over_time(
        self,
        performance_df: pd.DataFrame,
        metrics: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> str:
        """
        Plot model performance metrics over time.

        Args:
            performance_df: DataFrame with performance metrics
            metrics: List of metrics to plot (default: all available)
            save_path: Path to save plot (default: plots_dir)

        Returns:
            Path to saved plot
        """
        logger.info("Creating performance over time plot...")

        if metrics is None:
            metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]

        # Filter to available metrics
        available_metrics = [m for m in metrics if m in performance_df.columns]

        if not available_metrics:
            logger.warning("No performance metrics found to plot")
            return ""

        # Create plot
        fig, ax = plt.subplots(figsize=(14, 7))

        for metric in available_metrics:
            ax.plot(
                performance_df["snapshot_date"],
                performance_df[metric],
                marker="o",
                label=metric.upper(),
                linewidth=2,
                markersize=6
            )

        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("Model Performance Over Time", fontsize=14, fontweight="bold")
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3)

        # Rotate x-axis labels
        plt.xticks(rotation=45, ha="right")

        plt.tight_layout()

        # Save plot
        if save_path is None:
            save_path = self.plots_dir / "performance_over_time.png"

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved performance plot to: {save_path}")
        return str(save_path)

    def plot_default_rate_comparison(
        self,
        performance_df: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> str:
        """
        Plot actual vs predicted default rates over time.

        Args:
            performance_df: DataFrame with performance metrics
            save_path: Path to save plot

        Returns:
            Path to saved plot
        """
        logger.info("Creating default rate comparison plot...")

        fig, ax = plt.subplots(figsize=(14, 7))

        # Plot actual default rate
        ax.plot(
            performance_df["snapshot_date"],
            performance_df["actual_default_rate"] * 100,
            marker="o",
            label="Actual Default Rate",
            linewidth=2,
            markersize=6,
            color="#e74c3c"
        )

        # Plot predicted default rate
        ax.plot(
            performance_df["snapshot_date"],
            performance_df["predicted_default_rate"] * 100,
            marker="s",
            label="Predicted Default Rate",
            linewidth=2,
            markersize=6,
            color="#3498db"
        )

        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Default Rate (%)", fontsize=12)
        ax.set_title("Actual vs Predicted Default Rate Over Time", fontsize=14, fontweight="bold")
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        if save_path is None:
            save_path = self.plots_dir / "default_rate_comparison.png"

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved default rate plot to: {save_path}")
        return str(save_path)

    def plot_stability_metrics(
        self,
        stability_df: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> str:
        """
        Plot stability metrics (PSI, KS, drift) over time.

        Args:
            stability_df: DataFrame with stability metrics
            save_path: Path to save plot

        Returns:
            Path to saved plot
        """
        logger.info("Creating stability metrics plot...")

        # Identify available metrics
        metric_cols = [col for col in stability_df.columns if col in ["mean_psi", "max_psi", "mean_ks", "max_ks", "prediction_drift", "label_drift"]]

        if not metric_cols:
            logger.warning("No stability metrics found to plot")
            return ""

        # Create subplots
        n_metrics = len(metric_cols)
        fig, axes = plt.subplots(n_metrics, 1, figsize=(14, 4 * n_metrics))

        if n_metrics == 1:
            axes = [axes]

        for ax, metric in zip(axes, metric_cols):
            ax.plot(
                stability_df["snapshot_date"],
                stability_df[metric],
                marker="o",
                linewidth=2,
                markersize=6,
                color="#9b59b6"
            )

            # Add threshold lines for PSI
            if "psi" in metric:
                ax.axhline(y=0.1, color="orange", linestyle="--", label="Warning (0.1)", alpha=0.7)
                ax.axhline(y=0.25, color="red", linestyle="--", label="Critical (0.25)", alpha=0.7)
                ax.legend(loc="best")

            ax.set_xlabel("Date", fontsize=11)
            ax.set_ylabel(metric.replace("_", " ").title(), fontsize=11)
            ax.set_title(f"{metric.replace('_', ' ').title()} Over Time", fontsize=12, fontweight="bold")
            ax.grid(True, alpha=0.3)

            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

        plt.tight_layout()

        if save_path is None:
            save_path = self.plots_dir / "stability_metrics.png"

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved stability plot to: {save_path}")
        return str(save_path)

    def plot_confusion_matrix_evolution(
        self,
        performance_df: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> str:
        """
        Plot how confusion matrix metrics evolve over time.

        Args:
            performance_df: DataFrame with performance metrics
            save_path: Path to save plot

        Returns:
            Path to saved plot
        """
        logger.info("Creating confusion matrix evolution plot...")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        metrics_to_plot = [
            ("accuracy", "Accuracy", axes[0, 0]),
            ("precision", "Precision", axes[0, 1]),
            ("recall", "Recall", axes[1, 0]),
            ("f1", "F1-Score", axes[1, 1])
        ]

        for metric, title, ax in metrics_to_plot:
            if metric in performance_df.columns:
                ax.plot(
                    performance_df["snapshot_date"],
                    performance_df[metric],
                    marker="o",
                    linewidth=2,
                    markersize=6
                )

                ax.set_xlabel("Date", fontsize=11)
                ax.set_ylabel("Score", fontsize=11)
                ax.set_title(title, fontsize=12, fontweight="bold")
                ax.grid(True, alpha=0.3)
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

        plt.suptitle("Classification Metrics Evolution", fontsize=14, fontweight="bold", y=1.00)
        plt.tight_layout()

        if save_path is None:
            save_path = self.plots_dir / "confusion_matrix_evolution.png"

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved confusion matrix evolution plot to: {save_path}")
        return str(save_path)

    def plot_sample_counts(
        self,
        performance_df: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> str:
        """
        Plot sample counts over time.

        Args:
            performance_df: DataFrame with performance metrics
            save_path: Path to save plot

        Returns:
            Path to saved plot
        """
        logger.info("Creating sample counts plot...")

        fig, ax = plt.subplots(figsize=(14, 6))

        ax.bar(
            performance_df["snapshot_date"],
            performance_df["sample_count"],
            color="#3498db",
            alpha=0.7,
            edgecolor="black"
        )

        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Sample Count", fontsize=12)
        ax.set_title("Sample Counts Per Time Window", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        if save_path is None:
            save_path = self.plots_dir / "sample_counts.png"

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved sample counts plot to: {save_path}")
        return str(save_path)

    def create_monitoring_dashboard(
        self,
        performance_df: pd.DataFrame,
        stability_df: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> str:
        """
        Create comprehensive monitoring dashboard.

        Args:
            performance_df: DataFrame with performance metrics
            stability_df: DataFrame with stability metrics
            save_path: Path to save dashboard

        Returns:
            Path to saved dashboard
        """
        logger.info("Creating monitoring dashboard...")

        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        # 1. Performance metrics
        ax1 = fig.add_subplot(gs[0, :])
        metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        for metric in metrics:
            if metric in performance_df.columns:
                ax1.plot(
                    performance_df["snapshot_date"],
                    performance_df[metric],
                    marker="o",
                    label=metric.upper(),
                    linewidth=2
                )
        ax1.set_title("Model Performance Over Time", fontsize=13, fontweight="bold")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Score")
        ax1.legend(loc="best")
        ax1.grid(True, alpha=0.3)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")

        # 2. Default rates
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(
            performance_df["snapshot_date"],
            performance_df["actual_default_rate"] * 100,
            marker="o",
            label="Actual",
            linewidth=2,
            color="#e74c3c"
        )
        ax2.plot(
            performance_df["snapshot_date"],
            performance_df["predicted_default_rate"] * 100,
            marker="s",
            label="Predicted",
            linewidth=2,
            color="#3498db"
        )
        ax2.set_title("Default Rates", fontsize=13, fontweight="bold")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Default Rate (%)")
        ax2.legend(loc="best")
        ax2.grid(True, alpha=0.3)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

        # 3. PSI (if available)
        ax3 = fig.add_subplot(gs[1, 1])
        if "mean_psi" in stability_df.columns:
            ax3.plot(
                stability_df["snapshot_date"],
                stability_df["mean_psi"],
                marker="o",
                linewidth=2,
                color="#9b59b6"
            )
            ax3.axhline(y=0.1, color="orange", linestyle="--", alpha=0.7, label="Warning")
            ax3.axhline(y=0.25, color="red", linestyle="--", alpha=0.7, label="Critical")
        ax3.set_title("Population Stability Index (PSI)", fontsize=13, fontweight="bold")
        ax3.set_xlabel("Date")
        ax3.set_ylabel("Mean PSI")
        ax3.legend(loc="best")
        ax3.grid(True, alpha=0.3)
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha="right")

        # 4. Sample counts
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.bar(
            performance_df["snapshot_date"],
            performance_df["sample_count"],
            color="#3498db",
            alpha=0.7
        )
        ax4.set_title("Sample Counts", fontsize=13, fontweight="bold")
        ax4.set_xlabel("Date")
        ax4.set_ylabel("Count")
        ax4.grid(True, alpha=0.3, axis="y")
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha="right")

        # 5. Prediction drift (if available)
        ax5 = fig.add_subplot(gs[2, 1])
        if "prediction_drift" in stability_df.columns:
            ax5.plot(
                stability_df["snapshot_date"],
                stability_df["prediction_drift"],
                marker="o",
                linewidth=2,
                color="#e67e22"
            )
        ax5.set_title("Prediction Drift", fontsize=13, fontweight="bold")
        ax5.set_xlabel("Date")
        ax5.set_ylabel("Drift")
        ax5.grid(True, alpha=0.3)
        plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha="right")

        plt.suptitle(
            "ML Model Monitoring Dashboard",
            fontsize=16,
            fontweight="bold",
            y=0.995
        )

        if save_path is None:
            save_path = self.plots_dir / "monitoring_dashboard.png"

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved monitoring dashboard to: {save_path}")
        return str(save_path)

    def create_all_visualizations(self) -> Dict[str, str]:
        """
        Create all monitoring visualizations.

        Returns:
            Dictionary of plot paths
        """
        logger.info("="*80)
        logger.info("CREATING ALL VISUALIZATIONS")
        logger.info("="*80)

        # Load metrics
        performance_df = pd.read_csv(self.monitoring_dir / "performance_metrics.csv")
        stability_df = pd.read_csv(self.monitoring_dir / "stability_metrics.csv")

        # Convert dates
        performance_df["snapshot_date"] = pd.to_datetime(performance_df["snapshot_date"])
        stability_df["snapshot_date"] = pd.to_datetime(stability_df["snapshot_date"])

        # Create all plots
        plots = {}

        plots["performance_over_time"] = self.plot_performance_over_time(performance_df)
        plots["default_rate_comparison"] = self.plot_default_rate_comparison(performance_df)
        plots["stability_metrics"] = self.plot_stability_metrics(stability_df)
        plots["confusion_matrix_evolution"] = self.plot_confusion_matrix_evolution(performance_df)
        plots["sample_counts"] = self.plot_sample_counts(performance_df)
        plots["monitoring_dashboard"] = self.create_monitoring_dashboard(performance_df, stability_df)

        logger.info("\n" + "="*80)
        logger.info("ALL VISUALIZATIONS CREATED")
        logger.info("="*80)
        for name, path in plots.items():
            if path:
                logger.info(f"  {name}: {path}")
        logger.info("="*80 + "\n")

        return plots


# Convenience function
def create_visualizations(config_path: str = "config/pipeline_config.yaml") -> Dict[str, str]:
    """
    Convenience function to create all visualizations.

    Args:
        config_path: Path to pipeline configuration

    Returns:
        Dictionary of plot paths
    """
    viz = Visualizer(config_path)
    return viz.create_all_visualizations()
