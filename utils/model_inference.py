"""
Model Inference Module

Makes predictions using trained models across time periods.

Features:
- Load saved models
- Generate predictions on new data
- Store predictions with metadata
- Batch prediction support
- Temporal prediction tracking
"""

from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import json

from .config_loader import ConfigLoader
from .logger import get_logger
from .gold_processing import GoldProcessor

logger = get_logger(__name__)


class ModelInference:
    """
    Performs model inference on feature data.

    Handles:
    - Loading trained models
    - Making predictions across time periods
    - Storing predictions with metadata
    - Tracking prediction history

    Example:
        >>> inference = ModelInference()
        >>> predictions = inference.predict_all_snapshots()
    """

    def __init__(self, config_path: str = "config/pipeline_config.yaml"):
        """
        Initialize Model Inference.

        Args:
            config_path: Path to pipeline configuration file
        """
        self.config = ConfigLoader(config_path)
        self.gold_processor = GoldProcessor(config_path)

        self.models_dir = Path(self.config.get("paths.models_dir"))
        self.gold_dir = Path(self.config.get("paths.gold_dir"))

        self.predictions_dir = self.gold_dir / "predictions"
        self.predictions_dir.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.model_metadata = None

        logger.info("ModelInference initialized")

    def load_model(self, model_path: Optional[str] = None) -> None:
        """
        Load trained model.

        Args:
            model_path: Path to model file (default: latest best model)
        """
        if model_path is None:
            model_path = self.models_dir / "best_model.joblib"
        else:
            model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.model = joblib.load(model_path)
        logger.info(f"Loaded model from: {model_path}")

        # Load metadata if available
        metadata_path = self.models_dir / "model_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                self.model_metadata = json.load(f)
            logger.info(f"Loaded model metadata: {self.model_metadata.get('model_name')}")

    def predict(
        self,
        X: pd.DataFrame,
        return_proba: bool = True
    ) -> pd.DataFrame:
        """
        Make predictions on feature data.

        Args:
            X: Feature DataFrame
            return_proba: Whether to return prediction probabilities

        Returns:
            DataFrame with predictions
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load_model() first.")

        # Make predictions
        y_pred = self.model.predict(X)

        # Create results DataFrame
        results = X[["Customer_ID", "feature_snapshot_date", "label_snapshot_date"]].copy()
        results["prediction"] = y_pred

        if return_proba:
            y_pred_proba = self.model.predict_proba(X)
            results["prediction_proba_0"] = y_pred_proba[:, 0]
            results["prediction_proba_1"] = y_pred_proba[:, 1]

        results["prediction_timestamp"] = datetime.now()

        if self.model_metadata:
            results["model_name"] = self.model_metadata.get("model_name")
            results["model_version"] = self.model_metadata.get("version")

        return results

    def predict_all_snapshots(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Generate predictions across all snapshot dates.

        Args:
            start_date: Start date for predictions (YYYY-MM-DD)
            end_date: End date for predictions (YYYY-MM-DD)

        Returns:
            DataFrame with all predictions
        """
        logger.info("Generating predictions for all snapshots...")

        # Load model if not already loaded
        if self.model is None:
            self.load_model()

        # Load feature store
        feature_store = self.gold_processor.read_feature_store()

        # Filter by date range if specified
        if start_date:
            feature_store = feature_store[
                feature_store["feature_snapshot_date"] >= pd.to_datetime(start_date)
            ]

        if end_date:
            feature_store = feature_store[
                feature_store["feature_snapshot_date"] <= pd.to_datetime(end_date)
            ]

        logger.info(f"Predicting on {len(feature_store):,} samples")

        # Make predictions
        predictions = self.predict(feature_store, return_proba=True)

        logger.info(f"Generated {len(predictions):,} predictions")
        logger.info(
            f"Predicted default rate: {predictions['prediction'].mean()*100:.2f}%"
        )

        # Save predictions
        self._save_predictions(predictions)

        return predictions

    def predict_by_snapshot(
        self,
        snapshot_date: str
    ) -> pd.DataFrame:
        """
        Generate predictions for a specific snapshot date.

        Args:
            snapshot_date: Snapshot date (YYYY-MM-DD)

        Returns:
            DataFrame with predictions for that snapshot
        """
        logger.info(f"Generating predictions for snapshot: {snapshot_date}")

        # Load model if not already loaded
        if self.model is None:
            self.load_model()

        # Load feature store
        feature_store = self.gold_processor.read_feature_store()

        # Filter to specific snapshot
        snapshot_features = feature_store[
            feature_store["feature_snapshot_date"] == pd.to_datetime(snapshot_date)
        ]

        if len(snapshot_features) == 0:
            raise ValueError(f"No features found for snapshot date: {snapshot_date}")

        # Make predictions
        predictions = self.predict(snapshot_features, return_proba=True)

        logger.info(f"Generated {len(predictions):,} predictions for {snapshot_date}")

        return predictions

    def _save_predictions(self, predictions: pd.DataFrame):
        """
        Save predictions to Gold layer.

        Args:
            predictions: DataFrame with predictions
        """
        output_path = self.predictions_dir / "all_predictions.parquet"

        predictions.to_parquet(
            output_path,
            engine="pyarrow",
            compression="snappy",
            index=False
        )

        logger.info(f"Saved predictions to: {output_path}")

        # Also save as CSV for easy inspection
        csv_path = self.predictions_dir / "all_predictions.csv"
        predictions.to_csv(csv_path, index=False)

    def read_predictions(self) -> pd.DataFrame:
        """
        Read saved predictions from Gold layer.

        Returns:
            DataFrame with predictions
        """
        parquet_path = self.predictions_dir / "all_predictions.parquet"

        if not parquet_path.exists():
            raise FileNotFoundError(
                "No predictions found. Run predict_all_snapshots() first."
            )

        predictions = pd.read_parquet(parquet_path)
        logger.info(f"Loaded {len(predictions):,} predictions")

        return predictions


# Convenience function
def generate_predictions(
    config_path: str = "config/pipeline_config.yaml",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Convenience function to generate predictions.

    Args:
        config_path: Path to pipeline configuration
        start_date: Start date for predictions
        end_date: End date for predictions

    Returns:
        DataFrame with predictions
    """
    inference = ModelInference(config_path)
    return inference.predict_all_snapshots(start_date, end_date)
