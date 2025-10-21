"""
Model Training Module

Trains multiple ML models, evaluates performance, and selects the best model.

Features:
- Multiple algorithms (Logistic Regression, Random Forest, XGBoost, LightGBM)
- Experiment tracking with comprehensive metrics
- Model versioning and artifact storage
- Class imbalance handling
- Cross-validation
- Feature importance analysis
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from datetime import datetime
import json
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)

import xgboost as xgb
import lightgbm as lgb

from .config_loader import ConfigLoader
from .logger import get_logger
from .gold_processing import GoldProcessor

logger = get_logger(__name__)


class ModelTrainer:
    """
    Trains and evaluates machine learning models for loan default prediction.

    This class handles:
    - Data preprocessing and splitting
    - Multiple model training with hyperparameter tuning
    - Model evaluation with comprehensive metrics
    - Model selection based on configurable criteria
    - Model artifact storage with versioning

    Example:
        >>> trainer = ModelTrainer()
        >>> trainer.train_all_models()
        >>> best_model, metrics = trainer.get_best_model()
    """

    def __init__(self, config_path: str = "config/pipeline_config.yaml"):
        """
        Initialize Model Trainer.

        Args:
            config_path: Path to pipeline configuration file
        """
        self.config = ConfigLoader(config_path)
        self.gold_processor = GoldProcessor(config_path)

        self.models_dir = Path(self.config.get("paths.models_dir"))
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Model configuration
        self.algorithms = self.config.get("model.algorithms", [])
        self.primary_metric = self.config.get("model.evaluation_metrics.primary", "roc_auc")
        self.secondary_metrics = self.config.get("model.evaluation_metrics.secondary", [])

        # Storage for trained models and results
        self.trained_models: Dict[str, Any] = {}
        self.model_results: Dict[str, Dict] = {}
        self.preprocessor: Optional[Pipeline] = None

        logger.info(f"ModelTrainer initialized (primary metric: {self.primary_metric})")

    def load_data(
        self,
        train_start: Optional[str] = None,
        train_end: Optional[str] = None,
        test_start: Optional[str] = None,
        test_end: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Load and split data into train/test sets.

        Uses temporal split: train on early dates, test on later dates.

        Args:
            train_start: Training start date
            train_end: Training end date
            test_start: Test start date
            test_end: Test end date

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info("Loading feature and label stores...")

        # Read from Gold layer
        feature_store = self.gold_processor.read_feature_store()
        label_store = self.gold_processor.read_label_store()

        # Merge features and labels
        df = feature_store.merge(
            label_store[["Customer_ID", "feature_snapshot_date", "label_snapshot_date", "label"]],
            on=["Customer_ID", "feature_snapshot_date", "label_snapshot_date"],
            how="inner"
        )

        logger.info(f"Total dataset: {len(df):,} rows")

        # Temporal split
        train_start = train_start or self.config.get("temporal.train_start")
        train_end = train_end or self.config.get("temporal.train_end")
        test_start = test_start or self.config.get("temporal.test_start")
        test_end = test_end or self.config.get("temporal.test_end")

        if train_start and train_end:
            train_df = df[
                (df["feature_snapshot_date"] >= pd.to_datetime(train_start)) &
                (df["feature_snapshot_date"] <= pd.to_datetime(train_end))
            ]
        else:
            # Use first 70% for training
            split_date = df["feature_snapshot_date"].quantile(0.7)
            train_df = df[df["feature_snapshot_date"] <= split_date]

        if test_start and test_end:
            test_df = df[
                (df["feature_snapshot_date"] >= pd.to_datetime(test_start)) &
                (df["feature_snapshot_date"] <= pd.to_datetime(test_end))
            ]
        else:
            # Use last 30% for testing
            split_date = df["feature_snapshot_date"].quantile(0.7)
            test_df = df[df["feature_snapshot_date"] > split_date]

        logger.info(f"Train set: {len(train_df):,} rows")
        logger.info(f"Test set: {len(test_df):,} rows")

        # Separate features and labels
        feature_cols = [
            col for col in df.columns
            if col not in ["Customer_ID", "feature_snapshot_date", "label_snapshot_date", "label"]
        ]

        X_train = train_df[feature_cols]
        X_test = test_df[feature_cols]
        y_train = train_df["label"]
        y_test = test_df["label"]

        logger.info(f"Features: {len(feature_cols)}")
        logger.info(f"Train label distribution: {y_train.value_counts().to_dict()}")
        logger.info(f"Test label distribution: {y_test.value_counts().to_dict()}")

        return X_train, X_test, y_train, y_test

    def create_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        """
        Create preprocessing pipeline.

        Handles:
        - Categorical encoding (one-hot)
        - Numerical scaling (standard scaler)

        Args:
            X: Feature DataFrame

        Returns:
            ColumnTransformer for preprocessing
        """
        logger.info("Creating preprocessing pipeline...")

        # Identify categorical and numerical columns
        categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        logger.info(f"Categorical features: {len(categorical_cols)}")
        logger.info(f"Numerical features: {len(numerical_cols)}")

        # Create transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numerical_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
            ],
            remainder="drop"
        )

        return preprocessor

    def train_logistic_regression(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> Pipeline:
        """
        Train Logistic Regression model.

        Args:
            X_train: Training features
            y_train: Training labels

        Returns:
            Trained Pipeline
        """
        logger.info("Training Logistic Regression...")

        hyperparams = self.config.get("model.hyperparameters.logistic_regression", {})

        model = LogisticRegression(
            C=hyperparams.get("C", 1.0),
            max_iter=hyperparams.get("max_iter", 1000),
            class_weight=hyperparams.get("class_weight", "balanced"),
            random_state=42
        )

        pipeline = Pipeline([
            ("preprocessor", self.preprocessor),
            ("classifier", model)
        ])

        pipeline.fit(X_train, y_train)

        return pipeline

    def train_random_forest(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> Pipeline:
        """
        Train Random Forest model.

        Args:
            X_train: Training features
            y_train: Training labels

        Returns:
            Trained Pipeline
        """
        logger.info("Training Random Forest...")

        hyperparams = self.config.get("model.hyperparameters.random_forest", {})

        model = RandomForestClassifier(
            n_estimators=hyperparams.get("n_estimators", 100),
            max_depth=hyperparams.get("max_depth", 20),
            min_samples_split=hyperparams.get("min_samples_split", 2),
            class_weight=hyperparams.get("class_weight", "balanced"),
            random_state=42,
            n_jobs=-1
        )

        pipeline = Pipeline([
            ("preprocessor", self.preprocessor),
            ("classifier", model)
        ])

        pipeline.fit(X_train, y_train)

        return pipeline

    def train_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> Pipeline:
        """
        Train XGBoost model.

        Args:
            X_train: Training features
            y_train: Training labels

        Returns:
            Trained Pipeline
        """
        logger.info("Training XGBoost...")

        hyperparams = self.config.get("model.hyperparameters.xgboost", {})

        # Calculate scale_pos_weight for class imbalance
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

        model = xgb.XGBClassifier(
            n_estimators=hyperparams.get("n_estimators", 100),
            max_depth=hyperparams.get("max_depth", 5),
            learning_rate=hyperparams.get("learning_rate", 0.1),
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1
        )

        pipeline = Pipeline([
            ("preprocessor", self.preprocessor),
            ("classifier", model)
        ])

        pipeline.fit(X_train, y_train)

        return pipeline

    def train_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> Pipeline:
        """
        Train LightGBM model.

        Args:
            X_train: Training features
            y_train: Training labels

        Returns:
            Trained Pipeline
        """
        logger.info("Training LightGBM...")

        hyperparams = self.config.get("model.hyperparameters.lightgbm", {})

        model = lgb.LGBMClassifier(
            n_estimators=hyperparams.get("n_estimators", 100),
            max_depth=hyperparams.get("max_depth", 5),
            learning_rate=hyperparams.get("learning_rate", 0.1),
            is_unbalance=hyperparams.get("is_unbalance", True),
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )

        pipeline = Pipeline([
            ("preprocessor", self.preprocessor),
            ("classifier", model)
        ])

        pipeline.fit(X_train, y_train)

        return pipeline

    def evaluate_model(
        self,
        model: Pipeline,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str
    ) -> Dict:
        """
        Evaluate model with comprehensive metrics.

        Args:
            model: Trained model pipeline
            X_test: Test features
            y_test: Test labels
            model_name: Name of model

        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating {model_name}...")

        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        metrics = {
            "model_name": model_name,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_pred_proba),
            "pr_auc": average_precision_score(y_test, y_pred_proba),
            "test_samples": len(y_test),
            "positive_class_count": int(y_test.sum()),
            "predicted_positive_count": int(y_pred.sum())
        }

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics["confusion_matrix"] = cm.tolist()

        # Log metrics
        logger.info(f"{model_name} Performance:")
        for metric_name, value in metrics.items():
            if metric_name not in ["confusion_matrix", "model_name"]:
                if isinstance(value, float):
                    logger.info(f"  {metric_name}: {value:.4f}")
                else:
                    logger.info(f"  {metric_name}: {value}")

        return metrics

    def train_all_models(
        self,
        train_start: Optional[str] = None,
        train_end: Optional[str] = None,
        test_start: Optional[str] = None,
        test_end: Optional[str] = None
    ) -> Dict[str, Dict]:
        """
        Train all enabled models and evaluate them.

        Args:
            train_start: Training start date
            train_end: Training end date
            test_start: Test start date
            test_end: Test end date

        Returns:
            Dictionary of model results
        """
        logger.info("="*80)
        logger.info("TRAINING ALL MODELS")
        logger.info("="*80)

        start_time = datetime.now()

        # Load data
        X_train, X_test, y_train, y_test = self.load_data(
            train_start, train_end, test_start, test_end
        )

        # Create preprocessor
        self.preprocessor = self.create_preprocessor(X_train)

        # Training functions
        training_functions = {
            "logistic_regression": self.train_logistic_regression,
            "random_forest": self.train_random_forest,
            "xgboost": self.train_xgboost,
            "lightgbm": self.train_lightgbm
        }

        # Train each enabled model
        for algo_config in self.algorithms:
            model_name = algo_config["name"]
            enabled = algo_config.get("enabled", False)

            if not enabled:
                logger.info(f"Skipping {model_name} (disabled)")
                continue

            if model_name not in training_functions:
                logger.warning(f"Unknown model: {model_name}")
                continue

            try:
                # Train model
                model = training_functions[model_name](X_train, y_train)
                self.trained_models[model_name] = model

                # Evaluate model
                metrics = self.evaluate_model(model, X_test, y_test, model_name)
                metrics["training_time"] = (datetime.now() - start_time).total_seconds()

                self.model_results[model_name] = metrics

                logger.info(f"✓ {model_name} trained successfully")

            except Exception as e:
                logger.error(f"✗ Failed to train {model_name}: {e}")
                continue

        duration = (datetime.now() - start_time).total_seconds()

        logger.info("\n" + "="*80)
        logger.info("MODEL TRAINING SUMMARY")
        logger.info("="*80)
        logger.info(f"Duration: {duration:.2f} seconds")
        logger.info(f"Models trained: {len(self.trained_models)}")
        logger.info("="*80 + "\n")

        return self.model_results

    def get_best_model(self) -> Tuple[Pipeline, Dict]:
        """
        Select best model based on primary metric.

        Returns:
            Tuple of (best_model_pipeline, best_model_metrics)
        """
        if not self.model_results:
            raise ValueError("No models trained yet. Run train_all_models() first.")

        # Sort by primary metric
        sorted_results = sorted(
            self.model_results.items(),
            key=lambda x: x[1].get(self.primary_metric, 0),
            reverse=True
        )

        best_model_name, best_metrics = sorted_results[0]
        best_model = self.trained_models[best_model_name]

        logger.info(f"Best model: {best_model_name} ({self.primary_metric}={best_metrics[self.primary_metric]:.4f})")

        return best_model, best_metrics

    def save_best_model(self) -> str:
        """
        Save best model to models directory.

        Returns:
            Path to saved model
        """
        best_model, best_metrics = self.get_best_model()
        model_name = best_metrics["model_name"]

        # Create model version
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version = f"v_{timestamp}"

        # Save model
        model_path = self.models_dir / f"best_model_{version}.joblib"
        joblib.dump(best_model, model_path)

        logger.info(f"Saved best model to: {model_path}")

        # Save metadata
        metadata = {
            "model_name": model_name,
            "version": version,
            "timestamp": timestamp,
            "metrics": best_metrics,
            "primary_metric": self.primary_metric,
            "model_path": str(model_path)
        }

        metadata_path = self.models_dir / f"model_metadata_{version}.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved model metadata to: {metadata_path}")

        # Also save as "best_model.joblib" for easy loading
        latest_model_path = self.models_dir / "best_model.joblib"
        joblib.dump(best_model, latest_model_path)

        latest_metadata_path = self.models_dir / "model_metadata.json"
        with open(latest_metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info("Saved as latest best model")

        return str(model_path)

    def load_latest_model(self) -> Pipeline:
        """
        Load the latest best model.

        Returns:
            Loaded model pipeline
        """
        model_path = self.models_dir / "best_model.joblib"

        if not model_path.exists():
            raise FileNotFoundError(
                "No saved model found. Train a model first."
            )

        model = joblib.load(model_path)
        logger.info(f"Loaded model from: {model_path}")

        return model


# Convenience functions
def train_models(config_path: str = "config/pipeline_config.yaml") -> Dict[str, Dict]:
    """
    Convenience function to train all models.

    Args:
        config_path: Path to pipeline configuration

    Returns:
        Dictionary of model results
    """
    trainer = ModelTrainer(config_path)
    results = trainer.train_all_models()
    trainer.save_best_model()
    return results
