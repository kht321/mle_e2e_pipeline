"""
ML Pipeline DAG

End-to-end ML pipeline for loan default prediction.

This DAG orchestrates:
1. Bronze layer: Raw data ingestion
2. Silver layer: Data cleaning and validation
3. Gold layer: Feature engineering and label creation
4. Model training: Train and select best model
5. Model inference: Generate predictions
6. Monitoring: Track performance and stability
7. Visualization: Create monitoring charts

Schedule: Monthly (can be backfilled)
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

# Import pipeline modules
import sys
sys.path.append("/opt/airflow")

from utils.bronze_processing import ingest_bronze_tables
from utils.silver_processing import process_silver_tables
from utils.gold_processing import create_gold_tables
from utils.model_training import train_models
from utils.model_inference import generate_predictions
from utils.model_monitoring import generate_monitoring_reports
from utils.visualizations import create_visualizations


# Default arguments
default_args = {
    "owner": "mle_team",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# DAG definition
dag = DAG(
    dag_id="ml_pipeline_end_to_end",
    default_args=default_args,
    description="End-to-end ML pipeline for loan default prediction",
    schedule_interval="@monthly",  # Run monthly
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2024, 3, 31),
    catchup=True,  # Enable backfilling
    max_active_runs=1,
    tags=["ml", "production", "loan-default"],
)


# Task 1: Ingest Bronze Layer
def ingest_bronze(**context):
    """Ingest raw data into Bronze layer."""
    print("="*80)
    print("TASK 1: INGESTING BRONZE LAYER")
    print("="*80)

    results = ingest_bronze_tables()

    print(f"\nIngested {len(results)} sources:")
    for source, df in results.items():
        print(f"  - {source}: {len(df):,} rows")

    return f"Bronze ingestion complete: {len(results)} sources"


task_ingest_bronze = PythonOperator(
    task_id="ingest_bronze_layer",
    python_callable=ingest_bronze,
    provide_context=True,
    dag=dag,
)


# Task 2: Process Silver Layer
def process_silver(**context):
    """Clean and validate data for Silver layer."""
    print("="*80)
    print("TASK 2: PROCESSING SILVER LAYER")
    print("="*80)

    results = process_silver_tables()

    print(f"\nProcessed {len(results)} sources:")
    for source, df in results.items():
        print(f"  - {source}: {len(df):,} rows")

    return f"Silver processing complete: {len(results)} sources"


task_process_silver = PythonOperator(
    task_id="process_silver_layer",
    python_callable=process_silver,
    provide_context=True,
    dag=dag,
)


# Task 3: Create Gold Layer
def create_gold(**context):
    """Create feature and label stores."""
    print("="*80)
    print("TASK 3: CREATING GOLD LAYER")
    print("="*80)

    feature_store, label_store = create_gold_tables()

    print(f"\nFeature store: {feature_store.shape}")
    print(f"Label store: {label_store.shape}")
    print(f"Default rate: {label_store['label'].mean()*100:.2f}%")

    return f"Gold layer complete: {feature_store.shape[0]} rows, {feature_store.shape[1]} features"


task_create_gold = PythonOperator(
    task_id="create_gold_layer",
    python_callable=create_gold,
    provide_context=True,
    dag=dag,
)


# Task 4: Train Models
def train_ml_models(**context):
    """Train ML models and select best."""
    print("="*80)
    print("TASK 4: TRAINING ML MODELS")
    print("="*80)

    results = train_models()

    print(f"\nTrained {len(results)} models:")
    for model_name, metrics in results.items():
        print(f"  - {model_name}:")
        print(f"      Accuracy: {metrics.get('accuracy', 0):.4f}")
        print(f"      F1: {metrics.get('f1', 0):.4f}")
        print(f"      ROC-AUC: {metrics.get('roc_auc', 0):.4f}")

    # Find best model
    best_model = max(results.items(), key=lambda x: x[1].get("roc_auc", 0))
    print(f"\nBest model: {best_model[0]} (ROC-AUC: {best_model[1]['roc_auc']:.4f})")

    return f"Training complete: {len(results)} models, best={best_model[0]}"


task_train_models = PythonOperator(
    task_id="train_ml_models",
    python_callable=train_ml_models,
    provide_context=True,
    dag=dag,
)


# Task 5: Generate Predictions
def generate_model_predictions(**context):
    """Generate predictions across time periods."""
    print("="*80)
    print("TASK 5: GENERATING PREDICTIONS")
    print("="*80)

    predictions = generate_predictions()

    print(f"\nGenerated {len(predictions):,} predictions")
    print(f"Predicted default rate: {predictions['prediction'].mean()*100:.2f}%")

    # Summary by snapshot date
    summary = predictions.groupby("feature_snapshot_date").agg({
        "prediction": ["count", "mean"],
        "prediction_proba_1": "mean"
    })

    print("\nPrediction summary by snapshot:")
    print(summary)

    return f"Predictions complete: {len(predictions):,} predictions"


task_generate_predictions = PythonOperator(
    task_id="generate_predictions",
    python_callable=generate_model_predictions,
    provide_context=True,
    dag=dag,
)


# Task 6: Monitor Performance and Stability
def monitor_model(**context):
    """Monitor model performance and stability."""
    print("="*80)
    print("TASK 6: MONITORING MODEL")
    print("="*80)

    reports = generate_monitoring_reports()

    print("\nMonitoring reports generated:")
    for report_name, report_path in reports.items():
        print(f"  - {report_name}: {report_path}")

    return f"Monitoring complete: {len(reports)} reports"


task_monitor_model = PythonOperator(
    task_id="monitor_model",
    python_callable=monitor_model,
    provide_context=True,
    dag=dag,
)


# Task 7: Create Visualizations
def create_viz(**context):
    """Create monitoring visualizations."""
    print("="*80)
    print("TASK 7: CREATING VISUALIZATIONS")
    print("="*80)

    plots = create_visualizations()

    print("\nVisualizations created:")
    for plot_name, plot_path in plots.items():
        if plot_path:
            print(f"  - {plot_name}: {plot_path}")

    return f"Visualizations complete: {len(plots)} plots"


task_create_visualizations = PythonOperator(
    task_id="create_visualizations",
    python_callable=create_viz,
    provide_context=True,
    dag=dag,
)


# Task dependencies
task_ingest_bronze >> task_process_silver >> task_create_gold

# After Gold layer, three parallel paths
task_create_gold >> [task_train_models, task_generate_predictions]

# Training and inference must complete before monitoring
[task_train_models, task_generate_predictions] >> task_monitor_model

# Monitoring must complete before visualization
task_monitor_model >> task_create_visualizations


# Document DAG
dag.doc_md = """
# ML Pipeline for Loan Default Prediction

This DAG implements an end-to-end machine learning pipeline with the following stages:

## Architecture: Medallion (Bronze → Silver → Gold)

### Bronze Layer
- Ingests raw CSV data
- Converts to Parquet format
- Partitions temporal data by snapshot_date
- Performs basic validation

### Silver Layer
- Cleans and validates data
- Handles missing values
- Enforces business rules
- Calculates MOB (Months on Book) and DPD (Days Past Due)
- NO feature engineering (clean separation)

### Gold Layer (CRITICAL for Data Leakage Prevention)
- Multi-snapshot temporal join
  - Features from time T
  - Labels from time T+6 months
  - Explicit validation prevents data leakage
- Advanced feature engineering
  - Financial ratios (debt-to-income, EMI-to-salary)
  - Log transformations
  - Behavioral aggregations
- Label creation: Default = 1 if DPD >= 30 at MOB = 6

### Model Training
- Trains multiple algorithms:
  - Logistic Regression
  - Random Forest
  - XGBoost
  - LightGBM
- Handles class imbalance
- Selects best model based on ROC-AUC
- Saves model artifacts with versioning

### Model Inference
- Loads best trained model
- Generates predictions across time periods
- Stores predictions with probabilities

### Monitoring (Evidently AI)
- Performance metrics: Accuracy, Precision, Recall, F1, AUC
- Stability metrics: PSI, KS statistic, drift detection
- Comprehensive HTML reports
- Time-series tracking

### Visualization
- Performance trends over time
- Actual vs Predicted default rates
- Data drift metrics
- Comprehensive dashboard

## Configuration
- Config file: `config/pipeline_config.yaml`
- Customize: MOB threshold, DPD threshold, metrics, algorithms

## Backfilling
This DAG supports backfilling to simulate historical predictions:
```bash
airflow dags backfill ml_pipeline_end_to_end \\
    --start-date 2023-01-01 \\
    --end-date 2024-03-01
```

## Outputs
- Bronze: `datamart/bronze/`
- Silver: `datamart/silver/`
- Gold: `datamart/gold/`
- Models: `models/`
- Monitoring: `monitoring/`
- Plots: `monitoring/plots/`

## Key Features
✅ Prevents data leakage with multi-snapshot join
✅ Comprehensive validation at each layer
✅ Production-grade monitoring with Evidently AI
✅ Professional visualizations for stakeholders
✅ Model versioning and experiment tracking
✅ Scalable and maintainable code structure

---
**Version:** 1.0.0
**Author:** MLE Team
**Contact:** mle-team@example.com
"""
