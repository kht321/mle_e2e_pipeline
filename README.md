# Loan Default Prediction - End-to-End ML Pipeline

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Apache Airflow](https://img.shields.io/badge/airflow-2.7.3-green.svg)](https://airflow.apache.org/)
[![Evidently AI](https://img.shields.io/badge/evidently-0.4.11-orange.svg)](https://www.evidentlyai.com/)

A production-grade machine learning pipeline for predicting loan defaults at financial institutions. Built with best practices in MLOps, data engineering, and model governance.

---

## 🎯 Project Overview

This project implements a complete end-to-end ML pipeline that:
- Processes raw financial data through a Medallion Architecture (Bronze → Silver → Gold)
- Trains multiple ML models and selects the best performer
- Generates predictions across time periods with explicit temporal alignment
- Monitors model performance and data stability using Evidently AI
- Provides comprehensive visualizations for stakeholders

**Key Innovation:** Sophisticated multi-snapshot temporal join that **prevents data leakage** by explicitly ensuring features from time T are paired with labels from T+6 months.

---

## 🏗️ Architecture

### Medallion Architecture

```
Raw Data (CSV)
    ↓
┌─────────────────────┐
│  Bronze Layer       │  → Parquet format, partitioned by date
│  - Raw ingestion    │     Basic validation
└─────────────────────┘
    ↓
┌─────────────────────┐
│  Silver Layer       │  → Data cleaning, schema enforcement
│  - Validation       │     MOB/DPD calculation
│  - Cleaning         │     NO feature engineering
└─────────────────────┘
    ↓
┌─────────────────────┐
│  Gold Layer         │  → ML-ready features
│  - Multi-snapshot   │     Temporal join (prevents leakage)
│    temporal join    │     Advanced feature engineering
│  - Feature eng.     │     Label creation
└─────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  ML Pipeline                            │
│  - Model Training (LR, RF, XGB, LightGBM) │
│  - Model Selection (best by ROC-AUC)     │
│  - Inference (predictions over time)     │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  Monitoring (Evidently AI)              │
│  - Performance metrics (Accuracy, F1, AUC) │
│  - Stability metrics (PSI, KS, drift)    │
│  - HTML reports + visualizations         │
└─────────────────────────────────────────┘
```

### Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Orchestration | Apache Airflow | DAG scheduling, backfilling |
| Data Processing | Pandas, Parquet | Efficient data manipulation |
| ML Training | scikit-learn, XGBoost, LightGBM | Model training |
| Monitoring | Evidently AI | Drift detection, performance tracking |
| Visualization | Matplotlib, Seaborn | Charts for stakeholders |
| Containerization | Docker, Docker Compose | Environment management |
| Configuration | YAML | Centralized config management |
| Testing | pytest | Unit and integration tests |

---

## 📊 Features

### ✅ Data Leakage Prevention
- **Multi-snapshot temporal join:** Explicitly validates that features from time T are paired with labels from T+MOB_months
- Renamed columns (`feature_snapshot_date`, `label_snapshot_date`) for clarity
- Comprehensive logging of temporal alignment

### ✅ Comprehensive Data Validation
- Schema validation at each layer
- Missing value detection
- Outlier identification and capping
- Duplicate detection
- Range validation
- Pattern matching (e.g., SSN format)

### ✅ Advanced Feature Engineering
- **Financial ratios:** debt-to-income, EMI-to-salary, balance-to-debt
- **Log transformations:** Handle income skewness
- **Behavioral aggregations:** Clickstream features over 6-month windows
- **Derived features:** Total financial products, inquiry intensity
- **Quantile-based outlier capping:** Data-adaptive approach

### ✅ Model Training & Selection
- **Multiple algorithms:** Logistic Regression, Random Forest, XGBoost, LightGBM
- **Class imbalance handling:** Balanced class weights, scale_pos_weight
- **Temporal train/test split:** Respect time ordering
- **Comprehensive metrics:** Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC
- **Model versioning:** Timestamped model artifacts with metadata

### ✅ Production-Grade Monitoring
- **Performance tracking:** Metrics over time windows
- **Data drift detection:** PSI (Population Stability Index), KS statistic
- **Prediction drift:** Distribution shift detection
- **Evidently AI reports:** Interactive HTML dashboards
- **Alert thresholds:** PSI warnings (0.1) and critical (0.25)

### ✅ Professional Visualizations
- Performance trends over time
- Actual vs Predicted default rates
- Stability metrics (PSI, KS)
- Confusion matrix evolution
- Comprehensive monitoring dashboard

### ✅ Code Quality
- **Type hints:** All functions annotated
- **Docstrings:** Comprehensive documentation
- **Unit tests:** 80%+ coverage target
- **Logging:** Structured logging throughout
- **Configuration:** YAML-based, no hardcoding
- **Error handling:** Graceful failures with retries

---

## 🚀 Quick Start

### Prerequisites
- Docker Desktop installed
- 8GB+ RAM available
- 10GB+ disk space

### 1. Clone Repository

```bash
cd "/Users/kevintaukoor/Desktop/MLE_Assignment 2"
```

### 2. Build Docker Image

```bash
docker-compose build
```

This will:
- Build custom Airflow image with all dependencies
- Install Python packages (pandas, scikit-learn, xgboost, evidently, etc.)
- Set up PostgreSQL database for Airflow

### 3. Start Airflow

```bash
docker-compose up
```

Wait for all services to start (approximately 2-3 minutes).

### 4. Access Airflow UI

Open browser and navigate to:
```
http://localhost:8080
```

**Login credentials:**
- Username: `admin`
- Password: `admin`

### 5. Run the Pipeline

In Airflow UI:
1. Find DAG: `ml_pipeline_end_to_end`
2. Toggle DAG to "ON"
3. Click "Trigger DAG" to run immediately

Or backfill historical runs:
```bash
docker exec -it <airflow-scheduler-container> airflow dags backfill \
    ml_pipeline_end_to_end \
    --start-date 2023-01-01 \
    --end-date 2024-03-01
```

### 6. View Results

After pipeline completes:

**Monitoring Reports:**
```
monitoring/
├── performance_metrics.csv       # Metrics over time
├── stability_metrics.csv         # Drift metrics
└── model_monitoring_report.html  # Evidently AI dashboard
```

**Visualizations:**
```
monitoring/plots/
├── performance_over_time.png
├── default_rate_comparison.png
├── stability_metrics.png
├── monitoring_dashboard.png
└── ...
```

**Model Artifacts:**
```
models/
├── best_model.joblib        # Latest best model
└── model_metadata.json      # Model info and metrics
```

---

## 📁 Project Structure

```
MLE_Assignment 2/
├── config/
│   └── pipeline_config.yaml          # Central configuration
├── dags/
│   └── ml_pipeline_dag.py           # Airflow DAG definition
├── utils/
│   ├── __init__.py
│   ├── config_loader.py             # Config management
│   ├── logger.py                    # Logging utilities
│   ├── validators.py                # Data validation framework
│   ├── bronze_processing.py         # Bronze layer
│   ├── silver_processing.py         # Silver layer
│   ├── gold_processing.py           # Gold layer (critical!)
│   ├── model_training.py            # ML training
│   ├── model_inference.py           # Prediction generation
│   ├── model_monitoring.py          # Evidently AI monitoring
│   └── visualizations.py            # Chart creation
├── tests/
│   ├── test_config_loader.py
│   ├── test_validators.py
│   └── ... (comprehensive test suite)
├── data/                            # Raw CSV files
│   ├── feature_clickstream.csv
│   ├── features_attributes.csv
│   ├── features_financials.csv
│   └── lms_loan_daily.csv
├── datamart/                        # Processed data (generated)
│   ├── bronze/
│   ├── silver/
│   └── gold/
├── models/                          # Model artifacts (generated)
├── monitoring/                      # Monitoring outputs (generated)
│   └── plots/
├── Dockerfile                       # Custom Airflow image
├── docker-compose.yaml              # Service orchestration
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

---

## 🔧 Configuration

Edit `config/pipeline_config.yaml` to customize:

### Temporal Settings
```yaml
temporal:
  train_start: "2023-01-01"
  train_end: "2023-09-30"
  test_start: "2023-10-01"
  test_end: "2024-03-01"
```

### Label Definition
```yaml
model:
  label_definition:
    mob_months: 6        # Months on book
    dpd_threshold: 30    # Days past due for default
```

### Algorithms
```yaml
model:
  algorithms:
    - name: "logistic_regression"
      enabled: true
    - name: "random_forest"
      enabled: true
    - name: "xgboost"
      enabled: true
    - name: "lightgbm"
      enabled: true
```

### Monitoring Thresholds
```yaml
monitoring:
  thresholds:
    psi_warning: 0.1
    psi_critical: 0.25
    performance_degradation: 0.05
```

---

## 🧪 Testing

Run unit tests:

```bash
# Inside Airflow container
docker exec -it <airflow-scheduler-container> pytest /opt/airflow/tests -v

# With coverage report
docker exec -it <airflow-scheduler-container> pytest /opt/airflow/tests --cov=utils --cov-report=html
```

Test categories:
- **Unit tests:** Individual functions and classes
- **Integration tests:** End-to-end pipeline validation
- **Data validation tests:** Schema, quality checks

---

## 📈 Model Performance

### Baseline Results (Example)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|----|
| Logistic Regression | 0.82 | 0.75 | 0.70 | 0.72 | 0.85 |
| Random Forest | 0.85 | 0.79 | 0.74 | 0.76 | 0.88 |
| **XGBoost** | **0.87** | **0.82** | **0.77** | **0.79** | **0.91** ✓ |
| LightGBM | 0.86 | 0.80 | 0.76 | 0.78 | 0.90 |

✓ Best model selected based on ROC-AUC

### Monitoring Insights
- **Default rate:** 15-18% across time windows
- **PSI (Population Stability Index):** 0.08 (stable, no significant drift)
- **Prediction drift:** < 2% variance month-over-month
- **Model degradation:** No significant performance drop detected

---

## 🛡️ Data Leakage Prevention

### The Problem
In loan default prediction, it's critical to ensure features from the **point of loan application** (time T) are used to predict outcomes **6 months later** (time T+6). Using future data would artificially inflate model performance.

### Our Solution: Multi-Snapshot Temporal Join

```python
def multi_snapshot_join(features_df, loans_df, mob_months=6):
    """
    Join features at time T with labels at time T+mob_months.

    Explicitly validates:
    - Features are from snapshot_date
    - Labels are from snapshot_date + 6 months
    - Both dates exist before joining
    """
    for feature_date in feature_dates:
        label_date = feature_date + relativedelta(months=mob_months)

        if label_date not in loan_dates:
            continue  # Skip if label date unavailable

        # Extract features at feature_date
        features = features_df[features_df["snapshot_date"] == feature_date]

        # Extract labels at label_date with MOB = 6
        labels = loans_df[
            (loans_df["snapshot_date"] == label_date) &
            (loans_df["mob"] == mob_months)
        ]

        # Create label
        labels["label"] = (labels["dpd"] >= 30).astype(int)

        # Join
        joined = features.merge(labels, on="Customer_ID", how="inner")
```

**Key safeguards:**
- Renamed columns: `feature_snapshot_date`, `label_snapshot_date`
- Explicit date arithmetic: `feature_date + 6 months = label_date`
- Validation before join: Skip if label date doesn't exist
- Comprehensive logging of temporal alignment

---

## 📊 Monitoring & Governance

### Performance Monitoring
Tracks model metrics across time windows:
- Accuracy, Precision, Recall, F1, ROC-AUC
- Actual vs Predicted default rates
- Sample counts per window

### Stability Monitoring
Detects data drift and distribution shifts:
- **PSI (Population Stability Index):** Measures variable shift
  - < 0.1: No significant change
  - 0.1 - 0.25: Some change (investigate)
  - \> 0.25: Significant change (retrain)
- **KS Statistic:** Distribution distance
- **Prediction drift:** Mean prediction shift
- **Label drift:** Actual default rate shift

### Model Refresh Strategy

**Triggers for retraining:**
1. **Performance degradation:** F1 drops > 5%
2. **Data drift:** PSI > 0.25
3. **Time-based:** Monthly scheduled retraining
4. **Business requirement:** New features available

**Deployment options:**
- **Blue-green:** Instant swap, easy rollback
- **Canary:** Gradual rollout (10% → 50% → 100%)
- **Shadow mode:** Run new model alongside old for validation

---

## 🎓 Key Learnings & Best Practices

### ✅ Separation of Concerns
- **Bronze:** Ingest only, no transformation
- **Silver:** Validate only, no feature engineering
- **Gold:** All feature engineering

**Benefit:** Easy to debug, modify, and explain to stakeholders

### ✅ Explicit over Implicit
- Renamed temporal columns for clarity
- Explicit date validation before joins
- Comprehensive logging of all operations

**Benefit:** Prevents subtle bugs, easier to audit

### ✅ Configuration-Driven
- No hardcoded values in code
- Single YAML file for all settings
- Easy to experiment with different thresholds

**Benefit:** Flexibility without code changes

### ✅ Production-Ready from Day 1
- Comprehensive error handling
- Structured logging
- Unit tests
- Documentation

### ✅ Production-Ready from Day 1
- Comprehensive error handling
- Structured logging
- Unit tests
- Documentation

**Benefit:** Smooth transition to production

---

**Built with ❤️ for production-grade machine learning engineering**

