# Dockerfile for ML Pipeline with Airflow
# Based on official Airflow image with custom dependencies

FROM apache/airflow:2.7.3-python3.10

# Switch to root to install system dependencies and Python deps globally
USER root

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies globally (disable user-only installs)
ENV PIP_USER=false

# Copy requirements file
COPY requirements.txt /tmp/requirements.txt

# Install Python dependencies
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt
# Ensure base Airflow installation remains intact after dependency resolution
RUN python -m pip install --no-cache-dir "apache-airflow==2.7.3"

# Create necessary directories
RUN mkdir -p \
    /opt/airflow/dags \
    /opt/airflow/utils \
    /opt/airflow/data \
    /opt/airflow/datamart \
    /opt/airflow/models \
    /opt/airflow/monitoring \
    /opt/airflow/config \
    /opt/airflow/logs && \
    chown -R airflow:root /opt/airflow

# Restore default pip behavior for runtime installs
ENV PIP_USER=true

# Set environment variables
ENV PYTHONPATH="/opt/airflow:${PYTHONPATH}"
ENV AIRFLOW__CORE__LOAD_EXAMPLES=False
ENV AIRFLOW__CORE__DAGS_FOLDER=/opt/airflow/dags
ENV AIRFLOW__CORE__ENABLE_XCOM_PICKLING=True

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD airflow jobs check --job-type SchedulerJob --hostname "$${HOSTNAME}" || exit 1

# Switch back to airflow user for runtime
USER airflow

WORKDIR /opt/airflow
