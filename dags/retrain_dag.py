"""Apache Airflow DAG: automated model retraining with drift-triggered execution."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import Any

from airflow import DAG
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./prices.db")
DRIFT_THRESHOLD_KS: float = float(os.getenv("DRIFT_THRESHOLD_KS", "0.05"))
DRIFT_THRESHOLD_PSI: float = float(os.getenv("DRIFT_THRESHOLD_PSI", "0.2"))
MIN_SAMPLES_FOR_DRIFT: int = int(os.getenv("MIN_SAMPLES_FOR_DRIFT", "50"))

default_args: dict[str, Any] = {
    "owner": "ml-team",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "start_date": datetime(2024, 1, 1),
}


def check_drift(**context: Any) -> bool:
    """Query the database for drift and push result to XCom.

    Returns:
        True if retraining is needed, False otherwise.
    """
    from sqlalchemy import create_engine, text

    from app.monitoring import compute_drift

    engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT predicted_price FROM predictions ORDER BY created_at ASC LIMIT 2000")
        ).fetchall()
    prices = [float(r[0]) for r in rows if r[0] is not None]

    if len(prices) < MIN_SAMPLES_FOR_DRIFT:
        logger.info("Insufficient data (%d samples). Skipping drift check.", len(prices))
        context["ti"].xcom_push(key="should_retrain", value=False)
        return False

    mid = len(prices) // 2
    drift = compute_drift(prices[:mid], prices[mid:])
    psi = float(drift.get("psi", 0.0))
    ks_p = float(drift.get("p_value", 1.0))

    should_retrain = drift["is_drifted"]
    logger.info(
        "Drift check: KS_p=%.4f PSI=%.4f should_retrain=%s",
        ks_p,
        psi,
        should_retrain,
    )
    context["ti"].xcom_push(key="should_retrain", value=should_retrain)
    context["ti"].xcom_push(key="drift_result", value=drift)
    return should_retrain


def retrain_model(**context: Any) -> dict[str, Any]:
    """Retrain the model if drift was detected.

    Returns:
        Training metrics dict, or empty dict if retraining was skipped.
    """
    should_retrain = context["ti"].xcom_pull(key="should_retrain", task_ids="check_drift")
    if not should_retrain:
        logger.info("No drift detected. Skipping retraining.")
        return {}

    logger.info("Drift detected. Starting model retraining...")
    from app.features import generate_synthetic_training_data
    from app.model import train_model

    df = generate_synthetic_training_data(8000)
    metrics = train_model(df)
    logger.info("Retraining complete: %s", metrics)
    return metrics


def send_notification(**context: Any) -> None:
    """Log a notification summary after retraining."""
    should_retrain = context["ti"].xcom_pull(key="should_retrain", task_ids="check_drift")
    drift_result = context["ti"].xcom_pull(key="drift_result", task_ids="check_drift") or {}
    status = "RETRAINED" if should_retrain else "SKIPPED (no drift)"
    logger.info(
        "Retraining pipeline complete. Status=%s KS=%.4f PSI=%.4f",
        status,
        drift_result.get("ks_statistic", 0.0),
        drift_result.get("psi", 0.0),
    )


with DAG(
    dag_id="price_prophet_retrain",
    default_args=default_args,
    description="Automated retraining pipeline with KS + PSI drift detection",
    schedule_interval="0 2 * * *",
    catchup=False,
    tags=["ml", "price-prophet", "retraining"],
) as dag:
    check_drift_task = PythonOperator(
        task_id="check_drift",
        python_callable=check_drift,
    )

    retrain_task = PythonOperator(
        task_id="retrain_model",
        python_callable=retrain_model,
    )

    notify_task = PythonOperator(
        task_id="send_notification",
        python_callable=send_notification,
    )

    check_drift_task >> retrain_task >> notify_task
