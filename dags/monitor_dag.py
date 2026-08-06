"""Hourly monitoring - the task that closes the retraining loop.

    predictions + realised demand -> rolling MAE
    served features vs training features -> Evidently drift
    either threshold breached -> TriggerDagRunOperator -> retrain_dag

``TriggerDagRunOperator`` is used rather than the REST API: it needs no
credentials, shows the causal link between the monitoring run and the retrain it
caused directly in the Airflow UI, and cannot silently fail on an auth change.
"""

from __future__ import annotations

import pendulum
from airflow.decorators import dag, task
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

from src.config import load_config

config = load_config()


@dag(
    dag_id="monitor_dag",
    description="Rolling accuracy + drift; triggers retraining when breached",
    schedule="@hourly",
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 1, "retry_delay": pendulum.duration(minutes=2)},
    tags=["monitoring", "evidently"],
)
def monitor_dag():
    @task
    def check_liveness() -> dict:
        """Alert if ingestion has stopped.

        Runs before the model checks and independently of them: a stalled
        pipeline is an operational failure that retraining cannot fix, and it
        makes the drift/error signals meaningless rather than alarming.
        """
        from src.monitoring.liveness import alert_on_stalled_pipeline

        return alert_on_stalled_pipeline(config).as_dict()

    @task
    def evaluate() -> dict:
        """Score recent predictions and measure drift."""
        from src.monitoring.pipeline import run_monitoring_cycle

        outcome = run_monitoring_cycle(config)
        return {
            "should_retrain": outcome.should_retrain,
            "summary": outcome.summary(),
            "accuracy": outcome.accuracy,
            "drift": outcome.drift,
        }

    @task.branch
    def decide(result: dict) -> str:
        """Route to a retrain or to a no-op, based on the thresholds in config."""
        return "trigger_retrain" if result["should_retrain"] else "no_action"

    @task
    def no_action() -> str:
        return "within thresholds; no retraining triggered"

    trigger_retrain = TriggerDagRunOperator(
        task_id="trigger_retrain",
        trigger_dag_id="retrain_dag",
        # Blend true historical labels with the live proxy for the retrain.
        conf={"source": "blend"},
        wait_for_completion=False,
        reset_dag_run=True,
    )

    liveness = check_liveness()
    result = evaluate()
    branch = decide(result)
    liveness >> result
    branch >> [trigger_retrain, no_action()]


monitor_dag()
