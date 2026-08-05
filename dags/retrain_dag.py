"""Rolling-window retraining, daily and on demand.

    blended labels -> train LightGBM + deep model -> MLflow -> champion/challenger

Triggered on a schedule and by ``monitor_dag`` when error or drift breaches a
threshold. That second path is what closes the loop.

Training uses the ``blend`` source: historical trip CSVs supply true labels,
and the live GBFS proxy extends the window forward. Where the two overlap,
ground truth wins.
"""

from __future__ import annotations

import pendulum
from airflow.decorators import dag, task

from src.config import load_config

config = load_config()


@dag(
    dag_id="retrain_dag",
    description="Retrain candidates and run champion/challenger promotion",
    schedule="0 3 * * *",  # 03:00 UTC daily, off-peak
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 1, "retry_delay": pendulum.duration(minutes=5)},
    tags=["training", "mlflow"],
)
def retrain_dag():
    @task
    def train_and_promote(params: dict | None = None) -> dict:
        """Train every configured candidate and promote only a clear winner.

        ``run_training`` handles the comparison; a challenger that fails to beat
        the champion by the configured margin leaves production untouched.
        """
        from src.models.train import run_training

        params = params or {}
        result = run_training(
            source=params.get("source", "blend"),
            window_days=params.get("window_days"),
            config=config,
            promote=True,
        )
        return {
            "promoted": result["promoted"],
            "reason": result["promotion_reason"],
            "candidates": {k: v.get("mae") for k, v in result["candidates"].items()},
            "label_provenance": result["label_provenance"],
        }

    @task
    def refresh_serving(result: dict) -> dict:
        """Tell the API to pick up a newly promoted model.

        Best-effort: the API also refreshes on its own, so a failure here delays
        the switchover rather than breaking it.
        """
        if not result.get("promoted"):
            return {"reloaded": False, "reason": "no promotion this run"}

        import requests

        url = "http://api:8000/reload"
        try:
            response = requests.post(url, timeout=30)
            response.raise_for_status()
            return {"reloaded": True, "model_version": response.json().get("model_version")}
        except Exception as exc:  # noqa: BLE001
            return {"reloaded": False, "reason": f"reload call failed: {exc}"}

    refresh_serving(train_and_promote())


retrain_dag()
