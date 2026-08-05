"""Live GBFS ingestion, every 5 minutes.

    poll station_status -> Postgres snapshots -> proxy departures -> hourly table

The DAG is intentionally thin: it schedules, it does not compute. All logic
lives in ``src.ingestion.pipeline`` so it can be tested without Airflow.

``catchup=False`` and ``max_active_runs=1`` matter here. This task polls a live
feed that only reports the present moment, so replaying missed intervals would
just re-record the current inventory under historical timestamps - fabricating
data that never existed.
"""

from __future__ import annotations

import pendulum
from airflow.decorators import dag, task

from src.config import load_config

config = load_config()
POLL_MINUTES = int(config.get_path("ingestion.poll_interval_minutes", 5))


@dag(
    dag_id="ingest_dag",
    description="Poll the GBFS feed and derive proxy hourly departures",
    schedule=f"*/{POLL_MINUTES} * * * *",
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 2, "retry_delay": pendulum.duration(seconds=30)},
    tags=["ingestion", "gbfs"],
)
def ingest_dag():
    @task
    def refresh_station_metadata() -> dict:
        from src.ingestion.pipeline import ingest_station_information

        return ingest_station_information(config)

    @task
    def poll_station_status() -> dict:
        from src.ingestion.pipeline import ingest_station_status

        return ingest_station_status(config)

    @task
    def derive_proxy_departures(_: dict) -> dict:
        """Inventory deltas -> hourly demand PROXY (not true ride counts)."""
        from src.ingestion.pipeline import derive_and_store_proxy_departures

        return derive_and_store_proxy_departures(config=config)

    refresh_station_metadata() >> derive_proxy_departures(poll_station_status())


ingest_dag()
