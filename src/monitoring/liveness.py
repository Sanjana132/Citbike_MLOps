"""Pipeline liveness: is data still arriving at all?

This exists because of a real incident. The Airflow bind mounts went stale on
the host, ``/opt/airflow/dags`` became empty inside the container, Airflow
quietly marked every DAG inactive, and **ingestion stopped for 42 hours without
a single alert**. Nothing was broken in a way any existing check could see: the
containers were healthy, the API served predictions, the scheduler heartbeat was
fine, and drift/error monitoring reported nothing because it needs *new* data to
detect a problem and no new data was arriving.

That is the blind spot this module covers. Drift and error monitoring answer
"is the model still good?". They cannot answer "is the pipeline still running?",
and a silent pipeline looks identical to a calm one:

* No fresh snapshots means no fresh predictions to score, so rolling MAE simply
  stops updating rather than degrading.
* No fresh features means the drift comparison has nothing current to compare,
  so it reports "not evaluated" - which is easy to read as "fine".

So liveness is checked directly against the newest row in each table, and a
breach is escalated by email rather than only logged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

import pandas as pd

from src.config import Config, load_config

logger = logging.getLogger(__name__)


@dataclass
class LivenessCheck:
    """Freshness of one data stream."""

    name: str
    latest: datetime | None
    age_minutes: float | None
    threshold_minutes: float
    healthy: bool
    detail: str

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "latest": self.latest.isoformat() if self.latest else None,
            "age_minutes": self.age_minutes,
            "threshold_minutes": self.threshold_minutes,
            "healthy": self.healthy,
            "detail": self.detail,
        }


@dataclass
class LivenessReport:
    """Outcome of all liveness checks."""

    checks: list[LivenessCheck] = field(default_factory=list)

    @property
    def healthy(self) -> bool:
        return all(check.healthy for check in self.checks)

    @property
    def failures(self) -> list[LivenessCheck]:
        return [check for check in self.checks if not check.healthy]

    def summary(self) -> str:
        if self.healthy:
            return "All data streams are fresh"
        return "; ".join(check.detail for check in self.failures)

    def as_dict(self) -> dict:
        return {
            "healthy": self.healthy,
            "checks": [check.as_dict() for check in self.checks],
            "summary": self.summary(),
        }


def _age_minutes(latest: datetime | None) -> float | None:
    if latest is None:
        return None
    stamp = pd.Timestamp(latest)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("UTC").tz_localize(None)
    now = pd.Timestamp(datetime.now(tz=UTC)).tz_localize(None)
    return float((now - stamp).total_seconds() / 60.0)


def _check(name: str, latest: datetime | None, threshold_minutes: float) -> LivenessCheck:
    age = _age_minutes(latest)
    if latest is None:
        return LivenessCheck(
            name, None, None, threshold_minutes, False,
            f"{name}: no rows at all - the pipeline has never written data",
        )
    if age is not None and age > threshold_minutes:
        return LivenessCheck(
            name, latest, age, threshold_minutes, False,
            f"{name}: newest row is {age:.0f} min old, over the "
            f"{threshold_minutes:.0f} min threshold (last seen {latest})",
        )
    return LivenessCheck(
        name, latest, age, threshold_minutes, True,
        f"{name}: fresh ({age:.0f} min old)" if age is not None else f"{name}: fresh",
    )


def _latest_timestamp(engine, table: str, column: str) -> datetime | None:
    from sqlalchemy import text

    try:
        with engine.connect() as connection:
            value = connection.execute(text(f"SELECT MAX({column}) FROM {table}")).scalar()
        return pd.Timestamp(value).to_pydatetime() if value is not None else None
    except Exception as exc:  # noqa: BLE001
        logger.error("Could not read freshness of %s.%s: %s", table, column, exc)
        return None


def check_pipeline_liveness(config: Config | None = None) -> LivenessReport:
    """Check that each ingestion stream is still receiving data.

    Thresholds are generous multiples of the polling interval, so a single
    missed run does not page anyone - only a genuinely stopped pipeline does.
    """
    cfg = config or load_config()
    from src.storage.db import get_engine

    poll_minutes = float(cfg.get_path("ingestion.poll_interval_minutes", 5))
    snapshot_threshold = float(
        cfg.get_path("monitoring.liveness.snapshot_max_age_minutes", poll_minutes * 6)
    )
    departures_threshold = float(
        cfg.get_path("monitoring.liveness.departures_max_age_minutes", 180)
    )

    engine = get_engine()
    report = LivenessReport()
    report.checks.append(
        _check(
            "gbfs_snapshots",
            _latest_timestamp(engine, "status_snapshots", "snapshot_ts"),
            snapshot_threshold,
        )
    )
    report.checks.append(
        _check(
            "hourly_departures (proxy)",
            _latest_timestamp(engine, "hourly_departures", "hour_ts"),
            departures_threshold,
        )
    )

    if report.healthy:
        logger.info("Pipeline liveness: %s", report.summary())
    else:
        logger.error("Pipeline liveness FAILED: %s", report.summary())
    return report


def alert_on_stalled_pipeline(config: Config | None = None) -> LivenessReport:
    """Check liveness and email if a stream has stalled.

    Deliberately separate from the drift/error triggers: a stalled pipeline is
    an *operational* failure, not a model-quality one, and retraining would not
    fix it. The right response is to tell a human.
    """
    cfg = config or load_config()
    report = check_pipeline_liveness(cfg)
    if report.healthy:
        return report

    if not bool(cfg.get_path("monitoring.liveness.alert_enabled", True)):
        logger.info("Liveness alerting is disabled; not sending an email")
        return report

    try:
        from src.alerting.email import alert_recipients, send_email

        body = "\n".join(
            [
                "A data pipeline has stopped producing rows.",
                "",
                report.summary(),
                "",
                "Detail:",
                *[
                    f"  {c.name}: latest={c.latest} age={c.age_minutes and round(c.age_minutes)}min "
                    f"threshold={c.threshold_minutes:.0f}min healthy={c.healthy}"
                    for c in report.checks
                ],
                "",
                "This is an operational failure, not model degradation, so no",
                "retraining has been triggered - retraining cannot fix an",
                "ingestion outage. Check that the ingestion DAG is unpaused and",
                "that the Airflow dags/ mount is present inside the container.",
                "",
                "-- citibike-demand-mlops",
            ]
        )
        send_email(
            "[citibike-mlops] ALERT: ingestion pipeline has stalled",
            body,
            alert_recipients(cfg),
        )
    except Exception as exc:  # noqa: BLE001 - alerting must not mask the finding
        logger.error("Liveness alert could not be sent: %s", exc)

    return report
