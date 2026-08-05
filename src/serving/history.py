"""Supplies the recent demand history that lag features are computed from.

Primary source is Postgres (``hourly_departures``), populated by the ingestion
DAG and by ``scripts/bootstrap_db.py``. If the database is unreachable, the
service falls back to the local parquet cache so that a dev machine can still
serve predictions - but the fallback is reported in ``/health`` rather than
hidden, because serving on stale history is a degraded state, not a normal one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from src.config import Config, data_dir, load_config

logger = logging.getLogger(__name__)


@dataclass
class HistoryResult:
    """Recent history plus provenance, so callers can report freshness."""

    departures: pd.DataFrame
    source: str
    latest_hour: pd.Timestamp | None

    @property
    def is_empty(self) -> bool:
        return self.departures.empty


def _minimum_history_hours(config: Config) -> int:
    """History deep enough for the longest lag and rolling window, plus slack."""
    longest_lag = max(config.get_path("features.lag_hours", [1, 2, 3, 24, 168]))
    longest_window = max(config.get_path("features.rolling_windows_hours", [24, 168]))
    return int(max(longest_lag, longest_window) * 2)


def load_recent_history(
    config: Config | None = None, *, hours: int | None = None
) -> HistoryResult:
    """Load recent hourly departures, preferring the database."""
    cfg = config or load_config()
    window_hours = hours or _minimum_history_hours(cfg)
    window_days = max(1, int(window_hours / 24) + 1)

    try:
        from src.storage.db import read_hourly_departures

        frame = read_hourly_departures(window_days=window_days, config=cfg)
        if not frame.empty:
            return HistoryResult(frame, "postgres", pd.Timestamp(frame["hour_ts"].max()))

        # The recent window is empty, but the database may still hold older
        # history - which is exactly the situation when the ingestion DAG has not
        # run yet and only the trip-CSV backfill is loaded. Widen the lookback
        # and take the newest slice, rather than falling through to the parquet
        # cache and pretending the database had nothing.
        stale_lookback_days = int(cfg.get_path("serving.stale_history_lookback_days", 730))
        frame = read_hourly_departures(window_days=stale_lookback_days, config=cfg)
        if not frame.empty:
            cutoff = frame["hour_ts"].max() - pd.Timedelta(hours=window_hours)
            frame = frame[frame["hour_ts"] >= cutoff]
            logger.warning(
                "No recent history in the database; serving from the newest "
                "available slice ending %s", frame["hour_ts"].max(),
            )
            return HistoryResult(
                frame, "postgres_stale", pd.Timestamp(frame["hour_ts"].max())
            )
        logger.warning("Database returned no hourly departures; trying the parquet cache")
    except Exception as exc:  # noqa: BLE001 - fall back rather than fail the request
        logger.warning("Database history unavailable (%s); trying the parquet cache", exc)

    cache_path = data_dir("cache", "hourly_departures_all.parquet")
    if cache_path.exists():
        frame = pd.read_parquet(cache_path)
        frame["hour_ts"] = pd.to_datetime(frame["hour_ts"])
        cutoff = frame["hour_ts"].max() - pd.Timedelta(hours=window_hours)
        frame = frame[frame["hour_ts"] >= cutoff]
        return HistoryResult(frame, "parquet_cache", pd.Timestamp(frame["hour_ts"].max()))

    return HistoryResult(
        pd.DataFrame(columns=["station_short_name", "hour_ts", "departures"]), "none", None
    )
