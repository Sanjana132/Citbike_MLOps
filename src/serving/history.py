"""Supplies the recent demand history that lag features are computed from.

Primary source is Postgres (``hourly_departures``), populated by the ingestion
DAG and by ``scripts/bootstrap_db.py``. If the database is unreachable, the
service falls back to the local parquet cache so that a dev machine can still
serve predictions - but the fallback is reported in ``/health`` rather than
hidden, because serving on stale history is a degraded state, not a normal one.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime

import pandas as pd

from src.config import Config, data_dir, load_config

logger = logging.getLogger(__name__)

# Hourly departures change at most once an hour, but every prediction re-read
# them. The cache is keyed on the requested window and guarded by a lock so
# concurrent requests do not stampede the database.
_CACHE: dict[int, tuple[float, HistoryResult]] = {}
_CACHE_LOCK = threading.Lock()


def invalidate_history_cache() -> None:
    """Drop cached history. Called after ingestion writes new rows."""
    with _CACHE_LOCK:
        _CACHE.clear()


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
    """History deep enough for the longest lag and rolling window, plus a buffer.

    Computing a 720-hour rolling mean at time *T* needs the 720 hours before it,
    not 1440. The old ``* 2`` was arbitrary slack, and once a 30-day seasonal
    window was added it doubled the serving read to 60 days - 790k rows and
    ~970ms per cache miss. A fixed buffer covers the ``shift(1)`` and any hour
    with no rows at all, at a fraction of the cost.
    """
    longest_lag = max(config.get_path("features.lag_hours", [1, 2, 3, 24, 168]))
    longest_window = max(config.get_path("features.rolling_windows_hours", [24, 168]))
    buffer_hours = int(config.get_path("serving.history_buffer_hours", 24))
    return int(max(longest_lag, longest_window) + buffer_hours)


def load_recent_history(
    config: Config | None = None, *, hours: int | None = None, use_cache: bool = True
) -> HistoryResult:
    """Load recent hourly departures, preferring the database.

    Results are cached for ``serving.history_cache_seconds``. The underlying
    table only gains rows when the ingestion DAG runs, so re-reading it on every
    prediction was pure overhead - it was the single largest cost in a request.
    """
    cfg = config or load_config()
    window_hours = hours or _minimum_history_hours(cfg)
    ttl = float(cfg.get_path("serving.history_cache_seconds", 300))

    if use_cache and ttl > 0:
        with _CACHE_LOCK:
            entry = _CACHE.get(window_hours)
        if entry and (time.monotonic() - entry[0]) < ttl:
            return entry[1]

    result = _load_history_uncached(cfg, window_hours)

    if use_cache and ttl > 0 and not result.is_empty:
        with _CACHE_LOCK:
            _CACHE[window_hours] = (time.monotonic(), result)
    return result


def _load_history_uncached(cfg: Config, window_hours: int) -> HistoryResult:
    """Actually hit the database (or fall back), bypassing the cache."""
    try:
        from src.storage.db import read_latest_hourly_departures

        # One SQL query, anchored on the newest observation. This replaces a
        # recent-window read plus a 730-day fallback read that between them
        # transferred the whole table just to keep its tail.
        frame = read_latest_hourly_departures(window_hours, config=cfg)
        if not frame.empty:
            latest = pd.Timestamp(frame["hour_ts"].max())
            now = pd.Timestamp(datetime.now(tz=UTC)).tz_localize(None)
            # "Stale" is about how old the newest observation is, not about
            # which query found it. The threshold is deliberately tight: with
            # ingestion polling every 5 minutes, data hours old means the
            # pipeline is broken, and calling that "fresh" would hide it.
            # Reusing the lag window (336h) here would let two-week-old data
            # pass as current.
            freshness_hours = float(cfg.get_path("serving.history_freshness_hours", 6))
            fresh = (now - latest) <= pd.Timedelta(hours=freshness_hours)
            if not fresh:
                logger.warning(
                    "No recent history in the database; serving from the newest "
                    "available slice ending %s", latest,
                )
            return HistoryResult(frame, "postgres" if fresh else "postgres_stale", latest)
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
