"""Ingestion orchestration: the functions the Airflow DAGs actually call.

DAGs stay thin on purpose - they wire schedules and dependencies, and the
business logic lives here where it can be tested without an Airflow instance.

Every function is written to survive partial failure. A single bad station, a
GBFS hiccup, or one unparseable snapshot must not fail a run: ingestion that
crashes on bad input stops collecting the very history the model depends on.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

import pandas as pd

from src.config import Config, load_config

logger = logging.getLogger(__name__)


def ingest_station_information(config: Config | None = None) -> dict:
    """Refresh static station metadata. Cheap, and stations do move."""
    cfg = config or load_config()
    from src.ingestion.gbfs_client import GBFSClient
    from src.storage.db import write_station_info

    station_info = GBFSClient(cfg).fetch_station_information()
    if station_info.empty:
        logger.warning("station_information returned no rows; keeping existing metadata")
        return {"stations_written": 0}
    written = write_station_info(station_info)
    logger.info("Refreshed metadata for %d stations", written)
    return {"stations_written": written}


def ingest_station_status(config: Config | None = None) -> dict:
    """Poll one live inventory snapshot and persist it.

    This writes INVENTORY, not demand. Converting it into a demand proxy is a
    separate, explicitly-named step.
    """
    cfg = config or load_config()
    from src.ingestion.gbfs_client import GBFSClient
    from src.storage.db import write_status_snapshots

    snapshot = GBFSClient(cfg).fetch_station_status()
    if snapshot.empty:
        logger.error("station_status returned no usable rows; nothing ingested this run")
        return {"snapshots_written": 0}

    written = write_status_snapshots(snapshot)
    logger.info("Wrote %d station status rows", written)
    return {
        "snapshots_written": written,
        "snapshot_ts": str(snapshot["snapshot_ts"].max()),
    }


def derive_and_store_proxy_departures(
    lookback_hours: int = 3, config: Config | None = None
) -> dict:
    """Turn recent snapshots into hourly proxy departures and persist them.

    Reprocesses a few hours each run rather than only the newest interval, so
    that a late-arriving or retried snapshot corrects the hour it belongs to.
    The write is an upsert keyed on (station, hour, label_source), making
    reprocessing idempotent.

    The output is labelled ``gbfs_proxy`` and is NOT ground truth - see
    ``derive_departures.py`` for exactly how it is biased.
    """
    cfg = config or load_config()
    from src.ingestion.derive_departures import (
        aggregate_proxy_to_hourly,
        derive_interval_departures,
        proxy_quality_report,
    )
    from src.storage.db import read_recent_snapshots, read_station_info, write_hourly_departures

    snapshots = read_recent_snapshots(hours=lookback_hours)
    if snapshots.empty:
        logger.warning("No recent snapshots to derive departures from")
        return {"rows_written": 0}

    intervals = derive_interval_departures(snapshots, cfg)
    if intervals.empty:
        logger.warning("No usable snapshot pairs after applying proxy corrections")
        return {"rows_written": 0}

    station_info = read_station_info(config=cfg)
    hourly = aggregate_proxy_to_hourly(intervals, station_info, cfg)
    if hourly.empty:
        logger.warning("No station-hours passed the snapshot-coverage threshold")
        return {"rows_written": 0}

    # Never overwrite the most recent hour while it is still accumulating - it
    # would be written as a complete hour when it is only partly observed.
    latest_complete = pd.Timestamp(hourly["hour_ts"].max())
    complete = hourly[hourly["hour_ts"] < latest_complete]
    if complete.empty:
        logger.info("Only the in-progress hour is available; deferring the write")
        return {"rows_written": 0}

    written = write_hourly_departures(complete, label_source="gbfs_proxy")
    quality = proxy_quality_report(intervals)
    logger.info("Wrote %d proxy hourly departure rows; quality=%s", written, quality)
    return {"rows_written": written, **quality}


def ingest_once(config: Config | None = None) -> dict:
    """Whole ingestion cycle in one call. Used by the DAG and by manual runs."""
    cfg = config or load_config()
    result = {}
    result.update(ingest_station_status(cfg))
    result.update(derive_and_store_proxy_departures(config=cfg))
    return result


def backfill_weather(days: int = 2, config: Config | None = None) -> dict:
    """Warm the weather cache so serving and training never wait on the API."""
    cfg = config or load_config()
    from src.ingestion.weather import load_historical_weather

    end = datetime.now(tz=UTC).date()
    start = end - timedelta(days=days)
    frame = load_historical_weather(start, end, cfg, use_cache=False)
    return {"weather_hours": len(frame)}
