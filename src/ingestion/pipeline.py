"""Ingestion orchestration: the functions the Airflow DAGs actually call.

DAGs stay thin on purpose - they wire schedules and dependencies, and the
business logic lives here where it can be tested without an Airflow instance.

Every function is written to survive partial failure. A single bad station, a
GBFS hiccup, or one unparseable snapshot must not fail a run: ingestion that
crashes on bad input stops collecting the very history the model depends on.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
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


def ingest_station_status(
    config: Config | None = None, *, sleep: Callable[[float], None] | None = None
) -> dict:
    """Sample the live inventory feed several times and persist each snapshot.

    This writes INVENTORY, not demand. Converting it into a demand proxy is a
    separate, explicitly-named step.

    **Why several samples per run rather than one.** The feed advances its
    ``last_updated`` every 60 seconds, and ``snapshot_ts`` comes from that field
    rather than from our clock - so polling once every 5 minutes captured only
    one in five available updates, and re-polling within the same minute wrote a
    duplicate that the upsert discarded. Two consequences, both bad:

    * **Fragility.** At 12 samples an hour, the hourly-coverage guard needs 6.
      A few minutes of scheduler lag dropped an hour below the threshold and
      discarded it entirely, which is why proxy departures stalled for a day
      while snapshots kept arriving.
    * **Accuracy.** Coarser sampling makes the netting bias worse: the longer
      the interval, the more departures and returns cancel inside it before the
      difference is taken.

    Sampling once a minute within each run fixes both. Coverage becomes ~60
    intervals an hour, so a whole missed run costs 5 of 60 rather than half the
    hour, and each interval is short enough that less demand nets out inside it.
    """
    cfg = config or load_config()
    import time

    from src.ingestion.gbfs_client import GBFSClient
    from src.storage.db import write_status_snapshots

    # Injectable so tests exercise the sampling loop without waiting minutes of
    # real time - the loop's logic is what matters, not the delay.
    wait = sleep if sleep is not None else time.sleep

    samples = max(1, int(cfg.get_path("ingestion.samples_per_run", 5)))
    spacing_seconds = float(cfg.get_path("ingestion.sample_spacing_seconds", 60))

    # Hard wall-clock budget for the whole run. A single fetch once hung for 67
    # minutes: `requests` applies its timeout per socket read, so a server that
    # trickles bytes can hold a connection open far longer than the configured
    # 30s. With max_active_runs=1 that one stuck run blocked every scheduled run
    # behind it for over an hour, and ingestion looked dead from the outside.
    # The budget bounds the damage to a single cycle.
    budget_seconds = float(
        cfg.get_path("ingestion.run_budget_seconds", spacing_seconds * samples + 60)
    )
    deadline = time.monotonic() + budget_seconds

    client = GBFSClient(cfg)
    written = 0
    timestamps: list[str] = []

    for index in range(samples):
        if time.monotonic() > deadline:
            logger.warning(
                "Run budget of %.0fs exhausted after %d/%d samples; ending this cycle "
                "so the next scheduled run is not blocked",
                budget_seconds, index, samples,
            )
            break
        try:
            snapshot = client.fetch_station_status()
        except Exception as exc:  # noqa: BLE001 - one bad sample must not lose the rest
            logger.error("Sample %d/%d failed: %s", index + 1, samples, exc)
            snapshot = None

        if snapshot is not None and not snapshot.empty:
            written += write_status_snapshots(snapshot)
            timestamps.append(str(snapshot["snapshot_ts"].max()))
        elif snapshot is not None:
            logger.warning("Sample %d/%d returned no usable rows", index + 1, samples)

        # No sleep after the final sample - the next scheduled run continues the
        # sequence, keeping intervals contiguous across run boundaries.
        if index < samples - 1 and time.monotonic() + spacing_seconds < deadline:
            wait(spacing_seconds)

    if not written:
        logger.error("station_status produced nothing this run")
        return {"snapshots_written": 0, "samples": samples}

    logger.info(
        "Wrote %d station status rows from %d/%d samples (%d distinct feed timestamps)",
        written, len(timestamps), samples, len(set(timestamps)),
    )
    return {
        "snapshots_written": written,
        "samples": samples,
        "distinct_timestamps": len(set(timestamps)),
        "snapshot_ts": max(timestamps),
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

    # New rows landed, so any cached serving history is now out of date. The
    # cache TTL is only an upper bound; this makes fresh data visible at once.
    try:
        from src.serving.history import invalidate_history_cache

        invalidate_history_cache()
    except Exception as exc:  # noqa: BLE001 - cache invalidation is never critical
        logger.debug("Could not invalidate the serving history cache: %s", exc)

    quality = proxy_quality_report(intervals)

    # Prune only after the hours have been derived, so nothing is deleted
    # before the data it feeds has been written.
    try:
        from src.storage.db import prune_status_snapshots

        pruned = prune_status_snapshots(
            int(cfg.get_path("ingestion.snapshot_retention_days", 3))
        )
    except Exception as exc:  # noqa: BLE001 - retention is housekeeping, not critical
        logger.warning("Could not prune old snapshots: %s", exc)
        pruned = 0

    logger.info("Wrote %d proxy hourly departure rows; quality=%s", written, quality)
    return {"rows_written": written, "snapshots_pruned": pruned, **quality}


def ingest_once(
    config: Config | None = None, *, sleep: Callable[[float], None] | None = None
) -> dict:
    """Whole ingestion cycle in one call. Used by the DAG and by manual runs."""
    cfg = config or load_config()
    result = {}
    result.update(ingest_station_status(cfg, sleep=sleep))
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
