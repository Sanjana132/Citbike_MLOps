"""Seed Postgres with schema, station metadata, and historical departures.

Run once after ``docker compose up`` so the API can serve immediately instead of
waiting for the ingestion DAG to accumulate enough history for a 168-hour lag::

    python -m scripts.bootstrap_db

Historical rows are written with ``label_source='trip_csv'`` to keep them
distinguishable from the ``gbfs_proxy`` rows the live loop produces. That
distinction is load-bearing: one is ground truth, the other is an estimate.
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from src.config import data_dir, load_config
from src.storage.db import (
    get_engine,
    init_schema,
    write_hourly_departures,
    write_station_info,
)

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialise and seed the database")
    parser.add_argument("--skip-history", action="store_true", help="Schema and stations only")
    parser.add_argument(
        "--history-months",
        type=int,
        default=None,
        help="Only load the most recent N months of history (default: all cached)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s")
    config = load_config()
    engine = get_engine()

    init_schema(engine)

    # --- station metadata, straight from the live feed ---------------------
    info_cache = data_dir("cache", "station_information.parquet")
    if info_cache.exists():
        station_info = pd.read_parquet(info_cache)
        logger.info("Loaded %d stations from cache", len(station_info))
    else:
        from src.ingestion.gbfs_client import GBFSClient

        station_info = GBFSClient(config).fetch_station_information()
        station_info.to_parquet(info_cache, index=False)
    written = write_station_info(station_info, engine)
    logger.info("Wrote %d stations to station_info", written)

    if args.skip_history:
        logger.info("Skipping history load as requested")
        return

    # --- historical hourly departures (TRUE labels) ------------------------
    cache_path = data_dir("cache", "hourly_departures_all.parquet")
    if cache_path.exists():
        departures = pd.read_parquet(cache_path)
    else:
        from src.ingestion.trip_csv_loader import load_hourly_departures

        live = {s for s in station_info["short_name"].dropna().tolist() if s}
        departures = load_hourly_departures(config=config, live_station_short_names=live)
        departures.to_parquet(cache_path, index=False)

    departures["hour_ts"] = pd.to_datetime(departures["hour_ts"])
    if args.history_months:
        cutoff = departures["hour_ts"].max() - pd.DateOffset(months=args.history_months)
        departures = departures[departures["hour_ts"] >= cutoff]

    departures = departures.assign(label_source="trip_csv", n_intervals=None)

    # Chunked so a multi-million-row insert does not build one enormous
    # parameter list.
    total = 0
    chunk_size = 100_000
    for start in range(0, len(departures), chunk_size):
        chunk = departures.iloc[start : start + chunk_size]
        total += write_hourly_departures(chunk, engine, label_source="trip_csv")
        logger.info("Wrote %d/%d departure rows", total, len(departures))

    logger.info(
        "Bootstrap complete: %d stations, %d hourly departure rows (%s..%s)",
        len(station_info), total, departures["hour_ts"].min(), departures["hour_ts"].max(),
    )


if __name__ == "__main__":
    main()
