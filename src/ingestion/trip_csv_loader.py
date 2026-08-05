"""Historical Citi Bike trip-history loader.

This is the source of **true labels** for the initial (offline) model: every row
in these CSVs is a completed trip, so aggregating trip start times into hourly
buckets per station gives real departure counts. Contrast this with the live
GBFS path (``derive_departures.py``), which can only produce a noisy *proxy*.

Two practical problems this module solves:

1. **Size.** A single NYC month is ~1 GB uncompressed, split across several part
   files inside one zip. The loader streams the archive over HTTP range requests
   and reduces each chunk to hourly counts immediately, so peak memory stays in
   the tens of MB rather than gigabytes.

2. **Schema drift.** Citi Bike has changed the CSV schema several times over the
   years (``starttime`` -> ``started_at``, ``start station id`` ->
   ``start_station_id``, integer station ids -> string codes like ``5506.14``).
   Column resolution is therefore done by alias lookup, not by position, and a
   month whose schema cannot be resolved is skipped with a loud warning rather
   than crashing the run.
"""

from __future__ import annotations

import logging
import re
import zipfile
from dataclasses import dataclass
from typing import Iterator

import pandas as pd

from src.config import Config, data_dir, load_config
from src.ingestion.http import get_with_retries, open_remote_zip_stream

logger = logging.getLogger(__name__)

# Alias tables covering every Citi Bike schema generation seen in the bucket.
# Matching is done on a normalised column name (lowercased, non-alphanumerics
# collapsed to underscores), so "Start Time", "start time" and "start_time" all
# reduce to the same key.
START_TIME_ALIASES = ("started_at", "starttime", "start_time")
END_TIME_ALIASES = ("ended_at", "stoptime", "stop_time", "end_time")
START_STATION_ID_ALIASES = (
    "start_station_id",
    "start_station",
    "from_station_id",
)
START_STATION_NAME_ALIASES = ("start_station_name", "from_station_name")
START_LAT_ALIASES = ("start_lat", "start_station_latitude")
START_LON_ALIASES = ("start_lng", "start_lon", "start_station_longitude")

HOURLY_DEPARTURES_COLUMNS = ["station_short_name", "hour_ts", "departures"]


def _normalise(column: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", column.strip().lower()).strip("_")


@dataclass(frozen=True)
class TripSchema:
    """Resolved column names for one CSV file."""

    start_time: str
    start_station_id: str
    start_lat: str | None
    start_lon: str | None
    start_station_name: str | None

    @property
    def usecols(self) -> list[str]:
        cols = [self.start_time, self.start_station_id]
        for optional in (self.start_lat, self.start_lon, self.start_station_name):
            if optional:
                cols.append(optional)
        return cols


def resolve_schema(columns: list[str]) -> TripSchema:
    """Map a raw CSV header onto the fields we need.

    Raises ``ValueError`` if a required field is absent - callers skip that file.
    """
    lookup = {_normalise(c): c for c in columns}

    def first_match(aliases: tuple[str, ...]) -> str | None:
        for alias in aliases:
            if alias in lookup:
                return lookup[alias]
        return None

    start_time = first_match(START_TIME_ALIASES)
    start_station_id = first_match(START_STATION_ID_ALIASES)
    if not start_time or not start_station_id:
        raise ValueError(
            "Unrecognised trip CSV schema; could not find a start-time and "
            f"start-station-id column among: {sorted(columns)}"
        )
    return TripSchema(
        start_time=start_time,
        start_station_id=start_station_id,
        start_lat=first_match(START_LAT_ALIASES),
        start_lon=first_match(START_LON_ALIASES),
        start_station_name=first_match(START_STATION_NAME_ALIASES),
    )


def aggregate_chunk(chunk: pd.DataFrame, schema: TripSchema) -> pd.DataFrame:
    """Reduce a chunk of raw trips to hourly departure counts.

    Rows with an unparseable timestamp or a blank station id are dropped and
    counted, never allowed to propagate as nulls into the training set.
    """
    frame = chunk[[schema.start_time, schema.start_station_id]].copy()
    frame.columns = ["start_time", "station_short_name"]

    frame["start_time"] = pd.to_datetime(frame["start_time"], errors="coerce", format="mixed")
    frame["station_short_name"] = (
        frame["station_short_name"].astype("string").str.strip()
    )
    frame = frame[
        frame["start_time"].notna()
        & frame["station_short_name"].notna()
        & (frame["station_short_name"] != "")
    ]
    if frame.empty:
        return pd.DataFrame(columns=HOURLY_DEPARTURES_COLUMNS)

    frame["hour_ts"] = frame["start_time"].dt.floor("h")
    counts = (
        frame.groupby(["station_short_name", "hour_ts"], observed=True)
        .size()
        .reset_index(name="departures")
    )
    return counts[HOURLY_DEPARTURES_COLUMNS]


def _list_bucket_keys(config: Config) -> list[str]:
    """List every object key in the trip-data bucket."""
    index_url = config.require("trip_history.bucket_index_url")
    response = get_with_retries(index_url, params={"list-type": "2", "max-keys": "1000"})
    keys = re.findall(r"<Key>(.*?)</Key>", response.text, flags=re.S)
    return [k.strip() for k in keys]


def resolve_month_url(month: str, config: Config | None = None) -> str:
    """Find the archive URL for a ``YYYYMM`` month.

    The bucket is inconsistent about suffixes - some months are
    ``202606-citibike-tripdata.zip`` and others ``...-tripdata.csv.zip`` - so the
    key is discovered from a live listing rather than assumed.

    Jersey City (``JC-``) and Hoboken files are excluded: they cover a different
    station set from the NYC system feed we serve against.
    """
    cfg = config or load_config()
    keys = _list_bucket_keys(cfg)
    candidates = [
        k
        for k in keys
        if k.startswith(month)
        and k.endswith(".zip")
        and not k.startswith("JC-")
        and "__MACOSX" not in k
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No NYC trip archive found for month {month}. "
            f"Available prefixes: {sorted({k[:6] for k in keys if k[:6].isdigit()})}"
        )
    # Prefer the shortest key; some months have both a base and a variant file.
    key = sorted(candidates, key=len)[0]
    base = cfg.require("trip_history.bucket_index_url").rstrip("/")
    return f"{base}/{key}"


def _iter_member_chunks(
    archive: zipfile.ZipFile, member: zipfile.ZipInfo, chunk_size: int
) -> Iterator[pd.DataFrame]:
    """Yield chunks of one CSV member, resolving its schema from the header."""
    with archive.open(member) as handle:
        reader = pd.read_csv(
            handle,
            chunksize=chunk_size,
            dtype=str,
            low_memory=False,
            on_bad_lines="warn",
        )
        schema: TripSchema | None = None
        for chunk in reader:
            if schema is None:
                schema = resolve_schema(list(chunk.columns))
                logger.info(
                    "%s: resolved start_time=%r station_id=%r",
                    member.filename, schema.start_time, schema.start_station_id,
                )
            yield aggregate_chunk(chunk, schema)


def load_month_hourly_departures(
    month: str, config: Config | None = None, *, use_cache: bool = True
) -> pd.DataFrame:
    """Stream one month of trips and return hourly departures per station.

    Result is cached as parquet under ``data/cache`` so repeated training runs
    do not re-download ~1 GB per month.
    """
    cfg = config or load_config()
    cache_path = data_dir("cache", f"hourly_departures_{month}.parquet")
    if use_cache and cache_path.exists():
        logger.info("Using cached hourly departures for %s (%s)", month, cache_path)
        return pd.read_parquet(cache_path)

    url = resolve_month_url(month, cfg)
    chunk_size = int(cfg.get_path("trip_history.chunk_size", 500_000))
    logger.info("Streaming trip archive for %s from %s", month, url)

    partials: list[pd.DataFrame] = []
    total_rows = 0
    with open_remote_zip_stream(url) as stream:
        archive = zipfile.ZipFile(stream)
        members = [
            m
            for m in archive.infolist()
            if m.filename.lower().endswith(".csv") and "__MACOSX" not in m.filename
        ]
        if not members:
            raise FileNotFoundError(f"Archive for {month} contained no CSV members")
        logger.info("%s: %d CSV member(s) in archive", month, len(members))
        for member in members:
            try:
                for partial in _iter_member_chunks(archive, member, chunk_size):
                    if not partial.empty:
                        partials.append(partial)
                        total_rows += int(partial["departures"].sum())
            except ValueError as exc:
                # Schema we cannot interpret - skip this member, keep the month.
                logger.error("Skipping %s: %s", member.filename, exc)
            except Exception as exc:  # noqa: BLE001 - one bad member must not kill the month
                logger.error("Unexpected failure reading %s: %s", member.filename, exc)

    if not partials:
        raise ValueError(f"No usable trip rows parsed for month {month}")

    # Chunk boundaries split hours, so a final regroup is required.
    monthly = (
        pd.concat(partials, ignore_index=True)
        .groupby(["station_short_name", "hour_ts"], observed=True)["departures"]
        .sum()
        .reset_index()
    )
    logger.info(
        "%s: %d trips -> %d (station, hour) rows across %d stations",
        month, total_rows, len(monthly), monthly["station_short_name"].nunique(),
    )
    monthly.to_parquet(cache_path, index=False)
    return monthly


def load_hourly_departures(
    months: list[str] | None = None,
    config: Config | None = None,
    *,
    live_station_short_names: set[str] | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Load and concatenate hourly departures for the configured months.

    If ``restrict_to_live_stations`` is set and live short names are supplied,
    stations absent from the current GBFS feed are dropped - there is no point
    training on docks the serving path can never see.
    """
    cfg = config or load_config()
    target_months = months or list(cfg.require("trip_history.months"))

    frames: list[pd.DataFrame] = []
    for month in target_months:
        try:
            frames.append(load_month_hourly_departures(month, cfg, use_cache=use_cache))
        except Exception as exc:  # noqa: BLE001 - a missing month should not abort training
            logger.error("Failed to load month %s: %s", month, exc)
    if not frames:
        raise RuntimeError(f"No trip-history months could be loaded (requested: {target_months})")

    combined = (
        pd.concat(frames, ignore_index=True)
        .groupby(["station_short_name", "hour_ts"], observed=True)["departures"]
        .sum()
        .reset_index()
    )

    if cfg.get_path("trip_history.restrict_to_live_stations", True) and live_station_short_names:
        before = combined["station_short_name"].nunique()
        combined = combined[combined["station_short_name"].isin(live_station_short_names)]
        logger.info(
            "Restricted to live stations: %d -> %d stations",
            before, combined["station_short_name"].nunique(),
        )

    min_departures = int(cfg.get_path("features.min_departures_per_station", 0))
    if min_departures > 0:
        totals = combined.groupby("station_short_name")["departures"].sum()
        keep = set(totals[totals >= min_departures].index)
        before = combined["station_short_name"].nunique()
        combined = combined[combined["station_short_name"].isin(keep)]
        logger.info(
            "Dropped low-volume stations (<%d departures): %d -> %d stations",
            min_departures, before, combined["station_short_name"].nunique(),
        )

    return combined.sort_values(["station_short_name", "hour_ts"]).reset_index(drop=True)
