"""GBFS client for the Citi Bike (Lyft) feeds.

Two feeds matter here:

* ``station_information`` - static metadata (name, lat/lon, capacity, short_name).
* ``station_status`` - the live inventory snapshot polled every 5 minutes.

IMPORTANT - ``station_status`` reports *inventory* (how many bikes are docked
right now), not *demand* (how many rides were taken). Converting one into the
other is the job of ``derive_departures.py``, and the result is a **proxy**, not
ground truth. Nothing in this module produces a demand figure.

Parsing is deliberately tolerant: GBFS producers add, rename, and drop optional
fields, and a single malformed station must never take down an ingestion run.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from src.config import Config, load_config
from src.ingestion.http import get_with_retries

logger = logging.getLogger(__name__)

# Columns produced by this module, in order. Downstream code (DB schema,
# feature builder) relies on these names.
STATION_INFO_COLUMNS = [
    "station_id",
    "short_name",
    "name",
    "lat",
    "lon",
    "capacity",
    "region_id",
]

STATION_STATUS_COLUMNS = [
    "station_id",
    "num_bikes_available",
    "num_ebikes_available",
    "num_bikes_disabled",
    "num_docks_available",
    "num_docks_disabled",
    "is_installed",
    "is_renting",
    "is_returning",
    "last_reported",
    "snapshot_ts",
]


def _to_int(value: Any, default: int | None = None) -> int | None:
    """Coerce a GBFS numeric field to int, tolerating strings/None/floats."""
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    # GBFS occasionally emits nulls as 0.0 lat/lon; those are not valid NYC coords.
    return result


def _to_bool_int(value: Any, default: int = 0) -> int:
    """GBFS 1.1 encodes booleans as 0/1 ints, but some producers emit true/false."""
    if isinstance(value, bool):
        return int(value)
    coerced = _to_int(value, None)
    if coerced is not None:
        return 1 if coerced else 0
    if isinstance(value, str):
        if value.strip().lower() in {"true", "yes", "y"}:
            return 1
        if value.strip().lower() in {"false", "no", "n"}:
            return 0
    return default


class GBFSClient:
    """Fetches and normalises GBFS station feeds."""

    def __init__(self, config: Config | None = None) -> None:
        self._config = config or load_config()
        self._region = self._config.require("gbfs.region")
        self._timeout_s = int(self._config.get_path("gbfs.request_timeout_s", 30))
        self._retries = int(self._config.get_path("gbfs.max_retries", 3))

    def _url(self, key: str) -> str:
        template = self._config.require(f"gbfs.{key}")
        return template.format(region=self._region)

    def _fetch_stations(self, key: str) -> tuple[list[dict[str, Any]], datetime]:
        """Fetch a feed and return (station dicts, feed last_updated timestamp)."""
        url = self._url(key)
        payload = get_with_retries(
            url, timeout_s=self._timeout_s, retries=self._retries
        ).json()
        stations = payload.get("data", {}).get("stations")
        if not isinstance(stations, list):
            raise ValueError(f"Feed {url} did not contain data.stations list")
        last_updated_raw = _to_int(payload.get("last_updated"))
        last_updated = (
            datetime.fromtimestamp(last_updated_raw, tz=UTC)
            if last_updated_raw
            else datetime.now(tz=UTC)
        )
        return stations, last_updated

    def fetch_station_information(self) -> pd.DataFrame:
        """Static station metadata.

        ``short_name`` is retained because it is the join key to the historical
        trip CSVs - those use values like ``5506.14`` in ``start_station_id``,
        which correspond to GBFS ``short_name``, *not* GBFS ``station_id``
        (a UUID). Verified against the live feed at build time.
        """
        stations, _ = self._fetch_stations("station_information_url")
        rows: list[dict[str, Any]] = []
        skipped = 0
        for station in stations:
            station_id = station.get("station_id")
            lat = _to_float(station.get("lat"))
            lon = _to_float(station.get("lon"))
            if not station_id or lat is None or lon is None:
                skipped += 1
                continue
            # A handful of feed entries carry placeholder (0, 0) coordinates.
            if abs(lat) < 1e-6 and abs(lon) < 1e-6:
                skipped += 1
                continue
            rows.append(
                {
                    "station_id": str(station_id),
                    "short_name": (
                        str(station["short_name"]).strip()
                        if station.get("short_name") is not None
                        else None
                    ),
                    "name": station.get("name"),
                    "lat": lat,
                    "lon": lon,
                    "capacity": _to_int(station.get("capacity"), 0),
                    "region_id": (
                        str(station["region_id"]) if station.get("region_id") is not None else None
                    ),
                }
            )
        if skipped:
            logger.warning("station_information: skipped %d malformed stations", skipped)
        logger.info("station_information: parsed %d stations", len(rows))
        return pd.DataFrame(rows, columns=STATION_INFO_COLUMNS)

    def fetch_station_status(self, snapshot_ts: datetime | None = None) -> pd.DataFrame:
        """One live inventory snapshot.

        The returned frame is inventory state at a point in time. It says
        nothing on its own about how many rides were taken.
        """
        stations, feed_updated = self._fetch_stations("station_status_url")
        observed_at = snapshot_ts or feed_updated
        rows: list[dict[str, Any]] = []
        skipped = 0
        for station in stations:
            station_id = station.get("station_id")
            bikes = _to_int(station.get("num_bikes_available"))
            if not station_id or bikes is None:
                skipped += 1
                continue
            last_reported_raw = _to_int(station.get("last_reported"))
            rows.append(
                {
                    "station_id": str(station_id),
                    "num_bikes_available": bikes,
                    "num_ebikes_available": _to_int(station.get("num_ebikes_available"), 0),
                    "num_bikes_disabled": _to_int(station.get("num_bikes_disabled"), 0),
                    "num_docks_available": _to_int(station.get("num_docks_available"), 0),
                    "num_docks_disabled": _to_int(station.get("num_docks_disabled"), 0),
                    "is_installed": _to_bool_int(station.get("is_installed")),
                    "is_renting": _to_bool_int(station.get("is_renting")),
                    "is_returning": _to_bool_int(station.get("is_returning")),
                    "last_reported": (
                        datetime.fromtimestamp(last_reported_raw, tz=UTC)
                        if last_reported_raw
                        else None
                    ),
                    "snapshot_ts": observed_at,
                }
            )
        if skipped:
            logger.warning("station_status: skipped %d malformed stations", skipped)
        logger.info(
            "station_status: parsed %d stations at %s", len(rows), observed_at.isoformat()
        )
        return pd.DataFrame(rows, columns=STATION_STATUS_COLUMNS)


def fetch_live_station_short_names(config: Config | None = None) -> set[str]:
    """Short names of every station currently in the live feed.

    Used by the trip-history loader to drop stations that no longer exist, so
    the offline-trained model only covers stations it can actually be served for.
    """
    info = GBFSClient(config).fetch_station_information()
    return {s for s in info["short_name"].dropna().tolist() if s}
