"""Shared fixtures. Everything here is synthetic - tests never hit the network."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import load_config


@pytest.fixture(scope="session")
def config():
    return load_config()


@pytest.fixture
def station_info() -> pd.DataFrame:
    """Twelve stations in three tight geographic pockets."""
    rows = []
    for cluster_index, (base_lat, base_lon) in enumerate(
        [(40.70, -73.99), (40.75, -73.98), (40.80, -73.95)]
    ):
        for offset in range(4):
            station_number = cluster_index * 4 + offset
            rows.append(
                {
                    "station_id": f"uuid-{station_number}",
                    "short_name": f"{1000 + station_number}.01",
                    "name": f"Station {station_number}",
                    "lat": base_lat + offset * 0.001,
                    "lon": base_lon + offset * 0.001,
                    "capacity": 20 + station_number,
                    "region_id": "71",
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def hourly_departures(station_info) -> pd.DataFrame:
    """Six weeks of hourly departures with a realistic daily/weekly shape."""
    rng = np.random.default_rng(0)
    hours = pd.date_range("2026-04-01", periods=24 * 42, freq="h")
    rows = []
    for index, short_name in enumerate(station_info["short_name"]):
        base = 3.0 + index * 0.4
        for hour in hours:
            commute = 2.5 if hour.hour in (8, 9, 17, 18) else 1.0
            weekend = 0.6 if hour.dayofweek >= 5 else 1.0
            expected = base * commute * weekend
            rows.append(
                {
                    "station_short_name": short_name,
                    "hour_ts": hour,
                    "departures": float(rng.poisson(expected)),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def weather() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    hours = pd.date_range("2026-03-31", periods=24 * 44, freq="h")
    return pd.DataFrame(
        {
            "hour_ts": hours,
            "temperature_c": 18 + 6 * np.sin(np.arange(len(hours)) / 24.0),
            "precipitation_mm": rng.gamma(0.3, 0.5, len(hours)).round(2),
            "wind_speed_kmh": rng.uniform(3, 25, len(hours)).round(1),
        }
    )


@pytest.fixture
def snapshots() -> pd.DataFrame:
    """Two stations sampled once a minute.

    Matches production: each ingest run takes several samples 60s apart, so
    intervals are minutes rather than the 5-minute polls of the original design.
    """
    times = pd.date_range("2026-08-01T12:00:00Z", periods=13, freq="1min")
    rows = []
    for station_id, inventory in (
        ("uuid-a", [20, 19, 18, 18, 16, 15, 15, 14, 13, 13, 12, 11, 10]),
        ("uuid-b", [10, 10, 11, 12, 12, 11, 11, 10, 10, 9, 9, 8, 8]),
    ):
        for timestamp, bikes in zip(times, inventory, strict=True):
            rows.append(
                {
                    "station_id": station_id,
                    "snapshot_ts": timestamp,
                    "num_bikes_available": bikes,
                    "is_renting": 1,
                    "is_installed": 1,
                    "is_returning": 1,
                    "last_reported": timestamp,
                }
            )
    return pd.DataFrame(rows)
