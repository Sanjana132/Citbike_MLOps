"""Tests for the inventory-delta demand proxy and its correction rules."""

from __future__ import annotations

import pandas as pd
import pytest

from src.ingestion.derive_departures import (
    aggregate_proxy_to_hourly,
    derive_interval_departures,
    proxy_quality_report,
)


def test_net_decrease_becomes_proxy_departures(snapshots, config):
    result = derive_interval_departures(snapshots, config)
    station_a = result[result["station_id"] == "uuid-a"]
    # Inventory 20 -> 10 over twelve intervals, monotone non-increasing.
    assert station_a["proxy_departures"].sum() == pytest.approx(10.0)


def test_returns_do_not_produce_negative_departures(snapshots, config):
    """Inventory increases are returns; the departures target clamps at zero."""
    result = derive_interval_departures(snapshots, config)
    assert (result["proxy_departures"] >= 0).all()

    station_b = result[result["station_id"] == "uuid-b"]
    # Station B goes 10 -> 11 -> 12 (returns) before declining.
    assert station_b["proxy_departures"].min() == 0.0


def test_netting_understates_true_demand(config):
    """Documents the proxy's core bias rather than pretending it does not exist.

    Within one interval 5 bikes leave and 4 arrive. Five rides actually started,
    but inventory only fell by 1, so the proxy reports 1.
    """
    frame = pd.DataFrame(
        {
            "station_id": ["s1", "s1"],
            "snapshot_ts": pd.to_datetime(["2026-08-01T12:00:00Z", "2026-08-01T12:05:00Z"]),
            "num_bikes_available": [20, 19],
            "is_renting": [1, 1],
            "is_installed": [1, 1],
            "last_reported": pd.to_datetime(["2026-08-01T12:00:00Z", "2026-08-01T12:05:00Z"]),
        }
    )
    result = derive_interval_departures(frame, config)
    assert result["proxy_departures"].sum() == 1.0  # not the true 5


def test_non_renting_intervals_are_excluded(config):
    """A dock outage drains inventory without anyone riding."""
    frame = pd.DataFrame(
        {
            "station_id": ["s1"] * 3,
            "snapshot_ts": pd.to_datetime(
                ["2026-08-01T12:00:00Z", "2026-08-01T12:05:00Z", "2026-08-01T12:10:00Z"]
            ),
            "num_bikes_available": [20, 5, 4],
            "is_renting": [1, 0, 1],  # middle snapshot: station offline
            "is_installed": [1, 1, 1],
            "last_reported": pd.to_datetime(
                ["2026-08-01T12:00:00Z", "2026-08-01T12:05:00Z", "2026-08-01T12:10:00Z"]
            ),
        }
    )
    result = derive_interval_departures(frame, config)
    # Both pairs touch the non-renting snapshot, so neither survives.
    assert result.empty


def test_implausible_drop_is_capped_as_rebalancing(config):
    """A 15-bike drop in five minutes is a van, not fifteen simultaneous riders."""
    frame = pd.DataFrame(
        {
            "station_id": ["s1", "s1"],
            "snapshot_ts": pd.to_datetime(["2026-08-01T12:00:00Z", "2026-08-01T12:05:00Z"]),
            "num_bikes_available": [20, 5],
            "is_renting": [1, 1],
            "is_installed": [1, 1],
            "last_reported": pd.to_datetime(["2026-08-01T12:00:00Z", "2026-08-01T12:05:00Z"]),
        }
    )
    cap = float(config.get_path("ingestion.proxy.max_plausible_drop_per_interval"))
    result = derive_interval_departures(frame, config)
    assert result["proxy_departures"].iloc[0] == cap


def test_missed_polls_are_rejected(config):
    """A 3-hour gap cannot be attributed to one interval of demand."""
    frame = pd.DataFrame(
        {
            "station_id": ["s1", "s1"],
            "snapshot_ts": pd.to_datetime(["2026-08-01T12:00:00Z", "2026-08-01T15:00:00Z"]),
            "num_bikes_available": [20, 12],
            "is_renting": [1, 1],
            "is_installed": [1, 1],
            "last_reported": pd.to_datetime(["2026-08-01T12:00:00Z", "2026-08-01T15:00:00Z"]),
        }
    )
    assert derive_interval_departures(frame, config).empty


def test_stale_reporters_are_rejected(config):
    """Inventory from a station that stopped reporting hours ago is not current."""
    frame = pd.DataFrame(
        {
            "station_id": ["s1", "s1"],
            "snapshot_ts": pd.to_datetime(["2026-08-01T12:00:00Z", "2026-08-01T12:05:00Z"]),
            "num_bikes_available": [20, 15],
            "is_renting": [1, 1],
            "is_installed": [1, 1],
            "last_reported": pd.to_datetime(["2026-08-01T06:00:00Z", "2026-08-01T06:00:00Z"]),
        }
    )
    assert derive_interval_departures(frame, config).empty


def test_empty_and_malformed_input_do_not_raise(config):
    """Ingestion must never crash a DAG run on bad data."""
    assert derive_interval_departures(pd.DataFrame(), config).empty

    malformed = pd.DataFrame(
        {
            "station_id": ["s1", "s1"],
            "snapshot_ts": ["not-a-date", None],
            "num_bikes_available": ["abc", None],
        }
    )
    assert derive_interval_departures(malformed, config).empty


def test_hourly_aggregation_drops_low_coverage_hours(snapshots, config):
    """Three surviving intervals must not be reported as a full hour of demand."""
    intervals = derive_interval_departures(snapshots, config)
    station_info = pd.DataFrame(
        {"station_id": ["uuid-a", "uuid-b"], "short_name": ["1001.01", "1002.01"]}
    )

    # 12 intervals of an expected 12 -> full coverage, kept.
    hourly = aggregate_proxy_to_hourly(intervals, station_info, config)
    assert not hourly.empty
    assert set(hourly["station_short_name"]) == {"1001.01", "1002.01"}

    # Keep only 3 intervals -> 25% coverage, dropped.
    sparse = intervals.groupby("station_id", as_index=False).head(3)
    assert aggregate_proxy_to_hourly(sparse, station_info, config).empty


def test_quality_report_summarises_proxy(snapshots, config):
    intervals = derive_interval_departures(snapshots, config)
    report = proxy_quality_report(intervals)
    assert report["n_stations"] == 2
    assert 0.0 <= report["zero_share"] <= 1.0
