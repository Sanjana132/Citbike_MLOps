"""Schema tolerance for historical trip CSVs.

Citi Bike has changed this schema repeatedly. The loader resolves columns by
alias, so these tests pin the generations we know about and prove that an
unknown schema fails loudly rather than silently producing garbage.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.ingestion.trip_csv_loader import (
    aggregate_chunk,
    resolve_schema,
)

# The three generations present in the public bucket.
MODERN_2020_ONWARDS = [
    "ride_id", "rideable_type", "started_at", "ended_at",
    "start_station_name", "start_station_id", "end_station_name",
    "end_station_id", "start_lat", "start_lng", "end_lat", "end_lng",
    "member_casual",
]
LEGACY_UNDERSCORE = [
    "tripduration", "starttime", "stoptime", "start_station_id",
    "start_station_name", "start_station_latitude", "start_station_longitude",
    "end_station_id", "bikeid", "usertype",
]
LEGACY_SPACED = [
    "tripduration", "starttime", "stoptime", "start station id",
    "start station name", "start station latitude", "start station longitude",
    "end station id", "bikeid", "usertype", "birth year", "gender",
]


@pytest.mark.parametrize(
    "columns,expected_time,expected_station",
    [
        (MODERN_2020_ONWARDS, "started_at", "start_station_id"),
        (LEGACY_UNDERSCORE, "starttime", "start_station_id"),
        (LEGACY_SPACED, "starttime", "start station id"),
    ],
)
def test_resolves_every_known_schema_generation(columns, expected_time, expected_station):
    schema = resolve_schema(columns)
    assert schema.start_time == expected_time
    assert schema.start_station_id == expected_station


def test_column_matching_is_case_and_separator_insensitive():
    schema = resolve_schema(["Started_At", "START STATION ID", "Start Lat", "Start Lng"])
    assert schema.start_time == "Started_At"
    assert schema.start_station_id == "START STATION ID"
    assert schema.start_lat == "Start Lat"


def test_unknown_schema_fails_loudly():
    """Better a skipped month with a clear error than silent bad training data."""
    with pytest.raises(ValueError, match="Unrecognised trip CSV schema"):
        resolve_schema(["foo", "bar", "baz"])


def test_aggregates_trips_into_hourly_counts():
    schema = resolve_schema(MODERN_2020_ONWARDS)
    chunk = pd.DataFrame(
        {
            "started_at": [
                "2026-06-01 08:14:00.123",
                "2026-06-01 08:59:59.999",
                "2026-06-01 09:00:00.000",
            ],
            "start_station_id": ["5506.14", "5506.14", "5506.14"],
        }
    )
    result = aggregate_chunk(chunk, schema)
    counts = dict(zip(result["hour_ts"].astype(str), result["departures"]))
    assert counts["2026-06-01 08:00:00"] == 2
    assert counts["2026-06-01 09:00:00"] == 1


def test_bad_rows_are_dropped_not_propagated():
    """One unparseable row must not poison the month."""
    schema = resolve_schema(MODERN_2020_ONWARDS)
    chunk = pd.DataFrame(
        {
            "started_at": ["2026-06-01 08:14:00", "not-a-timestamp", "2026-06-01 08:30:00", None],
            "start_station_id": ["5506.14", "5506.14", None, "5506.14"],
        }
    )
    result = aggregate_chunk(chunk, schema)
    # Only the first row is fully valid: row 1 has an unparseable timestamp,
    # row 2 has no station id, and row 3 has no timestamp.
    assert result["departures"].sum() == 1
    assert result["hour_ts"].notna().all()
    assert result["station_short_name"].notna().all()


def test_station_ids_keep_their_string_form():
    """Modern ids like "5506.14" must not be coerced to floats.

    Float coercion would turn 5506.10 into 5506.1 and break the join to the
    GBFS short_name, silently dropping stations from training.
    """
    schema = resolve_schema(MODERN_2020_ONWARDS)
    chunk = pd.DataFrame(
        {
            "started_at": ["2026-06-01 08:14:00", "2026-06-01 08:20:00"],
            "start_station_id": ["5506.10", "5506.1"],
        }
    )
    result = aggregate_chunk(chunk, schema)
    assert set(result["station_short_name"]) == {"5506.10", "5506.1"}


def test_whitespace_in_station_ids_is_stripped():
    schema = resolve_schema(MODERN_2020_ONWARDS)
    chunk = pd.DataFrame(
        {
            "started_at": ["2026-06-01 08:14:00", "2026-06-01 08:20:00"],
            "start_station_id": [" 5506.14", "5506.14 "],
        }
    )
    result = aggregate_chunk(chunk, schema)
    assert len(result) == 1
    assert result["departures"].iloc[0] == 2


def test_empty_chunk_returns_empty_frame():
    schema = resolve_schema(MODERN_2020_ONWARDS)
    empty = pd.DataFrame({"started_at": [], "start_station_id": []})
    assert aggregate_chunk(empty, schema).empty
