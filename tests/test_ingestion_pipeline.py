"""GBFS parsing and ingestion orchestration.

Feeds change shape without warning, so the parser is tested against the kinds of
malformation real producers emit. The rule under test throughout: log and skip,
never crash — an ingestion run that dies stops collecting the history the model
depends on.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.ingestion.gbfs_client import GBFSClient, _to_bool_int, _to_float, _to_int


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def json(self) -> dict:
        return self._payload


@pytest.fixture
def client(config):
    return GBFSClient(config)


def _patch_feed(monkeypatch, payload: dict) -> None:
    from src.ingestion import gbfs_client as module

    monkeypatch.setattr(module, "get_with_retries", lambda *a, **k: FakeResponse(payload))


# --------------------------------------------------------------------------
# Coercion helpers
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "value,expected",
    [(5, 5), ("7", 7), (3.0, 3), (None, None), ("abc", None), ([], None)],
)
def test_to_int_coerces_tolerantly(value, expected):
    assert _to_int(value) == expected


@pytest.mark.parametrize(
    "value,expected",
    [(True, 1), (False, 0), (1, 1), (0, 0), ("true", 1), ("no", 0), (None, 0), ("?", 0)],
)
def test_to_bool_int_handles_every_encoding(value, expected):
    """GBFS 1.1 says 0/1, but producers emit booleans and strings too."""
    assert _to_bool_int(value) == expected


def test_to_float_rejects_unparseable_values():
    assert _to_float("12.5") == 12.5
    assert _to_float("not-a-number") is None
    assert _to_float(None) is None


# --------------------------------------------------------------------------
# station_information
# --------------------------------------------------------------------------

def test_station_information_parses_valid_stations(monkeypatch, client):
    _patch_feed(
        monkeypatch,
        {
            "last_updated": 1785887009,
            "data": {
                "stations": [
                    {
                        "station_id": "uuid-1",
                        "short_name": "5506.14",
                        "name": "E 6 St & Ave D",
                        "lat": 40.72,
                        "lon": -73.97,
                        "capacity": 31,
                        "region_id": "71",
                    }
                ]
            },
        },
    )
    frame = client.fetch_station_information()
    assert len(frame) == 1
    # short_name is the join key to the historical trip CSVs.
    assert frame["short_name"].iloc[0] == "5506.14"
    assert frame["capacity"].iloc[0] == 31


def test_station_information_skips_malformed_stations(monkeypatch, client):
    """One bad station must not lose the rest of the feed."""
    _patch_feed(
        monkeypatch,
        {
            "data": {
                "stations": [
                    {"station_id": "good", "short_name": "1.1", "lat": 40.7, "lon": -74.0, "capacity": 10},
                    {"station_id": "no-coords", "short_name": "2.2"},
                    {"short_name": "no-id", "lat": 40.7, "lon": -74.0},
                    {"station_id": "null-island", "short_name": "3.3", "lat": 0.0, "lon": 0.0},
                    {"station_id": "bad-lat", "short_name": "4.4", "lat": "abc", "lon": -74.0},
                ]
            }
        },
    )
    frame = client.fetch_station_information()
    assert list(frame["station_id"]) == ["good"]


def test_missing_capacity_defaults_to_zero(monkeypatch, client):
    _patch_feed(
        monkeypatch,
        {"data": {"stations": [{"station_id": "a", "short_name": "1.1", "lat": 40.7, "lon": -74.0}]}},
    )
    assert client.fetch_station_information()["capacity"].iloc[0] == 0


# --------------------------------------------------------------------------
# station_status
# --------------------------------------------------------------------------

def test_station_status_parses_a_snapshot(monkeypatch, client):
    _patch_feed(
        monkeypatch,
        {
            "last_updated": 1785887009,
            "ttl": 60,
            "data": {
                "stations": [
                    {
                        "station_id": "uuid-1",
                        "num_bikes_available": 12,
                        "num_ebikes_available": 3,
                        "num_docks_available": 19,
                        "is_installed": 1,
                        "is_renting": 1,
                        "is_returning": 1,
                        "last_reported": 1785886900,
                    }
                ]
            },
        },
    )
    frame = client.fetch_station_status()
    assert frame["num_bikes_available"].iloc[0] == 12
    assert frame["is_renting"].iloc[0] == 1
    assert pd.notna(frame["snapshot_ts"].iloc[0])


def test_station_status_skips_rows_without_inventory(monkeypatch, client):
    """A station with no bike count carries no usable inventory signal."""
    _patch_feed(
        monkeypatch,
        {
            "data": {
                "stations": [
                    {"station_id": "a", "num_bikes_available": 5},
                    {"station_id": "b"},
                    {"num_bikes_available": 3},
                ]
            }
        },
    )
    frame = client.fetch_station_status()
    assert list(frame["station_id"]) == ["a"]


def test_unusable_feed_shape_raises(monkeypatch, client):
    """A structurally wrong feed is a real failure, not a row to skip."""
    _patch_feed(monkeypatch, {"data": {}})
    with pytest.raises(ValueError, match="data.stations"):
        client.fetch_station_status()


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def test_ingest_status_reports_zero_on_an_empty_feed(monkeypatch, config):
    """An empty feed must return a clean zero, not raise."""
    from src.ingestion import pipeline

    monkeypatch.setattr(
        GBFSClient, "fetch_station_status", lambda self, snapshot_ts=None: pd.DataFrame()
    )
    # Inject a no-op sleep: the loop's behaviour is under test, not the delay.
    result = pipeline.ingest_station_status(config, sleep=lambda _: None)
    assert result["snapshots_written"] == 0


def test_ingest_takes_several_samples_per_run(monkeypatch, config):
    """One sample per run captured only a fifth of the feed's updates.

    The feed advances last_updated every 60s, so a single poll every 5 minutes
    left ~12 intervals an hour against a 50% coverage guard - a little scheduler
    lag then discarded whole hours.
    """
    from src.ingestion import pipeline
    from src.storage import db

    calls = {"fetch": 0, "sleeps": []}

    def fake_fetch(self, snapshot_ts=None):
        calls["fetch"] += 1
        stamp = pd.Timestamp("2026-08-01T12:00:00Z") + pd.Timedelta(minutes=calls["fetch"])
        return pd.DataFrame(
            {"station_id": ["a"], "snapshot_ts": [stamp], "num_bikes_available": [5]}
        )

    monkeypatch.setattr(GBFSClient, "fetch_station_status", fake_fetch)
    monkeypatch.setattr(db, "write_status_snapshots", lambda frame, *a, **k: len(frame))

    expected = int(config.get_path("ingestion.samples_per_run"))
    spacing = float(config.get_path("ingestion.sample_spacing_seconds"))
    result = pipeline.ingest_station_status(config, sleep=calls["sleeps"].append)

    assert calls["fetch"] == expected
    assert result["distinct_timestamps"] == expected
    # Sleeps happen between samples, never after the last one - the next
    # scheduled run continues the sequence.
    assert calls["sleeps"] == [spacing] * (expected - 1)


def test_one_failed_sample_does_not_lose_the_others(monkeypatch, config):
    """A transient GBFS error must cost one sample, not the whole run."""
    from src.ingestion import pipeline
    from src.storage import db

    state = {"n": 0}

    def flaky_fetch(self, snapshot_ts=None):
        state["n"] += 1
        if state["n"] == 2:
            raise RuntimeError("transient upstream failure")
        stamp = pd.Timestamp("2026-08-01T12:00:00Z") + pd.Timedelta(minutes=state["n"])
        return pd.DataFrame(
            {"station_id": ["a"], "snapshot_ts": [stamp], "num_bikes_available": [5]}
        )

    monkeypatch.setattr(GBFSClient, "fetch_station_status", flaky_fetch)
    monkeypatch.setattr(db, "write_status_snapshots", lambda frame, *a, **k: len(frame))

    expected = int(config.get_path("ingestion.samples_per_run"))
    result = pipeline.ingest_station_status(config, sleep=lambda _: None)
    assert result["snapshots_written"] == expected - 1


def test_derive_step_defers_the_in_progress_hour(monkeypatch, config, snapshots):
    """The newest hour is still accumulating; writing it would understate it."""
    from src.ingestion import pipeline
    from src.storage import db

    monkeypatch.setattr(db, "read_recent_snapshots", lambda hours=3, engine=None: snapshots)
    monkeypatch.setattr(
        db,
        "read_station_info",
        lambda config=None, engine=None: pd.DataFrame(
            {"station_id": ["uuid-a", "uuid-b"], "short_name": ["1001.01", "1002.01"]}
        ),
    )
    written: list = []
    monkeypatch.setattr(
        db, "write_hourly_departures", lambda frame, **kw: written.append(frame) or len(frame)
    )

    result = pipeline.derive_and_store_proxy_departures(config=config)
    # The fixture spans a single hour, which is therefore the in-progress hour.
    assert result["rows_written"] == 0
    assert not written


def test_derive_step_handles_no_snapshots(monkeypatch, config):
    from src.ingestion import pipeline
    from src.storage import db

    monkeypatch.setattr(db, "read_recent_snapshots", lambda hours=3, engine=None: pd.DataFrame())
    assert pipeline.derive_and_store_proxy_departures(config=config)["rows_written"] == 0
