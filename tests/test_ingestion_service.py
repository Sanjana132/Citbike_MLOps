"""The always-on GBFS poller.

Ingestion moved out of Airflow because the numbers were unambiguous: over 6
hours, 4 DAG runs completed of an expected 72, single runs ran 1.5-2.5 hours
against a 4-minute execution_timeout that never fired, and the scheduler lost
its heartbeat 10 times in 3 hours. A 60-second feed is not a batch job.

These tests pin the properties that make a long-running poller trustworthy: it
never dies, and its cadence does not drift.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.ingestion import service as service_module
from src.ingestion.gbfs_client import GBFSClient
from src.ingestion.service import PollerState, run_poller


@pytest.fixture
def stub_feed(monkeypatch):
    """A working feed plus counters, with all persistence stubbed out."""
    calls = {"fetch": 0, "written": 0, "derive": 0, "stations": 0}

    def fake_fetch(self, snapshot_ts=None):
        calls["fetch"] += 1
        stamp = pd.Timestamp("2026-08-01T12:00:00Z") + pd.Timedelta(minutes=calls["fetch"])
        return pd.DataFrame(
            {"station_id": ["a"], "snapshot_ts": [stamp], "num_bikes_available": [5]}
        )

    monkeypatch.setattr(GBFSClient, "fetch_station_status", fake_fetch)
    monkeypatch.setattr(
        "src.storage.db.write_status_snapshots",
        lambda frame, *a, **k: calls.__setitem__("written", calls["written"] + len(frame))
        or len(frame),
    )
    monkeypatch.setattr(
        service_module,
        "_derive_once",
        lambda cfg, state: calls.__setitem__("derive", calls["derive"] + 1),
    )
    monkeypatch.setattr(
        service_module,
        "_refresh_stations",
        lambda cfg: calls.__setitem__("stations", calls["stations"] + 1),
    )
    return calls


@pytest.fixture
def instant_clock(monkeypatch):
    """Fast-forward a fake clock instead of really waiting.

    Returning immediately from ``wait`` without advancing time would busy-spin
    for a real 60 seconds per poll, since the loop only acts when the fixed grid
    says it is due. The clock has to move, not the wait be skipped.
    """
    clock = {"t": 1000.0}
    monkeypatch.setattr(service_module.time, "monotonic", lambda: clock["t"])

    def fake_wait(self, timeout=None):
        clock["t"] += timeout if timeout else 0.1
        return False

    monkeypatch.setattr(service_module.threading.Event, "wait", fake_wait)
    return clock


def test_poller_samples_repeatedly(stub_feed, instant_clock, config):
    state = run_poller(config, max_iterations=5)
    assert stub_feed["fetch"] == 5
    assert state.polls_ok == 5
    assert state.rows_written == 5


def test_poller_refreshes_station_metadata_on_start(stub_feed, instant_clock, config):
    run_poller(config, max_iterations=1)
    assert stub_feed["stations"] == 1


def test_a_failed_poll_does_not_stop_the_loop(monkeypatch, stub_feed, instant_clock, config):
    """The whole point of a service: it must survive a bad response."""
    state_holder = {"n": 0}

    def flaky(self, snapshot_ts=None):
        state_holder["n"] += 1
        if state_holder["n"] % 2 == 0:
            raise RuntimeError("upstream 503")
        stamp = pd.Timestamp("2026-08-01T12:00:00Z") + pd.Timedelta(minutes=state_holder["n"])
        return pd.DataFrame(
            {"station_id": ["a"], "snapshot_ts": [stamp], "num_bikes_available": [5]}
        )

    monkeypatch.setattr(GBFSClient, "fetch_station_status", flaky)

    state = run_poller(config, max_iterations=6)
    assert state.polls_ok == 3
    assert state.polls_failed == 3


def test_an_empty_response_is_counted_not_fatal(monkeypatch, stub_feed, instant_clock, config):
    monkeypatch.setattr(
        GBFSClient, "fetch_station_status", lambda self, snapshot_ts=None: pd.DataFrame()
    )
    state = run_poller(config, max_iterations=3)
    assert state.polls_ok == 0
    assert state.polls_failed == 3


def test_a_persistence_failure_does_not_stop_the_loop(monkeypatch, stub_feed, instant_clock, config):
    def explode(*args, **kwargs):
        raise RuntimeError("database unreachable")

    monkeypatch.setattr("src.storage.db.write_status_snapshots", explode)
    state = run_poller(config, max_iterations=3)
    assert state.polls_failed == 3
    assert state.polls_ok == 0


def test_a_derivation_failure_does_not_stop_the_loop(monkeypatch, stub_feed, instant_clock, config):
    """Deriving is downstream work; if it breaks, keep collecting raw snapshots."""

    def explode(cfg, state):
        state.derives_failed += 1
        raise RuntimeError("derivation exploded")

    monkeypatch.setattr(service_module, "_derive_once", explode)
    original = config["ingestion"].get("derive_interval_seconds")
    try:
        # Derive on essentially every tick so the failure is definitely hit.
        config["ingestion"]["derive_interval_seconds"] = 1
        state = run_poller(config, max_iterations=3)
    finally:
        config["ingestion"]["derive_interval_seconds"] = original

    # The loop kept polling despite the derivation blowing up every cycle.
    assert state.polls_ok == 3
    assert state.derives_failed >= 1


def test_cadence_is_computed_on_a_fixed_grid(stub_feed, instant_clock, config, monkeypatch):
    """Ticks come from a schedule, not from sleeping after variable work.

    Sleeping a fixed amount *after* each poll would let slow polls slowly
    desynchronise the sampling grid.
    """
    clock = instant_clock
    fetches = []

    def slow_fetch(self, snapshot_ts=None):
        fetches.append(clock["t"])
        clock["t"] += 25.0  # each poll takes a variable, non-trivial time
        stamp = pd.Timestamp("2026-08-01T12:00:00Z") + pd.Timedelta(seconds=clock["t"])
        return pd.DataFrame(
            {"station_id": ["a"], "snapshot_ts": [stamp], "num_bikes_available": [5]}
        )

    monkeypatch.setattr(GBFSClient, "fetch_station_status", slow_fetch)
    run_poller(config, max_iterations=3)

    spacing = float(config.get_path("ingestion.sample_spacing_seconds"))
    # Grid positions, not "previous finish + spacing".
    assert fetches[1] - fetches[0] == pytest.approx(spacing, abs=1.0)


def test_state_summary_reports_what_an_operator_needs(config, stub_feed, instant_clock):
    state = run_poller(config, max_iterations=2)
    summary = state.summary()
    for field in ("uptime=", "polls_ok=", "polls_failed=", "rows=", "derives_ok="):
        assert field in summary


def test_fresh_state_starts_at_zero():
    state = PollerState()
    assert state.polls_ok == 0
    assert state.polls_failed == 0
    assert "polls_ok=0" in state.summary()
