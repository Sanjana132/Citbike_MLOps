"""Storage layer, exercised against SQLite.

Airflow retries tasks, so every write must be idempotent: a retried ingestion
run must not duplicate snapshots or double-count departures. That property is
what these tests actually check.
"""

from __future__ import annotations

import pandas as pd
import pytest
from sqlalchemy import create_engine, text

from src.storage import db as db_module
from src.storage.db import (
    init_schema,
    split_sql_statements,
    write_hourly_departures,
    write_predictions,
    write_station_info,
    write_status_snapshots,
)


@pytest.fixture
def engine(tmp_path, monkeypatch):
    url = f"sqlite:///{tmp_path/'test.db'}"
    test_engine = create_engine(url, future=True)
    monkeypatch.setattr(db_module, "get_engine", lambda url=None: test_engine)
    init_schema(test_engine)
    return test_engine


def test_schema_splitter_ignores_semicolons_inside_comments():
    """A semicolon in comment prose must not cut a statement in half.

    sql/init.sql really contains one ("...*proxy*; nothing here is a ride
    count."), which previously produced an unparseable fragment.
    """
    ddl = """
    -- a comment with a semicolon; and more prose
    CREATE TABLE a (x INT);
    -- another; comment
    CREATE TABLE b (y INT);
    """
    statements = split_sql_statements(ddl)
    assert len(statements) == 2
    assert all(s.startswith("CREATE TABLE") for s in statements)


def test_init_schema_creates_every_table(engine):
    tables = {
        row[0]
        for row in engine.connect().execute(
            text("SELECT name FROM sqlite_master WHERE type='table'")
        )
    }
    assert {
        "station_info", "status_snapshots", "hourly_departures",
        "predictions", "features", "model_metrics",
    }.issubset(tables)


def test_init_schema_is_rerunnable(engine):
    init_schema(engine)  # must not raise on an existing schema


def test_station_info_upsert_is_idempotent(engine, station_info):
    write_station_info(station_info, engine)
    write_station_info(station_info, engine)
    count = pd.read_sql("SELECT COUNT(*) AS n FROM station_info", engine)["n"].iloc[0]
    assert count == len(station_info)


def test_station_info_upsert_updates_changed_rows(engine, station_info):
    write_station_info(station_info, engine)
    updated = station_info.copy()
    updated.loc[0, "capacity"] = 999
    write_station_info(updated, engine)

    result = pd.read_sql(
        text("SELECT capacity FROM station_info WHERE station_id = :sid"),
        engine,
        params={"sid": station_info.loc[0, "station_id"]},
    )
    assert result["capacity"].iloc[0] == 999


def test_snapshot_writes_are_idempotent(engine, snapshots):
    write_status_snapshots(snapshots, engine)
    write_status_snapshots(snapshots, engine)  # simulate an Airflow retry
    count = pd.read_sql("SELECT COUNT(*) AS n FROM status_snapshots", engine)["n"].iloc[0]
    assert count == len(snapshots)


def test_proxy_departures_never_overwrite_true_labels(engine):
    """label_source is part of the key, so a proxy row cannot clobber truth."""
    row = {"station_short_name": "1001.01", "hour_ts": pd.Timestamp("2026-06-01T08:00:00")}

    write_hourly_departures(
        pd.DataFrame([{**row, "departures": 42.0}]), engine, label_source="trip_csv"
    )
    write_hourly_departures(
        pd.DataFrame([{**row, "departures": 7.0}]), engine, label_source="gbfs_proxy"
    )

    stored = pd.read_sql("SELECT label_source, departures FROM hourly_departures", engine)
    assert len(stored) == 2
    by_source = dict(zip(stored["label_source"], stored["departures"], strict=True))
    assert by_source["trip_csv"] == 42.0
    assert by_source["gbfs_proxy"] == 7.0


def test_reprocessing_an_hour_updates_rather_than_duplicates(engine):
    """The ingestion DAG reprocesses recent hours; that must correct, not append."""
    row = {
        "station_short_name": "1001.01",
        "hour_ts": pd.Timestamp("2026-06-01T08:00:00"),
        "label_source": "gbfs_proxy",
    }
    write_hourly_departures(pd.DataFrame([{**row, "departures": 3.0}]), engine)
    write_hourly_departures(pd.DataFrame([{**row, "departures": 5.0}]), engine)

    stored = pd.read_sql("SELECT departures FROM hourly_departures", engine)
    assert len(stored) == 1
    assert stored["departures"].iloc[0] == 5.0


def test_predictions_append_for_audit(engine):
    """Predictions are an append-only log; history must be preserved."""
    frame = pd.DataFrame(
        [
            {
                "entity_id": "cluster_000",
                "target_hour_ts": pd.Timestamp("2026-08-01T10:00:00"),
                "predicted": 12.5,
                "model_name": "citibike-demand-forecaster",
                "model_version": "1",
            }
        ]
    )
    write_predictions(frame, engine)
    write_predictions(frame.assign(predicted=13.5, model_version="2"), engine)

    stored = pd.read_sql("SELECT * FROM predictions ORDER BY id", engine)
    assert len(stored) == 2
    assert list(stored["model_version"]) == ["1", "2"]


def test_empty_writes_are_no_ops(engine):
    assert write_station_info(pd.DataFrame(columns=["station_id", "short_name", "name", "lat", "lon", "capacity", "region_id"]), engine) == 0
    assert write_predictions(pd.DataFrame(), engine) == 0


# --------------------------------------------------------------------------
# Engine caching
# --------------------------------------------------------------------------

def test_engine_is_created_once_and_reused(monkeypatch):
    """Regression test for the defect behind every ingestion stall.

    The cache check compared ``str(_ENGINE.url)`` against the URL it was built
    from, but SQLAlchemy renders the password as "***" - so the comparison never
    matched and each call built a new engine and connection pool while leaking
    the previous one. It exhausted Postgres (39 idle connections against
    max_connections=100) and caused 67-minute hangs in both the Airflow DAG and
    the poller, which looked for a long time like a scheduler problem.
    """
    import src.storage.db as module

    monkeypatch.setattr(module, "_ENGINE", None)
    monkeypatch.setattr(module, "_ENGINE_URL", None)
    monkeypatch.setenv("DATABASE_URL", "sqlite://")

    created = []
    real_create = module.create_engine

    def counting_create(*args, **kwargs):
        created.append(args[0])
        return real_create(*args, **kwargs)

    monkeypatch.setattr(module, "create_engine", counting_create)

    first = module.get_engine()
    for _ in range(20):
        assert module.get_engine() is first

    assert len(created) == 1, f"engine rebuilt {len(created)} times; the pool leaks"


def test_password_bearing_url_still_hits_the_cache(monkeypatch):
    """The exact shape that broke: a URL containing a password.

    Without a separately stored URL this matched "***" and missed every time.
    """
    import src.storage.db as module

    monkeypatch.setattr(module, "_ENGINE", None)
    monkeypatch.setattr(module, "_ENGINE_URL", None)

    url = "postgresql+psycopg2://user:secret@localhost:5432/db"
    created = []

    class FakeEngine:
        def __init__(self, target):
            from sqlalchemy.engine.url import make_url

            self.url = make_url(target)

        def dispose(self):
            pass

    monkeypatch.setattr(
        module, "create_engine", lambda target, **kw: created.append(target) or FakeEngine(target)
    )

    first = module.get_engine(url)
    second = module.get_engine(url)
    assert first is second
    assert len(created) == 1


def test_switching_url_disposes_the_old_pool(monkeypatch):
    """A genuine URL change must release the old connections, not orphan them."""
    import src.storage.db as module

    monkeypatch.setattr(module, "_ENGINE", None)
    monkeypatch.setattr(module, "_ENGINE_URL", None)

    disposed = []

    class FakeEngine:
        def __init__(self, target):
            from sqlalchemy.engine.url import make_url

            self.url = make_url(target)
            self._target = target

        def dispose(self):
            disposed.append(self._target)

    monkeypatch.setattr(module, "create_engine", lambda target, **kw: FakeEngine(target))

    module.get_engine("sqlite:///first.db")
    module.get_engine("sqlite:///second.db")
    assert disposed == ["sqlite:///first.db"]
