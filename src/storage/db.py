"""Postgres access layer.

SQLAlchemy is used rather than raw psycopg2 so that the test suite can point
``DATABASE_URL`` at SQLite and exercise the same code paths without a server.
The production stack always runs Postgres.

Writes are idempotent (``ON CONFLICT DO UPDATE``) because Airflow retries tasks,
and a retried ingestion run must not duplicate snapshots or double-count
departures.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from src.config import PROJECT_ROOT, Config

logger = logging.getLogger(__name__)

DEFAULT_DATABASE_URL = "postgresql+psycopg2://citibike:citibike@localhost:5432/citibike"

_ENGINE: Engine | None = None


def database_url() -> str:
    return os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)


def get_engine(url: str | None = None) -> Engine:
    """Process-wide engine, created lazily."""
    global _ENGINE
    resolved = url or database_url()
    if _ENGINE is None or str(_ENGINE.url) != resolved:
        _ENGINE = create_engine(resolved, pool_pre_ping=True, future=True)
        logger.info("Database engine created for %s", _ENGINE.url.render_as_string(hide_password=True))
    return _ENGINE


def is_sqlite(engine: Engine | None = None) -> bool:
    return (engine or get_engine()).dialect.name == "sqlite"


def init_schema(engine: Engine | None = None) -> None:
    """Apply ``sql/init.sql``.

    SQLite cannot parse the Postgres DDL (``BIGSERIAL``, ``JSONB``,
    ``TIMESTAMPTZ``), so a small translation is applied for the test path.
    """
    engine = engine or get_engine()
    ddl = (Path(PROJECT_ROOT) / "sql" / "init.sql").read_text(encoding="utf-8")
    if is_sqlite(engine):
        ddl = (
            ddl.replace("BIGSERIAL PRIMARY KEY", "INTEGER PRIMARY KEY AUTOINCREMENT")
            .replace("TIMESTAMPTZ", "TIMESTAMP")
            .replace("JSONB", "TEXT")
            .replace("DOUBLE PRECISION", "REAL")
            .replace("NOW()", "CURRENT_TIMESTAMP")
        )
    with engine.begin() as connection:
        for statement in split_sql_statements(ddl):
            connection.execute(text(statement))
    logger.info("Schema initialised from sql/init.sql")


def split_sql_statements(ddl: str) -> list[str]:
    """Split a DDL script into executable statements.

    Line comments are stripped *before* splitting on semicolons. Splitting first
    is wrong: prose in a comment can contain a semicolon (init.sql really does),
    which cuts a statement in half and leaves the remainder of the comment
    prepended to the next CREATE TABLE.
    """
    lines = [line for line in ddl.splitlines() if not line.strip().startswith("--")]
    return [s.strip() for s in "\n".join(lines).split(";") if s.strip()]


def _upsert(
    frame: pd.DataFrame,
    table: str,
    conflict_columns: list[str],
    engine: Engine | None = None,
    *,
    update: bool = True,
) -> int:
    """Bulk idempotent insert."""
    if frame.empty:
        return 0
    engine = engine or get_engine()
    columns = list(frame.columns)
    placeholders = ", ".join(f":{c}" for c in columns)
    conflict = ", ".join(conflict_columns)

    if update:
        assignments = ", ".join(
            f"{c} = EXCLUDED.{c}" for c in columns if c not in conflict_columns
        )
        action = f"DO UPDATE SET {assignments}" if assignments else "DO NOTHING"
    else:
        action = "DO NOTHING"

    statement = text(
        f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders}) "
        f"ON CONFLICT ({conflict}) {action}"
    )
    with engine.begin() as connection:
        connection.execute(statement, _to_records(frame))
    return len(frame)


def _to_records(frame: pd.DataFrame) -> list[dict]:
    """Convert a frame to bind parameters every DBAPI driver accepts.

    pandas ``Timestamp`` and numpy scalars survive ``astype(object)``. psycopg2
    tolerates them, but sqlite3 raises "type 'Timestamp' is not supported", so
    datetime columns are converted to plain Python datetimes and NaN/NaT to None.
    """
    prepared = frame.copy()
    for column in prepared.columns:
        if pd.api.types.is_datetime64_any_dtype(prepared[column]):
            series = prepared[column]
            # sqlite cannot bind tz-aware datetimes either; normalise to naive UTC.
            if isinstance(series.dtype, pd.DatetimeTZDtype):
                series = series.dt.tz_convert("UTC").dt.tz_localize(None)
            # Built as an explicit object-dtype Series: .astype(object) leaves
            # pd.Timestamp in place, and .map() re-infers straight back to
            # datetime64. Only forcing dtype=object survives both.
            prepared[column] = pd.Series(
                [value.to_pydatetime() if pd.notna(value) else None for value in series],
                index=series.index,
                dtype=object,
            )
    prepared = prepared.astype(object).where(pd.notna(prepared), None)
    return prepared.to_dict(orient="records")


# --------------------------------------------------------------------------
# Writes
# --------------------------------------------------------------------------

def write_station_info(frame: pd.DataFrame, engine: Engine | None = None) -> int:
    keep = ["station_id", "short_name", "name", "lat", "lon", "capacity", "region_id"]
    return _upsert(frame[keep].copy(), "station_info", ["station_id"], engine)


def write_status_snapshots(frame: pd.DataFrame, engine: Engine | None = None) -> int:
    keep = [
        "station_id", "snapshot_ts", "num_bikes_available", "num_ebikes_available",
        "num_bikes_disabled", "num_docks_available", "num_docks_disabled",
        "is_installed", "is_renting", "is_returning", "last_reported",
    ]
    return _upsert(
        frame[[c for c in keep if c in frame.columns]].copy(),
        "status_snapshots",
        ["station_id", "snapshot_ts"],
        engine,
    )


def write_hourly_departures(
    frame: pd.DataFrame, engine: Engine | None = None, *, label_source: str = "gbfs_proxy"
) -> int:
    """Persist hourly departures.

    ``label_source`` is part of the primary key so proxy-derived rows can never
    silently overwrite true trip-CSV counts for the same hour.
    """
    out = frame.copy()
    out["label_source"] = out.get("label_source", label_source)
    keep = [c for c in ["station_short_name", "hour_ts", "departures", "label_source", "n_intervals"] if c in out.columns]
    return _upsert(
        out[keep], "hourly_departures", ["station_short_name", "hour_ts", "label_source"], engine
    )


def write_predictions(frame: pd.DataFrame, engine: Engine | None = None) -> int:
    """Append served predictions so monitoring can score them later."""
    if frame.empty:
        return 0
    engine = engine or get_engine()
    keep = ["entity_id", "target_hour_ts", "predicted", "model_name", "model_version"]
    subset = frame[[c for c in keep if c in frame.columns]].copy()
    subset.to_sql("predictions", engine, if_exists="append", index=False)
    return len(subset)


def write_features(frame: pd.DataFrame, engine: Engine | None = None) -> int:
    """Persist the exact feature rows used for a prediction (drift reference)."""
    if frame.empty:
        return 0
    payload = frame.drop(columns=["entity_id", "hour_ts"], errors="ignore")
    records = pd.DataFrame(
        {
            "entity_id": frame["entity_id"].astype(str),
            "hour_ts": pd.to_datetime(frame["hour_ts"]),
            "payload": [
                json.dumps({k: (None if pd.isna(v) else v) for k, v in row.items()})
                for row in payload.to_dict(orient="records")
            ],
        }
    )
    return _upsert(records, "features", ["entity_id", "hour_ts"])


def write_model_metrics(record: dict, engine: Engine | None = None) -> None:
    engine = engine or get_engine()
    pd.DataFrame([record]).to_sql("model_metrics", engine, if_exists="append", index=False)


# --------------------------------------------------------------------------
# Reads
# --------------------------------------------------------------------------

def prune_status_snapshots(retention_days: int, engine: Engine | None = None) -> int:
    """Delete snapshots older than the retention window.

    Snapshots are raw material for the hourly proxy, not a long-term record:
    once the hours they feed have been derived into ``hourly_departures`` they
    are dead weight. At ~2,460 stations sampled every minute they accumulate
    roughly 3.5M rows a day, so without pruning the table would dominate the
    database within a week.
    """
    if retention_days <= 0:
        return 0
    engine = engine or get_engine()
    cutoff = datetime.now(tz=UTC) - timedelta(days=retention_days)
    with engine.begin() as connection:
        deleted = connection.execute(
            text("DELETE FROM status_snapshots WHERE snapshot_ts < :cutoff"),
            {"cutoff": cutoff},
        ).rowcount
    if deleted:
        logger.info("Pruned %d status snapshots older than %d days", deleted, retention_days)
    return int(deleted or 0)


def read_station_info(config: Config | None = None, engine: Engine | None = None) -> pd.DataFrame:
    engine = engine or get_engine()
    return pd.read_sql(
        "SELECT station_id, short_name, name, lat, lon, capacity, region_id FROM station_info",
        engine,
    )


def read_latest_hourly_departures(
    window_hours: int,
    config: Config | None = None,
    engine: Engine | None = None,
) -> pd.DataFrame:
    """The most recent ``window_hours`` of departures, anchored on the newest row.

    The serving path needs a trailing slice relative to the *latest observation*,
    not to wall-clock now, because the two can be weeks apart. Doing that with a
    wide `window_days` read and trimming in pandas meant transferring the whole
    2.6M-row table to discard almost all of it - the single largest cost in a
    prediction request. Anchoring in SQL moves the filter to the database.
    """
    engine = engine or get_engine()
    query = text(
        "SELECT station_short_name, hour_ts, SUM(departures) AS departures "
        "FROM hourly_departures "
        "WHERE hour_ts >= ("
        "  SELECT MAX(hour_ts) - CAST(:window AS interval) FROM hourly_departures"
        ") "
        "GROUP BY station_short_name, hour_ts "
        "ORDER BY station_short_name, hour_ts"
    )
    if is_sqlite(engine):
        # SQLite has no interval type; it stores timestamps as text.
        query = text(
            "SELECT station_short_name, hour_ts, SUM(departures) AS departures "
            "FROM hourly_departures "
            "WHERE hour_ts >= datetime((SELECT MAX(hour_ts) FROM hourly_departures), "
            "                          '-' || :window_hours || ' hours') "
            "GROUP BY station_short_name, hour_ts "
            "ORDER BY station_short_name, hour_ts"
        )
        params: dict[str, object] = {"window_hours": int(window_hours)}
    else:
        params = {"window": f"{int(window_hours)} hours"}

    frame = pd.read_sql(query, engine, params=params)
    if not frame.empty:
        frame["hour_ts"] = pd.to_datetime(frame["hour_ts"])
    return frame


def read_hourly_departures(
    window_days: int = 90,
    config: Config | None = None,
    engine: Engine | None = None,
    *,
    label_source: str | None = None,
) -> pd.DataFrame:
    engine = engine or get_engine()
    cutoff = datetime.now(tz=UTC).replace(tzinfo=None) - timedelta(days=window_days)
    query = (
        "SELECT station_short_name, hour_ts, SUM(departures) AS departures "
        "FROM hourly_departures WHERE hour_ts >= :cutoff "
    )
    params: dict[str, object] = {"cutoff": cutoff}
    if label_source:
        query += "AND label_source = :label_source "
        params["label_source"] = label_source
    query += "GROUP BY station_short_name, hour_ts ORDER BY station_short_name, hour_ts"
    frame = pd.read_sql(text(query), engine, params=params)
    if not frame.empty:
        frame["hour_ts"] = pd.to_datetime(frame["hour_ts"])
    return frame


def read_recent_snapshots(
    hours: int = 6, engine: Engine | None = None
) -> pd.DataFrame:
    engine = engine or get_engine()
    cutoff = datetime.now(tz=UTC) - timedelta(hours=hours)
    frame = pd.read_sql(
        text(
            "SELECT * FROM status_snapshots WHERE snapshot_ts >= :cutoff "
            "ORDER BY station_id, snapshot_ts"
        ),
        engine,
        params={"cutoff": cutoff},
    )
    if not frame.empty:
        frame["snapshot_ts"] = pd.to_datetime(frame["snapshot_ts"])
    return frame


def read_predictions(
    start: datetime, end: datetime, engine: Engine | None = None
) -> pd.DataFrame:
    engine = engine or get_engine()
    frame = pd.read_sql(
        text(
            "SELECT entity_id, target_hour_ts, predicted, model_name, model_version, "
            "predicted_at FROM predictions "
            "WHERE target_hour_ts >= :start AND target_hour_ts < :end"
        ),
        engine,
        params={"start": start, "end": end},
    )
    if not frame.empty:
        frame["target_hour_ts"] = pd.to_datetime(frame["target_hour_ts"])
    return frame


def read_model_metrics(limit: int = 500, engine: Engine | None = None) -> pd.DataFrame:
    engine = engine or get_engine()
    return pd.read_sql(
        text("SELECT * FROM model_metrics ORDER BY computed_at DESC LIMIT :limit"),
        engine,
        params={"limit": limit},
    )


def read_features(start: datetime, end: datetime, engine: Engine | None = None) -> pd.DataFrame:
    """Reconstruct persisted feature rows into a flat frame for drift analysis."""
    engine = engine or get_engine()
    raw = pd.read_sql(
        text("SELECT entity_id, hour_ts, payload FROM features WHERE hour_ts >= :start AND hour_ts < :end"),
        engine,
        params={"start": start, "end": end},
    )
    if raw.empty:
        return raw
    expanded = pd.json_normalize(
        [json.loads(p) if isinstance(p, str) else p for p in raw["payload"]]
    )
    expanded.insert(0, "hour_ts", pd.to_datetime(raw["hour_ts"]).to_numpy())
    expanded.insert(0, "entity_id", raw["entity_id"].to_numpy())
    return expanded
