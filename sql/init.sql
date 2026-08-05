-- Schema for the Citi Bike demand platform.
-- Applied automatically by the Postgres container on first start, and by
-- src/storage/db.py::init_schema() for local/dev runs.

CREATE TABLE IF NOT EXISTS station_info (
    station_id      TEXT PRIMARY KEY,
    short_name      TEXT,
    name            TEXT,
    lat             DOUBLE PRECISION NOT NULL,
    lon             DOUBLE PRECISION NOT NULL,
    capacity        INTEGER,
    region_id       TEXT,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_station_info_short_name ON station_info (short_name);

-- Raw GBFS inventory snapshots. This is INVENTORY (bikes docked right now),
-- not demand. derive_departures.py turns consecutive snapshots into a
-- departures *proxy*; nothing here is a ride count.
CREATE TABLE IF NOT EXISTS status_snapshots (
    station_id           TEXT        NOT NULL,
    snapshot_ts          TIMESTAMPTZ NOT NULL,
    num_bikes_available  INTEGER,
    num_ebikes_available INTEGER,
    num_bikes_disabled   INTEGER,
    num_docks_available  INTEGER,
    num_docks_disabled   INTEGER,
    is_installed         SMALLINT,
    is_renting           SMALLINT,
    is_returning         SMALLINT,
    last_reported        TIMESTAMPTZ,
    PRIMARY KEY (station_id, snapshot_ts)
);
CREATE INDEX IF NOT EXISTS idx_snapshots_ts ON status_snapshots (snapshot_ts);

-- Hourly departures per station.
-- label_source distinguishes ground truth from the proxy:
--   'trip_csv'   -> true completed-trip counts from historical CSVs
--   'gbfs_proxy' -> estimated from inventory deltas (NOISY - see README)
CREATE TABLE IF NOT EXISTS hourly_departures (
    station_short_name TEXT        NOT NULL,
    hour_ts            TIMESTAMP   NOT NULL,
    departures         DOUBLE PRECISION NOT NULL,
    label_source       TEXT        NOT NULL DEFAULT 'gbfs_proxy',
    n_intervals        INTEGER,
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (station_short_name, hour_ts, label_source)
);
CREATE INDEX IF NOT EXISTS idx_hourly_departures_ts ON hourly_departures (hour_ts);

-- Every prediction served, so monitoring can join them to realised demand.
CREATE TABLE IF NOT EXISTS predictions (
    id             BIGSERIAL PRIMARY KEY,
    entity_id      TEXT        NOT NULL,
    target_hour_ts TIMESTAMP   NOT NULL,
    predicted      DOUBLE PRECISION NOT NULL,
    model_name     TEXT,
    model_version  TEXT,
    predicted_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_predictions_target ON predictions (target_hour_ts);
CREATE INDEX IF NOT EXISTS idx_predictions_entity ON predictions (entity_id, target_hour_ts);

-- Feature rows persisted at prediction time. Kept as the drift reference:
-- Evidently compares these against the training feature distribution.
CREATE TABLE IF NOT EXISTS features (
    entity_id  TEXT        NOT NULL,
    hour_ts    TIMESTAMP   NOT NULL,
    payload    JSONB       NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (entity_id, hour_ts)
);
CREATE INDEX IF NOT EXISTS idx_features_hour ON features (hour_ts);

-- Rolling monitoring output written by monitor_dag.
CREATE TABLE IF NOT EXISTS model_metrics (
    id             BIGSERIAL PRIMARY KEY,
    computed_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    window_start   TIMESTAMP,
    window_end     TIMESTAMP,
    model_name     TEXT,
    model_version  TEXT,
    n_samples      INTEGER,
    rolling_mae    DOUBLE PRECISION,
    rolling_rmse   DOUBLE PRECISION,
    drift_share    DOUBLE PRECISION,
    dataset_drift  BOOLEAN,
    breached       BOOLEAN,
    breach_reason  TEXT
);
CREATE INDEX IF NOT EXISTS idx_model_metrics_computed ON model_metrics (computed_at);
