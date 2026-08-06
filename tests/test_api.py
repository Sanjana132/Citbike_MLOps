"""API contract tests.

A stub model stands in for the registry so these run without MLflow, Postgres,
or a network. What is being tested is the service's behaviour - especially that
it degrades honestly instead of returning confident nonsense.
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.features.build_features import feature_columns
from src.features.station_clusters import fit_entity_mapping
from src.serving.history import HistoryResult
from src.serving.model_loader import ModelBundle


class StubModel:
    """Returns the row sum of lag_1h, so predictions track the inputs."""

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.nan_to_num(features["lag_1h"].to_numpy(dtype=float), nan=1.0) + 0.5


@pytest.fixture
def client(monkeypatch, station_info, hourly_departures, config):
    from src.serving import api as api_module

    mapping = fit_entity_mapping(station_info, config)
    bundle = ModelBundle(
        model=StubModel(),
        mapping=mapping,
        metadata={
            "features": feature_columns(config),
            "model_kind": "lightgbm",
            "trained_at": "2026-08-01T00:00:00+00:00",
            "train_start": "2026-04-01 00:00:00",
            "train_end": "2026-05-01 00:00:00",
            "granularity": mapping.granularity,
            "label_provenance": "trip_csv_true_labels",
        },
        model_name="citibike-demand-forecaster",
        model_version="7",
        run_id="stub-run",
    )

    monkeypatch.setattr(api_module.loader, "_bundle", bundle, raising=False)
    monkeypatch.setattr(api_module.loader, "load", lambda force=False: bundle)
    monkeypatch.setattr(
        api_module,
        "load_recent_history",
        lambda cfg=None, **kw: HistoryResult(
            hourly_departures, "postgres", pd.Timestamp(hourly_departures["hour_ts"].max())
        ),
    )
    # Keep the weather providers off the network.
    monkeypatch.setattr(api_module, "_load_weather", lambda hours, target: pd.DataFrame())
    # Prediction logging is exercised separately; keep these tests DB-free.
    monkeypatch.setattr(api_module, "_log_prediction", lambda *a, **k: None)
    # Liveness hits the database; it has its own tests in test_liveness.py.
    monkeypatch.setattr(
        "src.monitoring.liveness.check_pipeline_liveness",
        lambda cfg=None: SimpleNamespace(healthy=True, summary=lambda: "fresh"),
    )

    with TestClient(api_module.app) as test_client:
        yield test_client


def test_health_reports_ok_when_everything_is_available(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"] == "7"
    assert body["history_source"] == "postgres"
    assert body["pipeline_live"] is True


def test_health_degrades_when_ingestion_has_stalled(monkeypatch, client):
    """A stalled pipeline must show up here, not only in the hourly DAG.

    This is the signal that was missing when ingestion stopped for 42 hours
    while every other check reported healthy.
    """
    monkeypatch.setattr(
        "src.monitoring.liveness.check_pipeline_liveness",
        lambda cfg=None: SimpleNamespace(
            healthy=False, summary=lambda: "gbfs_snapshots: newest row is 2520 min old"
        ),
    )
    body = client.get("/health").json()
    assert body["status"] == "degraded"
    assert body["pipeline_live"] is False
    assert any("stalled" in w for w in body["warnings"])


def test_model_info_exposes_provenance_and_version(client):
    body = client.get("/model-info").json()
    assert body["model_version"] == "7"
    assert body["model_kind"] == "lightgbm"
    assert body["granularity"] == "cluster"
    # Whether the model learned from true trips or from the proxy must be visible.
    assert body["label_provenance"] == "trip_csv_true_labels"
    assert body["n_features"] == len(feature_columns())


def test_predict_returns_one_row_per_entity(client, station_info, config):
    response = client.post("/predict", json={"log_prediction": False})
    assert response.status_code == 200
    body = response.json()

    # n_clusters is capped at the number of stations, so the 12-station fixture
    # yields 12 entities rather than the configured 40.
    expected_entities = min(int(config.get_path("features.n_clusters")), len(station_info))
    assert len(body["predictions"]) == expected_entities
    assert len({p["entity_id"] for p in body["predictions"]}) == expected_entities
    assert all(p["predicted_departures"] >= 0 for p in body["predictions"])
    assert all(p["lat"] is not None and p["lon"] is not None for p in body["predictions"])
    assert body["model_version"] == "7"


def test_predict_defaults_to_the_hour_after_the_last_observation(client, hourly_departures):
    body = client.post("/predict", json={"log_prediction": False}).json()
    expected = pd.Timestamp(hourly_departures["hour_ts"].max()) + pd.Timedelta(hours=1)
    assert pd.Timestamp(body["target_hour"]) == expected


def test_predict_filters_by_station_id(client, station_info):
    target_station = station_info["short_name"].iloc[0]
    body = client.post(
        "/predict", json={"station_ids": [target_station], "log_prediction": False}
    ).json()
    assert len(body["predictions"]) == 1


def test_unknown_station_is_reported_not_silently_dropped(client, station_info):
    known = station_info["short_name"].iloc[0]
    body = client.post(
        "/predict", json={"station_ids": [known, "9999.99"], "log_prediction": False}
    ).json()
    assert any("Unknown station ids" in w for w in body["warnings"])

    response = client.post(
        "/predict", json={"station_ids": ["9999.99"], "log_prediction": False}
    )
    assert response.status_code == 404


def test_forecasting_beyond_the_horizon_is_rejected(client, hourly_departures):
    """A far-future hour would have all-NaN lags; refuse rather than guess."""
    far = pd.Timestamp(hourly_departures["hour_ts"].max()) + pd.Timedelta(days=90)
    response = client.post(
        "/predict", json={"target_hour": far.isoformat(), "log_prediction": False}
    )
    assert response.status_code == 422
    assert "beyond the latest observation" in response.json()["detail"]


def test_multi_hour_horizon_warns(client, hourly_departures):
    target = pd.Timestamp(hourly_departures["hour_ts"].max()) + pd.Timedelta(hours=6)
    body = client.post(
        "/predict", json={"target_hour": target.isoformat(), "log_prediction": False}
    ).json()
    assert any("ahead" in w for w in body["warnings"])


def test_missing_history_returns_503_not_a_fabricated_forecast(monkeypatch, client):
    """No history means no lags. Better a clear 503 than an invented number."""
    from src.serving import api as api_module

    monkeypatch.setattr(
        api_module,
        "load_recent_history",
        lambda cfg=None, **kw: HistoryResult(
            pd.DataFrame(columns=["station_short_name", "hour_ts", "departures"]), "none", None
        ),
    )
    response = client.post("/predict", json={"log_prediction": False})
    assert response.status_code == 503
    assert "history" in response.json()["detail"].lower()


def test_cache_fallback_is_surfaced_as_degraded(monkeypatch, client, hourly_departures):
    """Serving from the local cache is a degraded state and must be visible."""
    from src.serving import api as api_module

    monkeypatch.setattr(
        api_module,
        "load_recent_history",
        lambda cfg=None, **kw: HistoryResult(
            hourly_departures, "parquet_cache", pd.Timestamp(hourly_departures["hour_ts"].max())
        ),
    )
    assert client.get("/health").json()["status"] == "degraded"
    body = client.post("/predict", json={"log_prediction": False}).json()
    assert any("parquet cache" in w for w in body["warnings"])


def test_stale_history_is_flagged(monkeypatch, client, hourly_departures):
    """Old lag features make the forecast weaker than its backtest suggests."""
    from src.serving import api as api_module

    stale = hourly_departures.copy()
    shift = pd.Timestamp(datetime.now()) - pd.Timestamp(stale["hour_ts"].max()) - pd.Timedelta(days=30)
    stale["hour_ts"] = stale["hour_ts"] + shift.floor("h")

    monkeypatch.setattr(
        api_module,
        "load_recent_history",
        lambda cfg=None, **kw: HistoryResult(
            stale, "postgres", pd.Timestamp(stale["hour_ts"].max())
        ),
    )
    body = client.post("/predict", json={"log_prediction": False}).json()
    assert body["history_lag_hours"] > 3
    assert any("stale" in w for w in body["warnings"])
