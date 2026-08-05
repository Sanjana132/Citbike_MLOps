"""Monitoring: breach rules, drift detection, and the prediction/actual join.

These decide when the system retrains itself, so the thresholds are tested
directly rather than only through the DAG.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift import compute_drift, population_stability_index
from src.monitoring.metrics import _judge_breach, join_predictions_to_realised

# --------------------------------------------------------------------------
# Breach rules
# --------------------------------------------------------------------------

def test_absolute_threshold_triggers_retraining(config):
    breached, reason = _judge_breach(rolling_mae=99.0, config=config, reference_mae=None)
    assert breached
    assert "absolute threshold" in reason


def test_within_threshold_does_not_trigger(config):
    breached, reason = _judge_breach(rolling_mae=0.5, config=config, reference_mae=1.0)
    assert not breached
    assert reason is None


def test_degradation_against_training_mae_triggers(config):
    """The meaningful rule: worse than the model was at training time."""
    ratio = float(config.get_path("monitoring.mae_degradation_ratio"))
    reference = 1.0
    breached, reason = _judge_breach(
        rolling_mae=reference * ratio + 0.01, config=config, reference_mae=reference
    )
    assert breached
    assert "training holdout" in reason


def test_slight_degradation_does_not_trigger(config):
    ratio = float(config.get_path("monitoring.mae_degradation_ratio"))
    breached, _ = _judge_breach(
        rolling_mae=1.0 * ratio - 0.01, config=config, reference_mae=1.0
    )
    assert not breached


def test_missing_reference_falls_back_to_the_absolute_rule(config):
    """With no baseline, only the absolute ceiling can fire."""
    threshold = float(config.get_path("monitoring.mae_threshold"))
    assert not _judge_breach(threshold - 0.1, config, None)[0]
    assert _judge_breach(threshold + 0.1, config, None)[0]


# --------------------------------------------------------------------------
# Prediction / actual join
# --------------------------------------------------------------------------

@pytest.fixture
def mapping(station_info, config):
    from src.features.station_clusters import fit_entity_mapping

    return fit_entity_mapping(station_info, config)


def test_join_matches_predictions_to_realised_demand(mapping, hourly_departures):
    hours = sorted(hourly_departures["hour_ts"].unique())[:3]
    entities = mapping.entities["entity_id"].tolist()[:2]
    predictions = pd.DataFrame(
        [
            {"entity_id": entity, "target_hour_ts": hour, "predicted": 5.0}
            for entity in entities
            for hour in hours
        ]
    )

    joined = join_predictions_to_realised(predictions, hourly_departures, mapping)
    assert len(joined) == len(entities) * len(hours)
    assert {"predicted", "actual"}.issubset(joined.columns)
    assert joined["actual"].notna().all()


def test_join_drops_predictions_without_realised_demand(mapping, hourly_departures):
    """An hour with no observed demand yet must not be scored as a miss."""
    future = pd.Timestamp(hourly_departures["hour_ts"].max()) + pd.Timedelta(days=5)
    predictions = pd.DataFrame(
        [{"entity_id": mapping.entities["entity_id"].iloc[0], "target_hour_ts": future, "predicted": 5.0}]
    )
    assert join_predictions_to_realised(predictions, hourly_departures, mapping).empty


def test_join_keeps_only_the_most_recent_prediction_per_hour(mapping, hourly_departures):
    """Re-requests and model reloads produce duplicates; the last one was served."""
    entity = mapping.entities["entity_id"].iloc[0]
    hour = pd.Timestamp(hourly_departures["hour_ts"].min())
    predictions = pd.DataFrame(
        [
            {"entity_id": entity, "target_hour_ts": hour, "predicted": 1.0,
             "predicted_at": pd.Timestamp("2026-08-01T10:00:00")},
            {"entity_id": entity, "target_hour_ts": hour, "predicted": 9.0,
             "predicted_at": pd.Timestamp("2026-08-01T11:00:00")},
        ]
    )
    joined = join_predictions_to_realised(predictions, hourly_departures, mapping)
    assert len(joined) == 1
    assert joined["predicted"].iloc[0] == 9.0


def test_join_handles_empty_inputs(mapping, hourly_departures):
    empty = pd.DataFrame(columns=["entity_id", "target_hour_ts", "predicted"])
    assert join_predictions_to_realised(empty, hourly_departures, mapping).empty
    assert join_predictions_to_realised(empty, empty, mapping).empty


# --------------------------------------------------------------------------
# Drift
# --------------------------------------------------------------------------

@pytest.fixture
def drift_frames():
    rng = np.random.default_rng(7)
    n = 700
    reference = pd.DataFrame(
        {
            "temperature_c": rng.normal(18, 4, n),
            "lag_1h": rng.poisson(30, n).astype(float),
            "hour": rng.integers(0, 24, n).astype(float),
        }
    )
    stable = pd.DataFrame(
        {
            "temperature_c": rng.normal(18, 4, n),
            "lag_1h": rng.poisson(30, n).astype(float),
            "hour": rng.integers(0, 24, n).astype(float),
        }
    )
    shifted = pd.DataFrame(
        {
            "temperature_c": rng.normal(32, 4, n),
            "lag_1h": rng.poisson(80, n).astype(float),
            "hour": rng.integers(0, 24, n).astype(float),
        }
    )
    return reference, stable, shifted


def test_stable_distributions_report_no_drift(drift_frames, config):
    reference, stable, _ = drift_frames
    report = compute_drift(reference, stable, config, save_html=False)
    assert report.evaluated
    assert not report.dataset_drift
    assert report.drift_share == 0.0


def test_shifted_distributions_report_drift(drift_frames, config):
    reference, _, shifted = drift_frames
    report = compute_drift(reference, shifted, config, save_html=False)
    assert report.evaluated
    assert report.dataset_drift
    assert report.drift_share > float(config.get_path("monitoring.drift_share_threshold"))
    assert "temperature_c" in report.drifted_features


def test_too_little_current_data_is_not_evaluated(drift_frames, config):
    """Ten rows must not be allowed to trigger an expensive retrain."""
    reference, stable, _ = drift_frames
    report = compute_drift(reference, stable.head(10), config, save_html=False)
    assert not report.evaluated
    assert "insufficient" in report.reason


def test_empty_frames_are_not_evaluated(config):
    report = compute_drift(pd.DataFrame(), pd.DataFrame(), config, save_html=False)
    assert not report.evaluated


def test_psi_separates_stable_from_shifted(drift_frames):
    reference, stable, shifted = drift_frames
    assert population_stability_index(reference["temperature_c"], stable["temperature_c"]) < 0.1
    assert population_stability_index(reference["temperature_c"], shifted["temperature_c"]) > 0.25


def test_psi_handles_degenerate_input():
    """Constant or tiny series must return 0, not raise or emit NaN."""
    constant = pd.Series([5.0] * 100)
    assert population_stability_index(constant, constant) == 0.0
    assert population_stability_index(pd.Series([1.0]), pd.Series([2.0])) == 0.0


# --------------------------------------------------------------------------
# Retrain cooldown
# --------------------------------------------------------------------------

def _metrics_history(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["computed_at", "breached", "breach_reason"])


def test_cooldown_blocks_a_second_retrain_too_soon(monkeypatch, config):
    """Drift persists until a new model serves; without this the queue never drains."""
    from src.monitoring import pipeline
    from src.storage import db

    recent = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(hours=1)
    monkeypatch.setattr(
        db, "read_model_metrics",
        lambda limit=500, engine=None: _metrics_history(
            [{"computed_at": recent, "breached": True, "breach_reason": "drift: 80%"}]
        ),
    )
    remaining = pipeline._cooldown_remaining_hours(config)
    cooldown = float(config.get_path("monitoring.retrain_cooldown_hours"))
    assert remaining is not None
    assert remaining == pytest.approx(cooldown - 1.0, abs=0.2)


def test_cooldown_expires(monkeypatch, config):
    from src.monitoring import pipeline
    from src.storage import db

    cooldown = float(config.get_path("monitoring.retrain_cooldown_hours"))
    old = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(hours=cooldown + 2)
    monkeypatch.setattr(
        db, "read_model_metrics",
        lambda limit=500, engine=None: _metrics_history(
            [{"computed_at": old, "breached": True, "breach_reason": "drift: 80%"}]
        ),
    )
    assert pipeline._cooldown_remaining_hours(config) == 0.0


def test_suppressed_cycles_do_not_extend_their_own_cooldown(monkeypatch, config):
    """A cycle blocked by the cooldown must not reset the clock."""
    from src.monitoring import pipeline
    from src.storage import db

    recent = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(minutes=10)
    monkeypatch.setattr(
        db, "read_model_metrics",
        lambda limit=500, engine=None: _metrics_history(
            [{"computed_at": recent, "breached": False,
              "breach_reason": "drift: 80%; [suppressed] cooldown remaining"}]
        ),
    )
    assert pipeline._cooldown_remaining_hours(config) is None


def test_no_history_means_no_cooldown(monkeypatch, config):
    """A first-ever retrain must never be blocked."""
    from src.monitoring import pipeline
    from src.storage import db

    monkeypatch.setattr(db, "read_model_metrics", lambda limit=500, engine=None: pd.DataFrame())
    assert pipeline._cooldown_remaining_hours(config) is None


def test_cooldown_fails_open_on_a_database_error(monkeypatch, config):
    """Never let a monitoring lookup failure block retraining entirely."""
    from src.monitoring import pipeline
    from src.storage import db

    def boom(limit=500, engine=None):
        raise RuntimeError("database unreachable")

    monkeypatch.setattr(db, "read_model_metrics", boom)
    assert pipeline._cooldown_remaining_hours(config) is None
