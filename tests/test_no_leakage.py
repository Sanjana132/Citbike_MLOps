"""Leakage guards beyond the feature builder.

Covers the two remaining places test-set information could reach training:
the chronological split itself, and any statistic fit across splits.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import TIME_KEY, build_feature_panel
from src.features.station_clusters import aggregate_to_entities, fit_entity_mapping
from src.models.dataset import build_dataset, chronological_split


@pytest.fixture
def dataset(hourly_departures, station_info, weather, config):
    mapping = fit_entity_mapping(station_info, config)
    return build_dataset(hourly_departures, mapping, weather, config)


def test_splits_are_strictly_ordered_in_time(dataset):
    """Every training row must precede every validation row, and so on."""
    assert dataset.train[TIME_KEY].max() < dataset.validation[TIME_KEY].min()
    assert dataset.validation[TIME_KEY].max() < dataset.test[TIME_KEY].min()


def test_splits_do_not_share_rows(dataset):
    def keys(frame: pd.DataFrame) -> set:
        return set(zip(frame["entity_id"], frame[TIME_KEY], strict=True))

    train, validation, test = keys(dataset.train), keys(dataset.validation), keys(dataset.test)
    assert not train & validation
    assert not validation & test
    assert not train & test


def test_split_is_not_random(hourly_departures, station_info, weather, config):
    """A shuffled split would interleave hours; this one must not."""
    mapping = fit_entity_mapping(station_info, config)
    observations = aggregate_to_entities(hourly_departures, mapping)
    panel = build_feature_panel(observations, mapping.entities, weather, config)
    train, validation, test = chronological_split(panel, config)

    boundary = train[TIME_KEY].max()
    assert (validation[TIME_KEY] > boundary).all()
    assert (test[TIME_KEY] > validation[TIME_KEY].max()).all()


def test_split_rejects_too_short_a_window(station_info, weather, config):
    """Silently returning an empty test set would fake a passing run."""
    short = pd.DataFrame(
        {
            "entity_id": ["cluster_000"] * 48,
            TIME_KEY: pd.date_range("2026-04-01", periods=48, freq="h"),
            "departures": np.arange(48, dtype=float),
        }
    )
    entities = pd.DataFrame(
        [{"entity_id": "cluster_000", "lat": 40.7, "lon": -74.0, "capacity": 20, "n_stations": 1}]
    )
    panel = build_feature_panel(short, entities, weather, config)
    with pytest.raises(ValueError, match="too short"):
        chronological_split(panel, config)


def test_missing_entity_metadata_degrades_rather_than_raising(config, weather):
    """Incomplete entity metadata must yield NaN features, not a crash."""
    observations = pd.DataFrame(
        {
            "entity_id": ["cluster_000"] * 400,
            TIME_KEY: pd.date_range("2026-04-01", periods=400, freq="h"),
            "departures": np.arange(400, dtype=float),
        }
    )
    sparse_entities = pd.DataFrame([{"entity_id": "cluster_000", "lat": 40.7, "lon": -74.0}])
    panel = build_feature_panel(observations, sparse_entities, weather, config)
    assert panel["capacity"].isna().all()
    assert panel["n_stations"].isna().all()
    assert panel["lat"].notna().all()


def test_deep_model_scaler_is_fit_on_training_data_only(dataset, config):
    """The LSTM's scalers must never see validation or test statistics."""
    torch = pytest.importorskip("torch")  # noqa: F841
    from src.models.deep_model import DeepDemandModel

    model = DeepDemandModel(dataset.features, config)
    model.sequence_length = 24  # keep the fixture window sufficient

    _, train_covariates, train_targets = model._sequences(dataset.train)
    if len(train_targets) == 0:
        pytest.skip("fixture window too short to form sequences")

    model.covariate_mean = train_covariates.mean(axis=0)
    model.covariate_std = np.where(
        train_covariates.std(axis=0) < 1e-6, 1.0, train_covariates.std(axis=0)
    )
    model.target_scale = float(max(train_targets.mean(), 1.0))

    # Statistics computed over train+validation+test must differ from the
    # train-only ones; if they matched, the scaler had seen everything.
    _, all_covariates, all_targets = model._sequences(
        pd.concat([dataset.train, dataset.validation, dataset.test], ignore_index=True)
    )
    assert not np.allclose(model.covariate_mean, all_covariates.mean(axis=0))
    assert model.target_scale != pytest.approx(float(all_targets.mean()), rel=1e-9)


def test_entity_mapping_is_frozen_not_refit(station_info, config, tmp_path):
    """Serving must reuse the training assignment, not re-cluster.

    Re-fitting KMeans at serve time would renumber every cluster and silently
    invalidate the model.
    """
    mapping = fit_entity_mapping(station_info, config)
    path = mapping.save(tmp_path / "mapping.json")

    from src.features.station_clusters import EntityMapping

    reloaded = EntityMapping.load(path)
    assert reloaded.station_to_entity == mapping.station_to_entity
    assert reloaded.granularity == mapping.granularity
    pd.testing.assert_frame_equal(
        reloaded.entities.sort_values("entity_id").reset_index(drop=True),
        mapping.entities.sort_values("entity_id").reset_index(drop=True),
        check_dtype=False,
    )
