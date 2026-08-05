"""Train/serve feature parity and leakage guards.

These are the load-bearing tests of the whole project. If features drift apart
between training and serving, every metric downstream becomes a fiction - and
the failure is silent, which is what makes it dangerous.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import (
    TARGET_COLUMN,
    TIME_KEY,
    build_feature_panel,
    complete_panel,
    feature_columns,
)
from src.features.station_clusters import aggregate_to_entities, fit_entity_mapping


@pytest.fixture
def mapping(station_info, config):
    return fit_entity_mapping(station_info, config)


@pytest.fixture
def observations(hourly_departures, mapping):
    return aggregate_to_entities(hourly_departures, mapping)


def test_serving_reproduces_training_features_exactly(observations, mapping, weather, config):
    """THE parity test.

    Training builds features over the full history at once. Serving sees only a
    trailing window and asks for one future hour. For the hours both paths
    cover, every feature value must be bit-identical - otherwise the model is
    scored on one distribution and served another.
    """
    training_panel = build_feature_panel(observations, mapping.entities, weather, config)

    # Serving path: only the trailing 3 weeks of history are available.
    cutoff = observations[TIME_KEY].max() - pd.Timedelta(days=21)
    recent = observations[observations[TIME_KEY] > cutoff]
    serving_panel = build_feature_panel(recent, mapping.entities, weather, config)

    # Compare only rows far enough into the serving window that its longest lag
    # (168h) is fully populated - before that, serving genuinely has less history.
    comparable_from = recent[TIME_KEY].min() + pd.Timedelta(hours=168)
    features = feature_columns(config)

    left = (
        training_panel[training_panel[TIME_KEY] >= comparable_from]
        .set_index(["entity_id", TIME_KEY])[features]
        .sort_index()
    )
    right = (
        serving_panel[serving_panel[TIME_KEY] >= comparable_from]
        .set_index(["entity_id", TIME_KEY])[features]
        .sort_index()
    )

    assert not left.empty, "No overlapping rows to compare - fixture window too short"
    assert list(left.index) == list(right.index)
    pd.testing.assert_frame_equal(left, right, check_exact=False, rtol=1e-12)


def test_feature_columns_are_stable_and_ordered(config):
    """Column order is part of the model contract; both paths call this function."""
    first = feature_columns(config)
    second = feature_columns(config)
    assert first == second
    assert len(first) == len(set(first)), "Duplicate feature names"


def test_future_hour_gets_features_but_no_label(observations, mapping, weather, config):
    """Serving must be able to request an unseen hour and still get full lags."""
    last_hour = observations[TIME_KEY].max()
    target_hour = last_hour + pd.Timedelta(hours=1)

    panel = build_feature_panel(
        observations,
        mapping.entities,
        weather,
        config,
        extra_hours=pd.DatetimeIndex([target_hour]),
    )
    future = panel[panel[TIME_KEY] == target_hour]

    assert not future.empty
    assert future[TARGET_COLUMN].isna().all(), "Future target must be unknown"
    # lag_1h at t+1 is the observed value at t, which is known.
    assert future["lag_1h"].notna().all()
    assert future["lag_24h"].notna().all()


def test_no_target_leakage_into_lag_features(observations, mapping, weather, config):
    """A lag must equal the target shifted, never the target itself."""
    panel = build_feature_panel(observations, mapping.entities, weather, config)
    entity = panel["entity_id"].iloc[0]
    single = panel[panel["entity_id"] == entity].sort_values(TIME_KEY).reset_index(drop=True)

    expected_lag_1 = single[TARGET_COLUMN].shift(1)
    pd.testing.assert_series_equal(
        single["lag_1h"], expected_lag_1, check_names=False, check_dtype=False
    )

    # A lag identical to the target would mean the label leaked in.
    overlap = single["lag_1h"].notna()
    assert not np.allclose(
        single.loc[overlap, "lag_1h"], single.loc[overlap, TARGET_COLUMN]
    )


def test_rolling_features_exclude_the_current_hour(observations, mapping, weather, config):
    """The classic leak: rolling(24).mean() includes t and leaks the label."""
    panel = build_feature_panel(observations, mapping.entities, weather, config)
    entity = panel["entity_id"].iloc[0]
    single = panel[panel["entity_id"] == entity].sort_values(TIME_KEY).reset_index(drop=True)

    expected = single[TARGET_COLUMN].shift(1).rolling(24, min_periods=6).mean()
    pd.testing.assert_series_equal(
        single["roll_mean_24h"], expected, check_names=False, check_dtype=False
    )

    # And prove the leaky version would have been different.
    leaky = single[TARGET_COLUMN].rolling(24, min_periods=6).mean()
    comparable = single["roll_mean_24h"].notna() & leaky.notna()
    assert not np.allclose(single.loc[comparable, "roll_mean_24h"], leaky[comparable])


def test_missing_hours_become_zero_not_nan(observations, mapping, weather, config):
    """A station with no rides in an hour had zero departures, not unknown ones."""
    sparse = observations[observations[TIME_KEY].dt.hour != 3]
    panel = build_feature_panel(sparse, mapping.entities, weather, config)
    hour_three = panel[panel[TIME_KEY].dt.hour == 3]

    assert not hour_three.empty
    assert (hour_three[TARGET_COLUMN] == 0).all()


def test_build_survives_missing_weather(observations, mapping, config):
    """A weather outage must degrade the model, not crash the training run."""
    panel = build_feature_panel(observations, mapping.entities, None, config)
    for column in ("temperature_c", "precipitation_mm", "wind_speed_kmh"):
        assert column in panel.columns
        assert panel[column].isna().all()
    assert set(feature_columns(config)).issubset(panel.columns)


def test_complete_panel_rejects_empty_input():
    with pytest.raises(ValueError, match="zero observations"):
        complete_panel(pd.DataFrame(columns=["entity_id", TIME_KEY, TARGET_COLUMN]))
