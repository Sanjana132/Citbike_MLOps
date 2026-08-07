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

    # Serving keeps only a trailing window. Its size is taken from the serving
    # layer itself rather than hardcoded, because it is derived from the feature
    # config: adding a 30-day rolling mean widened it from 14 to 60 days, and a
    # hardcoded 21 days silently stopped covering the longest feature. That is a
    # real contract, so the test tracks it instead of guessing.
    from src.serving.history import _minimum_history_hours

    window_hours = _minimum_history_hours(config)
    cutoff = observations[TIME_KEY].max() - pd.Timedelta(hours=window_hours)
    recent = observations[observations[TIME_KEY] > cutoff]
    serving_panel = build_feature_panel(recent, mapping.entities, weather, config)

    # Compare only rows far enough into the serving window that every lag and
    # rolling window is fully populated - before that, serving genuinely has
    # less history and the two are not expected to agree.
    lookback = max(
        max(config.get_path("features.lag_hours")),
        max(config.get_path("features.rolling_windows_hours")),
    )
    comparable_from = recent[TIME_KEY].min() + pd.Timedelta(hours=lookback)
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


# --------------------------------------------------------------------------
# Gap handling: absent data must not become fabricated zeros
# --------------------------------------------------------------------------

def test_long_gaps_are_not_filled_with_zeros(mapping, weather, config):
    """A multi-week hole is absent data, not weeks of zero demand.

    Blending the historical backfill with live proxy data leaves exactly such a
    hole. Filling it with zeros trains the model on invented data and, when the
    hole lands in the validation window, produces a meaningless near-zero
    val_mae alongside a flattering test MAE.
    """
    entity = mapping.entities["entity_id"].iloc[0]
    early = pd.date_range("2026-04-01", periods=200, freq="h")
    late = pd.date_range("2026-06-01", periods=200, freq="h")  # ~5 weeks later
    observations = pd.DataFrame(
        {
            "entity_id": entity,
            TIME_KEY: early.append(late),
            TARGET_COLUMN: 5.0,
        }
    )

    panel = build_feature_panel(observations, mapping.entities, weather, config)
    gap = panel[
        (panel[TIME_KEY] > early.max()) & (panel[TIME_KEY] < late.min())
    ]
    assert not gap.empty, "fixture should span a gap"
    assert gap[TARGET_COLUMN].isna().all(), "long gap must stay NaN, not become 0"

    # Observed hours are untouched.
    observed = panel[panel[TIME_KEY].isin(early)]
    assert (observed[TARGET_COLUMN] == 5.0).all()


def test_short_gaps_are_still_filled_with_zeros(mapping, weather, config):
    """A quiet overnight hour with no rides really is zero demand."""
    entity = mapping.entities["entity_id"].iloc[0]
    hours = pd.date_range("2026-04-01", periods=300, freq="h")
    kept = hours.delete(150)  # a single missing hour
    observations = pd.DataFrame(
        {"entity_id": entity, TIME_KEY: kept, TARGET_COLUMN: 5.0}
    )

    panel = build_feature_panel(observations, mapping.entities, weather, config)
    filled = panel[panel[TIME_KEY] == hours[150]]
    assert len(filled) == 1
    assert filled[TARGET_COLUMN].iloc[0] == 0.0


def test_gap_rows_are_dropped_from_training(mapping, weather, config):
    """Unfilled gap rows must not reach the model as NaN targets."""
    from src.features.station_clusters import aggregate_to_entities  # noqa: F401

    entity = mapping.entities["entity_id"].iloc[0]
    early = pd.date_range("2026-04-01", periods=800, freq="h")
    late = pd.date_range("2026-06-15", periods=800, freq="h")
    observations = pd.DataFrame(
        {"entity_id": entity, TIME_KEY: early.append(late), TARGET_COLUMN: 5.0}
    )
    panel = build_feature_panel(observations, mapping.entities, weather, config)
    assert panel[TARGET_COLUMN].isna().any(), "fixture should produce gap rows"

    trainable = panel[panel[TARGET_COLUMN].notna()]
    assert not trainable[TARGET_COLUMN].isna().any()
