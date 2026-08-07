"""Seasonal and daylight features.

Added because the original feature set described the annual cycle only through
`month` and `season` - two coarse step functions - while the model was trained
on three spring/summer months, so `season` was nearly constant and it had never
seen winter at all.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import (
    SEASONAL_FEATURES,
    TARGET_COLUMN,
    TIME_KEY,
    add_seasonal_features,
    build_feature_panel,
    drop_warmup_rows,
    feature_columns,
    solar_daylight_hours,
)
from src.features.station_clusters import aggregate_to_entities, fit_entity_mapping

NYC_LATITUDE = 40.73


@pytest.fixture
def mapping(station_info, config):
    return fit_entity_mapping(station_info, config)


# --------------------------------------------------------------------------
# Daylight physics
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "label,day_of_year,published_hours",
    [
        ("June solstice", 172, 15.05),
        ("December solstice", 355, 9.25),
        ("March equinox", 79, 12.12),
        ("September equinox", 265, 12.15),
    ],
)
def test_daylight_matches_published_nyc_values(label, day_of_year, published_hours):
    """Within 15 minutes of almanac values for NYC."""
    computed = float(solar_daylight_hours(np.array([day_of_year]), NYC_LATITUDE)[0])
    assert computed == pytest.approx(published_hours, abs=0.25), label


def test_daylight_peaks_at_the_summer_solstice():
    days = np.arange(1, 366)
    daylight = solar_daylight_hours(days, NYC_LATITUDE)
    # Solstices are day 172 and 355; allow a few days for the 365-day approximation.
    assert abs(int(days[daylight.argmax()]) - 172) <= 3
    assert abs(int(days[daylight.argmin()]) - 355) <= 3


def test_daylight_swing_is_about_six_hours_in_nyc():
    daylight = solar_daylight_hours(np.arange(1, 366), NYC_LATITUDE)
    assert 5.0 < (daylight.max() - daylight.min()) < 7.0


def test_sunrise_and_sunset_match_published_nyc_times():
    """Solar noon must account for DST and longitude, not be assumed at 12:00."""
    from src.config import load_config
    from src.features.build_features import _solar_noon_local_hour

    cfg = load_config()
    stamps = pd.Series(pd.to_datetime(["2026-06-21 12:00", "2026-12-21 12:00"]))
    solar_noon = _solar_noon_local_hour(stamps, cfg)
    daylight = solar_daylight_hours(
        pd.DatetimeIndex(stamps).dayofyear.to_numpy(), NYC_LATITUDE
    )

    # NYC: 21 Jun sunrise 05:25 / sunset 20:31; 21 Dec sunrise 07:16 / sunset 16:32.
    for index, (sunrise_actual, sunset_actual) in enumerate(
        [(5.42, 20.52), (7.27, 16.53)]
    ):
        sunrise = solar_noon[index] - daylight[index] / 2
        sunset = solar_noon[index] + daylight[index] / 2
        assert sunrise == pytest.approx(sunrise_actual, abs=0.15)
        assert sunset == pytest.approx(sunset_actual, abs=0.15)


def test_daylight_never_returns_nan_even_in_the_arctic():
    """Polar day/night pushes the arccos argument out of range; clip must hold."""
    for latitude in (0.0, 40.73, 68.0, 89.9, -89.9):
        values = solar_daylight_hours(np.arange(1, 366), latitude)
        assert np.isfinite(values).all(), latitude
        assert ((values >= 0) & (values <= 24)).all(), latitude


# --------------------------------------------------------------------------
# Seasonal encodings
# --------------------------------------------------------------------------

def test_day_of_year_encoding_is_continuous_across_new_year():
    """31 December and 1 January are one day apart, not 364.

    This is exactly what `month` and `season` get wrong at the year boundary.
    """
    frame = pd.DataFrame(
        {TIME_KEY: pd.to_datetime(["2026-12-31 12:00", "2027-01-01 12:00"])}
    )
    out = add_seasonal_features(frame)
    distance = np.hypot(
        out["day_of_year_sin"].iloc[0] - out["day_of_year_sin"].iloc[1],
        out["day_of_year_cos"].iloc[0] - out["day_of_year_cos"].iloc[1],
    )
    assert distance < 0.05

    # Sanity: mid-year is genuinely far away on the same circle.
    july = add_seasonal_features(pd.DataFrame({TIME_KEY: pd.to_datetime(["2026-07-01 12:00"])}))
    far = np.hypot(
        out["day_of_year_sin"].iloc[0] - july["day_of_year_sin"].iloc[0],
        out["day_of_year_cos"].iloc[0] - july["day_of_year_cos"].iloc[0],
    )
    assert far > 1.5


def test_is_daylight_tracks_the_season():
    """The same clock hour is light in June and dark in December.

    Requires the solar-noon correction: assuming noon at 12:00 shifts sunrise
    and sunset by nearly an hour in NYC and gets the 19:00 and 06:00 cases wrong.
    """
    frame = pd.DataFrame(
        {
            TIME_KEY: pd.to_datetime(
                [
                    "2026-06-21 19:00",  # summer evening: sunset ~20:29
                    "2026-12-21 19:00",  # winter evening: sunset ~16:33
                    "2026-06-21 06:00",  # summer dawn: sunrise ~05:23
                    "2026-12-21 06:00",  # winter dawn: sunrise ~07:18
                    "2026-06-21 13:00",  # midday: always light
                    "2026-12-21 13:00",
                ]
            )
        }
    )
    out = add_seasonal_features(frame)
    assert list(out["is_daylight"]) == [1, 0, 1, 0, 1, 1]


def test_seasonal_features_are_deterministic_from_the_timestamp():
    """No data goes in, so nothing can leak out."""
    timestamps = pd.to_datetime(["2026-03-15 08:00", "2026-09-15 18:00"])
    first = add_seasonal_features(pd.DataFrame({TIME_KEY: timestamps}))
    # Same timestamps, arbitrary extra columns - the features must not move.
    second = add_seasonal_features(
        pd.DataFrame({TIME_KEY: timestamps, TARGET_COLUMN: [999.0, -5.0]})
    )
    pd.testing.assert_frame_equal(first[SEASONAL_FEATURES], second[SEASONAL_FEATURES])


def test_seasonal_features_are_in_the_model_contract(config):
    columns = feature_columns(config)
    for name in SEASONAL_FEATURES:
        assert name in columns


def test_seasonal_features_reach_the_panel(hourly_departures, mapping, weather, config):
    observations = aggregate_to_entities(hourly_departures, mapping)
    panel = build_feature_panel(observations, mapping.entities, weather, config)
    for name in SEASONAL_FEATURES:
        assert name in panel.columns
        assert panel[name].notna().all(), f"{name} should never be NaN"


# --------------------------------------------------------------------------
# Warmup decoupling
# --------------------------------------------------------------------------

def test_warmup_is_not_dictated_by_the_longest_rolling_window(
    hourly_departures, mapping, weather, config
):
    """A 30-day seasonal window must not cost 30 days of training data.

    Before warmup was decoupled, adding roll_mean_720h would have silently
    discarded the first month of every training set.
    """
    observations = aggregate_to_entities(hourly_departures, mapping)
    panel = build_feature_panel(observations, mapping.entities, weather, config)
    trimmed = drop_warmup_rows(panel, config)

    warmup = int(config.get_path("features.warmup_hours"))
    longest_window = max(config.get_path("features.rolling_windows_hours"))
    assert longest_window > warmup, "fixture should exercise the decoupling"

    expected_start = panel[TIME_KEY].min() + pd.Timedelta(hours=warmup)
    assert trimmed[TIME_KEY].min() == expected_start
    # Rows are kept even though the 720h window is only partly filled there.
    assert trimmed["roll_mean_720h"].isna().any()
    assert not trimmed["lag_168h"].isna().all()
