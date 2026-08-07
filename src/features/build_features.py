"""THE shared feature builder. Used by training and by serving - no exceptions.

Train/serve skew is the classic way a forecasting system quietly rots, so there
is exactly one function that turns observed history into a model-ready matrix:
:func:`build_feature_panel`. ``train.py`` calls it; ``serving/api.py`` calls it;
``tests/test_features_parity.py`` asserts the two produce identical rows for the
same inputs.

Leakage rules enforced here
---------------------------
* Every lag is at least 1 hour, so the target hour never feeds its own features.
* Rolling statistics are computed on a ``shift(1)`` series - a plain
  ``rolling(24).mean()`` would include the current hour and leak the label.
* Nothing in this module is fit on data. Scalers/encoders live in the model
  layer where they are fit on the training split only.

A note on the "current fill ratio" feature
------------------------------------------
The brief lists ``num_bikes_available / capacity`` as a station feature. It is
deliberately **excluded**. Historical trip CSVs contain no inventory data, so
that feature would be entirely NaN across the training set and fully populated
at serving time - a guaranteed train/serve distribution skew that would make the
model behave differently in production than in backtest. It can be added once
the live GBFS loop has accumulated enough snapshot history to train on, at which
point it would be present on both sides. This is recorded in the README's
limitations section.
"""

from __future__ import annotations

import logging

import holidays
import numpy as np
import pandas as pd

from src.config import Config, load_config

logger = logging.getLogger(__name__)

TARGET_COLUMN = "departures"
ENTITY_KEY = "entity_id"
TIME_KEY = "hour_ts"

# Static entity attributes joined onto every row.
ENTITY_FEATURES = ["capacity", "n_stations", "lat", "lon"]
# Weather columns; NaN is tolerated (LightGBM handles missing values natively).
WEATHER_FEATURES = ["temperature_c", "precipitation_mm", "wind_speed_kmh"]
CALENDAR_FEATURES = [
    "hour",
    "day_of_week",
    "is_weekend",
    "is_holiday",
    "month",
    "season",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
]

# Annual-cycle features. `month` and `season` are coarse step functions - they
# say nothing about where inside a season a day falls, and they jump
# discontinuously at month boundaries. These describe the smooth annual cycle
# and the physical driver behind it.
#
# Daylight matters in its own right for cycling: demand collapses after dark
# well beyond what temperature alone explains, and the dark hour moves by ~3
# hours between June and December. It is derived from date and latitude by
# closed-form astronomy, so it needs no extra data source and cannot leak.
# Grouped so each can be enabled independently - they were measured separately
# and they do not behave the same way. See `seasonal_feature_names`.
SEASONAL_FEATURE_GROUPS = {
    "annual_cycle": ["day_of_year_sin", "day_of_year_cos", "week_of_year"],
    "daylight": ["daylight_hours", "is_daylight", "hours_from_solar_noon"],
}
# Every seasonal column this module can produce, enabled or not.
SEASONAL_FEATURES = [
    name for group in SEASONAL_FEATURE_GROUPS.values() for name in group
]

_SEASON_BY_MONTH = {
    12: 0, 1: 0, 2: 0,   # winter
    3: 1, 4: 1, 5: 1,    # spring
    6: 2, 7: 2, 8: 2,    # summer
    9: 3, 10: 3, 11: 3,  # autumn
}


def lag_feature_names(config: Config | None = None) -> list[str]:
    cfg = config or load_config()
    return [f"lag_{h}h" for h in cfg.get_path("features.lag_hours", [1, 2, 3, 24, 168])]


def rolling_feature_names(config: Config | None = None) -> list[str]:
    cfg = config or load_config()
    names: list[str] = []
    for window in cfg.get_path("features.rolling_windows_hours", [24, 168]):
        names.extend([f"roll_mean_{window}h", f"roll_std_{window}h"])
    return names


def seasonal_feature_names(config: Config | None = None) -> list[str]:
    """Seasonal columns that are actually fed to the model.

    Defaults to **none**, which is a measured decision rather than caution.
    Trained on 14 months, adding these made the model *worse* on a held-out
    fortnight (LightGBM, identical data and split):

        base (no seasonal)      MAE 20.034
        + 30-day rolling only   MAE 19.978   -0.28%
        + daylight only         MAE 20.795   +3.80%
        + annual cycle only     MAE 21.163   +5.64%
        + all seasonal          MAE 20.984   +4.74%

    They ranked among the most-used features while making predictions worse,
    which is the signature of memorisation rather than learning. With ~14 months
    each calendar date occurs once or twice, so `day_of_year` can key directly
    on "late June 2025" instead of learning an annual shape - and late June 2025
    is not late June 2026. Daylight fails the same way, being a deterministic
    function of day-of-year.

    The 30-day rolling mean, which tracks the *recent* level rather than
    position in the year, is the one piece that did not hurt; it stays on.

    Enable these once the training window spans several years, where an annual
    cycle can be estimated instead of memorised.
    """
    cfg = config or load_config()
    enabled = cfg.get_path("features.seasonal_feature_groups", []) or []
    names: list[str] = []
    for group in enabled:
        if group not in SEASONAL_FEATURE_GROUPS:
            logger.warning(
                "Unknown seasonal feature group %r; known groups: %s",
                group, sorted(SEASONAL_FEATURE_GROUPS),
            )
            continue
        names.extend(SEASONAL_FEATURE_GROUPS[group])
    return names


def feature_columns(config: Config | None = None) -> list[str]:
    """The exact ordered feature list the model consumes.

    Order is part of the contract: the serving path builds its matrix with this
    same call, so a config change moves both sides together.
    """
    cfg = config or load_config()
    return (
        CALENDAR_FEATURES
        + seasonal_feature_names(cfg)
        + lag_feature_names(cfg)
        + rolling_feature_names(cfg)
        + ENTITY_FEATURES
        + WEATHER_FEATURES
    )


def _holiday_dates(years: list[int], country: str) -> set:
    try:
        return set(holidays.country_holidays(country, years=years).keys())
    except Exception as exc:  # noqa: BLE001 - never let a holiday lookup break training
        logger.warning("Holiday calendar unavailable for %s (%s); is_holiday=0", country, exc)
        return set()


def solar_daylight_hours(day_of_year: np.ndarray, latitude: float) -> np.ndarray:
    """Hours between sunrise and sunset, from date and latitude.

    Solar-declination geometry, with sunrise/sunset taken at a solar altitude of
    -0.833 degrees rather than 0. That offset is the conventional correction for
    atmospheric refraction plus the sun's angular radius, and it is what
    almanacs mean by "sunrise". Omitting it understates NYC daylight by a
    consistent ~0.2 h, which checks against published values confirmed.

    The ``clip`` guards polar day/night, where the arccos argument leaves
    [-1, 1]. NYC never gets near that, but a silent NaN would propagate into
    every downstream feature.
    """
    declination = np.deg2rad(23.44) * np.sin(2 * np.pi * (day_of_year - 81) / 365.0)
    latitude_rad = np.deg2rad(latitude)
    altitude = np.deg2rad(-0.833)

    cos_hour_angle = (
        np.sin(altitude) - np.sin(latitude_rad) * np.sin(declination)
    ) / (np.cos(latitude_rad) * np.cos(declination))
    hour_angle = np.arccos(np.clip(cos_hour_angle, -1.0, 1.0))
    return 2.0 * np.rad2deg(hour_angle) / 15.0


def add_seasonal_features(frame: pd.DataFrame, config: Config | None = None) -> pd.DataFrame:
    """Smooth annual-cycle and daylight features.

    All are deterministic functions of the timestamp (plus a fixed latitude), so
    they are exactly as leakage-free as ``hour`` or ``day_of_week`` - no future
    information is involved.
    """
    cfg = config or load_config()
    out = frame.copy()
    timestamps = pd.to_datetime(out[TIME_KEY])

    day_of_year = timestamps.dt.dayofyear.to_numpy()
    # Cyclical so that 31 December and 1 January are adjacent, which `month`
    # and `season` both get wrong at the year boundary.
    out["day_of_year_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    out["day_of_year_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    out["week_of_year"] = (
        timestamps.dt.isocalendar().week.to_numpy().astype("int16")
    )

    latitude = float(cfg.get_path("weather.latitude", 40.73))
    daylight = solar_daylight_hours(day_of_year, latitude)
    out["daylight_hours"] = daylight

    # Each row is an hourly bucket, so it is represented by its midpoint. Using
    # the hour's start would call 05:00-06:00 "dark" on a day when the sun rises
    # at 05:23, even though most of that hour is light.
    hour_of_day = timestamps.dt.hour.to_numpy().astype(float) + 0.5
    solar_noon = _solar_noon_local_hour(timestamps, cfg)

    sunrise = solar_noon - daylight / 2.0
    sunset = solar_noon + daylight / 2.0
    out["is_daylight"] = (
        (hour_of_day >= sunrise) & (hour_of_day <= sunset)
    ).astype("int8")
    out["hours_from_solar_noon"] = np.abs(hour_of_day - solar_noon)
    return out


def _solar_noon_local_hour(timestamps: pd.Series, config: Config) -> np.ndarray:
    """Local clock hour of solar noon, accounting for DST and longitude.

    Assuming solar noon falls at 12:00 is wrong by nearly an hour in NYC, and
    wrong in a way that matters: it shifts sunrise and sunset by the same
    amount, which flips ``is_daylight`` exactly at dawn and dusk where demand is
    actually changing fastest. It reported 20:00 on 21 June as dark, when NYC
    sunset that day is about 20:31.

    Two corrections:

    * **Daylight saving.** Clocks move but the sun does not, so solar noon lands
      an hour later by the clock during EDT. Taken from the zone's real UTC
      offset rather than from month-based guesswork.
    * **Longitude.** NYC sits ~1 degree east of the 75W meridian its zone is
      built on, worth about 4 minutes.
    """
    longitude = float(config.get_path("weather.longitude", -73.99))
    timezone = config.get_path("project.timezone", "America/New_York")

    try:
        localised = pd.DatetimeIndex(timestamps).tz_localize(
            timezone, ambiguous=True, nonexistent="shift_forward"
        )
        offset_hours = (
            localised.map(lambda t: t.utcoffset().total_seconds() / 3600.0)
            .to_numpy()
            .astype(float)
        )
    except Exception as exc:  # noqa: BLE001 - a bad zone must not break features
        logger.warning(
            "Could not resolve UTC offset for %s (%s); assuming solar noon at 12:00",
            timezone, exc,
        )
        return np.full(len(timestamps), 12.0)

    # Solar noon (UTC) is 12:00 shifted by longitude at 15 deg/hour; converting
    # to local clock time adds the zone's offset.
    return 12.0 - longitude / 15.0 + offset_hours


def add_calendar_features(frame: pd.DataFrame, config: Config | None = None) -> pd.DataFrame:
    """Time-of-day / day-of-week / holiday features derived from ``hour_ts``."""
    cfg = config or load_config()
    out = frame.copy()
    timestamps = pd.to_datetime(out[TIME_KEY])

    out["hour"] = timestamps.dt.hour.astype("int16")
    out["day_of_week"] = timestamps.dt.dayofweek.astype("int16")
    out["is_weekend"] = (out["day_of_week"] >= 5).astype("int8")
    out["month"] = timestamps.dt.month.astype("int16")
    out["season"] = out["month"].map(_SEASON_BY_MONTH).astype("int16")

    years = sorted({int(y) for y in timestamps.dt.year.unique()})
    holiday_set = _holiday_dates(years, cfg.get_path("features.holiday_country", "US"))
    out["is_holiday"] = timestamps.dt.date.isin(holiday_set).astype("int8")

    # Cyclical encodings so the model sees 23:00 and 00:00 as adjacent.
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24.0)
    out["dow_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 7.0)
    return out


def add_lag_features(frame: pd.DataFrame, config: Config | None = None) -> pd.DataFrame:
    """Per-entity lags and rolling statistics.

    Assumes ``frame`` is a *complete* hourly panel sorted by (entity, hour):
    gaps would make a positional ``shift`` mean the wrong lag. Callers get that
    guarantee from :func:`complete_panel`.
    """
    cfg = config or load_config()
    out = frame.sort_values([ENTITY_KEY, TIME_KEY]).reset_index(drop=True)
    grouped = out.groupby(ENTITY_KEY, observed=True)[TARGET_COLUMN]

    for hours in cfg.get_path("features.lag_hours", [1, 2, 3, 24, 168]):
        out[f"lag_{hours}h"] = grouped.shift(hours)

    # shift(1) before rolling: without it the window includes the current hour
    # and the model would be trained on its own label.
    shifted = grouped.shift(1)
    for window in cfg.get_path("features.rolling_windows_hours", [24, 168]):
        roll = shifted.groupby(out[ENTITY_KEY], observed=True).rolling(
            window=window, min_periods=max(2, window // 4)
        )
        out[f"roll_mean_{window}h"] = roll.mean().reset_index(level=0, drop=True)
        out[f"roll_std_{window}h"] = roll.std().reset_index(level=0, drop=True)
    return out


def complete_panel(
    observations: pd.DataFrame,
    *,
    extra_hours: pd.DatetimeIndex | None = None,
    config: Config | None = None,
) -> pd.DataFrame:
    """Expand observations to a dense (entity x hour) grid.

    A missing (entity, hour) pair usually means *no departures were recorded*,
    which is a genuine zero rather than missing data - so short gaps are filled
    with 0.

    **Long gaps are not.** Blending the historical backfill with live proxy data
    leaves a hole between where the trip CSVs end and where live ingestion began
    (a 33-day hole on this stack). Filling that with zeros fabricates a month of
    "no demand": it trains the model on invented data, and if the hole lands in
    the validation window it makes the split entirely zeros, which showed up as
    a val_mae of 0.001 and a *better-looking* test MAE that would have been
    promoted over a genuinely better model. Runs of missing hours longer than
    ``features.max_gap_fill_hours`` are therefore left NaN and dropped from
    training instead.

    ``extra_hours`` appends rows whose target is unknown (NaN); the serving path
    uses this to request features for a future hour while still deriving its
    lags from observed history.
    """
    cfg = config or load_config()
    max_gap = int(cfg.get_path("features.max_gap_fill_hours", 24))
    if observations.empty:
        raise ValueError("Cannot build a feature panel from zero observations")

    frame = observations[[ENTITY_KEY, TIME_KEY, TARGET_COLUMN]].copy()
    frame[TIME_KEY] = pd.to_datetime(frame[TIME_KEY])

    entities = frame[ENTITY_KEY].unique()
    start, end = frame[TIME_KEY].min(), frame[TIME_KEY].max()
    if extra_hours is not None and len(extra_hours) > 0:
        end = max(end, pd.Timestamp(max(extra_hours)))
    full_hours = pd.date_range(start=start, end=end, freq="h")

    grid = pd.MultiIndex.from_product(
        [entities, full_hours], names=[ENTITY_KEY, TIME_KEY]
    ).to_frame(index=False)

    dense = grid.merge(frame, on=[ENTITY_KEY, TIME_KEY], how="left")

    dense = dense.sort_values([ENTITY_KEY, TIME_KEY]).reset_index(drop=True)

    # Hours beyond the observed window (the serving horizon) must stay NaN -
    # they are not yet known.
    observed_end = frame[TIME_KEY].max()
    within_observed = dense[TIME_KEY] <= observed_end
    missing = dense[TARGET_COLUMN].isna() & within_observed

    # Measure each run of consecutive missing hours per entity, and fill only
    # the short ones. A run boundary is either a change in the missing flag or a
    # change of entity.
    run_change = (missing != missing.shift(1)) | (
        dense[ENTITY_KEY] != dense[ENTITY_KEY].shift(1)
    )
    run_id = run_change.cumsum()
    run_length = run_id.map(run_id.value_counts())

    fillable = missing & (run_length <= max_gap)
    dense.loc[fillable, TARGET_COLUMN] = 0.0

    unfilled = int((missing & ~fillable).sum())
    if unfilled:
        logger.warning(
            "Left %d hours unfilled: they fall in gaps longer than %dh and are "
            "absent data, not observed zeros",
            unfilled, max_gap,
        )
    return dense


def build_feature_panel(
    observations: pd.DataFrame,
    entities: pd.DataFrame,
    weather: pd.DataFrame | None = None,
    config: Config | None = None,
    *,
    extra_hours: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Build the model-ready feature panel. **The single shared entry point.**

    Parameters
    ----------
    observations
        Long frame of observed history: ``entity_id``, ``hour_ts``, ``departures``.
        In training these are true counts from trip CSVs; in the live loop they
        are proxy counts derived from GBFS inventory deltas. The feature logic is
        identical either way - only the label's provenance differs.
    entities
        Static entity metadata: ``entity_id``, ``lat``, ``lon``, ``capacity``,
        ``n_stations``.
    weather
        Hourly weather. May be ``None`` or empty; weather columns then come
        through as NaN rather than failing the build.
    extra_hours
        Future hours to emit feature rows for, with an unknown (NaN) target.

    Returns
    -------
    Panel with ``entity_id``, ``hour_ts``, ``departures`` and every column in
    :func:`feature_columns`, sorted by (entity, hour).
    """
    cfg = config or load_config()

    panel = complete_panel(observations, extra_hours=extra_hours, config=cfg)
    panel = add_lag_features(panel, cfg)
    panel = add_calendar_features(panel, cfg)
    panel = add_seasonal_features(panel, cfg)

    # Select only the entity columns actually present; any that are absent are
    # filled with NaN below. Indexing the full list here would raise instead,
    # making the documented tolerance a lie.
    available = [c for c in ENTITY_FEATURES if c in entities.columns]
    missing = [c for c in ENTITY_FEATURES if c not in entities.columns]
    if missing:
        logger.warning("Entity metadata is missing %s; those features will be NaN", missing)
    entity_meta = entities.drop_duplicates(subset=ENTITY_KEY)[[ENTITY_KEY] + available]
    panel = panel.merge(entity_meta, on=ENTITY_KEY, how="left")

    if weather is not None and not weather.empty:
        weather_frame = weather.copy()
        weather_frame[TIME_KEY] = pd.to_datetime(weather_frame[TIME_KEY])
        weather_frame = weather_frame.drop_duplicates(subset=TIME_KEY)
        keep = [TIME_KEY] + [c for c in WEATHER_FEATURES if c in weather_frame.columns]
        panel = panel.merge(weather_frame[keep], on=TIME_KEY, how="left")

    for column in WEATHER_FEATURES:
        if column not in panel.columns:
            panel[column] = np.nan
    for column in ENTITY_FEATURES:
        if column not in panel.columns:
            panel[column] = np.nan

    ordered = [ENTITY_KEY, TIME_KEY, TARGET_COLUMN] + feature_columns(cfg)
    return panel[ordered].sort_values([ENTITY_KEY, TIME_KEY]).reset_index(drop=True)


def drop_warmup_rows(panel: pd.DataFrame, config: Config | None = None) -> pd.DataFrame:
    """Drop early rows whose core lags could not be computed.

    Keeping them would train the model on rows where the strongest feature
    (same-hour-last-week) is always missing, which is not a regime it will ever
    see in production.

    The cutoff is governed by ``features.warmup_hours`` rather than by the
    longest lag or rolling window. Those are no longer the same thing: adding a
    30-day seasonal rolling mean would otherwise discard the first 30 days of
    every training set, and a year-over-year lag would discard a whole year.
    A long window being partly unfilled early on is fine - LightGBM handles NaN
    natively - whereas throwing away real data is not.
    """
    cfg = config or load_config()
    warmup = cfg.get_path("features.warmup_hours")
    if warmup is None:
        warmup = max(cfg.get_path("features.lag_hours", [1, 2, 3, 24, 168]))
    cutoff = panel[TIME_KEY].min() + pd.Timedelta(hours=int(warmup))
    return panel[panel[TIME_KEY] >= cutoff].reset_index(drop=True)
