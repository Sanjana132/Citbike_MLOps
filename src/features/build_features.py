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


def feature_columns(config: Config | None = None) -> list[str]:
    """The exact ordered feature list the model consumes.

    Order is part of the contract: the serving path builds its matrix with this
    same call, so a config change moves both sides together.
    """
    cfg = config or load_config()
    return (
        CALENDAR_FEATURES
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
    """Drop early rows whose longest lag could not be computed.

    Keeping them would train the model on rows where the strongest feature
    (same-hour-last-week) is always missing, which is not a regime it will ever
    see in production.
    """
    cfg = config or load_config()
    longest_lag = max(cfg.get_path("features.lag_hours", [1, 2, 3, 24, 168]))
    cutoff = panel[TIME_KEY].min() + pd.Timedelta(hours=longest_lag)
    return panel[panel[TIME_KEY] >= cutoff].reset_index(drop=True)
