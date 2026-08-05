"""Assembles the training dataset and performs the chronological split.

Kept separate from ``train.py`` so that the offline (trip-CSV) path and the
live retraining (Postgres) path can share identical splitting and evaluation
semantics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import Config, load_config
from src.features.build_features import (
    TARGET_COLUMN,
    TIME_KEY,
    build_feature_panel,
    drop_warmup_rows,
    feature_columns,
)
from src.features.station_clusters import EntityMapping, aggregate_to_entities

logger = logging.getLogger(__name__)


@dataclass
class Dataset:
    """A chronologically split, model-ready dataset."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    features: list[str]
    mapping: EntityMapping

    @property
    def target(self) -> str:
        return TARGET_COLUMN

    def xy(self, split: str) -> tuple[pd.DataFrame, pd.Series]:
        frame = getattr(self, split)
        return frame[self.features], frame[TARGET_COLUMN]

    def summary(self) -> dict[str, object]:
        return {
            "n_train": len(self.train),
            "n_validation": len(self.validation),
            "n_test": len(self.test),
            "n_features": len(self.features),
            "n_entities": int(self.train["entity_id"].nunique()),
            "train_start": str(self.train[TIME_KEY].min()),
            "train_end": str(self.train[TIME_KEY].max()),
            "test_start": str(self.test[TIME_KEY].min()),
            "test_end": str(self.test[TIME_KEY].max()),
            "granularity": self.mapping.granularity,
        }


def chronological_split(
    panel: pd.DataFrame, config: Config | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by time, never at random.

    Layout: ``[ ---- train ---- | validation | test ]`` with validation and test
    taken from the tail. A random split would let the model see hour ``t+1``
    while predicting hour ``t`` and produce meaningless, wildly optimistic scores.
    """
    cfg = config or load_config()
    test_hours = int(cfg.get_path("training.test_size_hours", 336))
    val_hours = int(cfg.get_path("training.validation_size_hours", 168))

    max_ts = panel[TIME_KEY].max()
    test_start = max_ts - pd.Timedelta(hours=test_hours)
    val_start = test_start - pd.Timedelta(hours=val_hours)

    train = panel[panel[TIME_KEY] < val_start]
    validation = panel[(panel[TIME_KEY] >= val_start) & (panel[TIME_KEY] < test_start)]
    test = panel[panel[TIME_KEY] >= test_start]

    if train.empty or validation.empty or test.empty:
        empty = [
            name
            for name, part in (("train", train), ("validation", validation), ("test", test))
            if part.empty
        ]
        # Distinguish "not enough history" from "history has a hole in it".
        # The blend source hits the second case: trip CSVs end weeks before live
        # ingestion begins, and a split window can land entirely inside the gap.
        # Reporting "too short" there would send an operator chasing the wrong
        # problem entirely.
        observed_hours = panel[TIME_KEY].drop_duplicates().sort_values()
        deltas = observed_hours.diff()
        largest_gap = deltas.max()
        detail = (
            f"Chronological split produced empty partition(s): {empty}. "
            f"Data spans {panel[TIME_KEY].min()} to {max_ts} "
            f"({len(observed_hours)} distinct hours) with "
            f"test_size_hours={test_hours} and validation_size_hours={val_hours}."
        )
        if pd.notna(largest_gap) and largest_gap > pd.Timedelta(hours=24):
            gap_end = observed_hours[deltas == largest_gap].iloc[0]
            detail += (
                f" The data contains a {largest_gap} gap ending {gap_end}, so a "
                "split window falls inside it. This is the expected state when "
                "blending historical trip CSVs with live proxy data before the "
                "ingestion DAG has accumulated enough history to bridge the gap "
                "- train with --source offline until it has."
            )
        else:
            detail += " The available history is too short for these split sizes."
        raise ValueError(detail)
    logger.info(
        "Split -> train %d rows (<%s), val %d rows, test %d rows (>=%s)",
        len(train), val_start, len(validation), len(test), test_start,
    )
    return (
        train.reset_index(drop=True),
        validation.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def build_dataset(
    hourly_departures: pd.DataFrame,
    mapping: EntityMapping,
    weather: pd.DataFrame | None,
    config: Config | None = None,
) -> Dataset:
    """Station-level hourly departures -> split, model-ready dataset."""
    cfg = config or load_config()

    entity_observations = aggregate_to_entities(hourly_departures, mapping)
    panel = build_feature_panel(entity_observations, mapping.entities, weather, cfg)
    panel = drop_warmup_rows(panel, cfg)
    panel = panel[panel[TARGET_COLUMN].notna()].reset_index(drop=True)

    train, validation, test = chronological_split(panel, cfg)
    return Dataset(train, validation, test, feature_columns(cfg), mapping)


def regression_metrics(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """MAE / RMSE / R2 plus a zero-safe percentage error.

    Plain MAPE is undefined on the many genuine zero-demand hours in this data,
    so two alternatives are reported instead:

    * ``smape`` - symmetric MAPE, defined everywhere.
    * ``mape_nonzero`` - classic MAPE restricted to hours with actual demand,
      with the coverage share reported alongside so the number is interpretable.
    """
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    error = predicted - actual

    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))

    denominator = np.abs(actual) + np.abs(predicted)
    smape = float(
        np.mean(np.where(denominator == 0, 0.0, 2.0 * np.abs(error) / np.where(denominator == 0, 1.0, denominator)))
        * 100.0
    )

    nonzero = actual > 0
    mape_nonzero = (
        float(np.mean(np.abs(error[nonzero] / actual[nonzero])) * 100.0)
        if nonzero.any()
        else float("nan")
    )

    variance = float(np.var(actual))
    r2 = float(1.0 - np.mean(error**2) / variance) if variance > 0 else float("nan")

    return {
        "mae": mae,
        "rmse": rmse,
        "smape": smape,
        "mape_nonzero": mape_nonzero,
        "nonzero_share": float(nonzero.mean()),
        "r2": r2,
        "mean_actual": float(np.mean(actual)),
    }


def seasonal_naive_baseline(frame: pd.DataFrame) -> dict[str, float]:
    """Same-hour-last-week baseline.

    Reported next to every model so the headline MAE is interpretable: a model
    that cannot beat this is not adding value.
    """
    if "lag_168h" not in frame.columns:
        return {}
    usable = frame[frame["lag_168h"].notna()]
    if usable.empty:
        return {}
    metrics = regression_metrics(usable[TARGET_COLUMN], usable["lag_168h"].to_numpy())
    return {f"baseline_{k}": v for k, v in metrics.items()}
