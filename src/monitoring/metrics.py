"""Rolling accuracy: join served predictions to realised demand.

This is the error half of the closed loop. ``monitor_dag`` calls
:func:`evaluate_recent_predictions` hourly; if the rolling MAE breaches its
threshold, retraining is triggered.

**What "realised" means here.** Once the live loop is running, realised demand
comes from the GBFS inventory-delta **proxy**, not from true ride counts. So the
rolling MAE measures agreement between the model and a noisy, downward-biased
estimate of demand - not true accuracy. It is a perfectly good *change detector*,
which is what the loop needs, but it should never be quoted as the model's real
error. The honest number is the holdout MAE from training on trip-CSV labels.

Rows are tagged with the label source they were scored against so this
distinction survives into the dashboard.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import pandas as pd

from src.config import Config, load_config
from src.models.dataset import regression_metrics

logger = logging.getLogger(__name__)


@dataclass
class AccuracyReport:
    """Rolling accuracy over a window, plus the breach verdict."""

    window_start: datetime
    window_end: datetime
    n_samples: int
    metrics: dict[str, float] = field(default_factory=dict)
    breached: bool = False
    breach_reason: str | None = None
    model_name: str | None = None
    model_version: str | None = None
    label_source: str = "gbfs_proxy"
    evaluated: bool = True

    @property
    def rolling_mae(self) -> float | None:
        return self.metrics.get("mae")

    def to_record(self) -> dict:
        return {
            "window_start": self.window_start,
            "window_end": self.window_end,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "n_samples": self.n_samples,
            "rolling_mae": self.metrics.get("mae"),
            "rolling_rmse": self.metrics.get("rmse"),
            "breached": self.breached,
            "breach_reason": self.breach_reason,
        }


def join_predictions_to_realised(
    predictions: pd.DataFrame,
    realised_departures: pd.DataFrame,
    mapping,
) -> pd.DataFrame:
    """Match each logged prediction to what actually happened that hour.

    ``realised_departures`` is station-level; predictions are entity-level, so
    the realised side is rolled up through the same frozen mapping the model was
    trained with. Using a freshly-fitted mapping would compare against different
    groupings entirely.
    """
    if predictions.empty or realised_departures.empty:
        return pd.DataFrame(columns=["entity_id", "target_hour_ts", "predicted", "actual"])

    from src.features.station_clusters import aggregate_to_entities

    actual = aggregate_to_entities(realised_departures, mapping).rename(
        columns={"hour_ts": "target_hour_ts", "departures": "actual"}
    )

    # Several predictions may exist for one hour (re-requests, model reloads);
    # the most recent one is what was actually being served.
    latest = (
        predictions.sort_values("predicted_at")
        .groupby(["entity_id", "target_hour_ts"], as_index=False)
        .last()
        if "predicted_at" in predictions.columns
        else predictions
    )

    joined = latest.merge(actual, on=["entity_id", "target_hour_ts"], how="inner")
    logger.info(
        "Joined %d of %d predictions to realised demand", len(joined), len(latest)
    )
    return joined


def evaluate_recent_predictions(
    config: Config | None = None,
    *,
    window_hours: int | None = None,
    reference_mae: float | None = None,
) -> AccuracyReport:
    """Compute rolling error over the recent window and decide if it breached.

    Returns ``evaluated=False`` when there is too little matched data to judge.
    Triggering an expensive retrain on six data points would be worse than
    waiting for the next hour.
    """
    cfg = config or load_config()
    window = window_hours or int(cfg.get_path("monitoring.rolling_mae_window_hours", 24))
    min_samples = int(cfg.get_path("monitoring.min_samples_for_evaluation", 200))

    now = datetime.now(tz=UTC).replace(tzinfo=None)
    # Skip the current hour: it is still accumulating and would look like a miss.
    window_end = now.replace(minute=0, second=0, microsecond=0)
    window_start = window_end - timedelta(hours=window)

    from src.serving.model_loader import ModelRegistryLoader
    from src.storage.db import read_hourly_departures, read_predictions

    predictions = read_predictions(window_start, window_end)
    if predictions.empty:
        logger.info("No predictions logged in %s..%s", window_start, window_end)
        return AccuracyReport(window_start, window_end, 0, evaluated=False,
                              breach_reason="no predictions logged")

    try:
        bundle = ModelRegistryLoader(cfg).load()
    except Exception as exc:  # noqa: BLE001
        logger.error("Cannot evaluate without the production model: %s", exc)
        return AccuracyReport(window_start, window_end, 0, evaluated=False,
                              breach_reason=f"model unavailable: {exc}")

    realised = read_hourly_departures(window_days=max(1, window // 24 + 1), config=cfg)
    joined = join_predictions_to_realised(predictions, realised, bundle.mapping)

    if len(joined) < min_samples:
        logger.info(
            "Only %d matched prediction/actual pairs (need %d); skipping evaluation",
            len(joined), min_samples,
        )
        return AccuracyReport(
            window_start, window_end, len(joined), evaluated=False,
            model_name=bundle.model_name, model_version=bundle.model_version,
            breach_reason=f"insufficient samples ({len(joined)} < {min_samples})",
        )

    metrics = regression_metrics(joined["actual"], joined["predicted"].to_numpy())
    report = AccuracyReport(
        window_start=window_start,
        window_end=window_end,
        n_samples=len(joined),
        metrics=metrics,
        model_name=bundle.model_name,
        model_version=bundle.model_version,
    )

    report.breached, report.breach_reason = _judge_breach(metrics["mae"], cfg, reference_mae)
    logger.info(
        "Rolling MAE=%.3f over %d samples; breached=%s (%s)",
        metrics["mae"], len(joined), report.breached, report.breach_reason,
    )
    return report


def _judge_breach(
    rolling_mae: float, config: Config, reference_mae: float | None
) -> tuple[bool, str | None]:
    """Two independent triggers: an absolute ceiling and relative degradation.

    The relative rule is the more meaningful of the two - it asks whether the
    model got worse than it was at training time, which is what "degradation"
    actually means. The absolute ceiling is a backstop for the case where the
    model was never good enough to begin with.
    """
    absolute_threshold = float(config.get_path("monitoring.mae_threshold", 4.0))
    degradation_ratio = float(config.get_path("monitoring.mae_degradation_ratio", 1.5))

    if rolling_mae > absolute_threshold:
        return True, (
            f"rolling MAE {rolling_mae:.3f} exceeds the absolute threshold "
            f"{absolute_threshold:.3f}"
        )

    if reference_mae is not None and reference_mae > 0:
        limit = reference_mae * degradation_ratio
        if rolling_mae > limit:
            return True, (
                f"rolling MAE {rolling_mae:.3f} is more than {degradation_ratio:.2f}x "
                f"the training holdout MAE {reference_mae:.3f} (limit {limit:.3f})"
            )

    return False, None


def training_reference_mae(config: Config | None = None) -> float | None:
    """Holdout MAE of the live model, used as the degradation baseline."""
    cfg = config or load_config()
    try:
        from mlflow.tracking import MlflowClient

        from src.serving.model_loader import ModelRegistryLoader

        bundle = ModelRegistryLoader(cfg).load()
        run = MlflowClient().get_run(bundle.run_id)
        value = run.data.metrics.get("test_mae")
        return float(value) if value is not None else None
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read the training reference MAE: %s", exc)
        return None
