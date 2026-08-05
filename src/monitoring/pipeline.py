"""Monitoring orchestration - the decision that closes the loop.

``monitor_dag`` calls :func:`run_monitoring_cycle` hourly. It answers one
question: *should the system retrain itself right now?*

Two independent triggers, either of which fires:

* **Error** - rolling MAE breached its absolute ceiling, or degraded past a
  multiple of the model's own training holdout MAE.
* **Drift** - too large a share of features have moved away from the training
  distribution.

Both are needed. Error monitoring is the ground truth signal but is late and
requires labels; drift is early and label-free but can fire on changes the model
handles fine. Retraining on either is the conservative choice: a needless
retrain costs a few minutes of compute, while a missed one silently serves a
degraded model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.config import Config, load_config

logger = logging.getLogger(__name__)


@dataclass
class MonitoringOutcome:
    """Everything the DAG needs to decide and to record."""

    should_retrain: bool
    reasons: list[str] = field(default_factory=list)
    accuracy: dict = field(default_factory=dict)
    drift: dict = field(default_factory=dict)

    def summary(self) -> str:
        if not self.reasons:
            return "No retraining trigger fired"
        return "; ".join(self.reasons)


def run_monitoring_cycle(config: Config | None = None) -> MonitoringOutcome:
    """Evaluate accuracy and drift, persist the result, and decide on retraining."""
    cfg = config or load_config()

    from src.monitoring.drift import compute_drift, load_current_features, load_reference_features
    from src.monitoring.metrics import evaluate_recent_predictions, training_reference_mae
    from src.storage.db import write_model_metrics

    reasons: list[str] = []

    # --- accuracy ----------------------------------------------------------
    reference_mae = training_reference_mae(cfg)
    accuracy = evaluate_recent_predictions(cfg, reference_mae=reference_mae)
    if accuracy.evaluated and accuracy.breached:
        reasons.append(f"accuracy: {accuracy.breach_reason}")
    elif not accuracy.evaluated:
        logger.info("Accuracy not evaluated this cycle: %s", accuracy.breach_reason)

    # --- drift -------------------------------------------------------------
    drift_result = None
    try:
        reference = load_reference_features(cfg)
        current = load_current_features(
            cfg, hours=int(cfg.get_path("monitoring.rolling_mae_window_hours", 24))
        )
        drift_result = compute_drift(reference, current, cfg)
        if drift_result.evaluated and drift_result.dataset_drift:
            threshold = cfg.get_path("monitoring.drift_share_threshold", 0.3)
            reasons.append(
                f"drift: {drift_result.drift_share:.0%} of features drifted "
                f"(threshold {float(threshold):.0%}); "
                f"worst: {drift_result.drifted_features[:5]}"
            )
        elif drift_result and not drift_result.evaluated:
            logger.info("Drift not evaluated this cycle: %s", drift_result.reason)
    except Exception as exc:  # noqa: BLE001 - a drift failure must not block the error trigger
        logger.error("Drift evaluation failed: %s", exc)

    outcome = MonitoringOutcome(
        should_retrain=bool(reasons),
        reasons=reasons,
        accuracy={
            "evaluated": accuracy.evaluated,
            "n_samples": accuracy.n_samples,
            "rolling_mae": accuracy.rolling_mae,
            "reference_mae": reference_mae,
            "breached": accuracy.breached,
            "reason": accuracy.breach_reason,
            # The rolling number is scored against proxy labels once the live
            # loop is running; it is a change detector, not true accuracy.
            "label_source": accuracy.label_source,
        },
        drift=drift_result.to_dict() if drift_result else {},
    )

    # --- persist -----------------------------------------------------------
    try:
        record = accuracy.to_record()
        record.update(
            {
                "drift_share": drift_result.drift_share if drift_result else None,
                "dataset_drift": bool(drift_result.dataset_drift) if drift_result else None,
                "breached": outcome.should_retrain,
                "breach_reason": outcome.summary(),
            }
        )
        write_model_metrics(record)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not persist monitoring metrics: %s", exc)

    logger.info(
        "Monitoring cycle: should_retrain=%s | %s", outcome.should_retrain, outcome.summary()
    )
    return outcome
