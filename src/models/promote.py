"""Champion/challenger promotion.

The rule, in one sentence: a challenger replaces the champion only if it beats
it on the holdout metric by more than a configured relative margin.

The margin matters. Holdout MAE wobbles run to run from resampling noise alone,
so promoting on any improvement at all would swap the production model most
days for no real gain, and would make every incident harder to explain. A 2%
floor means a promotion reflects a change in the model, not in the weather.

The comparison is deliberately **model-agnostic** - LightGBM and the deep model
are compared on exactly the same holdout with the same metric, and whichever
wins becomes production.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from src.config import Config, load_config

logger = logging.getLogger(__name__)

# Metrics where smaller is better. Everything else is treated as higher-is-better.
LOWER_IS_BETTER = {"mae", "rmse", "smape", "mape_nonzero"}


@dataclass
class Candidate:
    """One trained model competing for the production slot."""

    name: str
    run_id: str
    metrics: dict[str, float]
    model_uri: str
    extra: dict[str, Any] = field(default_factory=dict)

    def score(self, metric: str) -> float:
        if metric not in self.metrics:
            raise KeyError(f"Candidate {self.name!r} has no metric {metric!r}")
        return float(self.metrics[metric])


@dataclass
class PromotionDecision:
    """Outcome of a champion/challenger comparison."""

    promote: bool
    reason: str
    challenger: Candidate | None = None
    challenger_score: float | None = None
    champion_score: float | None = None
    relative_improvement: float | None = None


def select_best_candidate(candidates: list[Candidate], metric: str) -> Candidate:
    """Best candidate on ``metric``, respecting the metric's direction."""
    if not candidates:
        raise ValueError("No candidates to select from")
    reverse = metric not in LOWER_IS_BETTER
    return sorted(candidates, key=lambda c: c.score(metric), reverse=reverse)[0]


def relative_improvement(challenger: float, champion: float, metric: str) -> float:
    """Fractional improvement of challenger over champion.

    Positive means better. Guards against a zero champion score, which would
    otherwise divide by zero and promote unconditionally.
    """
    if champion == 0:
        # A champion scoring exactly 0 on an error metric is already perfect, so
        # any non-zero challenger is strictly worse. Getting the sign wrong here
        # would promote a worse model against a flawless one.
        if challenger == champion:
            return 0.0
        if metric in LOWER_IS_BETTER:
            return -1.0 if challenger > champion else 1.0
        return 1.0 if challenger > champion else -1.0
    if metric in LOWER_IS_BETTER:
        return (champion - challenger) / abs(champion)
    return (challenger - champion) / abs(champion)


def decide_promotion(
    candidates: list[Candidate],
    champion_score: float | None,
    *,
    metric: str = "mae",
    min_relative_improvement: float = 0.02,
) -> PromotionDecision:
    """Pure decision function - no MLflow, no I/O, fully unit-testable.

    ``champion_score is None`` means there is no production model yet, in which
    case the best candidate is promoted unconditionally (something must serve).
    """
    if not candidates:
        return PromotionDecision(False, "No candidates were trained")

    challenger = select_best_candidate(candidates, metric)
    challenger_score = challenger.score(metric)

    if champion_score is None:
        return PromotionDecision(
            True,
            f"No existing production model; promoting {challenger.name} "
            f"({metric}={challenger_score:.4f})",
            challenger,
            challenger_score,
            None,
            None,
        )

    improvement = relative_improvement(challenger_score, champion_score, metric)
    if improvement > min_relative_improvement:
        reason = (
            f"{challenger.name} improves {metric} by {improvement:.2%} "
            f"({champion_score:.4f} -> {challenger_score:.4f}), "
            f"clearing the {min_relative_improvement:.2%} bar"
        )
        return PromotionDecision(
            True, reason, challenger, challenger_score, champion_score, improvement
        )

    reason = (
        f"Best challenger {challenger.name} changed {metric} by {improvement:.2%} "
        f"({champion_score:.4f} -> {challenger_score:.4f}), which does not clear "
        f"the {min_relative_improvement:.2%} bar; keeping the current champion"
    )
    return PromotionDecision(
        False, reason, challenger, challenger_score, champion_score, improvement
    )


class ModelRegistry:
    """Thin MLflow Model Registry helper built on aliases.

    Aliases are used rather than the deprecated stage API; ``@production`` is
    what the serving layer resolves.
    """

    def __init__(self, config: Config | None = None) -> None:
        self._config = config or load_config()
        self.model_name = self._config.require("mlflow.registered_model_name")
        self.alias = self._config.get_path("mlflow.production_alias", "production")
        mlflow.set_tracking_uri(self._config.require("mlflow.tracking_uri"))
        self._client = MlflowClient()

    @property
    def client(self) -> MlflowClient:
        return self._client

    def production_version(self):
        """Current production model version, or ``None`` if unset."""
        try:
            return self._client.get_model_version_by_alias(self.model_name, self.alias)
        except Exception:  # noqa: BLE001 - covers "no such model" and "no such alias"
            return None

    def champion_metric(self, metric: str) -> float | None:
        """Holdout metric of the current production model.

        Read from the run that produced it, so champion and challenger are
        always compared on metrics computed the same way.
        """
        version = self.production_version()
        if version is None:
            return None
        try:
            run = self._client.get_run(version.run_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load champion run %s: %s", version.run_id, exc)
            return None
        value = run.data.metrics.get(f"test_{metric}", run.data.metrics.get(metric))
        return float(value) if value is not None else None

    def promote(self, candidate: Candidate, decision: PromotionDecision) -> str:
        """Register the candidate and point the production alias at it."""
        version = mlflow.register_model(candidate.model_uri, self.model_name)
        self._client.set_registered_model_alias(
            self.model_name, self.alias, version.version
        )
        self._client.set_model_version_tag(
            self.model_name, version.version, "model_kind", candidate.name
        )
        self._client.set_model_version_tag(
            self.model_name, version.version, "promotion_reason", decision.reason
        )
        logger.info(
            "Promoted %s to %s@%s (version %s)",
            candidate.name, self.model_name, self.alias, version.version,
        )
        return version.version


def run_promotion(
    candidates: list[Candidate], config: Config | None = None
) -> PromotionDecision:
    """Compare candidates against the champion and promote if warranted."""
    cfg = config or load_config()
    metric = cfg.get_path("promotion.metric", "mae")
    margin = float(cfg.get_path("promotion.min_relative_improvement", 0.02))

    registry = ModelRegistry(cfg)
    champion_score = registry.champion_metric(metric)
    decision = decide_promotion(
        candidates,
        champion_score,
        metric=metric,
        min_relative_improvement=margin,
    )
    logger.info("Promotion decision: %s", decision.reason)
    if decision.promote and decision.challenger is not None:
        registry.promote(decision.challenger, decision)
    return decision
