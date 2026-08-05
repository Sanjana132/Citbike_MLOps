"""LightGBM demand model - the primary baseline.

Wrapped in an ``mlflow.pyfunc`` model so that whatever wins champion/challenger
(gradient boosting or the deep model) is loaded through one identical interface
in ``serving/api.py``. The API never branches on model type.

The regression objective defaults to Poisson: departures are non-negative
over-dispersed counts, and an L2 objective happily predicts negative demand.
"""

from __future__ import annotations

import logging
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import Config, load_config

logger = logging.getLogger(__name__)

MODEL_NAME = "lightgbm"


class LightGBMDemandModel:
    """Thin wrapper: fixed feature order in, non-negative predictions out."""

    def __init__(self, features: list[str], config: Config | None = None) -> None:
        self._config = config or load_config()
        self.features = list(features)
        self.booster: lgb.LGBMRegressor | None = None
        self.best_iteration_: int | None = None

    def _params(self) -> dict[str, Any]:
        params = dict(self._config.get_path("models.lightgbm", {}) or {})
        params.pop("early_stopping_rounds", None)
        params.setdefault("objective", "poisson")
        params["random_state"] = int(self._config.get_path("project.seed", 42))
        # Single-threaded determinism is not worth the wall-clock cost here, but
        # a fixed seed plus deterministic bagging keeps runs reproducible.
        params.setdefault("deterministic", True)
        params.setdefault("force_row_wise", True)
        params.setdefault("verbose", -1)
        return params

    def fit(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_validation: pd.DataFrame | None = None,
        y_validation: pd.Series | None = None,
    ) -> LightGBMDemandModel:
        params = self._params()
        early_stopping = int(
            self._config.get_path("models.lightgbm.early_stopping_rounds", 100)
        )
        self.booster = lgb.LGBMRegressor(**params)

        callbacks = [lgb.log_evaluation(period=200)]
        eval_set = None
        if x_validation is not None and y_validation is not None and len(x_validation):
            eval_set = [(x_validation[self.features], y_validation)]
            callbacks.append(lgb.early_stopping(early_stopping, verbose=False))

        self.booster.fit(
            x_train[self.features],
            y_train,
            eval_set=eval_set,
            eval_metric="l1",
            callbacks=callbacks,
        )
        self.best_iteration_ = getattr(self.booster, "best_iteration_", None)
        logger.info(
            "LightGBM fitted on %d rows; best_iteration=%s", len(x_train), self.best_iteration_
        )
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        if self.booster is None:
            raise RuntimeError("Model has not been fitted")
        missing = [c for c in self.features if c not in x.columns]
        if missing:
            raise ValueError(f"Missing feature columns at predict time: {missing}")
        raw = self.booster.predict(x[self.features])
        # Demand cannot be negative; clip rather than let the API emit nonsense.
        return np.clip(np.asarray(raw, dtype=float), 0.0, None)

    def feature_importance(self) -> pd.DataFrame:
        if self.booster is None:
            raise RuntimeError("Model has not been fitted")
        return (
            pd.DataFrame(
                {
                    "feature": self.features,
                    "importance": self.booster.feature_importances_,
                }
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
