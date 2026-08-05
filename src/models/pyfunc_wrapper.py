"""One MLflow pyfunc interface for every candidate model.

Champion/challenger only works if promotion is model-agnostic, so LightGBM and
the deep model are both packaged behind this single wrapper. ``serving/api.py``
calls ``mlflow.pyfunc.load_model(...)`` and never learns which family won.

Bundled artifacts travel with the model:

* ``model`` - the serialised estimator (LightGBM text booster or a torch state dict).
* ``entity_mapping`` - the frozen station -> cluster assignment. Without it the
  serving path could not reproduce the entity ids the model was trained on.
* ``metadata`` - feature order, model kind, and the training window.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import mlflow.pyfunc
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ARTIFACT_MODEL = "model"
ARTIFACT_MAPPING = "entity_mapping"
ARTIFACT_METADATA = "metadata"


class DemandModelWrapper(mlflow.pyfunc.PythonModel):
    """Loads whichever estimator was saved and applies it to a feature frame."""

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        metadata_path = context.artifacts[ARTIFACT_METADATA]
        self.metadata: dict[str, Any] = json.loads(
            Path(metadata_path).read_text(encoding="utf-8")
        )
        self.features: list[str] = self.metadata["features"]
        self.model_kind: str = self.metadata["model_kind"]

        model_path = context.artifacts[ARTIFACT_MODEL]
        if self.model_kind == "lightgbm":
            import lightgbm as lgb

            self._booster = lgb.Booster(model_file=model_path)
        elif self.model_kind == "deep":
            from src.models.deep_model import load_deep_model

            self._booster = load_deep_model(model_path, self.metadata)
        else:
            raise ValueError(f"Unknown model_kind: {self.model_kind!r}")

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> np.ndarray:
        missing = [c for c in self.features if c not in model_input.columns]
        if missing:
            raise ValueError(
                f"Input is missing {len(missing)} feature column(s) required by this "
                f"model: {missing[:10]}"
            )
        # Everything downstream is numeric, so coerce to float. This keeps a
        # caller that supplies ints (or strings from a JSON payload) working
        # rather than failing deep inside the estimator.
        matrix = model_input[self.features].astype("float64")
        raw = self._booster.predict(matrix)
        return np.clip(np.asarray(raw, dtype=float), 0.0, None)


def build_artifacts(
    directory: Path,
    *,
    model_kind: str,
    features: list[str],
    entity_mapping_path: Path,
    metadata: dict[str, Any],
    model_path: Path,
) -> dict[str, str]:
    """Write the metadata sidecar and return the MLflow ``artifacts`` mapping."""
    directory.mkdir(parents=True, exist_ok=True)
    payload = dict(metadata)
    payload.update({"model_kind": model_kind, "features": list(features)})
    metadata_path = directory / "metadata.json"
    metadata_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return {
        ARTIFACT_MODEL: str(model_path),
        ARTIFACT_MAPPING: str(entity_mapping_path),
        ARTIFACT_METADATA: str(metadata_path),
    }
