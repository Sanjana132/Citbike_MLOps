"""Loads the current Production model from the MLflow registry.

The serving layer resolves ``models:/<name>@production`` and is deliberately
blind to which model family won champion/challenger - both are packaged behind
the same pyfunc interface. A promotion in the registry is therefore picked up by
a reload with no code change and no redeploy.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from src.config import Config, load_config
from src.features.station_clusters import EntityMapping

logger = logging.getLogger(__name__)


@dataclass
class ModelBundle:
    """A loaded model plus everything needed to build its inputs."""

    model: Any
    mapping: EntityMapping
    metadata: dict[str, Any]
    model_name: str
    model_version: str
    run_id: str
    loaded_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))

    @property
    def features(self) -> list[str]:
        return list(self.metadata.get("features", []))

    def info(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "model_kind": self.metadata.get("model_kind"),
            "run_id": self.run_id,
            "trained_at": self.metadata.get("trained_at"),
            "train_start": self.metadata.get("train_start"),
            "train_end": self.metadata.get("train_end"),
            "granularity": self.metadata.get("granularity"),
            # Surfaced on purpose: a model retrained on live data learned from
            # an inventory-derived proxy, not from true ride counts.
            "label_provenance": self.metadata.get("label_provenance"),
            "n_features": len(self.features),
            "n_entities": len(self.mapping.entities),
            "loaded_at": self.loaded_at.isoformat(),
        }


class ModelRegistryLoader:
    """Thread-safe holder for the active model, with hot reload."""

    def __init__(self, config: Config | None = None) -> None:
        self._config = config or load_config()
        self._model_name = self._config.require("mlflow.registered_model_name")
        self._alias = self._config.get_path("mlflow.production_alias", "production")
        self._lock = threading.Lock()
        self._bundle: ModelBundle | None = None
        self._last_error: str | None = None
        mlflow.set_tracking_uri(self._config.require("mlflow.tracking_uri"))

    @property
    def bundle(self) -> ModelBundle | None:
        return self._bundle

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def _download_sidecars(self, client: MlflowClient, run_id: str) -> tuple[EntityMapping, dict]:
        """Fetch the entity mapping and metadata bundled with the model.

        These live under ``model/artifacts/`` because they were logged as
        pyfunc artifacts, which keeps them versioned with the model itself
        rather than with a run that might be deleted independently.
        """
        mapping_path = client.download_artifacts(run_id, "model/artifacts/entity_mapping.json")
        metadata_path = client.download_artifacts(run_id, "model/artifacts/metadata.json")
        mapping = EntityMapping.load(mapping_path)
        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        return mapping, metadata

    def load(self, *, force: bool = False) -> ModelBundle:
        """Resolve and load the production model.

        Returns the cached bundle when the registry still points at the same
        version, so a periodic refresh is cheap.
        """
        with self._lock:
            client = MlflowClient()
            try:
                version = client.get_model_version_by_alias(self._model_name, self._alias)
            except Exception as exc:  # noqa: BLE001
                self._last_error = (
                    f"No model registered at {self._model_name}@{self._alias}: {exc}"
                )
                logger.error(self._last_error)
                raise RuntimeError(self._last_error) from exc

            if (
                not force
                and self._bundle is not None
                and self._bundle.model_version == version.version
            ):
                return self._bundle

            logger.info(
                "Loading %s version %s (run %s)", self._model_name, version.version, version.run_id
            )
            model = mlflow.pyfunc.load_model(f"models:/{self._model_name}@{self._alias}")
            mapping, metadata = self._download_sidecars(client, version.run_id)

            self._bundle = ModelBundle(
                model=model,
                mapping=mapping,
                metadata=metadata,
                model_name=self._model_name,
                # MLflow hands back an int here; the API schema and the
                # predictions table both treat versions as opaque strings.
                model_version=str(version.version),
                run_id=version.run_id,
            )
            self._last_error = None
            logger.info("Model ready: %s", self._bundle.info())
            return self._bundle

    def predict(self, features: pd.DataFrame) -> Any:
        bundle = self._bundle or self.load()
        return bundle.model.predict(features)
