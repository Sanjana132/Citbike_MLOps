"""LSTM sequence model - the second champion/challenger candidate.

Why an LSTM rather than a Temporal Fusion Transformer: ``pytorch-forecasting``
pins an older Lightning/PyTorch stack that conflicts with the versions the rest
of this project uses, and a dependency conflict at the centre of the retraining
loop is a worse outcome than a simpler architecture. A plain PyTorch LSTM has no
extra dependencies beyond torch, trains in minutes on CPU, and gives the
promotion logic a genuinely different model family to choose between. The choice
is exposed as ``models.deep_model`` in config so TFT can be added later without
touching the promotion path.

The model consumes, per entity, a sequence of the last ``sequence_length`` hours
of demand plus the covariates for the hour being predicted, and emits one value.

Leakage control: the scaler is fit on the **training split only** and then
applied unchanged to validation and test. Fitting it on the full panel would
leak test-set scale into training, which is the subtle version of the same
mistake a random split makes loudly.
"""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import Config, load_config
from src.features.build_features import ENTITY_KEY, TARGET_COLUMN, TIME_KEY
from src.models.promote import Candidate

logger = logging.getLogger(__name__)

MODEL_NAME = "deep"


def _torch():
    """Import torch lazily so the rest of the project runs without it."""
    import torch

    return torch


class _LSTMRegressor:
    """Thin nn.Module wrapper. Defined lazily to keep torch an optional import."""

    def __init__(self, n_covariates: int, config: Config) -> None:
        torch = _torch()
        nn = torch.nn

        params = config.get_path("models.lstm", {}) or {}
        hidden = int(params.get("hidden_size", 64))
        layers = int(params.get("num_layers", 2))
        dropout = float(params.get("dropout", 0.2))

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=1,
                    hidden_size=hidden,
                    num_layers=layers,
                    batch_first=True,
                    dropout=dropout if layers > 1 else 0.0,
                )
                self.head = nn.Sequential(
                    nn.Linear(hidden + n_covariates, hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, 1),
                )

            def forward(self, sequence, covariates):
                _, (hidden_state, _) = self.lstm(sequence)
                combined = torch.cat([hidden_state[-1], covariates], dim=1)
                # Softplus keeps predictions non-negative, matching the Poisson
                # objective used by the LightGBM candidate.
                return torch.nn.functional.softplus(self.head(combined)).squeeze(-1)

        self.net = Net()
        self.n_covariates = n_covariates


class DeepDemandModel:
    """Sequence model with the same predict() contract as the LightGBM model."""

    def __init__(self, features: list[str], config: Config | None = None) -> None:
        self._config = config or load_config()
        self.features = list(features)
        self.sequence_length = int(
            self._config.get_path("models.lstm.sequence_length", 168)
        )
        # Covariates are every feature except the raw lag columns, which the
        # LSTM already receives as its input sequence.
        self.covariates = [f for f in self.features if not f.startswith("lag_")]
        self.net = None
        self.covariate_mean: np.ndarray | None = None
        self.covariate_std: np.ndarray | None = None
        self.target_scale: float = 1.0

    # -- data preparation --------------------------------------------------

    def prepend_context(
        self, split: pd.DataFrame, history: pd.DataFrame | None
    ) -> pd.DataFrame:
        """Prefix each entity's split with its preceding ``sequence_length`` hours.

        Without this, a split shorter than ``sequence_length`` yields **zero**
        sequences and the model silently scores nothing. The validation split is
        exactly 168 hours and the sequence length is 168, so it produced no
        predictions at all and early stopping fell back to the training loss.

        Prefixing also matches how serving works: the sequence for hour *t* comes
        from the history before *t*, which may sit in an earlier split.
        """
        return self.prepend_context_with_mask(split, history)[0]

    def prepend_context_with_mask(
        self, split: pd.DataFrame, history: pd.DataFrame | None
    ) -> tuple[pd.DataFrame, pd.Series]:
        """As :meth:`prepend_context`, plus a mask marking the split's own rows.

        The marker is attached before concatenating rather than recovered
        afterwards by comparing timestamps - context and split can share hours
        across entities, so a timestamp comparison would mislabel rows.
        """
        marker = "_is_split_row"
        tagged_split = split.assign(**{marker: True})
        if history is None or history.empty:
            combined = tagged_split.reset_index(drop=True)
            return combined.drop(columns=marker), combined[marker]

        tails = (
            history.sort_values([ENTITY_KEY, TIME_KEY])
            .groupby(ENTITY_KEY, observed=True, group_keys=False)
            .tail(self.sequence_length)
            .assign(**{marker: False})
        )
        combined = (
            pd.concat([tails, tagged_split], ignore_index=True)
            .sort_values([ENTITY_KEY, TIME_KEY])
            .reset_index(drop=True)
        )
        return combined.drop(columns=marker), combined[marker]

    def _sequences(
        self,
        panel: pd.DataFrame,
        *,
        target_mask: pd.Series | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build (sequence, covariates, target) arrays from a feature panel.

        The demand sequence is reconstructed from the panel's own history per
        entity, using only rows strictly before the target hour.

        ``target_mask`` restricts which rows become training targets while still
        allowing earlier rows to serve as sequence context - that is how a
        context-prefixed split is scored on its own rows only.
        """
        sequences, covariates, targets = [], [], []
        length = self.sequence_length

        working = panel
        if target_mask is not None:
            working = panel.assign(_is_target=target_mask.to_numpy())

        for _, group in working.groupby(ENTITY_KEY, observed=True):
            group = group.sort_values(TIME_KEY)
            demand = group[TARGET_COLUMN].to_numpy(dtype=float)
            covariate_matrix = group[self.covariates].to_numpy(dtype=float)
            is_target = (
                group["_is_target"].to_numpy(dtype=bool)
                if target_mask is not None
                else np.ones(len(group), dtype=bool)
            )

            for index in range(length, len(group)):
                if not is_target[index]:
                    continue
                window = demand[index - length : index]
                if np.isnan(window).any() or np.isnan(demand[index]):
                    continue
                sequences.append(window)
                covariates.append(covariate_matrix[index])
                targets.append(demand[index])

        if not sequences:
            return np.empty((0, length)), np.empty((0, len(self.covariates))), np.empty(0)
        return (
            np.asarray(sequences, dtype=np.float32),
            np.nan_to_num(np.asarray(covariates, dtype=np.float32)),
            np.asarray(targets, dtype=np.float32),
        )

    # -- training ----------------------------------------------------------

    def fit(self, train_panel: pd.DataFrame, validation_panel: pd.DataFrame) -> DeepDemandModel:
        torch = _torch()

        seed = int(self._config.get_path("project.seed", 42))
        torch.manual_seed(seed)
        np.random.seed(seed)

        x_seq, x_cov, y = self._sequences(train_panel)
        if len(y) == 0:
            raise ValueError("Not enough contiguous history to build LSTM sequences")

        # Scalers fit on TRAIN ONLY - see the module docstring.
        self.covariate_mean = x_cov.mean(axis=0)
        self.covariate_std = np.where(x_cov.std(axis=0) < 1e-6, 1.0, x_cov.std(axis=0))
        self.target_scale = float(max(y.mean(), 1.0))

        # Validation is scored with training history prefixed, so a split
        # shorter than sequence_length still produces sequences.
        validation_context, validation_mask = self.prepend_context_with_mask(
            validation_panel, train_panel
        )
        v_seq, v_cov, v_y = self._sequences(
            validation_context, target_mask=validation_mask
        )
        if len(v_y) == 0:
            logger.warning(
                "Validation produced no sequences even with context; early "
                "stopping will fall back to the training loss"
            )

        params = self._config.get_path("models.lstm", {}) or {}
        batch_size = int(params.get("batch_size", 256))
        max_epochs = int(params.get("max_epochs", 30))
        learning_rate = float(params.get("learning_rate", 1e-3))
        patience = int(params.get("early_stopping_patience", 5))

        model = _LSTMRegressor(len(self.covariates), self._config)
        self.net = model.net

        optimiser = torch.optim.Adam(self.net.parameters(), lr=learning_rate)
        loss_fn = torch.nn.L1Loss()  # matches the MAE promotion metric

        train_tensors = torch.utils.data.TensorDataset(
            torch.from_numpy(x_seq / self.target_scale).unsqueeze(-1),
            torch.from_numpy((x_cov - self.covariate_mean) / self.covariate_std),
            torch.from_numpy(y / self.target_scale),
        )
        loader = torch.utils.data.DataLoader(
            train_tensors, batch_size=batch_size, shuffle=True, drop_last=False
        )

        best_loss, best_state, epochs_without_improvement = float("inf"), None, 0
        for epoch in range(max_epochs):
            self.net.train()
            epoch_loss = 0.0
            for sequence, covariate, target in loader:
                optimiser.zero_grad()
                loss = loss_fn(self.net(sequence, covariate), target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                optimiser.step()
                epoch_loss += float(loss) * len(target)
            epoch_loss /= max(len(train_tensors), 1)

            validation_loss = epoch_loss
            if len(v_y):
                predictions = self._predict_arrays(v_seq, v_cov)
                validation_loss = float(np.mean(np.abs(predictions - v_y)))

            logger.info(
                "LSTM epoch %d/%d: train_loss=%.4f val_mae=%.4f",
                epoch + 1, max_epochs, epoch_loss, validation_loss,
            )

            if validation_loss < best_loss - 1e-4:
                best_loss = validation_loss
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break

        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    # -- inference ---------------------------------------------------------

    def _predict_arrays(self, x_seq: np.ndarray, x_cov: np.ndarray) -> np.ndarray:
        torch = _torch()
        if len(x_seq) == 0:
            return np.empty(0)
        self.net.eval()
        with torch.no_grad():
            predictions = self.net(
                torch.from_numpy(x_seq / self.target_scale).unsqueeze(-1),
                torch.from_numpy((x_cov - self.covariate_mean) / self.covariate_std),
            )
        return np.clip(predictions.numpy() * self.target_scale, 0.0, None)

    def predict_split(
        self, split: pd.DataFrame, history: pd.DataFrame | None = None
    ) -> pd.Series:
        """Predict a split, drawing sequence context from preceding data.

        Returns a series aligned to ``split.index``; rows that still lack enough
        history come back as NaN and are excluded from scoring.
        """
        combined = self.prepend_context(split, history)
        predictions = self.predict_panel(combined)
        scored = combined[[ENTITY_KEY, TIME_KEY]].assign(_pred=predictions.to_numpy())
        merged = split[[ENTITY_KEY, TIME_KEY]].merge(
            scored.drop_duplicates(subset=[ENTITY_KEY, TIME_KEY]),
            on=[ENTITY_KEY, TIME_KEY],
            how="left",
        )
        return pd.Series(merged["_pred"].to_numpy(), index=split.index)

    def predict_panel(self, panel: pd.DataFrame) -> pd.Series:
        """Predict for a feature panel, returning a series aligned to its index.

        Rows without enough history to form a sequence come back as NaN and are
        excluded from scoring rather than filled with a guess.
        """
        results = pd.Series(np.nan, index=panel.index, dtype=float)
        length = self.sequence_length

        for _, group in panel.groupby("entity_id", observed=True):
            group = group.sort_values(TIME_KEY)
            demand = group[TARGET_COLUMN].to_numpy(dtype=float)
            covariate_matrix = np.nan_to_num(group[self.covariates].to_numpy(dtype=float))

            indices, sequences, covariates = [], [], []
            for position in range(length, len(group)):
                window = demand[position - length : position]
                if np.isnan(window).any():
                    continue
                indices.append(group.index[position])
                sequences.append(window)
                covariates.append(covariate_matrix[position])

            if indices:
                predictions = self._predict_arrays(
                    np.asarray(sequences, dtype=np.float32),
                    np.asarray(covariates, dtype=np.float32),
                )
                results.loc[indices] = predictions
        return results

    def save(self, path: Path) -> Path:
        """Serialise weights plus the fitted scalers as one file."""
        torch = _torch()
        torch.save(
            {
                "state_dict": self.net.state_dict(),
                "covariates": self.covariates,
                "covariate_mean": self.covariate_mean,
                "covariate_std": self.covariate_std,
                "target_scale": self.target_scale,
                "sequence_length": self.sequence_length,
                "features": self.features,
            },
            path,
        )
        return path


def load_deep_model(model_path: str, metadata: dict[str, Any]):
    """Rehydrate a saved deep model for the pyfunc wrapper.

    Note the limitation this exposes: the wrapper's ``predict`` receives a plain
    feature matrix, but an LSTM needs a demand *sequence*. The saved lag columns
    are used to reconstruct a short sequence, which is why the deep candidate is
    evaluated through ``predict_panel`` during training rather than through this
    path. Serving a promoted deep model end-to-end is a documented roadmap item.
    """
    torch = _torch()
    payload = torch.load(model_path, map_location="cpu", weights_only=False)

    config = load_config()
    model = DeepDemandModel(payload["features"], config)
    model.covariates = payload["covariates"]
    model.covariate_mean = payload["covariate_mean"]
    model.covariate_std = payload["covariate_std"]
    model.target_scale = payload["target_scale"]
    model.sequence_length = payload["sequence_length"]

    net = _LSTMRegressor(len(model.covariates), config).net
    net.load_state_dict(payload["state_dict"])
    net.eval()
    model.net = net
    return _DeepPredictAdapter(model)


class _DeepPredictAdapter:
    """Adapts the sequence model to the wrapper's ``predict(matrix)`` contract."""

    def __init__(self, model: DeepDemandModel) -> None:
        self._model = model

    def predict(self, matrix: pd.DataFrame) -> np.ndarray:
        lag_columns = sorted(
            [c for c in matrix.columns if c.startswith("lag_")],
            key=lambda c: int(c.removeprefix("lag_").removesuffix("h")),
        )
        if not lag_columns:
            raise ValueError("Deep model needs lag columns to reconstruct a sequence")

        # Build the longest sequence the available lags allow, newest last, then
        # left-pad by repeating the oldest known value.
        lags = matrix[lag_columns].to_numpy(dtype=float)[:, ::-1]
        lags = np.nan_to_num(lags, nan=float(np.nanmean(lags)) if np.isfinite(lags).any() else 0.0)
        padded = np.repeat(lags[:, :1], self._model.sequence_length, axis=1)
        width = min(lags.shape[1], self._model.sequence_length)
        padded[:, -width:] = lags[:, -width:]

        covariates = np.nan_to_num(
            matrix[self._model.covariates].to_numpy(dtype=float)
        ).astype(np.float32)
        return self._model._predict_arrays(padded.astype(np.float32), covariates)


def train_deep_model(
    dataset,
    config: Config,
    mapping_path: Path,
    run_metadata: dict,
    evaluate_fn: Callable,
) -> Candidate | None:
    """Train, evaluate, and log the deep candidate as a nested MLflow run."""
    import mlflow

    from src.models.dataset import regression_metrics, seasonal_naive_baseline
    from src.models.pyfunc_wrapper import DemandModelWrapper, build_artifacts

    with mlflow.start_run(run_name="deep", nested=True) as run:
        mlflow.set_tag("model_kind", "deep")
        mlflow.log_params({f"lstm_{k}": v for k, v in (config.get_path("models.lstm", {}) or {}).items()})

        model = DeepDemandModel(dataset.features, config)
        try:
            model.fit(dataset.train, dataset.validation)
        except Exception as exc:  # noqa: BLE001 - a failed candidate must not fail the run
            logger.error("Deep model training failed: %s", exc)
            mlflow.set_tag("training_failed", str(exc))
            return None

        # Scored through predict_panel so the sequence is built from real
        # history rather than reconstructed from lag columns.
        metrics: dict[str, float] = {}
        contexts = {
            "validation": dataset.train,
            # Test draws context from everything chronologically before it.
            "test": pd.concat([dataset.train, dataset.validation], ignore_index=True),
        }
        for split_name, prefix in (("validation", "val"), ("test", "test")):
            split = getattr(dataset, split_name)
            predictions = model.predict_split(split, contexts[split_name])
            usable = predictions.notna()
            if not usable.any():
                logger.error("Deep model produced no usable predictions on %s", split_name)
                return None
            scored = regression_metrics(
                split.loc[usable, TARGET_COLUMN], predictions[usable].to_numpy()
            )
            metrics.update({f"{prefix}_{k}": v for k, v in scored.items()})
            metrics[f"{prefix}_coverage"] = float(usable.mean())

        metrics.update(seasonal_naive_baseline(dataset.test))
        mlflow.log_metrics(metrics)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            weights_path = model.save(tmp_path / "model.pt")
            artifacts = build_artifacts(
                tmp_path,
                model_kind="deep",
                features=dataset.features,
                entity_mapping_path=mapping_path,
                metadata={**run_metadata, "sequence_length": model.sequence_length},
                model_path=weights_path,
            )
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=DemandModelWrapper(),
                artifacts=artifacts,
                code_paths=["src"],
            )

        logger.info(
            "Deep model (in-process): test MAE=%.4f RMSE=%.4f (coverage %.1f%%)",
            metrics["test_mae"], metrics["test_rmse"], 100 * metrics["test_coverage"],
        )

        # Promotion uses the as-served score, not the one above. The two differ
        # sharply for this model: predict_split feeds it true 168-hour
        # sequences, while the pyfunc serving path reconstructs them from five
        # lag columns. Comparing candidates on the in-process number promoted an
        # LSTM that served 60% worse than the LightGBM it displaced.
        from src.models.train import _log_served_metrics

        model_uri = f"runs:/{run.info.run_id}/model"
        served = _log_served_metrics(model_uri, dataset, metrics, "Deep model")
        return Candidate(
            name="deep",
            run_id=run.info.run_id,
            metrics=served,
            model_uri=model_uri,
        )
