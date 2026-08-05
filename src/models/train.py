"""Training entrypoint: build dataset -> train candidates -> log to MLflow -> promote.

Run offline (initial model, true labels from trip CSVs)::

    python -m src.models.train

Run against the live proxy-labelled data accumulated in Postgres (retrain DAG)::

    python -m src.models.train --source db --window-days 90

Both paths converge on the same dataset builder, the same feature function, the
same holdout split, and the same champion/challenger comparison. The only
difference is where the labels came from - and that provenance is recorded as an
MLflow tag on every run, because a model trained on inventory-derived proxy
labels should never be silently mistaken for one trained on true trip counts.
"""

from __future__ import annotations

import argparse
import logging
import random
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from src.config import Config, data_dir, load_config
from src.features.station_clusters import fit_entity_mapping
from src.models.dataset import Dataset, build_dataset, regression_metrics, seasonal_naive_baseline
from src.models.lightgbm_model import LightGBMDemandModel
from src.models.promote import Candidate, run_promotion
from src.models.pyfunc_wrapper import DemandModelWrapper, build_artifacts, to_model_input

logger = logging.getLogger(__name__)


def set_seeds(seed: int) -> None:
    """Seed every RNG we might touch, so runs are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


# --------------------------------------------------------------------------
# Data sources
# --------------------------------------------------------------------------

def load_offline_departures(config: Config, *, use_cache: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Historical trip CSVs -> (hourly departures, station information).

    These are **true labels**: each row of the source CSV is a completed trip.
    """
    from src.ingestion.gbfs_client import GBFSClient
    from src.ingestion.trip_csv_loader import load_hourly_departures

    info_cache = data_dir("cache", "station_information.parquet")
    if use_cache and info_cache.exists():
        station_info = pd.read_parquet(info_cache)
    else:
        station_info = GBFSClient(config).fetch_station_information()
        station_info.to_parquet(info_cache, index=False)

    live_short_names = {s for s in station_info["short_name"].dropna().tolist() if s}
    departures = load_hourly_departures(
        config=config, live_station_short_names=live_short_names, use_cache=use_cache
    )
    return departures, station_info


def load_live_departures(config: Config, window_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Proxy departures accumulated by the ingestion DAG -> (departures, station info).

    NOTE: labels here are the **inventory-delta demand proxy**, not true demand.
    """
    from src.storage.db import read_hourly_departures, read_station_info

    departures = read_hourly_departures(window_days=window_days, config=config)
    station_info = read_station_info(config=config)
    return departures, station_info


def load_blended_departures(
    config: Config, window_days: int, *, use_cache: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Historical true labels + live proxy labels, as the brief specifies.

    Where both sources cover the same (station, hour) the historical true count
    wins - there is no reason to prefer a proxy when ground truth exists.
    """
    historical, station_info = load_offline_departures(config, use_cache=use_cache)
    historical = historical.assign(label_source="trip_csv")
    counts = {"historical_rows": len(historical), "live_proxy_rows": 0}

    try:
        live, live_info = load_live_departures(config, window_days)
        if not live.empty:
            live = live.assign(label_source="gbfs_proxy")
            counts["live_proxy_rows"] = len(live)
            if not live_info.empty:
                station_info = (
                    pd.concat([station_info, live_info], ignore_index=True)
                    .drop_duplicates(subset="short_name", keep="first")
                )
            combined = pd.concat([historical, live], ignore_index=True)
            combined = combined.sort_values("label_source").drop_duplicates(
                subset=["station_short_name", "hour_ts"], keep="first"
            )
            return combined.drop(columns="label_source"), station_info, counts
    except Exception as exc:  # noqa: BLE001 - live data is optional for a blend
        logger.warning("Live proxy data unavailable, training on history alone: %s", exc)

    return historical.drop(columns="label_source"), station_info, counts


# --------------------------------------------------------------------------
# Candidate training
# --------------------------------------------------------------------------

def _evaluate(model, dataset: Dataset) -> dict[str, float]:
    """Score a fitted model on validation and test, plus the naive baseline.

    These are *in-process* metrics, measured on the in-memory estimator. They are
    diagnostic only - :func:`score_as_served` produces the numbers that decide
    promotion.
    """
    metrics: dict[str, float] = {}
    for split in ("validation", "test"):
        x, y = dataset.xy(split)
        predictions = model.predict(x)
        for key, value in regression_metrics(y, predictions).items():
            metrics[f"{split.replace('validation', 'val')}_{key}"] = value
    metrics.update(seasonal_naive_baseline(dataset.test))
    return metrics


def score_as_served(model_uri: str, dataset: Dataset) -> dict[str, float]:
    """Score a logged model through the EXACT path that will serve it.

    This exists because measuring one code path and shipping another is how a
    worse model reaches production while its scoreboard says otherwise. It
    happened here: the LSTM was promoted on an in-process holdout MAE of 22.25,
    computed from true 168-hour sequences, but the pyfunc serving path rebuilds
    those sequences from five lag columns and actually scores **39.27** - 60%
    worse than the LightGBM champion it displaced.

    Loading the model back through ``mlflow.pyfunc`` and scoring the same
    holdout closes that gap by construction: whatever number promotion compares
    is a number the serving path can actually deliver.
    """
    import mlflow.pyfunc

    from src.models.pyfunc_wrapper import to_model_input

    served = mlflow.pyfunc.load_model(model_uri)
    x_test, y_test = dataset.xy("test")
    predictions = served.predict(to_model_input(x_test, dataset.features))
    return regression_metrics(y_test, predictions)


def train_lightgbm_candidate(
    dataset: Dataset,
    config: Config,
    mapping_path: Path,
    run_metadata: dict,
) -> Candidate:
    """Train, evaluate, and log the LightGBM candidate as a nested MLflow run."""
    with mlflow.start_run(run_name="lightgbm", nested=True) as run:
        mlflow.set_tag("model_kind", "lightgbm")
        mlflow.log_params(
            {f"lgb_{k}": v for k, v in (config.get_path("models.lightgbm", {}) or {}).items()}
        )

        model = LightGBMDemandModel(dataset.features, config)
        x_train, y_train = dataset.xy("train")
        x_val, y_val = dataset.xy("validation")
        model.fit(x_train, y_train, x_val, y_val)

        metrics = _evaluate(model, dataset)
        mlflow.log_metrics(metrics)
        if model.best_iteration_:
            mlflow.log_metric("best_iteration", model.best_iteration_)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            booster_path = tmp_path / "model.txt"
            assert model.booster is not None
            model.booster.booster_.save_model(str(booster_path))

            importance_path = tmp_path / "feature_importance.csv"
            model.feature_importance().to_csv(importance_path, index=False)
            mlflow.log_artifact(str(importance_path))

            artifacts = build_artifacts(
                tmp_path,
                model_kind="lightgbm",
                features=dataset.features,
                entity_mapping_path=mapping_path,
                metadata={**run_metadata, "best_iteration": model.best_iteration_},
                model_path=booster_path,
            )
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=DemandModelWrapper(),
                artifacts=artifacts,
                # Cast to float64 so the inferred signature declares every
                # feature as a double. With integer columns, MLflow's schema
                # enforcement rejects perfectly valid input that arrived as
                # float - which is exactly what happens to features read back
                # out of the JSONB features table. MLflow warns about this
                # itself at log time.
                input_example=to_model_input(x_val.head(5), dataset.features),
                code_paths=["src"],
            )

        model_uri = f"runs:/{run.info.run_id}/model"
        served = _log_served_metrics(model_uri, dataset, metrics, "LightGBM")
        return Candidate(
            name="lightgbm",
            run_id=run.info.run_id,
            metrics=served,
            model_uri=model_uri,
        )


def _log_served_metrics(
    model_uri: str, dataset: Dataset, in_process: dict[str, float], label: str
) -> dict[str, float]:
    """Score through the serving path, log the comparison, and return it.

    The gap between in-process and as-served metrics is logged explicitly, so a
    candidate whose serving path cannot reproduce its own holdout is visible in
    MLflow rather than silently promoted.
    """
    served = score_as_served(model_uri, dataset)
    mlflow.log_metrics({f"served_{k}": v for k, v in served.items()})

    in_process_mae = in_process.get("test_mae")
    if in_process_mae:
        gap = (served["mae"] - in_process_mae) / in_process_mae
        mlflow.log_metric("serving_gap_ratio", gap)
        if abs(gap) > 0.05:
            logger.warning(
                "%s: as-served MAE %.4f differs from in-process %.4f by %.1f%% - "
                "the serving path does not reproduce the holdout",
                label, served["mae"], in_process_mae, 100 * gap,
            )
    logger.info(
        "%s: as-served test MAE=%.4f RMSE=%.4f (in-process %.4f, baseline %.4f)",
        label, served["mae"], served["rmse"],
        in_process_mae if in_process_mae else float("nan"),
        in_process.get("baseline_mae", float("nan")),
    )
    return served


def train_deep_candidate(
    dataset: Dataset,
    config: Config,
    mapping_path: Path,
    run_metadata: dict,
) -> Candidate | None:
    """Train the sequence model candidate. Returns ``None`` if unavailable.

    A missing torch install must not break the retraining DAG - LightGBM alone
    is still a valid champion.
    """
    try:
        from src.models.deep_model import train_deep_model
    except ImportError as exc:
        logger.warning("Deep model unavailable (%s); skipping this candidate", exc)
        return None
    return train_deep_model(dataset, config, mapping_path, run_metadata, _evaluate)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def run_training(
    source: str = "offline",
    window_days: int | None = None,
    config: Config | None = None,
    *,
    candidates: list[str] | None = None,
    use_cache: bool = True,
    promote: bool = True,
) -> dict:
    """Full training run. Returns a summary dict for the DAG/CLI to report."""
    cfg = config or load_config()
    seed = int(cfg.get_path("project.seed", 42))
    set_seeds(seed)

    mlflow.set_tracking_uri(cfg.require("mlflow.tracking_uri"))
    mlflow.set_experiment(cfg.require("mlflow.experiment_name"))

    window = window_days or int(cfg.get_path("training.rolling_window_days", 90))
    wanted = candidates or list(cfg.get_path("models.candidates", ["lightgbm"]))

    # --- load labels -------------------------------------------------------
    label_counts: dict[str, int] = {}
    if source == "offline":
        departures, station_info = load_offline_departures(cfg, use_cache=use_cache)
        label_provenance = "trip_csv_true_labels"
    elif source == "db":
        departures, station_info = load_live_departures(cfg, window)
        label_provenance = "gbfs_inventory_delta_proxy"
    elif source == "blend":
        departures, station_info, label_counts = load_blended_departures(
            cfg, window, use_cache=use_cache
        )
        label_provenance = "blend_trip_csv_and_gbfs_proxy"
    else:
        raise ValueError(f"Unknown source: {source!r} (expected offline|db|blend)")

    if departures.empty:
        raise RuntimeError(f"No training data available from source={source!r}")

    # --- entities, weather, dataset ---------------------------------------
    stations_in_scope = set(departures["station_short_name"].astype(str).unique())
    mapping = fit_entity_mapping(station_info, cfg, stations_in_scope=stations_in_scope)

    from src.ingestion.weather import weather_window_for

    weather = weather_window_for(departures["hour_ts"], cfg)
    dataset = build_dataset(departures, mapping, weather, cfg)
    summary = dataset.summary()
    logger.info("Dataset: %s", summary)

    run_name = f"{source}-{datetime.now(tz=UTC):%Y%m%dT%H%M%SZ}"
    with mlflow.start_run(run_name=run_name) as parent_run:
        mlflow.set_tags(
            {
                "label_provenance": label_provenance,
                "training_source": source,
                "granularity": mapping.granularity,
                # Loud, machine-readable flag: anything trained on live data is
                # trained on a proxy, and downstream consumers must know.
                "labels_are_proxy": str(source in {"db", "blend"}).lower(),
            }
        )
        mlflow.log_params(
            {
                "source": source,
                "seed": seed,
                "granularity": mapping.granularity,
                "n_clusters": mapping.n_clusters,
                "window_days": window,
                "weather_hours": len(weather),
                **{k: v for k, v in summary.items() if isinstance(v, (int, float, str))},
                **label_counts,
            }
        )

        with tempfile.TemporaryDirectory() as tmp:
            mapping_path = mapping.save(Path(tmp) / "entity_mapping.json")
            mlflow.log_artifact(str(mapping_path))

            run_metadata = {
                "trained_at": datetime.now(tz=UTC).isoformat(),
                "train_start": summary["train_start"],
                "train_end": summary["train_end"],
                "granularity": mapping.granularity,
                "label_provenance": label_provenance,
                "seed": seed,
            }

            trained: list[Candidate] = []
            if "lightgbm" in wanted:
                trained.append(
                    train_lightgbm_candidate(dataset, cfg, mapping_path, run_metadata)
                )
            if "deep" in wanted:
                deep = train_deep_candidate(dataset, cfg, mapping_path, run_metadata)
                if deep is not None:
                    trained.append(deep)

            if not trained:
                raise RuntimeError("No candidate models were trained successfully")

            metric = cfg.get_path("promotion.metric", "mae")
            for candidate in trained:
                mlflow.log_metric(f"candidate_{candidate.name}_{metric}", candidate.score(metric))

            decision = None
            if promote:
                decision = run_promotion(trained, cfg)
                mlflow.set_tag("promoted", str(decision.promote).lower())
                mlflow.set_tag("promotion_reason", decision.reason)

        return {
            "parent_run_id": parent_run.info.run_id,
            "candidates": {c.name: c.metrics for c in trained},
            "dataset": summary,
            "label_provenance": label_provenance,
            "promoted": bool(decision.promote) if decision else False,
            "promotion_reason": decision.reason if decision else "promotion skipped",
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Citi Bike demand models")
    parser.add_argument(
        "--source",
        default="offline",
        choices=["offline", "db", "blend"],
        help="offline=historical trip CSVs (true labels); db=live GBFS proxy; blend=both",
    )
    parser.add_argument("--window-days", type=int, default=None)
    parser.add_argument(
        "--candidates",
        nargs="*",
        default=None,
        help="Subset of models to train, e.g. --candidates lightgbm",
    )
    parser.add_argument("--no-cache", action="store_true", help="Ignore cached raw data")
    parser.add_argument("--no-promote", action="store_true", help="Train without touching the registry")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
    )

    result = run_training(
        source=args.source,
        window_days=args.window_days,
        candidates=args.candidates,
        use_cache=not args.no_cache,
        promote=not args.no_promote,
    )

    print("\n" + "=" * 72)
    print("TRAINING COMPLETE")
    print("=" * 72)
    print(f"Label provenance : {result['label_provenance']}")
    print(f"Dataset          : {result['dataset']}")
    for name, metrics in result["candidates"].items():
        print(
            f"  {name:10s} MAE={metrics['mae']:.4f}  RMSE={metrics['rmse']:.4f}  "
            f"sMAPE={metrics['smape']:.2f}%"
        )
    print(f"Promoted         : {result['promoted']}")
    print(f"Reason           : {result['promotion_reason']}")


if __name__ == "__main__":
    main()
