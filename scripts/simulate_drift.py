"""Induce drift on purpose, to demonstrate that the loop actually closes.

This is the demo the whole project builds toward: shift the live data, watch
``monitor_dag`` notice, and watch ``retrain_dag`` fire without anyone touching
it.

    # 1. serve some predictions so there is something to score
    python -m scripts.simulate_drift --baseline

    # 2. shift the feature distribution and degrade the predictions
    python -m scripts.simulate_drift --inject --shift 2.5

    # 3. see what monitoring makes of it
    python -m scripts.simulate_drift --check

    # 4. put everything back
    python -m scripts.simulate_drift --reset

``--inject`` writes into ``features``, ``predictions`` and ``hourly_departures``
under a clearly marked model version so the effect can be undone. It is a demo
tool: it fabricates rows and must never be pointed at a real deployment.
"""

from __future__ import annotations

import argparse
import logging
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from src.config import load_config
from src.storage.db import get_engine

logger = logging.getLogger(__name__)

SIMULATED_VERSION = "drift-sim"


def _entities(config):
    from src.serving.model_loader import ModelRegistryLoader

    bundle = ModelRegistryLoader(config).load()
    return bundle, bundle.mapping.entities["entity_id"].tolist()


def emit_baseline(config, hours: int = 30) -> dict:
    """Serve predictions for recent hours and record matching realised demand.

    Gives the monitor a clean, in-threshold baseline before drift is injected,
    so the change is visible rather than being the only data point.
    """
    from src.features.build_features import TIME_KEY, feature_columns
    from src.features.station_clusters import aggregate_to_entities
    from src.ingestion.weather import weather_window_for
    from src.serving.history import load_recent_history
    from src.storage.db import write_features, write_hourly_departures, write_predictions

    bundle, _ = _entities(config)
    history = load_recent_history(config)
    if history.is_empty:
        raise RuntimeError("No history available; run scripts/bootstrap_db.py first")

    observations = aggregate_to_entities(history.departures, bundle.mapping)
    weather = weather_window_for(observations[TIME_KEY], config)

    from src.features.build_features import build_feature_panel

    panel = build_feature_panel(observations, bundle.mapping.entities, weather, config)
    recent = panel[panel[TIME_KEY] > panel[TIME_KEY].max() - pd.Timedelta(hours=hours)].copy()
    recent = recent[recent["lag_168h"].notna()]
    if recent.empty:
        raise RuntimeError("Not enough history to build a baseline window")

    features = feature_columns(config)
    predictions = bundle.model.predict(recent[features])

    write_predictions(
        pd.DataFrame(
            {
                "entity_id": recent["entity_id"].to_numpy(),
                "target_hour_ts": recent[TIME_KEY].to_numpy(),
                "predicted": predictions,
                "model_name": bundle.model_name,
                "model_version": bundle.model_version,
            }
        )
    )
    write_features(recent[["entity_id", TIME_KEY] + features])

    # Realised demand for these hours: the observed values the panel was built
    # from, written back at station granularity so the monitor's own join works.
    realised = history.departures[
        history.departures["hour_ts"].isin(recent[TIME_KEY].unique())
    ].copy()
    write_hourly_departures(
        realised.assign(label_source="gbfs_proxy"), label_source="gbfs_proxy"
    )

    logger.info("Baseline: %d predictions over %d hours", len(recent), recent[TIME_KEY].nunique())
    return {"rows": len(recent), "hours": int(recent[TIME_KEY].nunique())}


def inject_drift(config, shift: float = 2.5, hours: int = 24) -> dict:
    """Shift the served feature distribution and degrade realised demand.

    Simulates a regime change - a heatwave, a transit strike, a new bike-share
    competitor - where demand and its drivers both move away from what the model
    was trained on.
    """
    from src.features.build_features import feature_columns
    from src.storage.db import (
        read_features,
        write_features,
        write_hourly_departures,
        write_predictions,
    )

    bundle, _ = _entities(config)
    end = datetime.now(tz=UTC).replace(tzinfo=None)
    start = end - timedelta(hours=hours * 3)

    existing = read_features(start, end)
    if existing.empty:
        raise RuntimeError("No served features found; run --baseline first")

    recent_hours = sorted(existing["hour_ts"].unique())[-hours:]
    frame = existing[existing["hour_ts"].isin(recent_hours)].copy()

    features = [f for f in feature_columns(config) if f in frame.columns]

    # Move the drivers: hotter, wetter, and materially more demand in the lags.
    for column in ("temperature_c",):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce") + 12.0 * (shift / 2.5)
    for column in ("precipitation_mm",):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce") + 4.0 * (shift / 2.5)
    for column in [c for c in frame.columns if c.startswith(("lag_", "roll_mean"))]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce") * shift

    write_features(frame[["entity_id", "hour_ts"] + features])

    # The model still predicts from the drifted features...
    predictions = bundle.model.predict(frame[features].astype(float))
    write_predictions(
        pd.DataFrame(
            {
                "entity_id": frame["entity_id"].to_numpy(),
                "target_hour_ts": pd.to_datetime(frame["hour_ts"]).to_numpy(),
                "predicted": predictions,
                "model_name": bundle.model_name,
                "model_version": SIMULATED_VERSION,
            }
        )
    )

    # ...while realised demand moves somewhere else entirely, so error rises.
    rng = np.random.default_rng(int(config.get_path("project.seed", 42)))
    stations = _stations_for_entities(config, bundle, frame["entity_id"].unique())
    realised_rows = []
    for hour in recent_hours:
        for station, _entity in stations:
            base = float(rng.poisson(12) * shift)
            realised_rows.append(
                {
                    "station_short_name": station,
                    "hour_ts": pd.Timestamp(hour),
                    "departures": base,
                    "label_source": "gbfs_proxy",
                }
            )
    realised = pd.DataFrame(realised_rows)
    write_hourly_departures(realised, label_source="gbfs_proxy")

    logger.info(
        "Injected drift: shift=%.2f over %d hours, %d feature rows, %d realised rows",
        shift, len(recent_hours), len(frame), len(realised),
    )
    return {"feature_rows": len(frame), "realised_rows": len(realised), "shift": shift}


def _stations_for_entities(config, bundle, entity_ids) -> list[tuple[str, str]]:
    """A few stations per entity, so realised rows aggregate back correctly."""
    wanted = set(entity_ids)
    by_entity: dict[str, list[str]] = {}
    for station, entity in bundle.mapping.station_to_entity.items():
        if entity in wanted:
            by_entity.setdefault(entity, []).append(station)
    return [(s, e) for e, stations in by_entity.items() for s in stations[:4]]


def check(config) -> dict:
    """Run one monitoring cycle and print the verdict."""
    from src.monitoring.pipeline import run_monitoring_cycle

    outcome = run_monitoring_cycle(config)
    print("\n" + "=" * 72)
    print("MONITORING VERDICT")
    print("=" * 72)
    print(f"should_retrain : {outcome.should_retrain}")
    print(f"summary        : {outcome.summary()}")
    print(f"accuracy       : {outcome.accuracy}")
    print(f"drift          : { {k: v for k, v in outcome.drift.items() if k != 'drifted_features'} }")
    if outcome.drift.get("drifted_features"):
        print(f"drifted features: {outcome.drift['drifted_features'][:10]}")
    return {"should_retrain": outcome.should_retrain, "summary": outcome.summary()}


def reset(config) -> dict:
    """Remove everything the simulation wrote."""
    from sqlalchemy import text

    engine = get_engine()
    with engine.begin() as connection:
        deleted_predictions = connection.execute(
            text("DELETE FROM predictions WHERE model_version = :v"), {"v": SIMULATED_VERSION}
        ).rowcount
        deleted_proxy = connection.execute(
            text("DELETE FROM hourly_departures WHERE label_source = 'gbfs_proxy'")
        ).rowcount
        deleted_features = connection.execute(text("DELETE FROM features")).rowcount
        deleted_metrics = connection.execute(text("DELETE FROM model_metrics")).rowcount
    logger.info(
        "Reset: %d predictions, %d proxy departures, %d feature rows, %d metric rows",
        deleted_predictions, deleted_proxy, deleted_features, deleted_metrics,
    )
    return {
        "predictions": deleted_predictions,
        "proxy_departures": deleted_proxy,
        "features": deleted_features,
        "metrics": deleted_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate drift to demo the closed loop")
    parser.add_argument("--baseline", action="store_true", help="Emit clean predictions first")
    parser.add_argument("--inject", action="store_true", help="Shift the distribution")
    parser.add_argument("--check", action="store_true", help="Run one monitoring cycle")
    parser.add_argument("--reset", action="store_true", help="Delete simulated rows")
    parser.add_argument("--shift", type=float, default=2.5, help="Multiplier for the shift")
    parser.add_argument("--hours", type=int, default=24)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s")
    config = load_config()

    if not any([args.baseline, args.inject, args.check, args.reset]):
        parser.error("Choose at least one of --baseline / --inject / --check / --reset")

    if args.reset:
        print(reset(config))
    if args.baseline:
        print(emit_baseline(config, hours=args.hours + 6))
    if args.inject:
        print(inject_drift(config, shift=args.shift, hours=args.hours))
    if args.check:
        check(config)


if __name__ == "__main__":
    main()
