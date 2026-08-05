"""FastAPI prediction service.

Endpoints
---------
``GET  /health``     - liveness plus a frank account of degraded state.
``GET  /model-info`` - which model is live, when it was trained, on what labels.
``POST /predict``    - next-hour demand forecast per entity.
``POST /reload``     - re-resolve the production alias after a promotion.

Every prediction is built with :func:`src.features.build_features.build_feature_panel`
- the same function used in training. The API does not own a second copy of the
feature logic, which is the whole point.

Served predictions are logged to Postgres so ``monitoring/metrics.py`` can later
join them against realised demand and compute rolling error. A logging failure
never fails the request: forecasting is the service's job, bookkeeping is not.
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.config import load_config
from src.features.build_features import TIME_KEY, feature_columns
from src.features.station_clusters import aggregate_to_entities
from src.serving.history import load_recent_history
from src.serving.model_loader import ModelRegistryLoader

logger = logging.getLogger(__name__)

config = load_config()
loader = ModelRegistryLoader(config)


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------

class PredictRequest(BaseModel):
    station_ids: list[str] | None = Field(
        default=None,
        description="GBFS short_names (e.g. '5506.14'). Resolved to their entity.",
    )
    entity_ids: list[str] | None = Field(
        default=None, description="Entity ids (cluster_000...) to forecast."
    )
    target_hour: datetime | None = Field(
        default=None,
        description=(
            "Hour to forecast. Defaults to the hour after the most recent "
            "observation, which is the furthest ahead the lags actually support."
        ),
    )
    log_prediction: bool = Field(
        default=True, description="Persist the prediction for later scoring."
    )


class Prediction(BaseModel):
    entity_id: str
    predicted_departures: float
    lat: float | None = None
    lon: float | None = None
    n_stations: int | None = None


class PredictResponse(BaseModel):
    # "model_name"/"model_version" collide with pydantic's protected "model_"
    # namespace; the field names are worth keeping, so the guard is disabled.
    model_config = ConfigDict(protected_namespaces=())

    target_hour: datetime
    predictions: list[Prediction]
    model_name: str
    model_version: str
    history_source: str
    latest_observed_hour: datetime | None
    history_lag_hours: float | None = Field(
        default=None,
        description=(
            "Hours between the latest observation and now. Large values mean "
            "the lag features are stale and the forecast is weaker than its "
            "backtest suggests."
        ),
    )
    warnings: list[str] = []


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
    model_version: str | None = None
    history_source: str | None = None
    warnings: list[str] = []
    detail: str | None = None


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the production model at startup.

    A missing model is logged, not fatal: the container should come up and
    report unhealthy so an operator can see why, rather than crash-looping
    before anyone can read the logs.
    """
    try:
        loader.load()
    except Exception as exc:  # noqa: BLE001
        logger.error("Startup model load failed; serving will report unhealthy: %s", exc)
    yield


app = FastAPI(
    title="Citi Bike Demand Forecasting API",
    version="0.1.0",
    description=(
        "Next-hour bike demand forecasts. Live monitoring labels are derived "
        "from GBFS inventory deltas and are a PROXY for demand, not true ride "
        "counts - see the project README."
    ),
    lifespan=lifespan,
)


def _resolve_target_hour(requested: datetime | None, latest_observed: pd.Timestamp | None) -> tuple[pd.Timestamp, list[str]]:
    """Pick the hour to forecast and report anything unusual about it.

    A request far beyond the observed history is rejected outright. The panel
    builder would happily extend the grid to any future hour, but every lag
    would be NaN and the model would return a confident-looking number derived
    from nothing.
    """
    warnings: list[str] = []
    if requested is not None:
        target = (
            pd.Timestamp(requested).tz_convert(None)
            if requested.tzinfo
            else pd.Timestamp(requested)
        ).floor("h")
        if latest_observed is not None:
            horizon = int(config.get_path("serving.max_forecast_horizon_hours", 24))
            hours_ahead = (target - pd.Timestamp(latest_observed)).total_seconds() / 3600.0
            if hours_ahead > horizon:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"Target hour {target} is {hours_ahead:.0f}h beyond the latest "
                        f"observation ({latest_observed}); this model forecasts at most "
                        f"{horizon}h ahead. Ingest more recent data or request a nearer hour."
                    ),
                )
            if hours_ahead > 1:
                warnings.append(
                    f"Forecasting {hours_ahead:.0f}h ahead; the model is trained for "
                    "the next hour, so accuracy degrades with distance."
                )
        return target, warnings

    if latest_observed is None:
        target = pd.Timestamp(datetime.now(tz=UTC)).tz_localize(None).floor("h") + pd.Timedelta(hours=1)
        warnings.append("No observed history; lag features will be missing.")
        return target, warnings

    return pd.Timestamp(latest_observed).floor("h") + pd.Timedelta(hours=1), warnings


_WEATHER_CACHE: dict[str, tuple[float, pd.DataFrame]] = {}
_WEATHER_LOCK = threading.Lock()


def _load_weather(history_hours: pd.Series, target_hour: pd.Timestamp) -> pd.DataFrame:
    """Weather covering both the history window and the forecast hour.

    Cached for ``serving.weather_cache_seconds``. The uncached version made a
    live HTTP call to NOAA (or Open-Meteo) on *every* prediction, which cost
    more than the model inference itself. Hourly forecasts do not change between
    requests a few seconds apart, so paying that round trip each time bought
    nothing.
    """
    ttl = float(config.get_path("serving.weather_cache_seconds", 900))
    key = f"{pd.Timestamp(history_hours.min()):%Y%m%d}-{target_hour:%Y%m%d%H}"
    if ttl > 0:
        with _WEATHER_LOCK:
            entry = _WEATHER_CACHE.get(key)
        if entry and (time.monotonic() - entry[0]) < ttl:
            return entry[1]

    frame = _fetch_weather(history_hours, target_hour)
    if ttl > 0 and not frame.empty:
        with _WEATHER_LOCK:
            _WEATHER_CACHE[key] = (time.monotonic(), frame)
    return frame


def _fetch_weather(history_hours: pd.Series, target_hour: pd.Timestamp) -> pd.DataFrame:
    """Uncached weather fetch: archive for the history window, live for ahead."""
    from src.ingestion.weather import empty_weather_frame, load_live_weather, weather_window_for

    frames = []
    try:
        frames.append(weather_window_for(history_hours, config))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Historical weather unavailable: %s", exc)
    try:
        frames.append(load_live_weather(config))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Live weather unavailable: %s", exc)

    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return empty_weather_frame()
    combined = pd.concat(frames, ignore_index=True)
    return combined.drop_duplicates(subset="hour_ts", keep="last").sort_values("hour_ts")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness and an honest description of any degraded state."""
    bundle = loader.bundle
    if bundle is None:
        # Startup may have raced ahead of MLflow, or the first model may have
        # been promoted after this container booted. Retry cheaply: a cached
        # bundle short-circuits, so this is just an alias lookup.
        try:
            bundle = loader.load()
        except Exception:  # noqa: BLE001 - reported below as unhealthy
            bundle = None

    warnings: list[str] = []

    history = load_recent_history(config)
    if history.source == "parquet_cache":
        warnings.append(
            "Serving lag features from the local parquet cache, not the database."
        )
    elif history.source == "postgres_stale":
        warnings.append(
            "No recent history in the database; serving from the newest stored slice."
        )
    elif history.source == "none":
        warnings.append("No demand history available; predictions will be unreliable.")

    if bundle is None:
        return HealthResponse(
            status="degraded",
            model_loaded=False,
            history_source=history.source,
            warnings=warnings,
            detail=loader.last_error or "No production model loaded",
        )

    return HealthResponse(
        status="ok" if not warnings else "degraded",
        model_loaded=True,
        model_version=bundle.model_version,
        history_source=history.source,
        warnings=warnings,
    )


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    """Which model is live, trained when, on which kind of labels."""
    bundle = loader.bundle
    if bundle is None:
        try:
            bundle = loader.load()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    return bundle.info()


@app.post("/reload")
def reload_model() -> dict[str, Any]:
    """Re-resolve the production alias. Called after a promotion."""
    try:
        bundle = loader.load(force=True)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"reloaded": True, **bundle.info()}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest, background: BackgroundTasks) -> PredictResponse:
    bundle = loader.bundle
    if bundle is None:
        try:
            bundle = loader.load()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=503, detail=f"No production model available: {exc}"
            ) from exc

    history = load_recent_history(config)
    if history.is_empty:
        raise HTTPException(
            status_code=503,
            detail=(
                "No demand history available, so lag features cannot be built. "
                "Run scripts/bootstrap_db.py or wait for the ingestion DAG."
            ),
        )

    warnings: list[str] = []
    if history.source == "parquet_cache":
        warnings.append("Lag features came from the local parquet cache, not the database.")
    elif history.source == "postgres_stale":
        warnings.append(
            "No recent history in the database; lag features came from the newest stored slice."
        )

    target_hour, target_warnings = _resolve_target_hour(request.target_hour, history.latest_hour)
    warnings.extend(target_warnings)

    # Aggregate raw station history onto the entities this model was trained on.
    observations = aggregate_to_entities(history.departures, bundle.mapping)
    if observations.empty:
        raise HTTPException(
            status_code=503,
            detail="No history maps onto this model's entities; the model may be stale.",
        )

    weather = _load_weather(observations[TIME_KEY], target_hour)

    from src.features.build_features import build_feature_panel

    try:
        panel = build_feature_panel(
            observations,
            bundle.mapping.entities,
            weather,
            config,
            extra_hours=pd.DatetimeIndex([target_hour]),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Feature construction failed")
        raise HTTPException(status_code=500, detail=f"Feature build failed: {exc}") from exc

    rows = panel[panel[TIME_KEY] == target_hour]
    if rows.empty:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Target hour {target_hour} is outside the range features can be "
                f"built for (latest observation: {history.latest_hour})."
            ),
        )

    # Narrow to whatever the caller asked for.
    wanted: set[str] | None = None
    if request.entity_ids:
        wanted = set(request.entity_ids)
    if request.station_ids:
        resolved = {
            bundle.mapping.station_to_entity[s]
            for s in request.station_ids
            if s in bundle.mapping.station_to_entity
        }
        unknown = [s for s in request.station_ids if s not in bundle.mapping.station_to_entity]
        if unknown:
            warnings.append(f"Unknown station ids ignored: {unknown[:5]}")
        wanted = resolved if wanted is None else wanted | resolved
    if wanted is not None:
        rows = rows[rows["entity_id"].isin(wanted)]
        if rows.empty:
            raise HTTPException(status_code=404, detail="No requested entity is known to this model")

    from src.models.pyfunc_wrapper import to_model_input

    features = to_model_input(rows, feature_columns(config), bundle.model)
    try:
        predicted = bundle.model.predict(features)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Model inference failed")
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    result = rows[["entity_id", "lat", "lon", "n_stations"]].copy()
    result["predicted_departures"] = [round(float(p), 3) for p in predicted]

    history_lag_hours = None
    if history.latest_hour is not None:
        now = pd.Timestamp(datetime.now(tz=UTC)).tz_localize(None)
        history_lag_hours = round((now - pd.Timestamp(history.latest_hour)).total_seconds() / 3600.0, 2)
        # 3 hours is roughly where the 1h/2h/3h lags stop being informative.
        if history_lag_hours > 3:
            warnings.append(
                f"Most recent observation is {history_lag_hours:.1f}h old; lag "
                "features are stale and accuracy will be below backtest levels."
            )

    if request.log_prediction:
        # Off the response path: bookkeeping for the monitoring loop must not
        # make callers wait on two database writes.
        background.add_task(_log_prediction, result, rows, target_hour, bundle)

    return PredictResponse(
        target_hour=target_hour.to_pydatetime(),
        predictions=[
            Prediction(
                entity_id=row["entity_id"],
                predicted_departures=row["predicted_departures"],
                lat=None if pd.isna(row["lat"]) else float(row["lat"]),
                lon=None if pd.isna(row["lon"]) else float(row["lon"]),
                n_stations=None if pd.isna(row["n_stations"]) else int(row["n_stations"]),
            )
            for _, row in result.iterrows()
        ],
        model_name=bundle.model_name,
        model_version=bundle.model_version,
        history_source=history.source,
        latest_observed_hour=(
            history.latest_hour.to_pydatetime() if history.latest_hour is not None else None
        ),
        history_lag_hours=history_lag_hours,
        warnings=warnings,
    )


def _log_prediction(
    result: pd.DataFrame, rows: pd.DataFrame, target_hour: pd.Timestamp, bundle
) -> None:
    """Persist predictions and their feature rows for later scoring.

    Best-effort by design - a database hiccup must not turn a good forecast into
    a 500.
    """
    try:
        from src.storage.db import write_features, write_predictions

        write_predictions(
            pd.DataFrame(
                {
                    "entity_id": result["entity_id"],
                    "target_hour_ts": target_hour,
                    "predicted": result["predicted_departures"],
                    "model_name": bundle.model_name,
                    "model_version": bundle.model_version,
                }
            )
        )
        write_features(rows[["entity_id", TIME_KEY] + feature_columns(config)])
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not log predictions (continuing): %s", exc)
