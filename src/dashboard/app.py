"""Streamlit dashboard.

Four things an operator needs to see at a glance:

1. Which model is live, and what it was trained on.
2. Predicted vs realised demand on a map.
3. Rolling MAE over time, against the model's own training holdout.
4. Whether drift has been detected, and in which features.

Realised demand from the live loop is labelled "realised (proxy)" throughout,
because that is what it is. A dashboard that quietly calls an inventory-delta
estimate "actual demand" would be the single most misleading thing in this
project.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

import pandas as pd
import streamlit as st

from src.config import load_config

logging.basicConfig(level=logging.WARNING)

st.set_page_config(page_title="Citi Bike Demand Monitoring", page_icon="🚲", layout="wide")

config = load_config()
REFRESH_SECONDS = 60


@st.cache_data(ttl=REFRESH_SECONDS)
def load_model_info() -> dict:
    try:
        from src.serving.model_loader import ModelRegistryLoader

        return ModelRegistryLoader(config).load().info()
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


@st.cache_data(ttl=REFRESH_SECONDS)
def load_metrics_history(limit: int = 500) -> pd.DataFrame:
    try:
        from src.storage.db import read_model_metrics

        return read_model_metrics(limit=limit)
    except Exception as exc:  # noqa: BLE001
        st.session_state["metrics_error"] = str(exc)
        return pd.DataFrame()


@st.cache_data(ttl=REFRESH_SECONDS)
def load_predicted_vs_actual(hours: int = 24) -> pd.DataFrame:
    """Join logged predictions to realised (proxy) demand for the map."""
    try:
        from src.monitoring.metrics import join_predictions_to_realised
        from src.serving.model_loader import ModelRegistryLoader
        from src.storage.db import read_hourly_departures, read_predictions

        bundle = ModelRegistryLoader(config).load()
        end = datetime.now(tz=UTC).replace(tzinfo=None)
        start = end - timedelta(hours=hours)

        predictions = read_predictions(start, end)
        if predictions.empty:
            return pd.DataFrame()

        realised = read_hourly_departures(window_days=max(1, hours // 24 + 1), config=config)
        joined = join_predictions_to_realised(predictions, realised, bundle.mapping)
        if joined.empty:
            return joined

        entities = bundle.mapping.entities[["entity_id", "lat", "lon", "n_stations"]]
        return joined.merge(entities, on="entity_id", how="left")
    except Exception as exc:  # noqa: BLE001
        st.session_state["map_error"] = str(exc)
        return pd.DataFrame()


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("🚲 Citi Bike Demand Forecasting")
st.caption(
    "Next-hour demand per station cluster, with automatic drift/error-triggered "
    "retraining."
)

st.warning(
    "**Realised demand shown here is a PROXY.** Live labels are derived from "
    "differences in GBFS bike-inventory between snapshots, not from completed "
    "trips. Rebalancing vans, dock outages and within-interval netting all "
    "distort it, and it is biased low at busy stations. The model's honest "
    "accuracy figure is its training holdout MAE, measured against true "
    "trip-history labels.",
    icon="⚠️",
)

info = load_model_info()

# --------------------------------------------------------------------------
# Model + status row
# --------------------------------------------------------------------------
metrics_history = load_metrics_history()
latest = metrics_history.iloc[0] if not metrics_history.empty else None

col1, col2, col3, col4 = st.columns(4)

with col1:
    if "error" in info:
        st.metric("Production model", "none")
        st.caption(f"⚠️ {info['error'][:80]}")
    else:
        st.metric("Production model", f"v{info['model_version']}")
        st.caption(f"{info.get('model_kind', '?')} · {info.get('granularity', '?')}")

with col2:
    if latest is not None and pd.notna(latest.get("rolling_mae")):
        st.metric("Rolling MAE (proxy)", f"{latest['rolling_mae']:.2f}")
        st.caption(f"over {int(latest['n_samples'])} matched hours")
    else:
        st.metric("Rolling MAE (proxy)", "—")
        st.caption("awaiting matched predictions")

with col3:
    if latest is not None and pd.notna(latest.get("drift_share")):
        drifted = bool(latest.get("dataset_drift"))
        st.metric("Feature drift", f"{latest['drift_share']:.0%}")
        st.caption("🔴 drift detected" if drifted else "🟢 within threshold")
    else:
        st.metric("Feature drift", "—")
        st.caption("not yet evaluated")

with col4:
    if latest is not None and bool(latest.get("breached")):
        st.metric("Loop status", "RETRAIN")
        st.caption("threshold breached")
    else:
        st.metric("Loop status", "stable")
        st.caption("no trigger fired")

if latest is not None and latest.get("breach_reason"):
    st.info(f"**Latest monitoring verdict:** {latest['breach_reason']}")

# --------------------------------------------------------------------------
# Model details
# --------------------------------------------------------------------------
with st.expander("Production model details", expanded=False):
    if "error" in info:
        st.error(info["error"])
    else:
        provenance = info.get("label_provenance", "unknown")
        if provenance and "proxy" in provenance:
            st.warning(
                f"This model was trained on **{provenance}** - its labels include "
                "the inventory-derived proxy, not only true trip counts.",
                icon="⚠️",
            )
        st.json(info)

# --------------------------------------------------------------------------
# Map
# --------------------------------------------------------------------------
st.subheader("Predicted vs realised (proxy) demand")
window_hours = st.slider("Window (hours)", 1, 72, 24, key="map_window")
comparison = load_predicted_vs_actual(window_hours)

if comparison.empty:
    st.info(
        "No matched prediction/actual pairs yet. Predictions accumulate as the "
        "API serves requests, and realised demand as the ingestion DAG runs."
    )
    if "map_error" in st.session_state:
        st.caption(f"({st.session_state['map_error'][:160]})")
else:
    latest_hour = comparison["target_hour_ts"].max()
    snapshot = comparison[comparison["target_hour_ts"] == latest_hour].copy()
    snapshot["error"] = snapshot["predicted"] - snapshot["actual"]
    snapshot["abs_error"] = snapshot["error"].abs()

    map_col, table_col = st.columns([3, 2])
    with map_col:
        try:
            import pydeck as pdk

            layer = pdk.Layer(
                "ScatterplotLayer",
                data=snapshot.dropna(subset=["lat", "lon"]),
                get_position=["lon", "lat"],
                get_radius="60 + abs_error * 12",
                # Red where the model over-predicted, blue where it under-predicted.
                get_fill_color="[error > 0 ? 220 : 40, 90, error > 0 ? 60 : 200, 150]",
                pickable=True,
            )
            st.pydeck_chart(
                pdk.Deck(
                    layers=[layer],
                    initial_view_state=pdk.ViewState(
                        latitude=float(snapshot["lat"].mean()),
                        longitude=float(snapshot["lon"].mean()),
                        zoom=11,
                    ),
                    tooltip={
                        "text": "{entity_id}\nPredicted: {predicted}\nRealised (proxy): {actual}"
                    },
                )
            )
            st.caption(
                f"Hour {latest_hour} · circle size = |error| · "
                "red = over-predicted, blue = under-predicted"
            )
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Map unavailable ({exc}); showing the table instead.")

    with table_col:
        st.dataframe(
            snapshot[["entity_id", "predicted", "actual", "error"]]
            .sort_values("error", key=abs, ascending=False)
            .head(15)
            .round(2),
            hide_index=True,
            use_container_width=True,
        )

    st.subheader("Predicted vs realised (proxy) over time")
    over_time = (
        comparison.groupby("target_hour_ts")[["predicted", "actual"]].sum().reset_index()
    )
    over_time = over_time.rename(
        columns={"predicted": "Predicted", "actual": "Realised (proxy)"}
    )
    st.line_chart(over_time.set_index("target_hour_ts"))

# --------------------------------------------------------------------------
# Rolling MAE
# --------------------------------------------------------------------------
st.subheader("Rolling MAE over time")
if metrics_history.empty or metrics_history["rolling_mae"].isna().all():
    st.info("No monitoring runs recorded yet. monitor_dag writes a row every hour.")
    if "metrics_error" in st.session_state:
        st.caption(f"({st.session_state['metrics_error'][:160]})")
else:
    series = (
        metrics_history.dropna(subset=["rolling_mae"])
        .sort_values("computed_at")
        .set_index("computed_at")[["rolling_mae"]]
        .rename(columns={"rolling_mae": "Rolling MAE (vs proxy)"})
    )
    st.line_chart(series)

    threshold = float(config.get_path("monitoring.mae_threshold", 4.0))
    st.caption(
        f"Absolute retrain threshold: {threshold}. A retrain also fires if the "
        f"rolling MAE exceeds "
        f"{config.get_path('monitoring.mae_degradation_ratio', 1.5)}x the "
        "model's training holdout MAE."
    )

# --------------------------------------------------------------------------
# Drift detail
# --------------------------------------------------------------------------
st.subheader("Drift detail")
if metrics_history.empty or metrics_history["drift_share"].isna().all():
    st.info("No drift evaluations recorded yet.")
else:
    drift_series = (
        metrics_history.dropna(subset=["drift_share"])
        .sort_values("computed_at")
        .set_index("computed_at")[["drift_share"]]
        .rename(columns={"drift_share": "Share of drifted features"})
    )
    st.line_chart(drift_series)
    st.caption(
        f"Retrain threshold: "
        f"{float(config.get_path('monitoring.drift_share_threshold', 0.3)):.0%} "
        "of features drifted."
    )

    try:
        from src.monitoring.drift import latest_report_path

        report = latest_report_path()
        if report:
            st.caption(f"Latest Evidently HTML report: `{report}`")
    except Exception:  # noqa: BLE001
        pass

st.divider()
st.caption(
    "Data: Citi Bike GBFS (live inventory) + trip-history CSVs (true labels) + "
    "Open-Meteo/NOAA weather. Dashboard refreshes every 60s."
)
