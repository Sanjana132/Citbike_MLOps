"""Data and target drift reports via Evidently.

The drift half of the closed loop. ``monitor_dag`` compares the feature
distribution the model is *currently* seeing against the distribution it was
*trained* on; if too large a share of features have drifted, retraining is
triggered.

Why drift detection matters separately from error monitoring: rolling MAE only
tells you something is wrong *after* accuracy has already degraded, and only
where labels exist. Drift can fire earlier, and it fires even for hours whose
realised demand has not been observed yet. The two triggers are complementary,
so ``monitor_dag`` fires on either.

Evidently's Python API has changed substantially across versions, so the report
call is wrapped and the module degrades to "no drift detected, reason recorded"
rather than failing the DAG if the installed version does not match.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Config, data_dir, load_config

logger = logging.getLogger(__name__)

# Columns that are identifiers or targets rather than model inputs.
NON_FEATURE_COLUMNS = {"entity_id", "hour_ts", "departures", "target_hour_ts"}


@dataclass
class DriftReport:
    """Outcome of one drift comparison."""

    drift_share: float
    dataset_drift: bool
    n_features: int
    n_drifted: int
    drifted_features: list[str] = field(default_factory=list)
    reference_rows: int = 0
    current_rows: int = 0
    report_path: str | None = None
    evaluated: bool = True
    reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "drift_share": self.drift_share,
            "dataset_drift": self.dataset_drift,
            "n_features": self.n_features,
            "n_drifted": self.n_drifted,
            "drifted_features": self.drifted_features,
            "reference_rows": self.reference_rows,
            "current_rows": self.current_rows,
            "report_path": self.report_path,
            "evaluated": self.evaluated,
            "reason": self.reason,
        }


def _numeric_features(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if column not in NON_FEATURE_COLUMNS
        and pd.api.types.is_numeric_dtype(frame[column])
    ]


def population_stability_index(
    reference: pd.Series, current: pd.Series, bins: int = 10
) -> float:
    """PSI between two distributions.

    Used as the fallback detector when Evidently is unavailable or its API has
    moved. PSI is the standard rule of thumb in this space: < 0.1 is stable,
    0.1-0.25 is a moderate shift, > 0.25 is a substantial one.
    """
    reference = pd.to_numeric(reference, errors="coerce").dropna()
    current = pd.to_numeric(current, errors="coerce").dropna()
    if len(reference) < 2 or len(current) < 2:
        return 0.0

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    reference_counts = np.histogram(reference, bins=edges)[0].astype(float)
    current_counts = np.histogram(current, bins=edges)[0].astype(float)

    # Smoothing avoids log(0) when a bin is empty on one side.
    reference_share = (reference_counts + 0.5) / (reference_counts.sum() + 0.5 * len(edges))
    current_share = (current_counts + 0.5) / (current_counts.sum() + 0.5 * len(edges))
    return float(np.sum((current_share - reference_share) * np.log(current_share / reference_share)))


def _psi_drift_report(
    reference: pd.DataFrame, current: pd.DataFrame, threshold: float, reason: str
) -> DriftReport:
    """Fallback drift detection using PSI per feature."""
    features = _numeric_features(reference)
    shared = [f for f in features if f in current.columns]
    if not shared:
        return DriftReport(0.0, False, 0, 0, evaluated=False, reason="no shared numeric features")

    drifted = [f for f in shared if population_stability_index(reference[f], current[f]) > 0.25]
    share = len(drifted) / len(shared)
    return DriftReport(
        drift_share=share,
        dataset_drift=share > threshold,
        n_features=len(shared),
        n_drifted=len(drifted),
        drifted_features=drifted,
        reference_rows=len(reference),
        current_rows=len(current),
        reason=reason,
    )


def compute_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    config: Config | None = None,
    *,
    save_html: bool = True,
) -> DriftReport:
    """Compare current feature distributions against the training reference."""
    cfg = config or load_config()
    threshold = float(cfg.get_path("monitoring.drift_share_threshold", 0.3))
    min_rows = int(cfg.get_path("monitoring.min_samples_for_evaluation", 200))

    if reference.empty or current.empty:
        return DriftReport(0.0, False, 0, 0, evaluated=False, reason="empty reference or current")
    if len(current) < min_rows:
        return DriftReport(
            0.0, False, 0, 0,
            reference_rows=len(reference), current_rows=len(current),
            evaluated=False,
            reason=f"insufficient current rows ({len(current)} < {min_rows})",
        )

    # Calendar features are deterministic functions of the timestamp, so a short
    # current window differs from a long reference by construction. Comparing
    # them measures the window sizes, not the data. See the config comment.
    excluded = set(cfg.get_path("monitoring.drift_exclude_features", []) or [])
    features = [
        f
        for f in _numeric_features(reference)
        if f in current.columns and f not in excluded
    ]
    if not features:
        return DriftReport(0.0, False, 0, 0, evaluated=False, reason="no shared numeric features")

    reference_subset = reference[features]
    current_subset = current[features]

    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset

        report = Report(metrics=[DataDriftPreset()])
        result = report.run(reference_data=reference_subset, current_data=current_subset)

        summary = result.dict()
        share, drifted_count, drifted_names = _extract_drift_summary(summary, features)

        report_path = None
        if save_html:
            target = data_dir(
                "reports", f"drift_{datetime.now(tz=UTC):%Y%m%dT%H%M%SZ}.html"
            )
            try:
                result.save_html(str(target))
                report_path = str(target)
            except Exception as exc:  # noqa: BLE001 - the HTML is a nicety, not the signal
                logger.warning("Could not save the Evidently HTML report: %s", exc)

        return DriftReport(
            drift_share=share,
            dataset_drift=share > threshold,
            n_features=len(features),
            n_drifted=drifted_count,
            drifted_features=drifted_names,
            reference_rows=len(reference_subset),
            current_rows=len(current_subset),
            report_path=report_path,
        )
    except Exception as exc:  # noqa: BLE001
        # Evidently's API moves between releases. Rather than fail the DAG, fall
        # back to PSI so the loop keeps a working drift signal.
        logger.warning("Evidently unavailable or incompatible (%s); falling back to PSI", exc)
        return _psi_drift_report(
            reference_subset, current_subset, threshold, f"evidently fallback: {exc}"
        )


def _extract_drift_summary(
    summary: dict, features: list[str]
) -> tuple[float, int, list[str]]:
    """Pull the drift share out of an Evidently result dict.

    The payload shape differs across Evidently versions, so several known
    locations are probed before giving up and counting drifted features by hand.
    """
    metrics = summary.get("metrics", [])
    if not isinstance(metrics, list):
        return 0.0, 0, []

    # Per-column verdicts. Evidently 0.7 emits one
    # "ValueDrift(column=X,method=K-S p_value,threshold=0.05)" entry per feature
    # whose value is the p-value; below the threshold means drifted.
    drifted_names: list[str] = []
    for metric in metrics:
        if not isinstance(metric, dict):
            continue
        name = str(metric.get("metric_name", ""))
        if not name.startswith("ValueDrift"):
            continue
        match = re.search(r"column=([^,)]+)", name)
        value = metric.get("value")
        if not match or not isinstance(value, (int, float)):
            continue
        threshold_match = re.search(r"threshold=([0-9.eE+-]+)", name)
        threshold = float(threshold_match.group(1)) if threshold_match else 0.05
        # p_value methods drift below the threshold; distance methods above it.
        drifted = value < threshold if "p_value" in name else value > threshold
        if drifted:
            drifted_names.append(match.group(1))

    # Aggregate count, which Evidently computes itself.
    for metric in metrics:
        if not isinstance(metric, dict):
            continue
        value = metric.get("value")
        if isinstance(value, dict):
            if "share_of_drifted_columns" in value:
                share = float(value["share_of_drifted_columns"])
                return share, int(round(share * len(features))), drifted_names
            if "share" in value and "count" in value:
                return float(value["share"]), int(value["count"]), drifted_names

    return len(drifted_names) / max(len(features), 1), len(drifted_names), drifted_names


def load_reference_features(config: Config | None = None) -> pd.DataFrame:
    """Rebuild the feature distribution the live model was trained on.

    Uses the training window recorded in the model's own metadata, so the
    reference always corresponds to the model actually in production rather than
    to whatever data happens to be lying around.
    """
    cfg = config or load_config()
    from src.features.build_features import build_feature_panel, feature_columns
    from src.features.station_clusters import aggregate_to_entities
    from src.serving.model_loader import ModelRegistryLoader
    from src.storage.db import read_hourly_departures

    bundle = ModelRegistryLoader(cfg).load()
    train_start = pd.Timestamp(bundle.metadata.get("train_start"))
    train_end = pd.Timestamp(bundle.metadata.get("train_end"))

    window_days = max(1, int((datetime.now() - train_start.to_pydatetime()).days) + 1)
    departures = read_hourly_departures(window_days=window_days, config=cfg)
    if departures.empty:
        return pd.DataFrame()

    in_window = departures[
        (departures["hour_ts"] >= train_start) & (departures["hour_ts"] <= train_end)
    ]
    if in_window.empty:
        return pd.DataFrame()

    from src.ingestion.weather import weather_window_for

    observations = aggregate_to_entities(in_window, bundle.mapping)
    panel = build_feature_panel(
        observations, bundle.mapping.entities, weather_window_for(in_window["hour_ts"], cfg), cfg
    )
    return panel[["entity_id", "hour_ts"] + feature_columns(cfg)]


def load_current_features(
    config: Config | None = None, *, hours: int = 24
) -> pd.DataFrame:
    """Feature rows actually served recently, as persisted at prediction time.

    Reading back what was served - rather than recomputing it - means drift is
    measured on the real inputs the model saw, including any degradation in
    upstream data.
    """
    from src.storage.db import read_features

    end = datetime.now(tz=UTC).replace(tzinfo=None)
    start = end - timedelta(hours=hours)
    return read_features(start, end)


def latest_report_path() -> str | None:
    """Most recent saved Evidently HTML report, for the dashboard to link to."""
    reports_dir = Path(data_dir("reports", "x")).parent
    if not reports_dir.exists():
        return None
    candidates = sorted(reports_dir.glob("drift_*.html"))
    return str(candidates[-1]) if candidates else None
