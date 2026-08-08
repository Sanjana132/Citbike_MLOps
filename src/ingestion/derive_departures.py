"""Derive a DEMAND PROXY from GBFS inventory deltas.

READ THIS BEFORE TRUSTING ANY NUMBER THIS MODULE PRODUCES
=========================================================

GBFS ``station_status`` reports **inventory** - how many bikes are docked at a
station right now. It does **not** report **demand** - how many rides were
taken. This module estimates the latter from the former by differencing
consecutive snapshots:

    proxy_departures(t) = max(0, num_bikes_available(t-1) - num_bikes_available(t))

That quantity is a *net outflow*, and it is wrong in several knowable ways:

1. **Netting.** Within one polling interval a station can lose 4 bikes to riders
   and gain 3 from returns; the proxy sees a net drop of 1 and reports 1
   departure instead of 4. The proxy therefore **systematically underestimates**
   departures, and it does so worst at exactly the busy stations that matter
   most. Shorter polling intervals reduce but never eliminate this.

2. **Rebalancing.** Lyft trucks add and remove bikes in bulk. A van taking 15
   bikes looks identical to 15 people renting them. Capped, not solved (see
   ``max_plausible_drop_per_interval``).

3. **Dock and reporting outages.** A station that stops reporting, goes
   offline, or has its bikes marked disabled produces inventory jumps that are
   not rides at all. Filtered on ``is_renting`` / ``is_installed`` / staleness.

4. **E-bike churn.** Bikes with dead batteries flip between available and
   disabled without anyone riding them.

What this means in practice: the live loop's labels are **noisy and biased low**.
That is acceptable for *drift and degradation detection*, which is what they are
used for. It is why the initial model is trained on historical trip CSVs, where
every label is a real completed trip. Any model retrained on these proxy labels
is tagged ``labels_are_proxy=true`` in MLflow.

Nothing in this file produces ground truth, and nothing downstream should
describe its output as "actual demand". The dashboard labels it
"realised (proxy)" for the same reason.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import Config, load_config

logger = logging.getLogger(__name__)

PROXY_COLUMNS = ["station_id", "interval_end", "proxy_departures", "interval_minutes"]


def derive_interval_departures(
    snapshots: pd.DataFrame, config: Config | None = None
) -> pd.DataFrame:
    """Difference consecutive snapshots into per-interval proxy departures.

    Parameters
    ----------
    snapshots
        Rows from ``status_snapshots``: at minimum ``station_id``,
        ``snapshot_ts``, ``num_bikes_available``, and ideally the
        ``is_renting`` / ``is_installed`` / ``last_reported`` guard columns.

    Returns
    -------
    ``station_id``, ``interval_end``, ``proxy_departures``, ``interval_minutes``
    - one row per usable consecutive snapshot pair. Pairs failing any
    correction rule are dropped rather than guessed at.
    """
    cfg = config or load_config()
    rules = cfg.get_path("ingestion.proxy", {}) or {}
    require_renting = bool(rules.get("require_is_renting", True))
    require_installed = bool(rules.get("require_is_installed", True))
    max_drop = float(rules.get("max_plausible_drop_per_interval", 10))
    max_gap_minutes = float(rules.get("max_interval_gap_minutes", 20))
    max_staleness_minutes = float(rules.get("max_staleness_minutes", 30))

    if snapshots.empty:
        return pd.DataFrame(columns=PROXY_COLUMNS)

    frame = snapshots.copy()
    frame["snapshot_ts"] = pd.to_datetime(frame["snapshot_ts"], utc=True, errors="coerce")
    frame["num_bikes_available"] = pd.to_numeric(
        frame["num_bikes_available"], errors="coerce"
    )
    frame = frame.dropna(subset=["station_id", "snapshot_ts", "num_bikes_available"])
    if frame.empty:
        return pd.DataFrame(columns=PROXY_COLUMNS)

    frame = frame.sort_values(["station_id", "snapshot_ts"]).reset_index(drop=True)
    grouped = frame.groupby("station_id", observed=True)

    frame["prev_bikes"] = grouped["num_bikes_available"].shift(1)
    frame["prev_ts"] = grouped["snapshot_ts"].shift(1)
    for column in ("is_renting", "is_installed"):
        if column in frame.columns:
            frame[f"prev_{column}"] = grouped[column].shift(1)

    pairs = frame.dropna(subset=["prev_bikes", "prev_ts"]).copy()
    if pairs.empty:
        return pd.DataFrame(columns=PROXY_COLUMNS)

    total_pairs = len(pairs)
    pairs["interval_minutes"] = (
        pairs["snapshot_ts"] - pairs["prev_ts"]
    ).dt.total_seconds() / 60.0

    reasons: dict[str, int] = {}

    def drop(mask: pd.Series, reason: str) -> pd.DataFrame:
        """Drop rows failing a rule, recording how many and why."""
        nonlocal pairs
        removed = int((~mask).sum())
        if removed:
            reasons[reason] = reasons.get(reason, 0) + removed
        return pairs[mask]

    # Correction 1: the interval must be a plausible single polling gap.
    # A missed poll means the delta spans an unknown span of time and cannot be
    # attributed to one interval of demand.
    pairs = drop(
        (pairs["interval_minutes"] > 0) & (pairs["interval_minutes"] <= max_gap_minutes),
        "implausible_interval_length",
    )

    # Correction 2: both ends of the pair must have been rentable. Deltas across
    # an outage are dock state changes, not rides.
    if require_renting and "is_renting" in pairs.columns:
        pairs = drop(
            (pairs["is_renting"].fillna(0).astype(int) == 1)
            & (pairs.get("prev_is_renting", pd.Series(1, index=pairs.index)).fillna(0).astype(int) == 1),
            "station_not_renting",
        )
    if require_installed and "is_installed" in pairs.columns:
        pairs = drop(
            (pairs["is_installed"].fillna(0).astype(int) == 1)
            & (pairs.get("prev_is_installed", pd.Series(1, index=pairs.index)).fillna(0).astype(int) == 1),
            "station_not_installed",
        )

    # Correction 3: stale reporters. If the station's own last_reported is far
    # behind the snapshot time, its inventory figure is not current and the
    # delta is meaningless.
    if "last_reported" in pairs.columns:
        last_reported = pd.to_datetime(pairs["last_reported"], utc=True, errors="coerce")
        staleness = (pairs["snapshot_ts"] - last_reported).dt.total_seconds() / 60.0
        pairs = drop(
            staleness.isna() | (staleness <= max_staleness_minutes),
            "stale_last_reported",
        )

    if pairs.empty:
        logger.warning("All %d snapshot pairs were filtered out: %s", total_pairs, reasons)
        return pd.DataFrame(columns=PROXY_COLUMNS)

    # The proxy itself: a net DECREASE in inventory approximates bikes taken.
    # Increases are returns, which this target does not model, so they clamp to 0.
    delta = pairs["prev_bikes"] - pairs["num_bikes_available"]
    proxy = delta.clip(lower=0)

    # Correction 4: cap implausible single-interval drops. Losing more than
    # `max_drop` bikes in five minutes is a rebalancing van far more often than
    # it is a crowd. Capping (rather than dropping) keeps the busy-station
    # signal while bounding the damage; the count is logged.
    capped = int((proxy > max_drop).sum())
    if capped:
        reasons["capped_implausible_drop"] = capped
    proxy = proxy.clip(upper=max_drop)

    pairs["proxy_departures"] = proxy.astype(float)
    kept = len(pairs)
    logger.info(
        "Proxy derivation: %d/%d snapshot pairs usable (%.1f%%); corrections: %s",
        kept, total_pairs, 100.0 * kept / max(total_pairs, 1), reasons or "none",
    )
    return (
        pairs.rename(columns={"snapshot_ts": "interval_end"})[PROXY_COLUMNS]
        .reset_index(drop=True)
    )


def aggregate_proxy_to_hourly(
    interval_departures: pd.DataFrame,
    station_info: pd.DataFrame,
    config: Config | None = None,
    *,
    min_coverage: float = 0.5,
) -> pd.DataFrame:
    """Roll interval-level proxy departures up to hourly counts per station.

    Joins ``station_id`` -> ``short_name`` so the output shares a key space with
    the historical trip data.

    Hours with poor snapshot coverage are dropped: summing three surviving
    5-minute intervals into an "hourly" total would badly understate demand and
    silently poison the training set. ``min_coverage`` is the fraction of the
    hour's expected intervals that must be present.
    """
    cfg = config or load_config()
    if interval_departures.empty:
        return pd.DataFrame(columns=["station_short_name", "hour_ts", "departures", "n_intervals"])

    # Expected intervals are what this configuration is DESIGNED to collect:
    # samples_per_run x runs_per_hour. Two wrong denominators were tried first
    # and both distort the guard:
    #
    #   * runs_per_hour alone ignores multi-sampling and understates ~4x, so
    #     badly-covered hours sail through.
    #   * 3600/spacing is the feed's maximum rate, not ours. We sample in bursts
    #     with gaps between runs, so a perfectly healthy hour scores ~40/60 and
    #     gets discarded for missing samples it was never going to take.
    poll_minutes = float(cfg.get_path("ingestion.poll_interval_minutes", 5))
    samples_per_run = float(cfg.get_path("ingestion.samples_per_run", 1) or 1)
    runs_per_hour = max(1.0, 60.0 / max(poll_minutes, 1e-6))
    expected_intervals = max(1.0, samples_per_run * runs_per_hour)

    frame = interval_departures.copy()
    frame["interval_end"] = pd.to_datetime(frame["interval_end"], utc=True)
    # Hourly buckets are local time, matching the historical trip aggregation
    # and the weather join.
    local_tz = cfg.get_path("project.timezone", "America/New_York")
    frame["hour_ts"] = (
        frame["interval_end"].dt.tz_convert(local_tz).dt.tz_localize(None).dt.floor("h")
    )

    hourly = (
        frame.groupby(["station_id", "hour_ts"], observed=True)
        .agg(departures=("proxy_departures", "sum"), n_intervals=("proxy_departures", "size"))
        .reset_index()
    )

    coverage = hourly["n_intervals"] / expected_intervals
    dropped = int((coverage < min_coverage).sum())
    if dropped:
        logger.warning(
            "Dropping %d station-hours with <%.0f%% snapshot coverage", dropped, min_coverage * 100
        )
    hourly = hourly[coverage >= min_coverage]

    lookup = station_info.dropna(subset=["short_name"])[["station_id", "short_name"]]
    merged = hourly.merge(lookup, on="station_id", how="inner")
    merged = merged.rename(columns={"short_name": "station_short_name"})

    return merged[["station_short_name", "hour_ts", "departures", "n_intervals"]].reset_index(
        drop=True
    )


def proxy_quality_report(interval_departures: pd.DataFrame) -> dict[str, float]:
    """Summary statistics used by the dashboard to keep the caveat visible."""
    if interval_departures.empty:
        return {"n_intervals": 0}
    proxy = interval_departures["proxy_departures"]
    return {
        "n_intervals": int(len(proxy)),
        "zero_share": float((proxy == 0).mean()),
        "mean_proxy_departures": float(proxy.mean()),
        "p99_proxy_departures": float(np.percentile(proxy, 99)),
        "n_stations": int(interval_departures["station_id"].nunique()),
    }
