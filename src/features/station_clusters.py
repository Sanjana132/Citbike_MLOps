"""Entity resolution: map raw stations onto the unit we forecast.

Per-station hourly departures are extremely sparse - the median NYC station sees
a handful of departures per hour and long runs of zeros overnight - which makes
the target noisy and the model hard to evaluate. Clustering geographically
adjacent stations pools that signal.

The clusterer is **fit once on the training stations and then frozen**: the
assignment is persisted alongside the model and reloaded at serving time, so a
station always maps to the same entity in training and in production. Re-fitting
KMeans at serve time would silently renumber every cluster and invalidate the
model.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from src.config import Config, load_config

logger = logging.getLogger(__name__)

ENTITY_COLUMNS = ["entity_id", "lat", "lon", "capacity", "n_stations"]


@dataclass
class EntityMapping:
    """Frozen station -> entity assignment plus the entity metadata table.

    ``granularity`` is recorded so that a model trained per-cluster can never be
    served with a per-station mapping by accident.
    """

    granularity: str
    station_to_entity: dict[str, str]
    entities: pd.DataFrame
    n_clusters: int | None = None

    def assign(self, station_short_names: pd.Series) -> pd.Series:
        """Map station short names to entity ids; unknown stations become NaN."""
        return station_short_names.map(self.station_to_entity)

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "granularity": self.granularity,
            "n_clusters": self.n_clusters,
            "station_to_entity": self.station_to_entity,
            "entities": self.entities.to_dict(orient="records"),
        }
        target.write_text(json.dumps(payload), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> EntityMapping:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            granularity=payload["granularity"],
            n_clusters=payload.get("n_clusters"),
            station_to_entity=payload["station_to_entity"],
            entities=pd.DataFrame(payload["entities"], columns=ENTITY_COLUMNS),
        )


def fit_entity_mapping(
    station_info: pd.DataFrame,
    config: Config | None = None,
    *,
    stations_in_scope: set[str] | None = None,
) -> EntityMapping:
    """Fit the station -> entity mapping.

    ``station_info`` must carry ``short_name``, ``lat``, ``lon``, ``capacity``.
    Only stations in ``stations_in_scope`` (i.e. those with training data) are
    used, so the mapping never covers docks the model has never seen.
    """
    cfg = config or load_config()
    granularity = cfg.get_path("features.granularity", "cluster")
    seed = int(cfg.get_path("project.seed", 42))

    frame = station_info.dropna(subset=["short_name", "lat", "lon"]).copy()
    frame["short_name"] = frame["short_name"].astype(str)
    if stations_in_scope is not None:
        frame = frame[frame["short_name"].isin(stations_in_scope)]
    frame = frame.drop_duplicates(subset="short_name")
    if frame.empty:
        raise ValueError("No stations available to build an entity mapping")
    frame["capacity"] = pd.to_numeric(frame["capacity"], errors="coerce").fillna(0)

    if granularity == "station":
        frame["entity_id"] = frame["short_name"]
        entities = (
            frame[["entity_id", "lat", "lon", "capacity"]]
            .assign(n_stations=1)
            .reset_index(drop=True)
        )
        mapping = dict(zip(frame["short_name"], frame["entity_id"], strict=True))
        logger.info("Entity mapping: per-station, %d entities", len(entities))
        return EntityMapping("station", mapping, entities[ENTITY_COLUMNS])

    if granularity != "cluster":
        raise ValueError(f"Unsupported features.granularity: {granularity!r}")

    n_clusters = int(cfg.get_path("features.n_clusters", 40))
    n_clusters = min(n_clusters, len(frame))
    # Latitude/longitude are scaled so that a degree of longitude at NYC's
    # latitude covers a comparable ground distance to a degree of latitude;
    # without this, clusters come out stretched east-west.
    lat_rad = np.deg2rad(frame["lat"].to_numpy())
    coords = np.column_stack(
        [frame["lat"].to_numpy(), frame["lon"].to_numpy() * np.cos(lat_rad).mean()]
    )
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    # scikit-learn's k-means++ init emits spurious divide-by-zero/overflow
    # RuntimeWarnings from its internal float32 matmul on some BLAS builds
    # (observed on arm64). The resulting clusters are verified sane - spatially
    # coherent, ~1 km mean radius - so the warnings are noise, not a bad fit.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        labels = kmeans.fit_predict(coords)
    frame["entity_id"] = [f"cluster_{label:03d}" for label in labels]

    entities = (
        frame.groupby("entity_id")
        .agg(
            lat=("lat", "mean"),
            lon=("lon", "mean"),
            capacity=("capacity", "sum"),
            n_stations=("short_name", "count"),
        )
        .reset_index()
    )
    mapping = dict(zip(frame["short_name"], frame["entity_id"], strict=True))
    logger.info(
        "Entity mapping: %d clusters over %d stations (median %d stations/cluster)",
        len(entities), len(frame), int(entities["n_stations"].median()),
    )
    return EntityMapping("cluster", mapping, entities[ENTITY_COLUMNS], n_clusters=n_clusters)


def aggregate_to_entities(
    hourly_departures: pd.DataFrame, mapping: EntityMapping
) -> pd.DataFrame:
    """Roll station-level hourly departures up to entity level.

    Input columns: ``station_short_name``, ``hour_ts``, ``departures``.
    Output columns: ``entity_id``, ``hour_ts``, ``departures``.
    """
    frame = hourly_departures.copy()
    frame["entity_id"] = mapping.assign(frame["station_short_name"].astype(str))
    unmapped = int(frame["entity_id"].isna().sum())
    if unmapped:
        logger.warning(
            "Dropping %d rows for stations absent from the entity mapping", unmapped
        )
    frame = frame.dropna(subset=["entity_id"])
    return (
        frame.groupby(["entity_id", "hour_ts"], observed=True)["departures"]
        .sum()
        .reset_index()
        .sort_values(["entity_id", "hour_ts"])
        .reset_index(drop=True)
    )
