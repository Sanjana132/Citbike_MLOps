# Citi Bike Demand Forecasting — a closed-loop MLOps platform

Next-hour bike-demand forecasts for the NYC Citi Bike system, with a retraining
loop that closes itself: when accuracy degrades or the data drifts, the system
retrains, validates the challenger against the champion, and promotes the winner
without a human in the path.

---

## Architecture

```
                        ┌──────────────────────────────────────────┐
                        │  DATA SOURCES                            │
                        │  • GBFS station_status  (live inventory)  │
                        │  • trip-history CSVs    (true labels)     │
                        │  • Open-Meteo / NOAA    (weather)         │
                        └───────────────┬──────────────────────────┘
                                        │
        ┌───────────────────────────────┼────────────────────────────────┐
        │                    AIRFLOW (LocalExecutor)                     │
        │                               │                                │
        │  ingest_dag (*/5 min)         │      retrain_dag (daily + ⚡)   │
        │  ├─ poll station_status       │      ├─ blend labels            │
        │  ├─ derive PROXY departures   │      ├─ train LightGBM + LSTM   │
        │  └─ write hourly_departures   │      ├─ champion/challenger     │
        │                               │      └─ promote winner ─────┐   │
        │  monitor_dag (hourly)         │                             │   │
        │  ├─ rolling MAE (pred vs real)│                             │   │
        │  ├─ Evidently feature drift   │                             │   │
        │  └─ breach? ──TriggerDagRun──⚡┘                             │   │
        └───────────────────────────────┬─────────────────────────────┼───┘
                                        │                             │
              ┌─────────────────────────┴──────────┐                  │
              │                                    │                  │
      ┌───────▼────────┐                  ┌────────▼────────┐  ┌──────▼──────┐
      │   POSTGRES     │                  │     MLFLOW      │  │  FASTAPI    │
      │ station_info   │                  │  tracking +     │  │  /predict   │
      │ status_snapshots│◄────────────────│  model registry │─►│  /health    │
      │ hourly_departures                 │  @production    │  │  /model-info│
      │ predictions    │                  └─────────────────┘  └──────┬──────┘
      │ features       │                                              │
      │ model_metrics  │◄─────────────────────────────────────────────┘
      └───────┬────────┘                                       (logs predictions)
              │
      ┌───────▼────────┐
      │   STREAMLIT    │  predicted vs realised map · rolling MAE · drift · model version
      └────────────────┘
```

**The loop, precisely:** `monitor_dag` runs hourly. It joins logged predictions
to realised demand for rolling MAE, and compares the feature distribution being
served against the one the live model was trained on. If *either* the error
threshold or the drift threshold is breached, it fires `TriggerDagRunOperator`
against `retrain_dag`. That trains every configured candidate on a rolling
window, compares each against the current champion on an identical holdout, and
promotes only if a challenger wins by more than a configured margin. The API
picks up the new `@production` alias on reload.

---

## ⚠️ The demand proxy — read this before trusting any live number

GBFS reports **inventory** (bikes docked right now). It does **not** report
**demand** (rides taken). The live loop estimates the latter by differencing
consecutive snapshots:

```
proxy_departures(t) = max(0, bikes_available(t-1) − bikes_available(t))
```

This is a **proxy**, and it is wrong in knowable ways:

| Problem | Effect | Mitigation in code |
|---|---|---|
| **Within-interval netting** | 5 bikes leave, 4 arrive → proxy reports 1, not 5. **Systematically biased low, worst at busy stations.** | None possible. Documented and tested (`test_netting_understates_true_demand`). |
| **Rebalancing vans** | A truck removing 15 bikes looks like 15 riders. | Capped at `max_plausible_drop_per_interval`. Bounded, not solved. |
| **Dock / reporting outages** | Inventory jumps that are not rides. | Pairs dropped unless `is_renting` and `is_installed` on both ends. |
| **Stale reporters** | Station's inventory figure is not current. | Pairs dropped when `last_reported` is older than `max_staleness_minutes`. |
| **Missed polls** | Delta spans an unknown period. | Pairs dropped beyond `max_interval_gap_minutes`. |
| **Partial hours** | 3 intervals summed as a full hour understates it. | Station-hours below 50% snapshot coverage are dropped. |

**Consequences you must carry forward:**

- The **initial model is trained on historical trip CSVs**, where every label is
  a real completed trip. That is where the honest accuracy number comes from.
- The **rolling MAE from the live loop is scored against proxy labels.** It is a
  good *change detector* — which is all the loop needs — but it is **not** the
  model's true error and must never be quoted as such.
- Any model retrained on live data is tagged `labels_are_proxy=true` in MLflow,
  and the dashboard labels realised demand **"realised (proxy)"** everywhere.

---

## Results

Trained on **Apr–Jun 2026**: 13.85M trips, 2,270 stations, aggregated to
40 geographic clusters (median 58 stations, ~0.9 km mean radius). Test set is
the final 14 days, held out chronologically.

| Model | Test MAE | RMSE | sMAPE |
|---|---|---|---|
| **LightGBM** (Poisson) | **24.57** | 51.82 | 22.80% |
| Seasonal naive (same hour, last week) | 46.13 | — | — |

LightGBM is **47% better than the seasonal-naive baseline**. Mean demand is
~161 departures per cluster-hour, so MAE 24.6 is roughly 15% of the mean.

The baseline is reported alongside every run on purpose: a forecasting MAE
without one is uninterpretable.

---

## Quick start

```bash
cp .env.example .env          # then change the passwords
docker compose up -d          # postgres, mlflow, api, dashboard, airflow

# Seed schema + stations + historical departures (needed before /predict works)
python -m scripts.bootstrap_db

# Train the initial model and register it
MLFLOW_TRACKING_URI=http://localhost:5001 python -m src.models.train --source offline
```

| Service | URL |
|---|---|
| Prediction API (docs at `/docs`) | http://localhost:8000 |
| MLflow | http://localhost:5001 |
| Airflow | http://localhost:8080 |
| Streamlit dashboard | http://localhost:8501 |

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"station_ids": ["5506.14", "6575.03"]}'
```

### Demonstrating the closed loop

```bash
python -m scripts.simulate_drift --baseline        # serve clean predictions
python -m scripts.simulate_drift --inject --shift 2.5   # shift the distribution
python -m scripts.simulate_drift --check           # monitoring verdict
```

Then unpause `monitor_dag` in Airflow and watch it trigger `retrain_dag`.
`--reset` removes everything the simulation wrote.

---

## Design decisions worth defending

**One feature function, enforced by test.** `build_feature_panel()` is called by
training *and* by serving. `tests/test_features_parity.py` asserts the two
produce bit-identical rows for overlapping hours. Train/serve skew is silent and
fatal; this is the only reliable defence.

**Lags never touch their own label.** Rolling statistics are computed on a
`shift(1)` series. A plain `rolling(24).mean()` would include the current hour
and leak the target — tested explicitly, including a proof that the leaky
version *would* have differed.

**"Current fill ratio" is deliberately excluded.** The brief lists it as a
feature. Historical trip CSVs contain no inventory data, so it would be entirely
NaN in training and fully populated at serving — a guaranteed distribution skew.
It can be added once the live loop has accumulated enough snapshot history to
train on.

**Chronological splits only.** Validation and test are trailing time windows. A
random split on a time series lets the model see `t+1` while predicting `t`.

**Promotion needs a 2% margin.** Holdout MAE wobbles run-to-run from noise
alone; promoting on any improvement would churn production daily for no real
gain and make every incident harder to explain.

**Clustering over per-station.** Per-station hourly demand is mostly zeros
overnight, which makes the target noisy and evaluation meaningless. Configurable
via `features.granularity`.

**Weather comes from two providers.** NOAA was specified, and NOAA
**cannot serve the training window** — verified at build time,
`api.weather.gov` returns zero observations beyond ~7 days. Open-Meteo's archive
API supplies historical backfill; NOAA serves the live path. NOAA's hourly
forecast exposes precipitation *probability*, not amount, which is scaled to
keep one schema — an approximation, flagged in the code.

**LSTM rather than TFT.** `pytorch-forecasting` pins an older Lightning/PyTorch
stack that conflicts with the rest of this project. A dependency conflict at the
centre of the retraining loop is worse than a simpler architecture. The choice
is config-driven (`models.deep_model`), so TFT can be added without touching
promotion.

---

## Repository layout

```
config/config.yaml        every threshold, URL, and hyperparameter
src/
  ingestion/              GBFS client, trip-CSV loader, weather, PROXY derivation
  features/               build_features.py  ← the shared function
  models/                 lightgbm, LSTM, train, promote (champion/challenger)
  serving/                FastAPI app, model loader, history provider
  monitoring/             rolling metrics, Evidently drift, loop decision
  storage/                Postgres access layer
  dashboard/              Streamlit
dags/                     three thin DAGs — logic lives in src/
scripts/                  bootstrap_db.py, simulate_drift.py
tests/                    parity, leakage, proxy, CSV schema, promotion, API
```

Engineering notes worth knowing:

- **Trip archives are streamed.** An NYC month is ~1 GB uncompressed across six
  part-files. `HttpRangeFile` reads the zip over HTTP range requests and reduces
  each chunk to hourly counts in flight — peak memory stays in the tens of MB.
- **The join key is not obvious.** Trip CSV `start_station_id` (`5506.14`) maps
  to GBFS **`short_name`**, *not* GBFS `station_id` (a UUID). Joining on the
  latter matches zero rows.
- **Ingestion never crashes on bad data.** Malformed stations and rows are
  logged and skipped. A DAG that dies on one bad record stops collecting the
  history the model depends on.

---

## Limitations & roadmap

**Limitations — the honest list.**

1. **The demand proxy is biased low**, worst where demand is highest. See the
   table above. This is inherent to inventory-derived labels.
2. **Live rolling MAE is measured against that proxy**, so it tracks *change*
   rather than true accuracy.
3. **Three months of training data** (Apr–Jun) means no seasonal coverage — the
   model has never seen winter, and `season` is nearly constant in training.
4. **A single 14-day test window** carries real variance; there is no rolling-
   origin backtest yet.
5. **Cluster granularity hides per-station error.** 40 clusters averaging 58
   stations each is coarse for operational rebalancing decisions.
6. **The deep model is not fully servable.** The LSTM is trained, registered and
   compared honestly, but the pyfunc path reconstructs its input sequence from
   lag columns rather than true history. If it were promoted, serving would be
   approximate — so LightGBM is the practical champion today.
7. **NOAA precipitation is a probability, not an amount**, in the live path.
8. **The API image is large** (~10 GB) because torch is installed to support a
   promoted deep model.

**Roadmap.** Kubernetes deployment; Kafka for snapshot streaming; a dedicated
feature store (Feast); rolling-origin backtesting; per-station modelling with a
hierarchical reconciliation step; a true sequence-serving path for the LSTM;
conformal prediction intervals.

---

## Testing

```bash
pytest -q          # hermetic: no network, no database, no MLflow server
ruff check src tests dags scripts
```

CI runs lint, the test suite, an Airflow DAG-import check, and a compose build
on every push.
