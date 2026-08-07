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

### How far off is it, actually?

Measured on this stack against real GBFS traffic, rather than asserted:

| | value |
|---|---|
| Snapshot pairs usable after corrections | **9,472 / 9,852 (96.1%)** |
| Pairs rejected (`is_renting = 0`) | 380 |
| Proxy-implied system rate at ~00:00 local | **2,839 departures/hour** |
| True rate at 00:00 from trip CSVs (3-month mean) | **2,379 departures/hour** |
| Ratio | **1.19x — the proxy *overstates* here** |

The right order of magnitude, and wrong in the direction the theory predicts:
overnight, rebalancing vans move more bikes than riders do, and a van emptying a
dock is indistinguishable from a queue of customers. Netting pushes the other
way and dominates at busy daytime hours, so the bias is not a constant you can
calibrate away with a single factor.

Treat this as a sanity check, not a validation study: it is one 16-minute window
on one night compared against a three-month hourly average.

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

Trained on **May 2025 – Jun 2026**: 14 months, 54.6M trips, 2,436 stations,
aggregated to 40 geographic clusters. Test set is the final 14 days, held out
chronologically. Every figure is **as-served** — the model is loaded back
through `mlflow.pyfunc` and scored through the path that actually serves it.

| Model | Test MAE | RMSE | sMAPE | Serving gap |
|---|---|---|---|---|
| **LightGBM** (Poisson) — **champion, v3** | **20.27** | 48.68 | 20.87% | **0.0%** |
| LightGBM on 3 months (previous champion) | 24.57 | 51.82 | 22.80% | 0.0% |
| LSTM (deep candidate) | 39.27 | 92.53 | 40.66% | +76.5% |
| Seasonal naive (same hour, last week) | 46.11 | — | — | — |

Extending training from 3 to 14 months improved MAE by **17.5%** (24.57 → 20.27),
and champion/challenger promoted it automatically. The model is **56% better
than the seasonal-naive baseline**. Mean demand is ~161 departures per
cluster-hour, so MAE 20.3 is roughly 13% of the mean.

### Seasonal features: measured, and mostly rejected

More data was the win. The seasonal *features* were not — and the ablation is
worth recording, because the intuition was wrong. Identical data, identical
split, one change at a time:

| variant | Test MAE | vs base |
|---|---|---|
| base (no seasonal) | 20.034 | — |
| **+ 30-day rolling mean** | **19.978** | **−0.28%** |
| + daylight only | 20.795 | +3.80% |
| + annual cycle only | 21.163 | **+5.64%** |
| + all seasonal | 20.984 | +4.74% |

`day_of_year_sin/cos` ranked among the *most-used* features while making
predictions **worse**. That combination is the signature of memorisation rather
than learning: across 14 months each calendar date occurs once or twice, so the
model can key on "late June 2025" instead of learning an annual shape — and late
June 2025 is not late June 2026. Daylight fails the same way, being a
deterministic function of day-of-year.

So `features.seasonal_feature_groups` defaults to **empty**. The code, the solar
geometry and its tests all remain, verified against published NYC sunrise/sunset
to within 2 minutes; enable the groups once training spans several years, where
an annual cycle can be estimated instead of memorised. The 30-day rolling mean —
which tracks the *recent* level rather than position in the year — is the one
piece that did not hurt, and it stays on.

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

### Pipeline liveness alerts

Drift and error monitoring answer *"is the model still good?"*. Neither can
answer *"is the pipeline still running?"* — and the difference is not academic.

Ingestion on this stack once stopped for **42 hours with every other signal
green**: the Airflow bind mounts went stale, `/opt/airflow/dags` became empty
inside the container, Airflow quietly marked every DAG inactive, and nothing
noticed. Containers were healthy, the API served predictions, the scheduler
heartbeat was fine. The model checks stayed silent precisely *because* they need
new data to detect anything, and none was arriving:

- No fresh predictions to score, so rolling MAE stopped updating rather than
  degrading.
- No fresh features, so drift reported "not evaluated" — easy to read as "fine".

`src/monitoring/liveness.py` checks the newest row in each table directly.
`monitor_dag` runs it hourly ahead of the model checks and emails on breach;
`/health` reports it live via `pipeline_live`:

```json
{
  "status": "degraded",
  "pipeline_live": false,
  "warnings": ["Ingestion may have stalled - hourly_departures (proxy):
                newest row is 2739 min old, over the 180 min threshold"]
}
```

Thresholds are generous multiples of the poll interval
(`monitoring.liveness.*`), so one missed run does not page anyone. The alert
states explicitly that **retraining will not fix an ingestion outage**, so
nobody reaches for the wrong lever.

### Email alerts on model change

A promotion silently swaps the model behind a live API, so `promote.py` emails
the configured recipients whenever the production alias moves. Recipients live
in `config.yaml` (`alerting.email.recipients`, currently `skomma18@umd.edu`) and
can be overridden at runtime with `ALERT_EMAIL_RECIPIENTS`.

SMTP credentials come **only** from the environment — never from the committed
config:

```bash
SMTP_HOST=smtp.gmail.com     # leave unset to disable sending
SMTP_PORT=587                # 465 switches to implicit SSL, else STARTTLS
SMTP_USERNAME=you@gmail.com
SMTP_PASSWORD=<app password> # Gmail needs an App Password, not the account one
```

Two guarantees, both tested:

- **It never breaks a promotion.** Every failure path is caught. A dead mail
  server must not turn a successful model promotion into a failed training run.
- **It never claims a delivery it did not make.** With SMTP unconfigured it logs
  `SMTP is not configured ... would have emailed <address>` and returns `False`,
  rather than reporting success.

The alert carries the new version, model type, the as-served challenger and
champion scores, and the improvement. If the model was trained on blended labels
it also carries a prominent warning that its accuracy is measured against the
demand **proxy**, not true ride counts.

The SMTP path is verified end to end against a local mail server — a real
message was transmitted and received, correctly addressed:

```
Subject: [citibike-mlops] citibike-demand-forecaster v3 promoted to production (lightgbm)
To: skomma18@umd.edu

  Model            : citibike-demand-forecaster
  New version      : 3
  Challenger score : 21.5
  Previous champion: 24.566
  Improvement      : 12.48%
```

What is **not** verified is delivery through a real provider, because no
external SMTP credentials are configured on this machine. Supply them and the
same path applies unchanged.

---

## Latency

Measured end to end against the running container, `POST /predict` for all 40
clusters:

| | before | after |
|---|---|---|
| **Median warm request** | ~310 ms | **23 ms** (~13x faster) |
| Cold request | 1.37 s | 0.35 s |

The profile showed the model was never the bottleneck — the plumbing was:

| stage | before | after |
|---|---|---|
| `load_recent_history` | 157 ms | **0 ms** (cached) |
| `_load_weather` | 134 ms | **0 ms** (cached) |
| `build_feature_panel` | 38 ms | 16 ms |
| `model.predict` | 39 ms | 1 ms |
| prediction logging | on the response path | moved off it |

What changed:

1. **The history query returned the whole table.** The serving path needs a
   trailing slice anchored on the *newest observation*, which can be weeks
   behind wall-clock now. It did that with a wide read plus a 730-day fallback
   read, transferring ~2.6M rows to keep the tail. `read_latest_hourly_departures`
   anchors the window in SQL instead — one query, filtered in the database.
2. **Weather was fetched over HTTP on every single request**, costing more than
   inference itself. Hourly forecasts do not change between requests seconds
   apart, so both history and weather are now TTL-cached
   (`serving.history_cache_seconds`, `serving.weather_cache_seconds`). The
   ingestion DAG invalidates the history cache when it writes, so the TTL is an
   upper bound on staleness rather than a fixed delay.
3. **Prediction logging blocked the response** on two database writes. It now
   runs in a FastAPI background task: forecasting is the service's job,
   bookkeeping is not.

Caching freshness is not hidden. `/health` reports `postgres` only when the
newest observation is within `serving.history_freshness_hours` (6h); otherwise
it reports `postgres_stale` and degrades. Every response still carries
`history_lag_hours`.

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
make demo-drift        # baseline -> inject -> monitoring verdict
```

or step by step:

```bash
python -m scripts.simulate_drift --baseline             # serve predictions
python -m scripts.simulate_drift --inject --shift 3.0   # shift the distribution
python -m scripts.simulate_drift --check                # monitoring verdict
```

Then unpause `monitor_dag` in Airflow. `--reset` removes everything the
simulation wrote.

**Verified on this stack.** `monitor_dag` ran, evaluated, and triggered
retraining with no human involved:

```
Rolling MAE=23.844 over 838 samples; breached=False
Monitoring cycle: should_retrain=True | drift: 69% of features drifted
  (threshold 30%); worst: ['lag_1h','lag_2h','lag_3h','lag_24h','lag_168h']
{branch.py:38} Branch into trigger_retrain
{skipmixin.py:233} Following branch ('trigger_retrain',)
{skipmixin.py:281} Skipping tasks [('no_action', -1)]
```

`retrain_dag` runs were then created by `TriggerDagRunOperator`. Note that the
error trigger correctly did **not** fire — rolling MAE (23.84) was slightly
*better* than the training holdout MAE (24.57) — while the drift trigger did.
That is the complementarity the two-trigger design is for.

**And the retrain itself completes autonomously.** A full `retrain_dag` run in
Airflow, start to finish in 14 minutes, with no human in the loop:

```
WARNING Training on source='blend' failed (... 33 days 13:00:00 gap ...);
        retrying with source='offline'
INFO    LightGBM:   as-served test MAE=24.5660 (in-process 24.5660)
WARNING Deep model: as-served MAE 39.4733 differs from in-process 22.7873 by
        73.2% - the serving path does not reproduce the holdout
INFO    Promotion decision: Best challenger lightgbm changed mae by 0.00%,
        which does not clear the 2.00% bar; keeping the current champion
Returned: {'promoted': False, 'fell_back': True, ...}
```

Every safeguard fired in the right order: the gap diagnostic, the fallback, the
serving-gap warning, and a promotion correctly declined. Both tasks
(`train_and_promote`, `refresh_serving`) succeeded.

`monitor_dag` likewise runs all five tasks green, including the new
`check_liveness`, which detected the stalled proxy stream and attempted the
alert.

Because the simulation's timestamps are historical and the monitor's window is
measured backwards from now, `simulate_drift` shifts its rows forward by a whole
number of weeks (preserving hour-of-day and day-of-week). It fabricates rows by
design; never point it at a real deployment.

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

**Promotion scores candidates through the serving path, not an in-process
estimator.** This is the single most important correction in the project, and it
was earned. The LSTM was promoted on an in-process holdout MAE of **22.25**,
computed by feeding it true 168-hour sequences. Its pyfunc serving path rebuilds
those sequences from five lag columns and actually scores **39.27** — 76% worse
than the number it was promoted on, and **60% worse than the LightGBM champion
(24.57) it displaced**. Champion/challenger shipped a materially worse model
while its scoreboard said otherwise.

The feature-parity test could not catch this: features were identical on both
sides. The skew was in how the *model* was applied. So `train.py` now loads each
candidate back through `mlflow.pyfunc` and scores the same holdout through it,
and those as-served numbers are what promotion compares. The in-process metrics
are still logged for diagnosis, alongside an explicit `serving_gap_ratio` so a
candidate that cannot reproduce its own holdout is visible rather than silently
promoted.

The general rule, worth stating plainly: **measure the artifact you ship, not
the object in memory.**

**Monitoring thresholds are calibrated to this target, and that mattered.** The
first run of the loop fired on *clean* baseline data, for two reasons worth
recording:

- An absolute MAE ceiling of 4.0 is meaningless when mean demand is ~161
  departures per cluster-hour and the trained model's own holdout MAE is 24.6.
  The ceiling is now the seasonal-naive baseline (~46): if the model is worse
  than "same hour last week", it has stopped earning its keep. The primary rule
  is the scale-free one — degradation relative to the model's own holdout.
- **Calendar features are excluded from drift detection.** `hour`, `hour_sin`,
  `dow_sin` and friends are deterministic functions of the timestamp, so a
  24-hour serving window differs from a nine-week training reference *by
  construction*. They were responsible for most of an apparent 77% drift share
  on data that had not drifted at all. Genuine drift lives in the lag, weather
  and entity features, which the clock does not determine.

**The drift that remained after that fix was real.** Late-June demand runs 1.22x
the April–early-June training average (7,523 vs 6,189 departures/hour
system-wide) at 26.9 °C versus 16.8 °C. Serving late-June traffic from a model
trained on spring data genuinely warrants a retrain — a true positive, verified
against the data rather than assumed.

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

1. **The demand proxy is biased**, in both directions depending on time of day
   (measured at 1.19x overnight; netting pulls the other way at peak). This is
   inherent to inventory-derived labels and cannot be calibrated away with a
   single factor.
2. **Live rolling MAE is measured against that proxy**, so it tracks *change*
   rather than true accuracy.
3. **`--source blend` does not work on this data yet.** Trip CSVs end
   2026-06-30 and live ingestion began 2026-08-03, leaving a 33-day hole, so a
   split window lands inside it and training aborts with a clear diagnostic.
   `retrain_dag` therefore uses `run_training_with_fallback`: it attempts
   `blend` and falls back to `offline` (true trip-CSV labels) when the blended
   data is unusable, logging the fallback loudly and flagging it on the run
   summary. An automatically triggered retrain has to be able to finish on its
   own — aborting would leave the loop permanently open. Manual CLI runs still
   fail loudly rather than falling back silently.
4. **Three months of training data** (Apr–Jun) means no seasonal coverage — the
   model has never seen winter, and `season` is nearly constant in training.
5. **A single 14-day test window** carries real variance; there is no rolling-
   origin backtest yet.
6. **Cluster granularity hides per-station error.** 40 clusters averaging 58
   stations each is coarse for operational rebalancing decisions.
7. **The deep model is not servable as built, and is measurably worse.** The
   LSTM is trained, registered and compared honestly, but its pyfunc serving
   path reconstructs the input sequence from five lag columns instead of true
   history, costing it 76.5% (22.25 in-process -> 39.27 as served, worse than
   the naive baseline). It is correctly never promoted. Rebuilding that serving
   path is the top roadmap item.
8. **NOAA precipitation is a probability, not an amount**, in the live path.
9. **The API image is large** (~10 GB) because torch is installed so that a
   promoted deep model can be loaded.
10. **Airflow runs LocalExecutor on one host** with no resource isolation, so a
    retrain competes for CPU with ingestion.

### What running it actually caught

Several defects were only visible once the system was live, and they are the
kind that stay silent:

- **Fabricated zero demand.** `complete_panel` filled every missing
  (entity, hour) with 0. Across the 33-day hole between the historical backfill
  and live ingestion, that invented **32,160 rows (26.6% of the panel)** of
  fake zero demand. The validation window landed inside the hole, so it was all
  zeros, the model trivially predicted zero, `val_mae` collapsed to **0.001** —
  and `test_mae` came out at **20.70**, *better* than the honest 24.57, which
  means the promotion rule would have shipped it. Now only gaps shorter than
  `features.max_gap_fill_hours` are filled.
- **The deep model scored nothing.** Validation is exactly 168 hours and the
  LSTM's sequence length is 168, so zero sequences formed; the candidate logged
  no usable predictions and early stopping silently fell back to training loss.
  Splits now draw sequence context from preceding data (0 → 6,720 sequences).
- **Monitoring fired on clean data**, from an absolute MAE ceiling set for the
  wrong scale and from calendar features that drift by construction.
- **A retrain storm**: drift persists until a new model serves, so an hourly
  monitor queued a retrain every hour. Three had stacked up before the cooldown
  was added.
- **The auto-retrain could never finish.** `retrain_dag` defaults to `blend`,
  which the 33-day gap makes unusable, so all four early runs failed. An
  automatically triggered retrain that always aborts is not a closed loop; it
  now falls back to trip-CSV labels and completes.
- **A 42-hour silent ingestion outage.** Stale bind mounts emptied
  `/opt/airflow/dags`; Airflow marked every DAG inactive and nothing noticed,
  because drift and error monitoring need new data to detect anything. Liveness
  checks now cover it.

The suite is hermetic — verified by running it with `socket.connect` blocked,
not merely asserted.

**Roadmap**, most valuable first:

1. **A true sequence-serving path for the LSTM** — it is the only thing standing
   between a candidate that scores 22.25 in-process and one that can actually be
   promoted. Likely means passing recent demand history to the pyfunc wrapper
   rather than reconstructing it from lag columns.
2. **Rolling-origin backtesting**, so a single 14-day window stops being the
   whole evidence base.
3. **More history**, for seasonal coverage the model has never seen.
4. Per-station modelling with hierarchical reconciliation; conformal prediction
   intervals; Kubernetes deployment; Kafka for snapshot streaming; a dedicated
   feature store (Feast).

---

## Testing

```bash
pytest -q          # hermetic: no network, no database, no MLflow server
ruff check src tests dags scripts
```

CI runs lint, the test suite, an Airflow DAG-import check, and a compose build
on every push.
