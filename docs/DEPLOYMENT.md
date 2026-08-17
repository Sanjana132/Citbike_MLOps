# Continuous ingestion for free, on GitHub Actions

Collect live GBFS data around the clock without running or paying for a server:
a free hosted Postgres plus a scheduled GitHub Actions job.

Local `docker compose` still runs everything else — training, serving, MLflow,
the dashboard. Only *ingestion* needs to be awake continuously, and only
ingestion is deployed here.

---

## What this actually gives you, and what it does not

**Does:** live GBFS data collected day and night without a machine of your own,
hourly demand proxies derived automatically, and a database you can train
against from anywhere.

**Does not:** a true 60-second cadence, or a guarantee that every hour is
captured.

| | Always-on host | GitHub Actions (this) |
|---|---|---|
| Cost | ~$4–6/month | **£0** |
| Sampling cadence | 60s, uninterrupted | 60s *within* a run; runs every 5 min at best |
| Hours captured | essentially all | most, with gaps when cron is delayed |
| Ops burden | you patch a server | none |

Four limits worth knowing before you rely on it:

1. **GitHub's cron minimum is 5 minutes and runs are best-effort.** Scheduled
   jobs sit on shared infrastructure and are routinely late by 5–15 minutes,
   occasionally skipped under load. Each run therefore samples for ~4 minutes at
   60-second intervals rather than polling once, which recovers most of the
   cadence *inside* a run. Shorter intervals genuinely matter: the longer the
   gap, the more departures and returns cancel out unseen within it.
2. **Hours below 50% sample coverage are discarded**, not written as partial
   hours. A delayed cron costs a whole hour rather than corrupting it — the
   right trade, but it means gaps.
3. **Free Postgres tiers cap around 500 MB.** Three days of raw snapshots
   measured 505 MB on their own, so snapshot retention drops to **6 hours**
   here. Nothing is lost: derivation only looks back a few hours, and the
   durable output is `hourly_departures`.
4. **Scheduled workflows are disabled after 60 days of repository inactivity.**
   Any commit re-enables them.

---

## Setup, once

### 1. Create a free Postgres

Either works.

- **[Neon](https://neon.tech)** — 0.5 GB, scales to zero, wakes on connection.
- **[Supabase](https://supabase.com)** — 500 MB; free projects pause after a week
  of *inactivity*, which constant polling prevents.

### 2. Convert the connection string

Both hand you a URL starting `postgresql://`. This project uses psycopg2
explicitly, so insert the driver:

```
postgresql://user:pass@host/db?sslmode=require            ← what they give you
postgresql+psycopg2://user:pass@host/db?sslmode=require   ← what you need
```

Keep `sslmode=require`; both providers demand TLS.

### 3. Add the secret

Repo → **Settings** → **Secrets and variables** → **Actions** → **New repository
secret**:

| Name | Value |
|---|---|
| `DATABASE_URL` | the `postgresql+psycopg2://…` string |

Optional, for alerts when ingestion stalls:

| Name | Value |
|---|---|
| `SMTP_HOST` `SMTP_USERNAME` `SMTP_PASSWORD` | your mail provider (Gmail needs an App Password) |
| `ALERT_EMAIL_RECIPIENTS` | where alerts go |

### 4. Create the tables

**Actions** → **Initialise database schema** → **Run workflow**. It creates the
six tables and seeds station metadata from the live feed. Safe to re-run —
everything is `CREATE TABLE IF NOT EXISTS`.

### 5. Let it run

**Ingest GBFS** starts on its own schedule. Trigger it manually once to confirm.
Each run prints what landed:

```
distinct snapshots (last hour): 12
proxy hours derived (last 24h): 19
latest proxy hour: 2026-08-17 21:00:00
```

---

## Training against the hosted database

Point any local command at it — the code is identical, only the URL changes:

```bash
export DATABASE_URL='postgresql+psycopg2://user:pass@host/db?sslmode=require'

python -m src.models.train --source blend      # historical CSVs + live proxy
python -m scripts.simulate_drift --check       # monitoring verdict
```

`--source blend` needs live ingestion to have bridged the gap between where the
trip CSVs end and where collection began; until then it falls back to `offline`
and says so.

---

## If you outgrow it

The same code runs unchanged on a small VM — that is the point of keeping
ingestion in a container:

```bash
# on any host with Docker
git clone https://github.com/Sanjana132/Citbike_MLOps && cd Citbike_MLOps
cp .env.example .env && $EDITOR .env          # set POSTGRES_PASSWORD
docker compose -f docker-compose.ingest.yml up -d
```

That runs Postgres and the poller only — **~400 MB RAM measured** (ingestor
162 MB, Postgres 243 MB) and near-zero CPU, so the cheapest instance any
provider sells is enough. `restart: unless-stopped` handles reboots, and the
poller's healthcheck fails if no snapshot has landed in 15 minutes, because a
process can be running and stuck — which is exactly what a leaked connection
pool looked like here.

---

## Troubleshooting

**Workflow fails with `DATABASE_URL secret is not set`** — the secret is missing,
or was added to an Environment rather than to repository secrets.

**`psycopg2.OperationalError: SSL required`** — keep `?sslmode=require`.

**`relation "status_snapshots" does not exist`** — run *Initialise database
schema* first.

**`No station-hours passed the snapshot-coverage threshold`** — normal early on.
An hour needs ≥50% of its expected samples, so the first partial hour is always
skipped. Persisting for hours means cron runs are being dropped; check the
Actions tab for skipped schedules.

**Database is full** — lower `SNAPSHOT_RETENTION_HOURS` in the workflow (3 is
enough for derivation), and export `hourly_departures` periodically:

```bash
psql "$DATABASE_URL" -c "\copy (SELECT * FROM hourly_departures WHERE label_source='gbfs_proxy') TO 'proxy.csv' CSV HEADER"
```
