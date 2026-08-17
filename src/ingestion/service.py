"""Always-on GBFS poller.

Why this exists instead of an Airflow DAG
-----------------------------------------
Ingestion originally ran as ``ingest_dag`` on a 5-minute schedule. It kept
stalling, and the reason turned out to be a tool mismatch rather than a bug:

* The feed advances ``last_updated`` every **60 seconds**. Airflow's minimum
  practical granularity here is a DAG run, and each run carries scheduling
  latency, a task process spawn, and a DagBag parse - overhead measured in tens
  of seconds against a 60-second signal.
* Measured over 6 hours on this host: **4 runs completed out of an expected 72**,
  individual runs took 1.5-2.5 hours against a 4-minute ``execution_timeout``
  that never fired, and the scheduler lost its heartbeat 10 times in 3 hours.
  With ``max_active_runs=1`` a single slow run blocks every run behind it.

Airflow is a batch orchestrator. Polling a 60-second feed is a long-running
service, so this is a long-running service. Airflow keeps the work it is
genuinely good at - the daily retrain and the hourly monitor, where a minute of
scheduling jitter is irrelevant.

Design rules
------------
* **Never exit on error.** Any failure is logged and the loop continues; an
  ingester that dies on one bad response stops collecting the history the model
  depends on.
* **Drift-free cadence.** The next tick is computed from a fixed schedule rather
  than by sleeping a fixed amount after variable work, so slow polls do not
  slowly desynchronise the sampling grid.
* **Bounded work.** Each poll has a hard timeout, so a trickling server costs one
  sample rather than blocking the loop - the exact failure that stalled the DAG.
* **Graceful shutdown** on SIGTERM/SIGINT so ``docker compose down`` is clean.
"""

from __future__ import annotations

import logging
import signal
import threading
import time
from datetime import UTC, datetime

from src.config import Config, load_config

logger = logging.getLogger(__name__)


class PollerState:
    """Counters exposed in the heartbeat log line."""

    def __init__(self) -> None:
        self.polls_ok = 0
        self.polls_failed = 0
        self.rows_written = 0
        self.derives_ok = 0
        self.derives_failed = 0
        self.started_at = datetime.now(tz=UTC)

    def summary(self) -> str:
        uptime = (datetime.now(tz=UTC) - self.started_at).total_seconds() / 60.0
        return (
            f"uptime={uptime:.0f}min polls_ok={self.polls_ok} "
            f"polls_failed={self.polls_failed} rows={self.rows_written} "
            f"derives_ok={self.derives_ok} derives_failed={self.derives_failed}"
        )


def _poll_once(client, state: PollerState) -> None:
    """One snapshot. Every failure is contained here."""
    from src.storage.db import write_status_snapshots

    try:
        snapshot = client.fetch_station_status()
    except Exception as exc:  # noqa: BLE001 - a bad poll must never stop the loop
        state.polls_failed += 1
        logger.error("Poll failed: %s", exc)
        return

    if snapshot.empty:
        state.polls_failed += 1
        logger.warning("Poll returned no usable rows")
        return

    try:
        state.rows_written += write_status_snapshots(snapshot)
        state.polls_ok += 1
    except Exception as exc:  # noqa: BLE001
        state.polls_failed += 1
        logger.error("Could not persist snapshot: %s", exc)


def _derive_once(config: Config, state: PollerState) -> None:
    """Roll recent snapshots up into hourly proxy departures."""
    from src.ingestion.pipeline import derive_and_store_proxy_departures

    try:
        result = derive_and_store_proxy_departures(config=config)
        state.derives_ok += 1
        logger.info("Derived proxy departures: %s", result)
    except Exception as exc:  # noqa: BLE001
        state.derives_failed += 1
        logger.error("Derivation failed: %s", exc)


def _refresh_stations(config: Config) -> None:
    from src.ingestion.pipeline import ingest_station_information

    try:
        ingest_station_information(config)
    except Exception as exc:  # noqa: BLE001
        logger.error("Station metadata refresh failed: %s", exc)


def run_poller(
    config: Config | None = None,
    *,
    max_iterations: int | None = None,
    max_seconds: float | None = None,
) -> PollerState:
    """Poll the feed until stopped.

    ``max_iterations`` bounds it for tests. ``max_seconds`` bounds it by wall
    clock, which is what the GitHub Actions deployment uses: a scheduled job
    cannot run forever, so it samples for a few minutes and exits, and the next
    cron firing continues the sequence. Same loop either way.
    """
    cfg = config or load_config()
    from src.ingestion.gbfs_client import GBFSClient

    spacing = float(cfg.get_path("ingestion.sample_spacing_seconds", 60))
    derive_every = float(cfg.get_path("ingestion.derive_interval_seconds", 300))
    station_refresh_every = float(cfg.get_path("ingestion.station_refresh_seconds", 3600))

    state = PollerState()
    stop = threading.Event()

    def handle_signal(signum, _frame):
        logger.info("Received signal %s; shutting down after this poll", signum)
        stop.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, handle_signal)
        except ValueError:
            # Not on the main thread (e.g. under a test runner); shutdown is
            # then driven by max_iterations instead.
            pass

    client = GBFSClient(cfg)
    _refresh_stations(cfg)

    logger.info(
        "GBFS poller started: sampling every %.0fs, deriving every %.0fs", spacing, derive_every
    )

    # Fixed grid, so slow polls cannot make the cadence drift.
    start = time.monotonic()
    next_poll = start
    next_derive = start + derive_every
    next_station_refresh = start + station_refresh_every
    iterations = 0
    hard_deadline = start + max_seconds if max_seconds else None

    while not stop.is_set():
        if hard_deadline is not None and time.monotonic() >= hard_deadline:
            logger.info("Reached the %.0fs run budget; exiting cleanly", max_seconds)
            break
        now = time.monotonic()
        if now < next_poll:
            stop.wait(min(next_poll - now, 5.0))
            continue

        _poll_once(client, state)
        iterations += 1

        # Skip missed ticks outright rather than trying to catch up: the feed
        # only reports the present, so a backlog of polls would just re-record
        # the same instant.
        while next_poll <= time.monotonic():
            next_poll += spacing

        if time.monotonic() >= next_derive:
            # Guarded at the call site as well as inside _derive_once: the loop
            # must not depend on a callee's error handling to stay alive.
            # Deriving is downstream work - if it breaks, keep collecting raw
            # snapshots so nothing is lost while it is fixed.
            try:
                _derive_once(cfg, state)
            except Exception as exc:  # noqa: BLE001
                logger.error("Derivation raised past its own handler: %s", exc)
            logger.info("Poller heartbeat: %s", state.summary())
            while next_derive <= time.monotonic():
                next_derive += derive_every

        if time.monotonic() >= next_station_refresh:
            try:
                _refresh_stations(cfg)
            except Exception as exc:  # noqa: BLE001
                logger.error("Station refresh raised past its own handler: %s", exc)
            while next_station_refresh <= time.monotonic():
                next_station_refresh += station_refresh_every

        if max_iterations is not None and iterations >= max_iterations:
            break

    logger.info("GBFS poller stopped: %s", state.summary())
    return state


def main() -> None:
    """CLI entry point.

    Runs forever by default (the container deployment). ``--max-seconds`` makes
    it a bounded one-shot for schedulers that cannot host a daemon, such as
    GitHub Actions.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Poll the GBFS feed")
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="Exit after this many seconds instead of running forever",
    )
    parser.add_argument(
        "--derive-on-exit",
        action="store_true",
        help="Run one derivation before exiting, so a short run still produces hourly rows",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
    )
    config = load_config()
    state = run_poller(config, max_seconds=args.max_seconds)

    if args.derive_on_exit:
        # A 4-minute run would otherwise exit before its derive interval
        # elapsed, so the snapshots it collected would sit unprocessed.
        _derive_once(config, state)
        logger.info("Final state: %s", state.summary())


if __name__ == "__main__":
    main()
