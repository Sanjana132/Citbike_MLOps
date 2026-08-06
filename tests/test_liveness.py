"""Pipeline liveness checks.

Written after a real incident: stale Airflow bind mounts silently stopped
ingestion for 42 hours while every other signal reported healthy. These tests
pin the behaviour that would have caught it.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from src.monitoring import liveness as liveness_module
from src.monitoring.liveness import (
    LivenessCheck,
    LivenessReport,
    _check,
    alert_on_stalled_pipeline,
    check_pipeline_liveness,
)


def _minutes_ago(minutes: float) -> datetime:
    return datetime.now(tz=UTC).replace(tzinfo=None) - timedelta(minutes=minutes)


# --------------------------------------------------------------------------
# The freshness rule
# --------------------------------------------------------------------------

def test_recent_data_is_healthy():
    check = _check("gbfs_snapshots", _minutes_ago(4), threshold_minutes=30)
    assert check.healthy
    assert "fresh" in check.detail


def test_stale_data_is_unhealthy():
    check = _check("gbfs_snapshots", _minutes_ago(120), threshold_minutes=30)
    assert not check.healthy
    assert "over the" in check.detail


def test_the_42_hour_outage_is_detected():
    """The incident this module exists for.

    Containers healthy, API serving, scheduler heartbeating - and no data for
    42 hours. Nothing else noticed.
    """
    check = _check("gbfs_snapshots", _minutes_ago(42 * 60), threshold_minutes=30)
    assert not check.healthy
    assert check.age_minutes == pytest.approx(2520, rel=0.01)


def test_a_single_missed_poll_does_not_alert():
    """Thresholds are generous multiples of the poll interval; no flapping."""
    assert _check("gbfs_snapshots", _minutes_ago(11), threshold_minutes=30).healthy


def test_no_rows_at_all_is_unhealthy():
    """An empty table is a pipeline that never ran, not a fresh one."""
    check = _check("gbfs_snapshots", None, threshold_minutes=30)
    assert not check.healthy
    assert "never written" in check.detail


# --------------------------------------------------------------------------
# Report aggregation
# --------------------------------------------------------------------------

def test_report_is_unhealthy_if_any_stream_stalls():
    report = LivenessReport(
        checks=[
            LivenessCheck("a", _minutes_ago(1), 1.0, 30, True, "a: fresh"),
            LivenessCheck("b", _minutes_ago(500), 500.0, 30, False, "b: stalled"),
        ]
    )
    assert not report.healthy
    assert [c.name for c in report.failures] == ["b"]
    assert "stalled" in report.summary()


def test_healthy_report_summarises_cleanly():
    report = LivenessReport(
        checks=[LivenessCheck("a", _minutes_ago(1), 1.0, 30, True, "a: fresh")]
    )
    assert report.healthy
    assert report.summary() == "All data streams are fresh"


# --------------------------------------------------------------------------
# End-to-end against a stubbed database
# --------------------------------------------------------------------------

@pytest.fixture
def stub_timestamps(monkeypatch):
    """Control what each table reports as its newest row."""
    values: dict[str, datetime | None] = {}

    def fake_latest(engine, table, column):
        return values.get(table)

    monkeypatch.setattr(liveness_module, "_latest_timestamp", fake_latest)
    monkeypatch.setattr("src.storage.db.get_engine", lambda url=None: object())
    return values


def test_live_pipeline_reports_healthy(stub_timestamps, config):
    stub_timestamps["status_snapshots"] = _minutes_ago(3)
    stub_timestamps["hourly_departures"] = _minutes_ago(30)
    assert check_pipeline_liveness(config).healthy


def test_stalled_snapshots_report_unhealthy(stub_timestamps, config):
    stub_timestamps["status_snapshots"] = _minutes_ago(42 * 60)
    stub_timestamps["hourly_departures"] = _minutes_ago(42 * 60)

    report = check_pipeline_liveness(config)
    assert not report.healthy
    assert any("gbfs_snapshots" in c.name for c in report.failures)


def test_stalled_pipeline_sends_one_alert(stub_timestamps, config, monkeypatch):
    stub_timestamps["status_snapshots"] = _minutes_ago(42 * 60)
    stub_timestamps["hourly_departures"] = _minutes_ago(42 * 60)

    sent = {}
    monkeypatch.setattr(
        "src.alerting.email.send_email",
        lambda subject, body, recipients: sent.update(
            subject=subject, body=body, recipients=recipients
        )
        or True,
    )

    report = alert_on_stalled_pipeline(config)
    assert not report.healthy
    assert "stalled" in sent["subject"].lower()
    assert "skomma18@umd.edu" in sent["recipients"]
    # The alert must say retraining is not the fix, or someone will try it.
    assert "retraining cannot fix" in sent["body"]


def test_healthy_pipeline_sends_nothing(stub_timestamps, config, monkeypatch):
    stub_timestamps["status_snapshots"] = _minutes_ago(2)
    stub_timestamps["hourly_departures"] = _minutes_ago(20)

    def fail(*args, **kwargs):
        raise AssertionError("must not alert when the pipeline is healthy")

    monkeypatch.setattr("src.alerting.email.send_email", fail)
    assert alert_on_stalled_pipeline(config).healthy


def test_alert_failure_does_not_mask_the_finding(stub_timestamps, config, monkeypatch):
    """If the email fails, the unhealthy report must still be returned."""
    stub_timestamps["status_snapshots"] = _minutes_ago(42 * 60)
    stub_timestamps["hourly_departures"] = _minutes_ago(42 * 60)

    def explode(*args, **kwargs):
        raise RuntimeError("smtp down")

    monkeypatch.setattr("src.alerting.email.send_email", explode)
    assert not alert_on_stalled_pipeline(config).healthy


def test_liveness_alerting_can_be_disabled(stub_timestamps, config, monkeypatch):
    stub_timestamps["status_snapshots"] = _minutes_ago(42 * 60)
    stub_timestamps["hourly_departures"] = _minutes_ago(42 * 60)

    def fail(*args, **kwargs):
        raise AssertionError("alerting is disabled")

    monkeypatch.setattr("src.alerting.email.send_email", fail)
    monkeypatch.setitem(config["monitoring"]["liveness"], "alert_enabled", False)
    try:
        assert not alert_on_stalled_pipeline(config).healthy
    finally:
        monkeypatch.setitem(config["monitoring"]["liveness"], "alert_enabled", True)
