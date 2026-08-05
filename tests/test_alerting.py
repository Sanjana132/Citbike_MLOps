"""Email alerts on model promotion.

Two properties matter more than the message contents: an alert failure must
never break a promotion, and the code must never claim to have sent something it
did not.
"""

from __future__ import annotations

import pytest

from src.alerting.email import (
    _format_promotion_body,
    alert_recipients,
    load_smtp_settings,
    send_email,
    send_promotion_alert,
)

PROMOTION = {
    "model_name": "citibike-demand-forecaster",
    "model_version": "7",
    "model_kind": "lightgbm",
    "metric": "mae",
    "challenger_score": 21.5,
    "champion_score": 24.566,
    "relative_improvement": "12.48%",
    "reason": "lightgbm improves mae by 12.48%",
}


@pytest.fixture(autouse=True)
def clean_smtp_env(monkeypatch):
    for name in (
        "SMTP_HOST", "SMTP_PORT", "SMTP_USERNAME", "SMTP_PASSWORD",
        "SMTP_SENDER", "SMTP_USE_TLS", "ALERT_EMAIL_RECIPIENTS",
    ):
        monkeypatch.delenv(name, raising=False)


# --------------------------------------------------------------------------
# Recipients and configuration
# --------------------------------------------------------------------------

def test_recipients_come_from_config(config):
    assert "skomma18@umd.edu" in alert_recipients(config)


def test_env_overrides_configured_recipients(monkeypatch, config):
    """An operator can redirect alerts without editing a committed file."""
    monkeypatch.setenv("ALERT_EMAIL_RECIPIENTS", "a@example.com, b@example.com")
    assert alert_recipients(config) == ["a@example.com", "b@example.com"]


def test_smtp_is_unconfigured_by_default():
    """No credentials in the repo means no accidental sending."""
    assert not load_smtp_settings().is_configured


def test_smtp_settings_read_from_env(monkeypatch):
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("SMTP_PORT", "465")
    monkeypatch.setenv("SMTP_USERNAME", "bot@example.com")
    monkeypatch.setenv("SMTP_PASSWORD", "secret")

    settings = load_smtp_settings()
    assert settings.is_configured
    assert settings.port == 465
    # Sender falls back to the username, which is what most providers require.
    assert settings.sender == "bot@example.com"


# --------------------------------------------------------------------------
# Message contents
# --------------------------------------------------------------------------

def test_subject_and_body_carry_the_essentials():
    subject, body = _format_promotion_body(PROMOTION)
    assert "v7" in subject and "promoted" in subject.lower()
    for expected in ("citibike-demand-forecaster", "lightgbm", "24.566", "12.48%"):
        assert expected in body


def test_proxy_trained_models_are_flagged_in_the_alert():
    """A model trained on proxy labels must not look like one trained on truth."""
    _, body = _format_promotion_body(
        {**PROMOTION, "label_provenance": "blend_trip_csv_and_gbfs_proxy"}
    )
    assert "PROXY" in body
    assert "not a" in body and "true error" in body


def test_true_label_models_carry_no_proxy_warning():
    _, body = _format_promotion_body({**PROMOTION, "label_provenance": "trip_csv_true_labels"})
    assert "WARNING" not in body


# --------------------------------------------------------------------------
# Delivery honesty and failure isolation
# --------------------------------------------------------------------------

def test_unconfigured_smtp_reports_failure_rather_than_pretending():
    """Returning True here would make the logs claim a delivery that never happened."""
    assert send_email("subject", "body", ["someone@example.com"]) is False


def test_no_recipients_is_not_a_send():
    assert send_email("subject", "body", []) is False


def test_send_uses_starttls_and_delivers(monkeypatch):
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("SMTP_SENDER", "bot@example.com")
    monkeypatch.setenv("SMTP_USERNAME", "bot@example.com")
    monkeypatch.setenv("SMTP_PASSWORD", "secret")

    sent = {}

    class FakeSMTP:
        def __init__(self, host, port, timeout=None):
            sent["host"], sent["port"] = host, port

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def starttls(self, context=None):
            sent["starttls"] = True

        def login(self, user, password):
            sent["login"] = user

        def send_message(self, message):
            sent["to"] = message["To"]
            sent["subject"] = message["Subject"]

    monkeypatch.setattr("smtplib.SMTP", FakeSMTP)

    assert send_email("hello", "body", ["skomma18@umd.edu"]) is True
    assert sent["to"] == "skomma18@umd.edu"
    assert sent["starttls"] is True
    assert sent["port"] == 587


def test_smtp_failure_is_swallowed_and_reported(monkeypatch):
    """A mail outage must never propagate into the promotion path."""
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("SMTP_SENDER", "bot@example.com")

    def explode(*args, **kwargs):
        raise OSError("connection refused")

    monkeypatch.setattr("smtplib.SMTP", explode)
    assert send_email("hello", "body", ["skomma18@umd.edu"]) is False


def test_alerting_can_be_disabled(monkeypatch, config):
    monkeypatch.setitem(config["alerting"]["email"], "enabled", False)
    assert send_promotion_alert(PROMOTION, config) is False
    monkeypatch.setitem(config["alerting"]["email"], "enabled", True)


def test_promotion_still_succeeds_when_alerting_raises(monkeypatch):
    """The model is the deliverable; the notification is a courtesy."""
    from src.models import promote as promote_module

    def boom(*args, **kwargs):
        raise RuntimeError("alerting subsystem exploded")

    monkeypatch.setattr("src.alerting.email.send_promotion_alert", boom)

    promoted = {}

    class FakeRegistry:
        model_name = "citibike-demand-forecaster"

        def __init__(self, cfg=None):
            pass

        def champion_metric(self, metric):
            return 30.0

        def promote(self, candidate, decision):
            promoted["name"] = candidate.name
            return "9"

    monkeypatch.setattr(promote_module, "ModelRegistry", FakeRegistry)

    decision = promote_module.run_promotion(
        [promote_module.Candidate("lightgbm", "run-1", {"mae": 20.0}, "uri")]
    )
    assert decision.promote is True
    assert promoted["name"] == "lightgbm"
