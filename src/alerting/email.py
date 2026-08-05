"""Email alerts for model changes.

A promotion silently swaps the model behind a live API. That is exactly the kind
of event someone should hear about without having to watch a dashboard, so
:func:`send_promotion_alert` fires whenever the production alias moves.

Two rules govern this module:

* **It never breaks promotion.** Every failure path is caught and logged. A
  misconfigured SMTP host, an expired password, or a mail server outage must not
  turn a successful model promotion into a failed training run - the model is
  the deliverable, the notification is a courtesy.
* **It never invents delivery.** If SMTP is not configured, it says so and
  returns ``False`` rather than logging a cheerful "alert sent".

Configuration lives entirely in environment variables (see ``.env.example``);
no credential is ever read from ``config.yaml``, which is committed.
"""

from __future__ import annotations

import logging
import os
import smtplib
import ssl
from dataclasses import dataclass
from datetime import UTC, datetime
from email.message import EmailMessage

from src.config import Config, load_config

logger = logging.getLogger(__name__)


@dataclass
class SMTPSettings:
    """Resolved SMTP configuration, read from the environment."""

    host: str
    port: int
    username: str | None
    password: str | None
    sender: str
    use_tls: bool

    @property
    def is_configured(self) -> bool:
        return bool(self.host and self.sender)


def load_smtp_settings() -> SMTPSettings:
    return SMTPSettings(
        host=os.environ.get("SMTP_HOST", "").strip(),
        port=int(os.environ.get("SMTP_PORT", "587") or 587),
        username=os.environ.get("SMTP_USERNAME") or None,
        password=os.environ.get("SMTP_PASSWORD") or None,
        sender=(os.environ.get("SMTP_SENDER") or os.environ.get("SMTP_USERNAME") or "").strip(),
        use_tls=os.environ.get("SMTP_USE_TLS", "true").strip().lower() != "false",
    )


def alert_recipients(config: Config | None = None) -> list[str]:
    """Recipients for model-change alerts.

    ``ALERT_EMAIL_RECIPIENTS`` (comma separated) overrides the config list, so
    an operator can redirect alerts without editing a committed file.
    """
    cfg = config or load_config()
    override = os.environ.get("ALERT_EMAIL_RECIPIENTS", "").strip()
    if override:
        return [address.strip() for address in override.split(",") if address.strip()]
    configured = cfg.get_path("alerting.email.recipients", []) or []
    return [str(address).strip() for address in configured if str(address).strip()]


def _format_promotion_body(details: dict) -> tuple[str, str]:
    """Build the subject and body for a promotion alert."""
    model_name = details.get("model_name", "model")
    version = details.get("model_version", "?")
    kind = details.get("model_kind", "unknown")
    metric = details.get("metric", "mae")

    subject = f"[citibike-mlops] {model_name} v{version} promoted to production ({kind})"

    lines = [
        "A new model has been promoted to production.",
        "",
        f"  Model            : {model_name}",
        f"  New version      : {version}",
        f"  Model type       : {kind}",
        f"  Promoted at      : {datetime.now(tz=UTC).isoformat()}",
        "",
        "Comparison",
        f"  Metric           : {metric} (lower is better)",
        f"  Challenger score : {details.get('challenger_score')}",
        f"  Previous champion: {details.get('champion_score')}",
        f"  Improvement      : {details.get('relative_improvement')}",
        "",
        f"Reason: {details.get('reason', 'n/a')}",
        "",
    ]

    provenance = details.get("label_provenance")
    if provenance and "proxy" in str(provenance):
        lines += [
            "WARNING: this model was trained on labels that include the GBFS",
            "inventory-delta demand PROXY, not true completed-trip counts. Its",
            "reported accuracy is measured against that proxy and is not a",
            "true error figure. See the project README.",
            "",
        ]

    lines += [
        "Scores above are as-served: each candidate was loaded back through",
        "mlflow.pyfunc and scored through the path that actually serves it.",
        "",
        "-- citibike-demand-mlops",
    ]
    return subject, "\n".join(lines)


def send_email(subject: str, body: str, recipients: list[str]) -> bool:
    """Send a plain-text email. Returns True only if it was actually sent."""
    if not recipients:
        logger.info("No alert recipients configured; skipping email")
        return False

    settings = load_smtp_settings()
    if not settings.is_configured:
        logger.warning(
            "SMTP is not configured (set SMTP_HOST and SMTP_SENDER); "
            "would have emailed %s: %r",
            ", ".join(recipients), subject,
        )
        return False

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = settings.sender
    message["To"] = ", ".join(recipients)
    message.set_content(body)

    try:
        if settings.port == 465:
            with smtplib.SMTP_SSL(
                settings.host, settings.port, context=ssl.create_default_context(), timeout=30
            ) as server:
                if settings.username and settings.password:
                    server.login(settings.username, settings.password)
                server.send_message(message)
        else:
            with smtplib.SMTP(settings.host, settings.port, timeout=30) as server:
                if settings.use_tls:
                    server.starttls(context=ssl.create_default_context())
                if settings.username and settings.password:
                    server.login(settings.username, settings.password)
                server.send_message(message)
    except Exception as exc:  # noqa: BLE001 - a mail failure must never break promotion
        logger.error("Could not send alert email to %s: %s", ", ".join(recipients), exc)
        return False

    logger.info("Sent alert email to %s: %s", ", ".join(recipients), subject)
    return True


def send_promotion_alert(details: dict, config: Config | None = None) -> bool:
    """Notify recipients that the production model changed.

    Returns whether an email was actually delivered, so callers can log the
    truth rather than assume success.
    """
    cfg = config or load_config()
    if not bool(cfg.get_path("alerting.email.enabled", True)):
        logger.info("Email alerting is disabled in config; not sending")
        return False

    subject, body = _format_promotion_body(details)
    return send_email(subject, body, alert_recipients(cfg))
