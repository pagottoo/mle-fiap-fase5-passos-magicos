"""
Testes extras para aumentar cobertura do módulo de alertas.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.monitoring.alerts import (
    Alert,
    AlertManager,
    AlertSeverity,
    AlertType,
    AlertChannel,
    ConsoleChannel,
    EmailChannel,
    SlackChannel,
    WebhookChannel,
)


class _DummyResponse:
    def __init__(self, status: int):
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _BrokenChannel(AlertChannel):
    def is_configured(self) -> bool:
        return True

    def send(self, alert: Alert) -> bool:
        raise RuntimeError("channel failed")


class TestAlertFormatting:
    def test_to_slack_blocks_includes_metadata_and_divider(self):
        alert = Alert(
            alert_type=AlertType.DATA_DRIFT,
            severity=AlertSeverity.WARNING,
            title="Drift",
            message="Detectado",
            metadata={"feature": "INDE", "score": 0.22},
        )

        blocks = alert.to_slack_blocks()
        assert isinstance(blocks, list)
        assert blocks[-1]["type"] == "divider"
        assert any("fields" in block for block in blocks if block["type"] == "section")

    def test_to_email_html_renders_metadata(self):
        alert = Alert(
            alert_type=AlertType.CUSTOM,
            severity=AlertSeverity.INFO,
            title="Teste",
            message="Mensagem",
            metadata={"k": "v"},
        )

        html = alert.to_email_html()
        assert "Teste" in html
        assert "Mensagem" in html
        assert "Details" in html
        assert "k" in html
        assert "v" in html


class TestSlackChannel:
    def test_send_returns_false_when_not_configured(self):
        channel = SlackChannel(webhook_url=None)
        alert = Alert(AlertType.CUSTOM, AlertSeverity.INFO, "t", "m")
        assert channel.send(alert) is False

    def test_send_success(self):
        channel = SlackChannel(webhook_url="https://example.com/webhook")
        alert = Alert(AlertType.CUSTOM, AlertSeverity.INFO, "t", "m")

        with patch("src.monitoring.alerts.urllib.request.urlopen", return_value=_DummyResponse(200)):
            assert channel.send(alert) is True

    def test_send_non_200(self):
        channel = SlackChannel(webhook_url="https://example.com/webhook")
        alert = Alert(AlertType.CUSTOM, AlertSeverity.INFO, "t", "m")

        with patch("src.monitoring.alerts.urllib.request.urlopen", return_value=_DummyResponse(500)):
            assert channel.send(alert) is False

    def test_send_url_error(self):
        channel = SlackChannel(webhook_url="https://example.com/webhook")
        alert = Alert(AlertType.CUSTOM, AlertSeverity.INFO, "t", "m")

        with patch(
            "src.monitoring.alerts.urllib.request.urlopen",
            side_effect=Exception("network down"),
        ):
            assert channel.send(alert) is False


class TestEmailChannel:
    def test_not_configured_returns_false(self):
        channel = EmailChannel(
            smtp_host=None,
            smtp_user=None,
            smtp_password=None,
            from_email=None,
            to_emails=[],
        )
        alert = Alert(AlertType.CUSTOM, AlertSeverity.INFO, "t", "m")
        assert channel.send(alert) is False

    def test_send_success(self):
        channel = EmailChannel(
            smtp_host="smtp.example.com",
            smtp_port=587,
            smtp_user="user",
            smtp_password="pass",
            from_email="from@example.com",
            to_emails=["to@example.com"],
        )
        alert = Alert(AlertType.CUSTOM, AlertSeverity.INFO, "t", "m")

        smtp_server = MagicMock()
        smtp_cm = MagicMock()
        smtp_cm.__enter__.return_value = smtp_server
        smtp_cm.__exit__.return_value = False

        with patch("src.monitoring.alerts.smtplib.SMTP", return_value=smtp_cm):
            assert channel.send(alert) is True

        smtp_server.starttls.assert_called_once()
        smtp_server.login.assert_called_once_with("user", "pass")
        smtp_server.sendmail.assert_called_once()

    def test_send_exception_returns_false(self):
        channel = EmailChannel(
            smtp_host="smtp.example.com",
            smtp_port=587,
            smtp_user="user",
            smtp_password="pass",
            from_email="from@example.com",
            to_emails=["to@example.com"],
        )
        alert = Alert(AlertType.CUSTOM, AlertSeverity.INFO, "t", "m")

        with patch("src.monitoring.alerts.smtplib.SMTP", side_effect=RuntimeError("smtp error")):
            assert channel.send(alert) is False


class TestWebhookChannel:
    def test_send_returns_false_when_not_configured(self):
        channel = WebhookChannel(webhook_url=None)
        alert = Alert(AlertType.CUSTOM, AlertSeverity.INFO, "t", "m")
        assert channel.send(alert) is False

    def test_send_success(self):
        channel = WebhookChannel(webhook_url="https://example.com/hook")
        alert = Alert(AlertType.CUSTOM, AlertSeverity.INFO, "t", "m")

        with patch("src.monitoring.alerts.urllib.request.urlopen", return_value=_DummyResponse(202)):
            assert channel.send(alert) is True

    def test_send_non_success_status(self):
        channel = WebhookChannel(webhook_url="https://example.com/hook")
        alert = Alert(AlertType.CUSTOM, AlertSeverity.INFO, "t", "m")

        with patch("src.monitoring.alerts.urllib.request.urlopen", return_value=_DummyResponse(500)):
            assert channel.send(alert) is False

    def test_send_exception(self):
        channel = WebhookChannel(webhook_url="https://example.com/hook")
        alert = Alert(AlertType.CUSTOM, AlertSeverity.INFO, "t", "m")

        with patch(
            "src.monitoring.alerts.urllib.request.urlopen",
            side_effect=RuntimeError("boom"),
        ):
            assert channel.send(alert) is False


class TestAlertManagerExtended:
    def test_auto_configure_adds_multiple_channels(self, monkeypatch):
        monkeypatch.setenv("ENVIRONMENT", "development")
        monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://example.com")
        monkeypatch.setenv("SMTP_USER", "user")
        monkeypatch.setenv("SMTP_PASSWORD", "pass")
        monkeypatch.setenv("ALERT_FROM_EMAIL", "from@example.com")
        monkeypatch.setenv("ALERT_TO_EMAILS", "to@example.com")
        monkeypatch.setenv("ALERT_WEBHOOK_URL", "https://example.com/hook")

        manager = AlertManager(auto_configure=True)
        # Console + Slack + Email + Webhook
        assert len(manager.channels) >= 3

    def test_remove_channel(self):
        manager = AlertManager(auto_configure=False)
        manager.add_channel(ConsoleChannel())
        assert len(manager.channels) == 1

        manager.remove_channel(ConsoleChannel)
        assert len(manager.channels) == 0

    def test_send_alert_handles_channel_exception(self):
        manager = AlertManager(auto_configure=False)
        manager.add_channel(_BrokenChannel())

        result = manager.send_alert(
            alert_type=AlertType.CUSTOM,
            severity=AlertSeverity.ERROR,
            title="x",
            message="y",
        )
        assert result["_BrokenChannel"] is False

    def test_history_filters_and_max_history(self):
        manager = AlertManager(auto_configure=False)
        manager.add_channel(ConsoleChannel())
        manager.max_history = 2

        manager.send_alert(AlertType.CUSTOM, AlertSeverity.INFO, "a", "a")
        manager.send_alert(AlertType.DATA_DRIFT, AlertSeverity.WARNING, "b", "b")
        manager.send_alert(AlertType.API_ERROR, AlertSeverity.ERROR, "c", "c")

        history = manager.get_history(limit=10)
        assert len(history) == 2

        filtered = manager.get_history(alert_type=AlertType.API_ERROR, severity=AlertSeverity.ERROR, limit=10)
        assert len(filtered) == 1

    def test_convenience_alert_methods(self):
        manager = AlertManager(auto_configure=False)
        manager.add_channel(ConsoleChannel())

        r1 = manager.alert_data_drift("INDE", 0.4, 0.1)
        r2 = manager.alert_prediction_drift(True, {"0": 0.2, "1": 0.8})
        r3 = manager.alert_model_performance("f1", 0.7, 0.8)
        r4 = manager.alert_model_promoted("m", 2, "Production", {"f1": 0.9})
        r5 = manager.alert_training_complete("m", {"f1": 0.9}, 10.5)
        r6 = manager.alert_training_failed("m", "stacktrace")
        r7 = manager.alert_api_error("/predict", "boom", status_code=500)

        for result in [r1, r2, r3, r4, r5, r6, r7]:
            assert "ConsoleChannel" in result
