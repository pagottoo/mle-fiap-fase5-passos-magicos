"""
Módulo de monitoramento e logging
"""
from .logger import setup_logging, log_prediction
from .metrics import MetricsCollector
from .drift import DriftDetector
from .alerts import (
    AlertManager,
    AlertSeverity,
    AlertType,
    Alert,
    SlackChannel,
    EmailChannel,
    WebhookChannel,
    ConsoleChannel,
    get_alert_manager,
    send_alert
)

__all__ = [
    "setup_logging",
    "log_prediction",
    "MetricsCollector",
    "DriftDetector",
    "AlertManager",
    "AlertSeverity",
    "AlertType",
    "Alert",
    "SlackChannel",
    "EmailChannel",
    "WebhookChannel",
    "ConsoleChannel",
    "get_alert_manager",
    "send_alert"
]
