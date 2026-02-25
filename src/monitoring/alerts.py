"""
Alert System - notifications for MLOps events.

Supports multiple channels:
- Slack (webhook)
- Generic webhook
- Console (development)

Monitored events:
- Data drift detected
- Prediction drift detected
- Model promoted
- Training completed
- Critical errors
"""
import os
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import urllib.request
import urllib.error

import structlog

logger = structlog.get_logger()


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertType(Enum):
    """Supported alert types."""
    DATA_DRIFT = "data_drift"
    PREDICTION_DRIFT = "prediction_drift"
    MODEL_PERFORMANCE = "model_performance"
    MODEL_PROMOTED = "model_promoted"
    TRAINING_COMPLETE = "training_complete"
    TRAINING_FAILED = "training_failed"
    API_ERROR = "api_error"
    SYSTEM_ERROR = "system_error"
    CUSTOM = "custom"


@dataclass
class Alert:
    """Represents one alert event."""
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = "passos-magicos"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "alert_type": self.alert_type.value,
            "severity": self.severity.value,
            "title": self.title,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "source": self.source
        }
    
    def to_slack_blocks(self) -> List[Dict]:
        """Format alert payload for Slack Block Kit."""
        severity_emoji = {
            AlertSeverity.INFO: "ℹ️",
            AlertSeverity.WARNING: "[WARN]",
            AlertSeverity.ERROR: "[DRIFT]",
            AlertSeverity.CRITICAL: "🚨"
        }
        
        emoji = severity_emoji.get(self.severity, "📢")
        
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{emoji} {self.title}",
                    "emoji": True
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": self.message
                }
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Type:* {self.alert_type.value} | *Severity:* {self.severity.value} | *Time:* {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
                    }
                ]
            }
        ]
        
        # Attach metadata section if available
        if self.metadata:
            fields = []
            for key, value in list(self.metadata.items())[:10]:  # Slack limit: 10 fields
                fields.append({
                    "type": "mrkdwn",
                    "text": f"*{key}:* {value}"
                })
            
            if fields:
                blocks.append({
                    "type": "section",
                    "fields": fields[:10]
                })
        
        blocks.append({"type": "divider"})
        
        return blocks


class AlertChannel(ABC):
    """Base interface for alert channels."""
    
    @abstractmethod
    def send(self, alert: Alert) -> bool:
        """Send alert using this channel."""
        pass
    
    @abstractmethod
    def is_configured(self) -> bool:
        """Return whether this channel is configured."""
        pass


class SlackChannel(AlertChannel):
    """Slack webhook alert channel."""
    
    def __init__(self, webhook_url: Optional[str] = None):
        self.webhook_url = webhook_url or os.getenv("SLACK_WEBHOOK_URL")
    
    def is_configured(self) -> bool:
        return bool(self.webhook_url)
    
    def send(self, alert: Alert) -> bool:
        if not self.is_configured():
            logger.warning("slack_not_configured")
            return False
        
        try:
            payload = {
                "blocks": alert.to_slack_blocks(),
                "text": f"{alert.title}: {alert.message}"  # Fallback
            }
            
            data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(
                self.webhook_url,
                data=data,
                headers={'Content-Type': 'application/json'}
            )
            
            with urllib.request.urlopen(req, timeout=10) as response:
                if response.status == 200:
                    logger.info("slack_alert_sent", alert_type=alert.alert_type.value)
                    return True
                else:
                    logger.error("slack_alert_failed", status=response.status)
                    return False
                    
        except urllib.error.URLError as e:
            logger.error("slack_alert_error", error=str(e))
            return False
        except Exception as e:
            logger.error("slack_alert_exception", error=str(e))
            return False


class WebhookChannel(AlertChannel):
    """Generic webhook alert channel."""
    
    def __init__(self, webhook_url: Optional[str] = None, headers: Optional[Dict] = None):
        self.webhook_url = webhook_url or os.getenv("ALERT_WEBHOOK_URL")
        self.headers = headers or {"Content-Type": "application/json"}
    
    def is_configured(self) -> bool:
        return bool(self.webhook_url)
    
    def send(self, alert: Alert) -> bool:
        if not self.is_configured():
            logger.warning("webhook_not_configured")
            return False
        
        try:
            data = json.dumps(alert.to_dict()).encode('utf-8')
            req = urllib.request.Request(
                self.webhook_url,
                data=data,
                headers=self.headers
            )
            
            with urllib.request.urlopen(req, timeout=10) as response:
                if response.status in [200, 201, 202]:
                    logger.info("webhook_alert_sent", alert_type=alert.alert_type.value)
                    return True
                else:
                    logger.error("webhook_alert_failed", status=response.status)
                    return False
                    
        except Exception as e:
            logger.error("webhook_alert_error", error=str(e))
            return False


class ConsoleChannel(AlertChannel):
    """Console alert channel (development)."""
    
    def is_configured(self) -> bool:
        return True
    
    def send(self, alert: Alert) -> bool:
        logger.info(
            "console_alert",
            alert_type=alert.alert_type.value,
            severity=alert.severity.value,
            title=alert.title,
            message=alert.message,
            timestamp=alert.timestamp.isoformat(),
            metadata=alert.metadata or {},
            source=alert.source,
        )
        return True


class AlertManager:
    """
    Central alert manager.

    Usage example:
        manager = AlertManager()
        manager.add_channel(SlackChannel())
        
        manager.send_alert(
            alert_type=AlertType.DATA_DRIFT,
            severity=AlertSeverity.WARNING,
            title="Data Drift Detected",
            message="Feature 'inde' exceeded drift threshold",
            metadata={"feature": "inde", "drift_score": 0.15}
        )
    """
    
    def __init__(self, auto_configure: bool = True):
        """
        Initialize AlertManager.

        Args:
            auto_configure: If True, auto-configure channels from env vars.
        """
        self.channels: List[AlertChannel] = []
        self.alert_history: List[Alert] = []
        self.max_history = 1000
        
        if auto_configure:
            self._auto_configure_channels()
        
        logger.info("alert_manager_initialized", channels=len(self.channels))
    
    def _auto_configure_channels(self):
        """Auto-configure channels based on environment variables."""
        # Always add console channel in development
        if os.getenv("ENVIRONMENT", "development") == "development":
            self.add_channel(ConsoleChannel())
        
        # Slack
        if os.getenv("SLACK_WEBHOOK_URL"):
            self.add_channel(SlackChannel())
        
        # Generic webhook
        if os.getenv("ALERT_WEBHOOK_URL"):
            self.add_channel(WebhookChannel())
    
    def add_channel(self, channel: AlertChannel) -> None:
        """Add one alert channel."""
        if channel.is_configured():
            self.channels.append(channel)
            logger.info("alert_channel_added", channel=channel.__class__.__name__)
        else:
            logger.warning("alert_channel_not_configured", channel=channel.__class__.__name__)
    
    def remove_channel(self, channel_type: type) -> None:
        """Remove channels by type."""
        self.channels = [c for c in self.channels if not isinstance(c, channel_type)]
    
    def send_alert(
        self,
        alert_type: AlertType,
        severity: AlertSeverity,
        title: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, bool]:
        """
        Send alert to all configured channels.

        Returns:
            Delivery result by channel name.
        """
        alert = Alert(
            alert_type=alert_type,
            severity=severity,
            title=title,
            message=message,
            metadata=metadata or {}
        )
        
        # Persist in in-memory history
        self.alert_history.append(alert)
        if len(self.alert_history) > self.max_history:
            self.alert_history = self.alert_history[-self.max_history:]
        
        # Dispatch to all channels
        results = {}
        for channel in self.channels:
            channel_name = channel.__class__.__name__
            try:
                results[channel_name] = channel.send(alert)
            except Exception as e:
                logger.error("alert_send_error", channel=channel_name, error=str(e))
                results[channel_name] = False
        
        logger.info(
            "alert_sent",
            alert_type=alert_type.value,
            severity=severity.value,
            channels=len(self.channels),
            success=sum(results.values())
        )
        
        return results
    
    # Convenience helpers for common alert categories
    
    def alert_data_drift(
        self,
        feature: str,
        drift_score: float,
        threshold: float,
        severity: AlertSeverity = AlertSeverity.WARNING
    ) -> Dict[str, bool]:
        """Send data drift alert."""
        return self.send_alert(
            alert_type=AlertType.DATA_DRIFT,
            severity=severity,
            title=f"Data Drift Detected: {feature}",
            message=f"Feature '{feature}' exceeded drift threshold. Score: {drift_score:.4f} (threshold: {threshold})",
            metadata={
                "feature": feature,
                "drift_score": round(drift_score, 4),
                "threshold": threshold,
                "exceeded_by": round(drift_score - threshold, 4)
            }
        )
    
    def alert_prediction_drift(
        self,
        drift_detected: bool,
        current_distribution: Dict[str, float],
        severity: AlertSeverity = AlertSeverity.WARNING
    ) -> Dict[str, bool]:
        """Send prediction drift alert."""
        return self.send_alert(
            alert_type=AlertType.PREDICTION_DRIFT,
            severity=severity,
            title="Prediction Drift Detected",
            message="Prediction distribution changed significantly from baseline.",
            metadata={
                "drift_detected": drift_detected,
                **{f"class_{k}": f"{v:.2%}" for k, v in current_distribution.items()}
            }
        )
    
    def alert_model_performance(
        self,
        metric_name: str,
        current_value: float,
        expected_value: float,
        severity: AlertSeverity = AlertSeverity.WARNING
    ) -> Dict[str, bool]:
        """Send model performance degradation alert."""
        return self.send_alert(
            alert_type=AlertType.MODEL_PERFORMANCE,
            severity=severity,
            title=f"Performance Degradation: {metric_name}",
            message=f"Metric '{metric_name}' dropped below expectation. Current: {current_value:.4f}, Expected: {expected_value:.4f}",
            metadata={
                "metric": metric_name,
                "current_value": round(current_value, 4),
                "expected_value": round(expected_value, 4),
                "difference": round(expected_value - current_value, 4)
            }
        )
    
    def alert_model_promoted(
        self,
        model_name: str,
        version: int,
        stage: str,
        metrics: Optional[Dict[str, float]] = None
    ) -> Dict[str, bool]:
        """Send model promotion alert."""
        return self.send_alert(
            alert_type=AlertType.MODEL_PROMOTED,
            severity=AlertSeverity.INFO,
            title=f"Model Promoted: {model_name}",
            message=f"Model '{model_name}' version {version} was promoted to {stage}.",
            metadata={
                "model_name": model_name,
                "version": version,
                "stage": stage,
                **(metrics or {})
            }
        )
    
    def alert_training_complete(
        self,
        model_name: str,
        metrics: Dict[str, float],
        duration_seconds: float
    ) -> Dict[str, bool]:
        """Send training completed alert."""
        return self.send_alert(
            alert_type=AlertType.TRAINING_COMPLETE,
            severity=AlertSeverity.INFO,
            title=f"Training Completed: {model_name}",
            message=f"Training for model '{model_name}' completed successfully in {duration_seconds:.1f}s.",
            metadata={
                "model_name": model_name,
                "duration_seconds": round(duration_seconds, 1),
                **{f"metric_{k}": round(v, 4) for k, v in metrics.items()}
            }
        )
    
    def alert_training_failed(
        self,
        model_name: str,
        error: str,
        severity: AlertSeverity = AlertSeverity.ERROR
    ) -> Dict[str, bool]:
        """Send training failure alert."""
        return self.send_alert(
            alert_type=AlertType.TRAINING_FAILED,
            severity=severity,
            title=f"Training Failed: {model_name}",
            message=f"Training for model '{model_name}' failed.",
            metadata={
                "model_name": model_name,
                "error": error[:500]  # Limit error length
            }
        )
    
    def alert_api_error(
        self,
        endpoint: str,
        error: str,
        status_code: Optional[int] = None,
        severity: AlertSeverity = AlertSeverity.ERROR
    ) -> Dict[str, bool]:
        """Send API error alert."""
        return self.send_alert(
            alert_type=AlertType.API_ERROR,
            severity=severity,
            title=f"API Error: {endpoint}",
            message=f"Error on endpoint '{endpoint}': {error}",
            metadata={
                "endpoint": endpoint,
                "error": error[:500],
                "status_code": status_code
            }
        )
    
    def get_history(
        self,
        alert_type: Optional[AlertType] = None,
        severity: Optional[AlertSeverity] = None,
        limit: int = 100
    ) -> List[Dict]:
        """Return filtered alert history."""
        alerts = self.alert_history
        
        if alert_type:
            alerts = [a for a in alerts if a.alert_type == alert_type]
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        return [a.to_dict() for a in alerts[-limit:]]
    
    def get_status(self) -> Dict[str, Any]:
        """Return current alert system status."""
        return {
            "channels": [
                {
                    "name": c.__class__.__name__,
                    "configured": c.is_configured()
                }
                for c in self.channels
            ],
            "total_channels": len(self.channels),
            "configured_channels": sum(1 for c in self.channels if c.is_configured()),
            "alerts_in_history": len(self.alert_history),
            "recent_alerts": self.get_history(limit=5)
        }


# Global singleton used across the application
_alert_manager: Optional[AlertManager] = None


def get_alert_manager() -> AlertManager:
    """Return AlertManager singleton instance."""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager


def send_alert(
    alert_type: AlertType,
    severity: AlertSeverity,
    title: str,
    message: str,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, bool]:
    """Convenience function to send alerts."""
    return get_alert_manager().send_alert(alert_type, severity, title, message, metadata)
