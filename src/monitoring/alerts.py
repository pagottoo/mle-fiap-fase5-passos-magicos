"""
Alert System - Notificações para eventos de MLOps

Suporta múltiplos canais:
- Slack (webhook)
- Email (SMTP)
- Webhook genérico
- Console (para desenvolvimento)

Eventos monitorados:
- Data drift detectado
- Prediction drift detectado
- Modelo promovido
- Treinamento concluído
- Erros críticos
"""
import os
import json
import smtplib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from enum import Enum
from typing import Any, Dict, List, Optional
import urllib.request
import urllib.error

import structlog

logger = structlog.get_logger()


class AlertSeverity(Enum):
    """Níveis de severidade dos alertas."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertType(Enum):
    """Tipos de alertas do sistema."""
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
    """Representa um alerta do sistema."""
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = "passos-magicos"
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte alerta para dicionário."""
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
        """Formata alerta para Slack Block Kit."""
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
        
        # Adicionar metadata se existir
        if self.metadata:
            fields = []
            for key, value in list(self.metadata.items())[:10]:  # Limite de 10 campos
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
    
    def to_email_html(self) -> str:
        """Formata alerta para email HTML."""
        severity_color = {
            AlertSeverity.INFO: "#17a2b8",
            AlertSeverity.WARNING: "#ffc107",
            AlertSeverity.ERROR: "#dc3545",
            AlertSeverity.CRITICAL: "#721c24"
        }
        
        color = severity_color.get(self.severity, "#6c757d")
        
        metadata_html = ""
        if self.metadata:
            metadata_items = "".join([
                f"<tr><td><strong>{k}</strong></td><td>{v}</td></tr>"
                for k, v in self.metadata.items()
            ])
            metadata_html = f"""
            <h3>Details</h3>
            <table border="1" cellpadding="5" style="border-collapse: collapse;">
                {metadata_items}
            </table>
            """
        
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .alert-box {{ 
                    border-left: 4px solid {color}; 
                    padding: 15px; 
                    background-color: #f8f9fa;
                    margin-bottom: 20px;
                }}
                .severity {{ 
                    display: inline-block;
                    padding: 3px 10px;
                    background-color: {color};
                    color: white;
                    border-radius: 3px;
                    font-size: 12px;
                }}
                table {{ width: 100%; margin-top: 10px; }}
                th, td {{ text-align: left; padding: 8px; }}
            </style>
        </head>
        <body>
            <h2>{self.title}</h2>
            <div class="alert-box">
                <span class="severity">{self.severity.value.upper()}</span>
                <p>{self.message}</p>
            </div>
            <p><strong>Type:</strong> {self.alert_type.value}</p>
            <p><strong>Time:</strong> {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Source:</strong> {self.source}</p>
            {metadata_html}
            <hr>
            <p style="color: #6c757d; font-size: 12px;">
                This alert was sent by Passos Mágicos MLOps System
            </p>
        </body>
        </html>
        """


class AlertChannel(ABC):
    """Interface base para canais de alerta."""
    
    @abstractmethod
    def send(self, alert: Alert) -> bool:
        """Envia um alerta pelo canal."""
        pass
    
    @abstractmethod
    def is_configured(self) -> bool:
        """Verifica se o canal está configurado."""
        pass


class SlackChannel(AlertChannel):
    """Canal de alertas via Slack Webhook."""
    
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


class EmailChannel(AlertChannel):
    """Canal de alertas via Email SMTP."""
    
    def __init__(
        self,
        smtp_host: Optional[str] = None,
        smtp_port: Optional[int] = None,
        smtp_user: Optional[str] = None,
        smtp_password: Optional[str] = None,
        from_email: Optional[str] = None,
        to_emails: Optional[List[str]] = None
    ):
        self.smtp_host = smtp_host or os.getenv("SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = smtp_port or int(os.getenv("SMTP_PORT", "587"))
        self.smtp_user = smtp_user or os.getenv("SMTP_USER")
        self.smtp_password = smtp_password or os.getenv("SMTP_PASSWORD")
        self.from_email = from_email or os.getenv("ALERT_FROM_EMAIL")
        
        to_env = os.getenv("ALERT_TO_EMAILS", "")
        self.to_emails = to_emails or [e.strip() for e in to_env.split(",") if e.strip()]
    
    def is_configured(self) -> bool:
        return all([
            self.smtp_host,
            self.smtp_user,
            self.smtp_password,
            self.from_email,
            self.to_emails
        ])
    
    def send(self, alert: Alert) -> bool:
        if not self.is_configured():
            logger.warning("email_not_configured")
            return False
        
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
            msg['From'] = self.from_email
            msg['To'] = ", ".join(self.to_emails)
            
            # Versão texto simples
            text_content = f"{alert.title}\n\n{alert.message}\n\nType: {alert.alert_type.value}\nSeverity: {alert.severity.value}"
            
            # Versão HTML
            html_content = alert.to_email_html()
            
            msg.attach(MIMEText(text_content, 'plain'))
            msg.attach(MIMEText(html_content, 'html'))
            
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.sendmail(self.from_email, self.to_emails, msg.as_string())
            
            logger.info("email_alert_sent", alert_type=alert.alert_type.value, to=self.to_emails)
            return True
            
        except Exception as e:
            logger.error("email_alert_error", error=str(e))
            return False


class WebhookChannel(AlertChannel):
    """Canal de alertas via Webhook genérico."""
    
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
    """Canal de alertas para console (desenvolvimento)."""
    
    def is_configured(self) -> bool:
        return True
    
    def send(self, alert: Alert) -> bool:
        severity_icon = {
            AlertSeverity.INFO: "ℹ️",
            AlertSeverity.WARNING: "[WARN]",
            AlertSeverity.ERROR: "[DRIFT]",
            AlertSeverity.CRITICAL: "🚨"
        }
        
        icon = severity_icon.get(alert.severity, "📢")
        
        print(f"\n{'='*60}")
        print(f"{icon} ALERT: {alert.title}")
        print(f"{'='*60}")
        print(f"Type: {alert.alert_type.value}")
        print(f"Severity: {alert.severity.value}")
        print(f"Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\n{alert.message}")
        
        if alert.metadata:
            print(f"\nMetadata:")
            for key, value in alert.metadata.items():
                print(f"  - {key}: {value}")
        
        print(f"{'='*60}\n")
        
        logger.info("console_alert", alert_type=alert.alert_type.value, title=alert.title)
        return True


class AlertManager:
    """
    Gerenciador central de alertas.
    
    Exemplo de uso:
        manager = AlertManager()
        manager.add_channel(SlackChannel())
        manager.add_channel(EmailChannel())
        
        manager.send_alert(
            alert_type=AlertType.DATA_DRIFT,
            severity=AlertSeverity.WARNING,
            title="Data Drift Detectado",
            message="Feature 'inde' apresentou drift significativo",
            metadata={"feature": "inde", "drift_score": 0.15}
        )
    """
    
    def __init__(self, auto_configure: bool = True):
        """
        Inicializa o AlertManager.
        
        Args:
            auto_configure: Se True, configura canais automaticamente via env vars
        """
        self.channels: List[AlertChannel] = []
        self.alert_history: List[Alert] = []
        self.max_history = 1000
        
        if auto_configure:
            self._auto_configure_channels()
        
        logger.info("alert_manager_initialized", channels=len(self.channels))
    
    def _auto_configure_channels(self):
        """Configura canais automaticamente baseado em variáveis de ambiente."""
        # Sempre adicionar console em desenvolvimento
        if os.getenv("ENVIRONMENT", "development") == "development":
            self.add_channel(ConsoleChannel())
        
        # Slack
        if os.getenv("SLACK_WEBHOOK_URL"):
            self.add_channel(SlackChannel())
        
        # Email
        if os.getenv("SMTP_USER") and os.getenv("SMTP_PASSWORD"):
            self.add_channel(EmailChannel())
        
        # Webhook genérico
        if os.getenv("ALERT_WEBHOOK_URL"):
            self.add_channel(WebhookChannel())
    
    def add_channel(self, channel: AlertChannel) -> None:
        """Adiciona um canal de alertas."""
        if channel.is_configured():
            self.channels.append(channel)
            logger.info("alert_channel_added", channel=channel.__class__.__name__)
        else:
            logger.warning("alert_channel_not_configured", channel=channel.__class__.__name__)
    
    def remove_channel(self, channel_type: type) -> None:
        """Remove canais de um tipo específico."""
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
        Envia alerta para todos os canais configurados.
        
        Returns:
            Dicionário com resultado por canal
        """
        alert = Alert(
            alert_type=alert_type,
            severity=severity,
            title=title,
            message=message,
            metadata=metadata or {}
        )
        
        # Armazenar no histórico
        self.alert_history.append(alert)
        if len(self.alert_history) > self.max_history:
            self.alert_history = self.alert_history[-self.max_history:]
        
        # Enviar para todos os canais
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
    
    # Métodos de conveniência para tipos específicos de alertas
    
    def alert_data_drift(
        self,
        feature: str,
        drift_score: float,
        threshold: float,
        severity: AlertSeverity = AlertSeverity.WARNING
    ) -> Dict[str, bool]:
        """Alerta de data drift."""
        return self.send_alert(
            alert_type=AlertType.DATA_DRIFT,
            severity=severity,
            title=f"Data Drift Detectado: {feature}",
            message=f"A feature '{feature}' apresentou drift significativo. Score: {drift_score:.4f} (threshold: {threshold})",
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
        """Alerta de prediction drift."""
        return self.send_alert(
            alert_type=AlertType.PREDICTION_DRIFT,
            severity=severity,
            title="Prediction Drift Detectado",
            message="A distribuição das predições mudou significativamente em relação à referência.",
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
        """Alerta de degradação de performance do modelo."""
        return self.send_alert(
            alert_type=AlertType.MODEL_PERFORMANCE,
            severity=severity,
            title=f"Performance Degradada: {metric_name}",
            message=f"A métrica '{metric_name}' caiu abaixo do esperado. Atual: {current_value:.4f}, Esperado: {expected_value:.4f}",
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
        """Alerta de modelo promovido."""
        return self.send_alert(
            alert_type=AlertType.MODEL_PROMOTED,
            severity=AlertSeverity.INFO,
            title=f"Modelo Promovido: {model_name}",
            message=f"O modelo '{model_name}' versão {version} foi promovido para {stage}.",
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
        """Alerta de treinamento concluído."""
        return self.send_alert(
            alert_type=AlertType.TRAINING_COMPLETE,
            severity=AlertSeverity.INFO,
            title=f"Treinamento Concluído: {model_name}",
            message=f"O treinamento do modelo '{model_name}' foi concluído com sucesso em {duration_seconds:.1f}s.",
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
        """Alerta de falha no treinamento."""
        return self.send_alert(
            alert_type=AlertType.TRAINING_FAILED,
            severity=severity,
            title=f"Treinamento Falhou: {model_name}",
            message=f"O treinamento do modelo '{model_name}' falhou com erro.",
            metadata={
                "model_name": model_name,
                "error": error[:500]  # Limitar tamanho do erro
            }
        )
    
    def alert_api_error(
        self,
        endpoint: str,
        error: str,
        status_code: Optional[int] = None,
        severity: AlertSeverity = AlertSeverity.ERROR
    ) -> Dict[str, bool]:
        """Alerta de erro na API."""
        return self.send_alert(
            alert_type=AlertType.API_ERROR,
            severity=severity,
            title=f"Erro na API: {endpoint}",
            message=f"Erro no endpoint '{endpoint}': {error}",
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
        """Retorna histórico de alertas filtrado."""
        alerts = self.alert_history
        
        if alert_type:
            alerts = [a for a in alerts if a.alert_type == alert_type]
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        return [a.to_dict() for a in alerts[-limit:]]
    
    def get_status(self) -> Dict[str, Any]:
        """Retorna status do sistema de alertas."""
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


# Singleton global para uso em toda a aplicação
_alert_manager: Optional[AlertManager] = None


def get_alert_manager() -> AlertManager:
    """Retorna instância singleton do AlertManager."""
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
    """Função de conveniência para enviar alertas."""
    return get_alert_manager().send_alert(alert_type, severity, title, message, metadata)
