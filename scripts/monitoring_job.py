"""
Job de monitoramento de drift e performance (in-cluster).

O propósito deste job em rodar in cluster é ter acesso direto aos
 logs de predição e ao endpoint de métricas da API, 
 sem precisar expor esses dados externamente. 

Ele pode ser agendado para rodar periodicamente (ex: a cada hora)
usando um CronJob do Kubernetes.

"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict

import requests

# Adicionar o diretório raiz do projeto ao path para importar "src"
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.monitoring import DriftDetector


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _send_slack_alert(
    title: str,
    message: str,
    metadata: Dict[str, Any] | None = None,
    severity: str = "warning",
) -> bool:
    webhook_url = os.getenv("SLACK_WEBHOOK_URL", "").strip()
    if not webhook_url:
        print("[alert] SLACK_WEBHOOK_URL not configured; skipping Slack notification")
        return False

    metadata = metadata or {}
    details_lines = [f"- {k}: {v}" for k, v in metadata.items()]
    details_text = "\n".join(details_lines) if details_lines else "n/a"

    payload = {
        "text": f"[{severity.upper()}] {title} - {message}",
        "blocks": [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"[{severity.upper()}] {title}",
                },
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": message},
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*Details:*\n{details_text}"},
            },
        ],
    }

    try:
        response = requests.post(webhook_url, json=payload, timeout=10)
        response.raise_for_status()
        print("[alert] Slack notification sent")
        return True
    except Exception as exc:
        print(f"[alert] failed to send Slack notification: {exc}")
        return False


def _send_system_alert(title: str, message: str, metadata: Dict[str, Any]) -> None:
    _send_slack_alert(
        title=title,
        message=message,
        metadata=metadata,
        severity="error",
    )


def run_drift_check(window_size: int) -> bool:
    print(f"[drift] analyzing last {window_size} predictions from /app/logs/predictions.jsonl")
    detector = DriftDetector(enable_alerts=False)
    analysis = detector.analyze_recent_predictions(window_size=window_size)

    if "error" in analysis:
        print(f"[drift] skipped: {analysis['error']}")
        return False

    data_drift = analysis.get("data_drift", {}).get("drift_detected", False)
    pred_drift = analysis.get("prediction_drift", {}).get("drift_detected", False)

    print(f"[drift] data_drift={data_drift} prediction_drift={pred_drift}")
    has_drift = bool(data_drift or pred_drift)
    if has_drift:
        _send_slack_alert(
            title="Drift detected",
            message="Data drift and/or prediction drift was detected by scheduled monitoring.",
            metadata={
                "window_size": analysis.get("window_size"),
                "data_drift": data_drift,
                "prediction_drift": pred_drift,
                "drift_ratio": analysis.get("data_drift", {}).get("drift_ratio"),
                "class_1_ratio": analysis.get("summary", {}).get("class_1_ratio"),
            },
            severity="warning",
        )
    return has_drift


def run_performance_check(api_url: str, class1_min: float, class1_max: float) -> bool:
    metrics_url = f"{api_url.rstrip('/')}/metrics"
    print(f"[perf] fetching metrics from {metrics_url}")

    try:
        response = requests.get(metrics_url, timeout=10)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        print(f"[perf] failed to fetch metrics: {exc}")
        _send_system_alert(
            title="Monitoring job failed to fetch /metrics",
            message="Could not read API metrics endpoint during scheduled monitoring run.",
            metadata={"metrics_url": metrics_url, "error": str(exc)},
        )
        return True

    total_predictions = int(payload.get("total_predictions", 0) or 0)
    predictions_by_class = payload.get("predictions_by_class", {}) or {}
    class_1_count = int(predictions_by_class.get("1", predictions_by_class.get(1, 0)) or 0)

    if total_predictions <= 0:
        print("[perf] no predictions yet; skipping distribution check")
        return False

    class_1_ratio = class_1_count / total_predictions
    print(
        f"[perf] total_predictions={total_predictions} class_1_count={class_1_count} "
        f"class_1_ratio={class_1_ratio:.4f}"
    )

    imbalanced = class_1_ratio < class1_min or class_1_ratio > class1_max
    if not imbalanced:
        print("[perf] prediction distribution within configured bounds")
        return False

    _send_slack_alert(
        title="Prediction distribution out of bounds",
        message=(
            "Class 1 ratio is outside expected range. "
            f"Current={class_1_ratio:.2%}, expected between {class1_min:.2%} and {class1_max:.2%}."
        ),
        metadata={
            "total_predictions": total_predictions,
            "class_1_count": class_1_count,
            "class_1_ratio": round(class_1_ratio, 4),
            "class_1_min": class1_min,
            "class_1_max": class1_max,
        },
        severity="warning",
    )
    return True


def main() -> int:
    window_size = _env_int("MONITORING_WINDOW_SIZE", 100)
    api_url = os.getenv("MONITORING_API_URL", "http://passos-magicos-api:8000").strip()
    class1_min = _env_float("MONITORING_CLASS1_MIN", 0.10)
    class1_max = _env_float("MONITORING_CLASS1_MAX", 0.90)
    fail_on_alert = _env_bool("MONITORING_FAIL_ON_ALERT", False)

    print("=== PASSOS MAGICOS MONITORING JOB ===")
    print(f"window_size={window_size}")
    print(f"api_url={api_url}")
    print(f"class1_range=[{class1_min}, {class1_max}]")
    print(f"fail_on_alert={fail_on_alert}")

    has_alert = False

    try:
        has_alert = run_drift_check(window_size=window_size) or has_alert
    except Exception as exc:
        print(f"[drift] unexpected error: {exc}")
        _send_system_alert(
            title="Monitoring drift check failed",
            message="Unexpected error while running drift check.",
            metadata={"error": str(exc)},
        )
        has_alert = True

    try:
        has_alert = run_performance_check(
            api_url=api_url,
            class1_min=class1_min,
            class1_max=class1_max,
        ) or has_alert
    except Exception as exc:
        print(f"[perf] unexpected error: {exc}")
        _send_system_alert(
            title="Monitoring performance check failed",
            message="Unexpected error while running performance check.",
            metadata={"error": str(exc)},
        )
        has_alert = True

    if has_alert and fail_on_alert:
        print("monitoring completed with alerts; failing job because MONITORING_FAIL_ON_ALERT=true")
        return 1

    print("monitoring completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
