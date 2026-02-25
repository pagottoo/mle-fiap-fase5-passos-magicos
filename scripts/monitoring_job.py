"""
In-cluster monitoring job for drift and performance checks.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import requests
from structlog.contextvars import bind_contextvars, clear_contextvars

# Add project root to import path for "src" imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.monitoring import DriftDetector
from src.monitoring.job_metrics import JobMetricsPusher
from src.monitoring.logger import get_logger, setup_logging

setup_logging()
logger = get_logger(component="monitoring_job")


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


def _to_builtin_scalar(value: Any) -> Any:
    """Normalize numpy/object scalars into JSON-serializable values."""
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def _normalize_metadata(metadata: Dict[str, Any] | None) -> Dict[str, Any]:
    return {k: _to_builtin_scalar(v) for k, v in (metadata or {}).items()}


def _send_slack_alert(
    title: str,
    message: str,
    metadata: Dict[str, Any] | None = None,
    severity: str = "warning",
) -> bool:
    webhook_url = os.getenv("SLACK_WEBHOOK_URL", "").strip()
    if not webhook_url:
        logger.warning(
            "monitoring_slack_not_configured",
            title=title,
            severity=severity,
        )
        return False

    safe_metadata = _normalize_metadata(metadata)
    details_lines = [f"- {k}: {v}" for k, v in safe_metadata.items()]
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
        logger.info(
            "monitoring_slack_notification_sent",
            title=title,
            severity=severity,
            metadata=safe_metadata,
        )
        return True
    except Exception as exc:
        logger.error(
            "monitoring_slack_notification_failed",
            title=title,
            severity=severity,
            error=str(exc),
        )
        return False


def _send_system_alert(title: str, message: str, metadata: Dict[str, Any]) -> None:
    _send_slack_alert(
        title=title,
        message=message,
        metadata=metadata,
        severity="error",
    )


def run_drift_check(window_size: int) -> bool:
    logger.info(
        "monitoring_drift_check_started",
        window_size=window_size,
        predictions_log="/app/logs/predictions.jsonl",
    )

    detector = DriftDetector(enable_alerts=False)
    analysis = detector.analyze_recent_predictions(window_size=window_size)

    if "error" in analysis:
        logger.info(
            "monitoring_drift_check_skipped",
            reason=analysis["error"],
            window_size=window_size,
        )
        return False

    data_drift = bool(analysis.get("data_drift", {}).get("drift_detected", False))
    pred_drift = bool(analysis.get("prediction_drift", {}).get("drift_detected", False))

    logger.info(
        "monitoring_drift_check_result",
        data_drift=data_drift,
        prediction_drift=pred_drift,
        window_size=analysis.get("window_size", window_size),
    )

    has_drift = data_drift or pred_drift
    if has_drift:
        drift_ratio = _to_builtin_scalar(analysis.get("data_drift", {}).get("drift_ratio"))
        class_1_ratio = _to_builtin_scalar(analysis.get("summary", {}).get("class_1_ratio"))

        _send_slack_alert(
            title="Drift detected",
            message="Data drift and/or prediction drift was detected by scheduled monitoring.",
            metadata={
                "window_size": analysis.get("window_size"),
                "data_drift": data_drift,
                "prediction_drift": pred_drift,
                "drift_ratio": drift_ratio,
                "class_1_ratio": class_1_ratio,
            },
            severity="warning",
        )
    return has_drift


def run_performance_check(api_url: str, class1_min: float, class1_max: float) -> bool:
    metrics_url = f"{api_url.rstrip('/')}/metrics"
    logger.info(
        "monitoring_performance_check_started",
        metrics_url=metrics_url,
        class1_min=class1_min,
        class1_max=class1_max,
    )

    try:
        response = requests.get(metrics_url, timeout=10)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        logger.error(
            "monitoring_performance_check_fetch_failed",
            metrics_url=metrics_url,
            error=str(exc),
        )
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
        logger.info(
            "monitoring_performance_check_skipped",
            reason="no_predictions",
            metrics_url=metrics_url,
        )
        return False

    class_1_ratio = class_1_count / total_predictions
    logger.info(
        "monitoring_prediction_distribution",
        total_predictions=total_predictions,
        class_1_count=class_1_count,
        class_1_ratio=round(class_1_ratio, 6),
        class1_min=class1_min,
        class1_max=class1_max,
    )

    imbalanced = class_1_ratio < class1_min or class_1_ratio > class1_max
    if not imbalanced:
        logger.info("monitoring_performance_within_bounds")
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
            "class_1_ratio": round(class_1_ratio, 6),
            "class_1_min": class1_min,
            "class_1_max": class1_max,
        },
        severity="warning",
    )
    return True


def main() -> int:
    started_at = time.time()
    metrics_pusher = JobMetricsPusher(component="monitoring")
    exit_code = 0
    bind_contextvars(run_id=f"monitoring-{datetime.now().strftime('%Y%m%d-%H%M%S')}")

    window_size = _env_int("MONITORING_WINDOW_SIZE", 100)
    api_url = os.getenv("MONITORING_API_URL", "http://passos-magicos-api:8000").strip()
    class1_min = _env_float("MONITORING_CLASS1_MIN", 0.10)
    class1_max = _env_float("MONITORING_CLASS1_MAX", 0.90)
    fail_on_alert = _env_bool("MONITORING_FAIL_ON_ALERT", False)

    logger.info(
        "monitoring_job_started",
        window_size=window_size,
        api_url=api_url,
        class1_min=class1_min,
        class1_max=class1_max,
        fail_on_alert=fail_on_alert,
    )

    has_alert = False
    has_drift_alert = False
    has_distribution_alert = False

    try:
        has_drift_alert = run_drift_check(window_size=window_size)
        has_alert = has_drift_alert or has_alert
    except Exception as exc:
        logger.error(
            "monitoring_drift_check_unexpected_error",
            error=str(exc),
            exc_info=True,
        )
        _send_system_alert(
            title="Monitoring drift check failed",
            message="Unexpected error while running drift check.",
            metadata={"error": str(exc)},
        )
        has_alert = True

    try:
        has_distribution_alert = run_performance_check(
            api_url=api_url,
            class1_min=class1_min,
            class1_max=class1_max,
        )
        has_alert = has_distribution_alert or has_alert
    except Exception as exc:
        logger.error(
            "monitoring_performance_check_unexpected_error",
            error=str(exc),
            exc_info=True,
        )
        _send_system_alert(
            title="Monitoring performance check failed",
            message="Unexpected error while running performance check.",
            metadata={"error": str(exc)},
        )
        has_alert = True

    if has_alert and fail_on_alert:
        exit_code = 1
        logger.warning(
            "monitoring_job_completed_with_alerts_and_failed",
            fail_on_alert=fail_on_alert,
        )
    else:
        exit_code = 0
        logger.info(
            "monitoring_job_completed",
            has_alert=has_alert,
            fail_on_alert=fail_on_alert,
        )

    duration = time.time() - started_at
    metrics_pusher.push_run_metrics(
        success=exit_code == 0,
        duration_seconds=duration,
        exit_code=exit_code,
        extra_metrics={
            "monitoring_last_alert_detected": 1.0 if has_alert else 0.0,
            "monitoring_last_drift_alert_detected": 1.0 if has_drift_alert else 0.0,
            "monitoring_last_distribution_alert_detected": 1.0 if has_distribution_alert else 0.0,
            "monitoring_last_fail_on_alert": 1.0 if fail_on_alert else 0.0,
        },
    )

    logger.info(
        "monitoring_job_final_status",
        exit_code=exit_code,
        duration_seconds=round(duration, 4),
        has_alert=has_alert,
        has_drift_alert=has_drift_alert,
        has_distribution_alert=has_distribution_alert,
    )
    clear_contextvars()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
