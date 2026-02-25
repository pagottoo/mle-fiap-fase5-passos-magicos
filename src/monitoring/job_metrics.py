"""
Pushgateway helper for short-lived jobs (CronJobs).
"""
from __future__ import annotations

import os
import re
import time
from typing import Dict, Optional

from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
import structlog

logger = structlog.get_logger()


_METRIC_NAME_RE = re.compile(r"[^a-zA-Z0-9_:]")


def _sanitize_metric_name(name: str) -> str:
    candidate = _METRIC_NAME_RE.sub("_", name.strip())
    if not candidate:
        candidate = "passos_magicos_job_metric"
    if not candidate[0].isalpha() and candidate[0] != "_":
        candidate = f"passos_magicos_{candidate}"
    if not candidate.startswith("passos_magicos_"):
        candidate = f"passos_magicos_{candidate}"
    return candidate


class JobMetricsPusher:
    """
    Pushes run metrics to Prometheus Pushgateway.

    Expected environment variables:
    - PUSHGATEWAY_URL: Pushgateway base URL (optional; disables push when empty)
    - PUSHGATEWAY_JOB_NAME: logical job name in Pushgateway
    - PUSHGATEWAY_NAMESPACE: optional grouping label
    """

    def __init__(
        self,
        component: str,
        pushgateway_url: Optional[str] = None,
        job_name: Optional[str] = None,
        namespace: Optional[str] = None,
    ):
        self.component = component.strip().lower()
        self.pushgateway_url = (pushgateway_url or os.getenv("PUSHGATEWAY_URL", "")).strip()
        self.job_name = (
            job_name
            or os.getenv("PUSHGATEWAY_JOB_NAME", f"passos-magicos-{self.component}")
        ).strip()
        self.namespace = (namespace or os.getenv("PUSHGATEWAY_NAMESPACE", "")).strip()

    @property
    def enabled(self) -> bool:
        return bool(self.pushgateway_url)

    def push_run_metrics(
        self,
        *,
        success: bool,
        duration_seconds: float,
        exit_code: int,
        extra_metrics: Optional[Dict[str, float]] = None,
    ) -> bool:
        """
        Push latest run status for this component.
        """
        if not self.enabled:
            return False

        registry = CollectorRegistry()
        run_ts = Gauge(
            "passos_magicos_job_last_run_timestamp_seconds",
            "Unix timestamp of the latest job run.",
            registry=registry,
        )
        run_duration = Gauge(
            "passos_magicos_job_last_duration_seconds",
            "Duration in seconds of the latest job run.",
            registry=registry,
        )
        run_success = Gauge(
            "passos_magicos_job_last_success",
            "Whether latest job run succeeded (1) or failed (0).",
            registry=registry,
        )
        run_exit_code = Gauge(
            "passos_magicos_job_last_exit_code",
            "Exit code of latest job run.",
            registry=registry,
        )

        run_ts.set(time.time())
        run_duration.set(max(duration_seconds, 0.0))
        run_success.set(1 if success else 0)
        run_exit_code.set(float(exit_code))

        for metric_name, metric_value in (extra_metrics or {}).items():
            gauge = Gauge(
                _sanitize_metric_name(metric_name),
                f"Extra job metric: {metric_name}",
                registry=registry,
            )
            gauge.set(float(metric_value))

        grouping_key = {"component": self.component}
        if self.namespace:
            grouping_key["namespace"] = self.namespace

        try:
            push_to_gateway(
                self.pushgateway_url,
                job=self.job_name,
                registry=registry,
                grouping_key=grouping_key,
            )
            logger.info(
                "pushgateway_metrics_pushed",
                component=self.component,
                job=self.job_name,
                pushgateway_url=self.pushgateway_url,
            )
            return True
        except Exception as exc:
            logger.warning(
                "pushgateway_metrics_push_failed",
                component=self.component,
                job=self.job_name,
                pushgateway_url=self.pushgateway_url,
                error=str(exc),
            )
            return False
