"""
Unit tests for Pushgateway job metrics helper.
"""
from prometheus_client import generate_latest

import src.monitoring.job_metrics as job_metrics
from src.monitoring.job_metrics import JobMetricsPusher


def test_push_run_metrics_disabled_when_url_missing(monkeypatch):
    monkeypatch.delenv("PUSHGATEWAY_URL", raising=False)
    monkeypatch.delenv("PUSHGATEWAY_JOB_NAME", raising=False)

    pusher = JobMetricsPusher(component="training")
    assert pusher.enabled is False
    assert (
        pusher.push_run_metrics(
            success=True,
            duration_seconds=12.3,
            exit_code=0,
            extra_metrics={"training_last_f1_score": 0.91},
        )
        is False
    )


def test_push_run_metrics_enabled(monkeypatch):
    captured = {}

    def fake_push_to_gateway(url, job, registry, grouping_key):
        captured["url"] = url
        captured["job"] = job
        captured["grouping_key"] = grouping_key
        captured["payload"] = generate_latest(registry).decode("utf-8")

    monkeypatch.setenv("PUSHGATEWAY_URL", "http://pushgateway:9091")
    monkeypatch.setenv("PUSHGATEWAY_JOB_NAME", "passos-magicos-train")
    monkeypatch.setenv("PUSHGATEWAY_NAMESPACE", "passos-magicos")
    monkeypatch.setattr(job_metrics, "push_to_gateway", fake_push_to_gateway)

    pusher = JobMetricsPusher(component="training")
    ok = pusher.push_run_metrics(
        success=True,
        duration_seconds=17.8,
        exit_code=0,
        extra_metrics={"training_last_f1_score": 0.88},
    )

    assert ok is True
    assert captured["url"] == "http://pushgateway:9091"
    assert captured["job"] == "passos-magicos-train"
    assert captured["grouping_key"]["component"] == "training"
    assert captured["grouping_key"]["namespace"] == "passos-magicos"

    payload = captured["payload"]
    assert "passos_magicos_job_last_success 1.0" in payload
    assert "passos_magicos_job_last_exit_code 0.0" in payload
    assert "passos_magicos_training_last_f1_score 0.88" in payload
