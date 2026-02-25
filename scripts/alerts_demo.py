#!/usr/bin/env python3
"""
Alerting System Demo

This script demonstrates how to:
1. Configure AlertManager
2. Send alerts through different channels
3. Integrate alerts with drift detection
4. Inspect alert history

Execute: python scripts/alerts_demo.py
"""
import sys
import os

# Add project root to path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime
import structlog

from src.monitoring.logger import setup_logging

setup_logging()
logger = structlog.get_logger().bind(service="passos-magicos", component="alerts_demo")


def _console(message: object, level: str = "info", **kwargs) -> None:
    log_fn = getattr(logger, level, logger.info)
    log_fn("demo_output", message=str(message), **kwargs)


def demo_basic_alerts():
    """Demonstrate basic alert sending."""
    _console("\n" + "="*60)
    _console("📢 BASIC ALERTS DEMO")
    _console("="*60)
    
    from src.monitoring.alerts import (
        AlertManager,
        AlertType,
        AlertSeverity,
        ConsoleChannel,
        get_alert_manager
    )
    
    # Create AlertManager with a console channel.
    manager = AlertManager()
    manager.add_channel(ConsoleChannel())
    
    _console("\n Sending INFO alert...")
    manager.send_alert(
        alert_type=AlertType.CUSTOM,
        severity=AlertSeverity.INFO,
        title="System Started",
        message="The alerting system was initialized successfully.",
        metadata={"version": "1.0.0", "environment": "demo"}
    )
    
    _console("\n Sending WARNING alert...")
    manager.send_alert(
        alert_type=AlertType.DATA_DRIFT,
        severity=AlertSeverity.WARNING,
        title="Mild Data Drift Detected",
        message="A small change was detected in feature distributions.",
        metadata={
            "affected_features": ["INDE", "IAA"],
            "drift_score": 0.15
        }
    )
    
    _console("\n Sending ERROR alert...")
    manager.send_alert(
        alert_type=AlertType.MODEL_PERFORMANCE,
        severity=AlertSeverity.ERROR,
        title="Model Performance Degradation",
        message="Model accuracy dropped below the acceptable threshold.",
        metadata={
            "current_accuracy": 0.72,
            "expected_accuracy": 0.85,
            "drop": "15%"
        }
    )
    
    _console("\n Sending CRITICAL alert...")
    manager.send_alert(
        alert_type=AlertType.API_ERROR,
        severity=AlertSeverity.CRITICAL,
        title="API Unavailable",
        message="The production API is not responding!",
        metadata={
            "endpoint": "/predict",
            "status_code": 503,
            "attempts": 5
        }
    )
    
    # Show alert history.
    _console("\n Alert History:")
    _console("-" * 40)
    for alert in manager.get_history():
        _console(f"  [{alert['severity']}] {alert['title']}")


def demo_convenience_functions():
    """Demonstrate AlertManager convenience methods."""
    _console("\n" + "="*60)
    _console(" ALERTMANAGER CONVENIENCE METHODS")
    _console("="*60)
    
    from src.monitoring.alerts import (
        get_alert_manager,
        send_alert,
        ConsoleChannel
    )
    
    # Configure global manager.
    manager = get_alert_manager()
    manager.add_channel(ConsoleChannel())
    
    _console("\n Using manager.alert_data_drift()...")
    manager.alert_data_drift(
        feature="INDE",
        drift_score=0.35,
        threshold=0.20
    )
    
    _console("\n Using manager.alert_prediction_drift()...")
    manager.alert_prediction_drift(
        drift_detected=True,
        current_distribution={"0": 0.35, "1": 0.65}
    )
    
    _console("\n Using manager.alert_model_performance()...")
    manager.alert_model_performance(
        metric_name="F1-Score",
        current_value=0.78,
        expected_value=0.85
    )
    
    _console("\n Using standalone send_alert()...")
    from src.monitoring.alerts import AlertType, AlertSeverity
    send_alert(
        alert_type=AlertType.CUSTOM,
        severity=AlertSeverity.INFO,
        title="New Model Registered",
        message="Model v2.0.0 was registered in MLflow",
        metadata={"model_version": "2.0.0", "stage": "Staging"}
    )


def demo_drift_integration():
    """Demonstrate integration with DriftDetector."""
    _console("\n" + "="*60)
    _console(" DRIFT DETECTOR INTEGRATION")
    _console("="*60)
    
    from src.monitoring.alerts import ConsoleChannel, get_alert_manager
    from src.monitoring.drift import DriftDetector
    
    # Configure alerts.
    manager = get_alert_manager()
    manager.add_channel(ConsoleChannel())
    
    # Create reference data.
    np.random.seed(42)
    reference_data = pd.DataFrame({
        'INDE': np.random.normal(7.0, 1.5, 100),
        'IAA': np.random.normal(6.5, 1.2, 100),
        'IEG': np.random.normal(7.2, 1.0, 100),
        'IPS': np.random.normal(6.0, 1.5, 100)
    })
    
    # Create detector with alerts enabled.
    detector = DriftDetector(reference_data=reference_data, enable_alerts=True)
    
    _console("\n Testing with similar data (no drift)...")
    similar_data = pd.DataFrame({
        'INDE': np.random.normal(7.0, 1.5, 50),
        'IAA': np.random.normal(6.5, 1.2, 50),
        'IEG': np.random.normal(7.2, 1.0, 50),
        'IPS': np.random.normal(6.0, 1.5, 50)
    })
    result = detector.detect_data_drift(similar_data)
    _console(f"   Drift detected: {result['drift_detected']}")
    
    _console("\n Testing with different data (with drift)...")
    drifted_data = pd.DataFrame({
        'INDE': np.random.normal(9.0, 1.5, 50),  # Strongly shifted mean
        'IAA': np.random.normal(4.0, 1.2, 50),  # Strongly shifted mean
        'IEG': np.random.normal(5.0, 1.0, 50),  # Shifted mean
        'IPS': np.random.normal(8.0, 1.5, 50)   # Shifted mean
    })
    result = detector.detect_data_drift(drifted_data)
    _console(f"   Drift detected: {result['drift_detected']}")
    _console(f"   Features with drift: {result['features_with_drift']}")
    
    _console("\n Testing prediction drift...")
    # Balanced distribution.
    normal_predictions = [0, 1, 0, 1, 0, 1, 0, 0, 1, 1] * 10
    result = detector.detect_prediction_drift(normal_predictions)
    _console(f"   Prediction drift (normal): {result['drift_detected']}")
    
    # Biased distribution (many more class 1 samples).
    biased_predictions = [1, 1, 1, 1, 1, 1, 1, 1, 0, 1] * 10
    result = detector.detect_prediction_drift(biased_predictions)
    _console(f"   Prediction drift (biased): {result['drift_detected']}")


def demo_alert_status():
    """Demonstrate AlertManager status inspection."""
    _console("\n" + "="*60)
    _console(" ALERT SYSTEM STATUS")
    _console("="*60)
    
    from src.monitoring.alerts import get_alert_manager, ConsoleChannel
    import json
    
    manager = get_alert_manager()
    manager.add_channel(ConsoleChannel())
    
    status = manager.get_status()
    
    _console("\n Current status:")
    _console(json.dumps(status, indent=2, default=str))
    
    _console("\n Full history:")
    for i, alert in enumerate(manager.get_history(), 1):
        _console(f"\n  {i}. [{alert['severity']}] {alert['title']}")
        _console(f"     Type: {alert['alert_type']}")
        _console(f"     Time: {alert['timestamp']}")


def demo_slack_format():
    """Demonstrate Slack alert formatting."""
    _console("\n" + "="*60)
    _console("SLACK FORMATTING")
    _console("="*60)
    
    from src.monitoring.alerts import Alert, AlertType, AlertSeverity
    import json
    
    alert = Alert(
        alert_type=AlertType.DATA_DRIFT,
        severity=AlertSeverity.WARNING,
        title="Data Drift Detected",
        message="Significant change detected in input feature distributions.",
        source="drift_detector",
        metadata={
            "affected_features": ["INDE", "IAA", "IEG"],
            "drift_score": 0.35,
            "timestamp": datetime.now().isoformat()
        }
    )
    
    slack_blocks = alert.to_slack_blocks()
    
    _console("\n Slack Block Kit JSON:")
    _console(json.dumps(slack_blocks, indent=2))


def main():
    """Run all demo sections."""
    _console("\n" + "*"*30)
    _console("   ALERTING SYSTEM DEMO")
    _console("   Passos Mágicos MLOps")
    _console("*"*30)
    
    try:
        demo_basic_alerts()
        demo_convenience_functions()
        demo_drift_integration()
        demo_alert_status()
        demo_slack_format()
        
        _console("\n" + "="*60)
        _console(" DEMO COMPLETED SUCCESSFULLY!")
        _console("="*60)
        
    except Exception as e:
        _console(f"\n Demo error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
