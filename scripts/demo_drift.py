"""
Drift Detection system demo
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import structlog
from src.monitoring.drift import DriftDetector
from src.feature_store import FeatureStore
import pandas as pd
import numpy as np
from src.monitoring.logger import setup_logging

setup_logging()
logger = structlog.get_logger().bind(service="passos-magicos", component="demo_drift")


def _console(message: object, level: str = "info", **kwargs) -> None:
    log_fn = getattr(logger, level, logger.info)
    log_fn("demo_output", message=str(message), **kwargs)

def main():
    _console('=' * 60)
    _console(' DRIFT DETECTION DEMO')
    _console('=' * 60)

    # 1. Load reference data from Feature Store.
    _console('\n[1] Loading reference data from Feature Store...')
    fs = FeatureStore()
    ref_data = fs.get_training_data('passos_magicos_training')
    _console(f'   * Data loaded: {len(ref_data)} rows')

    # 2. Build detector with reference data.
    _console('\n[2] Initializing Drift Detector...')
    numeric_cols = ['INDE', 'IAA', 'IEG', 'IPS', 'IDA', 'IPP', 'IPV', 'IAN']
    detector = DriftDetector(reference_data=ref_data[numeric_cols])
    _console(f'   * Detector initialized with {len(numeric_cols)} features')
    _console('   * Method: Kolmogorov-Smirnov test (p-value threshold=0.05)')

    # 3. Simulate data without drift (same distribution).
    _console('\n[3] Testing data without drift (sample from same distribution)...')
    sample_normal = ref_data[numeric_cols].sample(50, random_state=42)
    drift_result_normal = detector.detect_data_drift(sample_normal)
    _console(f'   -> Drift detected: {drift_result_normal["drift_detected"]}')
    _console(f'   -> Features with drift: {drift_result_normal["features_with_drift"]}')

    # 4. Simulate data with drift (distribution shift).
    _console('\n[4] Testing data with drift (artificially shifted values)...')
    _console('   Simulation: INDE +3 points, IPV *1.5')
    sample_drift = sample_normal.copy()
    sample_drift['INDE'] = sample_drift['INDE'] + 3  # Add 3 points to INDE.
    sample_drift['IPV'] = sample_drift['IPV'] * 1.5   # Increase IPV by 50%.
    drift_result_drift = detector.detect_data_drift(sample_drift)
    _console(f'   -> Drift detected: {drift_result_drift["drift_detected"]}')
    _console(f'   -> Features with drift: {drift_result_drift["features_with_drift"]}')

    # 5. Per-feature statistical details.
    _console('\n[5] Statistical test details by feature:')
    _console('   ' + '-' * 50)
    _console(f'   {"Feature":<10} {"Z-Score":<12} {"Drift?"}')
    _console('   ' + '-' * 50)
    for feat, details in drift_result_drift['details'].items():
        status = '[DRIFT] DRIFT' if details.get('drift', False) else '[OK] OK'
        z_score = details.get('z_score', 0)
        _console(f'   {feat:<10} {z_score:<12.4f} {status}')
    _console('   ' + '-' * 50)

    # 6. Prediction drift.
    _console('\n[6] Testing Prediction Drift...')
    # Simulate normal predictions (balanced distribution).
    predictions_normal = [0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0]
    pred_drift_normal = detector.detect_prediction_drift(predictions_normal)
    _console(f'   Normal predictions: {predictions_normal}')
    _console(f'   -> Drift detected: {pred_drift_normal["drift_detected"]}')
    
    # Simulate predictions with drift (too many positives).
    predictions_drift = [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]
    pred_drift_result = detector.detect_prediction_drift(predictions_drift)
    _console(f'   Abnormal predictions: {predictions_drift}')
    _console(f'   -> Drift detected: {pred_drift_result["drift_detected"]}')

    # 7. Full drift report.
    _console('\n[7] Drift report:')
    report = detector.get_drift_report()
    _console(f'   * Total checks: {report["total_checks"]}')
    _console(f'   * Drifts detected: {report["drifts_detected"]}')
    _console(f'   * Last check: {report["last_check"]}')

    _console('\n' + '=' * 60)
    _console(' Drift Detection demo completed!')
    _console('=' * 60)
    
    # Conceptual summary.
    _console('\n CONCEPTS:')
    _console('''
    Data Drift: Changes in input feature distributions
    - Detected via Kolmogorov-Smirnov test
    - p-value < 0.05 indicates significant drift
    
    Prediction Drift: Changes in prediction distributions
    - Monitors class proportions
    - Raises alert when positive ratio shifts too much
    
    Production usage:
    - Monitor predictions continuously
    - Retrain model when drift persists
    - Alert the team to investigate root causes
    ''')

if __name__ == "__main__":
    main()
