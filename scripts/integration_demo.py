#!/usr/bin/env python
"""
Feature Store + API integration demo script

This script demonstrates the end-to-end flow:
1. Train a model (which populates the Feature Store)
2. Test API endpoints using Feature Store data
"""
import sys
from pathlib import Path

# Add project root to path.
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
import json
from time import sleep
import structlog

from src.monitoring.logger import setup_logging

setup_logging()
logger = structlog.get_logger().bind(service="passos-magicos", component="integration_demo")


def _console(message: object, level: str = "info", **kwargs) -> None:
    log_fn = getattr(logger, level, logger.info)
    log_fn("demo_output", message=str(message), **kwargs)


def wait_for_api(base_url: str, max_retries: int = 10) -> bool:
    """Wait until the API becomes available."""
    for i in range(max_retries):
        try:
            response = requests.get(f"{base_url}/health")
            if response.status_code == 200:
                return True
        except requests.ConnectionError:
            pass
        _console(f"   Waiting for API... ({i+1}/{max_retries})")
        sleep(1)
    return False


def demo_feature_store_endpoints(base_url: str):
    """Demonstrate Feature Store endpoints."""
    
    _console("\n" + "=" * 60)
    _console("TESTING FEATURE STORE ENDPOINTS")
    _console("=" * 60)
    
    # 1. Feature Store status.
    _console("\n[1/4] GET /features/status")
    response = requests.get(f"{base_url}/features/status")
    if response.status_code == 200:
        status = response.json()
        _console(f"   ✓ Registered features: {status['registry']['features']}")
        _console(f"   ✓ Groups: {status['registry']['groups']}")
        _console(f"   ✓ Offline datasets: {status['offline_store']['datasets']}")
        _console(f"   ✓ Online tables: {status['online_store']['tables']}")
    else:
        _console(f"   ✗ Error: {response.status_code} - {response.text}")
    
    # 2. Feature registry.
    _console("\n[2/4] GET /features/registry")
    response = requests.get(f"{base_url}/features/registry")
    if response.status_code == 200:
        registry = response.json()
        _console(f"   ✓ Total features: {registry['count']}")
        for feat in registry['features'][:3]:
            _console(f"      - {feat['name']}: {feat['description']}")
        if registry['count'] > 3:
            _console(f"      ... and {registry['count'] - 3} more features")
    else:
        _console(f"   ✗ Error: {response.status_code} - {response.text}")
    
    # 3. Feature groups.
    _console("\n[3/4] GET /features/groups")
    response = requests.get(f"{base_url}/features/groups")
    if response.status_code == 200:
        groups = response.json()
        _console(f"   ✓ Total groups: {groups['count']}")
        for group in groups['groups']:
            _console(f"      - {group['name']}: {group['description']}")
            _console(f"        Features: {', '.join(group['features'][:3])}...")
    else:
        _console(f"   ✗ Error: {response.status_code} - {response.text}")
    
    # 4. Prediction by aluno_id.
    _console("\n[4/4] GET /predict/aluno/{aluno_id}")
    response = requests.get(f"{base_url}/predict/aluno/1")
    if response.status_code == 200:
        prediction = response.json()
        _console(f"   ✓ Prediction for student 1:")
        _console(f"      - Label: {prediction['label']}")
        _console(f"      - Confidence: {prediction['confidence']:.2%}")
        _console(f"      - Turning-point probability: {prediction['probability_turning_point']:.2%}")
    elif response.status_code == 404:
        _console("   ⚠ Student 1 not found in Feature Store")
        _console("   (Run the training script to populate the Feature Store)")
    else:
        _console(f"   ✗ Error: {response.status_code} - {response.text}")


def demo_batch_prediction(base_url: str):
    """Demonstrate batch prediction using Feature Store."""
    
    _console("\n" + "=" * 60)
    _console("TESTING BATCH PREDICTION")
    _console("=" * 60)
    
    # Batch prediction.
    _console("\n[1/1] GET /predict/alunos?aluno_ids=1,2,3,4,5")
    response = requests.get(f"{base_url}/predict/alunos?aluno_ids=1,2,3,4,5")
    if response.status_code == 200:
        batch = response.json()
        _console(f"   ✓ Predictions returned: {batch['count']}")
        _console(f"   ✓ Source: {batch['source']}")
        for pred in batch['predictions'][:3]:
            _console(f"      - Student {pred.get('aluno_id', '?')}: {pred['label']} ({pred['confidence']:.2%})")
        if batch['count'] > 3:
            _console(f"      ... and {batch['count'] - 3} more predictions")
    elif response.status_code == 404:
        _console("   ⚠ No students found in Feature Store")
        _console("   (Run the training script to populate the Feature Store)")
    else:
        _console(f"   ✗ Error: {response.status_code} - {response.text}")


def demo_comparison(base_url: str):
    """Compare traditional endpoint vs Feature Store endpoint."""
    
    _console("\n" + "=" * 60)
    _console("COMPARISON: TRADITIONAL ENDPOINT vs FEATURE STORE")
    _console("=" * 60)
    
    # Traditional endpoint: all features must be provided.
    _console("\n[Traditional] POST /predict")
    _console("   -> You must send ALL features in the request:")
    _console("   → inde, ipv, ipp, ida, ieg, iaa, ips, ian, ipd, iap, NOTA_MAT, FASE, ANOS_PM, SITUACAO_2025")
    
    sample_data = {
        "inde": 7.5,
        "ipv": 7.0,
        "ipp": 6.5,
        "ida": 15.0,
        "ieg": 7.2,
        "iaa": 7.8,
        "ips": 6.9,
        "ian": 7.1,
        "ipd": 7.3,
        "iap": 7.4,
        "NOTA_MAT": 8.0,
        "FASE": 3,
        "ANOS_PM": 2,
        "SITUACAO_2025": 1
    }
    
    response = requests.post(f"{base_url}/predict", json=sample_data)
    if response.status_code == 200:
        _console("   ✓ Prediction successful")
    else:
        _console(f"   ✗ Error: {response.status_code}")
    
    # Feature Store endpoint: only student ID is required.
    _console("\n[Feature Store] GET /predict/aluno/1")
    _console("   -> Only the student ID is required.")
    _console("   -> Features are fetched automatically from Feature Store.")
    
    response = requests.get(f"{base_url}/predict/aluno/1")
    if response.status_code == 200:
        _console("   ✓ Prediction successful")
    elif response.status_code == 404:
        _console("   ⚠ Student not found (empty Feature Store)")
    
    _console("\n" + "=" * 60)
    _console("FEATURE STORE ADVANTAGES:")
    _console("=" * 60)
    _console("""
    1. CONSISTENCY: Same features for training and inference
    2. SIMPLICITY: Client sends only the ID
    3. SPEED: Pre-computed and cached features
    4. GOVERNANCE: Centralized versioning and metadata
    5. REUSE: Shared features across models
    """)


def main():
    base_url = "http://localhost:8000"
    
    _console("=" * 60)
    _console("DEMO: FEATURE STORE + API INTEGRATION")
    _console("=" * 60)
    
    _console("\nChecking API availability...")
    
    if not wait_for_api(base_url):
        _console("\n⚠ API is not available!")
        _console("To start the API, run:")
        _console("   uvicorn api.main:app --reload")
        _console("\nOr via Docker:")
        _console("   docker-compose up")
        return
    
    _console("✓ API is available!")
    
    # Run demos.
    demo_feature_store_endpoints(base_url)
    demo_batch_prediction(base_url)
    demo_comparison(base_url)
    
    _console("\n" + "=" * 60)
    _console("DEMO COMPLETED!")
    _console("=" * 60)


if __name__ == "__main__":
    main()
