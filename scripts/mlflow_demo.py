#!/usr/bin/env python
"""
MLflow demo script - Model Registry and Experiment Tracking

This script demonstrates a complete MLOps workflow with MLflow:
1. Experiment tracking
2. Model registration in Model Registry
3. Model promotion to production
4. Model loading for inference
"""
import sys
from pathlib import Path

# Add project root to path.
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime
import structlog

from src.data import DataPreprocessor, FeatureEngineer
from src.models.trainer import ModelTrainer
from src.mlflow_tracking import ExperimentTracker, ModelRegistry
from src.config import DATA_DIR
from src.monitoring.logger import setup_logging

setup_logging()
logger = structlog.get_logger().bind(service="passos-magicos", component="mlflow_demo")


def _console(message: object, level: str = "info", **kwargs) -> None:
    log_fn = getattr(logger, level, logger.info)
    log_fn("demo_output", message=str(message), **kwargs)


def main():
    _console("=" * 70)
    _console("DEMO: MLflow Model Registry and Experiment Tracking")
    _console("=" * 70)
    
    # 1. Load and prepare data.
    _console("\n[1/7] Loading and preparing data...")
    data_path = DATA_DIR / "Bases antigas" / "PEDE_PASSOS_DATASET_FIAP.csv"
    
    if not data_path.exists():
        _console(f"  File not found: {data_path}")
        _console("   Creating sample data for the demo...")
        df = create_sample_data()
    else:
        df = pd.read_csv(data_path, sep=";")
        _console(f"    Data loaded: {len(df)} rows")
    
    # Preprocessing.
    preprocessor = DataPreprocessor()
    df = preprocessor.clean_data(df)
    df = preprocessor.handle_missing_values(df)
    
    # Feature engineering.
    feature_engineer = FeatureEngineer()
    df = feature_engineer.fit_transform(df)
    X, y = feature_engineer.get_feature_matrix(df)
    
    _console(f"    Features: {X.shape[1]}, Samples: {X.shape[0]}")
    _console(f"    Target: {y.sum()} positives ({y.mean()*100:.1f}%)")
    
    # 2. Initialize trainer with MLflow enabled.
    _console("\n[2/7] Initializing ModelTrainer with MLflow...")
    trainer = ModelTrainer(
        model_type="random_forest",
        experiment_name="passos-magicos-demo",
        enable_mlflow=True
    )
    _console("    Trainer initialized with MLflow enabled")
    
    # 3. Start an MLflow run.
    _console("\n[3/7] Starting MLflow run...")
    run_name = f"demo-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    trainer.start_run(
        run_name=run_name,
        description="Demo of the end-to-end MLflow workflow"
    )
    _console(f"    Run started: {run_name}")
    _console(f"    Run ID: {trainer.run_id}")
    
    # 4. Train and evaluate model.
    _console("\n[4/7] Training and evaluating model...")
    
    # Split data.
    X_train, X_test, y_train, y_test = trainer.split_data(X, y)
    
    # Cross validation.
    cv_results = trainer.cross_validate(X_train, y_train)
    _console(f"    CV F1-Score: {cv_results['f1']['mean']:.4f} (±{cv_results['f1']['std']:.4f})")
    
    # Training.
    trainer.train(X_train, y_train)
    _console("    Model trained")
    
    # Evaluation.
    metrics = trainer.evaluate(X_test, y_test)
    _console(f"    Test F1-Score: {metrics['f1_score']:.4f}")
    _console(f"    Test ROC-AUC: {metrics['roc_auc']:.4f}")
    
    # 5. Register model in MLflow Model Registry.
    _console("\n[5/7] Registering model in MLflow Model Registry...")
    model_uri = trainer.log_model_to_mlflow(
        X_sample=X_test[:5],
        y_sample=y_test[:5],
        registered_model_name="passos-magicos-ponto-virada"
    )
    _console(f"    Model registered: {model_uri}")
    
    # End run.
    trainer.end_run()
    _console("    Run finished")
    
    # 6. Promote model to production.
    _console("\n[6/7] Promoting model to production...")
    success = trainer.promote_model_to_production(
        model_name="passos-magicos-ponto-virada"
    )
    if success:
        _console("    Model promoted to Production!")
    else:
        _console("  Failed to promote model")
    
    # 7. Check Model Registry status.
    _console("\n[7/7] Checking Model Registry status...")
    registry = ModelRegistry()
    
    status = registry.get_registry_status()
    _console(f"    Total registered models: {status['total_models']}")
    
    for model in status["models"]:
        _console(f"\n   Model: {model['name']}")
        _console(f"   Versions: {model['total_versions']}")
        for stage, info in model["latest_versions"].items():
            _console(f"      - {stage}: v{info['version']} ({info['status']})")
    
    # Demonstrate model loading.
    _console("\n" + "=" * 70)
    _console("LOADING MODEL FROM MLFLOW FOR INFERENCE")
    _console("=" * 70)
    
    try:
        from src.models.predictor import ModelPredictor
        
        predictor = ModelPredictor.from_mlflow(
            model_name="passos-magicos-ponto-virada",
            stage="Production"
        )
        
        _console("\n    Model loaded from MLflow!")
        _console(f"    Version: {predictor.get_model_info()['version']}")
        _console(f"    Source: {predictor.loaded_from}")
        
        # Build sample prediction input.
        sample = {
            "inde": 7.5, "ipv": 7.0, "ipp": 6.5, "ida": 15.0,
            "ieg": 7.2, "iaa": 7.8, "ips": 6.9, "ian": 7.1,
            "ipd": 7.3, "iap": 7.4, "NOTA_MAT": 8.0,
            "FASE": 3, "ANOS_PM": 2, "SITUACAO_2025": 1
        }
        
        # For MLflow-based prediction, input must contain processed features.
        _console("\n   Sample prediction:")
        _console(f"   Input: {sample}")
        
    except Exception as e:
        _console(f"\n  Error loading model: {e}")
        _console("   (This can happen if the MLflow server is not running)")
    
    # Summary.
    _console("\n" + "=" * 70)
    _console("RESUMO - MLflow Integration")
    _console("=" * 70)
    _console(f"""
     Experimento: passos-magicos-demo
     Run: {run_name}
     Registered model: passos-magicos-ponto-virada
     Current stage: Production
    
    NEXT STEPS:
    
    1. Iniciar MLflow UI:
       mlflow ui --port 5000
       
    2. Ou via Docker:
       docker-compose up mlflow
       
    3. Acessar: http://localhost:5000
    
    4. Explore:
       - Experiments and runs
       - Metrics and parameters
       - Model artifacts
       - Model Registry versions
    """)


def create_sample_data() -> pd.DataFrame:
    """Create sample data for demo purposes."""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        "NOME": [f"Aluno_{i}" for i in range(n_samples)],
        "INDE_2024": np.random.uniform(5, 9, n_samples),
        "IPV_2024": np.random.uniform(5, 9, n_samples),
        "IPP_2024": np.random.uniform(5, 9, n_samples),
        "IDA_2024": np.random.uniform(10, 18, n_samples),
        "IEG_2024": np.random.uniform(5, 9, n_samples),
        "IAA_2024": np.random.uniform(5, 9, n_samples),
        "IPS_2024": np.random.uniform(5, 9, n_samples),
        "IAN_2024": np.random.uniform(5, 9, n_samples),
        "IPD_2024": np.random.uniform(5, 9, n_samples),
        "IAP_2024": np.random.uniform(5, 9, n_samples),
        "NOTA_MAT_2024": np.random.uniform(5, 10, n_samples),
        "FASE_2024": np.random.choice([1, 2, 3, 4], n_samples),
        "ANOS_PM_2024": np.random.randint(1, 5, n_samples),
        "PONTO_VIRADA_2024": np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
    }
    
    return pd.DataFrame(data)


if __name__ == "__main__":
    main()
