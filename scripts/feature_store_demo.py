#!/usr/bin/env python
"""
Feature Store demo script

This script demonstrates how to use the Feature Store to:
1. Register feature definitions
2. Ingest training data (offline store)
3. Materialize features for serving (online store)
4. Retrieve features for inference
"""
import sys
from pathlib import Path

# Add project root to path.
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import structlog
from src.feature_store import FeatureStore
from src.data import DataPreprocessor, FeatureEngineer
from src.config import DATA_DIR
from src.monitoring.logger import setup_logging

setup_logging()
logger = structlog.get_logger().bind(service="passos-magicos", component="feature_store_demo")


def _console(message: object, level: str = "info", **kwargs) -> None:
    log_fn = getattr(logger, level, logger.info)
    log_fn("demo_output", message=str(message), **kwargs)


def main():
    _console("=" * 60)
    _console("FEATURE STORE DEMO")
    _console("=" * 60)
    
    # 1. Initialize Feature Store.
    _console("\n[1/6] Initializing Feature Store...")
    fs = FeatureStore()
    
    status = fs.get_status()
    _console(f"   * Registered features: {status['registry']['features']}")
    _console(f"   * Registered groups: {status['registry']['groups']}")
    
    # 2. List available features.
    _console("\n[2/6] Registered features:")
    for feat_name in fs.list_features()[:5]:
        feat = fs.get_feature_definition(feat_name)
        _console(f"   - {feat.name}: {feat.description} (source: {feat.source})")
    _console(f"   ... and {len(fs.list_features()) - 5} more features")
    
    # 3. Load and prepare data.
    _console("\n[3/6] Loading and preparing data...")
    data_path = DATA_DIR / "Bases antigas" / "PEDE_PASSOS_DATASET_FIAP.csv"
    
    if not data_path.exists():
        _console(f"   ⚠ File not found: {data_path}")
        _console("   Creating sample data...")
        
        # Create sample data.
        df = pd.DataFrame({
            "aluno_id": range(1, 101),
            "INDE": [round(5 + 4 * i/100, 2) for i in range(100)],
            "IAA": [round(5 + 4 * i/100, 2) for i in range(100)],
            "IEG": [round(5 + 4 * i/100, 2) for i in range(100)],
            "IPS": [round(5 + 4 * i/100, 2) for i in range(100)],
            "IDA": [round(5 + 4 * i/100, 2) for i in range(100)],
            "IPP": [round(5 + 4 * i/100, 2) for i in range(100)],
            "IPV": [round(5 + 4 * i/100, 2) for i in range(100)],
            "IAN": [round(5 + 4 * i/100, 2) for i in range(100)],
            "IDADE_ALUNO": [10 + i % 8 for i in range(100)],
            "ANOS_PM": [1 + i % 5 for i in range(100)],
            "INSTITUICAO_ENSINO_ALUNO": ["Pública" if i % 3 == 0 else "Privada" for i in range(100)],
            "FASE": [str(i % 4 + 1) for i in range(100)],
            "PEDRA": [["Quartzo", "Ametista", "Ágata", "Topázio"][i % 4] for i in range(100)],
            "BOLSISTA": ["Sim" if i % 2 == 0 else "Não" for i in range(100)],
            "PONTO_VIRADA": ["Sim" if i > 50 else "Não" for i in range(100)]
        })
    else:
        preprocessor = DataPreprocessor()
        df = preprocessor.prepare_dataset(data_path, year="2022")
        
        # Add ID.
        df["aluno_id"] = range(1, len(df) + 1)
        
        # Create target.
        feature_engineer = FeatureEngineer()
        df = feature_engineer.create_target_variable(df)
    
    _console(f"   * Data loaded: {len(df)} rows")
    
    # 4. Ingest to Offline Store.
    _console("\n[4/6] Ingesting data into Offline Store...")
    path = fs.ingest_training_data(df, "passos_magicos_2022", "aluno_id")
    _console(f"   * Data saved to: {path}")
    
    # Show stats.
    stats = fs.offline_store.compute_statistics("passos_magicos_2022")
    _console(f"   * Rows: {stats['num_rows']}")
    _console(f"   * Columns: {stats['num_columns']}")
    
    # 5. Materialize into Online Store.
    _console("\n[5/6] Materializing features into Online Store...")
    count = fs.materialize_for_serving(df, "alunos_features", "aluno_id")
    _console(f"   * {count} rows materialized")
    
    # 6. Demonstrate serving.
    _console("\n[6/6] Demonstrating feature serving...")
    
    # Fetch features for selected students.
    entity_ids = [1, 5, 10]
    features_df = fs.get_serving_features("alunos_features", entity_ids)
    _console(f"   * Features fetched for {len(entity_ids)} students:")
    _console(features_df.head())
    
    # Fetch feature vector for one student.
    _console("\n   Feature vector for student 1:")
    vector = fs.get_feature_vector("alunos_features", 1)
    for key, value in list(vector.items())[:5]:
        _console(f"   - {key}: {value}")
    
    # Final status.
    _console("\n" + "=" * 60)
    _console("FEATURE STORE STATUS")
    _console("=" * 60)
    final_status = fs.get_status()
    _console(f"Registered features: {final_status['registry']['features']}")
    _console(f"Registered groups: {final_status['registry']['groups']}")
    _console(f"Datasets offline: {final_status['offline_store']['datasets']}")
    _console(f"Online tables: {final_status['online_store']['tables']}")
    
    _console("\n Demo completed successfully!")
    _console("\nNext steps:")
    _console("- Use fs.get_training_data() to retrieve training data")
    _console("- Use fs.get_serving_features() for real-time inference")
    _console("- Use fs.sync_offline_to_online() to synchronize stores")


if __name__ == "__main__":
    main()
