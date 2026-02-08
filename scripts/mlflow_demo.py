#!/usr/bin/env python
"""
Script de demonstração do MLflow - Model Registry e Experiment Tracking

Este script demonstra o fluxo completo de MLOps com MLflow:
1. Tracking de experimentos
2. Registro de modelos no Model Registry
3. Promoção de modelos para produção
4. Carregamento de modelo para inferência
"""
import sys
from pathlib import Path

# Adicionar o diretório src ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime

from src.data import DataPreprocessor, FeatureEngineer
from src.models.trainer import ModelTrainer
from src.mlflow_tracking import ExperimentTracker, ModelRegistry
from src.config import DATA_DIR


def main():
    print("=" * 70)
    print("DEMONSTRAÇÃO: MLflow Model Registry e Experiment Tracking")
    print("=" * 70)
    
    # 1. Carregar e preparar dados
    print("\n[1/7] Carregando e preparando dados...")
    data_path = DATA_DIR / "Bases antigas" / "PEDE_PASSOS_DATASET_FIAP.csv"
    
    if not data_path.exists():
        print(f"   ⚠ Arquivo não encontrado: {data_path}")
        print("   Criando dados de exemplo para demonstração...")
        df = create_sample_data()
    else:
        df = pd.read_csv(data_path, sep=";")
        print(f"   ✓ Dados carregados: {len(df)} registros")
    
    # Preprocessamento
    preprocessor = DataPreprocessor()
    df = preprocessor.clean_data(df)
    df = preprocessor.handle_missing_values(df)
    
    # Feature engineering
    feature_engineer = FeatureEngineer()
    df = feature_engineer.fit_transform(df)
    X, y = feature_engineer.get_feature_matrix(df)
    
    print(f"   ✓ Features: {X.shape[1]}, Samples: {X.shape[0]}")
    print(f"   ✓ Target: {y.sum()} positivos ({y.mean()*100:.1f}%)")
    
    # 2. Inicializar trainer com MLflow
    print("\n[2/7] Inicializando ModelTrainer com MLflow...")
    trainer = ModelTrainer(
        model_type="random_forest",
        experiment_name="passos-magicos-demo",
        enable_mlflow=True
    )
    print("   ✓ Trainer inicializado com MLflow habilitado")
    
    # 3. Iniciar uma run MLflow
    print("\n[3/7] Iniciando run MLflow...")
    run_name = f"demo-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    trainer.start_run(
        run_name=run_name,
        description="Demonstração do fluxo MLflow completo"
    )
    print(f"   ✓ Run iniciada: {run_name}")
    print(f"   ✓ Run ID: {trainer.run_id}")
    
    # 4. Treinar e avaliar modelo
    print("\n[4/7] Treinando e avaliando modelo...")
    
    # Split dos dados
    X_train, X_test, y_train, y_test = trainer.split_data(X, y)
    
    # Validação cruzada
    cv_results = trainer.cross_validate(X_train, y_train)
    print(f"   ✓ CV F1-Score: {cv_results['f1']['mean']:.4f} (±{cv_results['f1']['std']:.4f})")
    
    # Treinamento
    trainer.train(X_train, y_train)
    print("   ✓ Modelo treinado")
    
    # Avaliação
    metrics = trainer.evaluate(X_test, y_test)
    print(f"   ✓ Test F1-Score: {metrics['f1_score']:.4f}")
    print(f"   ✓ Test ROC-AUC: {metrics['roc_auc']:.4f}")
    
    # 5. Registrar modelo no MLflow
    print("\n[5/7] Registrando modelo no MLflow Model Registry...")
    model_uri = trainer.log_model_to_mlflow(
        X_sample=X_test[:5],
        y_sample=y_test[:5],
        registered_model_name="passos-magicos-ponto-virada"
    )
    print(f"   ✓ Modelo registrado: {model_uri}")
    
    # Finalizar run
    trainer.end_run()
    print("   ✓ Run finalizada")
    
    # 6. Promover modelo para produção
    print("\n[6/7] Promovendo modelo para produção...")
    success = trainer.promote_model_to_production(
        model_name="passos-magicos-ponto-virada"
    )
    if success:
        print("   ✓ Modelo promovido para Production!")
    else:
        print("   ⚠ Falha ao promover modelo")
    
    # 7. Verificar status do Model Registry
    print("\n[7/7] Verificando status do Model Registry...")
    registry = ModelRegistry()
    
    status = registry.get_registry_status()
    print(f"   ✓ Total de modelos registrados: {status['total_models']}")
    
    for model in status["models"]:
        print(f"\n   Modelo: {model['name']}")
        print(f"   Versões: {model['total_versions']}")
        for stage, info in model["latest_versions"].items():
            print(f"      - {stage}: v{info['version']} ({info['status']})")
    
    # Demonstrar carregamento do modelo
    print("\n" + "=" * 70)
    print("CARREGANDO MODELO DO MLFLOW PARA INFERÊNCIA")
    print("=" * 70)
    
    try:
        from src.models.predictor import ModelPredictor
        
        predictor = ModelPredictor.from_mlflow(
            model_name="passos-magicos-ponto-virada",
            stage="Production"
        )
        
        print(f"\n   ✓ Modelo carregado do MLflow!")
        print(f"   ✓ Versão: {predictor.get_model_info()['version']}")
        print(f"   ✓ Fonte: {predictor.loaded_from}")
        
        # Fazer uma predição de exemplo
        sample = {
            "inde": 7.5, "ipv": 7.0, "ipp": 6.5, "ida": 15.0,
            "ieg": 7.2, "iaa": 7.8, "ips": 6.9, "ian": 7.1,
            "ipd": 7.3, "iap": 7.4, "NOTA_MAT": 8.0,
            "FASE": 3, "ANOS_PM": 2, "SITUACAO_2025": 1
        }
        
        # Nota: Para predição com modelo MLflow, precisamos passar features processadas
        print("\n   Predição de exemplo:")
        print(f"   Input: {sample}")
        
    except Exception as e:
        print(f"\n   ⚠ Erro ao carregar modelo: {e}")
        print("   (Isso pode ocorrer se o MLflow server não estiver rodando)")
    
    # Sumário
    print("\n" + "=" * 70)
    print("RESUMO - MLflow Integration")
    print("=" * 70)
    print(f"""
    ✓ Experimento: passos-magicos-demo
    ✓ Run: {run_name}
    ✓ Modelo registrado: passos-magicos-ponto-virada
    ✓ Stage atual: Production
    
    PRÓXIMOS PASSOS:
    
    1. Iniciar MLflow UI:
       mlflow ui --port 5000
       
    2. Ou via Docker:
       docker-compose up mlflow
       
    3. Acessar: http://localhost:5000
    
    4. Visualizar:
       - Experimentos e runs
       - Métricas e parâmetros
       - Artefatos do modelo
       - Model Registry com versões
    """)


def create_sample_data() -> pd.DataFrame:
    """Cria dados de exemplo para demonstração."""
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
