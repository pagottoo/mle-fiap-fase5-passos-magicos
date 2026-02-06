#!/usr/bin/env python
"""
Script de demonstração do Feature Store

Este script demonstra como usar o Feature Store para:
1. Registrar definições de features
2. Ingerir dados de treinamento (offline store)
3. Materializar features para serving (online store)
4. Obter features para inferência
"""
import sys
from pathlib import Path

# Adicionar o diretório src ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.feature_store import FeatureStore
from src.data import DataPreprocessor, FeatureEngineer
from src.config import DATA_DIR


def main():
    print("=" * 60)
    print("DEMONSTRAÇÃO DO FEATURE STORE")
    print("=" * 60)
    
    # 1. Inicializar Feature Store
    print("\n[1/6] Inicializando Feature Store...")
    fs = FeatureStore()
    
    status = fs.get_status()
    print(f"   * Features registradas: {status['registry']['features']}")
    print(f"   * Grupos registrados: {status['registry']['groups']}")
    
    # 2. Listar features disponíveis
    print("\n[2/6] Features registradas:")
    for feat_name in fs.list_features()[:5]:
        feat = fs.get_feature_definition(feat_name)
        print(f"   - {feat.name}: {feat.description} (source: {feat.source})")
    print(f"   ... e mais {len(fs.list_features()) - 5} features")
    
    # 3. Carregar e preparar dados
    print("\n[3/6] Carregando e preparando dados...")
    data_path = DATA_DIR / "Bases antigas" / "PEDE_PASSOS_DATASET_FIAP.csv"
    
    if not data_path.exists():
        print(f"   ⚠ Arquivo não encontrado: {data_path}")
        print("   Criando dados de exemplo...")
        
        # Criar dados de exemplo
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
            "PEDRA": ["Quartzo", "Ametista", "Ágata", "Topázio"][i % 4] for i in range(100)],
            "BOLSISTA": ["Sim" if i % 2 == 0 else "Não" for i in range(100)],
            "PONTO_VIRADA": ["Sim" if i > 50 else "Não" for i in range(100)]
        })
    else:
        preprocessor = DataPreprocessor()
        df = preprocessor.prepare_dataset(data_path, year="2022")
        
        # Adicionar ID
        df["aluno_id"] = range(1, len(df) + 1)
        
        # Criar target
        feature_engineer = FeatureEngineer()
        df = feature_engineer.create_target_variable(df)
    
    print(f"   * Dados carregados: {len(df)} registros")
    
    # 4. Ingerir no Offline Store
    print("\n[4/6] Ingerindo dados no Offline Store...")
    path = fs.ingest_training_data(df, "passos_magicos_2022", "aluno_id")
    print(f"   * Dados salvos em: {path}")
    
    # Mostrar estatísticas
    stats = fs.offline_store.compute_statistics("passos_magicos_2022")
    print(f"   * Registros: {stats['num_rows']}")
    print(f"   * Colunas: {stats['num_columns']}")
    
    # 5. Materializar no Online Store
    print("\n[5/6] Materializando features no Online Store...")
    count = fs.materialize_for_serving(df, "alunos_features", "aluno_id")
    print(f"   * {count} registros materializados")
    
    # 6. Demonstrar serving
    print("\n[6/6] Demonstrando serving de features...")
    
    # Obter features para alguns alunos
    entity_ids = [1, 5, 10]
    features_df = fs.get_serving_features("alunos_features", entity_ids)
    print(f"   * Features recuperadas para {len(entity_ids)} alunos:")
    print(features_df.head())
    
    # Obter vetor de features para um aluno
    print("\n   Vetor de features para aluno 1:")
    vector = fs.get_feature_vector("alunos_features", 1)
    for key, value in list(vector.items())[:5]:
        print(f"   - {key}: {value}")
    
    # Status final
    print("\n" + "=" * 60)
    print("STATUS DO FEATURE STORE")
    print("=" * 60)
    final_status = fs.get_status()
    print(f"Features registradas: {final_status['registry']['features']}")
    print(f"Grupos registrados: {final_status['registry']['groups']}")
    print(f"Datasets offline: {final_status['offline_store']['datasets']}")
    print(f"Tabelas online: {final_status['online_store']['tables']}")
    
    print("\n Demonstração concluída com sucesso!")
    print("\nPróximos passos:")
    print("- Use fs.get_training_data() para obter dados de treinamento")
    print("- Use fs.get_serving_features() para inferência em tempo real")
    print("- Use fs.sync_offline_to_online() para sincronizar stores")


if __name__ == "__main__":
    main()
