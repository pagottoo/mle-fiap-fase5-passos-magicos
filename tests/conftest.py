"""
Configuração dos testes pytest
"""
import sys
import os
from pathlib import Path

# Adicionar diretório raiz ao path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

import pytest
import pandas as pd
import numpy as np


@pytest.fixture(autouse=True)
def reset_mlflow_state():
    """Garante que cada teste tenha um estado limpo do MLflow."""
    # Garantir que cada teste tenha um novo run
    try:
        import mlflow
        if mlflow.active_run():
            mlflow.end_run()
    except:
        pass
    
    yield
    
    # Cleanup após cada teste
    try:
        import mlflow
        if mlflow.active_run():
            mlflow.end_run()
    except:
        pass


@pytest.fixture
def sample_data():
    """Fixture com dados de exemplo para testes (20 registros para split adequado)."""
    np.random.seed(42)
    n = 20
    
    # Gerar índices balanceados
    indices_positivos = [2, 4, 7, 9, 12, 14, 17, 19]  # 8 positivos
    indices_negativos = [i for i in range(n) if i not in indices_positivos]  # 12 negativos
    
    return pd.DataFrame({
        "NOME": [f"ALUNO-{i+1}" for i in range(n)],
        "INDE": np.random.uniform(4.0, 9.0, n).round(1),
        "IAA": np.random.uniform(4.0, 9.0, n).round(1),
        "IEG": np.random.uniform(4.0, 9.0, n).round(1),
        "IPS": np.random.uniform(4.0, 9.0, n).round(1),
        "IDA": np.random.uniform(4.0, 9.0, n).round(1),
        "IPP": np.random.uniform(4.0, 9.0, n).round(1),
        "IPV": np.random.uniform(4.0, 9.0, n).round(1),
        "IAN": np.random.uniform(2.0, 7.0, n).round(1),
        "IDADE_ALUNO": np.random.randint(10, 15, n),
        "ANOS_PM": np.random.randint(1, 4, n),
        "INSTITUICAO_ENSINO_ALUNO": np.random.choice(["Escola Pública", "Rede Decisão"], n),
        "FASE": np.random.choice(["1", "2", "3", "4"], n),
        "PEDRA": np.random.choice(["Quartzo", "Ametista", "Topázio", "Ágata"], n),
        "PONTO_VIRADA": ["Sim" if i in indices_positivos else "Não" for i in range(n)],
    })


@pytest.fixture
def sample_input():
    """Fixture com dados de entrada para predição - campos obrigatórios."""
    return {
        "INDE": 7.5,
        "IAA": 8.0,
        "IEG": 7.0,
        "IPS": 6.5,
        "IDA": 7.2,
        "IPP": 6.8,
        "IPV": 7.5,
        "IAN": 5.0,
        "FASE": "Fase 7",
        "PEDRA": "Ametista",
        "BOLSISTA": "Não",
        # Campos opcionais
        "IDADE_ALUNO": 12,
        "ANOS_PM": 2,
        "INSTITUICAO_ENSINO_ALUNO": "Escola Pública",
        "PONTO_VIRADA": "Não"
    }


@pytest.fixture
def sample_csv(tmp_path, sample_data):
    """Fixture que cria um arquivo CSV temporário."""
    csv_path = tmp_path / "test_data.csv"
    sample_data.to_csv(csv_path, sep=";", index=False)
    return csv_path
