#!/usr/bin/env python
"""
Script de demonstração da integração Feature Store + API

Este script demonstra o fluxo completo:
1. Treinar modelo (que popula o Feature Store)
2. Testar endpoints da API usando Feature Store
"""
import sys
from pathlib import Path

# Adicionar o diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
import json
from time import sleep


def wait_for_api(base_url: str, max_retries: int = 10) -> bool:
    """Aguarda a API estar disponível."""
    for i in range(max_retries):
        try:
            response = requests.get(f"{base_url}/health")
            if response.status_code == 200:
                return True
        except requests.ConnectionError:
            pass
        print(f"   Aguardando API... ({i+1}/{max_retries})")
        sleep(1)
    return False


def demo_feature_store_endpoints(base_url: str):
    """Demonstra os endpoints do Feature Store."""
    
    print("\n" + "=" * 60)
    print("TESTANDO ENDPOINTS DO FEATURE STORE")
    print("=" * 60)
    
    # 1. Status do Feature Store
    print("\n[1/4] GET /features/status")
    response = requests.get(f"{base_url}/features/status")
    if response.status_code == 200:
        status = response.json()
        print(f"   ✓ Features registradas: {status['registry']['features']}")
        print(f"   ✓ Grupos: {status['registry']['groups']}")
        print(f"   ✓ Datasets offline: {status['offline_store']['datasets']}")
        print(f"   ✓ Tabelas online: {status['online_store']['tables']}")
    else:
        print(f"   ✗ Erro: {response.status_code} - {response.text}")
    
    # 2. Registry de Features
    print("\n[2/4] GET /features/registry")
    response = requests.get(f"{base_url}/features/registry")
    if response.status_code == 200:
        registry = response.json()
        print(f"   ✓ Total de features: {registry['count']}")
        for feat in registry['features'][:3]:
            print(f"      - {feat['name']}: {feat['description']}")
        if registry['count'] > 3:
            print(f"      ... e mais {registry['count'] - 3} features")
    else:
        print(f"   ✗ Erro: {response.status_code} - {response.text}")
    
    # 3. Grupos de Features
    print("\n[3/4] GET /features/groups")
    response = requests.get(f"{base_url}/features/groups")
    if response.status_code == 200:
        groups = response.json()
        print(f"   ✓ Total de grupos: {groups['count']}")
        for group in groups['groups']:
            print(f"      - {group['name']}: {group['description']}")
            print(f"        Features: {', '.join(group['features'][:3])}...")
    else:
        print(f"   ✗ Erro: {response.status_code} - {response.text}")
    
    # 4. Predição por aluno_id
    print("\n[4/4] GET /predict/aluno/{aluno_id}")
    response = requests.get(f"{base_url}/predict/aluno/1")
    if response.status_code == 200:
        prediction = response.json()
        print(f"   ✓ Predição para aluno 1:")
        print(f"      - Label: {prediction['label']}")
        print(f"      - Confiança: {prediction['confidence']:.2%}")
        print(f"      - Probabilidade Ponto de Virada: {prediction['probability_turning_point']:.2%}")
    elif response.status_code == 404:
        print("   ⚠ Aluno 1 não encontrado no Feature Store")
        print("   (Execute o script de treinamento para popular o Feature Store)")
    else:
        print(f"   ✗ Erro: {response.status_code} - {response.text}")


def demo_batch_prediction(base_url: str):
    """Demonstra predição em lote via Feature Store."""
    
    print("\n" + "=" * 60)
    print("TESTANDO PREDIÇÃO EM LOTE")
    print("=" * 60)
    
    # Predição em lote
    print("\n[1/1] GET /predict/alunos?aluno_ids=1,2,3,4,5")
    response = requests.get(f"{base_url}/predict/alunos?aluno_ids=1,2,3,4,5")
    if response.status_code == 200:
        batch = response.json()
        print(f"   ✓ Predições realizadas: {batch['count']}")
        print(f"   ✓ Fonte: {batch['source']}")
        for pred in batch['predictions'][:3]:
            print(f"      - Aluno {pred.get('aluno_id', '?')}: {pred['label']} ({pred['confidence']:.2%})")
        if batch['count'] > 3:
            print(f"      ... e mais {batch['count'] - 3} predições")
    elif response.status_code == 404:
        print("   ⚠ Nenhum aluno encontrado no Feature Store")
        print("   (Execute o script de treinamento para popular o Feature Store)")
    else:
        print(f"   ✗ Erro: {response.status_code} - {response.text}")


def demo_comparison(base_url: str):
    """Compara endpoint tradicional vs Feature Store."""
    
    print("\n" + "=" * 60)
    print("COMPARAÇÃO: ENDPOINT TRADICIONAL vs FEATURE STORE")
    print("=" * 60)
    
    # Endpoint tradicional - precisa enviar todas as features
    print("\n[Tradicional] POST /predict")
    print("   → Precisa enviar TODAS as features na requisição:")
    print("   → inde, ipv, ipp, ida, ieg, iaa, ips, ian, ipd, iap, NOTA_MAT, FASE, ANOS_PM, SITUACAO_2025")
    
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
        print(f"   ✓ Predição realizada")
    else:
        print(f"   ✗ Erro: {response.status_code}")
    
    # Endpoint Feature Store - só precisa do ID
    print("\n[Feature Store] GET /predict/aluno/1")
    print("   → Só precisa enviar o ID do aluno!")
    print("   → Features são buscadas automaticamente do Feature Store")
    
    response = requests.get(f"{base_url}/predict/aluno/1")
    if response.status_code == 200:
        print(f"   ✓ Predição realizada")
    elif response.status_code == 404:
        print("   ⚠ Aluno não encontrado (Feature Store vazio)")
    
    print("\n" + "=" * 60)
    print("VANTAGENS DO FEATURE STORE:")
    print("=" * 60)
    print("""
    1. CONSISTÊNCIA: Mesmas features no treino e inferência
    2. SIMPLICIDADE: Cliente só envia o ID
    3. VELOCIDADE: Features pré-computadas e cacheadas
    4. GOVERNANÇA: Versionamento e metadados centralizados
    5. REUSO: Features compartilhadas entre modelos
    """)


def main():
    base_url = "http://localhost:8000"
    
    print("=" * 60)
    print("DEMONSTRAÇÃO: INTEGRAÇÃO FEATURE STORE + API")
    print("=" * 60)
    
    print("\nVerificando se a API está disponível...")
    
    if not wait_for_api(base_url):
        print("\n⚠ API não está disponível!")
        print("Para iniciar a API, execute:")
        print("   uvicorn api.main:app --reload")
        print("\nOu via Docker:")
        print("   docker-compose up")
        return
    
    print("✓ API disponível!")
    
    # Executar demonstrações
    demo_feature_store_endpoints(base_url)
    demo_batch_prediction(base_url)
    demo_comparison(base_url)
    
    print("\n" + "=" * 60)
    print("DEMONSTRAÇÃO CONCLUÍDA!")
    print("=" * 60)


if __name__ == "__main__":
    main()
