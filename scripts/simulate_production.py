"""
Production Simulation Script for Passos Magicos MLOps.
Generates traffic, drift, and feedback to populate dashboards.
"""
import os
import requests
import random
import time
import uuid
from datetime import datetime

# URL da API: Prioridade para variável de ambiente, fallback para localhost
API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

def get_student_payload(drift=False):
    """Generate a random student payload."""
    if drift:
        # Simulate drift: students with very low grades suddenly appearing
        return {
            "INDE": random.uniform(1.0, 4.0),
            "IAA": random.uniform(1.0, 4.0),
            "IEG": random.uniform(1.0, 4.0),
            "IPS": random.uniform(1.0, 4.0),
            "IDA": random.uniform(1.0, 4.0),
            "IPP": random.uniform(1.0, 4.0),
            "IPV": random.uniform(1.0, 4.0),
            "IAN": random.uniform(1.0, 4.0),
            "FASE": str(random.randint(0, 8)),
            "PEDRA": random.choice(["Quartzo", "Ágata"]),
            "BOLSISTA": "Não",
            "IDADE_ALUNO": random.randint(15, 20)
        }
    else:
        # Normal distribution
        return {
            "INDE": random.uniform(5.0, 9.5),
            "IAA": random.uniform(6.0, 10.0),
            "IEG": random.uniform(5.0, 10.0),
            "IPS": random.uniform(5.0, 10.0),
            "IDA": random.uniform(4.0, 9.0),
            "IPP": random.uniform(5.0, 9.0),
            "IPV": random.uniform(5.0, 9.5),
            "IAN": random.uniform(5.0, 10.0),
            "FASE": str(random.randint(0, 8)),
            "PEDRA": random.choice(["Ametista", "Topázio", "Ágata"]),
            "BOLSISTA": random.choice(["Sim", "Não"]),
            "IDADE_ALUNO": random.randint(7, 18)
        }

def get_actual_outcome(payload, noise_level=0.0):
    """
    Calculate the 'true' outcome based on business rules.
    Matches the logic in FeatureEngineer.create_target_variable.
    """
    # 1. Base logic: High INDE or IPV or specific stones
    is_turning_point = 0
    if payload.get("INDE", 0) >= 7.5 or payload.get("IPV", 0) >= 7.5:
        is_turning_point = 1
    elif payload.get("PEDRA") in ["Topázio", "Ágata"]:
        is_turning_point = 1
    
    # 2. Add noise to simulate real-world errors or model limitations
    if random.random() < noise_level:
        return 1 - is_turning_point
    
    return is_turning_point

def run_simulation(n_normal=50, n_drift=20, error_mode=False):
    print(f"Iniciando simulação em {API_URL} (Modo Erro: {error_mode})...")
    prediction_data = []

    # 1. Gerar tráfego normal
    print(f"Enviando {n_normal} predições normais...")
    for _ in range(n_normal):
        try:
            payload = get_student_payload(drift=False)
            endpoint = "/predict/explain" if random.random() > 0.7 else "/predict"
            resp = requests.post(f"{API_URL}{endpoint}", json=payload, timeout=5)
            if resp.status_code == 200:
                pid = resp.json()["prediction_id"]
                # Store payload to calculate correct outcome later
                prediction_data.append({"id": pid, "payload": payload})
            time.sleep(0.1)
        except Exception as e:
            print(f"Erro: {e}")

    # 2. Gerar Drift (Anomalias)
    print(f"Gerando {n_drift} predições com Drift (notas baixas)...")
    for _ in range(n_drift):
        try:
            payload = get_student_payload(drift=True)
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
            if resp.status_code == 200:
                pid = resp.json()["prediction_id"]
                prediction_data.append({"id": pid, "payload": payload})
            time.sleep(0.1)
        except Exception as e:
            print(f"Erro: {e}")

    # 3. Enviar Feedbacks (Ground Truth)
    # Se error_mode for True, aumentamos o ruído para 50% (derrubando acurácia)
    # Se False, ruído baixo (5%) para manter acurácia alta
    noise = 0.5 if error_mode else 0.05
    
    print(f"Enviando feedbacks (Ruído: {noise*100}%)...")
    sampled_data = random.sample(prediction_data, k=len(prediction_data)//2)
    for item in sampled_data:
        try:
            actual = get_actual_outcome(item["payload"], noise_level=noise)
            payload = {
                "prediction_id": item["id"],
                "actual_outcome": actual,
                "comment": "Simulação de campo" if not error_mode else "Simulação de erro proposital"
            }
            requests.post(f"{API_URL}/predict/feedback", json=payload, timeout=5)
            time.sleep(0.05)
        except Exception as e:
            print(f"Erro no feedback: {e}")

    print("\n Simulação concluída!")
    print(f"Total de predições: {n_normal + n_drift}")
    print(f"Total de feedbacks: {len(sampled_ids)}")
    print("Verifique o Grafana e o Dashboard Streamlit em alguns instantes.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Simulate production traffic and feedback")
    parser.add_argument("--error-mode", action="store_true", help="Simulate low accuracy and drift")
    parser.add_argument("--n-normal", type=int, default=50, help="Number of normal predictions")
    parser.add_argument("--n-drift", type=int, default=20, help="Number of drift predictions")
    args = parser.parse_args()
    
    run_simulation(n_normal=args.n_normal, n_drift=args.n_drift, error_mode=args.error_mode)
