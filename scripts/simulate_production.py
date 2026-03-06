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

def run_simulation(n_normal=50, n_drift=20):
    print(f"🚀 Iniciando simulação em {API_URL}...")
    prediction_ids = []

    # 1. Gerar tráfego normal
    print(f"📦 Enviando {n_normal} predições normais...")
    for _ in range(n_normal):
        try:
            payload = get_student_payload(drift=False)
            # Mix /predict and /predict/explain
            endpoint = "/predict/explain" if random.random() > 0.7 else "/predict"
            resp = requests.post(f"{API_URL}{endpoint}", json=payload, timeout=5)
            if resp.status_code == 200:
                prediction_ids.append(resp.json()["prediction_id"])
            time.sleep(0.1)
        except Exception as e:
            print(f"Erro: {e}")

    # 2. Gerar Drift (Anomalias)
    print(f"⚠️ Gerando {n_drift} predições com Drift (notas baixas)...")
    for _ in range(n_drift):
        try:
            payload = get_student_payload(drift=True)
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
            if resp.status_code == 200:
                prediction_ids.append(resp.json()["prediction_id"])
            time.sleep(0.1)
        except Exception as e:
            print(f"Erro: {e}")

    # 3. Enviar Feedbacks (Ground Truth)
    print(f"✅ Enviando feedbacks para {len(prediction_ids)//2} predições...")
    sampled_ids = random.sample(prediction_ids, k=len(prediction_ids)//2)
    for pid in sampled_ids:
        try:
            # Simulate real outcome (80% chance of model being right)
            # Since we don't store the prediction here, we just send a random outcome
            # the API will calculate correctness based on the logs
            actual = random.choice([0, 1])
            payload = {
                "prediction_id": pid,
                "actual_outcome": actual,
                "comment": "Simulação de campo"
            }
            requests.post(f"{API_URL}/predict/feedback", json=payload, timeout=5)
            time.sleep(0.05)
        except Exception as e:
            print(f"Erro no feedback: {e}")

    print("\n✨ Simulação concluída!")
    print(f"Total de predições: {n_normal + n_drift}")
    print(f"Total de feedbacks: {len(sampled_ids)}")
    print("Verifique o Grafana e o Dashboard Streamlit em alguns instantes.")

if __name__ == "__main__":
    run_simulation()
