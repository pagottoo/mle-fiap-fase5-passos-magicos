#!/usr/bin/env python3
"""
Demonstração do Sistema de Alertas

Este script demonstra como:
1. Configurar AlertManager
2. Enviar alertas via diferentes canais
3. Integrar alertas com detecção de drift
4. Consultar histórico de alertas

Execute: python scripts/alerts_demo.py
"""
import sys
import os

# Adicionar path do projeto
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime


def demo_basic_alerts():
    """Demonstra envio básico de alertas."""
    print("\n" + "="*60)
    print("📢 DEMONSTRAÇÃO DE ALERTAS BÁSICOS")
    print("="*60)
    
    from src.monitoring.alerts import (
        AlertManager,
        AlertType,
        AlertSeverity,
        ConsoleChannel,
        get_alert_manager
    )
    
    # Criar AlertManager com canal de console
    manager = AlertManager()
    manager.add_channel(ConsoleChannel())
    
    print("\n1️⃣ Enviando alerta INFO...")
    manager.send_alert(
        alert_type=AlertType.CUSTOM,
        severity=AlertSeverity.INFO,
        title="Sistema Iniciado",
        message="O sistema de alertas foi inicializado com sucesso.",
        metadata={"version": "1.0.0", "environment": "demo"}
    )
    
    print("\n2️⃣ Enviando alerta WARNING...")
    manager.send_alert(
        alert_type=AlertType.DATA_DRIFT,
        severity=AlertSeverity.WARNING,
        title="Data Drift Leve Detectado",
        message="Pequena mudança detectada na distribuição de features.",
        metadata={
            "features_afetadas": ["INDE", "IAA"],
            "drift_score": 0.15
        }
    )
    
    print("\n3️⃣ Enviando alerta ERROR...")
    manager.send_alert(
        alert_type=AlertType.MODEL_PERFORMANCE,
        severity=AlertSeverity.ERROR,
        title="Degradação de Performance",
        message="A acurácia do modelo caiu abaixo do threshold aceitável.",
        metadata={
            "accuracy_atual": 0.72,
            "accuracy_esperada": 0.85,
            "queda": "15%"
        }
    )
    
    print("\n4️⃣ Enviando alerta CRITICAL...")
    manager.send_alert(
        alert_type=AlertType.API_ERROR,
        severity=AlertSeverity.CRITICAL,
        title="API Indisponível",
        message="A API de produção não está respondendo!",
        metadata={
            "endpoint": "/predict",
            "status_code": 503,
            "tentativas": 5
        }
    )
    
    # Mostrar histórico
    print("\n📜 Histórico de Alertas:")
    print("-" * 40)
    for alert in manager.get_history():
        print(f"  [{alert['severity']}] {alert['title']}")


def demo_convenience_functions():
    """Demonstra funções de conveniência do AlertManager."""
    print("\n" + "="*60)
    print(" MÉTODOS DE CONVENIÊNCIA DO ALERTMANAGER")
    print("="*60)
    
    from src.monitoring.alerts import (
        get_alert_manager,
        send_alert,
        ConsoleChannel
    )
    
    # Configurar manager global
    manager = get_alert_manager()
    manager.add_channel(ConsoleChannel())
    
    print("\n1️⃣ Usando manager.alert_data_drift()...")
    manager.alert_data_drift(
        feature="INDE",
        drift_score=0.35,
        threshold=0.20
    )
    
    print("\n2️⃣ Usando manager.alert_prediction_drift()...")
    manager.alert_prediction_drift(
        drift_detected=True,
        current_distribution={"0": 0.35, "1": 0.65}
    )
    
    print("\n3️⃣ Usando manager.alert_model_performance()...")
    manager.alert_model_performance(
        metric_name="F1-Score",
        current_value=0.78,
        expected_value=0.85
    )
    
    print("\n4️⃣ Usando send_alert() standalone...")
    from src.monitoring.alerts import AlertType, AlertSeverity
    send_alert(
        alert_type=AlertType.CUSTOM,
        severity=AlertSeverity.INFO,
        title="Novo Modelo Registrado",
        message="Modelo v2.0.0 foi registrado no MLflow",
        metadata={"model_version": "2.0.0", "stage": "Staging"}
    )


def demo_drift_integration():
    """Demonstra integração com DriftDetector."""
    print("\n" + "="*60)
    print(" INTEGRAÇÃO COM DRIFT DETECTOR")
    print("="*60)
    
    from src.monitoring.alerts import ConsoleChannel, get_alert_manager
    from src.monitoring.drift import DriftDetector
    
    # Configurar alertas
    manager = get_alert_manager()
    manager.add_channel(ConsoleChannel())
    
    # Criar dados de referência
    np.random.seed(42)
    reference_data = pd.DataFrame({
        'INDE': np.random.normal(7.0, 1.5, 100),
        'IAA': np.random.normal(6.5, 1.2, 100),
        'IEG': np.random.normal(7.2, 1.0, 100),
        'IPS': np.random.normal(6.0, 1.5, 100)
    })
    
    # Criar detector com alertas habilitados
    detector = DriftDetector(reference_data=reference_data, enable_alerts=True)
    
    print("\n1️⃣ Testando com dados similares (sem drift)...")
    similar_data = pd.DataFrame({
        'INDE': np.random.normal(7.0, 1.5, 50),
        'IAA': np.random.normal(6.5, 1.2, 50),
        'IEG': np.random.normal(7.2, 1.0, 50),
        'IPS': np.random.normal(6.0, 1.5, 50)
    })
    result = detector.detect_data_drift(similar_data)
    print(f"   Drift detectado: {result['drift_detected']}")
    
    print("\n2️⃣ Testando com dados diferentes (com drift)...")
    drifted_data = pd.DataFrame({
        'INDE': np.random.normal(9.0, 1.5, 50),  # Média muito diferente
        'IAA': np.random.normal(4.0, 1.2, 50),  # Média muito diferente
        'IEG': np.random.normal(5.0, 1.0, 50),  # Média diferente
        'IPS': np.random.normal(8.0, 1.5, 50)   # Média diferente
    })
    result = detector.detect_data_drift(drifted_data)
    print(f"   Drift detectado: {result['drift_detected']}")
    print(f"   Features com drift: {result['features_with_drift']}")
    
    print("\n3️⃣ Testando prediction drift...")
    # Distribuição normal
    normal_predictions = [0, 1, 0, 1, 0, 1, 0, 0, 1, 1] * 10
    result = detector.detect_prediction_drift(normal_predictions)
    print(f"   Prediction drift (normal): {result['drift_detected']}")
    
    # Distribuição anormal (muito mais classe 1)
    biased_predictions = [1, 1, 1, 1, 1, 1, 1, 1, 0, 1] * 10
    result = detector.detect_prediction_drift(biased_predictions)
    print(f"   Prediction drift (biased): {result['drift_detected']}")


def demo_alert_status():
    """Demonstra consulta de status do AlertManager."""
    print("\n" + "="*60)
    print(" STATUS DO SISTEMA DE ALERTAS")
    print("="*60)
    
    from src.monitoring.alerts import get_alert_manager, ConsoleChannel
    import json
    
    manager = get_alert_manager()
    manager.add_channel(ConsoleChannel())
    
    status = manager.get_status()
    
    print("\n Status atual:")
    print(json.dumps(status, indent=2, default=str))
    
    print("\n📜 Histórico completo:")
    for i, alert in enumerate(manager.get_history(), 1):
        print(f"\n  {i}. [{alert['severity']}] {alert['title']}")
        print(f"     Tipo: {alert['alert_type']}")
        print(f"     Hora: {alert['timestamp']}")


def demo_slack_format():
    """Demonstra formatação de alerta para Slack."""
    print("\n" + "="*60)
    print("💬 FORMATAÇÃO SLACK")
    print("="*60)
    
    from src.monitoring.alerts import Alert, AlertType, AlertSeverity
    import json
    
    alert = Alert(
        alert_type=AlertType.DATA_DRIFT,
        severity=AlertSeverity.WARNING,
        title="Data Drift Detectado",
        message="Mudança significativa detectada nas features de entrada.",
        source="drift_detector",
        metadata={
            "features_afetadas": ["INDE", "IAA", "IEG"],
            "drift_score": 0.35,
            "timestamp": datetime.now().isoformat()
        }
    )
    
    slack_blocks = alert.to_slack_blocks()
    
    print("\n Slack Block Kit JSON:")
    print(json.dumps(slack_blocks, indent=2))
    
    print("\n📧 Email HTML Preview:")
    html = alert.to_email_html()
    # Mostrar apenas primeiras linhas
    lines = html.split('\n')[:30]
    print('\n'.join(lines))
    print("... (truncado)")


def main():
    """Executa todas as demonstrações."""
    print("\n" + "🔔"*30)
    print("   DEMONSTRAÇÃO DO SISTEMA DE ALERTAS")
    print("   Passos Mágicos MLOps")
    print("🔔"*30)
    
    try:
        demo_basic_alerts()
        demo_convenience_functions()
        demo_drift_integration()
        demo_alert_status()
        demo_slack_format()
        
        print("\n" + "="*60)
        print(" DEMONSTRAÇÃO CONCLUÍDA COM SUCESSO!")
        print("="*60)
        
        print("""
📝 Próximos passos para configurar alertas em produção:

1. Configure variáveis de ambiente:
   export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."
   export SMTP_HOST="smtp.gmail.com"
   export SMTP_USER="seu-email@gmail.com"
   export SMTP_PASSWORD="app-password"
   export ALERT_EMAIL_TO="equipe@exemplo.com"

2. Os alertas serão enviados automaticamente quando:
   - Drift de dados for detectado
   - Drift de predições for detectado
   - Erros críticos ocorrerem na API

3. Para testar manualmente via API:
   curl -X POST http://localhost:8000/alerts/test

4. Para enviar alerta via API:
   curl -X POST http://localhost:8000/alerts/send \\
     -H "Content-Type: application/json" \\
     -d '{"type": "system", "severity": "INFO", "title": "Teste", "message": "Mensagem de teste"}'
""")
        
    except Exception as e:
        print(f"\n❌ Erro na demonstração: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
