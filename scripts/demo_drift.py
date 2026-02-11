"""
Demonstração do sistema de Drift Detection
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.monitoring.drift import DriftDetector
from src.feature_store import FeatureStore
import pandas as pd
import numpy as np

def main():
    print('=' * 60)
    print(' DEMONSTRAÇÃO DE DRIFT DETECTION')
    print('=' * 60)

    # 1. Carregar dados de referência da Feature Store
    print('\n[1] Carregando dados de referência da Feature Store...')
    fs = FeatureStore()
    ref_data = fs.get_training_data('passos_magicos_training')
    print(f'   * Dados carregados: {len(ref_data)} registros')

    # 2. Criar detector com dados de referência
    print('\n[2] Inicializando Drift Detector...')
    numeric_cols = ['INDE', 'IAA', 'IEG', 'IPS', 'IDA', 'IPP', 'IPV', 'IAN']
    detector = DriftDetector(reference_data=ref_data[numeric_cols])
    print(f'   * Detector inicializado com {len(numeric_cols)} features')
    print(f'   * Método: Kolmogorov-Smirnov Test (p-value threshold=0.05)')

    # 3. Simular dados SEM drift (mesma distribuição)
    print('\n[3] Testando dados SEM drift (amostra da mesma distribuição)...')
    sample_normal = ref_data[numeric_cols].sample(50, random_state=42)
    drift_result_normal = detector.detect_data_drift(sample_normal)
    print(f'   -> Drift detectado: {drift_result_normal["drift_detected"]}')
    print(f'   -> Features com drift: {drift_result_normal["features_with_drift"]}')

    # 4. Simular dados COM drift (distribuição alterada)
    print('\n[4] Testando dados COM drift (valores alterados artificialmente)...')
    print('   Simulação: INDE +3 pontos, IPV *1.5')
    sample_drift = sample_normal.copy()
    sample_drift['INDE'] = sample_drift['INDE'] + 3  # Adiciona 3 pontos ao INDE
    sample_drift['IPV'] = sample_drift['IPV'] * 1.5   # Aumenta IPV em 50%
    drift_result_drift = detector.detect_data_drift(sample_drift)
    print(f'   -> Drift detectado: {drift_result_drift["drift_detected"]}')
    print(f'   -> Features com drift: {drift_result_drift["features_with_drift"]}')

    # 5. Detalhes por feature
    print('\n[5] Detalhes do teste estatístico por feature:')
    print('   ' + '-' * 50)
    print(f'   {"Feature":<10} {"Z-Score":<12} {"Drift?"}')
    print('   ' + '-' * 50)
    for feat, details in drift_result_drift['details'].items():
        status = '[DRIFT] DRIFT' if details.get('drift', False) else '[OK] OK'
        z_score = details.get('z_score', 0)
        print(f'   {feat:<10} {z_score:<12.4f} {status}')
    print('   ' + '-' * 50)

    # 6. Prediction Drift
    print('\n[6] Testando Prediction Drift...')
    # Simular predições normais (distribuição balanceada)
    predictions_normal = [0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0]
    pred_drift_normal = detector.detect_prediction_drift(predictions_normal)
    print(f'   Predições normais: {predictions_normal}')
    print(f'   -> Drift detectado: {pred_drift_normal["drift_detected"]}')
    
    # Simular predições com drift (muitos positivos)
    predictions_drift = [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]
    pred_drift_result = detector.detect_prediction_drift(predictions_drift)
    print(f'   Predições anormais: {predictions_drift}')
    print(f'   -> Drift detectado: {pred_drift_result["drift_detected"]}')

    # 7. Relatório completo
    print('\n[7] Relatório de Drift:')
    report = detector.get_drift_report()
    print(f'   * Total de verificações: {report["total_checks"]}')
    print(f'   * Drifts detectados: {report["drifts_detected"]}')
    print(f'   * Última verificação: {report["last_check"]}')

    print('\n' + '=' * 60)
    print(' Demonstração de Drift Detection concluída!')
    print('=' * 60)
    
    # Resumo conceitual
    print('\n CONCEITOS:')
    print('''
    Data Drift: Mudança na distribuição das features de entrada
    - Detectado via Kolmogorov-Smirnov Test
    - p-value < 0.05 indica drift significativo
    
    Prediction Drift: Mudança na distribuição das predições
    - Monitora proporção de classes
    - Alerta se % de positivos muda muito
    
    Uso em Produção:
    - Monitorar continuamente as predições
    - Re-treinar modelo quando drift persistir
    - Alertar equipe para investigar causa
    ''')

if __name__ == "__main__":
    main()
