"""
Coletor de métricas da API
"""
from typing import Dict, Any, List
from datetime import datetime
from collections import defaultdict
import threading


class MetricsCollector:
    """
    Coletor de métricas para monitoramento da API.
    """
    
    def __init__(self):
        self._lock = threading.Lock()
        self._reset_metrics()
    
    def _reset_metrics(self):
        """Reseta todas as métricas."""
        self.total_requests = 0
        self.total_predictions = 0
        self.predictions_by_class = defaultdict(int)
        self.request_durations: List[float] = []
        self.requests_by_endpoint = defaultdict(int)
        self.requests_by_status = defaultdict(int)
        self.start_time = datetime.now()
    
    def record_request(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        duration: float
    ) -> None:
        """
        Registra uma requisição.
        
        Args:
            endpoint: Endpoint acessado
            method: Método HTTP
            status_code: Código de status da resposta
            duration: Duração em segundos
        """
        with self._lock:
            self.total_requests += 1
            self.requests_by_endpoint[f"{method} {endpoint}"] += 1
            self.requests_by_status[status_code] += 1
            self.request_durations.append(duration)
            
            # Manter apenas últimas 10000 durações
            if len(self.request_durations) > 10000:
                self.request_durations = self.request_durations[-10000:]
    
    def record_prediction(self, prediction: int) -> None:
        """
        Registra uma predição.
        
        Args:
            prediction: Classe predita (0 ou 1)
        """
        with self._lock:
            self.total_predictions += 1
            self.predictions_by_class[prediction] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Retorna todas as métricas coletadas.
        
        Returns:
            Dicionário com métricas
        """
        with self._lock:
            uptime = (datetime.now() - self.start_time).total_seconds()
            
            # Calcular estatísticas de duração
            durations = self.request_durations
            avg_duration = sum(durations) / len(durations) if durations else 0
            max_duration = max(durations) if durations else 0
            min_duration = min(durations) if durations else 0
            
            # Calcular percentis
            sorted_durations = sorted(durations)
            p95_idx = int(len(sorted_durations) * 0.95)
            p99_idx = int(len(sorted_durations) * 0.99)
            p95_duration = sorted_durations[p95_idx] if sorted_durations else 0
            p99_duration = sorted_durations[p99_idx] if sorted_durations else 0
            
            # Calcular distribuição de predições
            total_preds = self.total_predictions
            pred_distribution = {}
            if total_preds > 0:
                pred_distribution = {
                    "class_0_pct": round(self.predictions_by_class[0] / total_preds * 100, 2),
                    "class_1_pct": round(self.predictions_by_class[1] / total_preds * 100, 2),
                }
            
            return {
                "uptime_seconds": round(uptime, 2),
                "total_requests": self.total_requests,
                "total_predictions": self.total_predictions,
                "requests_per_second": round(self.total_requests / uptime, 4) if uptime > 0 else 0,
                "predictions_by_class": dict(self.predictions_by_class),
                "prediction_distribution": pred_distribution,
                "latency": {
                    "avg_ms": round(avg_duration * 1000, 2),
                    "min_ms": round(min_duration * 1000, 2),
                    "max_ms": round(max_duration * 1000, 2),
                    "p95_ms": round(p95_duration * 1000, 2),
                    "p99_ms": round(p99_duration * 1000, 2),
                },
                "requests_by_endpoint": dict(self.requests_by_endpoint),
                "requests_by_status": dict(self.requests_by_status),
                "start_time": self.start_time.isoformat()
            }
    
    def reset(self) -> None:
        """Reseta todas as métricas."""
        with self._lock:
            self._reset_metrics()
