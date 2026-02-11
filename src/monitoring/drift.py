"""
Detector de drift para monitoramento contínuo do modelo
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import structlog

from ..config import LOGS_DIR, MONITORING_CONFIG

logger = structlog.get_logger()


# Flag para controlar envio de alertas (evitar import circular)
_alert_manager = None


def _get_alert_manager():
    """Obtém AlertManager de forma lazy para evitar import circular."""
    global _alert_manager
    if _alert_manager is None:
        try:
            from .alerts import get_alert_manager
            _alert_manager = get_alert_manager()
        except Exception as e:
            logger.warning("alert_manager_unavailable", error=str(e))
            _alert_manager = False  # Marca como indisponível
    return _alert_manager if _alert_manager else None


class DriftDetector:
    """
    Detector de drift para monitoramento do modelo em produção.
    
    Detecta:
    - Data drift: mudanças na distribuição dos dados de entrada
    - Concept drift: mudanças na relação entre features e target
    - Prediction drift: mudanças na distribuição das predições
    """
    
    def __init__(
        self, 
        reference_data: Optional[pd.DataFrame] = None,
        enable_alerts: bool = True
    ):
        """
        Inicializa o detector de drift.
        
        Args:
            reference_data: DataFrame de referência (dados de treino)
            enable_alerts: Se True, envia alertas quando drift é detectado
        """
        self.reference_data = reference_data
        self.reference_stats = {}
        self.drift_threshold = MONITORING_CONFIG["drift_threshold"]
        self.drift_history: List[Dict[str, Any]] = []
        self.enable_alerts = enable_alerts
        
        if reference_data is not None:
            self._compute_reference_stats()
    
    def _send_alert(
        self,
        alert_type: str,
        severity: str,
        title: str,
        message: str,
        details: Dict[str, Any]
    ) -> None:
        """
        Envia alerta se alertas estiverem habilitados.
        
        Args:
            alert_type: Tipo do alerta (data_drift, prediction_drift)
            severity: Severidade (INFO, WARNING, ERROR, CRITICAL)
            title: Título do alerta
            message: Mensagem do alerta
            details: Detalhes adicionais
        """
        if not self.enable_alerts:
            return
        
        alert_manager = _get_alert_manager()
        if alert_manager is None:
            return
        
        try:
            from .alerts import AlertType, AlertSeverity
            
            # Mapear tipos
            type_map = {
                "data_drift": AlertType.DATA_DRIFT,
                "prediction_drift": AlertType.PREDICTION_DRIFT
            }
            severity_map = {
                "INFO": AlertSeverity.INFO,
                "WARNING": AlertSeverity.WARNING,
                "ERROR": AlertSeverity.ERROR,
                "CRITICAL": AlertSeverity.CRITICAL
            }
            
            alert_manager.send_alert(
                alert_type=type_map.get(alert_type, AlertType.DATA_DRIFT),
                severity=severity_map.get(severity, AlertSeverity.WARNING),
                title=title,
                message=message,
                metadata=details
            )
        except Exception as e:
            logger.warning("failed_to_send_alert", error=str(e))
    
    def _compute_reference_stats(self) -> None:
        """Calcula estatísticas do conjunto de referência."""
        if self.reference_data is None:
            return
        
        numeric_cols = self.reference_data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            self.reference_stats[col] = {
                "mean": self.reference_data[col].mean(),
                "std": self.reference_data[col].std(),
                "min": self.reference_data[col].min(),
                "max": self.reference_data[col].max(),
                "median": self.reference_data[col].median(),
                "q25": self.reference_data[col].quantile(0.25),
                "q75": self.reference_data[col].quantile(0.75)
            }
        
        logger.info("reference_stats_computed", features=len(self.reference_stats))
    
    def detect_data_drift(
        self, 
        current_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Detecta drift nos dados de entrada.
        
        Usa teste PSI (Population Stability Index) simplificado.
        
        Args:
            current_data: DataFrame com dados atuais
            
        Returns:
            Dicionário com resultados do drift
        """
        if not self.reference_stats:
            return {"error": "Estatísticas de referência não disponíveis"}
        
        drift_results = {
            "timestamp": datetime.now().isoformat(),
            "features_analyzed": 0,
            "features_with_drift": 0,
            "drift_detected": False,
            "details": {}
        }
        
        numeric_cols = current_data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col not in self.reference_stats:
                continue
            
            drift_results["features_analyzed"] += 1
            
            ref_stats = self.reference_stats[col]
            curr_mean = current_data[col].mean()
            curr_std = current_data[col].std()
            
            # Calcular z-score da diferença de médias
            if ref_stats["std"] > 0:
                z_score = abs(curr_mean - ref_stats["mean"]) / ref_stats["std"]
            else:
                z_score = 0
            
            # Detectar drift se z-score > 2 (aproximadamente)
            has_drift = z_score > 2
            
            if has_drift:
                drift_results["features_with_drift"] += 1
            
            drift_results["details"][col] = {
                "reference_mean": round(ref_stats["mean"], 4),
                "current_mean": round(curr_mean, 4),
                "z_score": round(z_score, 4),
                "drift_detected": has_drift
            }
        
        # Drift geral se mais de 20% das features apresentam drift
        if drift_results["features_analyzed"] > 0:
            drift_ratio = drift_results["features_with_drift"] / drift_results["features_analyzed"]
            drift_results["drift_detected"] = drift_ratio > self.drift_threshold
            drift_results["drift_ratio"] = round(drift_ratio, 4)
        
        # Salvar no histórico
        self.drift_history.append(drift_results)
        
        if drift_results["drift_detected"]:
            logger.warning("data_drift_detected", **drift_results)
            
            # Enviar alerta
            features_com_drift = [
                col for col, info in drift_results["details"].items()
                if info.get("drift_detected", False)
            ]
            self._send_alert(
                alert_type="data_drift",
                severity="WARNING" if drift_results["drift_ratio"] < 0.5 else "ERROR",
                title="Data Drift Detectado",
                message=f"Drift detectado em {drift_results['features_with_drift']} de "
                        f"{drift_results['features_analyzed']} features "
                        f"({drift_results['drift_ratio']:.1%} das features afetadas)",
                details={
                    "features_com_drift": features_com_drift,
                    "drift_ratio": drift_results["drift_ratio"],
                    "timestamp": drift_results["timestamp"]
                }
            )
        else:
            logger.info("no_data_drift", features_analyzed=drift_results["features_analyzed"])
        
        return drift_results
    
    def detect_prediction_drift(
        self,
        predictions: List[int],
        reference_distribution: Dict[int, float] = None
    ) -> Dict[str, Any]:
        """
        Detecta drift na distribuição das predições.
        
        Args:
            predictions: Lista de predições recentes
            reference_distribution: Distribuição de referência {classe: proporção}
            
        Returns:
            Dicionário com resultados do drift
        """
        if not predictions:
            return {"error": "Nenhuma predição fornecida"}
        
        # Calcular distribuição atual
        predictions_array = np.array(predictions)
        current_distribution = {
            0: (predictions_array == 0).mean(),
            1: (predictions_array == 1).mean()
        }
        
        # Se não há referência, usar 50/50 como baseline
        if reference_distribution is None:
            reference_distribution = {0: 0.5, 1: 0.5}
        
        # Calcular divergência
        drift_score = abs(
            current_distribution[1] - reference_distribution.get(1, 0.5)
        )
        
        drift_results = {
            "timestamp": datetime.now().isoformat(),
            "sample_size": len(predictions),
            "current_distribution": {
                str(k): round(v, 4) for k, v in current_distribution.items()
            },
            "reference_distribution": {
                str(k): round(v, 4) for k, v in reference_distribution.items()
            },
            "drift_score": round(drift_score, 4),
            "drift_detected": drift_score > self.drift_threshold
        }
        
        if drift_results["drift_detected"]:
            logger.warning("prediction_drift_detected", **drift_results)
            
            # Enviar alerta
            self._send_alert(
                alert_type="prediction_drift",
                severity="WARNING" if drift_score < 0.3 else "ERROR",
                title="Prediction Drift Detectado",
                message=f"Mudança significativa na distribuição de predições. "
                        f"Classe 1: {current_distribution[1]:.1%} (esperado: {reference_distribution.get(1, 0.5):.1%}). "
                        f"Drift score: {drift_score:.4f}",
                details={
                    "sample_size": len(predictions),
                    "current_class_1_ratio": current_distribution[1],
                    "expected_class_1_ratio": reference_distribution.get(1, 0.5),
                    "drift_score": drift_score,
                    "timestamp": drift_results["timestamp"]
                }
            )
        
        return drift_results
    
    def analyze_recent_predictions(
        self,
        window_size: int = MONITORING_CONFIG["performance_window"]
    ) -> Dict[str, Any]:
        """
        Analisa predições recentes do log.
        
        Args:
            window_size: Número de predições a analisar
            
        Returns:
            Dicionário com análise
        """
        from .logger import get_recent_predictions
        
        recent = get_recent_predictions(window_size)
        
        if not recent:
            return {"error": "Nenhuma predição registrada"}
        
        predictions = [p["output"]["prediction"] for p in recent]
        
        # Extrair dados de entrada para análise de drift
        input_data = [p["input"] for p in recent]
        df_input = pd.DataFrame(input_data)
        
        # Detectar drifts
        prediction_drift = self.detect_prediction_drift(predictions)
        data_drift = self.detect_data_drift(df_input) if not df_input.empty else {}
        
        return {
            "window_size": len(recent),
            "prediction_drift": prediction_drift,
            "data_drift": data_drift,
            "summary": {
                "predictions_analyzed": len(predictions),
                "class_0_count": predictions.count(0),
                "class_1_count": predictions.count(1),
                "class_1_ratio": round(sum(predictions) / len(predictions), 4)
            }
        }
    
    def get_drift_report(self) -> Dict[str, Any]:
        """
        Gera relatório completo de drift.
        
        Returns:
            Relatório de drift
        """
        return {
            "generated_at": datetime.now().isoformat(),
            "reference_features": list(self.reference_stats.keys()),
            "drift_threshold": self.drift_threshold,
            "drift_history_count": len(self.drift_history),
            "recent_drift_events": self.drift_history[-10:] if self.drift_history else []
        }
    
    def save_reference_data(self, filepath: str) -> None:
        """Salva dados de referência para uso posterior."""
        if self.reference_data is not None:
            self.reference_data.to_parquet(filepath)
            logger.info("reference_data_saved", path=filepath)
    
    def load_reference_data(self, filepath: str) -> None:
        """Carrega dados de referência."""
        self.reference_data = pd.read_parquet(filepath)
        self._compute_reference_stats()
        logger.info("reference_data_loaded", path=filepath)
