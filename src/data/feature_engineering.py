"""
Módulo de engenharia de features
"""
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
import structlog

logger = structlog.get_logger()


class FeatureEngineer:
    """
    Classe responsável pela engenharia de features para o modelo de Ponto de Virada.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self._fitted = False
        self._feature_names = []
        self._numeric_features = []
        self._categorical_features = []
    
    def create_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria a variável target de predição de Ponto de Virada.
        
        O Ponto de Virada representa o momento transformador na vida do aluno,
        quando ele "vira a chave" e começa uma mudança real em sua trajetória.
        
        A variável é baseada em:
        - PONTO_VIRADA: coluna direta se existir (Sim/Não)
        - Combinação de indicadores altos (INDE, IPV, IDA)
        
        Args:
            df: DataFrame com os dados
            
        Returns:
            DataFrame com coluna target
        """
        logger.info("creating_target_variable")
        
        df = df.copy()
        
        # Inicializa como 0 (não atingiu ponto de virada)
        df["ponto_virada_pred"] = 0
        
        # Se existe coluna PONTO_VIRADA, usar diretamente
        if "PONTO_VIRADA" in df.columns:
            # Converter para binário
            df["ponto_virada_pred"] = df["PONTO_VIRADA"].apply(
                lambda x: 1 if str(x).strip().lower() in ["sim", "yes", "1", "true"] else 0
            )
        else:
            # Criar target baseado em indicadores de alta performance
            # IPV (Índice de Propensão à Virada) >= 7.5
            if "IPV" in df.columns:
                df["IPV_numeric"] = pd.to_numeric(df["IPV"], errors="coerce")
                df.loc[df["IPV_numeric"] >= 7.5, "ponto_virada_pred"] = 1
                df = df.drop(columns=["IPV_numeric"])
            
            # INDE alto (>= 7.5) também indica potencial
            if "INDE" in df.columns:
                df["INDE_numeric"] = pd.to_numeric(df["INDE"], errors="coerce")
                df.loc[df["INDE_numeric"] >= 7.5, "ponto_virada_pred"] = 1
                df = df.drop(columns=["INDE_numeric"])
            
            # PEDRA alta (Topázio ou Ametista com bom desempenho)
            if "PEDRA" in df.columns:
                df.loc[df["PEDRA"].isin(["Topázio", "Ágata"]), "ponto_virada_pred"] = 1
        
        logger.info(
            "target_created",
            total=len(df),
            ponto_virada_count=df["ponto_virada_pred"].sum(),
            ponto_virada_pct=round(df["ponto_virada_pred"].mean() * 100, 2)
        )
        
        return df
    
    def select_features(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Seleciona features relevantes para o modelo.
        
        Args:
            df: DataFrame com os dados
            
        Returns:
            Tupla com listas de features numéricas e categóricas
        """
        logger.info("selecting_features")
        
        # Features numéricas de interesse (índices de avaliação)
        # NOTA_PORT, NOTA_MAT, NOTA_ING removidas pois não existem no dataset atual
        numeric_candidates = [
            "INDE", "IAA", "IEG", "IPS", "IDA", "IPP", "IPV", "IAN",
            "IDADE_ALUNO", "ANOS_PM"
        ]
        
        # Features categóricas de interesse
        # PONTO_VIRADA REMOVIDO para evitar data leakage (target derivado desta coluna)
        categorical_candidates = [
            "INSTITUICAO_ENSINO_ALUNO", "FASE", "PEDRA",
            "BOLSISTA", "SINALIZADOR_INGRESSANTE"
        ]
        
        # Filtrar apenas colunas que existem no DataFrame
        numeric_features = [c for c in numeric_candidates if c in df.columns]
        categorical_features = [c for c in categorical_candidates if c in df.columns]
        
        logger.info(
            "features_selected",
            numeric=len(numeric_features),
            categorical=len(categorical_features)
        )
        
        return numeric_features, categorical_features
    
    def encode_categorical(
        self, 
        df: pd.DataFrame, 
        categorical_features: List[str],
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Codifica variáveis categóricas usando Label Encoding.
        
        Args:
            df: DataFrame com os dados
            categorical_features: Lista de features categóricas
            fit: Se True, ajusta os encoders; se False, usa encoders existentes
            
        Returns:
            DataFrame com variáveis codificadas
        """
        logger.info("encoding_categorical", features=categorical_features)
        
        df = df.copy()
        
        for col in categorical_features:
            if col not in df.columns:
                continue
                
            if fit:
                le = LabelEncoder()
                # Tratar valores desconhecidos
                df[col] = df[col].astype(str)
                df[f"{col}_encoded"] = le.fit_transform(df[col])
                self.label_encoders[col] = le
            else:
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    df[col] = df[col].astype(str)
                    # Mapear valores desconhecidos para -1
                    df[f"{col}_encoded"] = df[col].apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )
        
        return df
    
    def scale_numeric(
        self, 
        df: pd.DataFrame, 
        numeric_features: List[str],
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Normaliza variáveis numéricas usando StandardScaler.
        
        Args:
            df: DataFrame com os dados
            numeric_features: Lista de features numéricas
            fit: Se True, ajusta o scaler; se False, usa scaler existente
            
        Returns:
            DataFrame com variáveis normalizadas
        """
        logger.info("scaling_numeric", features=numeric_features)
        
        df = df.copy()
        
        # Converter para numérico
        for col in numeric_features:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # Filtrar features que existem
        existing_features = [c for c in numeric_features if c in df.columns]
        
        if not existing_features:
            return df
        
        # Preencher NaN com 0 antes de escalar
        df[existing_features] = df[existing_features].fillna(0)
        
        if fit:
            scaled_values = self.scaler.fit_transform(df[existing_features])
        else:
            scaled_values = self.scaler.transform(df[existing_features])
        
        # Criar colunas escaladas
        for i, col in enumerate(existing_features):
            df[f"{col}_scaled"] = scaled_values[:, i]
        
        return df
    
    def fit(self, df: pd.DataFrame) -> "FeatureEngineer":
        """
        Ajusta o engenheiro de features aos dados de treino.
        
        Args:
            df: DataFrame de treino
            
        Returns:
            Self
        """
        logger.info("fitting_feature_engineer")
        
        self._numeric_features, self._categorical_features = self.select_features(df)
        
        # Ajustar encoders e scalers
        df = self.encode_categorical(df, self._categorical_features, fit=True)
        df = self.scale_numeric(df, self._numeric_features, fit=True)
        
        # Salvar nomes das features finais
        self._feature_names = (
            [f"{c}_scaled" for c in self._numeric_features if c in df.columns] +
            [f"{c}_encoded" for c in self._categorical_features if c in df.columns]
        )
        
        self._fitted = True
        logger.info("feature_engineer_fitted", features=len(self._feature_names))
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforma os dados usando encoders e scalers ajustados.
        
        Args:
            df: DataFrame a ser transformado
            
        Returns:
            DataFrame transformado
        """
        if not self._fitted:
            raise ValueError("O FeatureEngineer precisa ser ajustado (fit) primeiro.")
        
        df = self.encode_categorical(df, self._categorical_features, fit=False)
        df = self.scale_numeric(df, self._numeric_features, fit=False)
        
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ajusta e transforma os dados em uma única operação.
        
        Args:
            df: DataFrame a ser processado
            
        Returns:
            DataFrame processado
        """
        self.fit(df)
        return self.transform(df)
    
    def get_feature_matrix(
        self, 
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Extrai matriz de features e target do DataFrame.
        
        Args:
            df: DataFrame processado
            
        Returns:
            Tupla (X, y) com matriz de features e vetor target
        """
        # Selecionar apenas colunas de features que existem
        available_features = [c for c in self._feature_names if c in df.columns]
        
        X = df[available_features].values
        
        y = None
        if "ponto_virada_pred" in df.columns:
            y = df["ponto_virada_pred"].values
        
        return X, y
    
    @property
    def feature_names(self) -> List[str]:
        """Retorna nomes das features."""
        return self._feature_names
