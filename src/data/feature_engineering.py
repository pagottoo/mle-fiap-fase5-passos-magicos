"""
Feature engineering module.
"""
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
import structlog

logger = structlog.get_logger()


class FeatureEngineer:
    """
    Handles feature engineering for the Turning Point model.
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
        Create target variable for Turning Point prediction.

        Target rule:
        - use `PONTO_VIRADA` directly when present (Yes/No)
        - otherwise use high-score heuristics (INDE, IPV, PEDRA)

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with target column.
        """
        logger.info("creating_target_variable")
        
        df = df.copy()
        
        # Initialize with 0 (no turning point)
        df["ponto_virada_pred"] = 0
        
        # If PONTO_VIRADA exists, use it directly
        if "PONTO_VIRADA" in df.columns:
            # Convert to binary
            df["ponto_virada_pred"] = df["PONTO_VIRADA"].apply(
                lambda x: 1 if str(x).strip().lower() in ["sim", "yes", "1", "true"] else 0
            )
        else:
            # Create target from high-performance signals
            # IPV (Turning Point Propensity Index) >= 7.5
            if "IPV" in df.columns:
                df["IPV_numeric"] = pd.to_numeric(df["IPV"], errors="coerce")
                df.loc[df["IPV_numeric"] >= 7.5, "ponto_virada_pred"] = 1
                df = df.drop(columns=["IPV_numeric"])
            
            # High INDE (>= 7.5) also indicates potential
            if "INDE" in df.columns:
                df["INDE_numeric"] = pd.to_numeric(df["INDE"], errors="coerce")
                df.loc[df["INDE_numeric"] >= 7.5, "ponto_virada_pred"] = 1
                df = df.drop(columns=["INDE_numeric"])
            
            # Higher PEDRA tiers indicate stronger probability
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
        Select relevant model features.

        Args:
            df: Input DataFrame.

        Returns:
            Tuple with numeric and categorical feature lists.
        """
        logger.info("selecting_features")
        
        # Main numeric features (evaluation indexes)
        # NOTA_PORT, NOTA_MAT, NOTA_ING removed because they are absent in the current dataset
        numeric_candidates = [
            "INDE", "IAA", "IEG", "IPS", "IDA", "IPP", "IPV", "IAN",
            "IDADE_ALUNO", "ANOS_PM"
        ]
        
        # Main categorical features
        # PONTO_VIRADA removed to avoid data leakage (target is derived from it)
        categorical_candidates = [
            "INSTITUICAO_ENSINO_ALUNO", "FASE", "PEDRA",
            "BOLSISTA", "SINALIZADOR_INGRESSANTE"
        ]
        
        # Keep only existing columns
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
        Encode categorical variables using LabelEncoding.

        Args:
            df: Input DataFrame.
            categorical_features: Categorical feature list.
            fit: If True, fit encoders; otherwise reuse existing ones.

        Returns:
            DataFrame with encoded variables.
        """
        logger.info("encoding_categorical", features=categorical_features)
        
        df = df.copy()
        
        for col in categorical_features:
            if col not in df.columns:
                continue
                
            if fit:
                le = LabelEncoder()
                # Handle unknown values
                df[col] = df[col].astype(str)
                df[f"{col}_encoded"] = le.fit_transform(df[col])
                self.label_encoders[col] = le
            else:
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    df[col] = df[col].astype(str)
                    # Map unknown values to -1
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
        Scale numeric variables using StandardScaler.

        Args:
            df: Input DataFrame.
            numeric_features: Numeric feature list.
            fit: If True, fit scaler; otherwise reuse existing scaler.

        Returns:
            DataFrame with scaled variables.
        """
        logger.info("scaling_numeric", features=numeric_features)
        
        df = df.copy()
        
        # Convert columns to numeric
        for col in numeric_features:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # Keep only existing features
        existing_features = [c for c in numeric_features if c in df.columns]
        
        if not existing_features:
            return df
        
        # Fill NaN with 0 before scaling
        df[existing_features] = df[existing_features].fillna(0)
        
        if fit:
            scaled_values = self.scaler.fit_transform(df[existing_features])
        else:
            scaled_values = self.scaler.transform(df[existing_features])
        
        # Create scaled columns
        for i, col in enumerate(existing_features):
            df[f"{col}_scaled"] = scaled_values[:, i]
        
        return df
    
    def fit(self, df: pd.DataFrame) -> "FeatureEngineer":
        """
        Fit feature engineer on training data.

        Args:
            df: Training DataFrame.

        Returns:
            Self.
        """
        logger.info("fitting_feature_engineer")
        
        self._numeric_features, self._categorical_features = self.select_features(df)
        
        # Fit encoders and scaler
        df = self.encode_categorical(df, self._categorical_features, fit=True)
        df = self.scale_numeric(df, self._numeric_features, fit=True)
        
        # Save final feature names
        self._feature_names = (
            [f"{c}_scaled" for c in self._numeric_features if c in df.columns] +
            [f"{c}_encoded" for c in self._categorical_features if c in df.columns]
        )
        
        self._fitted = True
        logger.info("feature_engineer_fitted", features=len(self._feature_names))
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted encoders/scaler.

        Args:
            df: DataFrame to transform.

        Returns:
            Transformed DataFrame.
        """
        if not self._fitted:
            raise ValueError("FeatureEngineer must be fitted before transform.")
        
        df = self.encode_categorical(df, self._categorical_features, fit=False)
        df = self.scale_numeric(df, self._numeric_features, fit=False)
        
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform in a single operation.

        Args:
            df: DataFrame to process.

        Returns:
            Processed DataFrame.
        """
        self.fit(df)
        return self.transform(df)
    
    def get_feature_matrix(
        self, 
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Extract feature matrix and target vector from DataFrame.

        Args:
            df: Processed DataFrame.

        Returns:
            Tuple `(X, y)` with feature matrix and target vector.
        """
        # Keep only existing feature columns
        available_features = [c for c in self._feature_names if c in df.columns]
        
        X = df[available_features].values
        
        y = None
        if "ponto_virada_pred" in df.columns:
            y = df["ponto_virada_pred"].values
        
        return X, y
    
    @property
    def feature_names(self) -> List[str]:
        """Return engineered feature names."""
        return self._feature_names
