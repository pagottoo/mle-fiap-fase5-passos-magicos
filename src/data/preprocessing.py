"""
Módulo de pré-processamento de dados
"""
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from pathlib import Path
import structlog

logger = structlog.get_logger()


class DataPreprocessor:
    """
    Classe responsável pelo pré-processamento dos dados da Passos Mágicos.
    """
    
    def __init__(self):
        self.columns_to_drop = ["NOME"]  # Colunas identificadoras
        self._fitted = False
        self._numeric_columns = []
        self._categorical_columns = []
    
    def load_data(self, file_path: Path) -> pd.DataFrame:
        """
        Carrega dados do arquivo CSV ou Excel.
        
        Args:
            file_path: Caminho para o arquivo de dados
            
        Returns:
            DataFrame com os dados carregados
        """
        logger.info("loading_data", file_path=str(file_path))
        
        if file_path.suffix == ".csv":
            df = pd.read_csv(file_path, delimiter=";")
        elif file_path.suffix in [".xlsx", ".xls"]:
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Formato de arquivo não suportado: {file_path.suffix}")
        
        logger.info("data_loaded", rows=len(df), columns=len(df.columns))
        return df
    
    def filter_columns_by_year(self, df: pd.DataFrame, year: str) -> pd.DataFrame:
        """
        Filtra colunas para um ano específico, removendo sufixos de ano.
        
        Args:
            df: DataFrame original
            year: Ano para filtrar (ex: "2022")
            
        Returns:
            DataFrame filtrado com colunas renomeadas
        """
        logger.info("filtering_columns_by_year", year=year)
        
        # Selecionar colunas do ano especificado ou sem ano
        year_suffix = f"_{year}"
        other_years = ["_2020", "_2021", "_2022", "_2023", "_2024"]
        other_years = [y for y in other_years if y != year_suffix]
        
        columns_to_keep = []
        for col in df.columns:
            # Manter se tem o sufixo do ano desejado ou não tem sufixo de ano
            has_year_suffix = any(y in col for y in other_years)
            if not has_year_suffix:
                columns_to_keep.append(col)
        
        df_filtered = df[columns_to_keep].copy()
        
        # Remover sufixo do ano das colunas
        df_filtered.columns = [
            col.replace(year_suffix, "") for col in df_filtered.columns
        ]
        
        logger.info("columns_filtered", columns=len(df_filtered.columns))
        return df_filtered
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpa dados removendo linhas com muitos valores nulos.
        
        Args:
            df: DataFrame a ser limpo
            
        Returns:
            DataFrame limpo
        """
        logger.info("cleaning_data", initial_rows=len(df))
        
        # Remover colunas identificadoras
        cols_to_drop = [c for c in self.columns_to_drop if c in df.columns]
        df = df.drop(columns=cols_to_drop, errors="ignore")
        
        # Remover linhas onde todas as colunas numéricas são nulas
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df = df.dropna(subset=numeric_cols, how="all")
        
        # Remover linhas com mais de 50% de valores nulos
        threshold = len(df.columns) * 0.5
        df = df.dropna(thresh=threshold)
        
        logger.info("data_cleaned", final_rows=len(df))
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Trata valores ausentes no DataFrame.
        
        Args:
            df: DataFrame com valores ausentes
            
        Returns:
            DataFrame com valores tratados
        """
        logger.info("handling_missing_values")
        
        df = df.copy()
        
        # Para colunas numéricas: preencher com a mediana
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                median_value = df[col].median()
                df[col] = df[col].fillna(median_value)
        
        # Para colunas categóricas: preencher com "Desconhecido"
        categorical_cols = df.select_dtypes(include=["object"]).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna("Desconhecido")
        
        logger.info("missing_values_handled")
        return df
    
    def fit(self, df: pd.DataFrame) -> "DataPreprocessor":
        """
        Ajusta o preprocessador aos dados (salva estatísticas).
        
        Args:
            df: DataFrame de treino
            
        Returns:
            Self
        """
        self._numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        self._categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()
        self._fitted = True
        logger.info("preprocessor_fitted", 
                   numeric_cols=len(self._numeric_columns),
                   categorical_cols=len(self._categorical_columns))
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica transformações aos dados.
        
        Args:
            df: DataFrame a ser transformado
            
        Returns:
            DataFrame transformado
        """
        if not self._fitted:
            raise ValueError("O preprocessador precisa ser ajustado (fit) primeiro.")
        
        df = self.clean_data(df)
        df = self.handle_missing_values(df)
        
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
    
    def prepare_dataset(
        self, 
        file_path: Path, 
        year: str = "2022"
    ) -> pd.DataFrame:
        """
        Pipeline completo de preparação dos dados.
        
        Args:
            file_path: Caminho para o arquivo de dados
            year: Ano dos dados a processar
            
        Returns:
            DataFrame preparado
        """
        logger.info("preparing_dataset", file_path=str(file_path), year=year)
        
        df = self.load_data(file_path)
        df = self.filter_columns_by_year(df, year)
        df = self.fit_transform(df)
        
        logger.info("dataset_prepared", rows=len(df), columns=len(df.columns))
        return df
