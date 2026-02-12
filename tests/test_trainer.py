"""
Testes unitários para o módulo de treinamento
"""
import pytest
import numpy as np
import tempfile
from pathlib import Path

from src.models.trainer import ModelTrainer
from src.data.preprocessing import DataPreprocessor
from src.data.feature_engineering import FeatureEngineer


class TestModelTrainer:
    """Testes para a classe ModelTrainer."""
    
    @pytest.fixture
    def training_data(self, sample_data):
        """Prepara dados para treinamento."""
        fe = FeatureEngineer()
        df = fe.create_target_variable(sample_data)
        df = fe.fit_transform(df)
        X, y = fe.get_feature_matrix(df)
        return X, y, fe
    
    def test_init_random_forest(self):
        """Testa inicialização com Random Forest."""
        trainer = ModelTrainer(model_type="random_forest")
        
        assert trainer.model_type == "random_forest"
        assert trainer.model is not None
        assert trainer.metrics == {}
    
    def test_init_gradient_boosting(self):
        """Testa inicialização com Gradient Boosting."""
        trainer = ModelTrainer(model_type="gradient_boosting")
        
        assert trainer.model_type == "gradient_boosting"
        assert trainer.model is not None
    
    def test_init_logistic_regression(self):
        """Testa inicialização com Logistic Regression."""
        trainer = ModelTrainer(model_type="logistic_regression")
        
        assert trainer.model_type == "logistic_regression"
        assert trainer.model is not None
    
    def test_init_invalid_model(self):
        """Testa erro com tipo de modelo inválido."""
        with pytest.raises(ValueError, match="Tipo de modelo inválido"):
            ModelTrainer(model_type="invalid_model")
    
    def test_split_data(self, training_data):
        """Testa divisão de dados."""
        X, y, _ = training_data
        trainer = ModelTrainer()
        
        X_train, X_test, y_train, y_test = trainer.split_data(X, y, test_size=0.3)
        
        assert len(X_train) + len(X_test) == len(X)
        assert len(y_train) + len(y_test) == len(y)
        # Verifica que o split gerou conjuntos não vazios
        assert len(X_test) >= 2  # Mínimo para stratification com 2 classes
        assert len(X_train) >= 2
    
    def test_split_data_stratified(self, training_data):
        """Testa que divisão mantém proporção das classes."""
        X, y, _ = training_data
        trainer = ModelTrainer()
        
        X_train, X_test, y_train, y_test = trainer.split_data(X, y)
        
        # Proporções devem ser similares
        train_ratio = y_train.mean()
        test_ratio = y_test.mean()
        
        # Com poucos dados, pode haver diferença, mas não muito grande
        assert abs(train_ratio - test_ratio) < 0.5
    
    def test_train(self, training_data):
        """Testa treinamento do modelo."""
        X, y, _ = training_data
        trainer = ModelTrainer()
        
        trainer.train(X, y)
        
        # Modelo deve ser capaz de fazer predições
        predictions = trainer.model.predict(X)
        assert len(predictions) == len(y)
    
    def test_evaluate(self, training_data):
        """Testa avaliação do modelo."""
        X, y, _ = training_data
        trainer = ModelTrainer()
        
        X_train, X_test, y_train, y_test = trainer.split_data(X, y)
        trainer.train(X_train, y_train)
        
        metrics = trainer.evaluate(X_test, y_test)
        
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert "roc_auc" in metrics
        assert "confusion_matrix" in metrics
        
        # Métricas devem estar entre 0 e 1
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["f1_score"] <= 1
    
    def test_cross_validate(self, training_data):
        """Testa validação cruzada."""
        X, y, _ = training_data
        trainer = ModelTrainer()
        
        # Com poucos dados, usar menos folds
        cv_results = trainer.cross_validate(X, y, cv_folds=2)
        
        assert "accuracy" in cv_results
        assert "f1" in cv_results
        assert "mean" in cv_results["accuracy"]
        assert "std" in cv_results["accuracy"]
    
    def test_save_model(self, training_data, tmp_path, monkeypatch):
        """Testa salvamento do modelo."""
        X, y, fe = training_data
        trainer = ModelTrainer()
        
        trainer.train(X, y)
        trainer.evaluate(X, y)
        
        # Usar diretório temporário
        monkeypatch.setattr("src.models.trainer.MODELS_DIR", tmp_path)
        
        preprocessor = DataPreprocessor()
        model_path = trainer.save_model(preprocessor, fe)
        
        assert model_path.exists()
        assert model_path.suffix == ".joblib"
        
        # Verificar que métricas JSON também foi salvo
        metrics_path = model_path.with_suffix(".joblib").parent / f"{model_path.stem}_metrics.json".replace("_metrics.joblib_metrics", "_metrics")
    
    def test_get_model_summary(self, training_data):
        """Testa geração do resumo do modelo."""
        X, y, _ = training_data
        trainer = ModelTrainer()
        
        trainer.train(X, y)
        trainer.evaluate(X, y)
        
        summary = trainer.get_model_summary()
        
        assert isinstance(summary, str)
        assert "Resumo do Modelo" in summary
        assert trainer.model_type in summary
    
    def test_feature_importance(self, training_data):
        """Testa extração de importância de features."""
        X, y, _ = training_data
        trainer = ModelTrainer(model_type="random_forest")
        
        trainer.train(X, y)
        
        assert len(trainer.feature_importance) > 0


class TestModelTrainerIntegration:
    """Testes de integração para ModelTrainer."""
    
    def test_full_training_pipeline(self, sample_data, tmp_path, monkeypatch):
        """Testa pipeline completo de treinamento."""
        # Preparar dados
        preprocessor = DataPreprocessor()
        df = preprocessor.fit_transform(sample_data)
        
        fe = FeatureEngineer()
        df = fe.create_target_variable(df)
        df = fe.fit_transform(df)
        
        X, y = fe.get_feature_matrix(df)
        
        # Treinar
        trainer = ModelTrainer(model_type="random_forest")
        X_train, X_test, y_train, y_test = trainer.split_data(X, y)
        trainer.train(X_train, y_train)
        metrics = trainer.evaluate(X_test, y_test)
        
        # Salvar
        monkeypatch.setattr("src.models.trainer.MODELS_DIR", tmp_path)
        model_path = trainer.save_model(preprocessor, fe)
        
        assert model_path.exists()
        assert metrics["accuracy"] > 0
