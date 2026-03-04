"""
Tests for Model Explainability (XAI) features.
"""
import pytest
from unittest.mock import MagicMock
from src.models.predictor import ModelPredictor

class TestModelXAI:
    """Test suite for SHAP-based model explanations using mocks for CI reliability."""

    @pytest.fixture
    def mock_predictor(self, monkeypatch):
        """Mock ModelPredictor to avoid loading real models in CI."""
        mock_explainer = MagicMock()
        mock_explainer.shap_values.return_value = [
            # Class 0 SHAP values
            [[-0.1, -0.05]],
            # Class 1 SHAP values (Ponto de Virada)
            [[0.2, 0.1]]
        ]
        mock_explainer.expected_value = [0.4, 0.6]
        
        # Mocking the initialization to skip file/mlflow loading
        monkeypatch.setattr(ModelPredictor, "_load_model", lambda x: None)
        
        predictor = ModelPredictor()
        predictor.model = MagicMock()
        predictor.explainer = mock_explainer
        predictor.feature_engineer = MagicMock()
        predictor.feature_engineer.feature_names = ["INDE_scaled", "IAA_scaled"]
        predictor.feature_engineer.get_feature_matrix.return_value = ([[0.5, 0.8]], None)
        predictor.preprocess_input = lambda x: MagicMock() # Return any object
        predictor.loaded_from = "file"
        
        return predictor

    def test_explain_single_prediction(self, mock_predictor, sample_input):
        """Test that explain returns valid SHAP contributions using mocked explainer."""
        explanation = mock_predictor.explain(sample_input, top_n=5)
        
        assert "base_value" in explanation
        assert explanation["base_value"] == 0.6
        assert "top_contributions" in explanation
        assert len(explanation["top_contributions"]) == 2
        
        # Verify contribution structure
        contribs = {c["feature"]: c["contribution"] for c in explanation["top_contributions"]}
        assert "INDE_scaled" in contribs
        assert contribs["INDE_scaled"] == 0.2

    def test_explain_without_explainer(self, mock_predictor, sample_input):
        """Test error handling when SHAP explainer is missing."""
        mock_predictor.explainer = None
        
        explanation = mock_predictor.explain(sample_input)
        assert "error" in explanation
        assert "not available" in explanation.get("error", "").lower()

    def test_explain_logic_branches(self, mock_predictor, sample_input):
        """Test logic branches for different SHAP output formats."""
        # Case: shap_values is a single array (new versions)
        import numpy as np
        mock_predictor.explainer.shap_values.return_value = np.array([[[0.1, 0.25], [0.3, 0.4]]]) # samples, features, classes
        
        explanation = mock_predictor.explain(sample_input)
        assert "base_value" in explanation
        # Top contribution should be the one with 0.4 (feature 1, class 1)
        assert explanation["top_contributions"][0]["contribution"] == 0.4
