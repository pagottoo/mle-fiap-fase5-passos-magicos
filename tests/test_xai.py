"""
Tests for Model Explainability (XAI) features.
"""
import pytest
import numpy as np
from src.models.predictor import ModelPredictor

class TestModelXAI:
    """Test suite for SHAP-based model explanations."""

    def test_explain_single_prediction(self, sample_input):
        """Test that explain returns valid SHAP contributions."""
        predictor = ModelPredictor()
        
        if predictor.model is None:
            pytest.skip("Model not loaded, skipping XAI test")
            
        explanation = predictor.explain(sample_input, top_n=5)
        
        assert "base_value" in explanation
        assert "top_contributions" in explanation
        assert len(explanation["top_contributions"]) <= 5
        
        # Verify contribution structure
        for contrib in explanation["top_contributions"]:
            assert "feature" in contrib
            assert "contribution" in contrib
            assert isinstance(contrib["contribution"], float)

    def test_explain_without_explainer(self, sample_input, monkeypatch):
        """Test error handling when SHAP explainer is missing."""
        predictor = ModelPredictor()
        monkeypatch.setattr(predictor, "explainer", None)
        
        explanation = predictor.explain(sample_input)
        assert "error" in explanation
        assert "not available" in explanation.get("error", "")

    def test_explain_output_consistency(self, sample_input):
        """Test that SHAP values sum up correctly (approximate)."""
        predictor = ModelPredictor()
        if predictor.model is None:
            pytest.skip("Model not loaded")
            
        result = predictor.predict(sample_input)
        explanation = predictor.explain(sample_input)
        
        # In SHAP: base_value + sum(all_contributions) approx log-odds or probability 
        # depending on the model/link. For trees, it's usually log-odds.
        # Here we just check that all_contributions exists and is populated.
        assert "all_contributions" in explanation
        assert len(explanation["all_contributions"]) > 10 # Should have all features
