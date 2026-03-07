from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional

class StudentData(BaseModel):
    """Schema for student input payload.

    Required fields for prediction:
    INDE, IAA, IEG, IPS, IDA, IPP, IPV, IAN, FASE, PEDRA, BOLSISTA.
    """
    # Required features for inference
    INDE: float = Field(..., description="Educational Development Index")
    IAA: float = Field(..., description="Self-Assessment Index")
    IEG: float = Field(..., description="Engagement Index")
    IPS: float = Field(..., description="Psychosocial Index")
    IDA: float = Field(..., description="Academic Performance Index")
    IPP: float = Field(..., description="Participation Index")
    IPV: float = Field(..., description="Turning Point Propensity Index")
    IAN: float = Field(..., description="Level Adequacy Index")
    FASE: str = Field(..., description="Student stage/class")
    PEDRA: str = Field(..., description="Student cluster (Quartzo, Ametista, etc.)")
    BOLSISTA: str = Field(..., description="Scholarship status (Yes/No)")

    # Optional fields (not used by the current model)
    IDADE_ALUNO: Optional[int] = Field(None, description="Student age")
    ANOS_PM: Optional[int] = Field(None, description="Years in Passos Magicos")
    INSTITUICAO_ENSINO_ALUNO: Optional[str] = Field(None, description="School institution type")
    PONTO_VIRADA: Optional[str] = Field(None, description="Observed turning point label (not used as feature)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "INDE": 7.5,
                "IAA": 8.0,
                "IEG": 7.0,
                "IPS": 6.5,
                "IDA": 7.2,
                "IPP": 6.8,
                "IPV": 7.5,
                "IAN": 5.0,
                "IDADE_ALUNO": 12,
                "ANOS_PM": 2,
                "INSTITUICAO_ENSINO_ALUNO": "Escola Pública",
                "FASE": "3",
                "PEDRA": "Ametista",
                "PONTO_VIRADA": "Não",
                "BOLSISTA": "Não"
            }
        }


class PredictionResponse(BaseModel):
    """Schema for prediction response."""
    prediction_id: str = Field(..., description="Unique identifier for this prediction")
    prediction: int
    label: str
    probability_no_turning_point: float
    probability_turning_point: float
    confidence: float
    model_version: str
    timestamp: str


class FeedbackRequest(BaseModel):
    """Schema for ground truth feedback."""
    prediction_id: str = Field(..., description="The ID of the original prediction")
    actual_outcome: int = Field(..., description="The actual outcome (0 or 1)")
    comment: Optional[str] = None


class FeatureContribution(BaseModel):
    """Individual feature contribution to the prediction."""
    feature: str
    contribution: float


class ExplanationResponse(BaseModel):
    """Schema for prediction explanation (XAI)."""
    prediction_id: str = Field(..., description="Unique identifier for this prediction")
    prediction: int
    label: str
    probability: float
    base_value: float
    top_contributions: List[FeatureContribution]
    model_version: str
    timestamp: str


class BatchPredictionRequest(BaseModel):
    """Schema for batch prediction request."""
    students: List[StudentData]


class HealthResponse(BaseModel):
    """Schema for health check response."""
    status: str
    model_loaded: bool
    timestamp: str


class ModelInfoResponse(BaseModel):
    """Schema for model metadata."""
    model_type: str
    version: str
    trained_at: str
    metrics: Dict[str, Any]
