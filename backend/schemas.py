"""Pydantic response models for the FastAPI layer.

These describe the shape callers get back; they don't change how the model
scores anything. Field names mostly follow the existing contract used by
patients_scored.json / score_upload.py (risk_tier, top_factors: [{factor,
impact}]) rather than inventing new ones, so nothing that already consumes
that shape has to change.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str


class TopFactor(BaseModel):
    """Raw SHAP-derived factor, exactly as the model/scoring code produces it
    (all factors, signed float impact -- positive raises risk, negative lowers it)."""

    factor: str
    impact: float


class RiskDriver(BaseModel):
    """A factor after model_adapter.adapt_record: risk-increasing only, with
    the patient's actual value and a bucketed high/medium/low impact. This is
    what explanation_layer used to build the narrative below."""

    factor: str
    value: Optional[Any] = None
    impact: str


class PatientResult(BaseModel):
    patient_id: Optional[str] = None
    age: Optional[int] = None
    condition: Optional[str] = None
    medication_name: Optional[str] = None
    risk_score: Optional[int] = None
    risk_tier: str
    reason: Optional[str] = None
    top_factors: list[TopFactor] = []
    risk_drivers: list[RiskDriver] = []
    explanation_category: str
    explanation: str
    recommendations: list[str] = []
    talking_points: list[str] = []


class PredictResponse(BaseModel):
    status: str
    total_patients: int
    scored_count: int
    insufficient_data_count: int
    high_risk_count: int
    results: list[PatientResult]
    high_risk_patients: list[PatientResult]


class ErrorResponse(BaseModel):
    status: str = "error"
    detail: str
