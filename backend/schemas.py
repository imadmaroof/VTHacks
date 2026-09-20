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


class SinglePatientRequest(BaseModel):
    """
    One hypothetical patient from the "customizable patient" demo form.

    Every field is Optional so a partially-filled form (or a field the UI
    intentionally left blank) still reaches the guard rather than failing
    Pydantic validation -- missing_required_fields() in scoring.py is what
    decides whether monthly_copay/medication_name being absent blocks the
    score, not this schema. Field names and types mirror patient_options.py
    (categorical/boolean fields as str/bool, everything else numeric) --
    that module is the single source of truth for which values are valid;
    this schema only shapes the request.
    """

    patient_id: Optional[str] = None
    age: Optional[float] = None
    condition: Optional[str] = None
    drug_class: Optional[str] = None
    medication_name: Optional[str] = None
    new_rx_or_refill: Optional[str] = None
    insurance_type: Optional[str] = None
    monthly_copay: Optional[float] = None
    deductible_met: Optional[bool] = None
    copay_assistance_available: Optional[bool] = None
    copay_assistance_enrolled: Optional[bool] = None
    prior_auth_required: Optional[bool] = None
    pa_turnaround_days: Optional[float] = None
    days_supply: Optional[float] = None
    distance_to_pharmacy_miles: Optional[float] = None
    delivery_option_available: Optional[bool] = None
    concurrent_medication_count: Optional[float] = None
    appointment_length_minutes: Optional[float] = None
    day_of_week_prescribed: Optional[str] = None
    prior_abandoned_rx_count: Optional[float] = None
    scheduled_visits_count: Optional[float] = None
    missed_visits_count: Optional[float] = None


class SinglePatientResponse(BaseModel):
    status: str
    error: Optional[str] = None
    patient: Optional[PatientResult] = None


class FieldOptionsResponse(BaseModel):
    """
    Everything the customizable-patient form needs to render itself: exact
    dropdown values (from the trained model's category vocabulary),
    condition-filtered medication lists, boolean fields, discrete-numeric
    dropdowns, free-numeric slider ranges, and a pre-filled default patient.
    See patient_options.py for what decides a field's category here.
    """

    categorical: dict[str, list[str]]
    condition_to_drug_class: dict[str, str]
    condition_to_medications: dict[str, list[str]]
    boolean_fields: list[str]
    discrete_numeric_options: dict[str, list[int]]
    numeric_ranges: dict[str, list[float]]
    default_patient: dict[str, Any]
