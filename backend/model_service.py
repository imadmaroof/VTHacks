"""
Backend-facing wrapper around the repo's EXISTING scoring + explanation code.

This module deliberately contains no model logic of its own. It imports
directly from the repo root:

    score_upload.py       - loads model.pkl, encodes, predicts, runs SHAP
                             (already used by the CLI and the GitHub Action)
    model_adapter.py       - turns a scored record into risk-driver factors
    explanation_layer.py   - turns risk-driver factors into a doctor-facing
                             explanation + suggestion (+ optional Gemini
                             talking points, gracefully skipped without a key)

so the FastAPI layer can never drift from the model/SHAP/explanation logic
used elsewhere in the repo. Nothing here retrains or modifies model.pkl.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# The repo keeps its model/scoring code at the repo root, not inside
# backend/. Add it to sys.path so it can be imported without copying or
# duplicating any of it.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from score_upload import preload_model, score_patient_upload, score_single_patient  # noqa: E402
from model_adapter import adapt_record  # noqa: E402
from explanation_layer import explain_scored_record  # noqa: E402
from patient_options import build_field_options  # noqa: E402

# explain_scored_record's use_llm flag is safe to leave on: explanation_layer
# checks for GEMINI_API_KEY itself and silently falls back to the rule-based
# template when it's unset or the call fails. Set USE_LLM_EXPLANATIONS=false
# to force the template path even when a key is present.
USE_LLM_EXPLANATIONS = os.environ.get("USE_LLM_EXPLANATIONS", "true").strip().lower() != "false"


def warm_up() -> None:
    """Load model.pkl into score_upload's cache before serving any requests.
    Call once from the FastAPI startup hook."""
    preload_model()


def _merge_raw_values(record: dict[str, Any], raw_row: dict[str, Any]) -> dict[str, Any]:
    """Copy the uploaded row's original feature values onto the scored record
    without overwriting anything the model already set. model_adapter reads
    the patient's actual value for each contributing factor (e.g. the real
    monthly_copay), which score_upload's output alone doesn't carry -- this
    mirrors the merge run_explanations.py does against patients_scored.json."""
    merged = dict(record)
    for key, value in raw_row.items():
        merged.setdefault(key, value)
    return merged


def _enrich(record: dict[str, Any], raw_row: dict[str, Any]) -> dict[str, Any]:
    merged = _merge_raw_values(record, raw_row)
    adapted = adapt_record(merged)
    explanation = explain_scored_record(merged, use_llm=USE_LLM_EXPLANATIONS)
    return {
        **record,
        "risk_drivers": adapted["top_factors"],
        "explanation_category": explanation["category"],
        "explanation": explanation["explanation"],
        "recommendations": [explanation["suggestion"]] if explanation.get("suggestion") else [],
        "talking_points": explanation.get("talking_points", []),
    }


def predict_patient_data(csv_path: str) -> dict[str, Any]:
    """Score every patient in the CSV and attach a doctor-facing explanation
    and recommendation to each. Delegates all scoring to score_patient_upload
    -- this function only enriches its output."""
    result = score_patient_upload(csv_path)
    if result["status"] == "error":
        return result

    raw_rows = pd.read_csv(csv_path).to_dict(orient="records")
    enriched = [
        _enrich(record, raw_rows[i] if i < len(raw_rows) else {})
        for i, record in enumerate(result["all_patients"])
    ]

    result["all_patients"] = enriched
    enriched_by_id = {r.get("patient_id"): r for r in enriched}
    result["high_risk_patients"] = [
        enriched_by_id.get(r.get("patient_id"), r) for r in result["high_risk_patients"]
    ]
    return result


def predict_single_patient(patient: dict[str, Any]) -> dict[str, Any]:
    """Score one hypothetical patient (from the customizable-patient demo
    form) and attach the same doctor-facing explanation/recommendation the
    batch path gets. Delegates all scoring to score_single_patient -- this
    function only enriches its output, exactly like predict_patient_data does
    for the batch/upload path."""
    result = score_single_patient(patient)
    if result["status"] == "error":
        return result
    return {
        "status": "ok",
        "error": None,
        "patient": _enrich(result["patient"], patient),
    }


def get_patient_field_options() -> dict[str, Any]:
    """Form spec for the customizable-patient demo: dropdown values, ranges,
    and a default patient. See patient_options.py for the policy itself."""
    return build_field_options()
