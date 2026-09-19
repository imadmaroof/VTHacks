"""
Scoring guards and filters for the medication adherence risk model.

This module is deliberately standalone and model-agnostic so that BOTH the
batch export in train_adherence_model.py and any future FastAPI endpoint
enforce identical rules. Import it; do not re-implement the checks.

WHY THE GUARD EXISTS
--------------------
evaluate_model.py swept every feature, blanking it across all 400 held-out
test patients, and found the model does NOT degrade gracefully for its two
most important features:

    feature            mean |score shift|    max    risk tier flipped
    monthly_copay              62.6 pts       92          83.2%
    medication_name            29.9 pts       70          60.2%
    ... every other feature     <10 pts        -          <22%

(Figures from the 150-tree model; regenerate with evaluate_model.py after any
retrain and update REQUIRED_FIELDS if a different feature becomes fragile.)

Critically, a missing value produces NO error -- XGBoost routes NaN down a
learned default branch and returns a confident, plausible-looking, wrong
number. Silent wrongness is worse than a crash in a live demo, because
nothing signals that the output should not be trusted.

So: if either required field is missing we refuse to score at all, rather
than emitting a number we know is unreliable. Every other field still goes
through the model as-is -- their mean shift is under 10 points, which is
well within acceptable degradation, and XGBoost's native missing handling
covers them.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

# Fields whose absence invalidates the score entirely. Derived from the
# missing-data sweep in test_results.json -> missing_data_sweep.fragile_features
# (threshold: mean absolute score shift >= 25 points).
REQUIRED_FIELDS: tuple[str, ...] = ("monthly_copay", "medication_name")

INSUFFICIENT_DATA_TIER = "insufficient_data"

# Risk tier cutoffs -- kept here so the API, the batch export and the
# dashboard filter can never drift apart.
TIER_HIGH = 70
TIER_MEDIUM = 40


def is_missing(value: Any) -> bool:
    """
    True if `value` is genuinely absent.

    Covers more than `is None` on purpose. A demo UI that "clears" a field
    almost always submits an empty string, and a JSON payload may carry the
    string "null" or "NaN" rather than a real null. Treating those as present
    would defeat the guard precisely in the scenario it exists for -- someone
    manually blanking a field in front of judges.
    """
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    # pandas NaT / NA / numpy nan, without importing pandas at module scope.
    try:
        import pandas as pd

        if pd.isna(value):
            return True
    except (ImportError, TypeError, ValueError):
        # pd.isna raises on array-likes; a non-scalar is "present" by default.
        pass
    if isinstance(value, str) and value.strip().lower() in {
        "", "null", "none", "nan", "n/a", "na",
    }:
        return True
    return False


def missing_required_fields(patient: Mapping[str, Any]) -> list[str]:
    """
    Return the names of required fields that are absent from `patient`.

    An empty list means the patient is safe to score. A key that is entirely
    absent from the mapping counts as missing, same as a null value.
    """
    return [
        field
        for field in REQUIRED_FIELDS
        if field not in patient or is_missing(patient[field])
    ]


def guard_reason(missing: list[str]) -> str:
    """Human-readable explanation naming the field(s) that were actually missing."""
    return f"Missing required field(s): {', '.join(missing)}"


def insufficient_data_record(
    patient: Mapping[str, Any], missing: list[str]
) -> dict[str, Any]:
    """
    Build the refusal record for a patient that cannot be scored.

    Identity fields are carried through (when present) so a dashboard can still
    render the row and show WHY it has no score. `risk_score` is null rather
    than 0 -- zero would read as "very low risk", the opposite of the truth,
    which is that we do not know. `top_factors` is empty because SHAP
    attributions derived from an unreliable score are themselves unreliable.
    """
    record: dict[str, Any] = {
        "patient_id": str(patient.get("patient_id", "")) or None,
        "risk_score": None,
        "risk_tier": INSUFFICIENT_DATA_TIER,
        "reason": guard_reason(missing),
        "top_factors": [],
    }
    # Carry identity context through only when it is actually available; a
    # missing medication_name is itself a possible trigger for this record.
    for field in ("age", "condition", "medication_name"):
        value = patient.get(field)
        record[field] = None if is_missing(value) else value
    if record.get("age") is not None:
        record["age"] = int(record["age"])
    return record


def tier_for(score: int) -> str:
    """Map a 0-100 risk score to its tier label."""
    if score >= TIER_HIGH:
        return "high"
    if score >= TIER_MEDIUM:
        return "medium"
    return "low"


def filter_high_risk(
    records: Iterable[Mapping[str, Any]], min_risk: int = TIER_HIGH
) -> list[dict[str, Any]]:
    """
    Return only patients scoring >= `min_risk`, sorted by risk_score descending.

    WHAT THIS FILTER ACTUALLY CATCHES -- measured on the held-out test set
    (see test_results.json -> tier_performance):

        Precision @70 = 79.1%  ->  roughly 4 in 5 patients flagged at this
                                   cutoff genuinely did discontinue.
        Recall    @70 = 63.1%  ->  BUT this cutoff only catches about 63% of
                                   everyone who actually discontinued. The
                                   remaining ~37% score below 70 and will not
                                   appear in this list at all.

    Do not describe this list as "the patients who will discontinue". It is
    "the patients we are most confident about", and it misses a substantial
    minority by design. Lowering min_risk to 40 raises recall to ~83% but
    drops precision to ~60% -- a deliberate tradeoff, not a free improvement.

    Patients with risk_tier == "insufficient_data" are excluded: their score is
    null, so they are neither above nor below any threshold. They still appear
    in the full patients_scored.json and should be surfaced separately as
    needing data completion, not silently dropped from the workflow.
    """
    scored = [
        r for r in records
        if r.get("risk_score") is not None
        and r.get("risk_tier") != INSUFFICIENT_DATA_TIER
    ]
    hits = [r for r in scored if r["risk_score"] >= min_risk]
    return sorted(hits, key=lambda r: r["risk_score"], reverse=True)
