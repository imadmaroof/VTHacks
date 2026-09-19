"""
Adapter: teammate's model output -> explanation layer input.

The model (train_adherence_model.py / scoring.py) emits records like:

    {
        "patient_id": "SYN-00001",
        "risk_score": 84,               # 0-100, or None for insufficient_data
        "risk_tier": "high",
        "reason": "cost" | None | "Missing required field(s): ...",
        "top_factors": [
            {"factor": "monthly_copay", "impact": 3.9},   # signed SHAP value
            ...
        ],
        "age": 61, "condition": "...", "medication_name": "...",
        # (patient feature values may also be present at top level)
    }

generate_explanation() expects each factor as
    {"factor": str, "value": <any>, "impact": "high"|"medium"|"low"}
and expects only genuine risk DRIVERS, not protective factors.

This adapter bridges the two. It does three things:
  1. drops factors with impact <= 0 (those lower risk; they are not drivers)
  2. buckets the remaining signed SHAP magnitude into high/medium/low
  3. pulls the patient's actual value for each factor from the record when present
It also flags records the model refused to score (insufficient_data).
"""

# Absolute-SHAP cutoffs for bucketing. Tuned to the model's own scale, where
# monthly_copay dominates at ~1.5-4 and everything else sits lower. Adjust if
# a retrain changes the SHAP magnitude distribution.
HIGH_CUTOFF = 2.0
MEDIUM_CUTOFF = 1.0

INSUFFICIENT_DATA_TIER = "insufficient_data"


def bucket_impact(shap_value):
    """Signed SHAP value -> 'high' | 'medium' | 'low' by absolute magnitude."""
    magnitude = abs(shap_value)
    if magnitude >= HIGH_CUTOFF:
        return "high"
    if magnitude >= MEDIUM_CUTOFF:
        return "medium"
    return "low"


def adapt_record(record):
    """Convert one model record into the explanation layer's contract.

    Returns a dict with:
      scorable    : bool  -- False when the model refused (insufficient_data)
      reason      : str | None
      risk_score  : int | None
      top_factors : [{"factor", "value", "impact"}]  -- positive drivers only,
                    highest impact first
    """
    if (
        record.get("risk_tier") == INSUFFICIENT_DATA_TIER
        or record.get("risk_score") is None
    ):
        return {
            "scorable": False,
            "reason": record.get("reason") or "Insufficient data to score.",
            "risk_score": None,
            "top_factors": [],
        }

    drivers = []
    for factor in record.get("top_factors", []):
        impact = factor.get("impact", 0)
        if impact <= 0:
            continue  # protective factor, not a risk driver
        drivers.append(
            {
                "factor": factor["factor"],
                "value": record.get(factor["factor"]),  # None if not on the record
                "impact": bucket_impact(impact),
                "_raw": impact,
            }
        )

    drivers.sort(key=lambda f: f["_raw"], reverse=True)
    for f in drivers:
        del f["_raw"]

    return {
        "scorable": True,
        "reason": record.get("reason"),
        "risk_score": record["risk_score"],
        "top_factors": drivers,
    }
