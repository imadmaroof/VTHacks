import pytest
from explanation_layer import generate_explanation, dominant_category


# ---- Fake inputs (Person 2's own examples, no dependency on the real model) ----

COST_EXAMPLE = {
    "risk_score": 78,
    "top_factors": [
        {"factor": "monthly_copay", "value": 145, "impact": "high"},
        {"factor": "copay_assistance_enrolled", "value": False, "impact": "medium"},
        {"factor": "prior_auth_required", "value": False, "impact": "low"},
    ],
}

ACCESS_EXAMPLE = {
    "risk_score": 64,
    "top_factors": [
        {"factor": "prior_auth_required", "value": True, "impact": "high"},
        {"factor": "pa_turnaround_days", "value": 14, "impact": "high"},
        {"factor": "distance_to_pharmacy_miles", "value": 22.3, "impact": "medium"},
    ],
}

SIDE_EFFECT_EXAMPLE = {
    "risk_score": 55,
    "top_factors": [
        {"factor": "side_effects_reported", "value": True, "impact": "high"},
        {"factor": "concurrent_medication_count", "value": 6, "impact": "medium"},
    ],
}

PERCEIVED_INEFFECTIVE_EXAMPLE = {
    "risk_score": 48,
    "top_factors": [
        {"factor": "missed_visits_count", "value": 2, "impact": "high"},
        {"factor": "new_rx_or_refill", "value": "New", "impact": "medium"},
    ],
}

TIE_EXAMPLE = {
    # cost and access factors both "medium" -> first-listed factor's category should win
    "risk_score": 60,
    "top_factors": [
        {"factor": "monthly_copay", "value": 90, "impact": "medium"},
        {"factor": "prior_auth_required", "value": True, "impact": "medium"},
    ],
}

UNKNOWN_FACTOR_EXAMPLE = {
    "risk_score": 40,
    "top_factors": [
        {"factor": "some_new_feature_nobody_mapped_yet", "value": 3, "impact": "high"},
    ],
}


def test_cost_dominant_category():
    assert dominant_category(COST_EXAMPLE["top_factors"]) == "cost"


def test_access_dominant_category():
    assert dominant_category(ACCESS_EXAMPLE["top_factors"]) == "access"


def test_side_effect_dominant_category():
    assert dominant_category(SIDE_EFFECT_EXAMPLE["top_factors"]) == "side_effect"


def test_perceived_ineffective_dominant_category():
    assert dominant_category(PERCEIVED_INEFFECTIVE_EXAMPLE["top_factors"]) == "perceived_ineffective"


def test_tie_break_uses_first_listed_factor():
    assert dominant_category(TIE_EXAMPLE["top_factors"]) == "cost"


def test_unknown_factor_falls_back_without_crashing():
    result = generate_explanation(**UNKNOWN_FACTOR_EXAMPLE)
    assert result["category"] == "general"
    assert result["suggestion"]  # still produces something usable


def test_output_shape():
    result = generate_explanation(**COST_EXAMPLE)
    assert set(result.keys()) == {"risk_score", "category", "explanation", "suggestion"}
    assert result["risk_score"] == 78
    assert result["category"] == "cost"


def test_explanation_is_grounded_in_top_factor():
    # the explanation must reference the actual top factor's name/value, not invent one
    result = generate_explanation(**COST_EXAMPLE)
    assert "145" in result["explanation"]
    assert "copay" in result["explanation"].lower()


def test_explanation_never_mentions_unrelated_category_language():
    # a cost case shouldn't talk about side effects or PA turnaround
    result = generate_explanation(**COST_EXAMPLE)
    assert "side effect" not in result["explanation"].lower()
    assert "prior authorization" not in result["suggestion"].lower()


def test_empty_top_factors_does_not_crash():
    result = generate_explanation(risk_score=10, top_factors=[])
    assert result["category"] == "general"
