import explanation_layer as el
from model_adapter import adapt_record, bucket_impact


# ---------- risk tier boundaries ----------

def _record(score, factors=None):
    return {
        "patient_id": "SYN-TEST",
        "monthly_copay": 200.0,
        "risk_score": score,
        "risk_tier": "high",
        "top_factors": factors
        if factors is not None
        else [{"factor": "monthly_copay", "impact": 3.0}],
    }


def test_score_exactly_at_low_cutoff_is_explained():
    # 40 is the medium cutoff in scoring.py -- must NOT be treated as low
    result = el.explain_scored_record(_record(40))
    assert result["category"] != "low_risk"


def test_score_just_below_cutoff_is_low_risk():
    result = el.explain_scored_record(_record(39))
    assert result["category"] == "low_risk"


def test_score_zero_is_low_risk():
    result = el.explain_scored_record(_record(0))
    assert result["category"] == "low_risk"


def test_score_100_explained():
    result = el.explain_scored_record(_record(100))
    assert result["category"] == "cost"
    assert "100" in result["explanation"]


# ---------- malformed factor input ----------

def test_factor_missing_impact_key_does_not_crash():
    factors = [{"factor": "monthly_copay", "value": 300}]
    result = el.generate_explanation(80, factors)
    assert result["category"] == "cost"


def test_factor_missing_value_key_does_not_crash():
    factors = [{"factor": "monthly_copay", "impact": "high"}]
    result = el.generate_explanation(80, factors)
    assert result["explanation"]


def test_none_value_does_not_leak_into_explanation():
    factors = [{"factor": "monthly_copay", "value": None, "impact": "high"}]
    result = el.generate_explanation(80, factors)
    assert "None" not in result["explanation"]


def test_unknown_impact_string_treated_as_low():
    factors = [{"factor": "monthly_copay", "value": 300, "impact": "catastrophic"}]
    result = el.generate_explanation(80, factors)
    assert result["category"] == "cost"


# ---------- adapter boundaries ----------

def test_bucket_impact_exactly_at_cutoffs():
    assert bucket_impact(2.0) == "high"
    assert bucket_impact(1.999) == "medium"
    assert bucket_impact(1.0) == "medium"
    assert bucket_impact(0.999) == "low"


def test_zero_impact_factor_is_dropped():
    record = {
        "patient_id": "X",
        "risk_score": 80,
        "risk_tier": "high",
        "top_factors": [
            {"factor": "monthly_copay", "impact": 0.0},
            {"factor": "prior_auth_required", "impact": 1.5},
        ],
    }
    drivers = [f["factor"] for f in adapt_record(record)["top_factors"]]
    assert "monthly_copay" not in drivers
    assert "prior_auth_required" in drivers


def test_all_factors_protective_yields_no_drivers():
    record = {
        "patient_id": "X",
        "risk_score": 80,
        "risk_tier": "high",
        "top_factors": [
            {"factor": "monthly_copay", "impact": -3.0},
            {"factor": "deductible_met", "impact": -1.0},
        ],
    }
    adapted = adapt_record(record)
    assert adapted["top_factors"] == []
    # and the layer above must still produce something usable
    result = el.explain_scored_record(record)
    assert result["explanation"]
    assert result["suggestion"]


def test_drivers_sorted_by_true_magnitude_not_bucket():
    record = {
        "patient_id": "X",
        "risk_score": 80,
        "risk_tier": "high",
        "top_factors": [
            {"factor": "prior_auth_required", "impact": 2.1},
            {"factor": "monthly_copay", "impact": 3.9},
        ],
    }
    drivers = adapt_record(record)["top_factors"]
    assert drivers[0]["factor"] == "monthly_copay"


# ---------- output guarantees ----------

def test_every_category_has_a_suggestion():
    for category in el.SUGGESTIONS:
        assert el.SUGGESTIONS[category].strip()


def test_talking_points_capped_at_three(monkeypatch):
    monkeypatch.setattr(
        el,
        "_gemini_output",
        lambda *a, **k: {"explanation": "x", "talking_points": ["a", "b", "c", "d", "e"]},
    )
    result = el.generate_explanation(
        80, [{"factor": "monthly_copay", "value": 300, "impact": "high"}], use_llm=True
    )
    assert len(result["talking_points"]) <= 3


def test_explanation_never_empty_across_categories():
    samples = [
        [{"factor": "monthly_copay", "value": 300, "impact": "high"}],
        [{"factor": "prior_auth_required", "value": True, "impact": "high"}],
        [{"factor": "side_effects_reported", "value": True, "impact": "high"}],
        [{"factor": "missed_visits_count", "value": 3, "impact": "high"}],
        [{"factor": "medication_name", "value": "Dermalis", "impact": "high"}],
        [],
    ]
    for factors in samples:
        result = el.generate_explanation(75, factors)
        assert result["explanation"].strip()
        assert result["suggestion"].strip()


def test_insufficient_data_never_shows_a_score():
    record = {
        "patient_id": "X",
        "risk_score": None,
        "risk_tier": "insufficient_data",
        "reason": "Missing required field(s): monthly_copay",
        "top_factors": [],
    }
    result = el.explain_scored_record(record)
    assert result["risk_score"] is None
    assert "0%" not in result["explanation"]
