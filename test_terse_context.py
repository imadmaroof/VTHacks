import explanation_layer as el


COST_FACTORS = [{"factor": "monthly_copay", "value": 300.72, "impact": "high"}]
ACCESS_FACTORS = [
    {"factor": "prior_auth_required", "value": True, "impact": "high"},
    {"factor": "pa_turnaround_days", "value": 14, "impact": "medium"},
]


def test_cost_explanation_mentions_insurance():
    ctx = {"insurance_type": "Medicare", "copay_assistance_available": False}
    result = el.generate_explanation(98, COST_FACTORS, context=ctx)
    assert "Medicare" in result["explanation"]


def test_cost_explanation_flags_assistance_available_not_enrolled():
    ctx = {
        "insurance_type": "Commercial",
        "copay_assistance_available": True,
        "copay_assistance_enrolled": False,
    }
    result = el.generate_explanation(98, COST_FACTORS, context=ctx)
    text = result["explanation"].lower()
    assert "assistance" in text
    assert "unused" in text


def test_cost_explanation_handles_string_booleans_from_csv():
    # CSV join yields strings, not real bools
    ctx = {
        "insurance_type": "Uninsured",
        "copay_assistance_available": "TRUE",
        "copay_assistance_enrolled": "FALSE",
    }
    result = el.generate_explanation(98, COST_FACTORS, context=ctx)
    assert "unused" in result["explanation"].lower()


def test_no_assistance_available_says_so():
    ctx = {
        "insurance_type": "Uninsured",
        "copay_assistance_available": False,
        "copay_assistance_enrolled": False,
    }
    result = el.generate_explanation(98, COST_FACTORS, context=ctx)
    assert "no assistance" in result["explanation"].lower()


def test_explanations_are_short():
    ctx = {"insurance_type": "Medicare", "copay_assistance_available": False}
    result = el.generate_explanation(98, COST_FACTORS, context=ctx)
    assert len(result["explanation"].split()) <= 20


def test_access_explanation_is_short_and_specific():
    result = el.generate_explanation(85, ACCESS_FACTORS)
    text = result["explanation"]
    assert len(text.split()) <= 20
    assert "prior auth" in text.lower()


def test_no_context_still_works():
    result = el.generate_explanation(98, COST_FACTORS)
    assert "300.72" in result["explanation"]
    assert result["category"] == "cost"


def test_score_still_present():
    result = el.generate_explanation(98, COST_FACTORS)
    assert "98" in result["explanation"]


def test_context_flows_through_explain_scored_record():
    record = {
        "patient_id": "SYN-00007",
        "monthly_copay": 300.72,
        "insurance_type": "Medicare",
        "copay_assistance_available": True,
        "copay_assistance_enrolled": False,
        "risk_score": 98,
        "risk_tier": "high",
        "top_factors": [{"factor": "monthly_copay", "impact": 3.9}],
    }
    result = el.explain_scored_record(record)
    assert "Medicare" in result["explanation"]
