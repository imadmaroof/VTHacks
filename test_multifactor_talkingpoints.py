import explanation_layer as el


MULTI_FACTOR = [
    {"factor": "monthly_copay", "value": 300.72, "impact": "high"},
    {"factor": "prior_auth_required", "value": True, "impact": "high"},
    {"factor": "distance_to_pharmacy_miles", "value": 22.3, "impact": "medium"},
]


def test_talking_points_present_when_llm_on(monkeypatch):
    monkeypatch.setattr(
        el,
        "_gemini_output",
        lambda *a, **k: {
            "explanation": "Cost and access barriers stack up here.",
            "talking_points": ["Acknowledge the cost.", "Ask about past affordability."],
        },
    )
    result = el.generate_explanation(90, MULTI_FACTOR, use_llm=True)
    assert result["explanation"] == "Cost and access barriers stack up here."
    assert len(result["talking_points"]) == 2


def test_talking_points_empty_when_llm_off():
    result = el.generate_explanation(90, MULTI_FACTOR, use_llm=False)
    assert result["talking_points"] == []
    # template still produces a grounded explanation
    assert "300.72" in result["explanation"]


def test_llm_failure_falls_back_with_empty_talking_points(monkeypatch):
    monkeypatch.setattr(el, "_gemini_output", lambda *a, **k: None)
    result = el.generate_explanation(90, MULTI_FACTOR, use_llm=True)
    assert result["talking_points"] == []
    assert "300.72" in result["explanation"]
    assert result["category"] == "cost"


def test_malformed_llm_response_does_not_crash(monkeypatch):
    # missing keys in the returned dict must not blow up
    monkeypatch.setattr(el, "_gemini_output", lambda *a, **k: {"explanation": "only this"})
    result = el.generate_explanation(90, MULTI_FACTOR, use_llm=True)
    assert result["explanation"] == "only this"
    assert result["talking_points"] == []


def test_suggestion_stays_rule_based_even_with_llm(monkeypatch):
    # the intervention choice must never come from the LLM
    monkeypatch.setattr(
        el,
        "_gemini_output",
        lambda *a, **k: {"explanation": "x", "talking_points": ["y"]},
    )
    result = el.generate_explanation(90, MULTI_FACTOR, use_llm=True)
    assert "copay assistance" in result["suggestion"].lower()


def test_output_shape_consistent_both_modes(monkeypatch):
    monkeypatch.setattr(
        el, "_gemini_output", lambda *a, **k: {"explanation": "x", "talking_points": []}
    )
    a = el.generate_explanation(90, MULTI_FACTOR, use_llm=False)
    b = el.generate_explanation(90, MULTI_FACTOR, use_llm=True)
    assert set(a.keys()) == set(b.keys())


def test_low_risk_and_insufficient_records_have_talking_points_key():
    low = {"patient_id": "X", "risk_score": 4, "risk_tier": "low", "top_factors": []}
    insuf = {
        "patient_id": "Y",
        "risk_score": None,
        "risk_tier": "insufficient_data",
        "reason": "Missing required field(s): monthly_copay",
        "top_factors": [],
    }
    for rec in (low, insuf):
        out = el.explain_scored_record(rec)
        assert "talking_points" in out
