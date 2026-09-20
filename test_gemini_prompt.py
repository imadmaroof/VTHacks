import explanation_layer as el


DRIVERS = [
    {"factor": "monthly_copay", "value": 300.72, "impact": "high"},
    {"factor": "copay_assistance_enrolled", "value": False, "impact": "medium"},
    {"factor": "prior_auth_required", "value": True, "impact": "medium"},
]

COST_CONTEXT = {
    "insurance_type": "Medicare",
    "copay_assistance_available": True,
    "copay_assistance_enrolled": False,
    "medication_name": "Dermalis",
}


def test_prompt_lists_every_driver():
    prompt = el._build_prompt(98, "cost", DRIVERS, COST_CONTEXT)
    assert "monthly copay" in prompt
    assert "300.72" in prompt
    assert "prior authorization" in prompt.lower()


def test_prompt_includes_insurance_for_cost_cases():
    prompt = el._build_prompt(98, "cost", DRIVERS, COST_CONTEXT)
    assert "Medicare" in prompt


def test_prompt_omits_insurance_for_non_cost_cases():
    prompt = el._build_prompt(70, "access", DRIVERS, COST_CONTEXT)
    assert "Medicare" not in prompt


def test_prompt_demands_brevity_and_multiple_factors():
    prompt = el._build_prompt(98, "cost", DRIVERS, COST_CONTEXT)
    low = prompt.lower()
    assert "30 words" in low
    assert "three" in low or "prose" in low


def test_prompt_keeps_grounding_rules():
    prompt = el._build_prompt(98, "cost", DRIVERS, COST_CONTEXT)
    low = prompt.lower()
    assert "only" in low
    assert "do not invent" in low
    assert "clinical" in low or "medical" in low


def test_prompt_handles_no_context():
    prompt = el._build_prompt(98, "cost", DRIVERS, None)
    assert "300.72" in prompt


def test_context_reaches_gemini_path(monkeypatch):
    captured = {}

    def fake(risk_score, category, drivers, context=None):
        captured["context"] = context
        return {"explanation": "x", "talking_points": []}

    monkeypatch.setattr(el, "_gemini_output", fake)
    record = {
        "patient_id": "SYN-1",
        "monthly_copay": 300.72,
        "insurance_type": "Medicare",
        "risk_score": 98,
        "risk_tier": "high",
        "top_factors": [{"factor": "monthly_copay", "impact": 3.9}],
    }
    el.explain_scored_record(record, use_llm=True)
    assert captured["context"]["insurance_type"] == "Medicare"
