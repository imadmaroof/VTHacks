import explanation_layer as el


COST_FACTORS = [
    {"factor": "monthly_copay", "value": 300, "impact": "high"},
    {"factor": "prior_auth_required", "value": True, "impact": "medium"},
]


def test_llm_off_uses_template(monkeypatch):
    # if the flag is off, Gemini is never called
    called = {"hit": False}

    def spy(*a, **k):
        called["hit"] = True
        return "SHOULD NOT APPEAR"

    monkeypatch.setattr(el, "_gemini_explanation", spy)
    result = el.generate_explanation(80, COST_FACTORS, use_llm=False)
    assert called["hit"] is False
    assert "SHOULD NOT APPEAR" not in result["explanation"]


def test_llm_on_uses_gemini_text(monkeypatch):
    monkeypatch.setattr(el, "_gemini_explanation", lambda *a, **k: "Polished LLM explanation.")
    result = el.generate_explanation(80, COST_FACTORS, use_llm=True)
    assert result["explanation"] == "Polished LLM explanation."
    # category and suggestion stay rule-based, not LLM-decided
    assert result["category"] == "cost"
    assert "copay assistance" in result["suggestion"].lower()


def test_llm_failure_falls_back_to_template(monkeypatch):
    # if Gemini returns None (api down, no key, bad response), use the template
    monkeypatch.setattr(el, "_gemini_explanation", lambda *a, **k: None)
    result = el.generate_explanation(80, COST_FACTORS, use_llm=True)
    assert "300" in result["explanation"]        # grounded template still runs
    assert result["category"] == "cost"


def test_output_shape_identical_regardless_of_llm(monkeypatch):
    monkeypatch.setattr(el, "_gemini_explanation", lambda *a, **k: "x")
    a = el.generate_explanation(80, COST_FACTORS, use_llm=False)
    b = el.generate_explanation(80, COST_FACTORS, use_llm=True)
    assert set(a.keys()) == set(b.keys())
