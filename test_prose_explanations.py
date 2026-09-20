import explanation_layer as el

MIXED = [
    {"factor": "monthly_copay", "value": 300.72, "impact": "high"},
    {"factor": "prior_auth_required", "value": True, "impact": "high"},
    {"factor": "distance_to_pharmacy_miles", "value": 22.3, "impact": "medium"},
]
FINANCIAL_ONLY = [
    {"factor": "monthly_copay", "value": 420.0, "impact": "high"},
    {"factor": "deductible_met", "value": False, "impact": "medium"},
]
ACCESS_ONLY = [
    {"factor": "prior_auth_required", "value": True, "impact": "high"},
    {"factor": "pa_turnaround_days", "value": 14, "impact": "medium"},
    {"factor": "missed_visits_count", "value": 3, "impact": "medium"},
]


def test_no_label_prefixes():
    text = el.generate_explanation(92, MIXED)["explanation"]
    assert "Financial:" not in text
    assert "Other:" not in text


def test_reads_as_a_sentence():
    text = el.generate_explanation(92, MIXED)["explanation"]
    assert text.endswith(".")
    # a connective word, not a bare comma-separated dump
    assert any(w in text.lower() for w in (" and ", " plus ", " driven by "))


def test_covers_both_financial_and_non_financial():
    text = el.generate_explanation(92, MIXED)["explanation"]
    assert "300.72" in text
    assert "prior auth" in text.lower()
    assert "22.3" in text


def test_financial_only_still_prose():
    text = el.generate_explanation(88, FINANCIAL_ONLY)["explanation"]
    assert "Financial:" not in text
    assert "420.00" in text
    assert "deductible" in text.lower()


def test_non_financial_only_still_prose():
    text = el.generate_explanation(76, ACCESS_ONLY)["explanation"]
    assert "Other:" not in text
    assert "prior auth" in text.lower()
    assert "missed visits" in text.lower()


def test_oxford_list_for_three_items():
    text = el.generate_explanation(76, ACCESS_ONLY)["explanation"]
    assert ", and " in text


def test_two_items_joined_with_and_not_comma():
    text = el.generate_explanation(88, FINANCIAL_ONLY)["explanation"]
    assert " and " in text


def test_starts_with_score():
    text = el.generate_explanation(92, MIXED)["explanation"]
    assert text.startswith("92% risk")


def test_still_concise():
    ctx = {"insurance_type": "Medicare", "copay_assistance_available": False}
    text = el.generate_explanation(92, MIXED, context=ctx)["explanation"]
    assert len(text.split()) <= 32


def test_insurance_context_woven_in():
    ctx = {
        "insurance_type": "Medicare",
        "copay_assistance_available": True,
        "copay_assistance_enrolled": False,
    }
    text = el.generate_explanation(98, [MIXED[0]], context=ctx)["explanation"]
    assert "Medicare" in text
    assert "assistance" in text.lower()


def test_single_factor_reads_naturally():
    text = el.generate_explanation(80, [MIXED[0]])["explanation"]
    assert "Financial:" not in text
    assert "300.72" in text


def test_empty_factors_safe():
    assert el.generate_explanation(50, [])["explanation"].strip()


def test_gemini_prompt_asks_for_a_sentence():
    prompt = el._build_prompt(92, "cost", MIXED, None).lower()
    assert "sentence" in prompt
    assert "financial:" not in prompt
