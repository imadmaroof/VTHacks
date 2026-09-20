import explanation_layer as el


def test_uninsured_not_repeated():
    # "while uninsured, Uninsured" -- the dedupe check was case-sensitive
    factors = [
        {"factor": "monthly_copay", "value": 304.99, "impact": "high"},
        {"factor": "insurance_type", "value": "Uninsured", "impact": "medium"},
    ]
    ctx = {"insurance_type": "Uninsured", "copay_assistance_available": False}
    text = el.generate_explanation(99, factors, context=ctx)["explanation"]
    assert text.lower().count("uninsured") == 1


def test_named_insurance_not_repeated():
    factors = [
        {"factor": "monthly_copay", "value": 300.0, "impact": "high"},
        {"factor": "insurance_type", "value": "Medicare", "impact": "medium"},
    ]
    ctx = {"insurance_type": "Medicare"}
    text = el.generate_explanation(95, factors, context=ctx)["explanation"]
    assert text.lower().count("medicare") == 1


def test_drug_name_not_listed_as_a_driver():
    factors = [
        {"factor": "monthly_copay", "value": 309.15, "impact": "high"},
        {"factor": "medication_name", "value": "Dermalis", "impact": "high"},
    ]
    text = el.generate_explanation(99, factors)["explanation"]
    assert "Dermalis" not in text
    assert "309.15" in text


def test_drug_name_slot_freed_for_real_factor():
    factors = [
        {"factor": "monthly_copay", "value": 300.0, "impact": "high"},
        {"factor": "medication_name", "value": "Cutinex", "impact": "high"},
        {"factor": "prior_auth_required", "value": True, "impact": "medium"},
        {"factor": "missed_visits_count", "value": 3, "impact": "medium"},
    ]
    text = el.generate_explanation(90, factors)["explanation"]
    assert "Cutinex" not in text
    # the slot the drug name would have taken goes to a real barrier
    assert "prior auth" in text.lower()
    assert "missed visits" in text.lower()


def test_drug_name_kept_when_it_is_all_there_is():
    # better to say something than produce an empty explanation
    factors = [{"factor": "medication_name", "value": "Dermalis", "impact": "high"}]
    text = el.generate_explanation(80, factors)["explanation"]
    assert "Dermalis" in text


def test_assistance_wording_is_consistent_from_factor():
    factors = [
        {"factor": "monthly_copay", "value": 300.0, "impact": "high"},
        {"factor": "copay_assistance_enrolled", "value": False, "impact": "medium"},
    ]
    text = el.generate_explanation(95, factors)["explanation"].lower()
    assert "unused copay assistance" in text
    assert "not enrolled" not in text


def test_assistance_wording_is_consistent_from_context():
    factors = [{"factor": "monthly_copay", "value": 300.0, "impact": "high"}]
    ctx = {"copay_assistance_available": True, "copay_assistance_enrolled": False}
    text = el.generate_explanation(95, factors, context=ctx)["explanation"].lower()
    assert "unused copay assistance" in text
    assert "not enrolled" not in text


def test_assistance_not_duplicated_across_factor_and_context():
    factors = [
        {"factor": "monthly_copay", "value": 300.0, "impact": "high"},
        {"factor": "copay_assistance_enrolled", "value": False, "impact": "medium"},
    ]
    ctx = {"copay_assistance_available": True, "copay_assistance_enrolled": False}
    text = el.generate_explanation(95, factors, context=ctx)["explanation"].lower()
    assert text.count("assistance") == 1
