import explanation_layer as el


def test_medication_name_does_not_decide_category():
    # medication_name leads by impact, but copay is the real actionable barrier
    factors = [
        {"factor": "medication_name", "value": "Dermalis", "impact": "high"},
        {"factor": "monthly_copay", "value": 300.72, "impact": "medium"},
    ]
    assert el.dominant_category(factors) == "cost"


def test_medication_name_alone_still_general():
    factors = [{"factor": "medication_name", "value": "Dermalis", "impact": "high"}]
    assert el.dominant_category(factors) == "general"


def test_general_explanation_lists_all_drivers():
    # medication_name is non-actionable, so it drops out once a real factor exists
    factors = [
        {"factor": "medication_name", "value": "Dermalis", "impact": "high"},
        {"factor": "some_unmapped_thing", "value": 7, "impact": "medium"},
    ]
    result = el.generate_explanation(80, factors)
    assert "Dermalis" not in result["explanation"]
    assert "7" in result["explanation"]
    assert "general barrier" not in result["explanation"]


def test_general_explanation_still_names_score():
    factors = [{"factor": "medication_name", "value": "Dermalis", "impact": "high"}]
    result = el.generate_explanation(80, factors)
    assert "80" in result["explanation"]


def test_specific_category_unchanged():
    # a clear cost case stays specific and grounded, not a generic list
    factors = [{"factor": "monthly_copay", "value": 300.72, "impact": "high"}]
    result = el.generate_explanation(90, factors)
    assert result["category"] == "cost"
    assert "300.72" in result["explanation"]
    assert "copay" in result["explanation"].lower()


def test_real_barrier_wins_and_explanation_is_specific():
    factors = [
        {"factor": "medication_name", "value": "Cutinex", "impact": "high"},
        {"factor": "prior_auth_required", "value": True, "impact": "medium"},
    ]
    result = el.generate_explanation(85, factors)
    assert result["category"] == "access"
    assert "prior auth" in result["explanation"].lower()


def test_empty_factors_still_safe():
    result = el.generate_explanation(50, [])
    assert result["category"] == "general"
    assert "50" in result["explanation"]
