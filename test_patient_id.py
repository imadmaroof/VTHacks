import explanation_layer as el


HIGH_RISK = {
    "patient_id": "SYN-00007",
    "monthly_copay": 300.72,
    "medication_name": "Dermalis",
    "risk_score": 98,
    "risk_tier": "high",
    "reason": "cost",
    "top_factors": [
        {"factor": "monthly_copay", "impact": 3.9},
        {"factor": "prior_auth_required", "impact": 2.1},
    ],
}

LOW_RISK = {
    "patient_id": "SYN-00001",
    "risk_score": 4,
    "risk_tier": "low",
    "top_factors": [],
}

INSUFFICIENT = {
    "patient_id": "SYN-00050",
    "risk_score": None,
    "risk_tier": "insufficient_data",
    "reason": "Missing required field(s): monthly_copay",
    "top_factors": [],
}


def test_patient_id_present_high_risk():
    assert el.explain_scored_record(HIGH_RISK)["patient_id"] == "SYN-00007"


def test_patient_id_present_low_risk():
    assert el.explain_scored_record(LOW_RISK)["patient_id"] == "SYN-00001"


def test_patient_id_present_insufficient_data():
    assert el.explain_scored_record(INSUFFICIENT)["patient_id"] == "SYN-00050"


def test_missing_patient_id_is_none_not_crash():
    record = {k: v for k, v in HIGH_RISK.items() if k != "patient_id"}
    assert el.explain_scored_record(record)["patient_id"] is None


def test_full_output_shape():
    result = el.explain_scored_record(HIGH_RISK)
    assert set(result.keys()) == {
        "patient_id",
        "risk_score",
        "category",
        "explanation",
        "suggestion",
        "talking_points",
    }


def test_generate_explanation_unchanged_no_patient_id():
    # the lower-level function has no record, so it must NOT invent the key
    result = el.generate_explanation(90, [{"factor": "monthly_copay", "value": 300, "impact": "high"}])
    assert "patient_id" not in result
