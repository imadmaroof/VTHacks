import pytest
from model_adapter import adapt_record, bucket_impact


# --- Real-shaped records matching patients_scored.json ---

LOW_RISK_RECORD = {
    "patient_id": "SYN-00001",
    "age": 24,
    "condition": "Major Depressive Disorder",
    "medication_name": "Serenol",
    "risk_score": 4,
    "risk_tier": "low",
    "reason": None,
    "top_factors": [
        {"factor": "monthly_copay", "impact": -1.6668},
        {"factor": "medication_name", "impact": -1.563},
        {"factor": "deductible_met", "impact": 0.4674},
    ],
}

HIGH_RISK_RECORD = {
    "patient_id": "SYN-09999",
    "age": 61,
    "condition": "Psoriasis",
    "medication_name": "Cutinex",
    "risk_score": 84,
    "risk_tier": "high",
    "reason": "cost",
    "top_factors": [
        {"factor": "monthly_copay", "impact": 3.9},
        {"factor": "prior_auth_required", "impact": 2.1},
        {"factor": "deductible_met", "impact": -0.4},
    ],
}

INSUFFICIENT_DATA_RECORD = {
    "patient_id": "SYN-00050",
    "risk_score": None,
    "risk_tier": "insufficient_data",
    "reason": "Missing required field(s): monthly_copay",
    "top_factors": [],
    "age": 40,
    "condition": "Asthma",
    "medication_name": None,
}


def test_bucket_impact_high_medium_low():
    # buckets are by absolute magnitude
    assert bucket_impact(3.9) == "high"
    assert bucket_impact(-1.6) == "medium"
    assert bucket_impact(0.4) == "low"


def test_adapt_keeps_only_positive_impact_factors_as_risk_drivers():
    # a factor pushing risk DOWN must not be presented as a driver
    adapted = adapt_record(HIGH_RISK_RECORD)
    driver_names = [f["factor"] for f in adapted["top_factors"]]
    assert "monthly_copay" in driver_names
    assert "prior_auth_required" in driver_names
    assert "deductible_met" not in driver_names  # negative impact = protective


def test_adapt_attaches_string_impact_buckets():
    adapted = adapt_record(HIGH_RISK_RECORD)
    for f in adapted["top_factors"]:
        assert f["impact"] in {"high", "medium", "low"}


def test_adapt_pulls_value_from_parent_when_available():
    record = dict(HIGH_RISK_RECORD)
    record["monthly_copay"] = 265
    adapted = adapt_record(record)
    copay = next(f for f in adapted["top_factors"] if f["factor"] == "monthly_copay")
    assert copay["value"] == 265


def test_adapt_missing_value_defaults_to_none_not_crash():
    adapted = adapt_record(HIGH_RISK_RECORD)  # no monthly_copay value at top level
    copay = next(f for f in adapted["top_factors"] if f["factor"] == "monthly_copay")
    assert copay["value"] is None


def test_insufficient_data_flagged_not_scored():
    adapted = adapt_record(INSUFFICIENT_DATA_RECORD)
    assert adapted["scorable"] is False
    assert adapted["reason"]


def test_scorable_record_marked_scorable():
    adapted = adapt_record(HIGH_RISK_RECORD)
    assert adapted["scorable"] is True


def test_all_negative_factors_yields_no_drivers():
    record = dict(LOW_RISK_RECORD)  # all top factors negative except deductible_met (0.4674)
    adapted = adapt_record(record)
    # only the one positive factor should survive
    driver_names = [f["factor"] for f in adapted["top_factors"]]
    assert driver_names == ["deductible_met"]
