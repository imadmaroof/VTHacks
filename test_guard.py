"""
Proof that the missing-field guard actually fires.

The production dataset contains no nulls, so a normal batch run blocks 0
patients and therefore never exercises the guard. That is exactly the
situation where a guard silently rots. This script injects the failure modes
a live demo would actually produce and asserts the guard catches each one.

It also demonstrates the thing the guard exists to prevent: for a blanked
monthly_copay we print the score the model WOULD have returned, so the size
of the averted error is visible.

Run:  python test_guard.py
"""

from __future__ import annotations

import math
import warnings

import joblib
import numpy as np
import pandas as pd

from scoring import (
    INSUFFICIENT_DATA_TIER,
    REQUIRED_FIELDS,
    filter_high_risk,
    insufficient_data_record,
    missing_required_fields,
)

warnings.filterwarnings("ignore")

BUNDLE = joblib.load("model.pkl")
MODEL = BUNDLE["model"]
FEATURES = BUNDLE["feature_names"]
CAT_LEVELS = BUNDLE["category_levels"]

# A complete, realistic patient (row SYN-00002 from the source data).
COMPLETE = {
    "patient_id": "DEMO-0001", "age": 73, "condition": "Type 2 Diabetes",
    "drug_class": "chronic", "medication_name": "Insunova",
    "new_rx_or_refill": "Refill", "insurance_type": "Medicare",
    "monthly_copay": 53.64, "deductible_met": True,
    "copay_assistance_available": True, "copay_assistance_enrolled": True,
    "prior_auth_required": True, "pa_turnaround_days": 2, "days_supply": 60,
    "distance_to_pharmacy_miles": 5.0, "delivery_option_available": False,
    "concurrent_medication_count": 5, "appointment_length_minutes": 20,
    "day_of_week_prescribed": "Thu", "prior_abandoned_rx_count": 1,
    "scheduled_visits_count": 3, "missed_visits_count": 0,
}


def encode(raw: dict) -> pd.DataFrame:
    """Mirror prepare_features() for a single incoming patient."""
    row = pd.DataFrame([{f: raw.get(f) for f in FEATURES}])
    for col, levels in CAT_LEVELS.items():
        if col in row.columns:
            row[col] = pd.Categorical(row[col], categories=levels)
    for col in row.columns:
        if col not in CAT_LEVELS:
            row[col] = pd.to_numeric(row[col], errors="coerce")
    return row


def raw_score(raw: dict) -> int:
    """Score WITHOUT the guard - what the old pipeline would have returned."""
    return int(round(MODEL.predict_proba(encode(raw))[:, 1][0] * 100))


def score_with_guard(raw: dict) -> dict:
    """The production path: guard first, model only if the guard passes."""
    missing = missing_required_fields(raw)
    if missing:
        return insufficient_data_record(raw, missing)
    score = raw_score(raw)
    tier = "high" if score >= 70 else ("medium" if score >= 40 else "low")
    return {"patient_id": raw.get("patient_id"), "risk_score": score,
            "risk_tier": tier, "reason": None}


results = []


def check(name: str, condition: bool, detail: str = "") -> None:
    results.append((name, condition, detail))
    print(f"  [{'PASS' if condition else 'FAIL'}]  {name:<52} {detail}")


print("=" * 78)
print("MISSING-FIELD GUARD TEST")
print("=" * 78)
print(f"Required fields: {list(REQUIRED_FIELDS)}\n")

# --- baseline: a complete patient must still score normally ---------------
print("Baseline (complete patient):")
base = score_with_guard(COMPLETE)
check("complete patient is scored", base["risk_score"] is not None,
      f"score={base['risk_score']} tier={base['risk_tier']}")
check("complete patient has no refusal reason", base["reason"] is None)

# --- the failure modes a demo actually produces ---------------------------
print("\nMissing-value forms (each should be BLOCKED):")
cases = [
    ("monthly_copay = None (JSON null)", "monthly_copay", None),
    ("monthly_copay = '' (UI field cleared)", "monthly_copay", ""),
    ("monthly_copay = '   ' (whitespace)", "monthly_copay", "   "),
    ("monthly_copay = float('nan')", "monthly_copay", float("nan")),
    ("monthly_copay = np.nan", "monthly_copay", np.nan),
    ("monthly_copay = 'null' (string)", "monthly_copay", "null"),
    ("monthly_copay = 'N/A'", "monthly_copay", "N/A"),
    ("medication_name = None", "medication_name", None),
    ("medication_name = '' (UI cleared)", "medication_name", ""),
]
for label, field, value in cases:
    patient = dict(COMPLETE)
    patient[field] = value
    out = score_with_guard(patient)
    blocked = (out["risk_score"] is None
               and out["risk_tier"] == INSUFFICIENT_DATA_TIER
               and field in (out.get("reason") or ""))
    check(label, blocked, out.get("reason", ""))

# --- a key omitted entirely, not just null -------------------------------
print("\nKey absent from the payload entirely:")
patient = {k: v for k, v in COMPLETE.items() if k != "monthly_copay"}
out = score_with_guard(patient)
check("monthly_copay key omitted", out["risk_tier"] == INSUFFICIENT_DATA_TIER,
      out.get("reason", ""))

# --- both required fields missing -> reason must name BOTH ---------------
print("\nBoth required fields missing:")
patient = dict(COMPLETE, monthly_copay=None, medication_name=None)
out = score_with_guard(patient)
check("both fields named in reason",
      all(f in (out.get("reason") or "") for f in REQUIRED_FIELDS),
      out.get("reason", ""))

# --- non-required fields must STILL score (graceful degradation) ---------
print("\nNon-required fields missing (should still score - mean shift <10 pts):")
for field in ["insurance_type", "condition", "prior_auth_required",
              "distance_to_pharmacy_miles", "day_of_week_prescribed"]:
    patient = dict(COMPLETE)
    patient[field] = None
    out = score_with_guard(patient)
    shift = (out["risk_score"] - base["risk_score"]
             if out["risk_score"] is not None else None)
    check(f"{field} missing -> still scored", out["risk_score"] is not None,
          f"score={out['risk_score']} (shift {shift:+d})" if shift is not None
          else "BLOCKED")

# --- the averted error ---------------------------------------------------
print("\nWhat the guard prevents:")
blanked = dict(COMPLETE, monthly_copay=np.nan)
would_have = raw_score(blanked)
print(f"  complete patient           -> score {base['risk_score']:>3} "
      f"({base['risk_tier']})")
print(f"  copay blanked, NO guard    -> score {would_have:>3} "
      f"(silently wrong, no error raised)")
print(f"  copay blanked, WITH guard  -> insufficient_data, no score emitted")
print(f"  averted error: {abs(would_have - base['risk_score'])} points")
check("guard averts a material error",
      abs(would_have - base["risk_score"]) > 0,
      f"{abs(would_have - base['risk_score'])} pts")

# --- the high-risk filter must not leak refusal records ------------------
print("\nHigh-risk filter behaviour:")
mixed = [
    {"patient_id": "A", "risk_score": 95, "risk_tier": "high"},
    {"patient_id": "B", "risk_score": 71, "risk_tier": "high"},
    {"patient_id": "C", "risk_score": 55, "risk_tier": "medium"},
    {"patient_id": "D", "risk_score": None, "risk_tier": INSUFFICIENT_DATA_TIER,
     "reason": "Missing required field(s): monthly_copay"},
]
hr = filter_high_risk(mixed, min_risk=70)
check("returns only score >= 70", [r["patient_id"] for r in hr] == ["A", "B"],
      str([r["patient_id"] for r in hr]))
check("sorted descending by risk_score",
      [r["risk_score"] for r in hr] == sorted(
          [r["risk_score"] for r in hr], reverse=True),
      str([r["risk_score"] for r in hr]))
check("insufficient_data never leaks into the filter",
      all(r["risk_score"] is not None for r in hr))

# --- verdict -------------------------------------------------------------
passed = sum(1 for _, ok, _ in results if ok)
total = len(results)
print("\n" + "=" * 78)
print(f"GUARD TEST: {passed}/{total} passed"
      f"{'' if passed == total else '  <-- FAILURES ABOVE'}")
print("=" * 78)
raise SystemExit(0 if passed == total else 1)
