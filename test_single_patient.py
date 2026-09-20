"""
Proof that score_single_patient() -- the interactive "customizable patient"
demo's scoring path -- behaves identically to the validated batch/upload
path, and that patient_options.py only ever offers choices the model
actually knows how to score.

Three things matter here, matching what the demo form depends on:

  1. A patient built from patient_options.DEFAULT_PATIENT and scored via
     score_single_patient() gets EXACTLY the same score as the identical
     patient scored via score_patient_upload() from a one-row CSV -- same
     model, same guard, same SHAP, because both call _build_records()
     underneath. If this ever drifts, the live-form demo would show a
     different number than a CSV upload of the same patient.
  2. The guard still fires for a single patient exactly as it does for a
     batch: missing monthly_copay/medication_name, or a medication not in
     the training vocabulary (the "Tylenol" case), comes back as
     insufficient_data instead of a silently wrong score.
  3. Every value patient_options.py offers as a dropdown option is one the
     model actually saw in training -- the condition -> medication mapping
     never points at a medication outside that condition's training pairing,
     and every categorical value listed is in the model's own category
     vocabulary. This is what stops a judge from ever seeing "Rheumatex" for
     "Type 2 Diabetes" mid-demo.

Run:  python test_single_patient.py
"""

from __future__ import annotations

import warnings

import pandas as pd

from patient_options import (
    CONDITION_TO_DRUG_CLASS,
    CONDITION_TO_MEDICATIONS,
    DEFAULT_PATIENT,
    NUMERIC_RANGES,
    build_field_options,
)
from score_upload import score_patient_upload, score_single_patient

warnings.filterwarnings("ignore")

SCRATCH_CSV = "_scratch_single_patient.csv"

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}]  {name:<58} {detail}")


print("=" * 78)
print("SINGLE-PATIENT SCORING TEST")
print("=" * 78)

# ===========================================================================
# 1. score_single_patient() agrees with score_patient_upload() exactly
# ===========================================================================
print("\nAgreement with the batch/upload path:")

patient = dict(DEFAULT_PATIENT, patient_id="DEMO-AGREEMENT")
single = score_single_patient(patient)
check("single patient scores ok", single["status"] == "ok", single.get("error") or "")

pd.DataFrame([patient]).to_csv(SCRATCH_CSV, index=False)
batch = score_patient_upload(SCRATCH_CSV)
check("batch upload of the same row scores ok", batch["status"] == "ok",
      batch.get("error") or "")

if single["status"] == "ok" and batch["status"] == "ok":
    batch_rec = batch["all_patients"][0]
    check("risk_score identical",
          single["patient"]["risk_score"] == batch_rec["risk_score"],
          f"single={single['patient']['risk_score']} batch={batch_rec['risk_score']}")
    check("risk_tier identical",
          single["patient"]["risk_tier"] == batch_rec["risk_tier"])
    check("top_factors identical",
          single["patient"]["top_factors"] == batch_rec["top_factors"])

# ===========================================================================
# 2. The guard still fires for a single patient
# ===========================================================================
print("\nGuard behaviour on a single patient:")

missing_copay = dict(DEFAULT_PATIENT, monthly_copay=None)
out = score_single_patient(missing_copay)
check("missing monthly_copay -> insufficient_data",
      out["patient"]["risk_tier"] == "insufficient_data"
      and out["patient"]["risk_score"] is None
      and "monthly_copay" in (out["patient"]["reason"] or ""),
      out["patient"].get("reason", ""))

unseen_drug = dict(DEFAULT_PATIENT, medication_name="Tylenol")
out = score_single_patient(unseen_drug)
check("medication never in training data -> insufficient_data (not silently scored)",
      out["patient"]["risk_tier"] == "insufficient_data"
      and out["patient"]["risk_score"] is None,
      out["patient"].get("reason", ""))

complete = score_single_patient(dict(DEFAULT_PATIENT, patient_id="DEMO-COMPLETE"))
check("a fully-specified patient is never blocked",
      complete["patient"]["risk_score"] is not None,
      f"score={complete['patient']['risk_score']}")

# ===========================================================================
# 3. Dragging monthly_copay produces the "live demo" score movement
# ===========================================================================
print("\nCopay slider moves the score (the flagship demo moment):")

low_copay = score_single_patient(dict(DEFAULT_PATIENT, monthly_copay=0))["patient"]
high_copay = score_single_patient(dict(DEFAULT_PATIENT, monthly_copay=500))["patient"]
check("raising monthly_copay from $0 to $500 raises the risk score",
      high_copay["risk_score"] > low_copay["risk_score"],
      f"$0 -> {low_copay['risk_score']}, $500 -> {high_copay['risk_score']}")

# ===========================================================================
# 4. patient_options.py never offers a choice the model can't score
# ===========================================================================
print("\npatient_options.py policy sanity (protects the live demo from a bad pairing):")

opts = build_field_options()
model_meds = set(opts["categorical"]["medication_name"])
model_conditions = set(opts["categorical"]["condition"])
model_drug_classes = set(opts["categorical"]["drug_class"])

check("every condition in the mapping is a real trained category",
      set(CONDITION_TO_MEDICATIONS) <= model_conditions)
check("every condition has a drug_class mapping",
      set(CONDITION_TO_MEDICATIONS) == set(CONDITION_TO_DRUG_CLASS))

all_mapped_meds = {m for meds in CONDITION_TO_MEDICATIONS.values() for m in meds}
check("every medication in the condition mapping is a real trained category",
      all_mapped_meds <= model_meds,
      f"unmapped: {sorted(all_mapped_meds - model_meds) or 'none'}")
check("every drug_class in the mapping is a real trained category",
      set(CONDITION_TO_DRUG_CLASS.values()) <= model_drug_classes)

# Confirm every condition -> medication pairing actually scores as "eligible"
# (never triggers the unrecognized-category guard), i.e. it is a real pairing
# the model has seen, not just individually-valid values combined arbitrarily.
bad_pairings = []
for condition, meds in CONDITION_TO_MEDICATIONS.items():
    for med in meds:
        patient = dict(DEFAULT_PATIENT, condition=condition,
                       drug_class=CONDITION_TO_DRUG_CLASS[condition],
                       medication_name=med)
        rec = score_single_patient(patient)["patient"]
        if rec["risk_tier"] == "insufficient_data":
            bad_pairings.append(f"{condition}/{med}")
check("every condition+medication pairing in the dropdown scores cleanly",
      not bad_pairings, f"failed: {bad_pairings or 'none'}")

# Default patient must itself use only mapped, in-vocabulary values -- it is
# what the form shows before a judge touches anything.
check("DEFAULT_PATIENT's medication matches its own condition",
      DEFAULT_PATIENT["medication_name"]
      in CONDITION_TO_MEDICATIONS[DEFAULT_PATIENT["condition"]])
check("DEFAULT_PATIENT's drug_class matches its own condition",
      DEFAULT_PATIENT["drug_class"] == CONDITION_TO_DRUG_CLASS[DEFAULT_PATIENT["condition"]])

for field, (lo, hi) in NUMERIC_RANGES.items():
    val = DEFAULT_PATIENT.get(field)
    in_range = val is not None and lo <= val <= hi
    check(f"DEFAULT_PATIENT.{field} within its own numeric range",
          in_range, f"{val} in [{lo}, {hi}]" if in_range else f"{val} NOT in [{lo}, {hi}]")

# ---- cleanup --------------------------------------------------------------
import os
if os.path.exists(SCRATCH_CSV):
    os.remove(SCRATCH_CSV)

passed = sum(1 for _, ok, _ in results if ok)
total = len(results)
print("\n" + "=" * 78)
print(f"SINGLE-PATIENT TEST: {passed}/{total} passed"
      f"{'' if passed == total else '  <-- FAILURES ABOVE'}")
print("=" * 78)
raise SystemExit(0 if passed == total else 1)
