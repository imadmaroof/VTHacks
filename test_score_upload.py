"""
Test score_patient_upload() against a simulated doctor's upload.

Builds a 25-row upload from the training CSV, blanks monthly_copay on 3 rows,
and checks three things:

  1. the blanked rows come back as insufficient_data,
  2. everyone else gets a real score,
  3. those scores match patients_scored.json EXACTLY -- same model, same rows,
     so any difference would mean the serving path encodes data differently
     from the training path, which is the bug this test exists to catch.

It then exercises the failure modes a real upload service will actually see:
unrecognized categories, missing columns, junk files.

Run:  python test_score_upload.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pandas as pd

import train_adherence_model as tam
from score_upload import score_patient_upload
from scoring import INSUFFICIENT_DATA_TIER

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
UPLOAD_CSV = HERE / "test_upload_sample.csv"
SCRATCH = HERE / "_scratch_uploads"

N_ROWS = 25
N_BLANKED = 3

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}]  {name:<50} {detail}")


# ===========================================================================
# Build the upload
# ===========================================================================
print("=" * 78)
print("SCORE_UPLOAD TEST")
print("=" * 78)

source = tam.load_data()
sample = source.sample(N_ROWS, random_state=99).reset_index(drop=True)

# A doctor's upload has no outcome columns -- those are what we predict.
sample = sample.drop(
    columns=[tam.TARGET, "primary_discontinue_reason", "abandoned"],
    errors="ignore",
)

blanked_ids = sample["patient_id"].iloc[:N_BLANKED].tolist()
# pandas 3.0 refuses to put "" into a float64 column, so widen it first. The
# blank is written as an empty string rather than a NaN on purpose: that is
# what a cleared spreadsheet cell actually produces on export.
sample["monthly_copay"] = sample["monthly_copay"].astype(object)
sample.loc[: N_BLANKED - 1, "monthly_copay"] = ""  # a cleared field, not a zero

sample.to_csv(UPLOAD_CSV, index=False)
print(f"\nBuilt upload: {UPLOAD_CSV.name} "
      f"({len(sample)} rows, monthly_copay blanked on {blanked_ids})\n")

# ===========================================================================
# 1. Happy path
# ===========================================================================
print("Core behaviour:")
res = score_patient_upload(str(UPLOAD_CSV))

check("status is ok", res["status"] == "ok", res.get("error") or "")
check("error is None", res["error"] is None)
check("total_patients == rows uploaded", res["total_patients"] == N_ROWS,
      str(res["total_patients"]))
check("insufficient_data_count == blanked rows",
      res["insufficient_data_count"] == N_BLANKED,
      str(res["insufficient_data_count"]))
check("scored_count == the rest",
      res["scored_count"] == N_ROWS - N_BLANKED, str(res["scored_count"]))
check("counts sum to total",
      res["scored_count"] + res["insufficient_data_count"] == res["total_patients"])
check("all_patients has one record per row",
      len(res["all_patients"]) == N_ROWS, str(len(res["all_patients"])))

by_id = {r["patient_id"]: r for r in res["all_patients"]}

# ---- the blanked rows ----
print("\nGuard on blanked monthly_copay:")
for pid in blanked_ids:
    rec = by_id[pid]
    ok = (rec["risk_score"] is None
          and rec["risk_tier"] == INSUFFICIENT_DATA_TIER
          and "monthly_copay" in (rec["reason"] or ""))
    check(f"{pid} flagged insufficient_data", ok, rec["reason"] or "")

# ---- everyone else ----
print("\nEveryone else scored:")
others = [r for r in res["all_patients"] if r["patient_id"] not in blanked_ids]
check("all non-blanked rows have a score",
      all(r["risk_score"] is not None for r in others), f"{len(others)} rows")
check("all scores within 0-100",
      all(0 <= r["risk_score"] <= 100 for r in others))
check("all non-blanked rows have reason None",
      all(r["reason"] is None for r in others))
check("every scored row has exactly 3 top_factors",
      all(len(r["top_factors"]) == 3 for r in others))

# ===========================================================================
# 2. Scores must match patients_scored.json exactly
# ===========================================================================
print("\nAgreement with the batch export (same model, same patients):")
batch = {r["patient_id"]: r for r in
         json.loads((HERE / "patients_scored.json").read_text(encoding="utf-8"))}

compared, mismatches = 0, []
for rec in others:
    ref = batch.get(rec["patient_id"])
    if ref is None or ref["risk_score"] is None:
        continue
    compared += 1
    if ref["risk_score"] != rec["risk_score"]:
        mismatches.append(
            f"{rec['patient_id']}: upload {rec['risk_score']} vs batch {ref['risk_score']}")

check("scores identical to patients_scored.json", not mismatches,
      f"{compared} compared, {len(mismatches)} mismatched")
for m in mismatches[:5]:
    print(f"        {m}")

tier_mismatch = [r["patient_id"] for r in others
                 if batch.get(r["patient_id"])
                 and batch[r["patient_id"]]["risk_tier"] != r["risk_tier"]]
check("risk_tier identical to patients_scored.json", not tier_mismatch,
      f"{len(tier_mismatch)} differ")

factor_mismatch = [
    r["patient_id"] for r in others
    if batch.get(r["patient_id"])
    and [f["factor"] for f in batch[r["patient_id"]]["top_factors"]]
    != [f["factor"] for f in r["top_factors"]]
]
check("top_factors identical to patients_scored.json", not factor_mismatch,
      f"{len(factor_mismatch)} differ")

# Show a couple of concrete side-by-sides, as asked.
print("\n  sample comparison (upload vs patients_scored.json):")
for rec in others[:3]:
    ref = batch.get(rec["patient_id"])
    if ref:
        print(f"    {rec['patient_id']}: upload={rec['risk_score']:>3} "
              f"batch={ref['risk_score']:>3}  "
              f"tier {rec['risk_tier']}/{ref['risk_tier']}  "
              f"top={rec['top_factors'][0]['factor']}")

# ===========================================================================
# 3. Output schema must match the existing files exactly
# ===========================================================================
print("\nSchema compatibility:")
expected_keys = set(next(iter(batch.values())).keys())
check("record keys match patients_scored.json",
      all(set(r.keys()) == expected_keys for r in res["all_patients"]),
      str(sorted(expected_keys)))

hr = res["high_risk_patients"]
check("high_risk_patients all >= 70",
      all(r["risk_score"] >= 70 for r in hr), f"{len(hr)} patients")
check("high_risk_patients sorted descending",
      [r["risk_score"] for r in hr] == sorted(
          (r["risk_score"] for r in hr), reverse=True))
check("high_risk_count matches list length", res["high_risk_count"] == len(hr))
check("no insufficient_data leaked into high_risk",
      all(r["risk_score"] is not None for r in hr))

# ===========================================================================
# 4. Real-world upload failures
# ===========================================================================
print("\nUnrecognized category values (the hospital-data-mismatch case):")
SCRATCH.mkdir(exist_ok=True)

# 4a. Unrecognized value in a REQUIRED categorical -> must be blocked, because
#     after encoding it is NaN and the score would be silently wrong.
bad_med = sample.copy()
bad_med["monthly_copay"] = source["monthly_copay"].iloc[:N_ROWS].values  # un-blank
bad_med.loc[0, "medication_name"] = "Zyxadrin"  # a drug we never trained on
p = SCRATCH / "unseen_med.csv"
bad_med.to_csv(p, index=False)
r2 = score_patient_upload(str(p))
rec0 = r2["all_patients"][0]
check("unrecognized medication_name -> insufficient_data",
      rec0["risk_tier"] == INSUFFICIENT_DATA_TIER and rec0["risk_score"] is None,
      (rec0["reason"] or "")[:60])
check("reason distinguishes it from a blank field",
      "Unrecognized" in (rec0["reason"] or ""))

# 4b. Unrecognized value in a NON-required categorical -> should still score.
#     The sweep measured these at under 10 points of mean shift.
bad_ins = bad_med.copy()
bad_ins.loc[:, "medication_name"] = sample["medication_name"].values
bad_ins.loc[0, "insurance_type"] = "TRICARE"
p = SCRATCH / "unseen_ins.csv"
bad_ins.to_csv(p, index=False)
r3 = score_patient_upload(str(p))
check("unrecognized insurance_type -> still scored",
      r3["all_patients"][0]["risk_score"] is not None,
      f"score={r3['all_patients'][0]['risk_score']}")

# 4c. Currency-formatted copay: a very plausible hospital export quirk.
money = bad_ins.copy()
money["insurance_type"] = sample["insurance_type"].values
money["monthly_copay"] = ["$" + str(v) for v in money["monthly_copay"]]
p = SCRATCH / "money.csv"
money.to_csv(p, index=False)
r4 = score_patient_upload(str(p))
check("currency-formatted copay -> all blocked, not silently wrong",
      r4["insufficient_data_count"] == N_ROWS,
      f"{r4['insufficient_data_count']}/{N_ROWS} blocked")

print("\nMalformed / wrong files:")
p = SCRATCH / "missing_cols.csv"
sample.drop(columns=["monthly_copay", "age", "condition"]).to_csv(p, index=False)
r5 = score_patient_upload(str(p))
check("missing whole columns -> status error",
      r5["status"] == "error" and "monthly_copay" in (r5["error"] or ""),
      (r5["error"] or "")[:60])

p = SCRATCH / "empty.csv"
p.write_text("", encoding="utf-8")
check("empty file -> status error",
      score_patient_upload(str(p))["status"] == "error")

p = SCRATCH / "garbage.csv"
p.write_text("this is not a csv\x00\x01 at all", encoding="utf-8")
r6 = score_patient_upload(str(p))
check("garbage file -> error, no crash", r6["status"] == "error",
      (r6["error"] or "")[:60])

p = SCRATCH / "headers_only.csv"
sample.head(0).to_csv(p, index=False)
r7 = score_patient_upload(str(p))
check("headers but no rows -> status error", r7["status"] == "error",
      (r7["error"] or "")[:50])

r8 = score_patient_upload(str(SCRATCH / "does_not_exist.csv"))
check("nonexistent path -> status error", r8["status"] == "error")

check("error responses still carry every key",
      all(k in r8 for k in ("status", "error", "total_patients", "scored_count",
                            "insufficient_data_count", "high_risk_count",
                            "all_patients", "high_risk_patients")))

# ===========================================================================
# 5. Side-effect check: this must not have touched anything
# ===========================================================================
print("\nNo side effects:")
import hashlib
h = hashlib.md5((HERE / "model.pkl").read_bytes()).hexdigest()
res_again = score_patient_upload(str(UPLOAD_CSV))
check("model.pkl byte-identical after scoring",
      hashlib.md5((HERE / "model.pkl").read_bytes()).hexdigest() == h, h[:12])
check("repeat call returns identical scores",
      [r["risk_score"] for r in res_again["all_patients"]]
      == [r["risk_score"] for r in res["all_patients"]])

# ---- cleanup scratch files, keep the sample upload for manual CLI use ----
for f in SCRATCH.glob("*.csv"):
    f.unlink()
SCRATCH.rmdir()

passed = sum(1 for _, ok, _ in results if ok)
total = len(results)
print("\n" + "=" * 78)
print(f"SCORE_UPLOAD TEST: {passed}/{total} passed"
      f"{'' if passed == total else '  <-- FAILURES ABOVE'}")
print(f"Sample upload kept at {UPLOAD_CSV.name} for: "
      f"python score_upload.py {UPLOAD_CSV.name}")
print("=" * 78)
raise SystemExit(0 if passed == total else 1)
