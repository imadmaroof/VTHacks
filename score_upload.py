"""
Score a newly uploaded patient CSV with the already-trained model.

This is the serving path: a doctor uploads their hospital's patient list and
gets risk scores back. It loads model.pkl and applies it. It never retrains,
never writes to model.pkl, and never touches the training CSV or
patients_scored.json. Everything it produces is returned in memory -- whether
to persist it is the API layer's decision, not this module's.

    from score_upload import score_patient_upload
    result = score_patient_upload("uploads/clinic_batch.csv")

Manual test without a backend:

    python score_upload.py test_upload_sample.csv


WHY THIS DOES NOT CALL prepare_features()
-----------------------------------------
prepare_features() is the right function for TRAINING and the wrong one for
SERVING, for two reasons that have nothing to do with code quality:

  1. It requires the label column. `y = df[TARGET].astype(int)` raises a
     KeyError on any upload, because an uploaded patient list has no
     `discontinued` column -- that is the thing we are predicting.

  2. It derives the category vocabulary from whatever frame you hand it:
     `X[col].astype("category")`. During training that frame is the full
     2,000-row dataset, so the vocabulary is complete. On an upload it would
     be the upload, so a clinic with no Medicaid patients would produce a
     different vocabulary than the model was trained on, and -- more
     importantly -- a value the model has NEVER seen would be silently
     accepted as a legitimate new category instead of being flagged.

     (Worth knowing: XGBoost 3.x re-maps categories by VALUE, not by integer
     code, so a differently-ordered vocabulary does not by itself corrupt
     predictions -- verified directly against this model. The vocabulary is
     still pinned to the training levels here, because relying on that
     re-mapping is version-specific behaviour and because pinning is what
     turns an unrecognized value into a detectable NaN.)

So this module reuses the POLICY from the training script -- which columns are
identifiers, leakage, redundant, categorical -- and re-implements only the
encoding step, against the category vocabulary saved in model.pkl at training
time. The policy cannot drift because it is imported, not copied.


HOW UNRECOGNIZED CATEGORY VALUES BEHAVE (read this before a live demo)
---------------------------------------------------------------------
A hospital's data will not perfectly match our training categories. Suppose a
clinic sends insurance_type "TRICARE", or a drug we have never seen:

    pd.Categorical(["Zyxadrin"], categories=<22 training drugs>)  ->  NaN

Pandas turns the unrecognized value into NaN. It does not raise. XGBoost then
treats that NaN as a missing value and routes it down a learned default
branch, and returns a perfectly confident number. Measured on this model,
replacing medication_name with an unrecognized value moved scores like so:

    real values : [99,  2, 52,  2, 17,  5, 58,  3]
    unrecognized: [100, 33, 80, 23, 72, 72, 77, 34]

Several of those cross a risk tier boundary. Nothing errors.

This is a DIFFERENT failure from a blank field, and the existing guard in
scoring.py does not catch it on its own: missing_required_fields() inspects
the raw submitted value, and "Zyxadrin" is a present, non-empty string. It is
only unusable after encoding.

So this module applies the guard in two stages:

    stage 1 (raw)      missing_required_fields()  -- nulls, "", "N/A", absent keys
    stage 2 (encoded)  required field became NaN  -- unrecognized categories,
                                                     and numerics that would not
                                                     parse, e.g. "$53.64"

A row failing either stage gets risk_score null and risk_tier
"insufficient_data", exactly like the batch export. Non-required fields are
left to XGBoost's native missing handling, which the evaluation measured at
under 10 points of mean score shift -- acceptable degradation.
"""

from __future__ import annotations

import functools
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap

# --- Column POLICY, imported from training so it cannot drift --------------
# Note prepare_features() itself is deliberately not imported; see the module
# docstring. These are the constants that define which columns mean what.
from train_adherence_model import (
    CATEGORICAL_COLUMNS,
    ID_COLUMNS,
    LEAKY_COLUMNS,
    REDUNDANT_COLUMNS,
    TARGET,
)

# --- Guard + tier logic, imported from the single source of truth ----------
# scoring.tier_for() is used rather than train_adherence_model's tier(), which
# is defined inside main() and therefore not importable. Same cutoffs: both
# modules now read TIER_HIGH / TIER_MEDIUM from scoring.py.
from scoring import (
    INSUFFICIENT_DATA_TIER,
    REQUIRED_FIELDS,
    TIER_HIGH,
    filter_high_risk,
    guard_reason,
    insufficient_data_record,
    is_missing,
    missing_required_fields,
    tier_for,
)

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "model.pkl"

# Columns the OUTPUT schema needs that are not model features. patient_id is
# dropped before training (it is an identifier) but every returned record is
# keyed by it, so an upload without it cannot produce actionable results.
IDENTITY_COLUMNS = ["patient_id"]


@functools.lru_cache(maxsize=1)
def _load_model_bundle() -> dict[str, Any]:
    """Load model.pkl once per process and reuse it on every call.

    A CLI invocation only ever calls this once anyway, so this changes
    nothing there. It matters for a long-running server (e.g. FastAPI),
    where reloading a joblib pickle on every request would be wasteful --
    lru_cache(maxsize=1) makes repeated calls in the same process free.
    """
    return joblib.load(MODEL_PATH)


def preload_model() -> None:
    """Force the model to load now rather than on the first prediction.

    Intended for a server startup hook, so the first real request isn't the
    one that pays for the load. Safe to call more than once -- the cache
    means only the first call actually touches disk.
    """
    _load_model_bundle()


def _empty_result(status: str, error: str | None = None) -> dict[str, Any]:
    """
    Build a complete response with zero patients.

    Every key is always present, including on failure, so a caller can read
    result["scored_count"] without first checking status and without risking
    a KeyError.
    """
    return {
        "status": status,
        "error": error,
        "total_patients": 0,
        "scored_count": 0,
        "insufficient_data_count": 0,
        "high_risk_count": 0,
        "all_patients": [],
        "high_risk_patients": [],
    }


def _safe_int(value: Any) -> int | None:
    """int() that returns None instead of raising on junk from a stranger's CSV."""
    if is_missing(value):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _safe_str(value: Any) -> str | None:
    return None if is_missing(value) else str(value)


def _encode(df: pd.DataFrame, feature_names: list[str],
            category_levels: dict[str, list]) -> pd.DataFrame:
    """
    Turn an uploaded frame into the exact matrix the model expects.

    Three things matter here:

      * Column ORDER. XGBoost validates feature names and raises
        "feature_names mismatch" if the order differs, so we reindex to the
        exact training order stored in model.pkl.

      * Categorical columns are pinned to the TRAINING vocabulary. Any value
        outside it becomes NaN (see module docstring) rather than being
        accepted as a new category.

      * Everything else is coerced with pd.to_numeric(errors="coerce").
        Booleans became ints during training, so True/False parse cleanly;
        anything unparseable ("N/A", "$53.64", "") becomes NaN. Under pandas
        3.0 an un-coerced text column arrives as `str` dtype, which XGBoost
        rejects outright, so this step is required, not cosmetic.
    """
    X = df.reindex(columns=feature_names).copy()
    for col in feature_names:
        if col in category_levels:
            X[col] = pd.Categorical(X[col], categories=category_levels[col])
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    return X


def score_patient_upload(csv_path: str) -> dict:
    """
    Score a new, uploaded patient CSV using the trained model.
    Does not retrain. Does not modify model.pkl.

    Args:
        csv_path: path to the uploaded CSV.

    Returns:
        {
          "status": "ok" | "error",
          "error": str | None,          # populated only if status == "error"
          "total_patients": int,
          "scored_count": int,
          "insufficient_data_count": int,
          "high_risk_count": int,
          "all_patients": [...],         # every row, same schema as
                                         #   patients_scored.json
          "high_risk_patients": [...],   # filtered + sorted, same schema
                                         #   as patients_high_risk.json
        }

    Records in both lists carry exactly the keys patients_scored.json uses:
    patient_id, age, condition, medication_name, risk_score, risk_tier,
    reason, top_factors. A frontend built against those files works against
    this function's output with no changes.
    """
    # ---- load the trained model ------------------------------------------
    if not MODEL_PATH.exists():
        return _empty_result("error", f"Trained model not found at {MODEL_PATH}. "
                                      "Run train_adherence_model.py first.")
    try:
        bundle = _load_model_bundle()
        model = bundle["model"]
        feature_names = bundle["feature_names"]
        category_levels = bundle["category_levels"]
    except Exception as exc:
        return _empty_result("error", f"Could not load model.pkl: {exc}")

    # ---- FILE-LEVEL VALIDATION: is this even the right kind of file? -----
    # This is deliberately separate from the per-row guard. Here we ask "does
    # this file have the right shape", before asking "is this row scoreable".
    path = Path(csv_path)
    if not path.exists():
        return _empty_result("error", f"File not found: {csv_path}")
    if path.stat().st_size == 0:
        return _empty_result("error", f"File is empty: {csv_path}")

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return _empty_result("error",
                             f"File contains no readable CSV data: {csv_path}")
    except pd.errors.ParserError as exc:
        return _empty_result("error",
                             f"Could not parse CSV (check delimiter/quoting): {exc}")
    except UnicodeDecodeError:
        return _empty_result("error",
                             "File is not valid UTF-8 text -- is it an .xlsx or a "
                             "binary file renamed to .csv?")
    except Exception as exc:
        return _empty_result("error", f"Could not read CSV: {exc}")

    if len(df) == 0:
        return _empty_result("error", "CSV parsed but contains no data rows.")

    # Whole columns absent is a different failure from individual nulls: we
    # cannot score at all, and filling them in would be exactly the "silently
    # produce garbage" behaviour we are trying to avoid.
    needed = IDENTITY_COLUMNS + list(feature_names)
    absent = [c for c in needed if c not in df.columns]
    if absent:
        return _empty_result(
            "error",
            f"Uploaded file is missing {len(absent)} required column(s): "
            f"{', '.join(absent)}. Expected a patient export containing: "
            f"{', '.join(needed)}.",
        )

    # An upload that still carries outcome columns is not an error -- a
    # hospital may export its full table -- but they are never fed to the
    # model. Dropping them here makes that explicit and keeps the leakage
    # policy identical to training.
    outcome_cols = [c for c in (LEAKY_COLUMNS + [TARGET] + REDUNDANT_COLUMNS)
                    if c in df.columns]
    df = df.drop(columns=outcome_cols, errors="ignore")

    total = len(df)

    # ---- ENCODE, then apply the guard in two stages ----------------------
    X = _encode(df, feature_names, category_levels)

    raw_rows = df.to_dict(orient="records")
    guards: list[tuple[list[str], list[str]]] = []
    for i, raw in enumerate(raw_rows):
        stage1 = missing_required_fields(raw)               # blank / absent
        stage2 = [                                          # unrecognized / unparseable
            f for f in REQUIRED_FIELDS
            if f not in stage1 and f in X.columns and pd.isna(X.iloc[i][f])
        ]
        guards.append((stage1, stage2))

    eligible = np.array([not (s1 or s2) for s1, s2 in guards])
    X_ok = X[eligible]
    pos_of = {orig: k for k, orig in enumerate(np.flatnonzero(eligible))}

    # ---- score + explain, eligible rows only -----------------------------
    if len(X_ok):
        try:
            proba = model.predict_proba(X_ok)[:, 1]
            shap_values = shap.TreeExplainer(model)(X_ok).values
        except Exception as exc:
            return _empty_result("error", f"Scoring failed: {exc}")
        scores = np.round(proba * 100).astype(int)
    else:
        scores = np.array([], dtype=int)
        shap_values = np.empty((0, len(feature_names)))

    # ---- assemble records ------------------------------------------------
    records: list[dict[str, Any]] = []
    for i, raw in enumerate(raw_rows):
        stage1, stage2 = guards[i]

        if stage1 or stage2:
            # Sanitize identity fields first: insufficient_data_record casts
            # age to int, which would raise on junk like "unknown".
            clean = dict(raw)
            clean["age"] = _safe_int(raw.get("age"))
            rec = insufficient_data_record(clean, stage1 + stage2)

            # Name the real cause. A blank field and an unrecognized value are
            # different problems for whoever has to fix the upload.
            reasons = []
            if stage1:
                reasons.append(guard_reason(stage1))
            if stage2:
                shown = ", ".join(f"{f}={raw.get(f)!r}" for f in stage2)
                reasons.append(
                    f"Unrecognized value for required field(s): {shown} "
                    "(not present in the training data, so the model cannot use it)"
                )
            rec["reason"] = " | ".join(reasons)
            records.append(rec)
            continue

        k = pos_of[i]
        row_shap = shap_values[k]
        top_idx = np.argsort(np.abs(row_shap))[::-1][:3]
        score = int(scores[k])
        records.append({
            "patient_id": _safe_str(raw.get("patient_id")),
            "age": _safe_int(raw.get("age")),
            "condition": _safe_str(raw.get("condition")),
            "medication_name": _safe_str(raw.get("medication_name")),
            "risk_score": score,
            "risk_tier": tier_for(score),
            "reason": None,
            "top_factors": [
                {"factor": feature_names[j], "impact": round(float(row_shap[j]), 4)}
                for j in top_idx
            ],
        })

    high_risk = filter_high_risk(records, min_risk=TIER_HIGH)
    n_insufficient = sum(
        1 for r in records if r["risk_tier"] == INSUFFICIENT_DATA_TIER)

    return {
        "status": "ok",
        "error": None,
        "total_patients": total,
        "scored_count": total - n_insufficient,
        "insufficient_data_count": n_insufficient,
        "high_risk_count": len(high_risk),
        "all_patients": records,
        "high_risk_patients": high_risk,
    }


# ===========================================================================
# Manual CLI test -- lets you sanity-check an upload before a backend exists.
# Kept out of the importable surface: nothing above depends on this block.
# ===========================================================================
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python score_upload.py <path/to/upload.csv>")
        raise SystemExit(2)

    result = score_patient_upload(sys.argv[1])

    print("=" * 70)
    print(f"UPLOAD SCORING: {sys.argv[1]}")
    print("=" * 70)

    if result["status"] == "error":
        print(f"STATUS : error\nREASON : {result['error']}")
        raise SystemExit(1)

    print(f"status                   : {result['status']}")
    print(f"total_patients           : {result['total_patients']}")
    print(f"scored_count             : {result['scored_count']}")
    print(f"insufficient_data_count  : {result['insufficient_data_count']}")
    print(f"high_risk_count          : {result['high_risk_count']}")

    if result["insufficient_data_count"]:
        print("\nRows that could not be scored:")
        for rec in result["all_patients"]:
            if rec["risk_tier"] == INSUFFICIENT_DATA_TIER:
                print(f"  {rec['patient_id']}: {rec['reason']}")

    print("\nTop 3 highest-risk patients:")
    if not result["high_risk_patients"]:
        print("  (none scored >= 70)")
    for rec in result["high_risk_patients"][:3]:
        factors = ", ".join(
            f"{f['factor']} {f['impact']:+.2f}" for f in rec["top_factors"])
        print(f"  {rec['patient_id']:<12} score {rec['risk_score']:>3} "
              f"({rec['risk_tier']})  {rec['condition']} / {rec['medication_name']}")
        print(f"      drivers: {factors}")
    raise SystemExit(0)
