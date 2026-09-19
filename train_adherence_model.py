"""
Medication Adherence Risk Model
===============================

Predicts `discontinued` (did the patient stop taking the medication?) from
patient / prescription / access attributes, then explains each individual
prediction with SHAP and exports per-patient risk scores to JSON.

PIPELINE OVERVIEW
-----------------
    CSV
     |
     +--> PART 1  Data prep      : drop leaky columns, stratified 80/20 split,
     |                             declare categoricals
     +--> PART 2  Training       : XGBoost + scale_pos_weight, evaluate with
     |                             AUC-ROC / precision / recall, save model.pkl
     +--> PART 3  Explainability : SHAP values per patient, global importance
     +--> PART 4  Export         : patients_scored.json
     +--> PART 5  Validation     : sanity checks against held-out truth columns

Run:  python train_adherence_model.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# Scoring guards and the dashboard filter live in scoring.py so that this batch
# export and any future API enforce the SAME rules. See that module for why
# monthly_copay and medication_name are treated as required.
from scoring import (
    REQUIRED_FIELDS,
    filter_high_risk,
    insufficient_data_record,
    missing_required_fields,
)

# `random_state` fixes the pseudo-random number generator so the split and the
# model are reproducible. Re-running gives byte-identical output. Without this,
# every run would shuffle rows differently and your metrics would drift.
RANDOM_STATE = 42

HERE = Path(__file__).resolve().parent

# The file ships with spaces in the name; accept either spelling.
CSV_CANDIDATES = [
    HERE / "Synthetic_Patients_Processed.csv",
    HERE / "Synthetic Patients Processed.csv",
    Path.home() / "Downloads" / "Synthetic Patients Processed.csv",
    Path.home() / "Downloads" / "Synthetic_Patients_Processed.csv",
]

MODEL_PATH = HERE / "model.pkl"
JSON_PATH = HERE / "patients_scored.json"
HIGH_RISK_JSON_PATH = HERE / "patients_high_risk.json"

# ---------------------------------------------------------------------------
# COLUMN POLICY
# ---------------------------------------------------------------------------
# TARGET is what we predict. Everything in the EXCLUDE lists is kept in the
# dataframe (we need it for the validation section at the end) but never handed
# to the model.
TARGET = "discontinued"

# --- Excluded because of FEATURE LEAKAGE -----------------------------------
# Feature leakage = giving the model information that would not exist at the
# moment you actually need a prediction. A leaky model scores brilliantly in
# testing and is useless in production, because in production that column is
# either blank or literally a restatement of the answer.
#
#   primary_discontinue_reason : equals "none" for every non-discontinued row.
#                                Knowing it != "none" IS knowing the label.
#   abandoned                  : a second outcome, not an input. Verified on
#                                this data: all 336 abandoned=True rows are
#                                also discontinued=True, and 86 discontinued
#                                rows are not abandoned -- so it is a strict
#                                subset flagging early/severe dropout. It only
#                                becomes known at the same time as the label,
#                                so it cannot be an input.
LEAKY_COLUMNS = ["primary_discontinue_reason", "abandoned"]

# --- Excluded because it is an IDENTIFIER -----------------------------------
# patient_id is unique per row. A tree could memorize it and learn nothing
# that generalizes to a patient it has never seen.
ID_COLUMNS = ["patient_id"]

# --- Excluded because they are REDUNDANT (not leakage) ----------------------
# These are pre-binned copies of numeric columns we already have at full
# resolution. Feeding both hurts in two ways:
#   1. The binned copy is strictly less informative ($24.99 and $0.50 collapse
#      into one "<$25" bucket) -- a gradient-boosted tree finds its own optimal
#      split points, so hand-made bins only throw away resolution.
#   2. It splits one real signal's SHAP importance across two columns, so the
#      importance chart understates how much "copay" actually matters.
REDUNDANT_COLUMNS = ["copay_bucket", "prior_abandoned_bucket"]

# NOTE / JUDGMENT CALL: `age_bracket` was not in your exclusion list, but it is
# a binned copy of `age` -- exactly the case above. Dropping it for the same
# reason. Flip this to False to keep it as a feature instead.
DROP_AGE_BRACKET = True
if DROP_AGE_BRACKET:
    REDUNDANT_COLUMNS.append("age_bracket")

# Categorical (non-numeric) features. `medication_name` was not in your list,
# but it is a text column that has to be handled somehow, and it is a genuine
# predictor (different drugs differ in tolerability and price). Included here
# as a categorical; drop it from this list if you would rather exclude it.
CATEGORICAL_COLUMNS = [
    "condition",
    "drug_class",
    "medication_name",
    "new_rx_or_refill",
    "insurance_type",
    "day_of_week_prescribed",
]

# Risk tier cutoffs on the 0-100 risk score.
TIER_HIGH = 70
TIER_MEDIUM = 40


def banner(title: str) -> None:
    """Print a section header so the console output is readable."""
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# ===========================================================================
# PART 1 - DATA PREP
# ===========================================================================
def load_data() -> pd.DataFrame:
    """Locate and load the CSV."""
    for path in CSV_CANDIDATES:
        if path.exists():
            print(f"Loading: {path}")
            return pd.read_csv(path)
    raise FileNotFoundError(
        "Could not find the dataset. Looked in:\n  "
        + "\n  ".join(str(p) for p in CSV_CANDIDATES)
    )


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Split the dataframe into a model matrix X, a label vector y, and the list
    of feature names.

    Returns X with categorical columns typed as pandas `category`, which is
    what lets XGBoost use its native categorical handling (see below).
    """
    drop_cols = ID_COLUMNS + LEAKY_COLUMNS + REDUNDANT_COLUMNS + [TARGET]
    feature_names = [c for c in df.columns if c not in drop_cols]

    X = df[feature_names].copy()

    # The label. Cast to int (0/1) because XGBoost wants a numeric target.
    y = df[TARGET].astype(int)

    # ---- CATEGORICAL ENCODING --------------------------------------------
    # Three common options, and why we picked the third:
    #
    #   One-hot encoding: one new 0/1 column per category value. Safe and
    #     universal, but it explodes the feature count (medication_name alone
    #     becomes 22 columns) and it fragments a tree's splits -- the tree can
    #     only ask "is it Serenol? yes/no" instead of partitioning all 22 drugs
    #     in one split. It also scatters SHAP importance across 22 columns,
    #     which is the same problem we just avoided by dropping the bucket
    #     columns.
    #
    #   Label encoding: map each category to an integer (Mon=0, Tue=1 ...).
    #     Compact, but it invents a false ordering. The tree can only split on
    #     ranges, so it is forced to treat "Mon < Tue < Wed" as meaningful.
    #
    #   Native categorical support (what we use): mark the column as pandas
    #     `category` dtype and pass enable_categorical=True. XGBoost then
    #     partitions the category values optimally at each split, with no fake
    #     ordering and no column explosion. One column in, one clean SHAP
    #     importance out.
    #
    # IMPORTANT: we set the category dtype BEFORE the train/test split so both
    # halves share one identical category->code mapping. Doing it after the
    # split would let "Medicaid" mean a different internal code in each half.
    for col in CATEGORICAL_COLUMNS:
        if col in X.columns:
            X[col] = X[col].astype("category")

    # Booleans become 0/1. Trees handle these natively as numeric.
    for col in X.columns:
        if X[col].dtype == bool:
            X[col] = X[col].astype(int)

    return X, y, feature_names


def main() -> None:
    banner("PART 1 - DATA PREP")

    df = load_data()
    print(f"Loaded {len(df):,} rows x {df.shape[1]} columns")

    # Make sure the target is boolean/int, not the strings "True"/"False".
    if df[TARGET].dtype == object:
        df[TARGET] = df[TARGET].map({"True": True, "False": False})

    X, y, feature_names = prepare_features(df)

    print(f"\nExcluded from features:")
    print(f"  identifier : {ID_COLUMNS}")
    print(f"  leakage    : {LEAKY_COLUMNS}")
    print(f"  redundant  : {REDUNDANT_COLUMNS}")
    print(f"\nUsing {len(feature_names)} features:")
    for name in feature_names:
        kind = "categorical" if name in CATEGORICAL_COLUMNS else "numeric"
        print(f"  - {name:<32} ({kind})")

    # ---- CLASS IMBALANCE --------------------------------------------------
    # Class imbalance = one outcome is far more common than the other. Here
    # ~21% discontinued vs ~79% not. The danger: a model that ignores every
    # feature and always answers "False" is 79% accurate. That is why accuracy
    # is the wrong thing to optimize -- it rewards a model that never finds a
    # single at-risk patient, which is the only thing we actually care about.
    pos = int(y.sum())
    neg = int(len(y) - pos)
    print(f"\nClass balance: {pos} discontinued ({pos / len(y):.1%}) "
          f"vs {neg} retained ({neg / len(y):.1%})")
    print(f"  -> a 'always predict False' model scores {neg / len(y):.1%} accuracy "
          f"and finds 0 at-risk patients.")

    # ---- STRATIFIED TRAIN/TEST SPLIT --------------------------------------
    # We hold out 20% of rows the model never sees during training, so our
    # metrics estimate performance on genuinely new patients.
    #
    # "Stratified" means the split preserves the class ratio in both halves --
    # ~21% discontinued in train AND ~21% in test. With a random unstratified
    # split on an imbalanced, smallish dataset you can get a test set with, say,
    # 17% positives purely by chance, which makes the metrics noisy and not
    # comparable between runs. `stratify=y` removes that source of noise.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    print(f"\nTrain: {len(X_train):,} rows ({y_train.mean():.1%} positive)")
    print(f"Test : {len(X_test):,} rows ({y_test.mean():.1%} positive)")

    # =======================================================================
    # PART 2 - MODEL TRAINING
    # =======================================================================
    banner("PART 2 - MODEL TRAINING")

    # ---- WHY GRADIENT-BOOSTED TREES ---------------------------------------
    # A gradient-boosted tree ensemble builds many small decision trees in
    # sequence, where each new tree is fit to correct the errors the ensemble
    # has made so far. For tabular data like this -- mixed numeric and
    # categorical columns, non-linear effects, interactions between features --
    # it is the strongest default, and it needs no feature scaling.
    #
    # XGBoost vs LightGBM:
    #   LightGBM is faster on large datasets (it grows trees leaf-wise and bins
    #     features aggressively) and has the longer-established native
    #     categorical support. On 2,000 rows that speed edge is irrelevant.
    #   XGBoost grows trees level-wise, which is a bit more conservative and
    #     tends to overfit less on small data like this. It also has the most
    #     mature SHAP integration and the single-knob `scale_pos_weight`.
    # On a 2,000-row set either is fine; we use XGBoost for the small-data
    # robustness and the cleaner explainability path.

    # ---- scale_pos_weight -------------------------------------------------
    # This is the explicit imbalance handling. Internally the model minimizes a
    # loss summed over rows; scale_pos_weight multiplies the loss contribution
    # of every POSITIVE (discontinued) row by this factor. At neg/pos, the two
    # classes contribute equal total weight, so "always guess False" stops
    # being a cheap win and the model is forced to actually learn which
    # features separate the classes.
    #
    # Computed on the TRAINING set only -- using the full dataset would leak
    # information about the test set's composition into training.
    train_pos = int(y_train.sum())
    train_neg = int(len(y_train) - train_pos)
    scale_pos_weight = train_neg / train_pos
    print(f"scale_pos_weight = {train_neg} / {train_pos} = {scale_pos_weight:.3f}")
    print("  -> each discontinued patient counts ~%.2fx a retained one in the loss."
          % scale_pos_weight)

    model = xgb.XGBClassifier(
        # --- capacity / regularization (tuned conservatively for 2k rows) ---
        # 150, reduced from an original 400 to curb overfitting. With 400 the
        # train/test AUC gap was 0.109 (train AUC 0.9995 -- near-total
        # memorization of the training rows). A sweep of 40-400 trees showed
        # 5-fold CV AUC is flat across that whole range (0.8716-0.8790, inside
        # one standard deviation), so the extra trees bought no generalization,
        # only memorization. At 150 the gap falls to 0.084 while held-out AUC
        # actually rises. See the summary table in evaluate_model.py output.
        n_estimators=150,        # number of boosting rounds (trees)
        learning_rate=0.05,      # how much each tree corrects; lower = steadier
        max_depth=4,             # shallow trees; deep trees memorize small data
        min_child_weight=5,      # min rows per leaf; blocks tiny overfit leaves
        subsample=0.8,           # each tree sees 80% of rows  -> less overfit
        colsample_bytree=0.8,    # each tree sees 80% of columns -> less overfit
        reg_lambda=1.0,          # L2 penalty on leaf weights
        # --- imbalance ---
        scale_pos_weight=scale_pos_weight,
        # --- mechanics ---
        tree_method="hist",      # required for native categorical support
        enable_categorical=True, # honor the pandas `category` dtype
        eval_metric="aucpr",     # track area under precision-recall while training
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    print(f"\nTrained {model.n_estimators} boosted trees.")

    # ---- EVALUATION -------------------------------------------------------
    # predict_proba returns P(discontinued) as a float in [0, 1]. We take
    # column 1 = probability of the positive class.
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )
    accuracy = (y_pred == y_test).mean()
    baseline_accuracy = 1 - y_test.mean()

    print("\n--- TEST SET METRICS ---")
    print(f"AUC-ROC                  : {auc:.4f}")
    print(f"Average precision (PR-AUC): {ap:.4f}   (baseline = {y_test.mean():.4f})")
    print(f"Precision (discontinued) : {precision:.4f}")
    print(f"Recall    (discontinued) : {recall:.4f}")
    print(f"F1        (discontinued) : {f1:.4f}")
    print(f"Accuracy                 : {accuracy:.4f}  "
          f"<- vs {baseline_accuracy:.4f} for always-guess-False")

    # What these mean, in plain terms:
    #
    #   AUC-ROC ("area under the receiver operating characteristic curve"):
    #     take one random discontinued patient and one random retained patient;
    #     AUC is the probability the model gives the discontinued one the higher
    #     risk score. 0.5 = coin flip, 1.0 = perfect ranking. It measures
    #     RANKING quality and is unaffected by class imbalance, which is why it
    #     is the headline metric here.
    #
    #   Precision: of the patients we flagged as high risk, what fraction
    #     actually discontinued? Low precision = wasted outreach on people who
    #     were never going to stop.
    #
    #   Recall (sensitivity): of the patients who actually discontinued, what
    #     fraction did we flag? Low recall = at-risk patients we silently
    #     missed. For an intervention program this is usually the costlier
    #     error, so you would tune the threshold to favor recall.
    #
    #   Precision and recall trade off against each other via the 0.5 decision
    #     threshold. Lower the threshold -> flag more people -> recall up,
    #     precision down. The threshold is a business decision (how much
    #     outreach capacity do you have?), not a modeling one.
    #
    #   Average precision / PR-AUC: summarizes that whole tradeoff curve in one
    #     number. Unlike AUC-ROC it IS sensitive to imbalance, so compare it to
    #     the positive rate baseline printed above, not to 0.5.
    #
    #   Accuracy: the fraction of all predictions that were right. Misleading
    #     here because 79% of rows are negative -- a model that finds nobody
    #     scores 79%. Reported only for contrast.

    print("\n--- CONFUSION MATRIX (rows = actual, cols = predicted) ---")
    cm = confusion_matrix(y_test, y_pred)
    print(f"                 pred:retained  pred:discontinued")
    print(f"  actual:retained      {cm[0, 0]:5d}              {cm[0, 1]:5d}")
    print(f"  actual:discontinued  {cm[1, 0]:5d}              {cm[1, 1]:5d}")

    print("\n--- FULL CLASSIFICATION REPORT ---")
    print(classification_report(
        y_test, y_pred,
        target_names=["retained", "discontinued"],
        zero_division=0,
    ))

    # ---- SAVE THE MODEL ---------------------------------------------------
    # joblib is preferred over raw pickle for scikit-learn-style objects: same
    # format, but far more efficient with the large numpy arrays inside.
    joblib.dump(
        {
            "model": model,
            "feature_names": feature_names,
            "categorical_columns": CATEGORICAL_COLUMNS,
            # Storing the category vocabularies matters: to score a new patient
            # later you must rebuild the SAME category dtype, or the internal
            # codes shift and predictions become silently wrong.
            "category_levels": {
                c: list(X[c].cat.categories)
                for c in CATEGORICAL_COLUMNS if c in X.columns
            },
            "metrics": {
                "auc_roc": float(auc),
                "average_precision": float(ap),
                "precision": float(precision),
                "recall": float(recall),
            },
        },
        MODEL_PATH,
    )
    print(f"Saved model -> {MODEL_PATH}")

    # =======================================================================
    # PART 3 - EXPLAINABILITY (SHAP)
    # =======================================================================
    banner("PART 3 - EXPLAINABILITY (SHAP)")

    # ---- WHAT SHAP IS -----------------------------------------------------
    # SHAP = SHapley Additive exPlanations. It comes from cooperative game
    # theory: if the model's prediction is a "payout", how much did each
    # feature "contribute" to it? Shapley values are the unique way to split
    # that payout fairly, by averaging each feature's marginal contribution
    # over every possible ordering in which features could be added.
    #
    # The property that makes it useful: it is ADDITIVE, per patient.
    #
    #     base_value + sum(shap values for this patient) = model output
    #
    # So for one patient you can say "their risk is above average specifically
    # because copay contributed +1.2 and having assistance enrolled contributed
    # -0.4" -- and those numbers sum exactly to the prediction. That is a
    # per-patient explanation, which is different from (and more useful than) a
    # global "feature importance" ranking.
    #
    # TreeExplainer computes this exactly (not by sampling) for tree ensembles,
    # in polynomial time, via the TreeSHAP algorithm.
    #
    # UNITS: for binary:logistic, the model output being explained is the
    # LOG-ODDS (the raw margin), not the probability. So a SHAP value of +1.0
    # means "this feature pushed the log-odds up by 1". Log-odds are used
    # because they are additive; probabilities are not. Sign convention:
    # positive = pushes toward discontinued, negative = pushes toward retained.

    explainer = shap.TreeExplainer(model)

    # We explain ALL 2,000 patients so every patient gets an exported score.
    #
    # CAVEAT worth knowing: 1,600 of these were training rows, so their risk
    # scores are in-sample and therefore optimistic (the model has already seen
    # their outcome). The honest performance estimate is the test-set block
    # above. For production scoring you would use cross-validated out-of-fold
    # predictions; for this export we score everyone with the final model.
    # ---- MISSING-FIELD GUARD ----------------------------------------------
    # Applied BEFORE the model is called. Patients missing monthly_copay or
    # medication_name are never passed to predict_proba or to SHAP at all --
    # we refuse to score them rather than emit a number the missing-data sweep
    # proved unreliable (mean shift 69 and 53 points respectively, flipping the
    # risk tier for 77-83% of patients, with no error raised). See scoring.py.
    #
    # The guard reads the RAW dataframe row, not the encoded matrix, because
    # that is what a real request or a demo UI would actually submit.
    guard_hits = [missing_required_fields(df.iloc[i].to_dict()) for i in range(len(df))]
    eligible = np.array([len(m) == 0 for m in guard_hits])
    n_blocked = int((~eligible).sum())

    print(f"Missing-field guard: required = {list(REQUIRED_FIELDS)}")
    print(f"  {int(eligible.sum()):,} patients eligible to score, "
          f"{n_blocked:,} blocked as insufficient_data")
    if n_blocked:
        from collections import Counter
        for fields, cnt in Counter(
            ", ".join(m) for m in guard_hits if m
        ).most_common():
            print(f"    blocked on [{fields}]: {cnt}")

    X_ok = X[eligible]
    # Map original row index -> position within the eligible-only arrays.
    pos_of = {orig: k for k, orig in enumerate(np.flatnonzero(eligible))}

    shap_explanation = explainer(X_ok)
    shap_values = shap_explanation.values  # shape: (n_eligible, n_features)

    print(f"\nComputed SHAP values: {shap_values.shape[0]:,} patients "
          f"x {shap_values.shape[1]} features")
    print(f"Base value (average log-odds): {explainer.expected_value:.4f}")

    # ---- GLOBAL FEATURE IMPORTANCE ----------------------------------------
    # Mean absolute SHAP value across all patients. "How much does this feature
    # move the prediction, on average, regardless of direction?" Absolute value
    # matters because a feature that pushes risk up for some patients and down
    # for others is still important -- averaging the signed values would cancel
    # it out to ~0 and hide it.
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs_shap})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    print("\n--- GLOBAL FEATURE IMPORTANCE (mean |SHAP|, log-odds) ---")
    max_imp = importance["mean_abs_shap"].max()
    for i, row in importance.iterrows():
        bar = "#" * int(40 * row["mean_abs_shap"] / max_imp) if max_imp > 0 else ""
        print(f"  {i + 1:2d}. {row['feature']:<32} {row['mean_abs_shap']:.4f}  {bar}")

    # ---- PER-PATIENT RISK SCORES ------------------------------------------
    # risk_score = P(discontinued) * 100, rounded to an integer 0-100.
    # Only eligible patients reach the model at all -- see the guard above.
    risk_proba = model.predict_proba(X_ok)[:, 1]
    risk_score = np.round(risk_proba * 100).astype(int)

    # NOTE: because scale_pos_weight re-weighted the classes, these
    # probabilities are deliberately NOT calibrated to the true 21% base rate --
    # they are shifted upward. That is fine and expected for a RANKING /
    # triage score, which is what a risk tier is. Do not read a risk_score of
    # 60 as "60% chance this patient discontinues". Read it as "this patient
    # ranks high". If you ever need true probabilities, calibrate separately
    # (e.g. sklearn's CalibratedClassifierCV on a held-out set).

    def tier(score: int) -> str:
        if score >= TIER_HIGH:
            return "high"
        if score >= TIER_MEDIUM:
            return "medium"
        return "low"

    print(f"\nRisk score distribution:")
    print(f"  min {risk_score.min()}  median {int(np.median(risk_score))}  "
          f"mean {risk_score.mean():.1f}  max {risk_score.max()}")
    tiers = pd.Series([tier(s) for s in risk_score])
    for t in ["high", "medium", "low"]:
        n = int((tiers == t).sum())
        print(f"  {t:<7}: {n:5,} patients ({n / len(tiers):.1%})")

    # =======================================================================
    # PART 4 - EXPORT
    # =======================================================================
    banner("PART 4 - EXPORT")

    # For each patient, rank features by ABSOLUTE SHAP value and take the top 3.
    # Absolute value, because the strongest protective factor is as explanatory
    # as the strongest risk factor -- we keep the signed value in `impact` so
    # direction is preserved in the output.
    # Guarded patients get a refusal record instead: risk_score null and
    # risk_tier "insufficient_data". Both kinds carry the same keys so a
    # dashboard can render one list without special-casing the schema.
    records = []
    for i in range(len(df)):
        raw = df.iloc[i].to_dict()

        if guard_hits[i]:
            records.append(insufficient_data_record(raw, guard_hits[i]))
            continue

        k = pos_of[i]
        row_shap = shap_values[k]
        top_idx = np.argsort(np.abs(row_shap))[::-1][:3]

        records.append({
            "patient_id": str(raw["patient_id"]),
            "age": int(raw["age"]),
            "condition": str(raw["condition"]),
            "medication_name": str(raw["medication_name"]),
            "risk_score": int(risk_score[k]),
            "risk_tier": tier(int(risk_score[k])),
            "reason": None,
            "top_factors": [
                {
                    "factor": feature_names[j],
                    "impact": round(float(row_shap[j]), 4),
                }
                for j in top_idx
            ],
        })

    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    size_kb = os.path.getsize(JSON_PATH) / 1024
    print(f"Wrote {len(records):,} patient records -> {JSON_PATH} ({size_kb:.1f} KB)")
    print(f"  scored: {sum(1 for r in records if r['risk_score'] is not None):,}  |  "
          f"insufficient_data: "
          f"{sum(1 for r in records if r['risk_score'] is None):,}")

    # ---- HIGH-RISK FILTER (dashboard feed) --------------------------------
    # Second file containing only score >= 70, sorted descending. At this
    # cutoff precision is 79.1% and recall 63.1% on the held-out test set --
    # about 4 in 5 flagged patients genuinely discontinue, but the cutoff
    # misses ~37% of those who do. filter_high_risk() carries the full caveat.
    high_risk = filter_high_risk(records, min_risk=TIER_HIGH)
    with open(HIGH_RISK_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(high_risk, f, indent=2)
    hr_kb = os.path.getsize(HIGH_RISK_JSON_PATH) / 1024
    print(f"Wrote {len(high_risk):,} high-risk patients (score >= {TIER_HIGH}) "
          f"-> {HIGH_RISK_JSON_PATH} ({hr_kb:.1f} KB)")
    if high_risk:
        print(f"  top score {high_risk[0]['risk_score']}, "
              f"lowest in file {high_risk[-1]['risk_score']}")

    # =======================================================================
    # PART 5 - VALIDATION / SANITY CHECKS
    # =======================================================================
    banner("PART 5 - VALIDATION")

    scored = pd.DataFrame({
        "patient_id": [r["patient_id"] for r in records],
        "risk_score": [r["risk_score"] for r in records],
        "risk_tier": [r["risk_tier"] for r in records],
        "top_factor_names": [[f["factor"] for f in r["top_factors"]] for r in records],
    })
    scored["discontinued"] = df[TARGET].values
    scored["abandoned"] = df["abandoned"].values
    scored["reason"] = df["primary_discontinue_reason"].values

    # Guarded patients have a null score, so they cannot participate in any
    # numeric check below. Drop them here rather than letting None propagate
    # into a mean() and silently produce a wrong answer.
    n_guarded = int(scored["risk_score"].isna().sum())
    if n_guarded:
        print(f"Excluding {n_guarded:,} insufficient_data patients "
              f"from the numeric checks below.\n")
        scored = scored[scored["risk_score"].notna()].copy()
    scored["risk_score"] = scored["risk_score"].astype(int)

    # ---- CHECK 1: do SHAP factors line up with the stated reason? ---------
    # Map each ground-truth reason to the features that would plausibly explain
    # it. This is a soft sanity check, not a metric -- the model never saw the
    # reason column, so any alignment above chance is a good sign that it
    # latched onto real structure rather than noise.
    REASON_TO_FEATURES = {
        "cost": {
            "monthly_copay", "copay_assistance_enrolled",
            "copay_assistance_available", "deductible_met", "insurance_type",
        },
        "access_logistics": {
            "distance_to_pharmacy_miles", "delivery_option_available",
            "pa_turnaround_days", "prior_auth_required", "days_supply",
        },
        "side_effects": {
            "concurrent_medication_count", "drug_class", "medication_name",
            "condition",
        },
        "perceived_ineffective": {
            "appointment_length_minutes", "scheduled_visits_count",
            "missed_visits_count", "condition", "medication_name",
        },
    }

    print("--- CHECK 1: SHAP top-factor vs stated discontinuation reason ---")
    print("(model never saw the reason column; 'baseline' = how often those same")
    print(" features land in the top 3 for a RANDOM patient, so lift > 1.0 means")
    print(" the alignment is better than chance)\n")

    disc = scored[scored["discontinued"] == True]  # noqa: E712
    print(f"{'reason':<24} {'n':>5} {'match':>8} {'baseline':>9} {'lift':>7}")
    print("-" * 58)
    for reason, feats in REASON_TO_FEATURES.items():
        subset = disc[disc["reason"] == reason]
        if len(subset) == 0:
            continue
        hits = subset["top_factor_names"].apply(lambda names: bool(feats & set(names)))
        match_rate = hits.mean()
        # Baseline: same feature set, evaluated over ALL patients.
        base = scored["top_factor_names"].apply(
            lambda names: bool(feats & set(names))
        ).mean()
        lift = match_rate / base if base > 0 else float("nan")
        print(f"{reason:<24} {len(subset):>5} {match_rate:>7.1%} "
              f"{base:>8.1%} {lift:>6.2f}x")

    print("\nNOTE: `cost` is 76% of all discontinuations in this dataset, so the")
    print("model is heavily incentivized to learn cost signals. A high match")
    print("rate there is expected; the lift column is the more honest read.")

    # ---- CHECK 2: risk score for abandoned vs non-abandoned ---------------
    # `abandoned` is the narrower, more severe subset of `discontinued`. If the
    # risk score captures severity at all, abandoned patients should average
    # higher. This is an expectation, not a requirement -- the model was never
    # trained to distinguish these two groups.
    print("\n--- CHECK 2: risk_score among discontinued, by `abandoned` ---")
    ab_true = disc[disc["abandoned"] == True]["risk_score"]   # noqa: E712
    ab_false = disc[disc["abandoned"] == False]["risk_score"]  # noqa: E712
    print(f"  abandoned=True   n={len(ab_true):4d}  mean risk = {ab_true.mean():.1f}  "
          f"median = {ab_true.median():.0f}")
    print(f"  abandoned=False  n={len(ab_false):4d}  mean risk = {ab_false.mean():.1f}  "
          f"median = {ab_false.median():.0f}")
    delta = ab_true.mean() - ab_false.mean()
    direction = "HIGHER" if delta > 0 else "LOWER"
    print(f"  -> abandoned patients score {abs(delta):.1f} points {direction} on average")
    print(f"     (expected higher, since `abandoned` marks the more severe subset)")

    # ---- CHECK 3: no nulls in the output ----------------------------------
    print("\n--- CHECK 3: null / completeness check on patients_scored.json ---")
    with open(JSON_PATH, encoding="utf-8") as f:
        reloaded = json.load(f)

    # "No nulls" now means: no UNEXPECTED nulls. An insufficient_data record is
    # null by design -- that is the guard working, not a defect -- so it is held
    # to a different contract: it must have a null score, the right tier, and a
    # reason naming the field(s) that were missing.
    problems = []
    required = ["patient_id", "age", "condition", "medication_name",
                "risk_score", "risk_tier", "top_factors"]
    n_guard_recs = 0
    for rec in reloaded:
        pid = rec.get("patient_id", "?")

        if rec.get("risk_tier") == "insufficient_data":
            n_guard_recs += 1
            if rec.get("risk_score") is not None:
                problems.append(f"{pid}: insufficient_data but risk_score is not null")
            if not rec.get("reason"):
                problems.append(f"{pid}: insufficient_data without a reason")
            elif not any(fld in rec["reason"] for fld in REQUIRED_FIELDS):
                problems.append(f"{pid}: reason does not name a required field")
            continue

        for key in required:
            if key not in rec or rec[key] is None:
                problems.append(f"{pid}: missing/null {key}")
        if len(rec.get("top_factors", [])) != 3:
            problems.append(f"{pid}: "
                            f"{len(rec.get('top_factors', []))} top_factors, expected 3")
        for fac in rec.get("top_factors", []):
            if fac.get("factor") is None or fac.get("impact") is None:
                problems.append(f"{rec.get('patient_id', '?')}: null inside top_factors")
            if fac.get("impact") is not None and not np.isfinite(fac["impact"]):
                problems.append(f"{rec.get('patient_id', '?')}: non-finite impact")

    if problems:
        print(f"  FAIL - {len(problems)} problems found:")
        for p in problems[:10]:
            print(f"    {p}")
    else:
        print(f"  PASS - {len(reloaded):,} records: "
              f"{len(reloaded) - n_guard_recs:,} scored with no nulls and exactly 3 "
              f"top_factors, {n_guard_recs:,} insufficient_data correctly formed")

    # Also confirm we exported exactly one row per patient, with no dupes.
    ids = [r["patient_id"] for r in reloaded]
    print(f"  PASS - {len(set(ids)):,} unique patient_ids "
          f"({'no duplicates' if len(set(ids)) == len(ids) else 'DUPLICATES FOUND'})")

    # ---- CHECK 4: three example records -----------------------------------
    # Show one from each tier where possible, so the examples are informative
    # rather than three near-identical low-risk rows.
    print("\n--- CHECK 4: example output records ---")
    examples = []
    for t in ["high", "medium", "low"]:
        match = [r for r in reloaded if r["risk_tier"] == t]
        if match:
            examples.append(match[0])
    for rec in examples[:3]:
        print(json.dumps(rec, indent=2))

    banner("DONE")
    print(f"  model  -> {MODEL_PATH}")
    print(f"  scores -> {JSON_PATH}")


if __name__ == "__main__":
    main()
