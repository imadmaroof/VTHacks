"""
Field metadata for the interactive "customizable patient" demo form.

Single source of truth for which patient fields are dropdowns, which are
free-form numeric ranges, and the sensible bounds/defaults for each -- so the
frontend form and the backend agree on exactly what a judge is allowed to
submit live on stage.

THE RULE THIS MODULE ENCODES
-----------------------------
  * A field where the model learned a FIXED set of values during training
    (a "category") must be a dropdown. Free text would let someone type an
    unrecognized value (e.g. "Tylenol", never in the training data) and
    silently trigger the insufficient_data guard mid-demo -- see
    score_upload.py's module docstring for exactly how that happens.
  * A boolean field is a Yes/No dropdown, not free text -- there is no
    "range" to speak of.
  * A genuine continuous number is free-form, but bounded to a realistic
    range so nobody types age=9999 and gets a nonsense result.
  * A number that only ever took a handful of real values in training
    (days_supply, appointment_length_minutes) is ALSO a dropdown, even
    though it is technically numeric -- letting someone type days_supply=47
    produces a shape the model never saw, not unsafe, just unrealistic.

WHY THIS IS SEPARATE FROM model.pkl
------------------------------------
model.pkl's category_levels already gives the exact allowed values for each
categorical column, loaded live in build_field_options() below so it can
never drift from the trained model. What model.pkl does NOT capture is which
VALUE COMBINATIONS are realistic: every medication_name is a valid category
dtype level, but "Rheumatex" should not be selectable while
condition="Type 2 Diabetes" is chosen -- that pairing never existed in
training. Both would pass the guard (they're recognized values) but produce
a meaningless score, so the mapping below restricts the medication dropdown
to the drugs that actually occur under each selected condition.

These mappings and ranges were derived once from
`Synthetic Patients Processed.csv` (grouped by condition; min/max/median per
numeric column) and are fixed here as plain policy, not recomputed at
runtime, so a retrain doesn't silently change what a slider goes up to
mid-demo. Regenerate them the same way if the training data changes.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# condition -> drug_class
# df.groupby('condition')['drug_class'].unique() -- exactly one drug_class
# per condition in the training data (verified). Not user-editable; the form
# should auto-fill and disable this once a condition is chosen, so the pairing
# stays visible without being a second independent choice.
# ---------------------------------------------------------------------------
CONDITION_TO_DRUG_CLASS: dict[str, str] = {
    "Acute Bacterial Infection": "acute",
    "Asthma": "chronic",
    "Hyperlipidemia": "chronic",
    "Hypertension": "chronic",
    "Major Depressive Disorder": "chronic",
    "Multiple Sclerosis": "specialty",
    "Post-Surgical Pain": "acute",
    "Psoriasis": "specialty",
    "Rheumatoid Arthritis": "specialty",
    "Type 2 Diabetes": "chronic",
}

# condition -> medication_name
# df.groupby('condition')['medication_name'].unique() -- restricts the
# medication dropdown to drugs that actually occur for the selected
# condition.
CONDITION_TO_MEDICATIONS: dict[str, list[str]] = {
    "Acute Bacterial Infection": ["Amoxinol", "Cephadex"],
    "Asthma": ["Airoflex", "Bronchara"],
    "Hyperlipidemia": ["Cholestara", "Lipatrex"],
    "Hypertension": ["Cardiozem", "Presolol", "Vaslorin"],
    "Major Depressive Disorder": ["Moodrex", "Serenol"],
    "Multiple Sclerosis": ["Myelara", "Neurovex"],
    "Post-Surgical Pain": ["Analgex", "Recuparin"],
    "Psoriasis": ["Cutinex", "Dermalis"],
    "Rheumatoid Arthritis": ["Immunara", "Rheumatex"],
    "Type 2 Diabetes": ["Glucotrol-X", "Insunova", "Metforal"],
}

# Boolean-valued features: rendered as Yes/No dropdowns, never free text.
BOOLEAN_FIELDS: list[str] = [
    "deductible_met",
    "copay_assistance_available",
    "copay_assistance_enrolled",
    "prior_auth_required",
    "delivery_option_available",
]

# Numeric fields that only ever took a handful of real values in training --
# treated as dropdowns rather than free-form sliders/inputs.
DISCRETE_NUMERIC_OPTIONS: dict[str, list[int]] = {
    "days_supply": [30, 60, 90],
    "appointment_length_minutes": [10, 15, 20, 30, 45],
}

# Genuine continuous numeric fields: free-form, bounded to a realistic
# [min, max] the form should use as slider bounds.
NUMERIC_RANGES: dict[str, list[float]] = {
    "age": [18, 89],
    # Training data goes up to ~$1,214 for rare specialty-drug outliers, but
    # $0-500 covers the realistic demo range without a slider that looks
    # absurd on stage.
    "monthly_copay": [0, 500],
    # Only ever nonzero when prior_auth_required is True in training data
    # (always 0 otherwise) -- the form should disable/zero this field when
    # prior_auth_required is "No".
    "pa_turnaround_days": [0, 21],
    "distance_to_pharmacy_miles": [0, 30],
    "concurrent_medication_count": [0, 9],
    "prior_abandoned_rx_count": [0, 4],
    "scheduled_visits_count": [1, 5],
    # Upper bound here is the field's own max; the form should further clamp
    # this at whatever scheduled_visits_count is currently set to, since you
    # cannot miss more visits than were scheduled.
    "missed_visits_count": [0, 4],
}

# A realistic median/mode patient (derived from the same training CSV) so the
# form never starts blank. `patient_id` is intentionally absent -- the caller
# assigns an identity for a hypothetical patient, it is not a scored feature.
DEFAULT_PATIENT: dict[str, Any] = {
    "age": 54,
    "condition": "Hypertension",
    "drug_class": "chronic",
    "medication_name": "Vaslorin",
    "new_rx_or_refill": "Refill",
    "insurance_type": "Commercial",
    "monthly_copay": 36,
    "deductible_met": True,
    "copay_assistance_available": False,
    "copay_assistance_enrolled": False,
    "prior_auth_required": False,
    "pa_turnaround_days": 0,
    "days_supply": 30,
    "distance_to_pharmacy_miles": 6,
    "delivery_option_available": True,
    "concurrent_medication_count": 2,
    "appointment_length_minutes": 10,
    "day_of_week_prescribed": "Wed",
    "prior_abandoned_rx_count": 0,
    "scheduled_visits_count": 3,
    "missed_visits_count": 0,
}


def build_field_options() -> dict[str, Any]:
    """
    Assemble the full form spec for the frontend in one call, so the
    dropdown lists (and their bounds) can never drift from what this module
    -- and, for category_levels, the trained model itself -- actually allow.

    `categorical` values (condition, drug_class, medication_name,
    new_rx_or_refill, insurance_type, day_of_week_prescribed) come straight
    from model.pkl's saved category_levels, not duplicated here, so a
    retrain with different categories is picked up automatically.
    """
    # Imported lazily so this module can be imported (and its static policy
    # tables above tested) without requiring model.pkl to exist yet.
    from score_upload import _load_model_bundle

    bundle = _load_model_bundle()
    category_levels = bundle["category_levels"]

    return {
        "categorical": {
            col: list(levels) for col, levels in category_levels.items()
        },
        "condition_to_drug_class": CONDITION_TO_DRUG_CLASS,
        "condition_to_medications": CONDITION_TO_MEDICATIONS,
        "boolean_fields": BOOLEAN_FIELDS,
        "discrete_numeric_options": DISCRETE_NUMERIC_OPTIONS,
        "numeric_ranges": NUMERIC_RANGES,
        "default_patient": DEFAULT_PATIENT,
    }
