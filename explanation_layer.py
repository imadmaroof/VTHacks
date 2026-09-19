"""
Explanation & Suggestion Layer (Person 2)

Takes a risk score + top contributing factors (as produced by Person 1's
model / SHAP output) and returns a short, grounded explanation plus one
concrete suggestion for the prescribing doctor.

Input contract (matches Person 1's expected output shape):
{
    "risk_score": 78,
    "top_factors": [
        {"factor": "monthly_copay", "value": 145, "impact": "high"},
        ...
    ]
}
"""

IMPACT_WEIGHTS = {"high": 3, "medium": 2, "low": 1}

FACTOR_CATEGORY_MAP = {
    # cost-dominant
    "monthly_copay": "cost",
    "copay_assistance_available": "cost",
    "copay_assistance_enrolled": "cost",
    "insurance_type": "cost",
    "deductible_met": "cost",
    # not an actionable barrier, but the model surfaces it; map so it doesn't warn
    "medication_name": "general",
    # access / logistics-dominant
    "prior_auth_required": "access",
    "pa_turnaround_days": "access",
    "distance_to_pharmacy_miles": "access",
    "delivery_option_available": "access",
    # side-effect-dominant
    "side_effects_reported": "side_effect",
    "concurrent_medication_count": "side_effect",
    # perceived-ineffectiveness-dominant
    "missed_visits_count": "perceived_ineffective",
    "new_rx_or_refill": "perceived_ineffective",
    "scheduled_visits_count": "perceived_ineffective",
}

FACTOR_LABELS = {
    "monthly_copay": "monthly copay",
    "copay_assistance_enrolled": "copay assistance enrollment",
    "copay_assistance_available": "copay assistance availability",
    "prior_auth_required": "prior authorization requirement",
    "pa_turnaround_days": "prior authorization turnaround time",
    "distance_to_pharmacy_miles": "distance to the nearest pharmacy",
    "delivery_option_available": "delivery option availability",
    "side_effects_reported": "reported side effects",
    "concurrent_medication_count": "number of concurrent medications",
    "missed_visits_count": "missed visit count",
    "new_rx_or_refill": "new-prescription status",
    "scheduled_visits_count": "scheduled visit count",
}

SUGGESTIONS = {
    "cost": (
        "Enroll the patient in the manufacturer's copay assistance program "
        "before they leave the visit, or check for a lower-cost formulary alternative."
    ),
    "access": (
        "Start prior authorization paperwork today to avoid delays, and flag "
        "home delivery or mail-order options if the nearest pharmacy is far away."
    ),
    "side_effect": (
        "Schedule a brief conversation about side effects experienced so far and "
        "whether a dose adjustment or alternative medication makes sense."
    ),
    "perceived_ineffective": (
        "Schedule a short check-in call within the first week to address any "
        "doubts about whether the medication is working."
    ),
    "general": (
        "Schedule a brief follow-up conversation to understand this patient's "
        "specific concerns about the prescription."
    ),
}

CATEGORY_LABELS = {
    "cost": "cost",
    "access": "access/logistics",
    "side_effect": "side-effect",
    "perceived_ineffective": "perceived-effectiveness",
    "general": "general",
}


def _label_for(factor_name):
    return FACTOR_LABELS.get(factor_name, factor_name.replace("_", " "))


def dominant_category(top_factors):
    """Pick the risk category with the highest impact-weighted score.
    Ties go to whichever category the first-listed factor belongs to."""
    if not top_factors:
        return "general"

    scores = {}
    for factor in top_factors:
        if factor["factor"] not in FACTOR_CATEGORY_MAP:
            print(f"[explanation_layer] unmapped factor name: '{factor['factor']}' -> defaulting to 'general'")
        category = FACTOR_CATEGORY_MAP.get(factor["factor"], "general")
        weight = IMPACT_WEIGHTS.get(factor.get("impact"), 1)
        scores[category] = scores.get(category, 0) + weight

    best_score = max(scores.values())
    # first-listed factor's category wins ties
    for factor in top_factors:
        category = FACTOR_CATEGORY_MAP.get(factor["factor"], "general")
        if scores[category] == best_score:
            return category
    return "general"


def _top_factor_in_category(top_factors, category):
    """The highest-impact factor that belongs to the winning category,
    used to ground the explanation in a real value."""
    for factor in top_factors:
        if FACTOR_CATEGORY_MAP.get(factor["factor"], "general") == category:
            return factor
    return top_factors[0] if top_factors else None


def explain_scored_record(record):
    """End-to-end: take one raw model record (teammate's output shape) and
    return the doctor-facing explanation. Handles the insufficient-data case."""
    from model_adapter import adapt_record

    adapted = adapt_record(record)
    if not adapted["scorable"]:
        return {
            "risk_score": None,
            "category": "insufficient_data",
            "explanation": "This patient cannot be scored yet. " + adapted["reason"],
            "suggestion": "Complete the missing field(s) before relying on a risk score.",
        }

    # Only high/medium-risk patients need an intervention. Low-risk patients
    # shouldn't clutter the doctor's queue with explanations. Cutoff matches
    # scoring.py's TIER_MEDIUM.
    LOW_RISK_CUTOFF = 40
    if adapted["risk_score"] < LOW_RISK_CUTOFF:
        return {
            "risk_score": adapted["risk_score"],
            "category": "low_risk",
            "explanation": f"This patient has a low {adapted['risk_score']}% risk of not filling this prescription.",
            "suggestion": "No action needed.",
        }

    return generate_explanation(adapted["risk_score"], adapted["top_factors"])


def generate_explanation(risk_score, top_factors):
    category = dominant_category(top_factors)
    primary = _top_factor_in_category(top_factors, category)

    if primary is None:
        explanation = f"This patient has a {risk_score}% risk of not filling this prescription."
    else:
        label = _label_for(primary["factor"])
        explanation = (
            f"This patient has a {risk_score}% risk of not filling this prescription. "
            f"The biggest driver is {label} ({primary['value']}), pointing to a "
            f"{CATEGORY_LABELS[category]} barrier."
        )

    return {
        "risk_score": risk_score,
        "category": category,
        "explanation": explanation,
        "suggestion": SUGGESTIONS[category],
    }
