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

# Factors that describe the prescription rather than a barrier the doctor can
# act on. They may dominate the model's SHAP output, but "the drug is Dermalis"
# gives a doctor nothing to do, so they never decide the intervention category.
NON_ACTIONABLE_FACTORS = {"medication_name", "insurance_type"}

# Short fragments for listing factors inline. {v} is the factor's value; a
# tuple means (phrase when true, phrase when false) for boolean factors.
FACTOR_FRAGMENTS = {
    "monthly_copay": "${v}/mo copay",
    "deductible_met": ("deductible met", "deductible not met"),
    "copay_assistance_available": ("assistance available", "no assistance available"),
    "copay_assistance_enrolled": ("enrolled in copay assistance", "unused copay assistance"),
    "insurance_type": "{v}",
    "prior_auth_required": ("prior auth required", "no prior auth"),
    "pa_turnaround_days": "{v}d PA turnaround",
    "distance_to_pharmacy_miles": "{v}mi to pharmacy",
    "delivery_option_available": ("delivery available", "no delivery option"),
    "side_effects_reported": ("side effects reported", "no side effects"),
    "concurrent_medication_count": "{v} concurrent meds",
    "missed_visits_count": "{v} missed visits",
    "scheduled_visits_count": "{v} scheduled visits",
    "prior_abandoned_rx_count": "{v} prior abandonments",
    "new_rx_or_refill": "{v} prescription",
    "days_supply": "{v}-day supply",
    "appointment_length_minutes": "{v}-min appointment",
    "age": "age {v}",
    "medication_name": "{v}",
}

# How many factors to name in one explanation. Three gives the doctor a real
# picture without turning the queue into a wall of text.
MAX_LISTED_FACTORS = 3

# Boolean factors read badly as "field: True", so each gets a plain phrasing.
BOOLEAN_PHRASES = {
    "prior_auth_required": "prior authorization required",
    "deductible_met": "deductible already met",
    "copay_assistance_available": "copay assistance available",
    "copay_assistance_enrolled": "enrolled in copay assistance",
    "delivery_option_available": "delivery option available",
    "side_effects_reported": "side effects reported",
}

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
    "prior_abandoned_rx_count": "perceived_ineffective",
    "days_supply": "perceived_ineffective",
    "age": "general",
    "day_of_week_prescribed": "general",
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


def _format_currency(value):
    """'$300.72' for anything numeric, None when there is no usable figure."""
    if value is None:
        return None
    try:
        return f"${float(value):.2f}"
    except (TypeError, ValueError):
        return None


def _is_true(value):
    """True for real booleans and for the strings a CSV join produces."""
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    return bool(value)


def _join_english(items):
    """['a'] -> 'a'; ['a','b'] -> 'a and b'; ['a','b','c'] -> 'a, b, and c'."""
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _fragment(factor):
    """One short phrase for a factor, or None when there is nothing to say."""
    name = factor["factor"]
    value = factor.get("value")
    if value is None:
        return None

    template = FACTOR_FRAGMENTS.get(name)
    if template is None:
        # Unmapped factor: still show the value so the doctor sees the number
        # rather than a bare label with no information in it.
        return f"{_label_for(name)} {value}"

    if isinstance(template, tuple):
        return template[0] if _is_true(value) else template[1]

    if name == "monthly_copay":
        amount = _format_currency(value)
        return f"{amount}/mo copay" if amount else None

    return template.format(v=value)


def _grouped_factors(top_factors, context=None):
    """Top factors split into financial and non-financial phrases.

    Financial is derived from FACTOR_CATEGORY_MAP's "cost" bucket rather than a
    second mapping, so the two can never drift apart.
    """
    # A drug name is not a barrier a doctor can act on, so it should not consume
    # one of the three slots. Keep them only if nothing else is left, since an
    # explanation naming no factors at all is worse than a weak one.
    listable = [f for f in top_factors if f["factor"] not in NON_ACTIONABLE_FACTORS]
    if not listable:
        listable = top_factors

    financial, other = [], []
    for factor in listable[:MAX_LISTED_FACTORS]:
        phrase = _fragment(factor)
        if not phrase:
            continue
        if FACTOR_CATEGORY_MAP.get(factor["factor"]) == "cost":
            financial.append(phrase)
        else:
            other.append(phrase)

    # Insurance and assistance status rarely rank as top SHAP factors but are
    # what makes a cost barrier actionable, so they ride along in the financial
    # group when the model surfaced a cost factor at all.
    if financial and context:
        # Insurance qualifies the copay rather than being a reason of its own:
        # "a $300 copay on Medicare" is true, "driven by being on Medicare" is
        # not. So it is attached to the copay phrase, not listed beside it.
        insurance = context.get("insurance_type")
        if insurance:
            qualifier = (
                "while uninsured"
                if insurance.lower() == "uninsured"
                else f"on {insurance}"
            )
            already_named = any(insurance.lower() in p.lower() for p in financial)
            if not already_named:  # dedupe is case-insensitive: "Uninsured" vs "uninsured"
                for i, phrase in enumerate(financial):
                    if "copay" in phrase:
                        financial[i] = f"{phrase} {qualifier}"
                        break

        available = context.get("copay_assistance_available")
        enrolled = context.get("copay_assistance_enrolled")
        if available is not None and not any("assistance" in f for f in financial):
            if _is_true(available) and not _is_true(enrolled):
                financial.append("unused copay assistance")
            elif not _is_true(available):
                financial.append("no assistance available")

    return financial, other


def _cost_context(context):
    """Short insurance/assistance clause for a cost-driven explanation."""
    if not context:
        return ""
    bits = []
    insurance = context.get("insurance_type")
    if insurance:
        # "on Uninsured" is not English; the others read fine as "on Medicare".
        bits.append("uninsured" if insurance.lower() == "uninsured" else f"on {insurance}")

    available = context.get("copay_assistance_available")
    enrolled = context.get("copay_assistance_enrolled")
    if _is_true(available) and not _is_true(enrolled):
        bits.append("assistance available but not enrolled")
    elif not _is_true(available):
        bits.append("no assistance available")

    return ", ".join(bits)


def _label_for(factor_name):
    return FACTOR_LABELS.get(factor_name, factor_name.replace("_", " "))


def dominant_category(top_factors):
    """Pick the risk category with the highest impact-weighted score.
    Ties go to whichever category the first-listed factor belongs to."""
    if not top_factors:
        return "general"

    for factor in top_factors:
        if factor["factor"] not in FACTOR_CATEGORY_MAP:
            print(f"[explanation_layer] unmapped factor name: '{factor['factor']}' -> defaulting to 'general'")

    # Let an actionable barrier decide the category even when a descriptive
    # factor outranks it -- otherwise a patient whose top driver is the drug
    # name gets a vague "general" result while a real copay barrier sits
    # unused one row below.
    actionable = [f for f in top_factors if f["factor"] not in NON_ACTIONABLE_FACTORS]
    if actionable:
        top_factors = actionable

    # The single strongest factor decides the category. top_factors arrives
    # sorted by true SHAP magnitude, so summing weights per category would let
    # two medium factors outvote one dominant driver -- e.g. a $300 copay losing
    # to "prior auth required" plus "distance". Rank by the strongest factor
    # instead, and only sum to break ties between equally-weighted leaders.
    best_weight = max(
        IMPACT_WEIGHTS.get(f.get("impact"), 1) for f in top_factors
    )
    leaders = [
        f for f in top_factors
        if IMPACT_WEIGHTS.get(f.get("impact"), 1) == best_weight
    ]
    if len(leaders) == 1:
        return FACTOR_CATEGORY_MAP.get(leaders[0]["factor"], "general")

    # Several factors tie at the top weight: sum within those leaders only,
    # and fall back to the first-listed (highest SHAP) on a further tie.
    scores = {}
    for factor in leaders:
        category = FACTOR_CATEGORY_MAP.get(factor["factor"], "general")
        scores[category] = scores.get(category, 0) + 1

    best_score = max(scores.values())
    for factor in leaders:
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


def explain_scored_record(record, use_llm=False):
    """End-to-end: take one raw model record (teammate's output shape) and
    return the doctor-facing explanation. Handles the insufficient-data case."""
    from model_adapter import adapt_record

    # Carried through so the caller knows which patient each explanation belongs
    # to -- generate_explanation() below never sees the record, only the score.
    patient_id = record.get("patient_id")

    adapted = adapt_record(record)
    if not adapted["scorable"]:
        return {
            "patient_id": patient_id,
            "risk_score": None,
            "category": "insufficient_data",
            "explanation": "This patient cannot be scored yet. " + adapted["reason"],
            "suggestion": "Complete the missing field(s) before relying on a risk score.",
            "talking_points": [],
        }

    # Only high/medium-risk patients need an intervention. Low-risk patients
    # shouldn't clutter the doctor's queue with explanations. Cutoff matches
    # scoring.py's TIER_MEDIUM.
    LOW_RISK_CUTOFF = 40
    if adapted["risk_score"] < LOW_RISK_CUTOFF:
        return {
            "patient_id": patient_id,
            "risk_score": adapted["risk_score"],
            "category": "low_risk",
            "explanation": f"{adapted['risk_score']}% risk. No action needed.",
            "suggestion": "No action needed.",
            "talking_points": [],
        }

    return {
        "patient_id": patient_id,
        **generate_explanation(
            adapted["risk_score"], adapted["top_factors"], use_llm=use_llm, context=record
        ),
    }


def _build_prompt(risk_score, category, drivers, context=None):
    """Build the Gemini prompt. Pure function so it can be tested without
    spending an API call."""
    factor_lines = "\n".join(
        f"- {_label_for(d['factor'])}: {d['value']}" for d in drivers
    )

    extra = ""
    if category == "cost" and context:
        # Insurance and assistance status rarely rank as top SHAP factors but
        # are exactly what a doctor needs to act on a cost barrier.
        insurance = context.get("insurance_type")
        available = context.get("copay_assistance_available")
        enrolled = context.get("copay_assistance_enrolled")
        details = []
        if insurance:
            details.append(f"insurance: {insurance}")
        if available is not None:
            details.append(
                "copay assistance available"
                if _is_true(available)
                else "no copay assistance available"
            )
        if enrolled is not None and _is_true(available) and not _is_true(enrolled):
            details.append("not currently enrolled in assistance")
        if details:
            extra = "\nAlso relevant: " + "; ".join(details)

    return (
        "You support doctors at the point of prescribing. A model flagged this "
        "patient as likely to abandon their prescription.\n\n"
        f"Risk score: {risk_score}%\n"
        f"Barrier type: {category}\n"
        f"Contributing factors:\n{factor_lines}{extra}\n\n"
        "Return ONLY valid JSON, no markdown fences, with exactly these keys:\n"
        '  "explanation": ONE sentence, UNDER 30 WORDS. Name up to THREE factors '
        "from above with their values, covering both financial and "
        "non-financial reasons when the list has both. Cost alone is thin "
        "justification. Write plain English prose, no labels, no bullet "
        "points, no preamble. Start with the risk percentage.\n"
        '  "talking_points": array of 2-3 very short prompts for the doctor\'s '
        "brief conversation with the patient about this barrier.\n\n"
        "HARD RULES: Use ONLY the factors listed above. Do not invent any other "
        "reason. Do not give medical or clinical advice. Do not recommend dose "
        "changes, alternative drugs, or treatments. Talking points are "
        "conversation framing only, never instructions for care."
    )


def _gemini_output(risk_score, category, drivers, context=None):
    """Optional: ask Gemini for a concise explanation + conversation points.

    Returns {"explanation": str, "talking_points": [str]} on success, or None on
    any failure so the caller falls back to the template. Gemini sees ONLY the
    real drivers from the model and is constrained to conversation framing --
    it never picks the intervention and never gives clinical advice.
    """
    import json
    import os

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key or not drivers:
        return None

    prompt = _build_prompt(risk_score, category, drivers, context)

    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-3.6-flash", contents=prompt
        )
        raw = (response.text or "").strip()
        # Models sometimes wrap JSON in fences despite instructions.
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        parsed = json.loads(raw)
        explanation = parsed.get("explanation")
        points = parsed.get("talking_points") or []
        if not explanation:
            return None
        if not isinstance(points, list):
            points = []
        return {
            "explanation": str(explanation),
            "talking_points": [str(p) for p in points][:3],
        }
    except Exception:
        # Any failure at all -> fall back silently to the template.
        return None


def generate_explanation(risk_score, top_factors, use_llm=False, context=None):
    category = dominant_category(top_factors)
    primary = _top_factor_in_category(top_factors, category)

    explanation = None
    talking_points = []
    if use_llm:
        llm = _gemini_output(risk_score, category, top_factors, context)
        if llm:
            explanation = llm.get("explanation")
            # Cap here as well as in _gemini_output: this is the boundary the
            # response actually crosses, whatever produced the points.
            talking_points = (llm.get("talking_points") or [])[:3]

    if explanation is None:
        # Terse by design: risk_score is its own field in the response, so the
        # prose lists the barriers rather than repeating the number in a full
        # sentence. Factors are grouped because "the copay is high" alone is
        # thin justification -- a doctor wants the financial and non-financial
        # picture side by side.
        financial, other = _grouped_factors(top_factors, context)

        if financial and other:
            explanation = (
                f"{risk_score}% risk, driven by {_join_english(financial)}, "
                f"plus {_join_english(other)}."
            )
        elif financial or other:
            explanation = f"{risk_score}% risk, driven by {_join_english(financial or other)}."
        else:
            explanation = f"{risk_score}% risk, with no dominant factor identified."

    return {
        "risk_score": risk_score,
        "category": category,
        "explanation": explanation,
        "suggestion": SUGGESTIONS[category],
        "talking_points": talking_points,
    }
