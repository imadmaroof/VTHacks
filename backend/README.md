# Backend (FastAPI)

Live prediction API. Wraps the repo's existing model/scoring/explanation code
(`score_upload.py`, `model_adapter.py`, `explanation_layer.py`, `model.pkl` --
all at the repo root, unchanged) behind two HTTP endpoints, so the website can
get a real-time prediction instead of waiting on a GitHub Actions run.

```
Website --(POST /predict, multipart CSV)--> FastAPI --> score_upload.py
  (existing model + SHAP) --> model_adapter.py + explanation_layer.py
  (existing explanation/suggestion logic) --> JSON --> Website
```

Nothing here retrains the model or duplicates its logic -- `model_service.py`
only imports and calls the existing functions.

## Setup

From the repo root:

```powershell
cd backend
python -m venv ..\venv        # skip if you already have a venv at the repo root
..\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

(`requirements.txt` here pulls in the repo root's `requirements.txt` too, so
this installs everything the model/scoring code needs plus FastAPI/uvicorn.)

## Run

```powershell
uvicorn main:app --reload --port 8000
```

The model loads once at startup (`warm_up()` in `main.py`'s startup hook), not
per-request. First request should still be fast; if `model.pkl` is missing
you'll see it fail at startup rather than on the first upload.

## Endpoints

### `GET /health`
```json
{"status": "ok"}
```

### `POST /predict`
`multipart/form-data` with a single field `file` holding a `.csv`.

```
curl -F "file=@../test_upload_sample.csv" http://localhost:8000/predict
```

Expected CSV columns: `patient_id` plus every feature the model was trained
on (see `train_adherence_model.py`'s `feature_names`/`CATEGORICAL_COLUMNS`) --
things like `age`, `condition`, `medication_name`, `monthly_copay`,
`insurance_type`, `prior_auth_required`, etc. A row missing `monthly_copay` or
`medication_name` isn't rejected -- it comes back with `risk_tier:
"insufficient_data"` instead of a score (see `scoring.py`'s guard).

Response (trimmed):
```json
{
  "status": "success",
  "total_patients": 25,
  "scored_count": 22,
  "insufficient_data_count": 3,
  "high_risk_count": 5,
  "results": [
    {
      "patient_id": "SYN-00042",
      "age": 61,
      "condition": "Psoriasis",
      "medication_name": "Cutinex",
      "risk_score": 84,
      "risk_tier": "high",
      "reason": null,
      "top_factors": [{"factor": "monthly_copay", "impact": 3.9}],
      "risk_drivers": [{"factor": "monthly_copay", "value": 265, "impact": "high"}],
      "explanation_category": "cost",
      "explanation": "This patient has an 84% risk of not filling this prescription. The biggest driver is monthly copay (265), pointing to a cost barrier.",
      "recommendations": ["Enroll the patient in the manufacturer's copay assistance program before they leave the visit, or check for a lower-cost formulary alternative."],
      "talking_points": []
    }
  ],
  "high_risk_patients": [ /* same shape, filtered to risk_score >= 70, sorted descending */ ]
}
```

Field notes:
- `risk_score` is 0-100 (a ranking score, deliberately not a calibrated
  probability -- see `train_adherence_model.py`), `null` for
  `insufficient_data` rows.
- `top_factors` is the raw model/SHAP output (every factor, signed impact).
  `risk_drivers` is the subset that actually raises risk, with the patient's
  real value and a high/medium/low bucket -- what `explanation` is grounded in.
- `recommendations`/`explanation`/`talking_points` come from
  `explanation_layer.py`. `talking_points` is only non-empty if
  `GEMINI_API_KEY` is set in the environment; otherwise the explanation still
  works, it's just template-based (see that file's docstring).

Errors: a malformed/empty/wrong-shaped CSV returns `422` with a `detail`
message naming the problem (missing columns, unreadable file, etc.), not a
crash. An upload endpoint hit with a non-CSV file returns `400`.

### `GET /patient-options`

Form spec for `docs/patient-simulator.html` (the "customizable patient" demo
page): exact dropdown values, condition-filtered medication lists, numeric
slider ranges, and a pre-filled default patient. See `patient_options.py` at
the repo root for what decides a field's category. `categorical` values come
straight from `model.pkl`'s saved vocabulary, so a retrain with different
categories is picked up automatically; the condition/medication pairings and
numeric ranges are policy constants derived once from the training CSV.

```json
{
  "categorical": {
    "condition": ["Acute Bacterial Infection", "...", "Type 2 Diabetes"],
    "medication_name": ["Airoflex", "...", "Vaslorin"],
    "...": "..."
  },
  "condition_to_drug_class": {"Type 2 Diabetes": "chronic", "...": "..."},
  "condition_to_medications": {"Type 2 Diabetes": ["Glucotrol-X", "Insunova", "Metforal"], "...": "..."},
  "boolean_fields": ["deductible_met", "..."],
  "discrete_numeric_options": {"days_supply": [30, 60, 90], "appointment_length_minutes": [10, 15, 20, 30, 45]},
  "numeric_ranges": {"age": [18, 89], "monthly_copay": [0, 500], "...": "..."},
  "default_patient": {"age": 54, "condition": "Hypertension", "...": "..."}
}
```

### `POST /predict-single`

The interactive sibling of `/predict`: score exactly one hypothetical
patient built live from form fields, instead of uploading a CSV. Same model,
same two-stage guard, same SHAP explanation -- `score_single_patient()` in
`score_upload.py` shares the same `_build_records()` core that
`score_patient_upload()` uses, so a patient scored through the form and the
same patient scored via CSV upload always get the identical number.

Body: JSON, one field per feature (all optional -- see `SinglePatientRequest`
in `schemas.py`; a key you omit is treated as null, same as a blank form
field). Values must come from `/patient-options`' vocabulary for categorical
fields -- an unrecognized value (e.g. a medication never in the training
data) is not rejected at the HTTP layer, it comes back as
`risk_tier: "insufficient_data"`, exactly like the batch guard.

```
curl -X POST http://localhost:8000/predict-single \
  -H "Content-Type: application/json" \
  -d '{"patient_id":"DEMO-1","age":54,"condition":"Hypertension","drug_class":"chronic","medication_name":"Vaslorin","new_rx_or_refill":"Refill","insurance_type":"Commercial","monthly_copay":36,"deductible_met":true,"copay_assistance_available":false,"copay_assistance_enrolled":false,"prior_auth_required":false,"pa_turnaround_days":0,"days_supply":30,"distance_to_pharmacy_miles":6,"delivery_option_available":true,"concurrent_medication_count":2,"appointment_length_minutes":10,"day_of_week_prescribed":"Wed","prior_abandoned_rx_count":0,"scheduled_visits_count":3,"missed_visits_count":0}'
```

Response:
```json
{
  "status": "ok",
  "error": null,
  "patient": {
    "patient_id": "DEMO-1", "age": 54, "condition": "Hypertension",
    "medication_name": "Vaslorin", "risk_score": 1, "risk_tier": "low",
    "reason": null, "top_factors": [ "..." ], "risk_drivers": [ "..." ],
    "explanation_category": "low_risk",
    "explanation": "This patient has a low 1% risk of not filling this prescription.",
    "recommendations": ["No action needed."], "talking_points": []
  }
}
```

Raising `monthly_copay` to `500` on that same patient moves this to
`risk_score: 83, risk_tier: "high"` -- this is the live "drag the copay
slider" demo moment `docs/patient-simulator.html` is built around.

## CORS

`main.py` allows `http://localhost:5500`, `http://127.0.0.1:5500`, and `null`
(the origin a browser sends for a page opened as a local `file://`, which is
how `docs/index.html` can still be opened directly). If you serve the
frontend from a different port, add it to `ALLOWED_ORIGINS` in `main.py`.

## Environment variables

Create a `backend/.env` file (already covered by `.gitignore`'s `.env`
pattern -- never commit it) to set these locally:

```
GEMINI_API_KEY=your-key-here
```

Then start uvicorn with `--env-file .env` so it's actually loaded:

```powershell
uvicorn main:app --reload --port 8000 --env-file .env
```

- `GEMINI_API_KEY` (optional) -- enables LLM-synthesized explanations +
  doctor talking points in `explanation_layer.py`. Unset by default; falls
  back to the rule-based template with no error. Requires the `google-genai`
  package (in `requirements.txt`) -- confirmed working end-to-end in testing.
- `USE_LLM_EXPLANATIONS=false` -- force the rule-based template even if
  `GEMINI_API_KEY` is set.
