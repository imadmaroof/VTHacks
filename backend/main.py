"""
FastAPI backend for live patient-risk predictions.

Website -> POST /predict (multipart CSV) -> this file -> model_service.py
-> score_upload.py (existing model + SHAP) -> model_adapter.py +
explanation_layer.py (existing explanation/suggestion logic) -> JSON.

Run from the backend/ folder:
    uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

# --- import bootstrap -------------------------------------------------------
# This file is imported two different ways and both have to work:
#
#   uvicorn main:app          -> run from inside backend/, so backend/ is
#                                already on sys.path (the local dev workflow
#                                documented in backend/README.md)
#   uvicorn backend.main:app  -> run from the repo root, which is what Vercel
#                                does via pyproject.toml's [tool.vercel]
#                                entrypoint. In THIS case backend/ is NOT on
#                                sys.path, so `from model_service import ...`
#                                below would raise ModuleNotFoundError.
#
# Adding backend/ explicitly keeps the flat imports working under both, rather
# than switching to relative imports, which would break `uvicorn main:app`.
_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from fastapi import FastAPI, File, HTTPException, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from model_service import (  # noqa: E402
    get_patient_field_options,
    predict_patient_data,
    predict_single_patient,
    warm_up,
)
from schemas import (  # noqa: E402
    FieldOptionsResponse,
    HealthResponse,
    PredictResponse,
    SinglePatientRequest,
    SinglePatientResponse,
)

app = FastAPI(title="Medication Adherence Risk API", version="1.0.0")

# Local-dev origins for docs/index.html. "null" covers opening the file
# directly (file://), which is how the front end originally ran. Add your
# static-server origin here if you serve docs/ from something other than
# port 5500 (see backend/README.md).
ALLOWED_ORIGINS = [
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "null",
]

# On Vercel the front end is served from a *.vercel.app domain (and possibly
# the custom domain in docs/CNAME), not localhost -- without this the browser
# would block every call to this API with a CORS error. Preview deploys get a
# fresh random subdomain each time, so match the whole vercel.app space by
# pattern rather than listing them.
ALLOWED_ORIGIN_REGEX = r"https://.*\.vercel\.app|https://(www\.)?impiricusupload\.com"

# Extra origins can be added at deploy time without a code change:
#   ALLOWED_ORIGINS=https://foo.com,https://bar.com
_extra = os.environ.get("ALLOWED_ORIGINS", "").strip()
if _extra:
    ALLOWED_ORIGINS += [o.strip() for o in _extra.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=ALLOWED_ORIGIN_REGEX,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _load_model() -> None:
    # Load model.pkl once, now, so the first real request isn't the one that
    # pays for it.
    warm_up()


@app.get("/health", response_model=HealthResponse)
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)) -> dict:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file.")

    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        result = predict_patient_data(tmp_path)
    except Exception as exc:  # noqa: BLE001 - never crash the server on bad input
        raise HTTPException(status_code=500, detail=f"Scoring failed: {exc}") from exc
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)

    if result["status"] == "error":
        raise HTTPException(status_code=422, detail=result["error"])

    return {
        "status": "success",
        "total_patients": result["total_patients"],
        "scored_count": result["scored_count"],
        "insufficient_data_count": result["insufficient_data_count"],
        "high_risk_count": result["high_risk_count"],
        "results": result["all_patients"],
        "high_risk_patients": result["high_risk_patients"],
    }


@app.get("/patient-options", response_model=FieldOptionsResponse)
def patient_options() -> dict:
    """Form spec for the "customizable patient" demo page: exact dropdown
    values (from the trained model), condition-filtered medication lists,
    numeric ranges, and a default patient to pre-fill the form with."""
    return get_patient_field_options()


@app.post("/predict-single", response_model=SinglePatientResponse)
def predict_single(patient: SinglePatientRequest) -> dict:
    """Score one hypothetical patient built live from form fields -- the
    interactive sibling of /predict. Same model, same guard, same SHAP
    explanation, just one in-memory patient instead of an uploaded CSV."""
    try:
        result = predict_single_patient(patient.model_dump())
    except Exception as exc:  # noqa: BLE001 - never crash the server on bad input
        raise HTTPException(status_code=500, detail=f"Scoring failed: {exc}") from exc

    if result["status"] == "error":
        raise HTTPException(status_code=422, detail=result["error"])

    return result


# ---------------------------------------------------------------------------
# Static front end
# ---------------------------------------------------------------------------
# Serve docs/ (index.html + patient-simulator.html) from this same app, so a
# Vercel deploy is ONE thing at ONE origin instead of a separate static site
# that then has to be pointed at a separate API host. Same-origin also means
# the browser never issues a cross-origin request, so CORS stops mattering for
# the deployed case entirely (the settings above still cover local file://
# and live-server use).
#
# Mounted LAST on purpose: StaticFiles at "/" is a catch-all, so mounting it
# before the routes above would shadow /health, /predict, /patient-options and
# /predict-single. Guarded by exists() so the API still boots if docs/ is
# absent (e.g. a backend-only deploy).
_DOCS_DIR = _BACKEND_DIR.parent / "docs"
if _DOCS_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(_DOCS_DIR), html=True), name="frontend")
