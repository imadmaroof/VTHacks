"""
FastAPI backend for live patient-risk predictions.

Website -> POST /predict (multipart CSV) -> this file -> model_service.py
-> score_upload.py (existing model + SHAP) -> model_adapter.py +
explanation_layer.py (existing explanation/suggestion logic) -> JSON.

Run from the backend/ folder:
    uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from model_service import predict_patient_data, warm_up
from schemas import HealthResponse, PredictResponse

app = FastAPI(title="Medication Adherence Risk API", version="1.0.0")

# Local-dev origins for docs/index.html. "null" covers opening the file
# directly (file://), which is how the front end originally ran. Add your
# static-server origin here if you serve docs/ from something other than
# port 5500 (see backend/README.md).
ALLOWED_ORIGINS = [
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "null",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
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
