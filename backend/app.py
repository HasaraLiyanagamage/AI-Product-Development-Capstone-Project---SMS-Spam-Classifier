from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Import training function to bootstrap a model if none exists
try:
    from ml_pipeline.train import train_and_save
except Exception:  # pragma: no cover
    train_and_save = None

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
MODEL_PATH = APP_DIR / "model" / "sms_spam_model.joblib"

app = FastAPI(title="SMS Spam Classifier API")


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    label: str
    probability: float


_model = None


@app.on_event("startup")
def _load_or_bootstrap_model() -> None:
    global _model
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        if train_and_save is None:
            raise RuntimeError("Model not found and training function unavailable.")
        # Train a model (downloads public dataset or uses fallback) and save it
        train_and_save(output_model_path=MODEL_PATH)

    _model = joblib.load(MODEL_PATH)


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest) -> PredictResponse:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="'text' must be a non-empty string")

    try:
        proba = _model.predict_proba([text])[0]
        classes = list(_model.classes_)
        idx = int(np.argmax(proba))
        return PredictResponse(label=str(classes[idx]), probability=float(proba[idx]))
    except AttributeError:
        # If the classifier doesn't support probabilities, fall back to hard label
        label = _model.predict([text])[0]
        return PredictResponse(label=str(label), probability=1.0)


# Serve frontend (index.html) from / if directory exists
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
else:
    @app.get("/")
    async def root() -> JSONResponse:
        return JSONResponse({"status": "ok", "message": "Frontend directory not found."})
