"""REST API for churn prediction.

Run locally:  uvicorn api.main:app --reload
Docs:         http://localhost:8000/docs
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from api.schemas import BatchRequest, BatchResponse, Customer, ModelInfo, Prediction
from churn import config
from churn.predict import ChurnModel, risk_level

MODEL_PATH = Path(os.getenv("MODEL_PATH", config.MODEL_PATH))
state: dict[str, ChurnModel] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # The model is loaded once at startup, not on every request.
    if MODEL_PATH.exists():
        state["model"] = ChurnModel.load(MODEL_PATH)
    yield
    state.clear()


app = FastAPI(
    title="Churn Prediction API",
    description="Telecom customer churn probability with SHAP explanations.",
    version="0.1.0",
    lifespan=lifespan,
)


def get_model() -> ChurnModel:
    if "model" not in state:
        raise HTTPException(503, "Model is not loaded. Run `make train` first.")
    return state["model"]


def _predict(customers: list[Customer], explain: bool) -> list[Prediction]:
    model = get_model()
    X = pd.DataFrame([c.model_dump() for c in customers])
    proba = model.predict_proba(X).to_numpy()
    factors = model.explainer.top_factors(X, k=5) if explain else [None] * len(X)
    return [
        Prediction(
            churn_probability=round(float(p), 4),
            will_churn=bool(p >= model.threshold),
            risk_level=risk_level(float(p), model.threshold),
            threshold=model.threshold,
            top_factors=f,
        )
        for p, f in zip(proba, factors, strict=True)
    ]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": "model" in state}


@app.get("/model", response_model=ModelInfo)
def model_info() -> ModelInfo:
    model = get_model()
    return ModelInfo(threshold=model.threshold, **model.metadata)


@app.post("/predict", response_model=Prediction)
def predict(
    customer: Customer,
    explain: bool = Query(True, description="Return the top-5 SHAP factors"),
) -> Prediction:
    return _predict([customer], explain)[0]


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(
    request: BatchRequest,
    explain: bool = Query(False, description="SHAP for every customer (slower)"),
) -> BatchResponse:
    return BatchResponse(predictions=_predict(request.customers, explain))
