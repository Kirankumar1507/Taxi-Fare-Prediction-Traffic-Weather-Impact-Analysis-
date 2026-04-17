"""FastAPI application for taxi fare prediction.

Run locally:
    uvicorn deploy.server.rest_server:app --reload --port 8000
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException

from deploy.server.config import ServerConfig
from deploy.server.health import check_health
from deploy.server.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ErrorResponse,
    HealthResponse,
    PredictionResponse,
    TripInput,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level predictor — loaded once at startup via lifespan
# ---------------------------------------------------------------------------
_predictor = None


def _load_predictor() -> None:
    """Import and instantiate FarePredictor (keeps import side-effects out of module top)."""
    global _predictor
    from src.predict import FarePredictor

    _predictor = FarePredictor(ServerConfig.MODEL_NAME)
    logger.info(
        "Model loaded: %s (%s), features=%d",
        ServerConfig.MODEL_NAME,
        ServerConfig.MODEL_VERSION,
        len(_predictor.feature_names),
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup: load model.  Shutdown: nothing to clean up."""
    logging.basicConfig(level=getattr(logging, ServerConfig.LOG_LEVEL, logging.INFO))
    _load_predictor()
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Taxi Fare Prediction API",
    version="1.0.0",
    description="REST API serving a CatBoost taxi fare model (R2=0.943, MAPE=5.99%)",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Liveness + readiness probe."""
    result = check_health(_predictor)
    status_code = 200 if result["status"] == "healthy" else 503
    if status_code == 503:
        raise HTTPException(status_code=503, detail=result)
    return HealthResponse(**result)


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={422: {"model": ErrorResponse}},
    tags=["inference"],
)
def predict(trip: TripInput) -> PredictionResponse:
    """Predict fare for a single trip."""
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.perf_counter()
    try:
        fare = _predictor.predict(trip.model_dump())
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info("predict  fare=%.2f  latency_ms=%.1f", fare, elapsed_ms)

    # Optional: structured prediction log
    _maybe_log_prediction(trip.model_dump(), fare, elapsed_ms)

    return PredictionResponse(
        predicted_fare=fare,
        model_name=ServerConfig.MODEL_NAME,
        model_version=ServerConfig.MODEL_VERSION,
    )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    responses={422: {"model": ErrorResponse}},
    tags=["inference"],
)
def predict_batch(req: BatchPredictionRequest) -> BatchPredictionResponse:
    """Predict fares for up to 100 trips."""
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.perf_counter()
    try:
        import pandas as pd

        df = pd.DataFrame([t.model_dump() for t in req.trips])
        preds = _predictor.predict_batch(df)
        fares = [round(float(p), 4) for p in preds]
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "predict_batch  count=%d  latency_ms=%.1f", len(fares), elapsed_ms
    )

    return BatchPredictionResponse(
        predictions=fares,
        model_name=ServerConfig.MODEL_NAME,
        model_version=ServerConfig.MODEL_VERSION,
        count=len(fares),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _maybe_log_prediction(input_data: dict, fare: float, latency_ms: float) -> None:
    """Append to JSONL prediction log when enabled (best-effort, never raises)."""
    try:
        from deploy.monitoring.prediction_log import log_prediction

        log_prediction(input_data, fare, latency_ms)
    except Exception:  # noqa: BLE001
        pass  # logging must never break inference


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "deploy.server.rest_server:app",
        host=ServerConfig.APP_HOST,
        port=ServerConfig.APP_PORT,
        reload=(ServerConfig.APP_ENV == "dev"),
    )
