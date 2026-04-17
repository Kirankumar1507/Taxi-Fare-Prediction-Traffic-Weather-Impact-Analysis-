"""SageMaker inference entry point.

SageMaker invokes these four functions automatically when serving
a model behind a real-time endpoint.

    model_fn   — load model artifacts from the model directory
    input_fn   — deserialise the incoming request body
    predict_fn — run inference
    output_fn  — serialise the prediction for the HTTP response

The model archive (model.tar.gz) is expected to contain:
    models/catboost_v1.cbm
    models/feature_names.joblib
    configs/config.yaml
    src/predict.py   (+ supporting modules)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def model_fn(model_dir: str) -> Any:
    """Load the FarePredictor from the SageMaker model directory.

    SageMaker extracts model.tar.gz into ``model_dir``.  We temporarily
    add it to sys.path so that ``src.predict`` resolves.
    """
    import sys

    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)

    # Patch get_project_root to point at model_dir instead of
    # the installed package location.
    from src import preprocess as _prep

    from pathlib import Path

    _prep.get_project_root = lambda: Path(model_dir)

    from src.predict import FarePredictor

    model_name = os.getenv("MODEL_NAME", "catboost")
    predictor = FarePredictor(model_name)
    logger.info("Model loaded from %s  features=%d", model_dir, len(predictor.feature_names))
    return predictor


def input_fn(request_body: str, content_type: str = "application/json") -> dict | list[dict]:
    """Deserialise the request body.

    Accepts:
        - ``application/json`` with a single trip dict **or** a list of dicts.
    """
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {content_type}")

    payload = json.loads(request_body)
    return payload


def predict_fn(input_data: dict | list[dict], model: Any) -> list[float]:
    """Run prediction(s) and return a list of floats."""
    if isinstance(input_data, dict):
        input_data = [input_data]

    results: list[float] = []
    for trip in input_data:
        fare = model.predict(trip)
        results.append(round(float(fare), 4))

    return results


def output_fn(prediction: list[float], accept: str = "application/json") -> str:
    """Serialise prediction list to JSON."""
    if accept != "application/json":
        raise ValueError(f"Unsupported accept type: {accept}")

    return json.dumps({"predictions": prediction})
