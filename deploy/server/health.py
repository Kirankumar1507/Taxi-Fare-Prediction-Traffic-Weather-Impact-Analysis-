"""Health-check logic — separated from the route handler for testability."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.predict import FarePredictor

logger = logging.getLogger(__name__)

EXPECTED_FEATURE_COUNT = 19


def check_health(predictor: "FarePredictor | None") -> dict:
    """Return a health-check dict.

    Checks
    ------
    1. Model object is not None and loaded.
    2. Feature schema length matches the expected count (19).

    Returns a dict consumable by ``HealthResponse``.
    """
    if predictor is None:
        return {
            "status": "unhealthy",
            "model_loaded": False,
            "model_name": "n/a",
            "model_version": "n/a",
            "feature_count": 0,
        }

    model_loaded = predictor.model is not None
    feature_count = len(predictor.feature_names) if predictor.feature_names else 0
    schema_ok = feature_count == EXPECTED_FEATURE_COUNT

    if not model_loaded or not schema_ok:
        status = "degraded"
        if not schema_ok:
            logger.warning(
                "Feature schema mismatch: expected %d, got %d",
                EXPECTED_FEATURE_COUNT,
                feature_count,
            )
    else:
        status = "healthy"

    from deploy.server.config import ServerConfig

    return {
        "status": status,
        "model_loaded": model_loaded,
        "model_name": ServerConfig.MODEL_NAME,
        "model_version": ServerConfig.MODEL_VERSION,
        "feature_count": feature_count,
    }
