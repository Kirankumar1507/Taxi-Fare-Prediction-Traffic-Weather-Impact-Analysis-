"""Structured JSONL prediction logger.

Every prediction is appended as a single JSON line to the log file
specified by PREDICTION_LOG_PATH (default: logs/predictions.jsonl).

This log feeds the drift-detection module and provides an audit trail
for model governance.

Log schema (one JSON object per line):
    timestamp     — ISO 8601 UTC
    model_name    — e.g. "catboost"
    model_version — e.g. "v1"
    input         — dict of raw trip fields
    predicted_fare — float
    latency_ms    — float
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_LOG_PATH: str = os.getenv("PREDICTION_LOG_PATH", "logs/predictions.jsonl")


def log_prediction(
    input_data: dict,
    predicted_fare: float,
    latency_ms: float,
    model_name: str | None = None,
    model_version: str | None = None,
) -> None:
    """Append one prediction record to the JSONL log file.

    Best-effort: failures are logged as warnings but never propagated.
    """
    from deploy.server.config import ServerConfig

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name or ServerConfig.MODEL_NAME,
        "model_version": model_version or ServerConfig.MODEL_VERSION,
        "input": input_data,
        "predicted_fare": predicted_fare,
        "latency_ms": round(latency_ms, 2),
    }

    try:
        path = Path(_LOG_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        logger.warning("Failed to write prediction log", exc_info=True)
