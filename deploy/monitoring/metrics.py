"""CloudWatch metrics sink for inference telemetry.

When ENABLE_METRICS=true, publishes custom metrics to CloudWatch.
When disabled (default for local dev), silently no-ops.

Metrics published:
    TaxiFareAPI/InferenceLatencyMs  — histogram of prediction latency
    TaxiFareAPI/PredictionCount     — running count of predictions
    TaxiFareAPI/ErrorCount          — running count of prediction errors
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_client = None
_NAMESPACE = "TaxiFareAPI"
_ENABLED = os.getenv("ENABLE_METRICS", "false").lower() == "true"


def _get_client():
    """Lazy-init boto3 CloudWatch client."""
    global _client
    if _client is None:
        import boto3

        region = os.getenv("AWS_REGION", "ap-southeast-1")
        _client = boto3.client("cloudwatch", region_name=region)
    return _client


def put_latency(latency_ms: float, endpoint: str = "/predict") -> None:
    """Record inference latency in milliseconds."""
    if not _ENABLED:
        return
    try:
        _get_client().put_metric_data(
            Namespace=_NAMESPACE,
            MetricData=[
                {
                    "MetricName": "InferenceLatencyMs",
                    "Value": latency_ms,
                    "Unit": "Milliseconds",
                    "Dimensions": [{"Name": "Endpoint", "Value": endpoint}],
                }
            ],
        )
    except Exception:
        logger.warning("Failed to push latency metric", exc_info=True)


def put_prediction_count(count: int = 1, endpoint: str = "/predict") -> None:
    """Increment prediction counter."""
    if not _ENABLED:
        return
    try:
        _get_client().put_metric_data(
            Namespace=_NAMESPACE,
            MetricData=[
                {
                    "MetricName": "PredictionCount",
                    "Value": count,
                    "Unit": "Count",
                    "Dimensions": [{"Name": "Endpoint", "Value": endpoint}],
                }
            ],
        )
    except Exception:
        logger.warning("Failed to push prediction count metric", exc_info=True)


def put_error_count(count: int = 1, error_type: Optional[str] = None) -> None:
    """Increment error counter."""
    if not _ENABLED:
        return
    dimensions = [{"Name": "Endpoint", "Value": "/predict"}]
    if error_type:
        dimensions.append({"Name": "ErrorType", "Value": error_type})
    try:
        _get_client().put_metric_data(
            Namespace=_NAMESPACE,
            MetricData=[
                {
                    "MetricName": "ErrorCount",
                    "Value": count,
                    "Unit": "Count",
                    "Dimensions": dimensions,
                }
            ],
        )
    except Exception:
        logger.warning("Failed to push error count metric", exc_info=True)
