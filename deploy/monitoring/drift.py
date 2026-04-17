"""Drift detection from prediction logs.

Reads the JSONL prediction log and compares recent input distributions
against a reference baseline (training data statistics) using
Population Stability Index (PSI).

Usage:
    python -m deploy.monitoring.drift                # CLI summary
    from deploy.monitoring.drift import detect_drift  # programmatic

PSI thresholds (standard industry values):
    PSI < 0.10  — no significant drift
    PSI 0.10-0.25 — moderate drift (investigate)
    PSI > 0.25  — significant drift (retrain recommended)
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

PSI_THRESHOLD_MODERATE = 0.10
PSI_THRESHOLD_SIGNIFICANT = 0.25

# Numeric features tracked for drift
TRACKED_FEATURES = [
    "Trip_Distance_km",
    "Passenger_Count",
    "Base_Fare",
    "Per_Km_Rate",
    "Per_Minute_Rate",
    "Trip_Duration_Minutes",
]


def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Population Stability Index between two distributions.

    Parameters
    ----------
    reference : array-like — baseline distribution (training data)
    current   : array-like — recent production distribution
    n_bins    : int — number of histogram bins

    Returns
    -------
    float — PSI value (0 = identical distributions)
    """
    eps = 1e-6

    # Use reference quantiles for bin edges so both are on the same scale
    breakpoints = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(reference, breakpoints)
    edges[-1] += eps  # ensure the max value falls in the last bin

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    ref_pct = ref_counts / len(reference) + eps
    cur_pct = cur_counts / len(current) + eps

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return psi


def load_prediction_log(
    log_path: Optional[str] = None,
    last_n: int = 200,
) -> list[dict]:
    """Load the most recent ``last_n`` records from the prediction log."""
    path = Path(log_path or os.getenv("PREDICTION_LOG_PATH", "logs/predictions.jsonl"))
    if not path.exists():
        logger.warning("Prediction log not found at %s", path)
        return []

    records: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return records[-last_n:]


def load_reference_stats(data_path: str = "data/splits/train.csv") -> dict[str, np.ndarray]:
    """Load reference distributions from the training split."""
    import pandas as pd

    df = pd.read_csv(data_path)
    return {col: df[col].dropna().values for col in TRACKED_FEATURES if col in df.columns}


def detect_drift(
    log_path: Optional[str] = None,
    reference_path: str = "data/splits/train.csv",
    last_n: int = 200,
) -> dict:
    """Run drift detection and return a summary dict.

    Returns
    -------
    dict with keys:
        status          — "no_drift" | "moderate_drift" | "significant_drift" | "insufficient_data"
        features        — dict of {feature_name: {"psi": float, "status": str}}
        sample_count    — number of recent predictions analysed
        recommendation  — human-readable action
    """
    records = load_prediction_log(log_path, last_n)
    if len(records) < 30:
        return {
            "status": "insufficient_data",
            "features": {},
            "sample_count": len(records),
            "recommendation": f"Need at least 30 predictions, have {len(records)}.",
        }

    # Build current distributions from logged inputs
    current_data: dict[str, list[float]] = {f: [] for f in TRACKED_FEATURES}
    for rec in records:
        inp = rec.get("input", {})
        for feat in TRACKED_FEATURES:
            if feat in inp:
                current_data[feat].append(float(inp[feat]))

    # Load reference
    ref_path = Path(reference_path)
    if not ref_path.exists():
        return {
            "status": "insufficient_data",
            "features": {},
            "sample_count": len(records),
            "recommendation": f"Reference data not found at {reference_path}.",
        }

    reference = load_reference_stats(reference_path)

    # Compute PSI per feature
    feature_results: dict[str, dict] = {}
    worst_status = "no_drift"

    for feat in TRACKED_FEATURES:
        ref_vals = reference.get(feat)
        cur_vals = current_data.get(feat, [])

        if ref_vals is None or len(cur_vals) < 10:
            feature_results[feat] = {"psi": None, "status": "insufficient_data"}
            continue

        psi = compute_psi(ref_vals, np.array(cur_vals))
        if psi > PSI_THRESHOLD_SIGNIFICANT:
            status = "significant_drift"
        elif psi > PSI_THRESHOLD_MODERATE:
            status = "moderate_drift"
        else:
            status = "no_drift"

        feature_results[feat] = {"psi": round(psi, 4), "status": status}

        # Track worst
        if status == "significant_drift":
            worst_status = "significant_drift"
        elif status == "moderate_drift" and worst_status != "significant_drift":
            worst_status = "moderate_drift"

    recommendations = {
        "no_drift": "No action needed.",
        "moderate_drift": "Investigate shifted features; consider retraining if trend continues.",
        "significant_drift": "Retrain recommended — input distributions have shifted significantly.",
    }

    return {
        "status": worst_status,
        "features": feature_results,
        "sample_count": len(records),
        "recommendation": recommendations[worst_status],
    }


if __name__ == "__main__":
    import pprint

    logging.basicConfig(level=logging.INFO)
    result = detect_drift()
    pprint.pprint(result)
