"""Tests for the training pipeline."""

import sys
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.train import (
    load_config,
    load_splits,
    get_xy,
    mean_absolute_percentage_error,
    evaluate_model,
)


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def splits(config):
    return load_splits(config)


class TestGetXY:
    def test_returns_correct_shapes(self, splits, config):
        """X should have len(model_features) columns, y should be 1D."""
        train, val, test = splits
        n_features = len(config["features"]["model_features"])
        for name, df in [("train", train), ("val", val), ("test", test)]:
            X, y = get_xy(df, config)
            assert X.shape[1] == n_features, f"{name}: expected {n_features} features, got {X.shape[1]}"
            assert X.shape[0] == len(df), f"{name}: row count mismatch"
            assert y.shape == (len(df),), f"{name}: y shape mismatch"

    def test_no_nans_in_features(self, splits, config):
        """No NaNs should be in the feature matrix."""
        train, _, _ = splits
        X, y = get_xy(train, config)
        assert not np.isnan(X).any(), "NaN found in feature matrix"
        assert not np.isnan(y).any(), "NaN found in target"


class TestMAPE:
    def test_perfect_prediction(self):
        """MAPE should be 0 for perfect predictions."""
        y = np.array([10.0, 20.0, 30.0])
        assert mean_absolute_percentage_error(y, y) == 0.0

    def test_known_mape(self):
        """MAPE for a known case."""
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([110.0, 180.0])
        # |10/100| + |20/200| = 0.10 + 0.10, mean = 0.10, * 100 = 10%
        assert mean_absolute_percentage_error(y_true, y_pred) == pytest.approx(10.0)

    def test_handles_zero_targets(self):
        """Should filter out zero targets to avoid division by zero."""
        y_true = np.array([0.0, 100.0])
        y_pred = np.array([5.0, 110.0])
        # Only non-zero: |110-100|/100 = 10%
        assert mean_absolute_percentage_error(y_true, y_pred) == pytest.approx(10.0)


class TestEvaluateModel:
    def test_returns_all_metrics(self, splits, config):
        """evaluate_model should return MAE, RMSE, MAPE, R2."""
        train, _, _ = splits
        X, y = get_xy(train, config)
        # Use the actual values as "predictions" for a quick test
        metrics = evaluate_model(y, y, "test_perfect")
        assert set(metrics.keys()) == {"MAE", "RMSE", "MAPE", "R2"}
        assert metrics["MAE"] == 0.0
        assert metrics["R2"] == 1.0
