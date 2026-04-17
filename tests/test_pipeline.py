"""Integration tests for the full preprocessing and training pipeline."""

import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocess import (
    load_config, load_raw_data, drop_null_targets, encode_categoricals,
    compute_impute_stats, apply_imputation, engineer_features,
    split_data, run_preprocessing, get_project_root,
)
from src.train import (
    load_splits, get_xy, train_ridge, train_xgboost, train_catboost,
)


@pytest.fixture
def config():
    return load_config()


class TestRunPreprocessing:
    def test_returns_splits_and_config(self):
        """run_preprocessing should return (train, val, test, config)."""
        train, val, test, config = run_preprocessing()
        assert isinstance(train, pd.DataFrame)
        assert isinstance(val, pd.DataFrame)
        assert isinstance(test, pd.DataFrame)
        assert isinstance(config, dict)

    def test_splits_have_model_features(self):
        """All model features should be present in splits."""
        train, val, test, config = run_preprocessing()
        for name, df in [("train", train), ("val", val), ("test", test)]:
            for col in config["features"]["model_features"]:
                assert col in df.columns, f"Missing {col} in {name}"

    def test_no_nulls_in_output(self):
        """Final output should have no nulls in model features."""
        train, val, test, config = run_preprocessing()
        for name, df in [("train", train), ("val", val), ("test", test)]:
            for col in config["features"]["model_features"]:
                assert df[col].isnull().sum() == 0, f"Nulls in {col} of {name}"

    def test_impute_stats_saved(self):
        """Imputation stats file should be created."""
        run_preprocessing()
        root = get_project_root()
        stats_path = root / "models" / "impute_stats.joblib"
        assert stats_path.exists()


class TestTrainModels:
    @pytest.fixture
    def train_val_data(self, config):
        train, val, _ = load_splits(config)
        X_train, y_train = get_xy(train, config)
        X_val, y_val = get_xy(val, config)
        return X_train, y_train, X_val, y_val

    def test_ridge_returns_model_and_metrics(self, train_val_data, config):
        """Ridge training should return a model and metrics dict."""
        X_train, y_train, X_val, y_val = train_val_data
        model, metrics = train_ridge(X_train, y_train, X_val, y_val, config)
        assert hasattr(model, "predict")
        assert "R2" in metrics
        assert metrics["R2"] > 0

    def test_xgboost_returns_model_and_metrics(self, train_val_data, config):
        """XGBoost training should return a model and metrics dict."""
        X_train, y_train, X_val, y_val = train_val_data
        model, metrics = train_xgboost(X_train, y_train, X_val, y_val, config)
        assert hasattr(model, "predict")
        assert "R2" in metrics

    def test_catboost_returns_model_and_metrics(self, train_val_data, config):
        """CatBoost training should return a model and metrics dict."""
        X_train, y_train, X_val, y_val = train_val_data
        model, metrics = train_catboost(X_train, y_train, X_val, y_val, config)
        assert hasattr(model, "predict")
        assert "R2" in metrics

    def test_catboost_val_r2_above_minimum(self, train_val_data, config):
        """CatBoost validation R2 should be reasonable."""
        X_train, y_train, X_val, y_val = train_val_data
        _, metrics = train_catboost(X_train, y_train, X_val, y_val, config)
        assert metrics["R2"] > 0.70, f"Val R2={metrics['R2']:.4f} too low"
