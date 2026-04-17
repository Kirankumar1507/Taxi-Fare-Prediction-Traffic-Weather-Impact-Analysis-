"""Tests for the evaluation pipeline."""

import sys
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluate import load_champion, full_evaluation, generate_report
from src.train import load_config, load_splits, get_xy, mean_absolute_percentage_error


@pytest.fixture
def config():
    return load_config()


class TestLoadChampion:
    def test_load_catboost(self, config):
        """Should load CatBoost model successfully."""
        model = load_champion(config, "catboost")
        assert hasattr(model, "predict")

    def test_load_xgboost(self, config):
        """Should load XGBoost model successfully."""
        model = load_champion(config, "xgboost")
        assert hasattr(model, "predict")

    def test_load_ridge(self, config):
        """Should load Ridge model successfully."""
        model = load_champion(config, "ridge")
        assert hasattr(model, "predict")

    def test_unknown_model_raises(self, config):
        """Unknown model name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown model"):
            load_champion(config, "random_forest")


class TestModelPredictions:
    def test_catboost_predicts_correct_shape(self, config):
        """CatBoost predictions should match test set size."""
        model = load_champion(config, "catboost")
        _, _, test = load_splits(config)
        X_test, y_test = get_xy(test, config)
        preds = model.predict(X_test)
        assert preds.shape == y_test.shape

    def test_catboost_r2_above_threshold(self, config):
        """CatBoost test R2 should be above 0.80."""
        from sklearn.metrics import r2_score
        model = load_champion(config, "catboost")
        _, _, test = load_splits(config)
        X_test, y_test = get_xy(test, config)
        preds = model.predict(X_test)
        r2 = r2_score(y_test, preds)
        assert r2 > 0.80, f"R2={r2:.4f} below threshold"

    def test_catboost_mape_below_threshold(self, config):
        """CatBoost test MAPE should be below 20%."""
        model = load_champion(config, "catboost")
        _, _, test = load_splits(config)
        X_test, y_test = get_xy(test, config)
        preds = model.predict(X_test)
        mape = mean_absolute_percentage_error(y_test, preds)
        assert mape < 20, f"MAPE={mape:.2f}% above threshold"


class TestGenerateReport:
    def test_report_contains_metrics(self):
        """Generated report should contain all metric values."""
        metrics = {"MAE": 4.0, "RMSE": 12.0, "MAPE": 6.0, "R2": 0.94}
        baseline = {"MAE": 8.0, "RMSE": 16.0, "MAPE": 16.0, "R2": 0.88}
        residuals = np.array([1.0, -2.0, 0.5, -0.3])
        import pandas as pd
        shap_df = pd.DataFrame({
            "Feature": ["metered_fare", "distance_cost"],
            "Mean_SHAP": [15.0, 6.0],
        })
        report = generate_report(metrics, baseline, "catboost", shap_df, residuals)
        assert "PASS" in report
        assert "catboost" in report
        assert "metered_fare" in report

    def test_report_pass_fail_logic(self):
        """Report should show FAIL when metrics don't meet thresholds."""
        metrics = {"MAE": 10.0, "RMSE": 20.0, "MAPE": 25.0, "R2": 0.70}
        baseline = {"MAE": 15.0, "RMSE": 25.0, "MAPE": 30.0, "R2": 0.50}
        residuals = np.array([5.0, -5.0])
        report = generate_report(metrics, baseline, "catboost", None, residuals)
        assert "FAIL" in report  # MAPE > 20 or R2 < 0.80


class TestFullEvaluation:
    def test_full_eval_returns_metrics(self, config):
        """full_evaluation should return dict with all metric keys."""
        metrics = full_evaluation("catboost")
        assert set(metrics.keys()) == {"MAE", "RMSE", "MAPE", "R2"}
        assert all(isinstance(v, float) for v in metrics.values())
