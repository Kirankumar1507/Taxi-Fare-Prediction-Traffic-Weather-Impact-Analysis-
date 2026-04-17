"""
Final evaluation on held-out test set + SHAP analysis.
Run standalone: python src/evaluate.py
"""

import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from catboost import CatBoostRegressor

from src.preprocess import get_project_root, load_config
from src.train import mean_absolute_percentage_error, load_splits, get_xy

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def load_champion(config: dict, model_name: str = "catboost"):
    """Load the champion model from disk."""
    root = get_project_root()
    models_dir = root / config["paths"]["models_dir"]

    if model_name == "catboost":
        model = CatBoostRegressor()
        model.load_model(str(models_dir / "catboost_v1.cbm"))
    elif model_name == "xgboost":
        model = joblib.load(models_dir / "xgboost_v1.joblib")
    elif model_name == "ridge":
        model = joblib.load(models_dir / "ridge_v1.joblib")
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model


def full_evaluation(model_name: str = "catboost") -> dict:
    """Evaluate champion on test set. Generate SHAP plots and report."""
    config = load_config()
    root = get_project_root()
    reports_dir = root / config["paths"]["reports_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("FINAL EVALUATION (Test Set)")
    print("=" * 60)

    # Load data and model
    train, val, test = load_splits(config)
    X_train, y_train = get_xy(train, config)
    X_test, y_test = get_xy(test, config)
    feature_names = config["features"]["model_features"]

    model = load_champion(config, model_name)
    y_pred = model.predict(X_test)

    # --- Metrics ---
    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "MAPE": mean_absolute_percentage_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred),
    }

    print(f"\n  Model: {model_name}")
    for k, v in metrics.items():
        print(f"    {k}: {v:.4f}")

    # --- Also evaluate baseline for comparison ---
    ridge = joblib.load(root / config["paths"]["models_dir"] / "ridge_v1.joblib")
    ridge_pred = ridge.predict(X_test)
    baseline_metrics = {
        "MAE": mean_absolute_error(y_test, ridge_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, ridge_pred)),
        "MAPE": mean_absolute_percentage_error(y_test, ridge_pred),
        "R2": r2_score(y_test, ridge_pred),
    }

    print(f"\n  Baseline (Ridge):")
    for k, v in baseline_metrics.items():
        print(f"    {k}: {v:.4f}")

    # --- Residual analysis ---
    residuals = y_test - y_pred
    print(f"\n  Residuals: mean={residuals.mean():.4f}, std={residuals.std():.4f}")
    print(f"  Max overpredict: {residuals.min():.2f}, Max underpredict: {residuals.max():.2f}")

    # --- SHAP Analysis ---
    print("\n--- SHAP Feature Importance ---")
    if model_name in ("xgboost", "catboost"):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        # SHAP summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
        plt.tight_layout()
        shap_path = reports_dir / "shap_summary.png"
        plt.savefig(shap_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved SHAP summary plot: {shap_path}")

        # SHAP bar plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, feature_names=feature_names,
                          plot_type="bar", show=False)
        plt.tight_layout()
        bar_path = reports_dir / "shap_importance_bar.png"
        plt.savefig(bar_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved SHAP bar plot: {bar_path}")

        # Mean absolute SHAP values
        mean_shap = np.abs(shap_values).mean(axis=0)
        shap_df = pd.DataFrame({
            "Feature": feature_names,
            "Mean_SHAP": mean_shap
        }).sort_values("Mean_SHAP", ascending=False)
        print("\n  Top features by mean |SHAP|:")
        for _, row in shap_df.head(10).iterrows():
            print(f"    {row['Feature']:<25} {row['Mean_SHAP']:.4f}")
    else:
        shap_df = None
        print("  SHAP skipped for Ridge (use permutation importance instead)")

    # --- Prediction vs Actual plot ---
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred, alpha=0.5, s=20)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    plt.plot(lims, lims, "r--", linewidth=1)
    plt.xlabel("Actual Trip Price")
    plt.ylabel("Predicted Trip Price")
    plt.title(f"{model_name} — Predicted vs Actual (Test Set)")
    plt.tight_layout()
    scatter_path = reports_dir / "predicted_vs_actual.png"
    plt.savefig(scatter_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved scatter plot: {scatter_path}")

    # --- Residual distribution ---
    plt.figure(figsize=(8, 5))
    plt.hist(residuals, bins=30, edgecolor="black", alpha=0.7)
    plt.axvline(0, color="red", linestyle="--")
    plt.xlabel("Residual (Actual - Predicted)")
    plt.ylabel("Count")
    plt.title("Residual Distribution (Test Set)")
    plt.tight_layout()
    resid_path = reports_dir / "residual_distribution.png"
    plt.savefig(resid_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved residual plot: {resid_path}")

    # --- Write Final Report ---
    report = generate_report(metrics, baseline_metrics, model_name, shap_df, residuals)
    report_path = reports_dir / "final_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\n  Final report: {report_path}")

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)

    return metrics


def generate_report(metrics: dict, baseline: dict, model_name: str,
                    shap_df: pd.DataFrame, residuals: np.ndarray) -> str:
    """Generate markdown evaluation report."""
    report = f"""# Taxi Fare Prediction — Final Evaluation Report

## Champion Model: {model_name}

## Test Set Metrics

| Metric | Baseline (Ridge) | Champion ({model_name}) | Improvement |
|--------|------------------|------------------------|-------------|
| MAE    | {baseline['MAE']:.4f} | {metrics['MAE']:.4f} | {((baseline['MAE'] - metrics['MAE']) / baseline['MAE'] * 100):.1f}% |
| RMSE   | {baseline['RMSE']:.4f} | {metrics['RMSE']:.4f} | {((baseline['RMSE'] - metrics['RMSE']) / baseline['RMSE'] * 100):.1f}% |
| MAPE   | {baseline['MAPE']:.4f}% | {metrics['MAPE']:.4f}% | {((baseline['MAPE'] - metrics['MAPE']) / baseline['MAPE'] * 100):.1f}% |
| R2     | {baseline['R2']:.4f} | {metrics['R2']:.4f} | {((metrics['R2'] - baseline['R2']) / (1 - baseline['R2']) * 100):.1f}% of remaining |

## Success Criteria Check
- MAPE < 20%: **{'PASS' if metrics['MAPE'] < 20 else 'FAIL'}** ({metrics['MAPE']:.2f}%)
- R2 > 0.80: **{'PASS' if metrics['R2'] > 0.80 else 'FAIL'}** ({metrics['R2']:.4f})

## Residual Analysis
- Mean residual: {residuals.mean():.4f}
- Std residual: {residuals.std():.4f}
- Max overprediction: {residuals.min():.2f}
- Max underprediction: {residuals.max():.2f}

## Feature Importance (SHAP)
See `reports/shap_summary.png` and `reports/shap_importance_bar.png` for visualizations.

"""
    if shap_df is not None:
        report += "| Rank | Feature | Mean |SHAP| |\n"
        report += "|------|---------|-------------|\n"
        for i, (_, row) in enumerate(shap_df.iterrows(), 1):
            report += f"| {i} | {row['Feature']} | {row['Mean_SHAP']:.4f} |\n"

    report += """
## Plots
- `reports/predicted_vs_actual.png` — Scatter plot of predicted vs actual prices
- `reports/residual_distribution.png` — Histogram of prediction residuals
- `reports/shap_summary.png` — SHAP beeswarm plot
- `reports/shap_importance_bar.png` — SHAP feature importance bar chart

## Known Limitations
1. Dataset is synthetic (metered_fare closely reconstructs Trip_Price for most rows)
2. Small sample size (1000 rows, ~950 after dropping null targets)
3. Uniform null pattern (exactly 50 nulls per column) — MCAR by design
4. Some outliers with Trip_Distance_km > 100 km create high-leverage points
"""
    return report


if __name__ == "__main__":
    full_evaluation("catboost")
