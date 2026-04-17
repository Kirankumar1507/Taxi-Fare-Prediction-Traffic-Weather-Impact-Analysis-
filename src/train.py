"""
Model training: Ridge baseline, XGBoost, CatBoost.
Run standalone: python src/train.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import yaml
import joblib
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.preprocess import get_project_root, load_config, run_preprocessing

warnings.filterwarnings("ignore", category=UserWarning)


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAPE — handles zero targets by filtering them."""
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> dict:
    """Compute all regression metrics and print them."""
    metrics = {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }
    print(f"\n  [{label}]")
    for k, v in metrics.items():
        print(f"    {k}: {v:.4f}")
    return metrics


def load_splits(config: dict) -> tuple:
    """Load train/val/test splits from disk."""
    root = get_project_root()
    train = pd.read_csv(root / config["paths"]["train_split"])
    val = pd.read_csv(root / config["paths"]["val_split"])
    test = pd.read_csv(root / config["paths"]["test_split"])
    return train, val, test


def get_xy(df: pd.DataFrame, config: dict) -> tuple:
    """Extract feature matrix X and target y from DataFrame."""
    features = config["features"]["model_features"]
    target = config["features"]["target"]
    return df[features].values, df[target].values


def train_ridge(X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray, y_val: np.ndarray,
                config: dict) -> tuple:
    """Train Ridge regression baseline (with StandardScaler to avoid overflow)."""
    print("\n--- Ridge Regression (Baseline) ---")
    model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=config["models"]["ridge"]["alpha"], random_state=config["random_seed"]),
    )
    model.fit(X_train, y_train)

    train_metrics = evaluate_model(y_train, model.predict(X_train), "Ridge Train")
    val_metrics = evaluate_model(y_val, model.predict(X_val), "Ridge Val")

    # Also train metered_fare-only model for diagnostic
    mf_idx = config["features"]["model_features"].index("metered_fare")
    from sklearn.linear_model import LinearRegression
    mf_model = LinearRegression()
    mf_model.fit(X_train[:, mf_idx:mf_idx+1], y_train)
    mf_pred = mf_model.predict(X_val[:, mf_idx:mf_idx+1])
    print("\n  [Ridge metered_fare-only on Val]")
    mf_r2 = r2_score(y_val, mf_pred)
    mf_mape = mean_absolute_percentage_error(y_val, mf_pred)
    print(f"    R2: {mf_r2:.4f}, MAPE: {mf_mape:.4f}%")
    if mf_r2 > 0.98:
        print("    WARNING: metered_fare alone explains >98% variance. Dataset likely synthetic.")

    return model, val_metrics


def train_xgboost(X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray,
                   config: dict) -> tuple:
    """Train XGBoost with early stopping."""
    print("\n--- XGBoost ---")
    params = config["models"]["xgboost"]
    model = XGBRegressor(
        max_depth=params["max_depth"],
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        reg_alpha=params["reg_alpha"],
        reg_lambda=params["reg_lambda"],
        min_child_weight=params.get("min_child_weight", 1),
        early_stopping_rounds=params["early_stopping_rounds"],
        random_state=config["random_seed"],
        verbosity=0,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    print(f"  Best iteration: {model.best_iteration}")

    train_metrics = evaluate_model(y_train, model.predict(X_train), "XGBoost Train")
    val_metrics = evaluate_model(y_val, model.predict(X_val), "XGBoost Val")
    return model, val_metrics


def train_catboost(X_train: np.ndarray, y_train: np.ndarray,
                    X_val: np.ndarray, y_val: np.ndarray,
                    config: dict) -> tuple:
    """Train CatBoost with early stopping."""
    print("\n--- CatBoost ---")
    params = config["models"]["catboost"]
    model = CatBoostRegressor(
        depth=params["depth"],
        iterations=params["iterations"],
        learning_rate=params["learning_rate"],
        l2_leaf_reg=params["l2_leaf_reg"],
        min_data_in_leaf=params.get("min_data_in_leaf", 1),
        early_stopping_rounds=params["early_stopping_rounds"],
        random_seed=config["random_seed"],
        verbose=params["verbose"],
    )
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        verbose=False,
    )
    print(f"  Best iteration: {model.best_iteration_}")

    train_metrics = evaluate_model(y_train, model.predict(X_train), "CatBoost Train")
    val_metrics = evaluate_model(y_val, model.predict(X_val), "CatBoost Val")
    return model, val_metrics


def run_cv(model, X: np.ndarray, y: np.ndarray, config: dict, label: str) -> dict:
    """Run RepeatedKFold cross-validation."""
    cv_config = config["cv"]
    rkf = RepeatedKFold(
        n_splits=cv_config["n_splits"],
        n_repeats=cv_config["n_repeats"],
        random_state=config["random_seed"],
    )
    scores_r2 = cross_val_score(model, X, y, cv=rkf, scoring="r2")
    scores_mae = -cross_val_score(model, X, y, cv=rkf, scoring="neg_mean_absolute_error")

    print(f"\n  [{label} CV]")
    print(f"    R2:  {scores_r2.mean():.4f} +/- {scores_r2.std():.4f}")
    print(f"    MAE: {scores_mae.mean():.4f} +/- {scores_mae.std():.4f}")
    return {"cv_r2_mean": scores_r2.mean(), "cv_r2_std": scores_r2.std(),
            "cv_mae_mean": scores_mae.mean(), "cv_mae_std": scores_mae.std()}


def train_all() -> dict:
    """Train all models, compare, and save champion."""
    config = load_config()
    root = get_project_root()

    # Check if splits exist, otherwise run preprocessing
    if not (root / config["paths"]["train_split"]).exists():
        print("Splits not found. Running preprocessing first...\n")
        run_preprocessing()

    train, val, test = load_splits(config)
    X_train, y_train = get_xy(train, config)
    X_val, y_val = get_xy(val, config)
    feature_names = config["features"]["model_features"]

    print("=" * 60)
    print("MODEL TRAINING")
    print("=" * 60)
    print(f"Train: {X_train.shape}, Val: {X_val.shape}")

    results = {}

    # 1. Ridge baseline
    ridge_model, ridge_metrics = train_ridge(X_train, y_train, X_val, y_val, config)
    results["Ridge"] = ridge_metrics

    # 2. XGBoost
    xgb_model, xgb_metrics = train_xgboost(X_train, y_train, X_val, y_val, config)
    results["XGBoost"] = xgb_metrics

    # 3. CatBoost
    cb_model, cb_metrics = train_catboost(X_train, y_train, X_val, y_val, config)
    results["CatBoost"] = cb_metrics

    # Comparison table
    print("\n" + "=" * 60)
    print("MODEL COMPARISON (Validation Set)")
    print("=" * 60)
    print(f"{'Model':<12} {'MAE':>8} {'RMSE':>8} {'MAPE%':>8} {'R2':>8}")
    print("-" * 48)
    for name, m in results.items():
        print(f"{name:<12} {m['MAE']:>8.3f} {m['RMSE']:>8.3f} {m['MAPE']:>8.3f} {m['R2']:>8.4f}")

    # Select champion (best R2 on val)
    champion_name = max(results, key=lambda k: results[k]["R2"])
    champion_models = {"Ridge": ridge_model, "XGBoost": xgb_model, "CatBoost": cb_model}
    champion = champion_models[champion_name]
    print(f"\nChampion model: {champion_name} (R2={results[champion_name]['R2']:.4f})")

    # Save all models
    models_dir = root / config["paths"]["models_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(ridge_model, models_dir / "ridge_v1.joblib")
    joblib.dump(xgb_model, models_dir / "xgboost_v1.joblib")
    cb_model.save_model(str(models_dir / "catboost_v1.cbm"))

    # Save feature names for inference
    joblib.dump(feature_names, models_dir / "feature_names.joblib")

    print(f"\nModels saved to {models_dir}/")

    return {
        "results": results,
        "champion_name": champion_name,
        "champion_model": champion,
        "config": config,
    }


if __name__ == "__main__":
    train_all()
