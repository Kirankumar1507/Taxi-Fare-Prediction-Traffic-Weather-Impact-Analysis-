"""
Data preprocessing: loading, cleaning, encoding, splitting, then imputation & feature engineering.
Imputation stats are fit on train only to prevent data leakage.
Run standalone: python -m src.preprocess
"""

import numpy as np
import pandas as pd
import yaml
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split


def get_project_root() -> Path:
    """Return project root (parent of src/)."""
    return Path(__file__).resolve().parent.parent


def load_config() -> dict:
    """Load config.yaml from configs/."""
    config_path = get_project_root() / "configs" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_raw_data(config: dict) -> pd.DataFrame:
    """Load raw CSV and print basic stats."""
    path = get_project_root() / config["paths"]["raw_data"]
    df = pd.read_csv(path)
    print(f"Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Null counts:\n{df.isnull().sum()}\n")
    return df


def drop_null_targets(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Drop rows where the target is null (cannot impute target)."""
    df = df.copy()
    target = config["features"]["target"]
    n_null = df[target].isnull().sum()
    if n_null > 0:
        df = df.dropna(subset=[target])
        print(f"  Dropped {n_null} rows with null target")
    return df


def encode_categoricals(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Ordinal-encode categorical features using config mappings."""
    df = df.copy()
    encoding = config["encoding"]

    df["Traffic_encoded"] = df["Traffic_Conditions"].map(encoding["Traffic_Conditions"])
    df["Weather_encoded"] = df["Weather"].map(encoding["Weather"])
    df["Time_of_Day_encoded"] = df["Time_of_Day"].map(encoding["Time_of_Day"])
    df["Day_of_Week_encoded"] = df["Day_of_Week"].map(encoding["Day_of_Week"])

    # Drop original categorical columns
    df = df.drop(columns=config["features"]["raw_categorical"])
    print("Encoded categoricals: Traffic, Weather, Time_of_Day, Day_of_Week")
    return df


def compute_impute_stats(train: pd.DataFrame, config: dict) -> dict:
    """Compute imputation statistics from training data only."""
    stats = {"numeric": {}, "categorical_encoded": {}}
    numeric_cols = config["features"]["raw_numeric"]
    encoded_cats = ["Traffic_encoded", "Weather_encoded",
                    "Time_of_Day_encoded", "Day_of_Week_encoded"]

    for col in numeric_cols:
        stats["numeric"][col] = train[col].median()

    for col in encoded_cats:
        stats["categorical_encoded"][col] = train[col].mode()[0]

    return stats


def apply_imputation(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """Apply pre-computed imputation statistics to a DataFrame."""
    df = df.copy()

    for col, median_val in stats["numeric"].items():
        if df[col].isnull().any():
            n_missing = df[col].isnull().sum()
            df[col] = df[col].fillna(median_val)
            print(f"  Imputed {col}: {n_missing} nulls -> median={median_val:.2f}")

    for col, mode_val in stats["categorical_encoded"].items():
        if df[col].isnull().any():
            n_missing = df[col].isnull().sum()
            df[col] = df[col].fillna(mode_val)
            print(f"  Imputed {col}: {n_missing} nulls -> mode={mode_val}")

    return df


def engineer_features(df: pd.DataFrame, avg_speed_median: float = None) -> pd.DataFrame:
    """Create engineered features. Uses provided median for avg_speed_kmh fill."""
    df = df.copy()

    # Priority 1: Algebraic reconstructions
    df["distance_cost"] = df["Trip_Distance_km"] * df["Per_Km_Rate"]
    df["duration_cost"] = df["Trip_Duration_Minutes"] * df["Per_Minute_Rate"]
    df["metered_fare"] = df["Base_Fare"] + df["distance_cost"] + df["duration_cost"]
    df["avg_speed_kmh"] = df["Trip_Distance_km"] / (df["Trip_Duration_Minutes"] / 60)
    # Handle inf/nan from division
    df["avg_speed_kmh"] = df["avg_speed_kmh"].replace([np.inf, -np.inf], np.nan)
    if avg_speed_median is not None:
        df["avg_speed_kmh"] = df["avg_speed_kmh"].fillna(avg_speed_median)
    else:
        df["avg_speed_kmh"] = df["avg_speed_kmh"].fillna(df["avg_speed_kmh"].median())

    # Priority 2: Interactions
    df["traffic_distance"] = df["Traffic_encoded"] * df["Trip_Distance_km"]
    df["weather_duration"] = df["Weather_encoded"] * df["Trip_Duration_Minutes"]

    # Priority 3: Temporal flags
    df["is_peak_hour"] = df["Time_of_Day_encoded"].isin([0, 2]).astype(int)  # Morning=0, Evening=2
    df["is_weekend"] = df["Day_of_Week_encoded"].astype(int)  # Already 0/1
    df["peak_traffic"] = df["is_peak_hour"] * df["Traffic_encoded"]

    print(f"Engineered 9 new features. Total columns: {df.shape[1]}")
    return df


def split_data(df: pd.DataFrame, config: dict) -> tuple:
    """Split into train/val/test. Returns (train, val, test) DataFrames."""
    seed = config["random_seed"]
    test_pct = config["split"]["test_pct"]
    val_pct = config["split"]["val_pct"]
    target = config["features"]["target"]

    # First split: train+val vs test
    train_val, test = train_test_split(df, test_size=test_pct, random_state=seed)
    # Second split: train vs val
    train, val = train_test_split(train_val, test_size=val_pct, random_state=seed)

    print(f"Split sizes: train={len(train)}, val={len(val)}, test={len(test)}")
    print(f"Target stats (train): mean={train[target].mean():.2f}, std={train[target].std():.2f}")
    print(f"Target stats (val):   mean={val[target].mean():.2f}, std={val[target].std():.2f}")
    print(f"Target stats (test):  mean={test[target].mean():.2f}, std={test[target].std():.2f}")
    return train, val, test


def run_preprocessing() -> tuple:
    """Full preprocessing pipeline. Imputation is fit on train only (no leakage).

    Pipeline order:
    1. Load raw data
    2. Drop null targets
    3. Encode categoricals (deterministic mapping, no leakage)
    4. Split into train/val/test
    5. Compute imputation stats on TRAIN only
    6. Apply imputation to all splits
    7. Engineer features (avg_speed median from TRAIN only)
    8. Save splits and imputation stats
    """
    config = load_config()
    root = get_project_root()

    print("=" * 60)
    print("PREPROCESSING PIPELINE")
    print("=" * 60)

    # Step 1: Load
    df = load_raw_data(config)

    # Step 2: Drop null targets (before split to avoid empty target rows)
    print("\n--- Drop Null Targets ---")
    df = drop_null_targets(df, config)

    # Step 3: Encode categoricals (deterministic mapping — no data-dependent stats)
    print("\n--- Encoding ---")
    df = encode_categoricals(df, config)

    # Step 4: Split BEFORE imputation
    print("\n--- Splitting ---")
    train, val, test = split_data(df, config)

    # Step 5: Compute imputation stats from TRAIN only
    print("\n--- Imputation (fit on train) ---")
    impute_stats = compute_impute_stats(train, config)
    print(f"  Imputation stats computed from {len(train)} training rows")

    # Step 6: Apply imputation to all splits using train stats
    print("\n  Applying to train:")
    train = apply_imputation(train, impute_stats)
    print("  Applying to val:")
    val = apply_imputation(val, impute_stats)
    print("  Applying to test:")
    test = apply_imputation(test, impute_stats)

    # Step 7: Feature engineering (avg_speed median from train only)
    print("\n--- Feature Engineering ---")
    # Compute avg_speed on train first to get the median
    train_speed = train["Trip_Distance_km"] / (train["Trip_Duration_Minutes"] / 60)
    train_speed = train_speed.replace([np.inf, -np.inf], np.nan)
    avg_speed_median = train_speed.median()
    print(f"  avg_speed_kmh median (from train): {avg_speed_median:.2f}")

    train = engineer_features(train, avg_speed_median=avg_speed_median)
    val = engineer_features(val, avg_speed_median=avg_speed_median)
    test = engineer_features(test, avg_speed_median=avg_speed_median)

    # Step 8: Save processed data and splits
    processed_path = root / config["paths"]["processed_data"]
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([train, val, test]).to_csv(processed_path, index=False)
    print(f"\nSaved processed data to {processed_path}")

    splits_dir = root / "data" / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(splits_dir / "train.csv", index=False)
    val.to_csv(splits_dir / "val.csv", index=False)
    test.to_csv(splits_dir / "test.csv", index=False)
    print(f"Saved splits to {splits_dir}/")

    # Save imputation stats for predict.py to use
    stats_path = root / config["paths"]["models_dir"] / "impute_stats.joblib"
    Path(stats_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"impute_stats": impute_stats, "avg_speed_median": avg_speed_median}, stats_path)
    print(f"Saved imputation stats to {stats_path}")

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    return train, val, test, config


if __name__ == "__main__":
    run_preprocessing()
