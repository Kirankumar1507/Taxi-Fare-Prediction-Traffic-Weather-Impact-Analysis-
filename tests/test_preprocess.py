"""Tests for the preprocessing pipeline."""

import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocess import (
    load_config,
    load_raw_data,
    drop_null_targets,
    encode_categoricals,
    engineer_features,
    split_data,
    compute_impute_stats,
    apply_imputation,
    get_project_root,
)


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def raw_df(config):
    return load_raw_data(config)


@pytest.fixture
def clean_df(raw_df, config):
    """Raw data with null targets dropped."""
    return drop_null_targets(raw_df, config)


@pytest.fixture
def encoded_df(clean_df, config):
    return encode_categoricals(clean_df, config)


@pytest.fixture
def split_dfs(encoded_df, config):
    """Return (train, val, test) from encoded data."""
    return split_data(encoded_df, config)


@pytest.fixture
def imputed_splits(split_dfs, config):
    """Train/val/test with imputation fit on train only."""
    train, val, test = split_dfs
    stats = compute_impute_stats(train, config)
    train = apply_imputation(train, stats)
    val = apply_imputation(val, stats)
    test = apply_imputation(test, stats)
    return train, val, test


@pytest.fixture
def engineered_splits(imputed_splits):
    """Train/val/test with feature engineering (avg_speed median from train)."""
    train, val, test = imputed_splits
    train_speed = train["Trip_Distance_km"] / (train["Trip_Duration_Minutes"] / 60)
    train_speed = train_speed.replace([np.inf, -np.inf], np.nan)
    median_speed = train_speed.median()
    train = engineer_features(train, avg_speed_median=median_speed)
    val = engineer_features(val, avg_speed_median=median_speed)
    test = engineer_features(test, avg_speed_median=median_speed)
    return train, val, test


class TestDropNullTargets:
    def test_no_null_target(self, clean_df, config):
        """Target should have no nulls after dropping."""
        assert clean_df[config["features"]["target"]].isnull().sum() == 0

    def test_shape_after_drop(self, raw_df, clean_df):
        """Should lose only rows where target was null."""
        max_dropped = raw_df[raw_df.columns[-1]].isnull().sum()
        assert len(clean_df) >= len(raw_df) - max_dropped


class TestEncoding:
    def test_encoded_columns_exist(self, encoded_df):
        """Encoded columns should be present."""
        expected = ["Traffic_encoded", "Weather_encoded",
                    "Time_of_Day_encoded", "Day_of_Week_encoded"]
        for col in expected:
            assert col in encoded_df.columns, f"Missing {col}"

    def test_original_categoricals_dropped(self, encoded_df, config):
        """Original categorical columns should be removed."""
        for col in config["features"]["raw_categorical"]:
            assert col not in encoded_df.columns

    def test_encoding_values_valid(self, encoded_df):
        """Encoded values should be within expected ranges (NaNs allowed pre-imputation)."""
        valid_or_nan = encoded_df["Traffic_encoded"].dropna().isin([0, 1, 2]).all()
        assert valid_or_nan


class TestImputationNoLeakage:
    def test_impute_stats_from_train_only(self, split_dfs, config):
        """Imputation stats should come from train split only."""
        train, val, test = split_dfs
        stats = compute_impute_stats(train, config)
        # Verify the median matches train, not full data
        for col in config["features"]["raw_numeric"]:
            assert stats["numeric"][col] == pytest.approx(train[col].median())

    def test_no_nulls_after_imputation(self, imputed_splits, config):
        """No nulls in numeric or encoded columns after imputation."""
        train, val, test = imputed_splits
        numeric_cols = config["features"]["raw_numeric"]
        for name, split_df in [("train", train), ("val", val), ("test", test)]:
            for col in numeric_cols:
                assert split_df[col].isnull().sum() == 0, f"Nulls remain in {col} of {name}"


class TestFeatureEngineering:
    def test_engineered_features_exist(self, engineered_splits):
        """All engineered features should be present in all splits."""
        expected = ["distance_cost", "duration_cost", "metered_fare",
                    "avg_speed_kmh", "traffic_distance", "weather_duration",
                    "is_peak_hour", "is_weekend", "peak_traffic"]
        for name, split_df in zip(["train", "val", "test"], engineered_splits):
            for col in expected:
                assert col in split_df.columns, f"Missing {col} in {name}"

    def test_no_inf_values(self, engineered_splits):
        """No infinite values in any numeric column."""
        for name, split_df in zip(["train", "val", "test"], engineered_splits):
            numeric = split_df.select_dtypes(include=[np.number])
            assert not np.isinf(numeric.values).any(), f"Inf values in {name}"

    def test_metered_fare_formula(self, engineered_splits):
        """metered_fare = Base_Fare + distance_cost + duration_cost."""
        train = engineered_splits[0]
        expected = (train["Base_Fare"] + train["distance_cost"] + train["duration_cost"])
        np.testing.assert_allclose(train["metered_fare"].values, expected.values, rtol=1e-10)

    def test_avg_speed_median_from_train(self, imputed_splits):
        """avg_speed median fill should use train value, not val/test."""
        train, val, test = imputed_splits
        train_speed = train["Trip_Distance_km"] / (train["Trip_Duration_Minutes"] / 60)
        train_speed = train_speed.replace([np.inf, -np.inf], np.nan)
        train_median = train_speed.median()

        # Engineer with explicit train median
        val_eng = engineer_features(val, avg_speed_median=train_median)
        assert not val_eng["avg_speed_kmh"].isnull().any()


class TestSplitting:
    def test_no_data_leakage(self, split_dfs):
        """Train/val/test sets should not share rows."""
        train, val, test = split_dfs
        train_idx = set(train.index)
        val_idx = set(val.index)
        test_idx = set(test.index)
        assert len(train_idx & val_idx) == 0, "Train/val overlap"
        assert len(train_idx & test_idx) == 0, "Train/test overlap"
        assert len(val_idx & test_idx) == 0, "Val/test overlap"

    def test_split_sizes(self, split_dfs):
        """Splits should be approximately correct proportions."""
        train, val, test = split_dfs
        total = len(train) + len(val) + len(test)
        test_ratio = len(test) / total
        assert 0.15 < test_ratio < 0.25, f"Test ratio {test_ratio} out of range"

    def test_no_nulls_in_final_splits(self, engineered_splits, config):
        """Final splits should have no nulls in model features."""
        model_features = config["features"]["model_features"]
        for name, split_df in zip(["train", "val", "test"], engineered_splits):
            for col in model_features:
                assert split_df[col].isnull().sum() == 0, \
                    f"Nulls in {col} of {name} split"
