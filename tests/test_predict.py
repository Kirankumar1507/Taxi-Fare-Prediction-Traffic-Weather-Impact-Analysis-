"""Tests for the prediction/inference pipeline."""

import sys
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.predict import FarePredictor, NUMERIC_FIELDS, CATEGORICAL_FIELDS


@pytest.fixture
def predictor():
    return FarePredictor("catboost")


@pytest.fixture
def valid_input():
    return {
        "Trip_Distance_km": 19.35,
        "Time_of_Day": "Morning",
        "Day_of_Week": "Weekday",
        "Passenger_Count": 3,
        "Traffic_Conditions": "Low",
        "Weather": "Clear",
        "Base_Fare": 3.56,
        "Per_Km_Rate": 0.80,
        "Per_Minute_Rate": 0.32,
        "Trip_Duration_Minutes": 53.82,
    }


class TestValidation:
    def test_valid_input_passes(self, predictor, valid_input):
        """Valid input should not raise."""
        predictor._validate(valid_input)

    def test_missing_field_raises(self, predictor, valid_input):
        """Missing required field should raise ValueError."""
        del valid_input["Trip_Distance_km"]
        with pytest.raises(ValueError, match="Missing required fields"):
            predictor._validate(valid_input)

    def test_non_numeric_raises(self, predictor, valid_input):
        """String in numeric field should raise TypeError."""
        valid_input["Trip_Distance_km"] = "not_a_number"
        with pytest.raises(TypeError, match="must be numeric"):
            predictor._validate(valid_input)

    def test_negative_value_raises(self, predictor, valid_input):
        """Negative numeric value should raise ValueError."""
        valid_input["Trip_Distance_km"] = -5.0
        with pytest.raises(ValueError, match="cannot be negative"):
            predictor._validate(valid_input)

    def test_invalid_categorical_raises(self, predictor, valid_input):
        """Invalid categorical value should raise ValueError."""
        valid_input["Weather"] = "Tornado"
        with pytest.raises(ValueError, match="must be one of"):
            predictor._validate(valid_input)

    def test_nan_raises(self, predictor, valid_input):
        """NaN in numeric field should raise ValueError."""
        valid_input["Base_Fare"] = float("nan")
        with pytest.raises(ValueError, match="must be finite"):
            predictor._validate(valid_input)


class TestBuildFeatures:
    def test_feature_count_matches(self, predictor, valid_input):
        """Feature vector length should match model expectation."""
        features = predictor._build_features(valid_input)
        assert len(features) == len(predictor.feature_names)

    def test_metered_fare_in_features(self, predictor, valid_input):
        """metered_fare should be correctly computed."""
        features = predictor._build_features(valid_input)
        mf_idx = predictor.feature_names.index("metered_fare")
        expected = 3.56 + (19.35 * 0.80) + (53.82 * 0.32)
        assert features[mf_idx] == pytest.approx(expected)

    def test_all_features_numeric(self, predictor, valid_input):
        """All features should be numeric (int or float)."""
        features = predictor._build_features(valid_input)
        for i, f in enumerate(features):
            assert isinstance(f, (int, float)), \
                f"Feature {predictor.feature_names[i]} is {type(f)}"


class TestPredict:
    def test_returns_float(self, predictor, valid_input):
        """Prediction should be a float."""
        result = predictor.predict(valid_input)
        assert isinstance(result, float)

    def test_prediction_positive(self, predictor, valid_input):
        """Fare prediction should be positive for valid input."""
        result = predictor.predict(valid_input)
        assert result > 0

    def test_prediction_reasonable_range(self, predictor, valid_input):
        """Prediction should be in a reasonable fare range."""
        result = predictor.predict(valid_input)
        # For a 19km, 54min trip: expect roughly $20-$80
        assert 5 < result < 200, f"Prediction ${result} outside reasonable range"

    def test_longer_trip_costs_more(self, predictor, valid_input):
        """A longer trip should cost more than a shorter one."""
        short_trip = valid_input.copy()
        short_trip["Trip_Distance_km"] = 5.0
        short_trip["Trip_Duration_Minutes"] = 15.0

        long_trip = valid_input.copy()
        long_trip["Trip_Distance_km"] = 50.0
        long_trip["Trip_Duration_Minutes"] = 120.0

        short_fare = predictor.predict(short_trip)
        long_fare = predictor.predict(long_trip)
        assert long_fare > short_fare
