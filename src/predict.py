"""
Inference entrypoint: load model + config, accept input, return predictions.
Run standalone: python -m src.predict
"""

import numpy as np
import pandas as pd
import joblib
import yaml
from pathlib import Path
from catboost import CatBoostRegressor

from src.preprocess import get_project_root, load_config


NUMERIC_FIELDS = [
    "Trip_Distance_km", "Passenger_Count", "Base_Fare",
    "Per_Km_Rate", "Per_Minute_Rate", "Trip_Duration_Minutes",
]

CATEGORICAL_FIELDS = {
    "Traffic_Conditions": ["Low", "Medium", "High"],
    "Weather": ["Clear", "Rain", "Snow"],
    "Time_of_Day": ["Morning", "Afternoon", "Evening", "Night"],
    "Day_of_Week": ["Weekday", "Weekend"],
}


class FarePredictor:
    """Load a trained model and predict taxi fares from raw input."""

    def __init__(self, model_name: str = "catboost"):
        self.config = load_config()
        self.root = get_project_root()
        self.model_name = model_name
        self.feature_names = joblib.load(
            self.root / self.config["paths"]["models_dir"] / "feature_names.joblib"
        )
        self.model = self._load_model()
        self.encoding = self.config["encoding"]

    def _load_model(self):
        """Load model artifact from disk."""
        models_dir = self.root / self.config["paths"]["models_dir"]
        if self.model_name == "catboost":
            model_path = models_dir / "catboost_v1.cbm"
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Model not found at {model_path}. Run 'python -m src.train' first."
                )
            model = CatBoostRegressor()
            model.load_model(str(model_path))
        elif self.model_name == "xgboost":
            model_path = models_dir / "xgboost_v1.joblib"
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Model not found at {model_path}. Run 'python -m src.train' first."
                )
            model = joblib.load(model_path)
        elif self.model_name == "ridge":
            model_path = models_dir / "ridge_v1.joblib"
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Model not found at {model_path}. Run 'python -m src.train' first."
                )
            model = joblib.load(model_path)
        else:
            raise ValueError(f"Unknown model: {self.model_name}. Choose from: catboost, xgboost, ridge")
        return model

    def predict(self, input_data: dict) -> float:
        """Predict fare from a single trip dict.

        Required keys: Trip_Distance_km, Time_of_Day, Day_of_Week,
            Passenger_Count, Traffic_Conditions, Weather,
            Base_Fare, Per_Km_Rate, Per_Minute_Rate, Trip_Duration_Minutes
        """
        self._validate(input_data)
        features = self._build_features(input_data)
        X = np.array([features])
        prediction = self.model.predict(X)[0]
        return round(float(prediction), 4)

    def predict_batch(self, df: pd.DataFrame) -> np.ndarray:
        """Predict fares for a DataFrame of trips."""
        rows = []
        for _, row in df.iterrows():
            rows.append(self._build_features(row.to_dict()))
        X = np.array(rows)
        return self.model.predict(X)

    def _validate(self, data: dict) -> None:
        """Check required fields are present with correct types and ranges."""
        required = list(NUMERIC_FIELDS) + list(CATEGORICAL_FIELDS.keys())
        missing = [k for k in required if k not in data]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")

        # Type and range checks for numeric fields
        for field in NUMERIC_FIELDS:
            val = data[field]
            if not isinstance(val, (int, float)):
                raise TypeError(f"{field} must be numeric, got {type(val).__name__}")
            if np.isnan(val) or np.isinf(val):
                raise ValueError(f"{field} must be finite, got {val}")
            if val < 0:
                raise ValueError(f"{field} cannot be negative, got {val}")

        # Categorical value checks
        for field, valid_values in CATEGORICAL_FIELDS.items():
            val = data[field]
            if val not in valid_values:
                raise ValueError(
                    f"{field} must be one of {valid_values}, got '{val}'"
                )

    def _build_features(self, data: dict) -> list:
        """Transform raw input dict into model feature vector."""
        # Encode categoricals
        traffic_enc = self.encoding["Traffic_Conditions"][data["Traffic_Conditions"]]
        weather_enc = self.encoding["Weather"][data["Weather"]]
        tod_enc = self.encoding["Time_of_Day"][data["Time_of_Day"]]
        dow_enc = self.encoding["Day_of_Week"][data["Day_of_Week"]]

        dist = float(data["Trip_Distance_km"])
        dur = float(data["Trip_Duration_Minutes"])
        per_km = float(data["Per_Km_Rate"])
        per_min = float(data["Per_Minute_Rate"])
        base = float(data["Base_Fare"])

        # Engineered features
        distance_cost = dist * per_km
        duration_cost = dur * per_min
        metered_fare = base + distance_cost + duration_cost
        avg_speed = dist / (dur / 60) if dur > 0 else 0
        traffic_distance = traffic_enc * dist
        weather_duration = weather_enc * dur
        is_peak = 1 if tod_enc in (0, 2) else 0  # Morning or Evening
        is_weekend = dow_enc
        peak_traffic = is_peak * traffic_enc

        # Must match config.features.model_features order exactly
        features = [
            dist,
            float(data["Passenger_Count"]),
            base,
            per_km,
            per_min,
            dur,
            tod_enc,
            dow_enc,
            traffic_enc,
            weather_enc,
            distance_cost,
            duration_cost,
            metered_fare,
            avg_speed,
            traffic_distance,
            weather_duration,
            is_peak,
            is_weekend,
            peak_traffic,
        ]

        if len(features) != len(self.feature_names):
            raise ValueError(
                f"Feature count mismatch: built {len(features)}, "
                f"expected {len(self.feature_names)}"
            )

        return features


def main():
    """Demo prediction with a sample trip."""
    predictor = FarePredictor("catboost")

    sample = {
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

    predicted_fare = predictor.predict(sample)
    print(f"Sample trip prediction: ${predicted_fare:.2f}")
    print(f"(Actual from dataset row 1: $36.26)")


if __name__ == "__main__":
    main()
