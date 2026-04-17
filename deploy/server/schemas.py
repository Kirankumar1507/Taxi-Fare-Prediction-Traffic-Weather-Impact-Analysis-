"""Pydantic request/response schemas for the Fare Prediction API."""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class TripInput(BaseModel):
    """Single trip input matching the 10 raw fields expected by FarePredictor."""

    Trip_Distance_km: float = Field(..., ge=0, description="Trip distance in kilometers")
    Passenger_Count: float = Field(..., ge=0, description="Number of passengers")
    Base_Fare: float = Field(..., ge=0, description="Base fare in dollars")
    Per_Km_Rate: float = Field(..., ge=0, description="Rate per kilometer")
    Per_Minute_Rate: float = Field(..., ge=0, description="Rate per minute")
    Trip_Duration_Minutes: float = Field(..., ge=0, description="Trip duration in minutes")
    Traffic_Conditions: Literal["Low", "Medium", "High"] = Field(
        ..., description="Traffic level"
    )
    Weather: Literal["Clear", "Rain", "Snow"] = Field(
        ..., description="Weather condition"
    )
    Time_of_Day: Literal["Morning", "Afternoon", "Evening", "Night"] = Field(
        ..., description="Time of day bucket"
    )
    Day_of_Week: Literal["Weekday", "Weekend"] = Field(
        ..., description="Weekday or Weekend"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "Trip_Distance_km": 19.35,
                    "Passenger_Count": 3,
                    "Base_Fare": 3.56,
                    "Per_Km_Rate": 0.80,
                    "Per_Minute_Rate": 0.32,
                    "Trip_Duration_Minutes": 53.82,
                    "Traffic_Conditions": "Low",
                    "Weather": "Clear",
                    "Time_of_Day": "Morning",
                    "Day_of_Week": "Weekday",
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Response for a single prediction."""

    predicted_fare: float = Field(..., description="Predicted trip fare in dollars")
    model_name: str = Field(..., description="Name of the model used")
    model_version: str = Field(..., description="Version of the model")


class BatchPredictionRequest(BaseModel):
    """Batch prediction request — list of trips."""

    trips: List[TripInput] = Field(..., min_length=1, max_length=100)


class BatchPredictionResponse(BaseModel):
    """Response for batch predictions."""

    predictions: List[float]
    model_name: str
    model_version: str
    count: int


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    model_name: str
    model_version: str
    feature_count: int


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    detail: Optional[str] = None
