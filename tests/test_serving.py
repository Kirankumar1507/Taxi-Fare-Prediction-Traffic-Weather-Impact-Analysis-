"""Unit tests for the FastAPI serving layer.

Uses FastAPI's TestClient so no live server is needed.
"""

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_TRIP = {
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


@pytest.fixture(scope="module")
def client():
    """Create a TestClient with the real model loaded."""
    from deploy.server.rest_server import app

    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
class TestHealth:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "healthy"
        assert body["model_loaded"] is True
        assert body["feature_count"] == 19

    def test_health_response_schema(self, client):
        body = client.get("/health").json()
        for key in ("status", "model_loaded", "model_name", "model_version", "feature_count"):
            assert key in body


# ---------------------------------------------------------------------------
# Single prediction
# ---------------------------------------------------------------------------
class TestPredict:
    def test_predict_returns_fare(self, client):
        resp = client.post("/predict", json=SAMPLE_TRIP)
        assert resp.status_code == 200
        body = resp.json()
        assert "predicted_fare" in body
        assert isinstance(body["predicted_fare"], float)
        assert body["predicted_fare"] > 0

    def test_predict_includes_model_info(self, client):
        body = client.post("/predict", json=SAMPLE_TRIP).json()
        assert body["model_name"] == "catboost"
        assert body["model_version"] == "v1"

    def test_predict_missing_field(self, client):
        bad_trip = {k: v for k, v in SAMPLE_TRIP.items() if k != "Weather"}
        resp = client.post("/predict", json=bad_trip)
        assert resp.status_code == 422

    def test_predict_invalid_categorical(self, client):
        bad_trip = {**SAMPLE_TRIP, "Weather": "Tornado"}
        resp = client.post("/predict", json=bad_trip)
        assert resp.status_code == 422

    def test_predict_negative_distance(self, client):
        bad_trip = {**SAMPLE_TRIP, "Trip_Distance_km": -5}
        resp = client.post("/predict", json=bad_trip)
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Batch prediction
# ---------------------------------------------------------------------------
class TestBatchPredict:
    def test_batch_predict(self, client):
        resp = client.post("/predict/batch", json={"trips": [SAMPLE_TRIP, SAMPLE_TRIP]})
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 2
        assert len(body["predictions"]) == 2
        assert all(isinstance(f, float) for f in body["predictions"])

    def test_batch_empty_list(self, client):
        resp = client.post("/predict/batch", json={"trips": []})
        assert resp.status_code == 422

    def test_batch_exceeds_limit(self, client):
        resp = client.post("/predict/batch", json={"trips": [SAMPLE_TRIP] * 101})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# OpenAPI docs available
# ---------------------------------------------------------------------------
class TestDocs:
    def test_openapi_json(self, client):
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        assert "paths" in resp.json()
