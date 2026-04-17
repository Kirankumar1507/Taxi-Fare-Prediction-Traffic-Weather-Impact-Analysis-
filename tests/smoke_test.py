"""Post-deploy integration / smoke test.

Run against a live server:
    SERVER_URL=http://localhost:8000 python -m pytest tests/smoke_test.py -v

Default: http://localhost:8000
"""

import os

import pytest
import requests

SERVER_URL = os.getenv("SERVER_URL", "http://localhost:8000")

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


def _url(path: str) -> str:
    return f"{SERVER_URL}{path}"


@pytest.fixture(scope="module")
def live_server():
    """Skip all tests if the server is unreachable."""
    try:
        requests.get(_url("/health"), timeout=5)
    except requests.ConnectionError:
        pytest.skip(f"Server not reachable at {SERVER_URL}")


class TestSmoke:
    def test_health(self, live_server):
        resp = requests.get(_url("/health"), timeout=5)
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"

    def test_single_prediction(self, live_server):
        resp = requests.post(_url("/predict"), json=SAMPLE_TRIP, timeout=10)
        assert resp.status_code == 200
        fare = resp.json()["predicted_fare"]
        # Sanity: fare should be positive and within a reasonable range
        assert 1.0 < fare < 500.0

    def test_batch_prediction(self, live_server):
        payload = {"trips": [SAMPLE_TRIP, SAMPLE_TRIP]}
        resp = requests.post(_url("/predict/batch"), json=payload, timeout=10)
        assert resp.status_code == 200
        assert resp.json()["count"] == 2

    def test_validation_error(self, live_server):
        bad = {**SAMPLE_TRIP, "Weather": "Tornado"}
        resp = requests.post(_url("/predict"), json=bad, timeout=10)
        assert resp.status_code == 422

    def test_openapi_available(self, live_server):
        resp = requests.get(_url("/openapi.json"), timeout=5)
        assert resp.status_code == 200
