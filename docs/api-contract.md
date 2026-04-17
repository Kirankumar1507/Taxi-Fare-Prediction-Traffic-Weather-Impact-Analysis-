# API Contract — Taxi Fare Prediction

Base URL: `http://localhost:8000` (local) or `https://<endpoint>/` (deployed)

---

## GET /health

Liveness + readiness probe.

**Response 200**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "catboost",
  "model_version": "v1",
  "feature_count": 19
}
```

**Response 503** — model not loaded or feature schema mismatch.

---

## POST /predict

Predict fare for a single trip.

**Request body**
```json
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
  "Day_of_Week": "Weekday"
}
```

| Field | Type | Constraints |
|-------|------|-------------|
| Trip_Distance_km | float | >= 0 |
| Passenger_Count | float | >= 0 |
| Base_Fare | float | >= 0 |
| Per_Km_Rate | float | >= 0 |
| Per_Minute_Rate | float | >= 0 |
| Trip_Duration_Minutes | float | >= 0 |
| Traffic_Conditions | string | "Low" / "Medium" / "High" |
| Weather | string | "Clear" / "Rain" / "Snow" |
| Time_of_Day | string | "Morning" / "Afternoon" / "Evening" / "Night" |
| Day_of_Week | string | "Weekday" / "Weekend" |

**Response 200**
```json
{
  "predicted_fare": 36.12,
  "model_name": "catboost",
  "model_version": "v1"
}
```

**Response 422** — validation error (missing field, invalid enum, negative value).

---

## POST /predict/batch

Predict fares for 1-100 trips.

**Request body**
```json
{
  "trips": [
    { ... trip 1 ... },
    { ... trip 2 ... }
  ]
}
```

**Response 200**
```json
{
  "predictions": [36.12, 18.45],
  "model_name": "catboost",
  "model_version": "v1",
  "count": 2
}
```

**Limits:** 1 to 100 trips per request. Returns 422 for empty or oversized batches.

---

## GET /openapi.json

Auto-generated OpenAPI 3.1 schema. Interactive docs at `/docs` (Swagger UI) and `/redoc`.
