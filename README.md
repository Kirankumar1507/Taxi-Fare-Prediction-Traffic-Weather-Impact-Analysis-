# Taxi Fare Prediction: How Traffic & Weather Shape What You Pay

> **An end-to-end ML system that predicts taxi fares by decomposing pricing into its base metered component and the dynamic surcharges driven by traffic congestion and adverse weather.**

<p align="center">
  <img src="reports/figures/17_price_sensitivity_heatmap.png" width="48%"/>
  <img src="reports/figures/07_traffic_weather_heatmap.png" width="48%"/>
</p>

---

## The Problem

Every taxi ride has a simple formula at its core: **base fare + (distance x rate) + (time x rate)**. But real-world pricing isn't that clean. Traffic jams inflate trip duration. Snow slows everything down. Rush hour compounds both effects.

This project investigates a fundamental question in ride-hailing economics:

> **How much of a taxi fare is explained by the base metered formula, and how much is driven by traffic and weather conditions?**

I built a full ML pipeline — from research to deployment — to decompose and predict these pricing dynamics.

---

## Key Findings

### 1. The metered formula explains ~88% of fare variance

The algebraic reconstruction `metered_fare = Base_Fare + (Distance x Per_Km_Rate) + (Duration x Per_Minute_Rate)` alone achieves R² = 0.88. The remaining 12% is where traffic and weather effects live.

<p align="center">
  <img src="reports/figures/10_metered_vs_actual.png" width="65%"/>
</p>

### 2. High traffic + snow = the most expensive combination

The Traffic x Weather interaction heatmap reveals that **High Traffic + Snow** pushes the average fare to **$85.60** — a **62% premium** over Low Traffic + Clear ($52.70).

| Condition | Avg Fare | Premium vs Baseline |
|-----------|----------|-------------------|
| Low Traffic + Clear | $54.30 | baseline |
| High Traffic + Clear | $64.30 | +18% |
| High Traffic + Rain | $64.50 | +19% |
| **High Traffic + Snow** | **$85.60** | **+58%** |

### 3. Weather premium scales with distance

Short trips (<15 km) see minimal weather impact. But on long trips (>35 km), **snow adds ~$8 and rain adds ~$6** on average. Weather's pricing effect is proportional to exposure time.

<p align="center">
  <img src="reports/figures/18_weather_premium.png" width="55%"/>
</p>

### 4. Traffic affects price through duration, not directly

The violin plots show that high traffic has a wider price spread but only a modest median increase ($48 → $55). Traffic's real effect is indirect: it inflates `Trip_Duration_Minutes`, which multiplies with `Per_Minute_Rate`. The `traffic_distance` interaction feature captures this compound effect.

<p align="center">
  <img src="reports/figures/05_price_by_traffic_violin.png" width="48%"/>
  <img src="reports/figures/06_price_by_weather_violin.png" width="48%"/>
</p>

---

## Model Performance

We trained three models and selected CatBoost as champion based on cross-validated performance:

<p align="center">
  <img src="reports/figures/13_model_comparison.png" width="90%"/>
</p>

| Metric | Ridge (Baseline) | XGBoost | CatBoost (Champion) | Target |
|--------|-----------------|---------|--------------------|----|
| **MAE** | $8.25 | $4.13 | **$4.02** | — |
| **RMSE** | $16.71 | $12.10 | **$11.49** | — |
| **MAPE** | 16.25% | 6.11% | **5.99%** | < 20% |
| **R²** | 0.881 | 0.937 | **0.943** | > 0.80 |
| **CV R² (10x3)** | 0.872 | 0.934 | **0.928** | — |

**Why CatBoost over XGBoost?** Nearly identical accuracy, but CatBoost has lower CV variance (std 0.038 vs 0.044), native categorical handling, and ordered boosting that reduces overfitting on our 1,000-row dataset.

<p align="center">
  <img src="reports/figures/14_predicted_vs_actual.png" width="48%"/>
  <img src="reports/figures/15_residual_analysis.png" width="48%"/>
</p>

---

## Explainability (SHAP)

Every prediction is decomposable. SHAP analysis confirms that `metered_fare` dominates, with traffic and weather interactions providing the marginal signal:

<p align="center">
  <img src="reports/figures/16_shap_waterfall.png" width="70%"/>
</p>

**Top 5 features by mean |SHAP|:**

| Rank | Feature | Mean |SHAP| | Interpretation |
|------|---------|-------------|----------------|
| 1 | `metered_fare` | 15.39 | Base formula — the pricing backbone |
| 2 | `distance_cost` | 5.64 | Distance component of the metered fare |
| 3 | `Trip_Distance_km` | 3.41 | Raw distance captures non-linear effects beyond rate x distance |
| 4 | `duration_cost` | 1.89 | Time component — amplified by traffic |
| 5 | `Trip_Duration_Minutes` | 1.06 | Raw duration captures congestion effects |

<p align="center">
  <img src="reports/shap_summary.png" width="48%"/>
  <img src="reports/shap_importance_bar.png" width="48%"/>
</p>

---

## Exploratory Data Analysis

### Data Quality

The dataset has **exactly 5% nulls in every column** — a uniform MCAR (Missing Completely At Random) pattern confirming synthetic generation. Imputation uses median (numeric) and mode (categorical), fit on training data only to prevent leakage.

<p align="center">
  <img src="reports/figures/01_missing_data_heatmap.png" width="48%"/>
  <img src="reports/figures/02_target_distribution.png" width="48%"/>
</p>

### Feature Relationships

<p align="center">
  <img src="reports/figures/03_correlation_matrix.png" width="48%"/>
  <img src="reports/figures/12_speed_vs_price.png" width="48%"/>
</p>

### Temporal Pricing Patterns

<p align="center">
  <img src="reports/figures/08_time_of_day_pricing.png" width="48%"/>
  <img src="reports/figures/09_weekday_weekend_faceted.png" width="48%"/>
</p>

---

## Feature Engineering

9 features engineered from domain knowledge, validated by SHAP importance:

| Priority | Feature | Formula | Why |
|----------|---------|---------|-----|
| 1 | `distance_cost` | Distance x Per_Km_Rate | Distance component of meter |
| 1 | `duration_cost` | Duration x Per_Minute_Rate | Time component of meter |
| 1 | `metered_fare` | Base + distance_cost + duration_cost | Full metered reconstruction |
| 1 | `avg_speed_kmh` | Distance / (Duration / 60) | Implicit congestion proxy |
| 2 | `traffic_distance` | Traffic_encoded x Distance | Congestion amplifies distance cost |
| 2 | `weather_duration` | Weather_encoded x Duration | Bad weather extends trip time |
| 3 | `is_peak_hour` | 1 if Morning or Evening | Rush hour flag |
| 3 | `is_weekend` | 1 if Weekend | Weekend demand pattern |
| 3 | `peak_traffic` | is_peak_hour x Traffic_encoded | Compound surge signal |

---

## ML Pipeline Design

### Leakage-Free Preprocessing

A critical design decision: **imputation statistics are computed on the training set only**, then applied to validation and test sets. This prevents the subtle but common data leakage pattern where test set information leaks through imputation medians.

```
Load raw data
    -> Drop null targets
    -> Encode categoricals (deterministic mapping — no leakage risk)
    -> Split into train / val / test
    -> Compute imputation stats on TRAIN only
    -> Apply imputation to all splits
    -> Engineer features (avg_speed median from TRAIN only)
```

### Overfitting Controls

With only 1,000 rows and 19 features, overfitting is the primary risk:

- **CatBoost:** depth=3, l2_leaf_reg=12, min_data_in_leaf=10, learning_rate=0.03
- **XGBoost:** max_depth=3, reg_lambda=5.0, min_child_weight=5, subsample=0.7
- **Evaluation:** RepeatedKFold (10 splits x 3 repeats) as primary metric, single test split as secondary
- **Early stopping:** 50-round patience on validation loss

---

## Quick Start

### Prerequisites

```bash
Python 3.9+
pip install -r requirements.txt
```

### Run the Full Pipeline

```bash
# 1. Preprocess (split, impute, engineer features)
python -m src.preprocess

# 2. Train all models (Ridge, XGBoost, CatBoost)
python -m src.train

# 3. Evaluate champion on test set + SHAP
python -m src.evaluate

# 4. Generate all analytics visualizations
python -m src.generate_analytics

# 5. Predict a single trip
python -m src.predict
```

### Python API

```python
from src.predict import FarePredictor

predictor = FarePredictor("catboost")
fare = predictor.predict({
    "Trip_Distance_km": 25.0,
    "Time_of_Day": "Evening",
    "Day_of_Week": "Weekday",
    "Passenger_Count": 2,
    "Traffic_Conditions": "High",
    "Weather": "Rain",
    "Base_Fare": 3.50,
    "Per_Km_Rate": 1.20,
    "Per_Minute_Rate": 0.30,
    "Trip_Duration_Minutes": 75.0,
})
print(f"Predicted fare: ${fare:.2f}")
```

### REST API (Local)

```bash
pip install -r requirements-serve.txt
python -m deploy.server.rest_server
# Swagger docs at http://localhost:8000/docs
```

### Docker

```bash
make -f deploy/Makefile build && make -f deploy/Makefile run
# Health: curl http://localhost:8000/health
# Predict: curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{...}'
```

---

## Testing

```bash
python -m pytest tests/ -v --cov=src --cov-report=term-missing
```

**62 tests passing | 83% coverage** across 7 test files:

| Test File | Tests | What It Covers |
|-----------|-------|----------------|
| `test_preprocess.py` | 14 | Imputation (train-only), encoding, feature engineering, splits, no leakage |
| `test_train.py` | 6 | Feature shapes, MAPE edge cases, metric computation |
| `test_predict.py` | 13 | Input validation (types, ranges, categoricals), prediction sanity |
| `test_evaluate.py` | 10 | Model loading, R²/MAPE thresholds, report generation |
| `test_pipeline.py` | 8 | End-to-end preprocessing, model training integration |
| `test_serving.py` | 11 | API health, predict, batch, validation errors |
| `smoke_test.py` | — | Post-deploy integration (skipped when server not running) |

---

## Deployment Architecture

```
                    GitHub Actions
                    (test.yml / build-api.yml)
                         |
                    Docker Image
                    (python:3.9-slim)
                         |
              +----------+----------+
              |                     |
         Docker Compose        SageMaker
         (local dev)           Endpoint
              |                     |
         FastAPI REST          inference.py
         /health               model_fn()
         /predict              predict_fn()
         /predict/batch        output_fn()
              |
         CloudWatch Metrics
         Prediction Logs (JSONL)
         Drift Detection (PSI)
```

See [docs/architecture.md](docs/architecture.md) for the full system diagram.

---

## Project Structure

```
taxi_pricing_route_weather/
├── src/                         # ML pipeline
│   ├── preprocess.py            # Load, impute (train-only), encode, split
│   ├── train.py                 # Ridge + XGBoost + CatBoost
│   ├── evaluate.py              # Test metrics + SHAP
│   ├── predict.py               # FarePredictor inference class
│   └── generate_analytics.py    # EDA & visualization generator
├── configs/config.yaml          # All hyperparams, features, encodings
├── models/                      # Artifacts (gitignored)
├── tests/                       # 62 tests, 83% coverage
├── reports/
│   ├── figures/                 # 18 EDA & analysis plots
│   ├── final_report.md          # Model evaluation report
│   └── INSIGHTS.md              # Business insights & interpretation
├── deploy/
│   ├── server/                  # FastAPI REST API
│   ├── sagemaker/               # SageMaker inference config
│   ├── monitoring/              # CloudWatch, prediction logs, drift
│   ├── Dockerfile               # Multi-stage, non-root
│   └── docker-compose.yaml      # Local dev
├── .github/workflows/           # CI/CD (test + build)
├── docs/                        # API contract, env vars, runbook, architecture
└── requirements.txt
```

---

## Research References

This project was informed by published research on taxi fare prediction and dynamic pricing:

1. [Taxi Fare Prediction Based on Multiple ML Models](https://www.researchgate.net/publication/382370761) — RF RMSE 1.264 vs Linear 1.718
2. [When Neural Nets Outperform Boosted Trees](https://arxiv.org/html/2305.02997v4) — CatBoost best for datasets <1,250 rows
3. [NYC Taxi/Uber Weather Shocks](https://www.sciencedirect.com/science/article/abs/pii/S0167268118301598) — Rain increases Uber rides +22%
4. [Weather-Aware AI Systems](https://arxiv.org/html/2507.17099) — Heavy rain +73% fares, extreme temps +42% demand
5. [TabPFN: Accurate Predictions on Small Data](https://www.nature.com/articles/s41586-024-08328-6) — Transformer baseline for small tabular datasets

Full research report: [reports/shodak_research_fare_prediction.md](reports/shodak_research_fare_prediction.md)

---

## Built With

| Component | Technology |
|-----------|------------|
| ML Framework | scikit-learn, XGBoost, CatBoost |
| Explainability | SHAP |
| API | FastAPI + Pydantic |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Monitoring | CloudWatch + custom JSONL |
| Visualization | Matplotlib, Seaborn, Plotly |
| Cloud | AWS SageMaker |

---

## License

This project is open source and available for educational and research purposes.

---

<p align="center">
  <i>Built as an end-to-end ML engineering showcase — from research to production deployment.</i>
</p>
