# Shodak Research Report — Taxi Fare Prediction with Traffic & Weather
**Date:** 2026-04-17 | **Agent:** Shodak (शोधक) | **Project:** taxi_pricing_traffic_weather

---

## 1. SOTA Approaches for Taxi Fare Prediction

**Dominant family: Gradient Boosted Decision Trees (GBDTs)**

| Model | Strengths | Weakness |
|---|---|---|
| **CatBoost** | Native categorical handling, ordered boosting reduces overfitting on small data | Slower training than LGBM (irrelevant at 1k rows) |
| **XGBoost** | L1/L2 regularization, second-order gradients, battle-tested | Needs manual categorical encoding |
| **LightGBM** | Fastest training, leaf-wise growth | **Overfits on small datasets** (<5k rows) — documented risk |
| **Random Forest** | Low variance, robust baseline | Higher bias than GBDTs |
| **Linear/Ridge** | Interpretable, ideal if rate formula dominates | Misses non-linear interactions |

**Key benchmark:** Ensemble of CatBoost + XGBoost + LGBM achieved RMSE 2.88 on NYC taxi data. RF RMSE 1.264 vs Linear Regression 1.718 in published comparison.

**Deep Learning:** BiLSTM + Attention achieved high accuracy but requires large data. **Not recommended at 1000 rows.**

**TabPFN (Nature 2024):** Transformer-based foundation model for tabular data — outperforms tuned GBDTs on datasets ≤1,250 rows with zero hyperparameter tuning. Worth a quick experiment.

---

## 2. Traffic Condition Impact on Pricing

- Traffic_Conditions (Low/Medium/High) acts as a **direct pricing modifier**
- High traffic on long routes disproportionately increases duration-based charges
- **Ordinal encoding** (0, 1, 2) preserves severity ordering — appropriate for tree models

**Key interaction effects to engineer:**
- `Traffic × Trip_Distance_km` — congestion amplifies distance cost
- `Traffic × Trip_Duration_Minutes` — stop-and-go inflates time charges
- `Traffic × Time_of_Day` — peak-hour + high traffic = compound surge

---

## 3. Weather Impact on Pricing

**Empirical findings from literature:**
- Heavy rainfall (>5mm/hr): **+73% average fare**, +107% wait time
- Uber rides: **+22% more rides/hour when raining** vs +5% for traditional taxis
- Extreme temperatures (<5°C or >35°C): **+42% demand increase**

**Encoding:** If weather categories show monotonic price increase with severity (Clear < Cloudy < Rain < Snow), use ordinal. Otherwise, one-hot or pass natively to CatBoost.

**Key interaction:** `Weather × Time_of_Day` — rain during peak hours compounds demand surge.

---

## 4. Feature Engineering Recommendations

### Priority 1 — Algebraic Reconstructions (highest impact)
| Feature | Formula | Rationale |
|---|---|---|
| `distance_cost` | `Trip_Distance_km * Per_Km_Rate` | Distance component of metered fare |
| `duration_cost` | `Trip_Duration_Minutes * Per_Minute_Rate` | Time component |
| `metered_fare` | `Base_Fare + distance_cost + duration_cost` | Near-deterministic estimate; residuals = surge/conditions |
| `avg_speed_kmh` | `Trip_Distance_km / (Trip_Duration_Minutes / 60)` | Implicit traffic proxy |

### Priority 2 — Interaction Features
| Feature | Formula | Rationale |
|---|---|---|
| `traffic_distance` | `Traffic_encoded * Trip_Distance_km` | Congestion × route length |
| `weather_duration` | `Weather_encoded * Trip_Duration_Minutes` | Weather slows trips |
| `peak_traffic` | `is_peak_hour * Traffic_encoded` | Compound surge signal |

### Priority 3 — Temporal Flags
- `is_peak_hour` — Morning + Evening = 1
- `is_weekend` — from Day_of_Week

**Critical note:** `metered_fare` will likely be the single strongest predictor. If Ridge R² > 0.98 on metered_fare alone, the dataset is likely synthetic and tree models risk overfitting noise.

---

## 5. Small Dataset Strategy (1000 rows)

| Aspect | Recommendation |
|---|---|
| CV strategy | **RepeatedKFold** (n_splits=10, n_repeats=3) for stable estimates |
| Hold-out test | Reserve 15–20% (150–200 rows) before any model selection |
| Early stopping | Always use within CV folds (patience=50 rounds) |

**Regularization by model:**
| Model | Key params |
|---|---|
| CatBoost | `depth=4–6`, `l2_leaf_reg=3–10`, `learning_rate=0.05`, `iterations=300–500` |
| XGBoost | `max_depth=3–5`, `n_estimators=100–300`, `learning_rate=0.05–0.1`, `subsample=0.8` |
| Ridge | Tune `alpha` via CV |

---

## 6. Model Selection — Final Ranking

### Rank 1: CatBoost Regressor
- Native categorical handling (no preprocessing for Traffic, Weather, Time_of_Day, Day_of_Week)
- Ordered boosting explicitly designed to reduce overfitting on small samples
- Top performer on datasets ≤1,250 rows in published benchmarks

### Rank 2: Ridge Regression (with engineered features)
- `metered_fare` feature likely explains 85–95% of variance
- Fastest to fit, most interpretable — establishes performance ceiling
- **Run first as baseline**

### Rank 3: XGBoost Regressor
- More regularization knobs than LGBM, less overfit-prone at small sample sizes
- Excellent SHAP support for explainability

### Not Recommended
- LightGBM (overfitting risk at 1k rows)
- Neural networks (insufficient data)

---

## 7. Evaluation Metrics
- **Primary:** RMSE (same units as Trip_Price, penalizes large errors)
- **Secondary:** MAE (robust to outliers), MAPE (percentage-based)
- **Report:** R² (explained variance)
- If Trip_Price is log-skewed, evaluate RMSLE as well

---

## Sources
1. [Taxi Fare Prediction — Multiple ML Models (ResearchGate 2024)](https://www.researchgate.net/publication/382370761)
2. [When Neural Nets Outperform Boosted Trees on Tabular Data (arXiv 2305.02997)](https://arxiv.org/html/2305.02997v4)
3. [Enhancing Ride-Hailing Fare Prediction — Deep Learning (Vilnius Tech)](https://journals.vilniustech.lt/index.php/NTCS/article/view/23932)
4. [NYC Taxi/Uber Weather Shocks — Empirical Analysis (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S0167268118301598)
5. [Weather-Aware AI Systems (arXiv 2507.17099)](https://arxiv.org/html/2507.17099)
6. [TabPFN: Accurate Predictions on Small Data (Nature 2024)](https://www.nature.com/articles/s41586-024-08328-6)
7. [Overfitting in GBDT (Google ML Guide)](https://developers.google.com/machine-learning/decision-forests/overfitting-gbdt)
8. [Peak Hour / Congested Area Taxi Surcharges (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S0967070X22000683)
9. [Dynamic Ride Pricing Model Using ML (IJSRET 2024)](https://ijsret.com/wp-content/uploads/2024/11/IJSRET_V10_issue6_542.pdf)
