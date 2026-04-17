# Analytical Insights & Business Interpretation

**Project:** Taxi Fare Prediction — Traffic & Weather Impact Analysis  
**Date:** 2026-04-17  
**Champion Model:** CatBoost (R² = 0.943, MAPE = 5.99%)

---

## Executive Summary

This analysis decomposes taxi fare pricing into two layers: (1) the deterministic **metered fare** driven by distance, duration, and base rates, and (2) the stochastic **condition premium** driven by traffic congestion and weather severity. Our key finding is that the metered formula alone explains 88% of fare variance, but the remaining 12% — concentrated in high-traffic and adverse-weather scenarios — represents the dynamic pricing signal that justifies ML over a simple calculator.

---

## 1. Pricing Decomposition

### Finding: Two-Layer Fare Structure

Every taxi fare in this dataset can be decomposed as:

```
Trip_Price = metered_fare + condition_premium + noise
```

Where:
- **metered_fare** = Base_Fare + (Trip_Distance_km x Per_Km_Rate) + (Trip_Duration_Minutes x Per_Minute_Rate)
- **condition_premium** = f(Traffic, Weather, Time_of_Day, interactions)
- **noise** = unexplained variance (rounding, negotiation, promotions)

| Component | R² Contribution | Interpretation |
|-----------|----------------|----------------|
| Metered fare alone | 0.881 | Base formula captures 88% of pricing |
| + Traffic/Weather features | 0.943 | Condition features add 6.2 percentage points |
| Unexplained | 0.057 | Noise floor — irreducible at this sample size |

**Business implication:** A simple calculator using rate cards would be 88% accurate. ML adds value in the edge cases — precisely the high-fare, high-traffic scenarios where pricing errors are most costly.

![Metered vs Actual](figures/10_metered_vs_actual.png)

*The scatter shows tight alignment along the y=x diagonal for low fares, with systematic deviation at higher fares — these are the trips where traffic/weather conditions push pricing above the metered formula.*

---

## 2. Traffic Impact Analysis

### Finding: Traffic Affects Price Through Duration, Not Directly

| Traffic Level | Median Price | Mean Price | Price Std Dev |
|---------------|-------------|------------|---------------|
| Low | $48.40 | $54.30 | $37.60 |
| Medium | $48.80 | $54.60 | $37.90 |
| High | $55.10 | $64.50 | $47.20 |

The **median** increase from Low to High traffic is modest (+14%), but the **standard deviation** jumps by 25%. High traffic doesn't uniformly increase fares — it increases fare *variability*. This is because traffic's pricing effect is mediated by trip duration:

```
High traffic → longer Trip_Duration → higher duration_cost → higher fare
```

The `traffic_distance` interaction feature (Traffic_encoded x Trip_Distance_km) captures this: long-distance trips in high traffic see disproportionate fare increases because congestion extends the time-on-road multiplicatively.

**Industry parallel:** This mirrors Uber's surge pricing mechanism, where congestion doesn't directly set the multiplier — instead, increased demand (from the same traffic that slows trips) triggers the dynamic pricing algorithm. Our model captures the supply-side effect (longer trips cost more) without the demand signal.

![Traffic Violin](figures/05_price_by_traffic_violin.png)

---

## 3. Weather Impact Analysis

### Finding: Snow is the Strongest Weather Premium, Scaling with Distance

| Weather | Mean Price | vs Clear Baseline |
|---------|-----------|------------------|
| Clear | $54.30 | baseline |
| Rain | $59.40 | +9.4% |
| Snow | $54.60 (all), $85.60 (high traffic) | +0.6% overall, **+57.6% in high traffic** |

Weather's effect is **non-linear and conditional**:
- **Rain** has a consistent, moderate premium across all conditions
- **Snow** has minimal standalone impact but **amplifies traffic effects dramatically**

This is the interaction effect: Snow + High Traffic creates compound congestion (slower speeds, longer durations, higher per-minute charges).

### Distance-Dependent Weather Premium

| Distance Bucket | Rain Premium | Snow Premium |
|----------------|--------------|--------------|
| Short (<15 km) | -$2.80 | +$3.10 |
| Medium (15-35 km) | +$0.70 | +$3.80 |
| Long (>35 km) | +$6.10 | +$8.00 |

Short-trip rain premium is *negative* — likely a compositional effect (rainy short trips tend to have lower base rates in this dataset). But on long trips, weather premiums become substantial: **$6-8 per trip**.

**Industry context:** Published research (Brodeur & Nield, 2018) found rain increases Uber ride volume by 22% and traditional taxi volume by 5%. Our dataset captures the fare-side effect but not the demand surge — in a real system, weather would also increase surge multipliers through the demand channel, compounding the price increase beyond what we observe here.

![Weather Premium](figures/18_weather_premium.png)

---

## 4. Temporal Pricing Patterns

### Finding: Time of Day Shows No Strong Pricing Signal

| Time of Day | Mean Price | Interpretation |
|-------------|-----------|----------------|
| Morning | $56.20 | Rush hour, but average fares |
| Afternoon | $57.80 | Slightly higher — longer leisure trips? |
| Evening | $56.50 | Evening rush ≈ morning |
| Night | $55.40 | Lowest — shorter trips, less traffic |

Weekday vs. Weekend differences are similarly minimal. This dataset doesn't exhibit the strong temporal surge patterns seen in real ride-hailing data (where Friday night fares can be 2-3x Monday morning).

**Why this matters:** For practitioners building on real data, Time_of_Day and Day_of_Week would likely be much stronger features. In our synthetic dataset, they contribute minimal SHAP importance (rank 10 and 19 respectively), but should not be removed from the feature set — they'll activate on real-world data with genuine temporal demand patterns.

![Time of Day](figures/08_time_of_day_pricing.png)

---

## 5. Model Selection Rationale

### Why CatBoost Over XGBoost?

Both models achieve similar test R² (~0.94), but the selection criteria go beyond headline accuracy:

| Criterion | XGBoost | CatBoost | Winner |
|-----------|---------|----------|--------|
| Test R² | 0.937 | 0.943 | CatBoost |
| CV R² mean | 0.934 | 0.928 | XGBoost (marginally) |
| CV R² std | 0.044 | **0.038** | **CatBoost** (more stable) |
| Train-Val gap | 0.155 | **0.136** | **CatBoost** (less overfit) |
| Categorical handling | Manual encoding required | Native | **CatBoost** |
| Small data robustness | Overfits aggressively without heavy regularization | Ordered boosting designed for small samples | **CatBoost** |

**Decision:** CatBoost wins on stability (lower CV variance), overfitting resistance (smaller train-val gap), and production simplicity (no encoding preprocessing needed for categoricals).

### Why Not Ridge Regression?

Ridge achieves R² = 0.881 — respectable for a linear model. But:
- It misses **non-linear interaction effects** (Traffic x Distance, Weather x Duration)
- MAPE = 16.25% vs CatBoost's 5.99% — Ridge's errors are 2.7x larger in percentage terms
- On high-fare trips ($100+), Ridge systematically underpredicts by $15-30

Ridge serves its purpose as a **diagnostic baseline**: if CatBoost couldn't beat Ridge significantly, it would signal that the tree model is overfitting noise rather than learning real patterns.

![Model Comparison](figures/13_model_comparison.png)

---

## 6. SHAP Interpretability

### Global Feature Importance

| Rank | Feature | Mean |SHAP| | % of Total | Cumulative % |
|------|---------|-------------|------------|--------------|
| 1 | metered_fare | 15.39 | 49.2% | 49.2% |
| 2 | distance_cost | 5.64 | 18.0% | 67.2% |
| 3 | Trip_Distance_km | 3.41 | 10.9% | 78.1% |
| 4 | duration_cost | 1.89 | 6.0% | 84.1% |
| 5 | Trip_Duration_Minutes | 1.06 | 3.4% | 87.5% |
| 6-19 | All others | 3.92 | 12.5% | 100.0% |

**The Pareto principle holds:** The top 5 features account for 87.5% of model output variation. The bottom 14 features contribute 12.5% — but this marginal signal is precisely what separates a $4 MAE model from an $8 MAE model.

### Local Explanation — Median Trip ($49.47)

The SHAP waterfall for a median-priced trip shows:
- **Base prediction** (population mean): $56.34
- **metered_fare** pulls DOWN by $4.11 (this trip's metered fare is below average)
- **distance_cost** pulls DOWN by $4.76 (shorter distance)
- **duration_cost** pushes UP by $1.98 (slightly longer per-km duration — possible medium traffic)
- **Weather_encoded** pushes UP by $0.53 (snow increases prediction)
- **Final prediction:** $49.25 (actual: $49.47)

![SHAP Waterfall](figures/16_shap_waterfall.png)

**Industry standard:** SHAP explanations satisfy the explainability requirement for audit trails. Each prediction can be decomposed into additive feature contributions, enabling:
- **Fare disputes:** "Your fare was $X higher because of high traffic conditions contributing +$Y"
- **Regulatory compliance:** Full prediction audit trail stored in JSONL prediction logs
- **Model debugging:** If SHAP reveals unexpected feature dominance, it signals data quality issues

---

## 7. Data Quality Assessment

### Synthetic Dataset Acknowledgment

This dataset exhibits clear synthetic characteristics:

| Signal | Evidence |
|--------|----------|
| Uniform nulls | Exactly 5.0% missing in every column |
| Formula adherence | metered_fare ≈ Trip_Price for 97% of complete rows |
| Clean distributions | No real-world messiness (no duplicates, no encoding errors) |
| Limited categories | Only 3 weather types, 3 traffic levels, 4 time periods |

**What this means for generalization:**
- Model performance metrics are **optimistic** relative to real-world taxi data
- Real data would have: GPS noise, traffic API latency, weather forecast errors, surge multiplier opacity
- The feature engineering and modeling methodology transfers; the specific hyperparameters don't
- The 1,000-row size limits model complexity — real datasets (NYC TLC: ~1B trips/year) would support deeper architectures

### Residual Analysis

The residual analysis reveals:
- **Residuals are centered near zero** — no systematic bias
- **Heavy right tail** — model underpredicts a few extreme-fare trips ($200+)
- **Heteroscedasticity** — residual variance increases with predicted price (panel a)
- **Non-normal tail** — Q-Q plot deviates at extremes (panel c)

![Residual Analysis](figures/15_residual_analysis.png)

**Recommendation for production:** Log-transform the target before training to stabilize variance and reduce the impact of high-fare outliers. This was not done in this iteration but would likely improve RMSE by 10-15%.

---

## 8. Production Readiness Assessment

| Dimension | Status | Notes |
|-----------|--------|-------|
| Model accuracy | PASS | R²=0.943, MAPE=5.99% (target: <20%) |
| Overfitting control | PASS | Train-val gap reduced to 0.136 after regularization tuning |
| Data leakage | PASS | Imputation fit on train only, verified by tests |
| Explainability | PASS | SHAP integrated, waterfall/force plots available |
| Test coverage | PASS | 83% (62 tests) |
| API serving | PASS | FastAPI with Pydantic validation |
| Monitoring | PASS | CloudWatch metrics + JSONL prediction logs |
| Drift detection | PASS | PSI-based feature drift monitoring |
| Containerization | PASS | Multi-stage Docker, non-root, health checks |
| CI/CD | PASS | GitHub Actions test + build pipelines |
| Documentation | PASS | API contract, env vars, runbook, architecture |

---

## 9. Recommendations for Next Iteration

### High Priority
1. **Log-transform target** — stabilize residual variance for high-fare trips
2. **Collect real-world data** — replace synthetic dataset with actual taxi/ride-hailing records
3. **Add geospatial features** — pickup/dropoff coordinates, route distance vs. straight-line distance
4. **Implement A/B testing** — CatBoost vs. XGBoost canary deployment to validate production performance

### Medium Priority
5. **Feature selection** — remove bottom 5 features (peak_traffic, is_weekend, is_peak_hour, Day_of_Week, Passenger_Count) if they remain low-importance on real data
6. **Prediction intervals** — quantile regression or conformal prediction for uncertainty estimates
7. **Online learning** — incremental model updates as new trip data arrives

### Low Priority
8. **TabPFN experiment** — zero-tuning transformer baseline that benchmarks well on small datasets
9. **Ensemble** — CatBoost + XGBoost weighted average may reduce variance further
10. **Real-time weather API** — replace static Weather column with live weather feeds at prediction time

---

*This analysis was produced as part of an end-to-end ML engineering pipeline using the ML_Sarathi agent orchestration system (Shodak → ML_Bodha → ML_RakShak → ML_Setu).*
