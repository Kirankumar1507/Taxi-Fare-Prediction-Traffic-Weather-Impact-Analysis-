# Taxi Fare Prediction — Final Evaluation Report

## Champion Model: catboost

## Test Set Metrics

| Metric | Baseline (Ridge) | Champion (catboost) | Improvement |
|--------|------------------|------------------------|-------------|
| MAE    | 8.2494 | 4.0168 | 51.3% |
| RMSE   | 16.7119 | 11.4911 | 31.2% |
| MAPE   | 16.2528% | 5.9886% | 63.2% |
| R2     | 0.8805 | 0.9435 | 52.7% of remaining |

## Success Criteria Check
- MAPE < 20%: **PASS** (5.99%)
- R2 > 0.80: **PASS** (0.9435)

## Residual Analysis
- Mean residual: 2.1118
- Std residual: 11.2954
- Max overprediction: -25.76
- Max underprediction: 81.91

## Feature Importance (SHAP)
See `reports/shap_summary.png` and `reports/shap_importance_bar.png` for visualizations.

| Rank | Feature | Mean |SHAP| |
|------|---------|-------------|
| 1 | metered_fare | 15.3852 |
| 2 | distance_cost | 5.6432 |
| 3 | Trip_Distance_km | 3.4053 |
| 4 | duration_cost | 1.8846 |
| 5 | Trip_Duration_Minutes | 1.0558 |
| 6 | Per_Minute_Rate | 0.5084 |
| 7 | traffic_distance | 0.4876 |
| 8 | Per_Km_Rate | 0.3972 |
| 9 | avg_speed_kmh | 0.2889 |
| 10 | Time_of_Day_encoded | 0.1035 |
| 11 | weather_duration | 0.0898 |
| 12 | Base_Fare | 0.0843 |
| 13 | Weather_encoded | 0.0736 |
| 14 | Passenger_Count | 0.0449 |
| 15 | is_weekend | 0.0332 |
| 16 | Traffic_encoded | 0.0165 |
| 17 | is_peak_hour | 0.0144 |
| 18 | peak_traffic | 0.0013 |
| 19 | Day_of_Week_encoded | 0.0000 |

## Plots
- `reports/predicted_vs_actual.png` — Scatter plot of predicted vs actual prices
- `reports/residual_distribution.png` — Histogram of prediction residuals
- `reports/shap_summary.png` — SHAP beeswarm plot
- `reports/shap_importance_bar.png` — SHAP feature importance bar chart

## Known Limitations
1. Dataset is synthetic (metered_fare closely reconstructs Trip_Price for most rows)
2. Small sample size (1000 rows, ~950 after dropping null targets)
3. Uniform null pattern (exactly 50 nulls per column) — MCAR by design
4. Some outliers with Trip_Distance_km > 100 km create high-leverage points
