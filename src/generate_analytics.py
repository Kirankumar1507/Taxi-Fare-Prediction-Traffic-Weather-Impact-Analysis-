"""
Comprehensive EDA & Analytics Visualizations for Taxi Fare Prediction.

Generates 18 publication-quality plots saved to reports/figures/.
Run: python3 -m src.generate_analytics   (from project root)
"""

import matplotlib
matplotlib.use("Agg")

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
import pandas as pd
import joblib
import yaml
from pathlib import Path
from scipy import stats
from catboost import CatBoostRegressor

# ── Project paths ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Style & colour constants ──────────────────────────────────────────────
sns.set_style("whitegrid")
TRAFFIC_COLORS = {"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"}
TRAFFIC_PALETTE = [TRAFFIC_COLORS["Low"], TRAFFIC_COLORS["Medium"], TRAFFIC_COLORS["High"]]
WEATHER_COLORS = {"Clear": "#3498db", "Rain": "#95a5a6", "Snow": "#ecf0f1"}
WEATHER_PALETTE = [WEATHER_COLORS["Clear"], WEATHER_COLORS["Rain"], WEATHER_COLORS["Snow"]]
MODEL_COLORS = {"Ridge": "#9b59b6", "XGBoost": "#e67e22", "CatBoost": "#1abc9c"}

TRAFFIC_ORDER = ["Low", "Medium", "High"]
WEATHER_ORDER = ["Clear", "Rain", "Snow"]
TOD_ORDER = ["Morning", "Afternoon", "Evening", "Night"]

saved_files: list[str] = []


def save(fig, name: str, dpi: int = 200) -> None:
    path = FIG_DIR / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    saved_files.append(str(path))
    print(f"  [saved] {name}")


# ═══════════════════════════════════════════════════════════════════════════
# Load data
# ═══════════════════════════════════════════════════════════════════════════
print("Loading data ...")
raw = pd.read_csv(ROOT / "data" / "raw" / "taxi_trip_pricing.csv")
test = pd.read_csv(ROOT / "data" / "splits" / "test.csv")
train = pd.read_csv(ROOT / "data" / "splits" / "train.csv")

with open(ROOT / "configs" / "config.yaml") as f:
    config = yaml.safe_load(f)

# Derived columns on raw (for Sections 1-3 where we use raw data)
raw["metered_fare"] = (
    raw["Base_Fare"]
    + raw["Trip_Distance_km"] * raw["Per_Km_Rate"]
    + raw["Trip_Duration_Minutes"] * raw["Per_Minute_Rate"]
)
raw["surge_factor"] = raw["Trip_Price"] / raw["metered_fare"]
raw["avg_speed_kmh"] = raw["Trip_Distance_km"] / (raw["Trip_Duration_Minutes"] / 60)
raw["avg_speed_kmh"] = raw["avg_speed_kmh"].replace([np.inf, -np.inf], np.nan)

# Distance buckets for Section 5
raw["distance_bucket"] = pd.cut(
    raw["Trip_Distance_km"],
    bins=[0, 15, 35, 200],
    labels=["Short (<15 km)", "Medium (15-35 km)", "Long (>35 km)"],
)

print(f"  Raw: {raw.shape[0]} rows, {raw.shape[1]} cols")
print(f"  Test split: {test.shape[0]} rows")

# ═══════════════════════════════════════════════════════════════════════════
# Load CatBoost model (used in Section 4 & 5)
# ═══════════════════════════════════════════════════════════════════════════
print("Loading CatBoost model ...")
cb_model = CatBoostRegressor()
cb_model.load_model(str(ROOT / "models" / "catboost_v1.cbm"))
feature_names = joblib.load(ROOT / "models" / "feature_names.joblib")

X_test = test[feature_names].values
y_test = test["Trip_Price"].values
y_pred_cb = cb_model.predict(X_test)

# Also load Ridge & XGB for residual comparison if needed
ridge_model = joblib.load(ROOT / "models" / "ridge_v1.joblib")
xgb_model = joblib.load(ROOT / "models" / "xgboost_v1.joblib")

print("Data & models loaded.\n")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1: Data Overview
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("SECTION 1: Data Overview")
print("=" * 60)

# ── Plot 1: Missing data heatmap ──────────────────────────────────────────
print("Plot 1: Missing data heatmap")
fig, ax = plt.subplots(figsize=(12, 6))
nulls = raw.drop(columns=["metered_fare", "surge_factor", "avg_speed_kmh", "distance_bucket"], errors="ignore")
sns.heatmap(
    nulls.isnull().astype(int),
    cbar_kws={"label": "Missing", "ticks": [0, 1]},
    yticklabels=False,
    cmap="YlOrRd",
    ax=ax,
)
ax.set_title("Missing Data Pattern Across All Columns", fontsize=15, fontweight="bold")
ax.set_xlabel("Feature", fontsize=12)
ax.set_ylabel("Row Index", fontsize=12)
ax.tick_params(axis="x", rotation=45)
save(fig, "01_missing_data_heatmap.png")

# ── Plot 2: Target distribution ──────────────────────────────────────────
print("Plot 2: Target distribution")
price = raw["Trip_Price"].dropna()
fig, ax1 = plt.subplots(figsize=(11, 6))
bins = np.linspace(price.min(), price.max(), 50)
ax1.hist(price, bins=bins, color="#3498db", alpha=0.7, edgecolor="white", label="Count")
ax1.set_xlabel("Trip Price ($)", fontsize=12)
ax1.set_ylabel("Frequency", fontsize=12, color="#3498db")
ax1.tick_params(axis="y", labelcolor="#3498db")

ax2 = ax1.twinx()
price.plot.kde(ax=ax2, color="#e74c3c", linewidth=2, label="KDE")
ax2.set_ylabel("Density", fontsize=12, color="#e74c3c")
ax2.tick_params(axis="y", labelcolor="#e74c3c")

mean_val = price.mean()
median_val = price.median()
skew_val = price.skew()
ax1.axvline(mean_val, color="#2c3e50", linestyle="--", linewidth=1.5, label=f"Mean={mean_val:.1f}")
ax1.axvline(median_val, color="#8e44ad", linestyle="-.", linewidth=1.5, label=f"Median={median_val:.1f}")
textstr = f"Mean: ${mean_val:.2f}\nMedian: ${median_val:.2f}\nSkew: {skew_val:.2f}"
ax1.text(
    0.97, 0.95, textstr, transform=ax1.transAxes, fontsize=11,
    verticalalignment="top", horizontalalignment="right",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.8),
)
ax1.set_title("Trip Price Distribution", fontsize=15, fontweight="bold")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=10)
save(fig, "02_target_distribution.png")

# ── Plot 3: Correlation matrix ──────────────────────────────────────────
print("Plot 3: Correlation matrix")
numeric_cols_raw = [
    "Trip_Distance_km", "Passenger_Count", "Base_Fare",
    "Per_Km_Rate", "Per_Minute_Rate", "Trip_Duration_Minutes", "Trip_Price",
]
corr = raw[numeric_cols_raw].dropna().corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
    center=0, vmin=-1, vmax=1, square=True,
    linewidths=0.5, ax=ax,
    annot_kws={"size": 10},
)
ax.set_title("Correlation Matrix — Numeric Features", fontsize=15, fontweight="bold")
save(fig, "03_correlation_matrix.png")

# ── Plot 4: Pair plot ────────────────────────────────────────────────────
print("Plot 4: Pair plot (this may take a moment) ...")
pair_df = raw[["Trip_Distance_km", "Trip_Duration_Minutes", "Trip_Price", "Traffic_Conditions"]].dropna()
g = sns.pairplot(
    pair_df, hue="Traffic_Conditions", hue_order=TRAFFIC_ORDER,
    palette=TRAFFIC_COLORS, diag_kind="kde",
    plot_kws={"alpha": 0.5, "s": 20},
)
g.figure.suptitle("Pair Plot — Distance, Duration & Price by Traffic", fontsize=15, fontweight="bold", y=1.02)
save(g.figure, "04_pair_plot.png")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2: Traffic & Weather Impact Analysis
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 2: Traffic & Weather Impact Analysis")
print("=" * 60)

# ── Plot 5: Price by Traffic — violin ────────────────────────────────────
print("Plot 5: Price by Traffic Conditions")
fig, ax = plt.subplots(figsize=(10, 7))
plot_df = raw.dropna(subset=["Trip_Price", "Traffic_Conditions"])
sns.violinplot(
    data=plot_df, x="Traffic_Conditions", y="Trip_Price",
    order=TRAFFIC_ORDER, palette=TRAFFIC_COLORS,
    inner=None, alpha=0.6, ax=ax,
)
sns.stripplot(
    data=plot_df, x="Traffic_Conditions", y="Trip_Price",
    order=TRAFFIC_ORDER, palette=TRAFFIC_COLORS,
    size=2.5, alpha=0.35, jitter=0.25, ax=ax,
)
for i, level in enumerate(TRAFFIC_ORDER):
    med = plot_df.loc[plot_df["Traffic_Conditions"] == level, "Trip_Price"].median()
    ax.text(i, med + 5, f"med={med:.1f}", ha="center", fontsize=10, fontweight="bold", color="#2c3e50")
ax.set_title("Trip Price Distribution by Traffic Conditions", fontsize=15, fontweight="bold")
ax.set_xlabel("Traffic Conditions", fontsize=12)
ax.set_ylabel("Trip Price ($)", fontsize=12)
save(fig, "05_price_by_traffic_violin.png")

# ── Plot 6: Price by Weather — violin ───────────────────────────────────
print("Plot 6: Price by Weather")
fig, ax = plt.subplots(figsize=(10, 7))
plot_df = raw.dropna(subset=["Trip_Price", "Weather"])
sns.violinplot(
    data=plot_df, x="Weather", y="Trip_Price",
    order=WEATHER_ORDER, palette=WEATHER_COLORS,
    inner=None, alpha=0.6, ax=ax,
)
sns.stripplot(
    data=plot_df, x="Weather", y="Trip_Price",
    order=WEATHER_ORDER, palette=WEATHER_COLORS,
    size=2.5, alpha=0.35, jitter=0.25, ax=ax,
)
for i, level in enumerate(WEATHER_ORDER):
    med = plot_df.loc[plot_df["Weather"] == level, "Trip_Price"].median()
    ax.text(i, med + 5, f"med={med:.1f}", ha="center", fontsize=10, fontweight="bold", color="#2c3e50")
ax.set_title("Trip Price Distribution by Weather", fontsize=15, fontweight="bold")
ax.set_xlabel("Weather", fontsize=12)
ax.set_ylabel("Trip Price ($)", fontsize=12)
save(fig, "06_price_by_weather_violin.png")

# ── Plot 7: Traffic x Weather heatmap ───────────────────────────────────
print("Plot 7: Traffic x Weather mean price heatmap")
fig, ax = plt.subplots(figsize=(9, 6))
pivot = (
    raw.dropna(subset=["Trip_Price", "Traffic_Conditions", "Weather"])
    .pivot_table(values="Trip_Price", index="Traffic_Conditions", columns="Weather", aggfunc="mean")
    .reindex(index=TRAFFIC_ORDER, columns=WEATHER_ORDER)
)
sns.heatmap(
    pivot, annot=True, fmt=".1f", cmap="YlOrRd",
    linewidths=1, ax=ax, annot_kws={"size": 13, "fontweight": "bold"},
)
ax.set_title("Mean Trip Price — Traffic x Weather", fontsize=15, fontweight="bold")
ax.set_xlabel("Weather", fontsize=12)
ax.set_ylabel("Traffic Conditions", fontsize=12)
save(fig, "07_traffic_weather_heatmap.png")

# ── Plot 8: Time of Day pricing pattern ─────────────────────────────────
print("Plot 8: Time of Day pricing by Traffic")
fig, ax = plt.subplots(figsize=(11, 7))
plot_df = raw.dropna(subset=["Trip_Price", "Time_of_Day", "Traffic_Conditions"])
grouped = (
    plot_df.groupby(["Time_of_Day", "Traffic_Conditions"])["Trip_Price"]
    .agg(["mean", "std"])
    .reset_index()
)
# Build grouped bar positions manually for full control
n_times = len(TOD_ORDER)
n_traffic = len(TRAFFIC_ORDER)
bar_width = 0.25
x = np.arange(n_times)

for j, traffic in enumerate(TRAFFIC_ORDER):
    subset = grouped[grouped["Traffic_Conditions"] == traffic]
    # Reorder to match TOD_ORDER
    subset = subset.set_index("Time_of_Day").reindex(TOD_ORDER)
    ax.bar(
        x + j * bar_width, subset["mean"], bar_width,
        yerr=subset["std"], capsize=3,
        color=TRAFFIC_COLORS[traffic], label=traffic, edgecolor="white",
    )

ax.set_xticks(x + bar_width)
ax.set_xticklabels(TOD_ORDER, fontsize=11)
ax.set_xlabel("Time of Day", fontsize=12)
ax.set_ylabel("Mean Trip Price ($)", fontsize=12)
ax.set_title("Mean Trip Price by Time of Day & Traffic", fontsize=15, fontweight="bold")
ax.legend(title="Traffic", fontsize=10)
save(fig, "08_time_of_day_pricing.png")

# ── Plot 9: Weekend vs Weekday pricing faceted ─────────────────────────
print("Plot 9: Weekend vs Weekday pricing by Weather")
plot_df = raw.dropna(subset=["Trip_Price", "Day_of_Week", "Weather"])
g = sns.catplot(
    data=plot_df, x="Day_of_Week", y="Trip_Price", col="Weather",
    col_order=WEATHER_ORDER, kind="box",
    palette={"Weekday": "#3498db", "Weekend": "#e67e22"},
    height=5, aspect=0.9,
)
g.figure.suptitle("Trip Price: Weekend vs Weekday (by Weather)", fontsize=15, fontweight="bold", y=1.03)
for ax in g.axes.flat:
    ax.set_xlabel("Day of Week", fontsize=11)
    ax.set_ylabel("Trip Price ($)", fontsize=11)
save(g.figure, "09_weekday_weekend_faceted.png")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3: Feature Engineering Insights
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 3: Feature Engineering Insights")
print("=" * 60)

# ── Plot 10: Metered fare vs actual price ────────────────────────────────
print("Plot 10: Metered fare vs actual price")
fig, ax = plt.subplots(figsize=(10, 8))
plot_df = raw.dropna(subset=["metered_fare", "Trip_Price", "Traffic_Conditions"])
for traffic in TRAFFIC_ORDER:
    sub = plot_df[plot_df["Traffic_Conditions"] == traffic]
    ax.scatter(
        sub["metered_fare"], sub["Trip_Price"],
        c=TRAFFIC_COLORS[traffic], label=traffic, s=25, alpha=0.6, edgecolors="none",
    )
lims = [0, max(plot_df["metered_fare"].max(), plot_df["Trip_Price"].max()) * 1.05]
ax.plot(lims, lims, "--", color="#2c3e50", linewidth=1.5, label="y = x (perfect metered)")
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel("Metered Fare ($)", fontsize=12)
ax.set_ylabel("Actual Trip Price ($)", fontsize=12)
ax.set_title("Metered Fare vs Actual Price", fontsize=15, fontweight="bold")
ax.legend(fontsize=10)
save(fig, "10_metered_vs_actual.png")

# ── Plot 11: Surge factor analysis ──────────────────────────────────────
print("Plot 11: Surge factor distribution")
fig, ax = plt.subplots(figsize=(10, 6))
plot_df = raw.dropna(subset=["surge_factor", "Weather"])
# Remove extreme outliers for readability
plot_df = plot_df[(plot_df["surge_factor"] > 0) & (plot_df["surge_factor"] < plot_df["surge_factor"].quantile(0.99))]
for weather in WEATHER_ORDER:
    sub = plot_df[plot_df["Weather"] == weather]
    ax.hist(
        sub["surge_factor"], bins=40, alpha=0.55,
        color=WEATHER_COLORS[weather], label=weather, edgecolor="white",
    )
ax.axvline(1.0, color="#2c3e50", linestyle="--", linewidth=1.5, label="Surge = 1.0 (no surge)")
ax.set_xlabel("Surge Factor (Actual / Metered)", fontsize=12)
ax.set_ylabel("Frequency", fontsize=12)
ax.set_title("Surge Factor Distribution by Weather", fontsize=15, fontweight="bold")
ax.legend(fontsize=10)
save(fig, "11_surge_factor_distribution.png")

# ── Plot 12: Speed vs Price ─────────────────────────────────────────────
print("Plot 12: Speed vs Price")
fig, ax = plt.subplots(figsize=(10, 7))
plot_df = raw.dropna(subset=["avg_speed_kmh", "Trip_Price", "Traffic_Conditions"])
for traffic in TRAFFIC_ORDER:
    sub = plot_df[plot_df["Traffic_Conditions"] == traffic]
    ax.scatter(
        sub["avg_speed_kmh"], sub["Trip_Price"],
        c=TRAFFIC_COLORS[traffic], label=traffic, s=25, alpha=0.6, edgecolors="none",
    )
ax.set_xlabel("Average Speed (km/h)", fontsize=12)
ax.set_ylabel("Trip Price ($)", fontsize=12)
ax.set_title("Average Speed vs Trip Price (by Traffic)", fontsize=15, fontweight="bold")
ax.legend(fontsize=10)
save(fig, "12_speed_vs_price.png")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4: Model Performance
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 4: Model Performance")
print("=" * 60)

# ── Plot 13: Model comparison bar chart ──────────────────────────────────
print("Plot 13: Model comparison bar chart")
metrics_data = {
    "Model":  ["Ridge", "XGBoost", "CatBoost"],
    "MAE":    [8.25, 4.13, 4.02],
    "RMSE":   [16.71, 12.10, 11.49],
    "MAPE":   [16.25, 6.11, 5.99],
    "R2":     [0.881, 0.937, 0.943],
}
metrics_df = pd.DataFrame(metrics_data)
metric_names = ["MAE", "RMSE", "MAPE", "R2"]

fig, axes = plt.subplots(1, 4, figsize=(18, 5))
for k, metric in enumerate(metric_names):
    ax = axes[k]
    for i, model in enumerate(["Ridge", "XGBoost", "CatBoost"]):
        val = metrics_df.loc[metrics_df["Model"] == model, metric].values[0]
        bar = ax.bar(i, val, color=MODEL_COLORS[model], edgecolor="white", width=0.6)
        ax.text(i, val + (val * 0.02 if val > 0 else 0.005), f"{val}", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Ridge", "XGBoost", "CatBoost"], fontsize=10)
    ax.set_title(metric, fontsize=13, fontweight="bold")
    ax.set_ylabel(metric, fontsize=11)
    # For R2, highlight the 'higher is better' with a different y range
    if metric == "R2":
        ax.set_ylim(0.85, 0.96)
fig.suptitle("Model Comparison — Test Set Metrics", fontsize=16, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.93])
save(fig, "13_model_comparison.png")

# ── Plot 14: Predicted vs Actual with confidence bands ──────────────────
print("Plot 14: Predicted vs Actual (CatBoost)")
fig, ax = plt.subplots(figsize=(10, 8))
residuals = y_test - y_pred_cb
abs_resid = np.abs(residuals)

scatter = ax.scatter(
    y_test, y_pred_cb, c=abs_resid, cmap="RdYlGn_r",
    s=30, alpha=0.7, edgecolors="none",
)
cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
cbar.set_label("|Residual|", fontsize=11)

lim_min = min(y_test.min(), y_pred_cb.min()) * 0.9
lim_max = max(y_test.max(), y_pred_cb.max()) * 1.05
ax.plot([lim_min, lim_max], [lim_min, lim_max], "--", color="#2c3e50", linewidth=1.5, label="Perfect prediction")

# +/-10% bands
x_line = np.linspace(lim_min, lim_max, 100)
ax.fill_between(x_line, x_line * 0.9, x_line * 1.1, alpha=0.12, color="#3498db", label="+/- 10% band")

ax.set_xlim(lim_min, lim_max)
ax.set_ylim(lim_min, lim_max)
ax.set_xlabel("Actual Trip Price ($)", fontsize=12)
ax.set_ylabel("Predicted Trip Price ($)", fontsize=12)
ax.set_title("CatBoost: Predicted vs Actual", fontsize=15, fontweight="bold")
ax.legend(fontsize=10)
save(fig, "14_predicted_vs_actual.png")

# ── Plot 15: Residual analysis 2x2 ─────────────────────────────────────
print("Plot 15: Residual analysis (2x2)")
fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# (a) Residuals vs predicted
ax = axes[0, 0]
ax.scatter(y_pred_cb, residuals, s=15, alpha=0.5, color="#3498db", edgecolors="none")
ax.axhline(0, color="#e74c3c", linestyle="--", linewidth=1.2)
ax.set_xlabel("Predicted Price ($)", fontsize=11)
ax.set_ylabel("Residual ($)", fontsize=11)
ax.set_title("(a) Residuals vs Predicted", fontsize=13, fontweight="bold")

# (b) Residual histogram
ax = axes[0, 1]
ax.hist(residuals, bins=40, color="#2ecc71", edgecolor="white", alpha=0.8)
ax.axvline(0, color="#e74c3c", linestyle="--", linewidth=1.2)
ax.set_xlabel("Residual ($)", fontsize=11)
ax.set_ylabel("Frequency", fontsize=11)
ax.set_title("(b) Residual Distribution", fontsize=13, fontweight="bold")

# (c) Q-Q plot
ax = axes[1, 0]
stats.probplot(residuals, dist="norm", plot=ax)
ax.set_title("(c) Q-Q Plot", fontsize=13, fontweight="bold")
ax.get_lines()[0].set_markerfacecolor("#3498db")
ax.get_lines()[0].set_markersize(4)
ax.get_lines()[1].set_color("#e74c3c")

# (d) Residuals vs Trip_Distance
ax = axes[1, 1]
ax.scatter(test["Trip_Distance_km"], residuals, s=15, alpha=0.5, color="#9b59b6", edgecolors="none")
ax.axhline(0, color="#e74c3c", linestyle="--", linewidth=1.2)
ax.set_xlabel("Trip Distance (km)", fontsize=11)
ax.set_ylabel("Residual ($)", fontsize=11)
ax.set_title("(d) Residuals vs Trip Distance", fontsize=13, fontweight="bold")

fig.suptitle("CatBoost — Residual Analysis", fontsize=16, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
save(fig, "15_residual_analysis.png")

# ── Plot 16: SHAP waterfall ────────────────────────────────────────────
print("Plot 16: SHAP waterfall for median-priced trip")
import shap

# Find the median-priced trip in test set
median_price = np.median(y_test)
median_idx = np.argmin(np.abs(y_test - median_price))

explainer = shap.TreeExplainer(cb_model)
shap_values = explainer(pd.DataFrame(X_test, columns=feature_names))

fig, ax = plt.subplots(figsize=(10, 8))
shap.plots.waterfall(shap_values[median_idx], max_display=15, show=False)
fig_current = plt.gcf()
fig_current.suptitle(
    f"SHAP Waterfall — Median-Priced Trip (${y_test[median_idx]:.2f})",
    fontsize=14, fontweight="bold", y=1.02,
)
save(fig_current, "16_shap_waterfall.png")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5: Business Insights
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 5: Business Insights")
print("=" * 60)

# ── Plot 17: Price sensitivity heatmap ──────────────────────────────────
print("Plot 17: Price sensitivity heatmap (simulated)")

# Build a grid of simulated trips; hold everything else at train medians
train_medians = train[feature_names].median()

distances = [5, 10, 20, 30, 50]
traffic_levels = TRAFFIC_ORDER
traffic_enc_map = config["encoding"]["Traffic_Conditions"]

grid_results = np.zeros((len(traffic_levels), len(distances)))
for i, traffic in enumerate(traffic_levels):
    for j, dist in enumerate(distances):
        row = train_medians.copy()
        row["Trip_Distance_km"] = dist
        row["Traffic_encoded"] = traffic_enc_map[traffic]
        # Recompute engineered features that depend on distance/traffic
        row["distance_cost"] = dist * row["Per_Km_Rate"]
        row["metered_fare"] = row["Base_Fare"] + row["distance_cost"] + row["duration_cost"]
        row["avg_speed_kmh"] = dist / (row["Trip_Duration_Minutes"] / 60) if row["Trip_Duration_Minutes"] > 0 else 0
        row["traffic_distance"] = traffic_enc_map[traffic] * dist
        row["peak_traffic"] = row["is_peak_hour"] * traffic_enc_map[traffic]
        grid_results[i, j] = cb_model.predict(row.values.reshape(1, -1))[0]

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(
    pd.DataFrame(grid_results, index=traffic_levels, columns=[f"{d} km" for d in distances]),
    annot=True, fmt=".1f", cmap="YlOrRd", linewidths=1,
    ax=ax, annot_kws={"size": 13, "fontweight": "bold"},
)
ax.set_title("Predicted Fare by Distance & Traffic Level", fontsize=15, fontweight="bold")
ax.set_xlabel("Trip Distance", fontsize=12)
ax.set_ylabel("Traffic Conditions", fontsize=12)
save(fig, "17_price_sensitivity_heatmap.png")

# ── Plot 18: Weather premium analysis ───────────────────────────────────
print("Plot 18: Weather premium by distance bucket")
plot_df = raw.dropna(subset=["Trip_Price", "Weather", "distance_bucket"])

# Compute mean price per (distance_bucket, Weather)
pivot_wp = plot_df.pivot_table(values="Trip_Price", index="distance_bucket", columns="Weather", aggfunc="mean")
pivot_wp = pivot_wp.reindex(columns=WEATHER_ORDER)

# Compute premiums relative to Clear
premium_rain = pivot_wp["Rain"] - pivot_wp["Clear"]
premium_snow = pivot_wp["Snow"] - pivot_wp["Clear"]

buckets = pivot_wp.index.tolist()
x = np.arange(len(buckets))
bar_w = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - bar_w / 2, premium_rain, bar_w, color=WEATHER_COLORS["Rain"], label="Rain vs Clear", edgecolor="white")
ax.bar(x + bar_w / 2, premium_snow, bar_w, color="#7f8c8d", label="Snow vs Clear", edgecolor="white")

for k in range(len(buckets)):
    ax.text(x[k] - bar_w / 2, premium_rain.iloc[k] + 0.3, f"${premium_rain.iloc[k]:.1f}", ha="center", fontsize=10, fontweight="bold")
    ax.text(x[k] + bar_w / 2, premium_snow.iloc[k] + 0.3, f"${premium_snow.iloc[k]:.1f}", ha="center", fontsize=10, fontweight="bold")

ax.axhline(0, color="#2c3e50", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(buckets, fontsize=11)
ax.set_xlabel("Trip Distance Bucket", fontsize=12)
ax.set_ylabel("Average Price Premium ($)", fontsize=12)
ax.set_title("Weather Premium by Trip Distance", fontsize=15, fontweight="bold")
ax.legend(fontsize=11)
save(fig, "18_weather_premium.png")


# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"DONE — {len(saved_files)} plots saved to {FIG_DIR}/")
print("=" * 60)
for f in saved_files:
    print(f"  {f}")
