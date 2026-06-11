"""
MODULE 02 — MODEL TRAINING
Honors Project: ML-Based Irrigation Recommendation System

PURPOSE:
    Trains the Random Forest soil moisture prediction model
    on 9 years of NASA POWER data and saves it to disk.

    Run this ONCE before using 03_predict_today.py.
    No user input required — just run and wait.

USAGE:
    python 02_ml_pipeline.py
"""

import os, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats

np.random.seed(42)

# ── Paths ─────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data")
FIG_DIR   = os.path.join(BASE_DIR, "figures")
MODEL_DIR = os.path.join(BASE_DIR, "models")
for d in [DATA_DIR, FIG_DIR, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Soil constants ────────────────────────────────────────
FC  = 0.35
PWP = 0.14

# ── NSE function ──────────────────────────────────────────
def nse(obs, sim):
    return 1 - np.sum((obs-sim)**2) / np.sum((obs-np.mean(obs))**2)

def nse_rating(n):
    if n > 0.75: return "Very Good"
    if n > 0.65: return "Good"
    if n > 0.50: return "Satisfactory"
    return "Unsatisfactory"

# ══════════════════════════════════════════════════════════
print("=" * 60)
print("  MODEL TRAINING — Irrigation ML Project")
print("  NASA POWER data | Yanco, NSW | 2016-2025")
print("=" * 60)

# ── Load data ─────────────────────────────────────────────
data_path = os.path.join(DATA_DIR, "ozflux_synthetic_full.csv")
if not os.path.exists(data_path):
    print("\n  ERROR: Data file not found.")
    print("  Please run 00_read_nasapower.py first.")
    exit(1)

print("\n  Loading data...", end=" ", flush=True)
df = pd.read_csv(data_path, parse_dates=["date"])
print(f"done  ({len(df)} records, {df['year'].nunique()} years)")

# ── Features ──────────────────────────────────────────────
FEATURES = [
    "soil_moisture",
    "sm_lag1", "sm_lag3", "sm_lag7",
    "soil_temperature", "air_temperature", "air_temp_max", "air_temp_min",
    "humidity", "rainfall",
    "ET0_mm", "VPD_kPa", "wind_speed_2m",
    "rain_3d", "rain_7d", "ET0_7d", "T_7d",
    "Kc", "season_day", "doy",
]

df["sm_next"] = df["soil_moisture"].shift(-1)
df = df.dropna(subset=["sm_next"]).reset_index(drop=True)

X = df[FEATURES].values
y = df["sm_next"].values

# ── Temporal train/test split (last year = test) ──────────
test_mask  = df["year"] == df["year"].max()
train_mask = ~test_mask
X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print(f"  Train : {X_train.shape[0]} samples "
      f"({df[train_mask]['year'].min()}–{df[train_mask]['year'].max()})")
print(f"  Test  : {X_test.shape[0]} samples "
      f"({df[test_mask]['year'].max()})")

# ── Train models ──────────────────────────────────────────
print("\n  Training models...")

MODELS = {
    "Random Forest": RandomForestRegressor(
        n_estimators=500, max_depth=15, min_samples_leaf=2,
        max_features=0.6, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=400, learning_rate=0.04, max_depth=6,
        subsample=0.75, min_samples_leaf=3, random_state=42),
    "Ridge Regression": Ridge(alpha=0.5),
}

results = {}
for name, model in MODELS.items():
    print(f"  Training {name}...", end=" ", flush=True)
    t0  = time.time()
    Xtr = X_train_s if name == "Ridge Regression" else X_train
    Xte = X_test_s  if name == "Ridge Regression" else X_test
    model.fit(Xtr, y_train)
    yp  = np.clip(model.predict(Xte), PWP, FC + 0.02)
    t1  = time.time()
    results[name] = dict(
        model=model, y_pred=yp,
        RMSE=np.sqrt(mean_squared_error(y_test, yp)),
        MAE =mean_absolute_error(y_test, yp),
        R2  =r2_score(y_test, yp),
        NSE =nse(y_test, yp))
    print(f"done ({t1-t0:.1f}s)  "
          f"R²={results[name]['R2']:.4f}  "
          f"NSE={results[name]['NSE']:.4f}  "
          f"— {nse_rating(results[name]['NSE'])}")

# Cross validation
print(f"\n  Running 5-fold cross-validation...", end=" ", flush=True)
cv    = TimeSeriesSplit(n_splits=5)
cv_sc = cross_val_score(
    RandomForestRegressor(n_estimators=300, max_depth=15,
                          min_samples_leaf=2, max_features=0.6,
                          random_state=42, n_jobs=-1),
    X_train, y_train, cv=cv,
    scoring="neg_root_mean_squared_error")
print(f"done  RMSE={-cv_sc.mean():.5f} ± {cv_sc.std():.5f}")

# ── Select best model ─────────────────────────────────────
best_name   = min(results, key=lambda k: results[k]["RMSE"])
best_result = results[best_name]
y_pred_best = best_result["y_pred"]
best_model  = best_result["model"]

print(f"\n  Best model : {best_name}")
print(f"  R²         : {best_result['R2']:.4f}")
print(f"  NSE        : {best_result['NSE']:.4f}  ({nse_rating(best_result['NSE'])})")
print(f"  RMSE       : {best_result['RMSE']:.5f} m³/m³")

# ── Save model ────────────────────────────────────────────
print("\n  Saving model...", end=" ", flush=True)
joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.pkl"))
joblib.dump(scaler,     os.path.join(MODEL_DIR, "feature_scaler.pkl"))
feat_imp = pd.Series(
    results["Random Forest"]["model"].feature_importances_,
    index=FEATURES).sort_values(ascending=False)
feat_imp.to_csv(os.path.join(DATA_DIR, "feature_importance.csv"),
                header=["importance"])
pd.DataFrame({n: {k: v for k,v in r.items()
                  if k not in ("model","y_pred")}
              for n,r in results.items()}).T.to_csv(
    os.path.join(DATA_DIR, "model_summary.csv"))
print("done")

# ── Irrigation recommendation engine ─────────────────────
print("\n  Computing irrigation recommendations...")

FC    = 0.35
PWP   = 0.14
Zr_mm = 600.0

test_mask   = df["year"] == df["year"].max()
test_df     = df[test_mask].copy().reset_index(drop=True)
test_df["sm_predicted"]       = y_pred_best
test_df["smd_pred_m3m3"]      = np.maximum(FC - test_df["sm_predicted"], 0)
test_df["smd_pred_mm"]        = test_df["smd_pred_m3m3"] * Zr_mm

# Dynamic trigger: 12th percentile of training SM
irr_trigger = float(np.percentile(y_train, 12))
irr_trigger = np.clip(irr_trigger, PWP + 0.03, FC - 0.05)

test_df["irr_recommended_mm"] = np.where(
    test_df["sm_predicted"] < irr_trigger,
    test_df["smd_pred_mm"], 0.0)

irr_rec = test_df[test_df["irr_recommended_mm"] > 0]
irr_act = test_df[test_df["irrigation_mm"] > 0]

test_df.to_csv(os.path.join(DATA_DIR, "test_predictions.csv"), index=False)

print(f"  Irrigation trigger : SM < {irr_trigger:.3f} m3/m3")
print(f"  Recommended events : {len(irr_rec)}")
print(f"  Actual events      : {len(irr_act)}")
if len(irr_rec):
    print(f"  Mean recommended   : {irr_rec['irr_recommended_mm'].mean():.1f} mm")
print(f"  Season total rec.  : {test_df['irr_recommended_mm'].sum():.0f} mm")
print(f"  Season total act.  : {test_df['irrigation_mm'].sum():.0f} mm")

# ── Generate figures ──────────────────────────────────────
print("\n  Generating figures...")
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10})
COLORS = {"Random Forest":"#1565C0",
          "Gradient Boosting":"#E65100",
          "Ridge Regression":"#6A1B9A"}
DPI = 150

# Fig 1 — Obs vs Predicted
print("  Saving fig1_obs_vs_pred.png...", end=" ", flush=True)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (name, res) in zip(axes, results.items()):
    yp  = res["y_pred"]
    lim = [min(y_test.min(),yp.min())-0.003,
           max(y_test.max(),yp.max())+0.003]
    ax.scatter(y_test, yp, alpha=0.45, s=16, color=COLORS[name])
    ax.plot(lim, lim, "k--", lw=1.2, label="1:1 line")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Observed SM(t+1) (m³/m³)")
    ax.set_ylabel("Predicted SM(t+1) (m³/m³)")
    ax.set_title(f"{name}\n"
                 f"R²={res['R2']:.4f}  "
                 f"NSE={res['NSE']:.4f}  "
                 f"RMSE={res['RMSE']:.5f}")
    ax.legend()
fig.suptitle("Figure 1. Observed vs Predicted Soil Moisture",
             fontweight="bold", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig1_obs_vs_pred.png"),
            dpi=DPI, bbox_inches="tight")
plt.close()
print("done")

# Fig 2 — Feature importance
# Fig 2 — Time series
print("  Saving fig2_timeseries.png...", end=" ", flush=True)
fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
dates = pd.to_datetime(test_df["date"])

ax = axes[0]
ax.plot(dates, test_df["soil_moisture"], color="#37474F", lw=1.6, label="Observed SM")
ax.plot(dates, test_df["sm_predicted"],  color="#D32F2F", lw=1.6, ls="--",
        label=f"{best_name} Predicted SM")
ax.axhline(FC,          color="#0D47A1", lw=1.1, ls=":",  label=f"FC={FC}")
ax.axhline(irr_trigger, color="#E65100", lw=1.1, ls="-.", label=f"Trigger={irr_trigger:.3f}")
ax.axhline(PWP,         color="#B71C1C", lw=1.1, ls=":",  label=f"PWP={PWP}")
ax.fill_between(dates, PWP, test_df["soil_moisture"], alpha=0.10, color="#607D8B")
ax.set_ylabel("Soil Moisture (m3/m3)")
ax.set_title("(a) Soil Moisture — Observed vs Predicted", fontweight="bold")
ax.legend(ncol=3, loc="lower left", fontsize=8)

ax = axes[1]
ax.bar(dates, test_df["smd_pred_mm"], color="#FF6F00", alpha=0.8, label="Predicted SMD")
ax.bar(dates, test_df["smd_mm"],      color="#B0BEC5", alpha=0.5, label="Observed SMD", width=0.6)
ax.axhline((FC-irr_trigger)*1000*0.6, color="red", ls="--", lw=1.3, label="Trigger")
ax.set_ylabel("Soil Moisture Deficit (mm)")
ax.set_title("(b) Soil Moisture Deficit", fontweight="bold")
ax.legend(fontsize=8)

ax = axes[2]
ax.bar(dates, test_df["irr_recommended_mm"], color="#1565C0", alpha=0.85,
       label=f"Recommended ({test_df['irr_recommended_mm'].sum():.0f} mm)")
ax.bar(dates, test_df["irrigation_mm"],      color="#2E7D32", alpha=0.55, width=0.6,
       label=f"Actual ({test_df['irrigation_mm'].sum():.0f} mm)")
ax.bar(dates, test_df["rainfall"],           color="#0097A7", alpha=0.4,  width=0.4, label="Rainfall")
ax.set_xlabel("Date")
ax.set_ylabel("Water Depth (mm)")
ax.set_title("(c) Irrigation Recommendation vs Actual", fontweight="bold")
ax.legend(fontsize=8)

fig.suptitle("Figure 2. ML Irrigation Decision Support — Full Season Analysis",
             fontweight="bold", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig2_timeseries.png"), dpi=DPI, bbox_inches="tight")
plt.close()
print("done")

print("  Saving fig3_feature_importance.png...", end=" ", flush=True)
top = feat_imp.head(15)
fig, ax = plt.subplots(figsize=(9, 6.5))
bars = ax.barh(range(len(top)), top.values, color="#1565C0", alpha=0.85)
ax.set_yticks(range(len(top)))
ax.set_yticklabels(top.index)
ax.invert_yaxis()
ax.set_xlabel("Mean Decrease in Impurity (MDI)")
ax.set_title("Figure 3. Feature Importances — Random Forest",
             fontweight="bold")
for b, v in zip(bars, top.values):
    ax.text(v+0.0005, b.get_y()+b.get_height()/2,
            f"{v:.4f}", va="center", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig3_feature_importance.png"),
            dpi=DPI, bbox_inches="tight")
plt.close()
print("done")

# Fig 3 — Model comparison
print("  Saving fig4_model_comparison.png...", end=" ", flush=True)
metrics_list = [("RMSE","RMSE"),("MAE","MAE"),("R²","R2"),("NSE","NSE")]
fig, axes = plt.subplots(1, 4, figsize=(15, 5))
for ax, (label, key) in zip(axes, metrics_list):
    vals = [results[n][key] for n in results]
    cols = [COLORS[n] for n in results]
    bars = ax.bar(list(results.keys()), vals,
                  color=cols, alpha=0.85, edgecolor="white")
    ax.set_title(label, fontweight="bold")
    ax.set_xticklabels(list(results.keys()), rotation=22, ha="right")
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2,
                b.get_height()+abs(b.get_height())*0.01,
                f"{v:.4f}", ha="center", fontsize=8.5)
    if key in ("R2","NSE"):
        ax.axhline(0.75, color="green", ls="--",
                   lw=1.2, label="Very Good (0.75)")
        ax.legend(fontsize=8)
fig.suptitle("Figure 4. Model Performance Comparison",
             fontweight="bold", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig4_model_comparison.png"),
            dpi=DPI, bbox_inches="tight")
plt.close()
print("done")

# Fig 4 — Residuals
print("  Saving fig5_residuals.png...", end=" ", flush=True)
residuals = y_test - y_pred_best
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
axes[0].scatter(y_pred_best, residuals, alpha=0.4, s=15, color="#1565C0")
axes[0].axhline(0, color="red", lw=1.2)
axes[0].set_xlabel("Predicted SM")
axes[0].set_ylabel("Residual")
axes[0].set_title("(a) Residuals vs Fitted", fontweight="bold")
axes[1].hist(residuals, bins=30, color="#43A047",
             edgecolor="white", alpha=0.85)
axes[1].axvline(0, color="red", lw=1.2)
axes[1].set_xlabel("Residual")
axes[1].set_ylabel("Count")
axes[1].set_title("(b) Residual Distribution", fontweight="bold")
stats.probplot(residuals, dist="norm", plot=axes[2])
axes[2].set_title("(c) Normal Q-Q Plot", fontweight="bold")
fig.suptitle(f"Figure 5. Residual Diagnostics — {best_name}",
             fontweight="bold", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig5_residuals.png"),
            dpi=DPI, bbox_inches="tight")
plt.close()
print("done")

# Fig 5 — Correlation heatmap
print("  Saving fig6_correlation.png...", end=" ", flush=True)
corr_cols = ["soil_moisture","soil_temperature","air_temperature",
             "humidity","rainfall","ET0_mm","VPD_kPa",
             "irrigation_mm","smd_mm"]
corr_df = df[corr_cols].rename(columns={
    "soil_moisture":"SM","soil_temperature":"T_soil",
    "air_temperature":"T_air","humidity":"RH",
    "rainfall":"Rain","ET0_mm":"ET0","VPD_kPa":"VPD",
    "irrigation_mm":"Irrigation","smd_mm":"SMD"})
fig, ax = plt.subplots(figsize=(8, 7))
mask = np.triu(np.ones_like(corr_df.corr(), dtype=bool))
sns.heatmap(corr_df.corr(), annot=True, fmt=".2f",
            cmap="RdBu_r", center=0, mask=mask,
            ax=ax, linewidths=0.5)
ax.set_title("Figure 6. Pearson Correlation Matrix",
             fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig6_correlation.png"),
            dpi=DPI, bbox_inches="tight")
plt.close()
print("done")

# ── Done ──────────────────────────────────────────────────
print()
print("=" * 60)
print("  TRAINING COMPLETE")
print("=" * 60)
print(f"  Best model : {best_name}")
print(f"  R²         : {best_result['R2']:.4f}")
print(f"  NSE        : {best_result['NSE']:.4f}  "
      f"({nse_rating(best_result['NSE'])})")
print(f"  RMSE       : {best_result['RMSE']:.5f} m³/m³")
print()
print(f"  Model saved : models/best_model.pkl")
print(f"  Figures     : figures/ (5 PNG files)")
print()
print("  Next step: run 03_predict_today.py")
print("=" * 60)
