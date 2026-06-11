"""
MODULE 03 — SINGLE DAY PREDICTION & IRRIGATION RECOMMENDATION
Honors Project: ML-Based Irrigation Recommendation System

PURPOSE:
    You enter today's weather and soil readings manually,
    and the system predicts tomorrow's soil moisture and
    tells you exactly how much to irrigate.

USAGE:
    python 03_predict_today.py

REQUIREMENT:
    Run 02_ml_pipeline.py at least once first so the
    trained model is saved in the models/ folder.
"""

import os, sys, time
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ── Soil & site constants ─────────────────────────────────
FC  = 0.35
PWP = 0.14
LAT = -34.99
ELEV = 134.0

# ── Crop definitions (same as pipeline) ───────────────────
CROPS = {
    1: {"name":"Wheat",     "key":"wheat",     "Kc_ini":0.70,"Kc_mid":1.15,"Kc_end":0.45,"root_depth":0.60,"MAD":0.55},
    2: {"name":"Cotton",    "key":"cotton",    "Kc_ini":0.35,"Kc_mid":1.20,"Kc_end":0.70,"root_depth":1.00,"MAD":0.65},
    3: {"name":"Sugarcane", "key":"sugarcane", "Kc_ini":0.40,"Kc_mid":1.25,"Kc_end":0.75,"root_depth":1.20,"MAD":0.65},
    4: {"name":"Rice",      "key":"rice",      "Kc_ini":1.05,"Kc_mid":1.20,"Kc_end":1.05,"root_depth":0.50,"MAD":0.20},
    5: {"name":"Canola",    "key":"canola",    "Kc_ini":0.35,"Kc_mid":1.10,"Kc_end":0.35,"root_depth":0.60,"MAD":0.60},
}

WIDTH = 62

# ── Helpers ───────────────────────────────────────────────
def clear():
    os.system("cls" if os.name == "nt" else "clear")

def banner():
    clear()
    print("=" * WIDTH)
    print("  IRRIGATION RECOMMENDATION — TODAY'S PREDICTION")
    print("  Enter today's readings → Get tomorrow's advice")
    print("=" * WIDTH)
    print()

def ask_float(prompt, low, high, unit=""):
    """Ask user for a number, validate it's in range."""
    while True:
        try:
            val = float(input(f"    {prompt} [{low}–{high} {unit}]: ").strip())
            if low <= val <= high:
                return val
            print(f"    Please enter a value between {low} and {high}.")
        except ValueError:
            print("    Please enter a valid number.")

def et0_pm(Tmax, Tmin, RH, u2, Ra, elev=ELEV):
    """FAO-56 Penman-Monteith ET0."""
    Tm    = (Tmax + Tmin) / 2
    es    = (0.6108*np.exp(17.27*Tmax/(Tmax+237.3)) +
             0.6108*np.exp(17.27*Tmin/(Tmin+237.3))) / 2
    ea    = (RH / 100) * es
    Delta = 4098*(0.6108*np.exp(17.27*Tm/(Tm+237.3))) / (Tm+237.3)**2
    P_atm = 101.3*((293 - 0.0065*elev)/293)**5.26
    gamma = 0.000665 * P_atm
    Rns   = (1 - 0.23) * Ra
    Rnl   = (4.903e-9 * ((Tmax+273.16)**4 + (Tmin+273.16)**4)/2 *
             (0.34 - 0.14*np.sqrt(max(ea, 0.01))) * (1.35*0.75 - 0.35))
    Rn    = max(Rns - Rnl, 0)
    return max((0.408*Delta*Rn + gamma*(900/(Tm+273))*u2*(es-ea)) /
               (Delta + gamma*(1 + 0.34*u2)), 0)

def ra(doy, lat=LAT):
    """Extraterrestrial radiation FAO-56."""
    lat  = np.radians(lat)
    dr   = 1 + 0.033 * np.cos(2*np.pi*doy/365)
    decl = np.radians(23.45 * np.sin(np.radians((360/365)*(doy-81))))
    ws   = np.arccos(np.clip(-np.tan(lat)*np.tan(decl), -1, 1))
    return max((24*60/np.pi)*0.0820*dr*(ws*np.sin(lat)*np.sin(decl)+
               np.cos(lat)*np.cos(decl)*np.sin(ws)), 0)

def compute_kc(doy, crop):
    sd = (doy - 274) % 365
    if sd < 30:   return crop["Kc_ini"]
    if sd < 75:   return crop["Kc_ini"] + (sd-30) * (crop["Kc_mid"]-crop["Kc_ini"])/45
    if sd < 135:  return crop["Kc_mid"]
    if sd < 180:  return crop["Kc_mid"] - (sd-135) * (crop["Kc_mid"]-crop["Kc_end"])/45
    return crop["Kc_end"]

# ══════════════════════════════════════════════════════════
#  CHECK MODEL EXISTS
# ══════════════════════════════════════════════════════════
banner()

# Find any saved model
model_files = [f for f in os.listdir(MODEL_DIR) if f.startswith("best_model")]
if not model_files:
    print("  ERROR: No trained model found in models/ folder.")
    print("  Please run 02_ml_pipeline.py first to train the model.")
    input("\n  Press Enter to exit...")
    exit(1)

# ══════════════════════════════════════════════════════════
#  STEP 1 — SELECT CROP
# ══════════════════════════════════════════════════════════
print("  STEP 1 of 3 — SELECT YOUR CROP")
print()
for num, crop in CROPS.items():
    print(f"  [{num}]  {crop['name']}")
print()

while True:
    try:
        choice = int(input("  Enter crop number (1-5): ").strip())
        if choice in CROPS:
            break
        print("  Please enter a number between 1 and 5.")
    except ValueError:
        print("  Please enter a number between 1 and 5.")

CROP      = CROPS[choice]
Zr_mm     = CROP["root_depth"] * 1000
RAW       = CROP["MAD"] * (FC - PWP)
# Dynamic trigger: 30th percentile of historical SM distribution
# More robust than theoretical FC-RAW for this dataset
_df_sm    = pd.read_csv(os.path.join(DATA_DIR, "ozflux_synthetic_full.csv"))
trigger   = float(_df_sm["soil_moisture"].quantile(0.30))
del _df_sm

# Load matching model or fall back to any available
model_path = os.path.join(MODEL_DIR, f"best_model_{CROP['key']}.pkl")
if not os.path.exists(model_path):
    # Use whichever model is available
    model_path = os.path.join(MODEL_DIR, model_files[0])
    loaded_crop = model_files[0].replace("best_model_","").replace(".pkl","")
    print(f"\n  Note: Using model trained for {loaded_crop.upper()}.")
    print(f"  For best results, run 02_ml_pipeline.py and select {CROP['name']}.")

model = joblib.load(model_path)

# ══════════════════════════════════════════════════════════
#  STEP 2 — ENTER TODAY'S READINGS
# ══════════════════════════════════════════════════════════
banner()
print(f"  Crop : {CROP['name'].upper()}")
print()
print("  STEP 2 of 3 — ENTER TODAY'S READINGS")
print()
print("  Enter your sensor and weather readings for today.")
print("  Press Enter after each value.")
print()

# Date
from datetime import date, timedelta
today_date    = date.today()
tomorrow_date = today_date + timedelta(days=1)
doy           = today_date.timetuple().tm_yday

print(f"  Today's date : {today_date}  (Day of year: {doy})")
print()

print("  ── Soil Sensors ──────────────────────────────────")
sm   = ask_float("Soil Moisture",     0.14, 0.40, "m³/m³")

print()
print("  ── Weather Station ───────────────────────────────")
ta   = ask_float("Air Temperature (mean)", 0.0, 50.0, "°C")

# Max and min — optional, estimated from mean if not entered
tmax_input = input(f"    Air Temp Max [{ta}–55.0 °C] (Enter=auto): ").strip()
tmin_input = input(f"    Air Temp Min [-5.0–{ta} °C] (Enter=auto): ").strip()

if tmax_input:
    tmax = float(tmax_input)
    print(f"    Tmax → {tmax} °C (entered)")
else:
    tmax = round(ta + 6.5, 1)
    print(f"    Tmax → {tmax} °C (estimated: mean + 6.5)")

if tmin_input:
    tmin = float(tmin_input)
    print(f"    Tmin → {tmin} °C (entered)")
else:
    tmin = round(ta - 6.5, 1)
    print(f"    Tmin → {tmin} °C (estimated: mean - 6.5)")
rh   = ask_float("Relative Humidity",      5.0, 100.0, "%")
rain = ask_float("Rainfall today",         0.0,  150.0, "mm")

print()
print("  ── Optional (press Enter to use defaults) ─────────")

# Wind speed — default 2.0 m/s if not available
ws_input = input("    Wind Speed at 2m [0.0–20.0 m/s] (Enter=2.0): ").strip()
ws = float(ws_input) if ws_input else 2.0
print(f"    Wind Speed → {ws} m/s")

# Soil temperature — estimated from air temp if not available
ts_input = input("    Soil Temperature [5.0–60.0 °C] (Enter=auto): ").strip()
if ts_input:
    ts = float(ts_input)
    print(f"    Soil Temperature → {ts} °C (entered)")
else:
    ts = round(ta * 0.85 + 4.2, 1)
    print(f"    Soil Temperature → {ts} °C (estimated from air temp: {ta} × 0.85 + 4.2)")

# ── Compute derived variables automatically ───────────────
Ra_val  = ra(doy)
ET0_val = et0_pm(tmax, tmin, rh, max(ws, 0.5), Ra_val)
es_val  = 0.6108 * np.exp(17.27*ta / (ta+237.3))
vpd_val = max(es_val * (1 - rh/100), 0)
Kc_val  = compute_kc(doy, CROP)

# ══════════════════════════════════════════════════════════
#  STEP 3 — LOAD HISTORY FOR LAG FEATURES
#  We need the past 7 days of SM and rainfall to build
#  the autoregressive features the model was trained on.
#  We load them from the saved dataset automatically.
# ══════════════════════════════════════════════════════════
data_path = os.path.join(DATA_DIR, "ozflux_synthetic_full.csv")
df_hist = pd.read_csv(data_path)

# Get recent history — use last 10 rows as proxy for recent conditions
# (In a real deployment these would be actual sensor logs)
recent = df_hist.tail(10).reset_index(drop=True)

sm_lag1  = recent["soil_moisture"].iloc[-1]
sm_lag3  = recent["soil_moisture"].iloc[-3]
sm_lag7  = recent["soil_moisture"].iloc[-7] if len(recent) >= 7 else sm
rain_3d  = recent["rainfall"].tail(3).sum()
rain_7d  = recent["rainfall"].tail(7).sum()
ET0_7d   = recent["ET0_mm"].tail(7).mean()
T_7d     = recent["air_temperature"].tail(7).mean()

# ── Build feature vector ──────────────────────────────────
# Must match FEATURES list in 02_ml_pipeline.py exactly
FEATURES = [
    "soil_moisture",
    "sm_lag1", "sm_lag3", "sm_lag7",
    "soil_temperature", "air_temperature", "air_temp_max", "air_temp_min",
    "humidity", "rainfall",
    "ET0_mm", "VPD_kPa", "wind_speed_2m",
    "rain_3d", "rain_7d", "ET0_7d", "T_7d",
    "Kc", "season_day", "doy",
]

feature_values = {
    "soil_moisture"  : sm,
    "sm_lag1"        : sm_lag1,
    "sm_lag3"        : sm_lag3,
    "sm_lag7"        : sm_lag7,
    "soil_temperature": ts,
    "air_temperature": ta,
    "air_temp_max"   : tmax,
    "air_temp_min"   : tmin,
    "humidity"       : rh,
    "rainfall"       : rain,
    "ET0_mm"         : ET0_val,
    "VPD_kPa"        : vpd_val,
    "wind_speed_2m"  : ws,
    "rain_3d"        : rain_3d + rain,   # include today
    "rain_7d"        : rain_7d + rain,
    "ET0_7d"         : (ET0_7d * 6 + ET0_val) / 7,
    "T_7d"           : (T_7d  * 6 + ta)  / 7,
    "Kc"             : Kc_val,
    "season_day"     : doy,
    "doy"            : doy,
}

X_input = np.array([[feature_values[f] for f in FEATURES]])

# ── Predict ───────────────────────────────────────────────
pred_sm_ml = float(np.clip(model.predict(X_input)[0], PWP, FC + 0.02))

# Physics-based check: estimate SM tomorrow from water balance
# SM(t+1) = SM(t) + (effective_rain - crop_ET) / root_zone
ETc_today  = ET0_val * Kc_val
delta_sm   = (rain * 0.85 - ETc_today) / Zr_mm
phys_sm    = float(np.clip(sm + delta_sm, PWP, FC + 0.02))

# Blend ML prediction with physics
# When soil is already dry → trust physics more (model over-predicts recovery)
# When soil is healthy    → trust ML more
if sm < trigger:
    pred_sm = 0.30 * pred_sm_ml + 0.70 * phys_sm
else:
    pred_sm = 0.80 * pred_sm_ml + 0.20 * phys_sm

pred_sm  = float(np.clip(pred_sm, PWP, FC + 0.02))
smd_m3m3 = max(FC - pred_sm, 0)
smd_mm   = smd_m3m3 * Zr_mm
irr_vol  = round(smd_mm, 1) if pred_sm < trigger else 0.0

# ══════════════════════════════════════════════════════════
#  DISPLAY RESULT
# ══════════════════════════════════════════════════════════
banner()
print(f"  Crop : {CROP['name'].upper()}")
print()
print("  STEP 3 of 3 — RESULT")
print()

# Summary of inputs
print("  ── Today's Inputs (" + str(today_date) + ") ──────────────────────")
print(f"    Soil moisture      : {sm:.3f} m³/m³")
print(f"    Soil temperature   : {ts:.1f} °C")
print(f"    Air temperature    : {ta:.1f} °C  (max {tmax:.1f} / min {tmin:.1f})")
print(f"    Humidity           : {rh:.1f} %")
print(f"    Rainfall           : {rain:.1f} mm")
print(f"    Wind speed         : {ws:.1f} m/s")
print()
print("  ── Computed Variables ────────────────────────────")
print(f"    ET₀ (FAO-56 P-M)   : {ET0_val:.2f} mm/day")
print(f"    VPD                : {vpd_val:.3f} kPa")
print(f"    Crop coefficient   : {Kc_val:.3f}  ({CROP['name']})")
print(f"    Crop ET demand     : {ET0_val*Kc_val:.2f} mm/day")
print()
print(f"  ── Prediction for Tomorrow ({tomorrow_date}) ──────────")
print(f"    Predicted soil moisture : {pred_sm:.3f} m³/m³")
print(f"    Soil moisture deficit   : {smd_mm:.1f} mm")
print(f"    Irrigation trigger      : SM < {trigger:.3f} m³/m³")
print()

# ── The recommendation box ────────────────────────────────
if irr_vol > 0:
    pct_depleted = (FC - pred_sm) / (FC - PWP) * 100
    print("  ╔══════════════════════════════════════════════╗")
    print("  ║   💧  IRRIGATE TOMORROW                      ║")
    print("  ╠══════════════════════════════════════════════╣")
    print(f"  ║   Recommended volume : {irr_vol:>6.1f} mm             ║")
    print(f"  ║   Root zone depleted : {pct_depleted:>5.1f} %              ║")
    print(f"  ║   Refills SM to FC   : {FC:.3f} m³/m³          ║")
    print("  ╠══════════════════════════════════════════════╣")
    print("  ║   Reason: Predicted SM will fall below the   ║")
    print(f"  ║   {CROP['name']} stress threshold ({trigger:.3f} m³/m³)    ║")
    print("  ╚══════════════════════════════════════════════╝")
else:
    pct_available = (pred_sm - PWP) / (FC - PWP) * 100
    print("  ╔══════════════════════════════════════════════╗")
    print("  ║   ✓   NO IRRIGATION NEEDED TOMORROW          ║")
    print("  ╠══════════════════════════════════════════════╣")
    print(f"  ║   Predicted SM       : {pred_sm:.3f} m³/m³          ║")
    print(f"  ║   Available water    : {pct_available:>5.1f} % of capacity   ║")
    print(f"  ║   Above trigger by   : {(pred_sm-trigger)*1000:>4.1f} mm³/m³         ║")
    print("  ╠══════════════════════════════════════════════╣")
    print("  ║   Re-check tomorrow with updated readings    ║")
    print("  ╚══════════════════════════════════════════════╝")

print()
print("=" * WIDTH)
print("  Prediction complete.")
print("  Model: trained on 9 years NASA POWER data (2016-2024)")
print(f"  Site : Yanco, NSW  |  Crop: {CROP['name']}")
print("=" * WIDTH)
input("\n  Press Enter to exit...")
