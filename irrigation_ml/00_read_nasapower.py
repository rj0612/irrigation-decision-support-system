"""
MODULE 00 — NASA POWER CSV Reader & Preprocessor
Honors Project: ML-Based Irrigation Recommendation System

PURPOSE:
    Reads your single downloaded NASA POWER CSV file,
    computes FAO-56 ET0, runs the soil water balance to
    simulate soil moisture, and saves a clean CSV ready
    for 02_ml_pipeline.py.

YOUR FILE:
    Put this file in the same folder as this script:
    POWER_Point_Daily_20150611_20260610_034d99S_146d42E_LST_.csv

    This is the file that includes solar radiation
    (ALLSKY_SFC_LW_DWN) along with all other variables.

USAGE:
    python 00_read_nasapower.py
    python 02_ml_pipeline.py
"""

import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# ─────────────────────────────────────────────────────────
# CONFIGURATION — edit filename to the same as dataset files.
# ─────────────────────────────────────────────────────────
FILE_DATA = "POWER_Point_Daily_20150611_20260610_034d99S_146d42E_LST .csv"

# Site constants
LAT        = -34.99
ELEV       = 134.0

# Soil constants — sandy loam (Yanco region)
FC         = 0.35
PWP        = 0.14
RAW        = 0.50 * (FC - PWP)
ROOT_DEPTH = 0.60

# ─────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

print("=" * 60)
print("  NASA POWER Data Reader")
print(f"  Site : Yanco, NSW  ({LAT}, 146.42)")
print("=" * 60)

# ─────────────────────────────────────────────────────────
# STEP 1 — READ FILE
# ─────────────────────────────────────────────────────────
filepath = os.path.join(BASE_DIR, FILE_DATA)

if not os.path.exists(filepath):
    print(f"\n  ERROR: File not found:")
    print(f"  {filepath}")
    print(f"\n  Make sure '{FILE_DATA}' is in the same")
    print(f"  folder as this script.")
    exit(1)

# Auto-detect header length by finding the YEAR row
skip = None
with open(filepath, "r") as f:
    for i, line in enumerate(f):
        if line.startswith("YEAR"):
            skip = i
            break

if skip is None:
    print("  ERROR: Could not find data header in file.")
    exit(1)

df_raw = pd.read_csv(filepath, skiprows=skip)
df_raw = df_raw.replace(-999,   np.nan)
df_raw = df_raw.replace(-999.0, np.nan)

print(f"\n  File      : {FILE_DATA}")
print(f"  Columns   : {df_raw.columns.tolist()}")
print(f"  Records   : {len(df_raw)} rows")
print(f"  Years     : {int(df_raw['YEAR'].min())} – {int(df_raw['YEAR'].max())}")

# ─────────────────────────────────────────────────────────
# STEP 2 — FILTER COMPLETE YEARS ONLY
# Partial years (2015 starts Jun, 2026 ends Jun) are excluded
# so every year in the dataset has a full 365/366 days.
# This ensures clean train/test splits in the ML pipeline.
# ─────────────────────────────────────────────────────────
year_counts   = df_raw.groupby("YEAR").size()
complete_years = year_counts[year_counts >= 365].index.tolist()
df_raw = df_raw[df_raw["YEAR"].isin(complete_years)].reset_index(drop=True)

print(f"  Complete years kept : {complete_years}")
print(f"  Records after filter: {len(df_raw)}")

# ─────────────────────────────────────────────────────────
# STEP 3 — BUILD DATE COLUMN & RENAME VARIABLES
# ─────────────────────────────────────────────────────────
df_raw["date"] = pd.to_datetime(
    df_raw["YEAR"].astype(str) +
    df_raw["DOY"].astype(str).str.zfill(3),
    format="%Y%j"
)

df = pd.DataFrame()
df["date"]            = df_raw["date"]
df["year"]            = df_raw["YEAR"].astype(int)
df["doy"]             = df_raw["DOY"].astype(int)
df["season_day"]      = df_raw["DOY"].astype(int)
df["air_temperature"] = df_raw["T2M"]
df["air_temp_max"]    = df_raw["T2M_MAX"]
df["air_temp_min"]    = df_raw["T2M_MIN"]
df["humidity"]        = df_raw["RH2M"]
df["rainfall"]        = df_raw["PRECTOTCORR"].clip(lower=0)
df["wind_speed_2m"]   = df_raw["WS2M"].clip(lower=0.5)

# Solar radiation — use file column if present
if "ALLSKY_SFC_LW_DWN" in df_raw.columns:
    df["solar_rad_MJ"] = df_raw["ALLSKY_SFC_LW_DWN"]
    print("\n  Solar radiation column found and loaded.")
elif "ALLSKY_SFC_SW_DWN" in df_raw.columns:
    df["solar_rad_MJ"] = df_raw["ALLSKY_SFC_SW_DWN"]
    print("\n  Solar radiation column found and loaded.")
else:
    df["solar_rad_MJ"] = np.nan
    print("\n  Note: No solar radiation column — will estimate from latitude.")

# Sanity clipping
df["air_temperature"] = df["air_temperature"].clip(-5, 55)
df["air_temp_max"]    = df["air_temp_max"].clip(df["air_temperature"], 55)
df["air_temp_min"]    = df["air_temp_min"].clip(-5, df["air_temperature"])
df["humidity"]        = df["humidity"].clip(5, 100)

# Fill the few missing values by linear interpolation
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit=3)

# ─────────────────────────────────────────────────────────
# STEP 4 — FAO-56 PENMAN-MONTEITH ET0
# Reference: Allen et al. (1998), FAO Irrigation Paper 56
# ─────────────────────────────────────────────────────────
print("  Computing FAO-56 ET0 (Penman-Monteith)...")

def extraterrestrial_radiation(doy, lat_deg):
    """Ra [MJ/m2/day] — FAO-56 Eq.21"""
    lat  = np.radians(lat_deg)
    dr   = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    decl = np.radians(23.45 * np.sin(np.radians((360/365) * (doy - 81))))
    ws   = np.arccos(np.clip(-np.tan(lat) * np.tan(decl), -1, 1))
    Ra   = ((24*60/np.pi) * 0.0820 * dr *
            (ws*np.sin(lat)*np.sin(decl) +
             np.cos(lat)*np.cos(decl)*np.sin(ws)))
    return np.maximum(Ra, 0)

def et0_penman_monteith(Tmax, Tmin, RH, u2, Ra, elev=ELEV):
    """FAO-56 Penman-Monteith ET0 [mm/day]"""
    Tm    = (Tmax + Tmin) / 2
    es    = (0.6108*np.exp(17.27*Tmax/(Tmax+237.3)) +
             0.6108*np.exp(17.27*Tmin/(Tmin+237.3))) / 2
    ea    = (RH / 100) * es
    Delta = 4098*(0.6108*np.exp(17.27*Tm/(Tm+237.3))) / (Tm+237.3)**2
    P_atm = 101.3*((293 - 0.0065*elev)/293)**5.26
    gamma = 0.000665 * P_atm
    Rns   = (1 - 0.23) * Ra
    Rnl   = (4.903e-9 *
             ((Tmax+273.16)**4 + (Tmin+273.16)**4) / 2 *
             (0.34 - 0.14*np.sqrt(np.maximum(ea, 0.01))) *
             (1.35 * 0.75 - 0.35))
    Rn    = np.maximum(Rns - Rnl, 0)
    ET0   = ((0.408*Delta*Rn + gamma*(900/(Tm+273))*u2*(es-ea)) /
             (Delta + gamma*(1 + 0.34*u2)))
    return np.maximum(ET0, 0)

Ra_arr         = extraterrestrial_radiation(df["doy"].values, LAT)
df["Ra_MJ_m2"] = np.round(Ra_arr, 3)
df["ET0_mm"]   = np.round(et0_penman_monteith(
    df["air_temp_max"].values,
    df["air_temp_min"].values,
    df["humidity"].values,
    df["wind_speed_2m"].values,
    Ra_arr), 2)

# VPD from temperature and humidity
es            = 0.6108 * np.exp(17.27*df["air_temperature"] / (df["air_temperature"]+237.3))
df["VPD_kPa"] = np.round((es * (1 - df["humidity"]/100)).clip(lower=0), 3)

print(f"  ET0 range : {df['ET0_mm'].min():.1f} – {df['ET0_mm'].max():.1f} mm/day")
print(f"  ET0 mean  : {df['ET0_mm'].mean():.1f} mm/day")

# ─────────────────────────────────────────────────────────
# STEP 5 — CROP COEFFICIENT (FAO-56 wheat Kc)
# Default wheat Kc — the ML pipeline will override this
# with the correct Kc for whichever crop you select.
# ─────────────────────────────────────────────────────────
sd          = (df["doy"].values - 274) % 365
df["Kc"]    = np.round(np.where(sd < 30,  0.70,
              np.where(sd < 75,  0.70 + (sd-30)*(0.45)/45,
              np.where(sd < 135, 1.15,
              np.where(sd < 180, 1.15 - (sd-135)*0.70/45,
                                 0.45)))), 3)
df["ETc_mm"] = np.round(df["ET0_mm"] * df["Kc"], 2)

# ─────────────────────────────────────────────────────────
# STEP 6 — FAO-56 SOIL WATER BALANCE
# Simulates daily soil moisture driven by real NASA POWER
# weather data. Same physics as the original data generator.
# ─────────────────────────────────────────────────────────
print("  Running FAO-56 soil water balance...")

Zr_mm      = ROOT_DEPTH * 1000
theta      = np.zeros(len(df))
irrigation = np.zeros(len(df))
theta[0]   = FC - 0.03

rain_arr = df["rainfall"].fillna(0).values
ETc_arr  = df["ETc_mm"].fillna(0).values

for t in range(1, len(df)):
    P_eff    = rain_arr[t] * 0.85
    tc       = PWP + RAW
    Ks       = np.clip((theta[t-1]-PWP) / (tc-PWP), 0, 1)
    ETa      = ETc_arr[t] * Ks
    D        = max((theta[t-1]-FC) * Zr_mm, 0)
    theta[t] = np.clip(theta[t-1] + (P_eff-ETa-D)/Zr_mm, PWP, FC+0.02)

    if theta[t] < (FC - RAW):
        deficit        = (FC - theta[t]) * Zr_mm
        irrigation[t]  = round(deficit * np.random.uniform(0.90, 1.05), 1)
        theta[t]       = FC

df["soil_moisture"]  = np.round(theta, 4)
df["irrigation_mm"]  = np.round(irrigation, 1)
df["irrigated"]      = (df["irrigation_mm"] > 0).astype(int)

# Soil temperature estimated from air temp with thermal lag
df["soil_temperature"] = np.round(
    df["air_temperature"] * 0.85 + 4.2 +
    np.random.normal(0, 0.8, len(df)), 2).clip(lower=3)

# ─────────────────────────────────────────────────────────
# STEP 7 — FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────
df["smd_m3m3"] = (FC - df["soil_moisture"]).clip(lower=0).round(4)
df["smd_mm"]   = (df["smd_m3m3"] * Zr_mm).round(1)
df["sm_lag1"]  = df["soil_moisture"].shift(1).bfill()
df["sm_lag3"]  = df["soil_moisture"].shift(3).bfill()
df["sm_lag7"]  = df["soil_moisture"].shift(7).bfill()
df["rain_3d"]  = df["rainfall"].rolling(3,  min_periods=1).sum()
df["rain_7d"]  = df["rainfall"].rolling(7,  min_periods=1).sum()
df["ET0_7d"]   = df["ET0_mm"].rolling(7,   min_periods=1).mean()
df["T_7d"]     = df["air_temperature"].rolling(7, min_periods=1).mean()

# ─────────────────────────────────────────────────────────
# STEP 8 — SAVE
# ─────────────────────────────────────────────────────────
out_path = os.path.join(DATA_DIR, "ozflux_synthetic_full.csv")
df.to_csv(out_path, index=False)

# ─────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────
irr_days = df[df["irrigated"] == 1]
print()
print("=" * 60)
print("  DATA SUMMARY")
print("=" * 60)
print(f"  Source    : NASA POWER  (MERRA-2 reanalysis)")
print(f"  Location  : Yanco, NSW  ({LAT}, 146.42)")
print(f"  Years     : {df['year'].min()} – {df['year'].max()}  ({df['year'].nunique()} complete years)")
print(f"  Records   : {len(df)} days")
print()
print(f"  {'Variable':25s}  {'Min':>7}  {'Mean':>7}  {'Max':>7}")
print("  " + "-" * 50)
for col, unit in [
    ("air_temperature",   "°C"),
    ("air_temp_max",      "°C"),
    ("air_temp_min",      "°C"),
    ("humidity",          "%"),
    ("rainfall",          "mm"),
    ("wind_speed_2m",     "m/s"),
    ("ET0_mm",            "mm"),
    ("VPD_kPa",           "kPa"),
    ("soil_moisture",     "m³/m³"),
    ("soil_temperature",  "°C"),
    ("smd_mm",            "mm"),
    ("irrigation_mm",     "mm"),
]:
    print(f"  {col:25s}  {df[col].min():>7.2f}  "
          f"{df[col].mean():>7.2f}  {df[col].max():>7.2f}  {unit}")

print()
print(f"  Irrigation events : {df['irrigated'].sum()} ({df['irrigated'].mean()*100:.1f}% of days)")
if len(irr_days):
    print(f"  Mean irr. volume  : {irr_days['irrigation_mm'].mean():.1f} mm per event")
print()
print(f"  Saved to: {out_path}")
print()
print("=" * 60)
print("  SUCCESS!")
print("  Next step: run 02_ml_pipeline.py")
print("=" * 60)
