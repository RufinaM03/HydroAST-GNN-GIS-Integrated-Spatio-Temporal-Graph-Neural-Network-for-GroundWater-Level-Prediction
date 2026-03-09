import pandas as pd
import numpy as np
import json
from pathlib import Path
from math import radians, sin, cos, sqrt, atan2

# =====================================================
# SETTINGS
# =====================================================
INPUT_CSV = "Dataset_v5.csv"
OUT_DIR = "stgnn_prepared"
Path(OUT_DIR).mkdir(exist_ok=True)

# =====================================================
# 1. LOAD DATA
# =====================================================
df = pd.read_csv(INPUT_CSV)

# REMOVE rows where gw_level is empty / NaN
# -----------------------------------------------------
df = df.dropna(subset=["gw_level"])

# If some rows have gw_level as blank strings, fix those too:
df["gw_level"].replace(["", " ", "NA", "na"], pd.NA, inplace=True)
df = df.dropna(subset=["gw_level"])

# Convert to numeric (forces any non-numeric leftover to NaN)
df["gw_level"] = pd.to_numeric(df["gw_level"], errors="coerce")
df = df.dropna(subset=["gw_level"])
# -----------------------------------------------------
# Ensure period is datetime and sort
df["period"] = pd.to_datetime(df["period"])
df = df.sort_values(["Ward_No", "period"])

# =====================================================
# 2. BUILD GLOBAL PERIOD INDEX
# =====================================================
all_periods = sorted(df["period"].unique())
wards = sorted(df["Ward_No"].unique())

# =====================================================
# 3. REINDEX EACH WARD TO FULL TIMELINE
# =====================================================
df_list = []
for w in wards:
    d = df[df["Ward_No"] == w].set_index("period")
    d = d.reindex(all_periods)
    d["Ward_No"] = w
    df_list.append(d)

df_full = pd.concat(df_list).reset_index().rename(columns={"index": "period"})

# =====================================================
# 4. IMPUTATION — ONLY NUMERIC COLUMNS
# =====================================================
numeric_cols = [
    col for col in df_full.columns
    if col not in ["period", "Ward_No"]
]

df_full[numeric_cols] = df_full.groupby("Ward_No")[numeric_cols].transform(
    lambda x: x.interpolate(method="linear").ffill().bfill()
)

# =====================================================
# 5. ADD SEASONAL FEATURES
# =====================================================
df_full["month"] = df_full["period"].dt.month
df_full["month_sin"] = np.sin(2 * np.pi * df_full["month"] / 12)
df_full["month_cos"] = np.cos(2 * np.pi * df_full["month"] / 12)

# =====================================================
# 6. ADD GROUNDWATER LAG FEATURES
# =====================================================
df_full = df_full.sort_values(["Ward_No", "period"])

for lag in [1, 3, 6, 12]:
    df_full[f"gw_lag_{lag}"] = df_full.groupby("Ward_No")["gw_level"].shift(lag)

for lag in [1, 3, 6, 12]:
    df_full[f"gw_lag_{lag}"] = df_full.groupby("Ward_No")[f"gw_lag_{lag}"].transform(
        lambda x: x.bfill().ffill()
    )

# =====================================================
# 7. BUILD TENSORS
# =====================================================
feature_cols = [
    "NDVI_mean", "Rainfall_mean", "Temperature_mean",
    "ET-mm_mean", "LULC_mean",
    "month_sin", "month_cos",
    "gw_lag_1", "gw_lag_3", "gw_lag_6", "gw_lag_12"
]

target_col = "gw_level"

N = len(wards)
T = len(all_periods)
F = len(feature_cols)

X = np.zeros((N, T, F), dtype=np.float32)
Y = np.zeros((N, T), dtype=np.float32)

for i, w in enumerate(wards):
    d = df_full[df_full["Ward_No"] == w].reset_index(drop=True)
    X[i] = d[feature_cols].values
    Y[i] = d[target_col].values

# =====================================================
# 8. WARD-WISE NORMALIZATION
# =====================================================
y_mean_per_ward = Y.mean(axis=1, keepdims=True)
y_std_per_ward = Y.std(axis=1, keepdims=True) + 1e-6

Y_scaled = (Y - y_mean_per_ward) / y_std_per_ward

# =====================================================
# 9. GLOBAL NORMALIZATION FOR FEATURES
# =====================================================
X_flat = X.reshape(-1, F)

X_mean = np.nanmean(X_flat, axis=0)
X_std = np.nanstd(X_flat, axis=0) + 1e-6

X_scaled = (X - X_mean.reshape(1, 1, -1)) / X_std.reshape(1, 1, -1)

# =====================================================
# 10. SAVE OUTPUT FILES
# =====================================================
np.save(f"{OUT_DIR}/X.npy", X_scaled)
np.save(f"{OUT_DIR}/Y.npy", Y_scaled)

# =====================================================
# 11. SAVE META
# =====================================================

def to_py(obj):
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, list):
        return [to_py(v) for v in obj]
    return obj

meta = {
    "wards": wards,
    "periods": [str(p) for p in all_periods],
    "features": feature_cols,
    "X_mean": X_mean.tolist(),
    "X_std": X_std.tolist(),
    "y_mean_per_ward": y_mean_per_ward.flatten().tolist(),
    "y_std_per_ward": y_std_per_ward.flatten().tolist()
}

meta_clean = {k: to_py(v) for k, v in meta.items()}

with open(f"{OUT_DIR}/meta.json", "w") as f:
    json.dump(meta_clean, f, indent=2)

print("✓ Tensors prepared successfully!")
print("X shape:", X_scaled.shape)
print("Y shape:", Y_scaled.shape)
