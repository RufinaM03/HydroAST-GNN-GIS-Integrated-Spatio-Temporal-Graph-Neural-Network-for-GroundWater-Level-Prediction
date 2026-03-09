# multi_step_forecast.py
"""
Iterative multi-step forecasting (auto-regressive) using a 1-step HydroASTGNN model.

Outputs:
 - multi_step_preds.npy   : shape (H, N) predictions in real units
 - multi_step_predictions.csv : readable CSV with step, Ward_No, prediction

Usage:
    python multi_step_forecast.py --steps 6
"""
import numpy as np
import json
import argparse
import torch
import pandas as pd
from HydroASTGNN_training import HydroASTGNN, L, device
import math
from datetime import datetime
from dateutil.relativedelta import relativedelta

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=6, help="Forecast horizon steps (months)")
parser.add_argument("--output", type=str, default="multi_step_predictions.csv")
args = parser.parse_args()

H = args.steps

# load meta + tensors
meta = json.load(open("stgnn_prepared/meta.json"))
wards = meta["wards"]
features = meta["features"]   # order must match X.npy columns
X_mean = np.array(meta["X_mean"])
X_std  = np.array(meta["X_std"])
y_mean_per_ward = np.array(meta["y_mean_per_ward"])
y_std_per_ward = np.array(meta["y_std_per_ward"])
periods = [p for p in meta["periods"]]

X_scaled = np.load("stgnn_prepared/X.npy")  # (N, T, F)
Y_scaled = np.load("stgnn_prepared/Y.npy")  # not used except maybe for lags

N, T, F = X_scaled.shape

# indices of gw lag features in features list
lag_names = ["gw_lag_1", "gw_lag_3", "gw_lag_6", "gw_lag_12"]
lag_indices = [features.index(name) for name in lag_names]

# indices for month_sin & month_cos
month_sin_idx = features.index("month_sin") if "month_sin" in features else None
month_cos_idx = features.index("month_cos") if "month_cos" in features else None

# model load
model = HydroASTGNN(N, F).to(device)
model.load_state_dict(torch.load("best_HydroASTGNN.pth", map_location=device))
model.eval()

# Starting last known window (scaled)
cur_X_window = X_scaled[:, -L:, :].copy()   # N x L x F

# find last period and generate future periods
last_period = datetime.fromisoformat(periods[-1])
future_periods = [last_period + relativedelta(months=i+1) for i in range(H)]

multi_preds = np.zeros((H, N), dtype=np.float32)

for h in range(H):
    # forward predict 1-step
    x_input = torch.tensor(cur_X_window).float().unsqueeze(0).to(device)  # 1,N,L,F
    with torch.no_grad():
        pred_norm, _ = model(x_input)
        pred_norm = pred_norm.squeeze(0).cpu().numpy()[:, 0]  # N

    # pred_norm is Y_scaled (per-ward normalized). Convert to real:
    pred_real = pred_norm * y_std_per_ward + y_mean_per_ward   # N

    multi_preds[h, :] = pred_real

    # Update cur_X_window for next step:
    # We need to create a new feature vector for each ward for the next time step,
    # then append it and drop the oldest.
    new_feat = np.zeros((N, F), dtype=np.float32)

    # For each ward, construct raw feature values for new time step:
    for i in range(N):
        # start by copying the last time-step's *raw* features:
        # cur_X_window is scaled; convert last step back to raw for manipulation:
        last_scaled = cur_X_window[i, -1, :]   # F
        last_raw = last_scaled * X_std + X_mean

        # update gw lags: new gw is pred_real[i]
        # compute raw new lag values (for gw_lag_1 we put last observed gw; but we use shift logic)
        # For simplicity we will compute new raw lag entries using the previous raw gw series:
        # Extract previous raw lag values:
        # We'll maintain a small raw_gw_history to compute realistic lags:
        # Build raw history from cur_X_window gw_lag_1 etc if available
        # If the lag features exist, we'll shift them:
        last_raw_copy = last_raw.copy()

        # shift gw lag features: gw_lag_12 <= gw_lag_11 etc - but we only stored specific lags.
        # We'll approximate shifting by:
        # new gw_lag_1   = previous pred_real (i.e., latest actual/pred)
        # new gw_lag_3   = value that was previously gw_lag_2 if it existed; in our simple case, approximate by previous gw_lag_2 ~ previous gw_lag_1
        # Simpler robust approach: keep gw_lag_1 = most recent (pred_real), gw_lag_3 = previous gw_lag_1, gw_lag_6 = previous gw_lag_3, gw_lag_12 = previous gw_lag_6

        # get previous lag raw values (if present in features)
        prev_lag_vals = {}
        for idx_name, idx_col in zip(lag_names, lag_indices):
            prev_lag_vals[idx_name] = last_raw[idx_col]

        # compute new raw lag values:
        new_lag_vals = {}
        new_lag_vals["gw_lag_1"] = pred_real[i]
        new_lag_vals["gw_lag_3"] = prev_lag_vals.get("gw_lag_1", pred_real[i])
        new_lag_vals["gw_lag_6"] = prev_lag_vals.get("gw_lag_3", pred_real[i])
        new_lag_vals["gw_lag_12"] = prev_lag_vals.get("gw_lag_6", pred_real[i])

        # fill new_feat raw with last_raw as base, then overwrite lag and seasonal
        new_raw = last_raw.copy()

        # assign new lag raw values into correct columns
        for name, val in new_lag_vals.items():
            col_idx = features.index(name)
            new_raw[col_idx] = val

        # update seasonal month features using future_periods[h]
        if month_sin_idx is not None and month_cos_idx is not None:
            m = future_periods[h].month
            new_raw[month_sin_idx] = math.sin(2 * math.pi * m / 12)
            new_raw[month_cos_idx] = math.cos(2 * math.pi * m / 12)

        # assign new_feat scaled values
        new_feat[i, :] = (new_raw - X_mean) / X_std

    # roll window: drop oldest, append new_feat
    cur_X_window = np.concatenate([cur_X_window[:, 1:, :], new_feat[:, None, :]], axis=1)

# save outputs
np.save("multi_step_preds.npy", multi_preds)   # shape H x N

# write CSV
rows = []
for h in range(H):
    for i, w in enumerate(wards):
        rows.append({"step": h+1, "ward": int(w), "period": future_periods[h].strftime("%Y-%m-%d"), "prediction": float(multi_preds[h, i])})

pd.DataFrame(rows).to_csv(args.output, index=False)
print("Saved multi-step predictions:", args.output)
