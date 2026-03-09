# pred_change_maps.py
"""
Creates QGIS-ready CSV + plots for predicted 3-month and 6-month groundwater change.

Inputs:
- stgnn_prepared/meta.json
- multi_step_preds.npy   (H x N)
- Last observed groundwater value (from Y.npy or direct denorm)

Outputs:
- gw_pred_change.csv  (QGIS-ready)
- plot_change_3m.png
- plot_change_6m.png
- barplot_change_3m.png
- barplot_change_6m.png
"""

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import geopandas as gpd
import os

BOUNDARY = "Chennai_shapefile_extraction\chennai_wards_clipped.geojson"
OUT_DIR = "prediction_change_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ------------------------------------------------------
# LOAD META + TENSORS
# ------------------------------------------------------
meta = json.load(open("stgnn_prepared/meta.json"))
wards = meta["wards"]
y_mean = np.array(meta["y_mean_per_ward"])
y_std = np.array(meta["y_std_per_ward"])

# Load predictions
multi_preds = np.load("multi_step_preds.npy")     # shape (H, N)
H, N = multi_preds.shape

if H < 6:
    raise ValueError("multi_step_preds.npy does not contain at least 6 steps.")

# Last observed real groundwater level:
Y_scaled = np.load("stgnn_prepared/Y.npy")        # (N, T)
Y_last_norm = Y_scaled[:, -1]                    # per ward
Y_last_real = Y_last_norm * y_std + y_mean       # de-normalized

# ------------------------------------------------------
# COMPUTE CHANGES (Δ = Forecasted - Current)
# ------------------------------------------------------
pred_3m = multi_preds[2, :]   # index 2 → +3 months
pred_6m = multi_preds[5, :]   # index 5 → +6 months

delta_3m = pred_3m - Y_last_real
delta_6m = pred_6m - Y_last_real

# ------------------------------------------------------
# SAVE QGIS-READY CSV
# ------------------------------------------------------
df = pd.DataFrame({
    "Ward_No": wards,
    "Last_Observed": Y_last_real,
    "Pred_3M": pred_3m,
    "Pred_6M": pred_6m,
    "Delta_3M": delta_3m,
    "Delta_6M": delta_6m
})

csv_path = os.path.join(OUT_DIR, "gw_pred_change.csv")
df.to_csv(csv_path, index=False)
print("✔ Saved:", csv_path)

# ------------------------------------------------------
# SIMPLE BAR PLOTS
# ------------------------------------------------------
plt.figure(figsize=(14,6))
plt.bar([str(w) for w in wards], delta_3m)
plt.xticks(rotation=90, fontsize=6)
plt.title("Predicted Groundwater Change After 3 Months (Δ3M)")
plt.ylabel("Change (m)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "barplot_change_3m.png"), dpi=300)
plt.close()

plt.figure(figsize=(14,6))
plt.bar([str(w) for w in wards], delta_6m, color="orange")
plt.xticks(rotation=90, fontsize=6)
plt.title("Predicted Groundwater Change After 6 Months (Δ6M)")
plt.ylabel("Change (m)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "barplot_change_6m.png"), dpi=300)
plt.close()

print("✔ Saved bar plots in", OUT_DIR)

# ------------------------------------------------------
# HEATMAPS (simple for quick view)
# ------------------------------------------------------
plt.figure(figsize=(10,4))
plt.imshow(delta_3m.reshape(1,-1), cmap="coolwarm", aspect="auto")
plt.colorbar(label="Δ3M (m)")
plt.title("3-Month GW Change Heatmap")
plt.yticks([])
plt.xticks(range(len(wards)), wards, rotation=90, fontsize=6)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "plot_change_3m.png"), dpi=300)
plt.close()

plt.figure(figsize=(10,4))
plt.imshow(delta_6m.reshape(1,-1), cmap="coolwarm", aspect="auto")
plt.colorbar(label="Δ6M (m)")
plt.title("6-Month GW Change Heatmap")
plt.yticks([])
plt.xticks(range(len(wards)), wards, rotation=90, fontsize=6)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "plot_change_6m.png"), dpi=300)
plt.close()

print("✔ Saved heatmaps in", OUT_DIR)

# ------------------------------------------------------
# OPTIONAL: JOIN SHAPEFILE + EXPORT GEOJSON
# ------------------------------------------------------
try:
    gdf = gpd.read_file(BOUNDARY)

    # Normalize Ward column name
    for col in ["Ward_No", "WARD_NO", "ward_no", "ward", "id"]:
        if col in gdf.columns:
            gdf.rename(columns={col: "Ward_No"}, inplace=True)
            break

    gdf["Ward_No"] = gdf["Ward_No"].astype(int)

    gdf_out = gdf.merge(df, on="Ward_No", how="left")

    gdf_out.to_file(os.path.join(OUT_DIR, "gw_pred_change.geojson"), driver="GeoJSON")

    print("✔ Exported spatial change map:", "gw_pred_change.geojson")
except Exception as e:
    print("⚠ Could not export GeoJSON:", e)

print("\nDone. Load 'gw_pred_change.csv' or 'gw_pred_change.geojson' in QGIS for full heatmaps.")
