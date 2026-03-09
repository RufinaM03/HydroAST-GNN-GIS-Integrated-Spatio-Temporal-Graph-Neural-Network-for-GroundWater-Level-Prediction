# spatial_error_heatmap_v4.py
import json
import numpy as np
import pandas as pd
import geopandas as gpd

# =====================================================
# Load meta + tensors
# =====================================================
meta = json.load(open("stgnn_prepared/meta.json"))
wards = meta["wards"]

X = np.load("stgnn_prepared/X.npy")
Y = np.load("stgnn_prepared/Y.npy")

# Load evaluation predictions
# (These npy files are created in evaluation_v4.py)
preds = np.load("preds_full.npy")  # shape (T-L, N)
trues = np.load("trues_full.npy")  # shape (T-L, N)

# =====================================================
# Metrics
# =====================================================
def rmse(y,p): return float(np.sqrt(np.mean((y-p)**2)))
def mae(y,p): return float(np.mean(np.abs(y-p)))
def corr(y,p):
    try: return float(np.corrcoef(y,p)[0,1])
    except: return 0.0
def smape(y,p):
    return float(np.mean(2*np.abs(y-p)/(np.abs(y)+np.abs(p)+1e-6))*100)
def kge(y,p):
    R = corr(y,p)
    a = p.mean()/(y.mean()+1e-6)
    b = p.std()/(y.std()+1e-6)
    return float(1 - np.sqrt((R-1)**2 + (a-1)**2 + (b-1)**2))

def bias(y,p): return float(np.mean(p-y))

# =====================================================
# Build metrics per ward
# =====================================================
records = []

for i, w in enumerate(wards):
    y = trues[:, i]
    p = preds[:, i]

    rec = {
        "Ward_No": int(w),
        "RMSE": rmse(y,p),
        "MAE": mae(y,p),
        "R": corr(y,p),
        "Bias": bias(y,p),
        "SMAPE": smape(y,p),
        "KGE": kge(y,p)
    }
    records.append(rec)

df_metrics = pd.DataFrame(records)
df_metrics.to_csv("gw_error_metrics.csv", index=False)
print("Saved gw_error_metrics.csv")

# =====================================================
# Load ward boundary file
# =====================================================
shp_path = "Chennai_shapefile_extraction\chennai_wards_clipped.geojson"

gdf = gpd.read_file(shp_path)

# Ensure Ward_No column exists
possible_cols = ["Ward_No", "ward_no", "wardno", "WARD_NO", "WardID"]
for col in possible_cols:
    if col in gdf.columns:
        gdf.rename(columns={col: "Ward_No"}, inplace=True)
        break

gdf["Ward_No"] = gdf["Ward_No"].astype(int)

# =====================================================
# Merge metrics with polygons
# =====================================================
gdf_join = gdf.merge(df_metrics, on="Ward_No", how="left")

# =====================================================
# Save outputs
# =====================================================
gdf_join.to_file("gw_error_map.geojson", driver="GeoJSON")
gdf_join.to_file("gw_error_map.shp")

print("\nExported:")
print("✔ gw_error_map.geojson (QGIS-ready)")
print("✔ gw_error_map.shp")
print("✔ gw_error_metrics.csv")

print("\nOpen gw_error_map.geojson in QGIS →")
print("   Layer Styling → Graduated → Column = RMSE or R or KGE")
print("   Choose Color Ramp: Reds/Blues")
