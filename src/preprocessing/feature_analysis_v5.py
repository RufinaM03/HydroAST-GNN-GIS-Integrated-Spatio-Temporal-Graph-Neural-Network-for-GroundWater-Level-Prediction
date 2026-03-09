# feature_spatiotemporal_analysis.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import calendar
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr

sns.set(style="whitegrid", context="talk")
plt.rcParams["figure.dpi"] = 120

# ---------- CONFIG ----------
DATA_CSV = "./cleaned_groundwater_dataset-V4.csv"  # change if needed
OUT_DIR = "./feature_spatiotemporal_figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- LOAD ----------
df = pd.read_csv(DATA_CSV)
print(" Loaded:", df.shape)

# Detect key columns
def detect_cols(df):
    date = next((c for c in df.columns if 'date' in c.lower() or 'period' in c.lower()), None)
    gw = next((c for c in df.columns if 'gw' in c.lower() or 'lvl' in c.lower()), None)
    lat = next((c for c in df.columns if 'lat' in c.lower()), None)
    lon = next((c for c in df.columns if 'lon' in c.lower()), None)
    ward = next((c for c in df.columns if 'ward' in c.lower() or 'zone' in c.lower() or 'place' in c.lower()), None)
    return date, gw, lat, lon, ward

DATE_COL, GW_COL, LAT_COL, LON_COL, WARD_COL = detect_cols(df)
print("Detected:", DATE_COL, GW_COL, LAT_COL, LON_COL, WARD_COL)

df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
df["year"] = df[DATE_COL].dt.year
df["month"] = df[DATE_COL].dt.month
df["month_name"] = df[DATE_COL].dt.month.apply(lambda m: calendar.month_abbr[m] if pd.notnull(m) else m)

# ---------- FEATURE SELECTION ----------
num_cols = df.select_dtypes(include=np.number).columns.tolist()
# exclude gw_level, coordinates
feature_cols = [c for c in num_cols if c not in [GW_COL, LAT_COL, LON_COL]]

print("Analyzing features:", feature_cols)

# ---------- TEMPORAL EVOLUTION BY WARD ----------
for feat in feature_cols:
    ward_avg = df.groupby([WARD_COL, "year"])[feat].mean().reset_index()
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=ward_avg, x="year", y=feat, hue=WARD_COL, marker="o")
    plt.title(f"{feat}: Yearly Trend by Ward")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{feat}_yearly_by_ward.png"))
    plt.close()

# ---------- MONTHLY HEATMAPS (by year) ----------
for feat in feature_cols:
    pivot = df.pivot_table(index="month_name", columns="year", values=feat, aggfunc="mean")
    pivot = pivot.reindex(index=[calendar.month_abbr[m] for m in range(1,13)])
    plt.figure(figsize=(10,6))
    sns.heatmap(pivot, cmap="coolwarm", annot=False)
    plt.title(f"Monthly average of {feat} across years")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{feat}_monthly_heatmap.png"))
    plt.close()

# ---------- SPATIAL VARIATION ----------
# Mean per ward across all time
spatial_df = df.groupby(WARD_COL).agg({
    LAT_COL: "mean",
    LON_COL: "mean",
    **{feat: "mean" for feat in feature_cols}
}).reset_index()

for feat in feature_cols:
    plt.figure(figsize=(8,6))
    sc = plt.scatter(spatial_df[LON_COL], spatial_df[LAT_COL], c=spatial_df[feat], cmap="viridis", s=120, edgecolor="k")
    plt.colorbar(sc, label=f"Mean {feat}")
    plt.title(f"Spatial distribution of {feat}")
    plt.xlabel("Longitude"); plt.ylabel("Latitude")
    for _, r in spatial_df.iterrows():
        plt.text(r[LON_COL]+0.005, r[LAT_COL], str(r[WARD_COL]), fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{feat}_spatial_map.png"))
    plt.close()

# ---------- SPATIAL CORRELATION BETWEEN WARDS ----------
coords = spatial_df[[LAT_COL, LON_COL]].values
dist_matrix = cdist(coords, coords, metric="euclidean")
dist_matrix = dist_matrix / dist_matrix.max()  # normalize 0–1

for feat in feature_cols:
    vals = spatial_df[feat].values.reshape(-1,1)
    corr_matrix = 1 - np.abs(vals - vals.T) / (np.nanmax(vals) - np.nanmin(vals) + 1e-9)
    combined = corr_matrix - dist_matrix  # high = similar & close
    plt.figure(figsize=(8,6))
    sns.heatmap(combined, cmap="RdYlGn", xticklabels=spatial_df[WARD_COL], yticklabels=spatial_df[WARD_COL])
    plt.title(f"Spatial similarity (value–distance) for {feat}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{feat}_spatial_similarity_heatmap.png"))
    plt.close()

# ---------- MONTHLY BOX PLOTS BY YEAR ----------
for feat in feature_cols:
    plt.figure(figsize=(12,5))
    sns.boxplot(x="month_name", y=feat, hue="year", data=df, showfliers=False)
    plt.title(f"Distribution of {feat} by Month across Years")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{feat}_month_year_boxplot.png"))
    plt.close()

# ---------- FEATURE INTERDEPENDENCE ----------
corr = df[feature_cols].corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr, cmap="RdBu_r", center=0, annot=True, fmt=".2f")
plt.title("Feature-to-feature correlation")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "feature_correlation_matrix.png"))
plt.close()

print(" All feature-based spatiotemporal visualizations saved to", OUT_DIR)
