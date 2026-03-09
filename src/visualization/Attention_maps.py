# attention_maps.py
"""
Extract and visualize spatial attention from HydroASTGNN.

Outputs:
 - attention_mean.npy      : mean attention matrix (N x N)
 - attention_mean_heatmap.png
 - ward_attention_summary.csv : top-10 influences per ward
 - ward_influence_{WARD}.png  : bar plot for each ward

Usage:
    python attention_maps.py
Requires:
    - best_HydroASTGNN.pth
    - HydroASTGNN_training_v4.py (model class HydroASTGNN)
    - stgnn_prepared/X.npy, Y.npy, meta.json
"""
import numpy as np
import json
import torch
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from HydroASTGNN_training import HydroASTGNN, L, device

OUT_DIR = "attention_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# load meta & tensors
meta = json.load(open("stgnn_prepared/meta.json"))
wards = meta["wards"]
X = np.load("stgnn_prepared/X.npy")  # (N,T,F)
Y = np.load("stgnn_prepared/Y.npy")

N, T, F = X.shape

# load model
model = HydroASTGNN(N, F).to(device)
model.load_state_dict(torch.load("best_HydroASTGNN.pth", map_location=device))
model.eval()

# Collect attention matrices across time windows
atts = []   # will store arrays shape (N,N)

for t in tqdm(range(T - L), desc="Collecting attention"):
    x_slice = X[:, t:t+L, :]
    x_tensor = torch.tensor(x_slice).float().unsqueeze(0).to(device)  # shape (1,N,L,F)
    with torch.no_grad():
        pred, att = model(x_tensor)   # att expected shape (1, N, N) per model impl
        if att is None:
            # if your model does not return attention, try to access via attribute (optional)
            raise RuntimeError("Model did not return attention matrix. Ensure HydroASTGNN returns att.")
        att_np = att.squeeze(0).cpu().numpy()   # N x N
        # ensure rows sum to 1 (softmax) — small numeric fix
        row_sums = att_np.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        att_np = att_np / row_sums
        atts.append(att_np)

atts = np.array(atts)   # (time_windows, N, N)
print("Collected attention shape:", atts.shape)

# Mean attention over time
att_mean = atts.mean(axis=0)   # N x N
np.save(os.path.join(OUT_DIR, "attention_mean.npy"), att_mean)

# Save heatmap
plt.figure(figsize=(10, 9))
vmin = att_mean.min()
vmax = att_mean.max()
plt.imshow(att_mean, cmap="viridis", vmin=vmin, vmax=vmax)
plt.colorbar(label="Attention weight")
plt.title("Mean Spatial Attention (averaged over time windows)")
plt.xlabel("Target Node index")
plt.ylabel("Source Node index")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "attention_mean_heatmap.png"), dpi=300)
plt.close()

# For each ward, find top K influential wards (columns with highest weight toward ward)
K = 10
records = []
for i, ward in enumerate(wards):
    # influence -> which source nodes most influence this target node i?
    # att_mean[:, i] is contribution from each source -> to target i
    influence = att_mean[:, i]  # shape N
    top_idx = np.argsort(influence)[::-1][:K]
    top_wards = [wards[j] for j in top_idx]
    top_vals = influence[top_idx].tolist()

    # Save CSV row
    rec = {"Ward_No": int(ward)}
    for rank, (w_j, v) in enumerate(zip(top_wards, top_vals), start=1):
        rec[f"top{rank}_ward"] = int(w_j)
        rec[f"top{rank}_val"] = float(v)
    records.append(rec)

import pandas as pd
df_top = pd.DataFrame(records)
df_top.to_csv(os.path.join(OUT_DIR, "ward_attention_summary.csv"), index=False)
print("Saved ward_attention_summary.csv")

# Save per-ward bar plots for top influences
for i, ward in enumerate(wards):
    influence = att_mean[:, i]
    top_idx = np.argsort(influence)[::-1][:K]
    top_wards = [wards[j] for j in top_idx]
    top_vals = influence[top_idx]

    plt.figure(figsize=(7, 3.5))
    plt.bar([str(w) for w in top_wards], top_vals)
    plt.title(f"Top {K} influences → Ward {ward}")
    plt.xlabel("Source Ward")
    plt.ylabel("Mean attention weight")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"ward_influence_{ward}.png"), dpi=200)
    plt.close()

print("Saved per-ward influence barplots in:", OUT_DIR)
print("Done. You can join ward_attention_summary.csv with your ward geojson to map top source wards.")
