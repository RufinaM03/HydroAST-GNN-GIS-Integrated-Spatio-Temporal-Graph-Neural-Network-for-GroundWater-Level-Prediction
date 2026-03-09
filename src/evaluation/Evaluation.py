# evaluation_v4.py
import numpy as np
import json
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from HydroASTGNN_training import HydroASTGNN, L, device

# =====================================================
# Load tensors + meta
# =====================================================
meta = json.load(open("stgnn_prepared/meta.json"))
wards = meta["wards"]

y_mean = np.array(meta["y_mean_per_ward"])
y_std = np.array(meta["y_std_per_ward"])

X = np.load("stgnn_prepared/X.npy")
Y = np.load("stgnn_prepared/Y.npy")

N, T, F = X.shape

# =====================================================
# Load trained model
# =====================================================
model = HydroASTGNN(N, F).to(device)
model.load_state_dict(torch.load("best_HydroASTGNN.pth", map_location=device))
model.eval()

# =====================================================
# Predict all timesteps
# =====================================================
preds = []
trues = []

for t in tqdm(range(T - L), desc="Evaluating"):
    x_slice = X[:, t:t+L, :]
    y_true_norm = Y[:, t+L]

    x_tensor = torch.tensor(x_slice).float().unsqueeze(0).to(device)

    with torch.no_grad():
        pred_norm, _ = model(x_tensor)
        pred_norm = pred_norm.squeeze(0).cpu().numpy()

    # De-normalize PER WARD
    pred_real = pred_norm[:, 0] * y_std + y_mean
    true_real = y_true_norm * y_std + y_mean

    preds.append(pred_real)
    trues.append(true_real)

preds = np.array(preds)   # (T-L, N)
trues = np.array(trues)

# =====================================================
# Hydrological Metrics
# =====================================================
def rmse(y, p):
    return np.sqrt(np.mean((y - p) ** 2))

def mae(y, p):
    return np.mean(np.abs(y - p))

def corr(y, p):
    return np.corrcoef(y.flatten(), p.flatten())[0, 1]

# symmetric MAPE (hydrology recommended)
def smape(y, p):
    return np.mean(2 * np.abs(y - p) / (np.abs(y) + np.abs(p) + 1e-6)) * 100

# Kling-Gupta Efficiency (KGE)
def kge(y, p):
    r = corr(y, p)
    a = (p.mean() / (y.mean() + 1e-6))
    b = (p.std() / (y.std() + 1e-6))
    return 1 - np.sqrt((r-1)**2 + (a-1)**2 + (b-1)**2)

# =====================================================
# Global Metrics
# =====================================================
RMSE = rmse(trues, preds)
MAE = mae(trues, preds)
R = corr(trues, preds)
SMAPE = smape(trues, preds)
KGE = kge(trues, preds)

print("\n================ GLOBAL METRICS ================")
print(f"RMSE : {RMSE:.4f}")
print(f"MAE  : {MAE:.4f}")
print(f"R    : {R:.4f}")
print(f"SMAPE: {SMAPE:.2f} %")
print(f"KGE  : {KGE:.4f}")

# =====================================================
# Per-Ward Metrics
# =====================================================
print("\n================ PER-WARD METRICS ================")
for i, w in enumerate(wards):
    r_w = rmse(trues[:,i], preds[:,i])
    m_w = mae(trues[:,i], preds[:,i])
    r2_w = corr(trues[:,i], preds[:,i])
    print(f"Ward {w}: RMSE={r_w:.3f}, MAE={m_w:.3f}, R={r2_w:.3f}")

# =====================================================
# Plot 1 — Global Average Trend
# =====================================================
plt.figure(figsize=(12, 5))
plt.plot(np.mean(trues, axis=1), label="Actual", linewidth=3)
plt.plot(np.mean(preds, axis=1), label="Predicted", linewidth=3, linestyle="--")
plt.title("Global Groundwater Trend (Average Across All Wards)")
plt.xlabel("Time Index")
plt.ylabel("GW Level")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("plot_global_trend.png", dpi=300)
plt.show()

# =====================================================
# Plot 2 — Scatter Plot: Actual vs Predicted
# =====================================================
plt.figure(figsize=(6, 6))
plt.scatter(trues.flatten(), preds.flatten(), alpha=0.4)
lims = [min(trues.min(), preds.min()), max(trues.max(), preds.max())]
plt.plot(lims, lims, "r--")
plt.xlabel("Actual GW Level")
plt.ylabel("Predicted GW Level")
plt.title("Actual vs Predicted (All Wards)")
plt.grid(True)
plt.tight_layout()
plt.savefig("plot_scatter_actual_vs_pred.png", dpi=300)
plt.show()

# =====================================================
# Plot 3 — Multi-Ward Grid Plot (4×4 panels per image)
# =====================================================
def plot_grid(start=0, per_page=16):
    end = min(start + per_page, N)
    subset = list(range(start, end))

    rows = 4
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(16, 12))

    for idx, ax in zip(subset, axes.flatten()):
        ax.plot(trues[:, idx], label="Actual")
        ax.plot(preds[:, idx], label="Pred", linestyle="--")
        ax.set_title(f"Ward {wards[idx]}")
        ax.grid(True)

    for empty_ax in axes.flatten()[len(subset):]:
        empty_ax.axis("off")

    fig.suptitle(f"Ward Predictions (Wards {start} to {end-1})")
    plt.tight_layout()
    plt.savefig(f"plot_wards_{start}_to_{end}.png", dpi=300)
    plt.show()

# make grid plots for all wards
for page in range(0, N, 16):
    plot_grid(start=page, per_page=16)

np.save("preds_full.npy", preds)
np.save("trues_full.npy", trues)
print("Saved preds_full.npy and trues_full.npy for spatial heatmaps.")

print("\nAll plots saved successfully.")
