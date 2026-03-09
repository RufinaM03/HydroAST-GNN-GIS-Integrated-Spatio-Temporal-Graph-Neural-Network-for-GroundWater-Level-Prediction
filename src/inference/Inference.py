# Inference_v4.py
import numpy as np
import torch
import json
from HydroASTGNN_training import HydroASTGNN, L, device

# Load metadata
meta = json.load(open("stgnn_prepared/meta.json"))
wards = meta["wards"]
y_mean = np.array(meta["y_mean_per_ward"])
y_std = np.array(meta["y_std_per_ward"])

# Load data
X = np.load("stgnn_prepared/X.npy")
Y = np.load("stgnn_prepared/Y.npy")

N, T, F = X.shape

# Load model
model = HydroASTGNN(N, F).to(device)
model.load_state_dict(torch.load("best_HydroASTGNN.pth", map_location=device))
model.eval()

# Use last window for forecasting
x_input = X[:, -L:, :]

x_tensor = torch.tensor(x_input).float().unsqueeze(0).to(device)

with torch.no_grad():
    pred_norm, _ = model(x_tensor)
    pred_norm = pred_norm.squeeze(0).cpu().numpy().flatten()

# De-normalize PER WARD
pred_real = pred_norm * y_std + y_mean

print("\n=== Prediction per Ward ===")
for w, p in zip(wards, pred_real):
    print(f"Ward {w}: {p:.3f}")
