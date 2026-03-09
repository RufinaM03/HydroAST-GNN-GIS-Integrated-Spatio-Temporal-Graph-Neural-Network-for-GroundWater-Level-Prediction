# HydroASTGNN_training_v4.py
import os
import json
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ---------------- CONFIG ----------------
DATA_DIR = "./stgnn_prepared"
X_NPY = f"{DATA_DIR}/X.npy"
Y_NPY = f"{DATA_DIR}/Y.npy"
META_JSON = f"{DATA_DIR}/meta.json"

MODEL_BEST = "best_HydroASTGNN.pth"
MODEL_FINAL = "final_HydroASTGNN.pth"
LOG_CSV = "training_log.csv"

# Model hyperparams
L = 12
H = 1
d_model = 64
nhead = 4
num_layers = 2
drop = 0.1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training params
BATCH_SIZE = 4
LR = 3e-4
EPOCHS = 200
PATIENCE = 20
MIN_EPOCHS = 20
GRAD_CLIP = 1.0

# -----------------------------------------------------
# Dataset
# -----------------------------------------------------
class STDataset(Dataset):
    def __init__(self, X, Y, L=12, H=1):
        self.X = X
        self.Y = Y
        self.N, self.T, self.F = X.shape
        self.L = L
        self.H = H
        self.indices = [t for t in range(self.T - L - H)]
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        t = self.indices[idx]
        x = self.X[:, t:t+self.L, :]
        y = self.Y[:, t+self.L:t+self.L+self.H]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# -----------------------------------------------------
# Model Components
# -----------------------------------------------------
class SpatialAttention(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.Wq = nn.Linear(in_dim, out_dim)
        self.Wk = nn.Linear(in_dim, out_dim)
        self.Wv = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        # x: B,N,d
        B, N, D = x.shape
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        A = torch.softmax(torch.matmul(Q, K.transpose(1, 2)) / math.sqrt(D), dim=-1)
        out = torch.matmul(A, V)
        return out, A

class TemporalBlock(nn.Module):
    def __init__(self, d_model, nhead, num_layers, drop, input_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=drop, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x):
        # x: B,N,L,F
        B, N, L, F = x.shape
        x = x.reshape(B * N, L, F)
        h = self.proj(x)
        h = self.encoder(h)
        h = h.mean(dim=1)
        return h.reshape(B, N, -1)

class HydroASTGNN(nn.Module):
    def __init__(self, N, F, d_model=64, nhead=4, num_layers=2, drop=0.1):
        super().__init__()
        self.temporal = TemporalBlock(d_model, nhead, num_layers, drop, F)
        self.spatial = SpatialAttention(d_model, d_model)

        self.readout = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, 1)
        )

    def forward(self, x):
        # x: B,N,L,F
        t_out = self.temporal(x)
        s_out, A = self.spatial(t_out)
        fused = t_out + s_out
        y = self.readout(fused)  # B,N,1
        return y, A

# -----------------------------------------------------
# Training function
# -----------------------------------------------------
def train():
    # Load tensors
    X = np.load(X_NPY)
    Y = np.load(Y_NPY)

    N, T, F = X.shape
    ds = STDataset(X, Y, L=L, H=H)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    model = HydroASTGNN(N, F, d_model, nhead, num_layers, drop).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    best_loss = float("inf")
    patience_counter = 0

    for ep in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        count = 0

        # -------------------------
        # Progress bar per epoch
        # -------------------------
        pbar = tqdm(dl, desc=f"Epoch {ep}/{EPOCHS}", leave=False)

        for xb, yb in pbar:
            xb = xb.to(device)
            yb = yb.to(device)

            pred, _ = model(xb)
            loss = loss_fn(pred, yb)

            opt.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()

            total_loss += loss.item()
            count += 1

            # Update progress bar details
            pbar.set_postfix({
                "batch_loss": f"{loss.item():.5f}",
                "grad_norm": f"{grad_norm:.3f}",
                "mean_pred": f"{pred.mean().item():.3f}",
                "mean_y": f"{yb.mean().item():.3f}"
            })

        avg_loss = total_loss / count
        lr = opt.param_groups[0]['lr']

        print(f"Epoch {ep}/{EPOCHS} | "
              f"Loss={avg_loss:.6f} | "
              f"LR={lr:.2e} | "
              f"Best={best_loss:.6f}")

        # -------------------------
        # Early stopping + checkpoint
        # -------------------------
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_BEST)
        else:
            patience_counter += 1

        if ep > MIN_EPOCHS and patience_counter >= PATIENCE:
            print("Early stopping!")
            break

    torch.save(model.state_dict(), MODEL_FINAL)
    print("Training complete.")

if __name__ == "__main__":
    train()