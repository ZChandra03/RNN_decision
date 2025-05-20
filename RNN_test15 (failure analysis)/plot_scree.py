#!/usr/bin/env python3
"""
scree_failure_analysis.py

Automatically load the trained model, generate trials, extract hidden-state averages,
perform full PCA, and plot a scree plot (\u03bb_j vs j).

Usage:
    python scree_failure_analysis.py

Outputs:
    - scree_plot.png in OUTPUT_DIR
"""

import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from rnn_model import RNNModel
from failure_count import generate_case_batch

# ────────────────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.abspath(os.path.dirname(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "models", "easy_trained")
OUTPUT_DIR = os.path.join(BASE_DIR, "scree_failure_analysis")

# Analysis parameters
BATCH_SIZE = 400    # trials per duration/order block
WINDOW_MS  = 200    # averaging window after 2nd interval offset
PCA_MAX    = None   # keep all components

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def load_model(model_dir):
    with open(os.path.join(model_dir, "hp.json"), "r") as f:
        hp = json.load(f)
    model = RNNModel(hp)
    ckpt = torch.load(os.path.join(model_dir, "checkpoint.pt"), map_location="cpu")
    model.load_state_dict(ckpt)
    model.to(device).eval()
    return model, hp


def average_hidden_window(h_seq, int2_ons, dt, window_ms):
    B, T, N = h_seq.shape
    w_steps = int(window_ms / dt)
    h_avg = torch.zeros(B, N, device=h_seq.device)
    for i in range(B):
        start = int2_ons[i].item()
        end = min(start + w_steps, T)
        h_avg[i] = h_seq[i, start:end].mean(dim=0)
    return h_avg.cpu().numpy()

# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model, hp = load_model(MODEL_DIR)
    dt = hp.get("dt", 10)

    # derive comparison durations
    if "comp_step" in hp:
        offsets = [-4, -3, -2, -1, 1, 2, 3, 4]
        comp_durs = [hp["std_dur"] + o * hp["comp_step"] for o in offsets]
    else:
        comp_durs = [120, 140, 160, 180, 220, 240, 260, 280]
    orders = [0, 1]

    # collect all hidden-state averages
    X_list = []
    for d in comp_durs:
        for o in orders:
            x, responds, int2_ons = generate_case_batch(hp, d, o, BATCH_SIZE)
            with torch.no_grad():
                h_seq = model.rnn(x.to(device))
            h_avg = average_hidden_window(h_seq, int2_ons, dt, WINDOW_MS)
            X_list.append(h_avg)
            print(f"Collected hidden averages for d={d}, order={o}")

    X = np.vstack(X_list)  # shape (n_trials, n_units)
    # standardize per feature
    Xc = X - X.mean(axis=0)

    # PCA
    pca = PCA(n_components=PCA_MAX, svd_solver='full')
    pca.fit(Xc)
    eigvals = pca.explained_variance_

    n = 30
    # Scree plot
    plt.figure(figsize=(6,4))
    j = np.arange(1, len(eigvals)+1)
    plt.plot(j[:n], eigvals[:n], 'o-', markersize=5)
    print(eigvals[:n])
    plt.xlabel('Principal component index $j$')
    plt.ylabel(r'Eigenvalue $\lambda_j$')
    plt.title('PCA Scree Plot')
    plt.grid(True)
    plt.tight_layout()
    out_file = os.path.join(OUTPUT_DIR, 'scree_plot.png')
    plt.savefig(out_file, dpi=300)
    plt.close()

    print(f"✓ Scree plot saved to {out_file}")


if __name__ == '__main__':
    main()
