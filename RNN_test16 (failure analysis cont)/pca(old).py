#!/usr/bin/env python3
"""
pca.py

Revised PCA‑based inspection for TP / FP / FN / TN hidden states.

Key changes:
- **Covariance PCA**: center only (Xc = X - mean), no feature scaling.
- **Print fix**: properly format unique label counts instead of building invalid sets.

Usage:
    python pca.py

Outputs:
    - pca_scatter.png and pca_variance.png in OUTPUT_DIR
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
OUTPUT_DIR = os.path.join(BASE_DIR, "pca_failure_analysis")

# Analysis hyper‑params
BATCH_SIZE = 500
THRESHOLD  = 0.4
PCA_NCOMP  = 10
WINDOW_MS  = 200

os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ────────────────────────────────────────────────────────────────────────────
# Helper functions
# ────────────────────────────────────────────────────────────────────────────
def load_model(model_dir):
    with open(os.path.join(model_dir, "hp.json"), "r") as f:
        hp = json.load(f)
    model = RNNModel(hp)
    ckpt = torch.load(os.path.join(model_dir, "checkpoint.pt"), map_location="cpu")
    model.load_state_dict(ckpt)
    model.to(DEVICE).eval()
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
    model, hp = load_model(MODEL_DIR)
    dt = hp.get("dt", 10)

    # Only focus on durations that show failures
    comp_durs = [180, 220]
    orders = [0, 1]

    # containers
    X_list, labels = [], []

    for d in comp_durs:
        for o in orders:
            x, responds, int2_ons = generate_case_batch(hp, d, o, BATCH_SIZE)
            with torch.no_grad():
                h_seq = model.rnn(x.to(DEVICE))
            # classify
            yhat = model(x.to(DEVICE)).cpu().squeeze(-1)
            fired = yhat > THRESHOLD
            fired_in = (fired & (torch.arange(yhat.size(1))[None, :] >= int2_ons.unsqueeze(1))).any(1)
            resp = responds
            lbl = np.where(
                resp & fired_in, 'TP',
                np.where(~resp & fired.any(1), 'FP',
                         np.where(resp & ~fired_in, 'FN', 'TN'))
            )
            # average hidden state
            h_avg = average_hidden_window(h_seq, int2_ons, dt, WINDOW_MS)
            X_list.append(h_avg)
            labels.append(lbl)

            # fixed printing of label counts
            uniq, counts = np.unique(lbl, return_counts=True)
            counts_str = ", ".join(f"{u}={c}" for u, c in zip(uniq, counts))
            print(f"d={d}, order={o} -> {counts_str}")

    # stack data
    X = np.vstack(X_list)
    y = np.concatenate(labels)

    # Covariance PCA: center only
    Xc = X - X.mean(axis=0)
    pca = PCA(n_components=PCA_NCOMP, svd_solver='full')
    Z = pca.fit_transform(Xc)

    # scatter plot
    colours = {'TP': 'g', 'FP': 'orange', 'FN': 'r', 'TN': 'b'}
    plt.figure(figsize=(6, 5))
    for lab in ["TP", "TN"]:
        sel = (y == lab)
        plt.scatter(Z[sel, 0], Z[sel, 1],
                    s=10, alpha=0.6, c=colours[lab], label=lab)

    for lab in ["FP", "FN"]:
        sel = (y == lab)
        plt.scatter(Z[sel, 0], Z[sel, 1],
                    s=10, alpha=1, c=colours[lab], label=lab)
    plt.xlabel('PC-1')
    plt.ylabel('PC-2')
    plt.title('Hidden-state Covariance-PCA')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'pca_scatter.png'), dpi=300)
    plt.close()

    # cumulative variance plot
    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
    plt.figure(figsize=(4, 3))
    plt.plot(np.arange(1, len(cumvar) + 1), cumvar, 'o-')
    plt.xlabel('PC index j')
    plt.ylabel('Cumulative variance (%)')
    plt.title('Covariance PCA variance')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'pca_variance.png'), dpi=300)
    plt.close()

    print(f"Saved plots to {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
