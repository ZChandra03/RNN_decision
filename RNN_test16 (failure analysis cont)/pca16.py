#!/usr/bin/env python3
"""
pca.py

Generate two PCA scatter plots:
1) TP vs FN (hits vs misses)
2) TN vs FP (correct rejections vs false alarms)

Usage:
    python pca.py

Outputs:
    - pca_tp_fn.png and pca_tn_fp.png in OUTPUT_DIR
    - pca_variance.png in OUTPUT_DIR
"""
import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from rnn_model import RNNModel
from failure_count import generate_case_batch

# Configuration
BASE_DIR   = os.path.abspath(os.path.dirname(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "models", "easy_trained")
OUTPUT_DIR = os.path.join(BASE_DIR, "pca_failure_analysis")

BATCH_SIZE = 5000
THRESHOLD  = 0.4
PCA_NCOMP  = 10
WINDOW_MS  = 200
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
def load_model(model_dir):
    with open(os.path.join(model_dir, "hp.json"), "r") as f:
        hp = json.load(f)
    model = RNNModel(hp)
    ckpt = torch.load(os.path.join(model_dir, "checkpoint.pt"), map_location="cpu")
    model.load_state_dict(ckpt)
    model.to(DEVICE).eval()
    return model, hp

# Average hidden states after int2 onset
def average_hidden_window(h_seq, int2_ons, dt, window_ms):
    B, T, N = h_seq.shape
    steps = int(window_ms / dt)
    h_avg = torch.zeros(B, N, device=h_seq.device)
    for i in range(B):
        start = int2_ons[i].item()
        end = min(start + steps, T)
        h_avg[i] = h_seq[i, start:end].mean(dim=0)
    return h_avg.cpu().numpy()

# Main
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model, hp = load_model(MODEL_DIR)
    dt = hp.get("dt", 10)

    # Focus on durations with failures
    comp_durs = [180, 220]
    orders    = [0, 1]
    X_list, labels = [], []

    for d in comp_durs:
        for o in orders:
            x, responds, int2_ons = generate_case_batch(hp, d, o, BATCH_SIZE)
            with torch.no_grad():
                h_seq = model.rnn(x.to(DEVICE))
            # classify
            yhat = model(x.to(DEVICE)).cpu().squeeze(-1)
            fired = yhat > THRESHOLD
            fired_in = (fired & (torch.arange(yhat.size(1))[None,:] >= int2_ons.unsqueeze(1))).any(1)
            resp = responds
            lbl = np.where(
                resp & fired_in, 'TP',
                np.where(~resp & fired.any(1), 'FP',
                         np.where(resp & ~fired_in, 'FN', 'TN'))
            )
            h_avg = average_hidden_window(h_seq, int2_ons, dt, WINDOW_MS)
            X_list.append(h_avg)
            labels.append(lbl)

    # Prepare data
    X = np.vstack(X_list)
    y = np.concatenate(labels)

    # Covariance PCA
    Xc  = X - X.mean(axis=0)
    pca = PCA(n_components=PCA_NCOMP, svd_solver='full')
    Z   = pca.fit_transform(Xc)

    # Plot 1: TP vs FN
    plt.figure(figsize=(6,5))
    for lab, col in [('TP','g'), ('FN','r')]:
        sel = (y == lab)
        plt.scatter(Z[sel,0], Z[sel,1], s=12, alpha=0.7, c=col, label=lab)
    plt.xlabel('PC-1'); plt.ylabel('PC-2')
    plt.title('PCA: Hits (TP) vs Misses (FN)')
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'pca_tp_fn.png'), dpi=300)
    plt.close()

    # Plot 2: TN vs FP
    plt.figure(figsize=(6,5))
    for lab, col in [('TN','b'), ('FP','orange')]:
        sel = (y == lab)
        plt.scatter(Z[sel,0], Z[sel,1], s=12, alpha=0.7, c=col, label=lab)
    plt.xlabel('PC-1'); plt.ylabel('PC-2')
    plt.title('PCA: Correct Rejections (TN) vs False Alarms (FP)')
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'pca_tn_fp.png'), dpi=300)
    plt.close()

    # Variance plot
    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
    plt.figure(figsize=(4,3))
    plt.plot(np.arange(1,len(cumvar)+1), cumvar, 'o-')
    plt.xlabel('PC index j'); plt.ylabel('Cumulative variance (%)')
    plt.title('Covariance PCA variance')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'pca_variance.png'), dpi=300)
    plt.close()

    print(f"Saved plots to {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
