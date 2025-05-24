#!/usr/bin/env python3
"""
PCA‑based inspection of TP / FP / FN / TN hidden states for the
interval‑discrimination RNN.

Usage
-----
python pca_failure_analysis.py            # generates PNG in OUTPUT_DIR

The script:
1. Loads the trained model (see MODEL_DIR below).
2. Builds 16 blocks of trials (8 comparison durations × 2 orders) using the
   same helper functions you used in *failure_count.py*.
3. Runs the network, classifies each trial as TP / FP / FN / TN, and extracts
   the mean hidden state within a 200‑ms window starting at the onset of the
   second interval.
4. Fits PCA (scikit‑learn) to those per‑trial vectors.
5. Saves a scatter plot of PC‑1 vs PC‑2 coloured by the four outcome classes,
   plus a bar chart of cumulative explained variance.

Edit the constants below if your folder layout or hyper‑parameters differ.
"""

# ────────────────────────────────────────────────────────────────────────────
# Imports & configuration
# ────────────────────────────────────────────────────────────────────────────
import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from rnn_model import RNNModel
from task import Trial   # only needed for generating bespoke batches
from failure_count import generate_case_batch   # re‑use existing util

# Location of the trained model folder (hp.json + checkpoint.pt)
BASE_DIR  = os.path.abspath(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models", "easy_trained")
OUTPUT_DIR = os.path.join(BASE_DIR, "pca_failure_analysis")

# Analysis hyper‑params
BATCH_SIZE = 1000          # trials per (duration, order) case
THRESHOLD  = 0.4          # decision threshold on network output
PCA_NCOMP  = 10           # number of PCs to retain
WINDOW_MS  = 200          # time window for hidden‑state averaging

# Make sure we have an output folder and CUDA if available
os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ────────────────────────────────────────────────────────────────────────────
# Helper functions
# ────────────────────────────────────────────────────────────────────────────

def load_model(model_dir):
    """Load model and its hp.json, move to eval mode on DEVICE."""
    with open(os.path.join(model_dir, "hp.json"), "r") as f:
        hp = json.load(f)
    model = RNNModel(hp)
    ckpt = torch.load(os.path.join(model_dir, "checkpoint.pt"), map_location="cpu")
    model.load_state_dict(ckpt)
    model.to(DEVICE).eval()
    return model, hp


def classify_labels(model, x, responds, int2_ons, threshold):
    """Return network output probabilities *and* a label string per trial."""
    with torch.no_grad():
        y_hat = model(x.to(DEVICE)).cpu().squeeze(-1)  # (B, T)

    fired = y_hat > threshold
    B, T = fired.shape
    t_idx = torch.arange(T).unsqueeze(0)
    epoch_mask = t_idx >= int2_ons.unsqueeze(1)        # after 2nd interval onset

    fired_in  = (fired & epoch_mask).any(dim=1)
    fired_out = (fired & ~epoch_mask).any(dim=1)
    fired_any = fired.any(dim=1)

    resp    = responds
    nonresp = ~resp

    labels = np.empty(B, dtype="<U2")
    labels[(resp & fired_in & ~fired_out).numpy()] = "TP"
    labels[(nonresp & fired_any).numpy()]          = "FP"
    labels[(resp & (~fired_in | fired_out)).numpy()] = "FN"
    labels[(nonresp & ~fired_any).numpy()]         = "TN"
    return y_hat, labels


def average_hidden_window(h_seq, int2_ons, dt, window_ms=200):
    """Return per‑trial averaged hidden state in window [int2_on, int2_on+window)."""
    B, T, N = h_seq.shape
    w_steps = int(window_ms / dt)
    h_avg = torch.empty(B, N, device=h_seq.device)
    for i in range(B):
        start = int2_ons[i].item()
        end = min(start + w_steps, T)
        h_avg[i] = h_seq[i, start:end].mean(dim=0)
    return h_avg.cpu().numpy()   # (B, N)

# ────────────────────────────────────────────────────────────────────────────
# Main analysis routine
# ────────────────────────────────────────────────────────────────────────────

def main():
    model, hp = load_model(MODEL_DIR)

    # # --- derive the 8 comparison durations (same logic as failure_count) ----
    # if "comp_step" in hp:
    #     offsets   = [-4, -3, -2, -1, 1, 2, 3, 4]
    #     comp_durs = [hp["std_dur"] + o * hp["comp_step"] for o in offsets]
    # else:
    #     comp_durs = [120, 140, 160, 180, 220, 240, 260, 280]  # fallback
    comp_durs = [180,220]

    orders = [0, 1]   # 0: standard first, 1: comparison first (or vice versa per your code)

    # containers
    X_list, label_list = [], []

    # iterate over all 16 (duration, order) cases
    for d in comp_durs:
        for o in orders:
            # generate deterministic batch with given params
            x, responds, int2_ons = generate_case_batch(hp, d, o, BATCH_SIZE)
            # run RNN to get hidden sequence
            with torch.no_grad():
                h_seq = model.rnn(x.to(DEVICE))          # (B, T, N)
            # classify trial outcomes
            _yhat, labels = classify_labels(model, x, responds, int2_ons, THRESHOLD)
            # average hidden state in task‑critical window
            h_avg = average_hidden_window(h_seq, int2_ons, hp["dt"], WINDOW_MS)
            # accumulate
            X_list.append(h_avg)
            label_list.append(labels)
            # console summary
            uniq, cnt = np.unique(labels, return_counts=True)
            print(f"d={d}, order={o}  :  " + ", ".join(f"{u}={c}" for u, c in zip(uniq, cnt)))

    X = np.vstack(X_list)             # shape (n_trials_total, n_units)
    y = np.concatenate(label_list)    # shape (n_trials_total,)

    # --- PCA ---------------------------------------------------------------
    # (z‑scoring per unit helps when variances differ greatly)
    X_std = (X - X.mean(0)) / X.std(0)
    pca = PCA(n_components=PCA_NCOMP, svd_solver="full")
    Z = pca.fit_transform(X_std)

    # save PC scatter
    colours = {"TP": "tab:green", "FP": "tab:orange", "FN": "tab:red", "TN": "tab:blue"}
    plt.figure(figsize=(6, 5))
    for lab in ["TP", "TN"]:
        sel = (y == lab)
        plt.scatter(Z[sel, 0], Z[sel, 1],
                    s=10, alpha=0.6, c=colours[lab], label=lab)

    for lab in ["FP", "FN"]:
        sel = (y == lab)
        plt.scatter(Z[sel, 0], Z[sel, 1],
                    s=10, alpha=1, c=colours[lab], label=lab)
    plt.xlabel("PC‑1")
    plt.ylabel("PC‑2")
    plt.title("Hidden‑state PCA (window = %d ms)" % WINDOW_MS)
    plt.legend(markerscale=1.5, frameon=False)
    plt.tight_layout()
    scatter_path = os.path.join(OUTPUT_DIR, "pca_scatter.png")
    plt.savefig(scatter_path, dpi=300)
    plt.close()

    # explained variance plot
    plt.figure(figsize=(4, 3))
    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
    plt.bar(range(1, len(cumvar) + 1), cumvar, width=0.7)
    plt.hlines(80, 0.5, len(cumvar) + 0.5, linestyles="dashed", linewidth=1)
    plt.xlabel("Principal component")
    plt.ylabel("Cumulative variance (%)")
    plt.title("PCA variance captured")
    plt.tight_layout()
    var_path = os.path.join(OUTPUT_DIR, "pca_variance.png")
    plt.savefig(var_path, dpi=300)
    plt.close()

    print(f"✓ Saved scatter to {scatter_path}\n✓ Saved variance plot to {var_path}")


if __name__ == "__main__":
    main()
