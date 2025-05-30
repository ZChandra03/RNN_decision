#!/usr/bin/env python3
"""
Static PCA mean‑trajectory plot.

This script loads a trained RNN and generates **one static Matplotlib plot**
showing the mean trajectory of the hidden state *per (order, comp_dur)*
condition in a 2‑D PCA space (PC‑1 vs PC‑2).  With the defaults below this
creates ≤ 16 coloured curves – one for every combination in
``ORDERS × COMP_DURS`` – and saves the figure as
``pca_mean_trajectories.png`` in the script folder.

Run
    python pca_mean_trajectory.py

Dependencies: torch, numpy, scikit‑learn, matplotlib.
"""

from __future__ import annotations

import os
import json
from collections import defaultdict

import numpy as np
import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Project‑level imports – *must* be importable from the script directory.
# ---------------------------------------------------------------------------
from rnn_model import RNNModel
from generate_trials import generate_case_batch

# ---------------- User‑tweakable params ------------------------------------
START_MS: int | None = -200     # clip window start (ms, relative to int‑2)
END_MS:   int | None = 1500     #          end   (ms, relative to int‑2)
PCA_NCOMP: int = 10             # keep first 10 PCs (plenty for this plot)
OUT_FIG:   str = "pca_mean_trajectories.png"
FIGSIZE = (6, 5)
LINEWIDTH = 1.5
ALPHA = 0.9

# ---------------- Constants -------------------------------------------------
BASE_DIR   = os.path.abspath(os.path.dirname(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "models", "easy_trained")
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 100
COMP_DURS  = [120, 140, 160, 180, 220, 240, 260, 280]
ORDERS     = [1]            # 0 → AB, 1 → BA

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_model(model_dir: str):
    """Load trained RNN *and* its hyper‑parameters from *model_dir*."""
    with open(os.path.join(model_dir, "hp.json"), "r") as f:
        hp = json.load(f)
    model = RNNModel(hp)
    ckpt_path = os.path.join(model_dir, "checkpoint.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt)
    model.to(DEVICE).eval()
    return model, hp


def collect_hidden_states(model: RNNModel, hp: dict[str, float]):
    """Generate all condition batches and return clipped hidden states.

    Returns
    -------
    h_by_cond : dict[tuple[int,int], torch.Tensor]
        Maps (order, comp_dur) → hidden states tensor of shape (trials, L, N).
    times : np.ndarray
        Time vector (ms) aligned so that *t = 0* is int‑2 onset.
    """
    dt = hp["dt"]                       # ms per simulation step
    min_pre, min_post = np.inf, np.inf  # across all trials
    raw_by_cond: dict[tuple[int,int], tuple[torch.Tensor, np.ndarray]] = {}

    # --- 1) Generate once for every requested condition --------------------
    for dur in COMP_DURS:
        for order in ORDERS:
            x, _, int2, _, _, _ = generate_case_batch(hp, dur, order, BATCH_SIZE)
            with torch.no_grad():
                h = model.rnn(x.to(DEVICE)).cpu()   # (B, T, N)
            pre  = int2.numpy()                     # samples before int‑2 (B,)
            post = h.shape[1] - pre                 # samples after     int‑2 (B,)
            min_pre  = min(min_pre,  pre.min())
            min_post = min(min_post, post.min())
            raw_by_cond[(order, dur)] = (h, int2.numpy())

    min_pre, min_post = int(min_pre), int(min_post)

    # --- 2) Clip every trial to the common window -------------------------
    h_by_cond: dict[tuple[int,int], torch.Tensor] = {}
    for key, (h, int2) in raw_by_cond.items():
        clipped = []
        for i in range(h.shape[0]):
            s = int(int2[i]) - min_pre      # inclusive start idx
            e = int(int2[i]) + min_post     # exclusive end idx
            clipped.append(h[i, s:e])       # (L, N)
        h_by_cond[key] = torch.stack(clipped)  # (B, L, N)

    frames = h_by_cond[ORDERS[0], COMP_DURS[0]].shape[1]
    int2_idx = min_pre                       # offset index in window
    times = (np.arange(frames) - int2_idx) * dt  # (L,)

    # Optional external window clip (START_MS / END_MS) ---------------------
    mask = np.ones_like(times, dtype=bool)
    if START_MS is not None:
        mask &= times >= START_MS
    if END_MS is not None:
        mask &= times <= END_MS

    if not mask.all():
        for key in h_by_cond:
            h_by_cond[key] = h_by_cond[key][:, mask]
        times = times[mask]

    return h_by_cond, times


def compute_pca(h_by_cond: dict[tuple[int,int], torch.Tensor]):
    """Fit PCA across *all* trials × time and project every condition.

    Returns a dict keyed like *h_by_cond* but holding numpy arrays of shape
    (trials, L, PCA_NCOMP).
    """
    # Concatenate across conditions for fitting – keeps orientation consistent.
    all_h = torch.cat([h.reshape(-1, h.shape[-1])  # (trials*L, N)
                       for h in h_by_cond.values()], dim=0)
    # z‑score per unit to equalise scale
    X = (all_h - all_h.mean(0)) / all_h.std(0)
    pca = PCA(n_components=PCA_NCOMP, svd_solver="full")
    pca.fit(X.numpy())

    Z_by_cond: dict[tuple[int,int], np.ndarray] = {}
    for key, h in h_by_cond.items():
        h_flat = h.reshape(-1, h.shape[-1])        # (trials*L, N)
        h_flat = (h_flat - all_h.mean(0)) / all_h.std(0)
        z_flat = pca.transform(h_flat.numpy())     # (trials*L, K)
        Z_by_cond[key] = z_flat.reshape(h.shape[0], h.shape[1], PCA_NCOMP)
    return Z_by_cond


def plot_mean_trajectories(Z_by_cond: dict[tuple[int,int], np.ndarray],
                           times: np.ndarray,
                           out_png: str):
    """Plot and save the mean PC‑1 vs PC‑2 trajectory for every condition."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    cmap = plt.get_cmap("tab20")               # up to 20 distinguishable colours

    for idx, ((order, dur), Z) in enumerate(sorted(Z_by_cond.items())):
        mean_traj = Z.mean(0)                  # (L, PCA_NCOMP)
        ax.plot(mean_traj[:, 0], mean_traj[:, 1],
                label=f"order {order} | dur {dur} ms",
                lw=LINEWIDTH, alpha=ALPHA, color=cmap(idx))

    ax.set(
        xlabel="PC‑1",
        ylabel="PC‑2",
        title="Mean hidden‑state trajectories (PC‑1 vs PC‑2)",
    )
    ax.legend(fontsize="small", ncol=2, frameon=False)
    fig.tight_layout()

    out_path = os.path.join(BASE_DIR, out_png)
    fig.savefig(out_path, dpi=300)
    print(f"✓ Saved figure → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    model, hp = load_model(MODEL_DIR)
    h_by_cond, times = collect_hidden_states(model, hp)
    Z_by_cond = compute_pca(h_by_cond)
    plot_mean_trajectories(Z_by_cond, times, OUT_FIG)


if __name__ == "__main__":
    main()
