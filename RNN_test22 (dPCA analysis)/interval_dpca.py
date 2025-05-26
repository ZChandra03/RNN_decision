#!/usr/bin/env python3
"""
interval_dpca.py  ―  Full demixed-PCA analysis for the Interval-Discrim RNN
-----------------------------------------------------------------------------

Outputs
-------
variance.png        – bar plot of variance explained per marginalisation
c_trajectories.png  – trajectories of the first two comparison-duration PCs

Author : ChatGPT (2025-05-25)
"""

import os, json, itertools, warnings
import numpy as np
import torch
import matplotlib.pyplot as plt
from dPCA.dPCA import dPCA

# ---------------------------------------------------------------------------
# Local imports from your codebase
# ---------------------------------------------------------------------------
from rnn_model     import RNNModel                 # :contentReference[oaicite:0]{index=0}
from failure_count import generate_case_batch      # :contentReference[oaicite:1]{index=1}

# ------------------------- USER-TWEAKABLE PARAMS ---------------------------
MODEL_DIR   = os.path.join(os.path.dirname(__file__), "models", "easy_trained")
REPEATS     = 20          # trials per condition
ALIGN_MS    = (-300, 300)  # window around int-2 onset (start, end) in ms
N_DPC       = 8            # dPCs to keep per marginal
REG         = 0       # regulariser (float or 'auto')
SAVE_FIGS   = True
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ---------------------------------------------------------------------------

# Comparison durations = 200 + 20*offsets  (taken from train.get_default_hp)
OFFSETS     = np.array([-2, -1, 1, 2], dtype=int)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_model(model_dir):
    """Load RNN + hyper-parameters (same as in failure_count)."""
    with open(os.path.join(model_dir, "hp.json"), "r") as f:
        hp = json.load(f)
    model = RNNModel(hp)
    ckpt  = torch.load(os.path.join(model_dir, "checkpoint.pt"), map_location="cpu")
    model.load_state_dict(ckpt)
    model.to(DEVICE).eval()
    return model, hp

def align_and_clip(h, int2_on, dt):
    """Return ±window hidden states around int-2 onset."""
    t0  = int(int2_on)                    # sample index of interval-2 onset
    pre = int(abs(ALIGN_MS[0]) / dt)
    post= int(abs(ALIGN_MS[1]) / dt)
    s, e = t0 - pre, t0 + post            # inclusive start, exclusive end
    return h[:, s:e]                      # (neurons, T_window)

# ---------------------------------------------------------------------------
# Build the complete data tensor
# ---------------------------------------------------------------------------
def build_tensor(model, hp):
    dt        = hp["dt"]
    std_dur   = hp["std_dur"]
    comp_step = hp.get("comp_step", 20)

    comp_durs = std_dur + comp_step * OFFSETS      # 8 durations
    orders    = [0, 1]                             # std-first / cmp-first

    # First pass to learn tensor dimensions
    with torch.no_grad():
        # forward one dummy trial to get (#neurons)
        x, _, int2_on, _ = generate_case_batch(hp, comp_durs[0], orders[0], 1)
        N = model.rnn(x.to(DEVICE)).shape[-1]
    T_window = int((ALIGN_MS[1] - ALIGN_MS[0]) / dt)

    # Allocate tensor: (neurons, time, n_c, n_o, repeats)
    X = np.zeros((N, T_window, len(comp_durs), len(orders), REPEATS), dtype=np.float32)

    # Fill tensor
    print("Generating trials …")
    for ic, cd in enumerate(comp_durs):
        for io, od in enumerate(orders):
            # gather hidden states for REPEATS trials
            x, _, int2_on, _ = generate_case_batch(hp, int(cd), od, REPEATS)
            with torch.no_grad():
                h = model.rnn(x.to(DEVICE)).cpu()    # (B, T, N)
            for r in range(REPEATS):
                X[:, :, ic, io, r] = align_and_clip(
                    h[r].T.numpy(), int2_on[r], dt)  # neurons first
    print("✓ Done.  Tensor shape:", X.shape)
    return X, comp_durs, orders

# ---------------------------------------------------------------------------
# dPCA + plotting
# ---------------------------------------------------------------------------
def run_dpca(X):
    """Fit dPCA; X has shape (neurons, time, c, o, r)."""
    N, T, C, O, R = X.shape
    X_flat = X.reshape(N, T, C*O*R)          # ←  keep (neurons, TIME, trials)
    dpca   = dPCA(labels='cto',
                  n_components=N_DPC,
                  regularizer=REG)           # no memory-hungry λ search
    Z      = dpca.fit_transform(X_flat)      #   and no duplicate trialX
    expl   = dpca.explained_variance_ratio_
    return dpca, Z, expl

def plot_variance(expl):
    cols = {'c':'tab:orange', 't':'tab:blue', 'o':'tab:green',
            'ct':'grey', 'co':'grey', 'to':'grey', 'cto':'grey'}
    fig, ax = plt.subplots(figsize=(5,4))
    pos = 0
    for marg in ('c','t','o','ct','co','to','cto'):
        if marg not in expl: continue
        vals = expl[marg] * 100
        ax.bar(np.arange(len(vals))+pos, vals, color=cols.get(marg,'grey'), label=marg)
        pos += len(vals)
    ax.set(xlabel="dPCA components", ylabel="% variance explained")
    ax.legend(frameon=False, ncol=4)
    fig.tight_layout()
    fig.savefig("variance.png", dpi=300)
    plt.close(fig)
    print("✓ Saved variance.png")

def plot_c_traj(Z, comp_durs):
    """First two comparison-duration PCs (order already averaged)."""
    C1 = Z['c'][0]              # (time, n_c[, n_o])
    C2 = Z['c'][1]

    fig, ax = plt.subplots(figsize=(5, 5))

    if C1.ndim == 2:            # (time, n_c)
        for ic, cd in enumerate(comp_durs):
            ax.plot(C1[:, ic], C2[:, ic], label=f"{cd} ms")
    else:                       # (time, n_c, n_o)  – keep order 0
        for ic, cd in enumerate(comp_durs):
            ax.plot(C1[:, ic, 0], C2[:, ic, 0], label=f"{cd} ms")

    ax.set(xlabel="c-PC1", ylabel="c-PC2")
    ax.legend(title="Comparison duration", bbox_to_anchor=(1, 1))
    fig.tight_layout()
    fig.savefig("c_trajectories.png", dpi=300)
    plt.close(fig)
    print("✓ Saved c_trajectories.png")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)   # cleaner logs
    model, hp = load_model(MODEL_DIR)
    X, comp_durs, orders = build_tensor(model, hp)
    dpca, Z, expl = run_dpca(X)

    if SAVE_FIGS:
        plot_variance(expl)
        plot_c_traj(Z, comp_durs)

    print("\nTop variances:")
    for k in ('c','t','o'):
        v = expl.get(k, None)
        if v is not None:
            print(f"{k}: {v[0]*100:5.1f}%  (first component)")

if __name__ == "__main__":
    main()
