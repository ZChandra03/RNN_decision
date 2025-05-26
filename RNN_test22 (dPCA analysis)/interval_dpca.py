#!/usr/bin/env python3
"""
interval_dpca.py  ―  Full demixed‑PCA analysis for the Interval‑Discrim RNN
-----------------------------------------------------------------------------

Outputs
-------
variance.png        – bar plot of variance explained per marginalisation
c_trajectories.png  – trajectories of the first two comparison‑duration PCs

"""

import os, json, warnings
import numpy as np
import torch
import matplotlib.pyplot as plt
from dPCA.dPCA import dPCA

# ---------------------------------------------------------------------------
# Local imports from your codebase
# ---------------------------------------------------------------------------
from rnn_model     import RNNModel            # local file
from failure_count import generate_case_batch # local file

# ------------------------- USER‑TWEAKABLE PARAMS ---------------------------
# Directory containing this script (for saving figures)
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_DIR   = os.path.join(SCRIPT_DIR, "models", "easy_trained")
REPEATS     = 10             # trials per condition
ALIGN_MS    = (-300, 300)    # window around int‑2 onset (start, end) in ms
N_DPC       = 8              # dPCs to keep per (joined) marginal
REG         = 0              # Tikhonov regulariser; 0 means none
SAVE_FIGS   = True
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ---------------------------------------------------------------------------

# Comparison durations = 200 + 20 * OFFSETS  (subset for quick test)
OFFSETS     = np.array([-2, -1, 1, 2], dtype=int)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model(model_dir):
    """Load RNN + hyper‑parameters (same as in failure_count)."""
    with open(os.path.join(model_dir, "hp.json"), "r") as f:
        hp = json.load(f)
    model = RNNModel(hp)
    ckpt  = torch.load(os.path.join(model_dir, "checkpoint.pt"), map_location="cpu")
    model.load_state_dict(ckpt)
    model.to(DEVICE).eval()
    return model, hp


def align_and_clip(h, int2_on, dt):
    """Return ±window hidden states around int‑2 onset."""
    t0   = int(int2_on)                     # sample index of interval‑2 onset
    pre  = int(abs(ALIGN_MS[0]) / dt)
    post = int(abs(ALIGN_MS[1]) / dt)
    s, e = t0 - pre, t0 + post             # inclusive start, exclusive end
    return h[:, s:e]                       # (neurons, T_window)

# ---------------------------------------------------------------------------
# Build the complete data tensor
# ---------------------------------------------------------------------------

def build_tensor(model, hp):
    dt        = hp["dt"]
    std_dur   = hp["std_dur"]
    comp_step = hp.get("comp_step", 20)

    comp_durs = std_dur + comp_step * OFFSETS      # 4 durations for quick run
    orders    = [0, 1]                             # std‑first / cmp‑first

    # First pass to learn tensor dimensions
    with torch.no_grad():
        x, _, int2_on, _ = generate_case_batch(hp, comp_durs[0], orders[0], 1)
        N = model.rnn(x.to(DEVICE)).shape[-1]
    T_window = int((ALIGN_MS[1] - ALIGN_MS[0]) / dt)

    # Allocate tensor: (neurons, time, n_c, n_o, repeats)
    X = np.zeros((N, T_window, len(comp_durs), len(orders), REPEATS), dtype=np.float32)

    print("Generating trials …")
    for ic, cd in enumerate(comp_durs):
        for io, od in enumerate(orders):
            x, _, int2_on, _ = generate_case_batch(hp, int(cd), od, REPEATS)
            with torch.no_grad():
                h = model.rnn(x.to(DEVICE)).cpu()    # (B, T, N)
            for r in range(REPEATS):
                X[:, :, ic, io, r] = align_and_clip(h[r].T.numpy(), int2_on[r], dt)
    print("✓ Done. Tensor shape:", X.shape)
    return X, comp_durs

# ---------------------------------------------------------------------------
# dPCA + plotting
# ---------------------------------------------------------------------------

def run_dpca(X):
    """Fit dPCA with joined marginal groups; X shape (neurons, time, c, o, r)."""
    N, T, C, O, R = X.shape
    X_flat = X.reshape(N, T, C * O * R)  # (neurons, time, trials)

    join_dict = {
        'time'       : ('t',),           # condition‑independent dynamics
        'comp'       : ('c', 'ct'),      # all comparison‑duration axes
        'order'      : ('o', 'to'),      # order‑related axes
        'interaction': ('co', 'cto'),    # higher‑order mixture
    }

    dpca = dPCA(labels='cto', join=join_dict, n_components=N_DPC, regularizer=REG)
    Z    = dpca.fit_transform(X_flat)
    expl = dpca.explained_variance_ratio_
    return dpca, Z, expl


def plot_variance(expl):
    """Bar plot of % variance per joined group as well as raw marginals."""
    cols = {
        'time':'tab:purple', 'comp':'tab:orange', 'order':'tab:green', 'interaction':'tab:red',
        'c':'tab:orange', 't':'tab:blue', 'o':'tab:green',
        'ct':'grey', 'co':'grey', 'to':'grey', 'cto':'grey'}

    fig, ax = plt.subplots(figsize=(6,4))
    pos = 0
    for marg in ('time','comp','order','interaction','c','t','o','ct','co','to','cto'):
        if marg not in expl: continue
        vals = expl[marg] * 100
        ax.bar(np.arange(len(vals))+pos, vals, color=cols.get(marg,'grey'), label=marg)
        pos += len(vals)
    ax.set(xlabel="dPCA components", ylabel="% variance explained")
    ax.legend(frameon=False, ncol=4, fontsize=8)
    fig.tight_layout()

    out_file = os.path.join(SCRIPT_DIR, "variance.png")
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"✓ Saved {out_file}")


def plot_c_traj(Z, comp_durs):
    """Low‑D trajectories in the first two comparison‑duration components."""
    C1, C2 = Z['comp'][0], Z['comp'][1]   # (time, n_c)

    fig, ax = plt.subplots(figsize=(5,5))
    for ic, cd in enumerate(comp_durs):
        ax.plot(C1[:, ic], C2[:, ic], label=f"{cd} ms")
    ax.set(xlabel="comp‑PC1", ylabel="comp‑PC2")
    ax.legend(title="Comparison duration", bbox_to_anchor=(1,1))
    fig.tight_layout()

    out_file = os.path.join(SCRIPT_DIR, "c_trajectories.png")
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"✓ Saved {out_file}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)   # cleaner logs
    model, hp = load_model(MODEL_DIR)
    X, comp_durs = build_tensor(model, hp)
    dpca, Z, expl = run_dpca(X)

    if SAVE_FIGS:
        plot_variance(expl)
        plot_c_traj(Z, comp_durs)

    print("\nFirst‑component variance by *raw* marginal:")
    for k in ('c','t','o'):
        if k in expl:
            print(f"{k:2s}: {expl[k][0]*100:5.1f}%")

    print("\nFirst‑component variance by *joined* group:")
    for k in ('time','comp','order','interaction'):
        if k in expl:
            print(f"{k:11s}: {expl[k][0]*100:5.1f}%")

if __name__ == "__main__":
    main()
