#!/usr/bin/env python3
# plot_saved_cases.py
# -------------------------------------------------------------------------
# Pick 4 examples from every .pt bundle in failure_conditions/ and
# visualise hidden activity + output, mimicking failure_analysis.py.
# -------------------------------------------------------------------------
import os, random, json, glob
import numpy as np
import torch
import matplotlib.pyplot as plt

from rnn_model import RNNModel                        # only to grab hp.json

# ─── Paths & constants ───────────────────────────────────────────────────
BASE_DIR   = os.path.abspath(os.path.dirname(__file__))
SAVED_DIR  = os.path.join(BASE_DIR, "failure_conditions")
FIG_DIR    = os.path.join(BASE_DIR, "figures")
MODEL_DIR  = os.path.join(BASE_DIR, "models", "easy_trained")

NUM_SAMPLES = 4              # per .pt file
THRESHOLD   = 0.4            # same as before

os.makedirs(FIG_DIR, exist_ok=True)

# ─── Grab dt from model hp (for x-axis in seconds) ───────────────────────
with open(os.path.join(MODEL_DIR, "hp.json"), "r") as f:
    hp = json.load(f)
dt_s = hp.get("dt", 10) / 1000.0        # ms → s

# ─── Helper: establish a common neuron order (first success bundle) ──────
def get_unit_order(h_tensor):           # h_tensor shape (T, N)
    arr = h_tensor.numpy().T            # (N, T)
    peaks = arr.argmax(axis=1)
    return np.argsort(peaks)

unit_order = None   # will be fixed on first success bundle encountered

# ─── Iterate over every saved bundle (*.pt) ──────────────────────────────
pt_files = sorted(glob.glob(os.path.join(SAVED_DIR, "*.pt")))
if not pt_files:
    raise RuntimeError("No .pt bundles found in failure_conditions/")

for fpath in pt_files:
    tag = os.path.basename(fpath).replace(".pt", "")
    data = torch.load(fpath, map_location="cpu")      # dict: x, h, y_hat
    x_all   = data['x']       # (B, T, n_in)  – unused here
    h_all   = data['h']       # (B, T, N)
    y_all   = data['y_hat']   # (B, T)

    B, T, N = h_all.shape
    n = 30
    idxs = [n + 0,n + 1,n + 2,n + 4]

    # lock unit order once (on first *_successes.pt seen)
    if unit_order is None and "success" in tag:
        unit_order = get_unit_order(h_all[idxs[0]])

    if unit_order is None:
        # fall-back: just sort by peak of first sample
        unit_order = get_unit_order(h_all[idxs[0]])

    times = np.arange(T) * dt_s

    # ---- set up figure (2 rows × n columns) ----------------------------
    fig, axes = plt.subplots(2, len(idxs),
                             figsize=(4*len(idxs), 4),
                             gridspec_kw={"height_ratios":[3,1],
                                          "wspace":0.3, "hspace":0.3})
    if len(idxs) == 1:          # matplotlib gives 1-D axes if n=1
        axes = np.array(axes).reshape(2,1)

    for j, idx in enumerate(idxs):
        h_seq = h_all[idx].numpy().T            # (N, T)
        y_hat = y_all[idx].numpy()              # (T,)

        h_sorted = np.clip(h_seq[unit_order], 0, None)

        # heat-map
        im = axes[0,j].imshow(h_sorted, aspect='auto', cmap='viridis',
                              vmin=0, vmax=1,
                              extent=[times[0], times[-1], 0, N])
        axes[0,j].set_title(f"{tag} #{j+1}", fontsize=9)
        axes[0,j].set_xlim(times[[0,-1]])
        if j == 0:
            axes[0,j].set_ylabel("Units")

        # output trace
        axes[1,j].plot(times, y_hat, linewidth=1.5)
        axes[1,j].axhline(THRESHOLD, linestyle='--', linewidth=1)
        axes[1,j].set_ylim([-0.05, 1.05])
        axes[1,j].set_xlim(times[[0,-1]])
        if j == 0:
            axes[1,j].set_ylabel("Output")

    # shared colorbar
    cbar = fig.colorbar(im, ax=axes[0,:].tolist() if len(idxs)>1 else axes[0,0],
                        orientation='vertical', shrink=0.8)
    cbar.set_label("Activity", rotation=270, labelpad=12)

    fig.suptitle(f"Samples from {tag}", fontsize=14)
    fig.tight_layout(rect=[0,0,1,0.95])

    out_png = os.path.join(FIG_DIR, f"{tag}_examples.png")
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"✓ saved {out_png}")
