"""
Demixed‑PCA (dPCA) pipeline for the Interval Discrimination RNN
===============================================================

* Mirrors the loading logic of **pca_animation_pointwise.py**
* Provides a tidy command‑line‑free interface (edit the **User‑tweakable
  params** block below).
* Supplies **trial‑by‑trial** data so the package can optimise its
  regularisation parameter (`regularizer='auto'`) without error.

-------------------------------------------
Usage
-------------------------------------------
```
$ python dpca_analysis.py   # edit params in‑file first
```
The script prints a variance breakdown and pops up a quick diagnostic plot of
how the comparison trajectories separate in the first dPCA component.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
from dPCA.dPCA import dPCA

# ──────────────────────────────────────────────────────────────────────────────
# User‑tweakable params
# ──────────────────────────────────────────────────────────────────────────────
ALIGN_EVENT: Literal[
    "int1_on", "int1_off", "int2_on", "int2_off"
] = "int1_on"               # alignment reference

COMP_DURS = (180, 220)       # comparison intervals to test (ms)
N_TRIALS  = 40               # per (comp, order) condition

BASE_DIR   = Path(__file__).resolve().parent
MODEL_DIR  = BASE_DIR / 'models' / 'easy_trained'
HP_FILE   = Path("models/easy_trained/hp.json")

# ──────────────────────────────────────────────────────────────────────────────
# Load helpers from repo (pca_animation_pointwise already placed them on path)
# ──────────────────────────────────────────────────────────────────────────────
from task import generate_trials
from rnn_model import RNNModel

# -----------------------------------------------------------------------------
# Dataset builder
# -----------------------------------------------------------------------------

def build_dataset(align_event: str) -> tuple[np.ndarray, np.ndarray]:
    """Return *(X, trialX)* for dPCA.

    X      : (N, T, 2, 2)   — trial‑averaged data
    trialX : (N, T, 2, 2, trials) — single‑trial data
    where axes are (neurons, time, comp, order [, trials]).
    """
    hp = json.loads(HP_FILE.read_text())
    model = RNNModel(hp)
    model.load_state_dict(torch.load(MODEL_DIR, map_location="cpu"))
    model.eval()

    orders = (0, 1)  # std‑first / std‑second

    hidden = np.empty((len(COMP_DURS), len(orders), N_TRIALS,  # comp, order, trials
                       None, None), dtype=np.float32)         # placeholder, we will fill later

    for i_c, comp in enumerate(COMP_DURS):
        for i_o, order in enumerate(orders):
            # Build a batch of trials for this condition
            batch_hp         = hp.copy()
            batch_hp.update({
                "dataset_size"  : N_TRIALS,
                "comp_dur_val"  : comp,
                "std_order_val" : order,
            })
            trials = generate_trials("Interval_Discrim", batch_hp,
                                     mode="random", noise_on=False,
                                     align_mode=align_event)  # helper added in generate_trials.py

            with torch.no_grad():
                h = model.rnn(torch.tensor(trials.x, dtype=torch.float32))  # (B,T,N)

            hidden[i_c, i_o] = h.numpy()                                   # store trials

    # hidden.shape == (comp, order, trials, T, N)
    comp, order, trials, T, N = hidden.shape
    hidden = hidden.transpose(4, 3, 0, 1, 2)   # now (N, T, comp, order, trials)
    trialX = hidden.copy()
    X      = hidden.mean(axis=-1)              # average over trials axis
    return X, trialX

# -----------------------------------------------------------------------------
# Run dPCA
# -----------------------------------------------------------------------------

def run_dpca(X: np.ndarray, trialX: np.ndarray, labels: str = "cto"):
    dpca = dPCA(labels=labels, regularizer="auto")
    Z    = dpca.fit_transform(X, trialX=trialX)
    return dpca, Z

# -----------------------------------------------------------------------------
# Diagnostics plot
# -----------------------------------------------------------------------------

def quick_plot(Z: dict[str, np.ndarray], title: str):
    comp_traj = Z["c"][0]                   # first PC of comparison marginal
    T         = comp_traj.shape[0]
    for i, comp in enumerate(COMP_DURS):
        plt.plot(np.arange(T), comp_traj[:, i, 0], label=f"{comp} ms")
    plt.xlabel("time (a.u.)")
    plt.title(title)
    plt.legend()
    plt.show()

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    print(f"Building dataset (ALIGN_EVENT = {ALIGN_EVENT}) …")
    X, trialX = build_dataset(ALIGN_EVENT)

    print("Running dPCA …")
    dpca, Z = run_dpca(X, trialX, labels="cto")

    # ── output ───────────────────────────────────────────────────────────────
    print("\nExplained‑variance by marginal (first 6 PCs each):")
    for marg, frac in dpca.explained_variance_ratio_.items():
        print(f"  {marg:5s}: {100*frac.sum():5.1f} %")

    quick_plot(Z, "dPCA comp‑marginal, PC‑1")


if __name__ == "__main__":
    main()
