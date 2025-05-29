"""
Demixed‑PCA (dPCA) pipeline for the Interval Discrimination RNN
===============================================================

This script mirrors the data‑loading logic of **pca_animation_pointwise.py** and
wraps it into a clean dPCA workflow.  Change the *User‑tweakable params* section
below to control the comparison durations, order conditions, number of trials
per condition, and—crucially—the **alignment event** used to time‑lock every
trial.

The output consists of:
    • the fitted ``dpca`` object (use it for cross‑validated decoding later),
    • a ``Z`` dictionary of low‑dimensional trajectories (one array per
      marginal),
    • a matplotlib figure showing the explained‑variance pie and the first
      comparison‑marginal trajectory.

Run::

    python dpca_analysis.py

Dependencies: ``torch``, ``numpy``, ``matplotlib``, ``dPCA``
"""

# ---------------------------------------------------------------------------
# Standard lib
# ---------------------------------------------------------------------------
import os, json, math
from pathlib import Path

# ---------------------------------------------------------------------------
# Third‑party
# ---------------------------------------------------------------------------
import numpy as np
import torch
import matplotlib.pyplot as plt
from dPCA.dPCA import dPCA

# ---------------------------------------------------------------------------
# Project‑level imports (same folder structure as pca_animation_pointwise)
# ---------------------------------------------------------------------------
from rnn_model      import RNNModel
from generate_trials import generate_case_batch   # helper provided earlier

# -------------------------- User‑tweakable params ---------------------------
# Comparison durations (ms) and interval orders to probe
COMP_DURS          = [180, 220]
ORDERS             = [0, 1]          # 0 = standard‑first, 1 = standard‑second
TRIALS_PER_COND    = 40              # balanced repetitions for averaging

# Choose which event each trial is aligned to before clipping.
# One of: 'int1_on', 'int1_off', 'int2_on', 'int2_off'
ALIGN_EVENT        = 'int2_on'

# Path to the trained model
BASE_DIR   = Path(__file__).resolve().parent
MODEL_DIR  = BASE_DIR / 'models' / 'easy_trained'
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------- Helper functions -------------------------------

def load_model(model_dir: Path):
    """Return a *(model, hp)* tuple loaded from *model_dir*."""
    with open(model_dir / 'hp.json', 'r') as f:
        hp = json.load(f)
    model = RNNModel(hp)
    ckpt  = torch.load(model_dir / 'checkpoint.pt', map_location='cpu')
    model.load_state_dict(ckpt)
    model.to(DEVICE).eval()
    return model, hp


def _pick_alignment_idx(event: str, int1_on, int1_off, int2_on, int2_off):
    """Return a 1‑D tensor of indices (one per trial) for the chosen *event*."""
    mapping = {
        'int1_on':  int1_on,
        'int1_off': int1_off,
        'int2_on':  int2_on,
        'int2_off': int2_off,
    }
    if event not in mapping:
        raise ValueError(f"Unknown ALIGN_EVENT '{event}'.")
    return mapping[event]


def build_aligned_dataset(model, hp, comp_durs, orders, n_trials, align_event):
    """Generate hidden‑state tensors for every (comp_dur, order) condition,
    time‑lock them on *align_event*, and clip to the largest common window.

    Returns
    -------
    X   : np.ndarray – shape (neurons, time, n_comp, n_order)
    tms : np.ndarray – time stamps (ms) relative to *align_event*
    """
    dt = hp['dt']
    hidden_blocks = []        # list of lists indexed [comp][order] later
    pre_lengths, post_lengths = [], []

    # 1) Generate all raw trials and track the min pre / post window
    for comp in comp_durs:
        row = []
        for order in orders:
            # x shape (B,T,1) — we discard respond flags for dPCA
            x, _, int1_on, int1_off, int2_on, int2_off = generate_case_batch(
                hp, comp, order, n_trials)
            with torch.no_grad():
                h = model.rnn(x.to(DEVICE)).cpu()     # (B,T,N)
            # Pick alignment index per trial
            align_idx = _pick_alignment_idx(align_event, int1_on, int1_off,
                                            int2_on, int2_off)               # (B,)

            # For each trial compute #samples before / after alignment
            pre  = align_idx.numpy()                    # inclusive counts pre‑event
            post = h.shape[1] - pre                    # counts from event to end
            pre_lengths.append(pre.min())
            post_lengths.append(post.min())
            row.append((h, align_idx))                 # stash to clip later
        hidden_blocks.append(row)

    min_pre  = int(min(pre_lengths))   # samples we can keep before event
    min_post = int(min(post_lengths))  # samples incl. event and after
    L        = min_pre + min_post      # final time‑points per trial

    # 2) Clip & average inside each condition → shape (N,L)
    blocks_mean = []
    for row in hidden_blocks:
        row_mean = []
        for h, align_idx in row:
            clips = []
            for i in range(h.shape[0]):
                s = int(align_idx[i]) - min_pre        # inclusive start
                e = s + L                             # exclusive end
                clips.append(h[i, s:e])               # (L,N)
            clips = torch.stack(clips)                # (B,L,N)
            row_mean.append(clips.mean(0).T.numpy())  # (N,L)
        blocks_mean.append(row_mean)

    # 3) Stack into (N, L, n_comp, n_order)
    X = np.stack([[bm for bm in row] for row in blocks_mean], axis=2)  # (N,L,comp,order)

    times = (np.arange(L) - min_pre) * dt                             # ms relative to align_event
    return X, times


def run_dpca(X, labels='cto'):
    """Fit dPCA on *X* and return *(dpca, Z)* where Z is the transformed dict."""
    dpca = dPCA(labels=labels, regularizer=None)
    Z    = dpca.fit_transform(X)
    return dpca, Z


# ------------------------------ Main ---------------------------------------

def main():
    print("Loading model …")
    model, hp = load_model(MODEL_DIR)

    print(f"Building dataset (ALIGN_EVENT = {ALIGN_EVENT}) …")
    X, times = build_aligned_dataset(model, hp, COMP_DURS, ORDERS,
                                     TRIALS_PER_COND, ALIGN_EVENT)

    # X has shape (N,T,C,O) — match dPCA expectation (neurons, time, cond1, cond2)
    print("Running dPCA …")
    dpca, Z = run_dpca(X, labels='cto')

    # ---------------------------------------------------------------------
    # Quick‑look diagnostic plots
    # ---------------------------------------------------------------------
    pie = dpca.explained_variance_ratio_            # dict with same keys as Z
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    # 1) Explained‑variance pie chart
    ax = axs[0]
    labels = list(pie.keys())

    # Some entries are ndarray (one value per component); make them scalars
    fracs = [float(np.sum(pie[k])) * 100 for k in labels]   # 1-D list

    ax.pie(fracs, labels=labels, autopct="%1.1f%%")
    ax.set_title("Variance by marginal")

    # ─── Make sure the time axis matches the length returned by dPCA ───
    T = Z['c'].shape[1]          # actual number of time-bins in the low-D output
    times = times[:T]            # trim the vector if it is longer


    # 2) First comp‑marginal trajectory
    ax = axs[1]
    for i, comp in enumerate(COMP_DURS):
        ax.plot(times, Z['c'][0, :, i, 0], label=f'{comp} ms')
    ax.axvline(0, c='k', lw=0.5)
    ax.set(xlabel=f"time relative to {ALIGN_EVENT} (ms)", ylabel='PC‑1 (comp)')
    ax.legend(title='first interval')
    fig.tight_layout()

    out = BASE_DIR / f"dpca_{ALIGN_EVENT}.png"
    fig.savefig(out, dpi=300)
    print(f"✓ Figures saved to {out}")

    # Print summary to stdout
    print("Explained variance by marginal:")
    for k, v in pie.items():
        frac = float(np.sum(v)) * 100          # collapse list → single % value
        print(f"  {k:<8s}: {frac:5.1f} %")


if __name__ == '__main__':
    main()