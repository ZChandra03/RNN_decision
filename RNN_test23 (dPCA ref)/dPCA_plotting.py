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
COMP_DURS          = [180,220]
ORDERS             = [0]          # 0 = standard‑first, 1 = standard‑second
TRIALS_PER_COND    = 40              # balanced repetitions for averaging

# Choose which event each trial is aligned to before clipping.
# One of: 'int1_on', 'int1_off', 'int2_on', 'int2_off'
ALIGN_EVENT        = 'int1_on'

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
    dpca = dPCA(labels=labels, regularizer=1e-5)
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
    # ❶  Explained-variance pie (all marginals together)
    # ---------------------------------------------------------------------
    pie_fig, pie_ax = plt.subplots(figsize=(4, 4))
    marginals = list(Z.keys())                  # e.g. ['c','t','o','ct','co','to','cto']
    fracs     = [float(np.sum(dpca.explained_variance_ratio_[k])) * 100
                for k in marginals]
    pie_ax.pie(fracs, labels=marginals, autopct="%1.1f%%")
    pie_ax.set_title("Explained variance by marginal")

    pie_out = BASE_DIR / f"dpca_{ALIGN_EVENT}_variance_pie.png"
    pie_fig.savefig(pie_out, dpi=300)
    plt.close(pie_fig)
    print(f"✓ Saved variance pie → {pie_out}")


    # ---------------------------------------------------------------------
    # ❷  One figure *per* marginal (first PC only)
    # ---------------------------------------------------------------------
    # Make a folder to keep things tidy
    fig_dir = BASE_DIR / f"dpca_{ALIGN_EVENT}"
    fig_dir.mkdir(exist_ok=True)

    # Ensure time vector matches low-dim output length
    T     = next(iter(Z.values())).shape[1]
    times = times[:T]              # shrink if dPCA truncated / padded

    for key, Zk in Z.items():
        pc1 = Zk[0]                # shape: (time, C, O) OR (time,) if nothing varies

        fig, ax = plt.subplots(figsize=(5, 3))
        time_axis = times[:pc1.shape[0]]

        # -----------------------------------------------------------------
        # Decide what varies in *this* marginal and plot only those traces
        # -----------------------------------------------------------------
        if pc1.ndim == 3:          # both factors still present
            n_C, n_O = pc1.shape[1], pc1.shape[2]

            if ('c' in key) and not ('o' in key):      # comparison-only marginal
                for i, comp in enumerate(COMP_DURS[:n_C]):
                    ax.plot(time_axis, pc1[:, i, 0], label=f"{comp} ms")

            elif ('o' in key) and not ('c' in key):    # order-only marginal
                for j, order in enumerate(ORDERS[:n_O]):
                    lbl = "std-first" if order == 0 else "std-second"
                    ax.plot(time_axis, pc1[:, 0, j], label=lbl)

            else:                                      # mixed (c & o) marginal
                for i, comp in enumerate(COMP_DURS[:n_C]):
                    for j, order in enumerate(ORDERS[:n_O]):
                        lbl = f"{comp} ms / " + ("1st" if order == 0 else "2nd")
                        ax.plot(time_axis, pc1[:, i, j], label=lbl,
                                lw=1, alpha=0.8)

        else:  # pc1.ndim == 1  → neither c nor o varies (e.g. pure 't')
            ax.plot(time_axis, pc1.squeeze(), c='k')

        # -----------------------------------------------------------------
        # Styling
        # -----------------------------------------------------------------
        ax.axvline(0, c='k', lw=0.5)
        ax.set(
            title = f"{key} marginal – PC1",
            xlabel=f"time relative to {ALIGN_EVENT} (ms)",
            ylabel="PC-1"
        )
        if pc1.ndim == 3:
            ax.legend(frameon=False, fontsize=8, ncol=1)

        fig.tight_layout()
        out_path = fig_dir / f"{key}_pc1.png"
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"✓ Saved {key} marginal → {out_path}")


    # ---------------------------------------------------------------------
    # ❸  Console summary
    # ---------------------------------------------------------------------
    print("\nExplained variance by marginal:")
    for k, v in dpca.explained_variance_ratio_.items():
        print(f"  {k:<5s}: {np.sum(v)*100:5.1f} %")



if __name__ == '__main__':
    main()
