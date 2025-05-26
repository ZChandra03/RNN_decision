#!/usr/bin/env python3
"""
Minimal dPCA analysis and visualisation for the interval‑discrimination RNN.

• Loads the trained model from `models/easy_trained`.
• Generates a balanced batch of trials (all comparison durations × orders).
• Aligns trials on the onset of the second interval (int‑2 on) and clips to the
  largest common window.
• Runs demixed PCA (labels = 'rt') and keeps the first N components.
• Produces two plots saved in the script directory:
    1. trajectories.png  – mean population trajectory in dPC‑1/2 space.
    2. timecourses.png   – variance‑normalised time‑courses of dPC‑1..3.

Run with:
    python dpca_basic.py  # produces PNGs next to the script
"""

import os, json, math
import numpy as np
import torch
import matplotlib.pyplot as plt
from dPCA.dPCA import dPCA

from rnn_model     import RNNModel
from failure_count import generate_case_batch

# ─── Configuration ─────────────────────────────────────────────────────────
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_DIR   = os.path.join(os.path.dirname(__file__), 'models', 'easy_trained')
BATCH_SIZE  = 512                 # trials per condition
COMP_DURS   = [120, 160, 180, 190, 210, 220, 240, 280]
ORDERS      = [0, 1]              # 0 = std first, 1 = comp first
N_DPC       = 10                  # number of dPCs to keep
ALIGN_MS    = (-500, 1000)        # window (ms) around int‑2 onset to keep

# ─── Helpers ──────────────────────────────────────────────────────────────

def load_model(model_dir):
    with open(os.path.join(model_dir, 'hp.json'), 'r') as f:
        hp = json.load(f)
    model = RNNModel(hp)
    ckpt  = torch.load(os.path.join(model_dir, 'checkpoint.pt'), map_location='cpu')
    model.load_state_dict(ckpt)
    model.to(DEVICE).eval()
    return model, hp


def build_dataset(model, hp):
    """Return hidden‑state tensor (trials, T, N) and time vector (ms)."""
    dt = hp['dt']
    batches = []
    min_pre = math.inf  # samples before int‑2
    min_post = math.inf # samples after int‑2

    # collect all conditions first so we know common window length
    for dur in COMP_DURS:
        for order in ORDERS:
            x, resp, int2_on, _ = generate_case_batch(hp, dur, order, BATCH_SIZE)
            with torch.no_grad():
                h = model.rnn(x.to(DEVICE)).cpu()      # (B,T,N)
            pre  = int2_on.numpy()
            post = h.shape[1] - pre
            min_pre  = min(min_pre,  pre.min())
            min_post = min(min_post, post.min())
            batches.append((h, int2_on))

    # clip every trial to common window
    clipped = []
    for h, int2_on in batches:
        for i in range(h.shape[0]):
            s = int(int2_on[i]) - min_pre
            e = int(int2_on[i]) + min_post
            clipped.append(h[i, s:e])

    h_all = torch.stack(clipped)            # (trials, L, N)
    times = (np.arange(h_all.shape[1]) - min_pre) * dt
    return h_all.numpy(), times


def run_dpca(h, n_comp=N_DPC):
    """Run dPCA on hidden activity and return low‑d trajectories & model."""
    X = np.transpose(h, (2, 0, 1)).astype(np.float32)  # (N, R, T)
    dpca = dPCA(labels='rt', n_components=n_comp)
    Z = dpca.fit_transform(X)['rt'].transpose(1, 2, 0)  # (trials, T, K)
    return dpca, Z


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    model, hp = load_model(MODEL_DIR)
    h, times  = build_dataset(model, hp)
    dpca, Z   = run_dpca(h)

    # 1) Mean trajectory in dPC‑1/2 space
    traj = Z.mean(axis=0)          # (T, K)
    plt.figure(figsize=(6, 5))
    plt.plot(traj[:, 0], traj[:, 1], lw=2)
    plt.scatter(traj[0, 0], traj[0, 1], label='start')
    plt.scatter(traj[len(times)//2, 0], traj[len(times)//2, 1], label='int‑2 on')
    plt.xlabel('dPC‑1'); plt.ylabel('dPC‑2'); plt.legend()
    plt.title('Mean population trajectory')
    plt.tight_layout(); plt.savefig('trajectories.png', dpi=150)

    # 2) Time‑courses of first 3 dPCs (variance‑normalised)
    var_expl = dpca.explained_variance_ratio_['rt']
    plt.figure(figsize=(7, 4))
    for k in range(3):
        plt.plot(times, traj[:, k] / traj[:, k].std(), label=f'dPC‑{k+1} ({var_expl[k]*100:.1f}% var)')
    plt.axvline(0, color='k', ls='--', lw=0.6)
    plt.xlabel('Time relative to int‑2 onset (ms)')
    plt.ylabel('z‑scored activity')
    plt.legend(); plt.tight_layout(); plt.savefig('timecourses.png', dpi=150)

    print('[✓] Saved trajectories.png and timecourses.png')


if __name__ == '__main__':
    main()
