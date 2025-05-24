#!/usr/bin/env python3
"""
Animated PCA scatter‑plot of RNN hidden states *without* temporal averaging.
At every animation frame we use the raw hidden activation at that exact time
step (aligned to int‑2 onset) instead of a moving window.

Changes compared to the previous version:
• Removed WINDOW_MS, avg_pool1d smoothing and any stride‑1 pooling.
• Time axis now centres int‑2 onset at t = 0 ms directly via
  ``times = (np.arange(frames) - int2_idx) * dt``.
• Everything else (classification, colouring, Pillow GIF export, PyQt5 live
  window) is unchanged.

Run:
    python pca_animation_pointwise.py

Dependencies: torch, numpy, scikit‑learn, PyQt5, matplotlib, Pillow.
"""

import os, sys, json
import numpy as np
import torch
from sklearn.decomposition import PCA

# GUI and plotting -----------------------------------------------------------
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtCore    import QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

# Project‑level modules -------------------------------------------------------
from rnn_model     import RNNModel
from failure_count import generate_case_batch

# ---------------- User‑tweakable params ------------------------------------
START_MS = -500       # clip start (ms relative to int‑2 onset)
END_MS   =  500       # clip end   (ms relative to int‑2 onset)
OUT_GIF  = "pca_pointwise" + str(START_MS) + "_" + str(END_MS) + ".gif"
FPS      = 10

# ---------------- Constants -------------------------------------------------
BASE_DIR   = os.path.abspath(os.path.dirname(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "models", "easy_trained")
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1000
COMP_DURS  = [180, 220]
ORDERS     = [0]
THRESHOLD  = 0.4
PCA_NCOMP  = 10
COLOUR_MAP = {"TP": "tab:green", "FP": "tab:orange",
              "FN": "tab:red",   "TN": "tab:blue"}

# ---------------- Helper functions -----------------------------------------

def load_model(model_dir):
    """Load trained RNN and hyper‑parameters from *model_dir*."""
    with open(os.path.join(model_dir, "hp.json"), "r") as f:
        hp = json.load(f)
    model = RNNModel(hp)
    ckpt  = torch.load(os.path.join(model_dir, "checkpoint.pt"), map_location="cpu")
    model.load_state_dict(ckpt)
    model.to(DEVICE).eval()
    return model, hp


def classify_trial(y_pred, int2_on, responds):
    """Return TP / FP / FN / TN label for one trial."""
    fired_any  = (y_pred > THRESHOLD).any().item()
    fired_post = (y_pred[int(int2_on):] > THRESHOLD).any().item()
    if responds:
        return "TP" if fired_post else "FN"
    else:
        return "FP" if fired_any else "TN"


def build_dataset(model, hp):
    """Generate trials, align on int‑2 onset and return raw hidden states."""
    dt = hp["dt"]
    batches, min_pre, min_post = [], np.inf, np.inf

    # 1) Generate all requested condition × order batches and track limits
    for dur in COMP_DURS:
        for order in ORDERS:
            x, resp, int2 = generate_case_batch(hp, dur, order, BATCH_SIZE)
            with torch.no_grad():
                h = model.rnn(x.to(DEVICE)).cpu()                # (B,T,N)
                y = model(x.to(DEVICE)).cpu().squeeze(-1)        # (B,T)
            pre  = int2.numpy()                     # samples before int‑2 per trial
            post = h.shape[1] - pre                # samples after int‑2 per trial
            min_pre  = min(min_pre,  pre.min())
            min_post = min(min_post, post.min())
            batches.append((h, y, int2, resp))

    min_pre, min_post = int(min_pre), int(min_post)

    # 2) Clip every trial to the common window and collect colours
    clipped, colours = [], []
    for h, y, int2, resp in batches:
        for i in range(h.shape[0]):
            s = int(int2[i]) - min_pre               # inclusive start idx
            e = int(int2[i]) + min_post              # exclusive end idx
            clipped.append(h[i, s:e])               # (L,N)
            lbl = classify_trial(y[i], int2[i], bool(resp[i]))
            colours.append(COLOUR_MAP[lbl])

    h_cl   = torch.stack(clipped)      # (trials, L, N)
    frames = h_cl.shape[1]
    int2_idx = min_pre                 # aligned onset index inside window
    times  = (np.arange(frames) - int2_idx) * dt

    return h_cl.numpy(), colours, times


def compute_pca(h):
    """Compute PCA across *all* trials × time, return time‑series in PC space."""
    X = h.reshape(-1, h.shape[-1])          # (trials*L, N)
    X = (X - X.mean(0)) / X.std(0)          # z‑score per unit
    pca = PCA(n_components=PCA_NCOMP, svd_solver="full")
    Z   = pca.fit_transform(X)              # (trials*L, K)
    return Z.reshape(h.shape[0], h.shape[1], PCA_NCOMP)

# ---------------- GIF export ------------------------------------------------

def save_gif(Z, colours, times, filename, fps):
    if not filename:
        return
    # ensure output is saved to the script's directory
    output_path = os.path.join(BASE_DIR, filename)

    true_idx  = [i for i,c in enumerate(colours) if c in (COLOUR_MAP["TP"], COLOUR_MAP["TN"])]
    false_idx = [i for i,c in enumerate(colours) if c in (COLOUR_MAP["FP"], COLOUR_MAP["FN"])]

    fig, ax = plt.subplots(figsize=(6,5))
    scat_t = ax.scatter(Z[true_idx,0,0], Z[true_idx,0,1], c=[colours[i] for i in true_idx], s=15, alpha=0.3)
    scat_f = ax.scatter(Z[false_idx,0,0], Z[false_idx,0,1], c=[colours[i] for i in false_idx], s=15, alpha=0.8)
    ax.set(xlabel="PC‑1", ylabel="PC‑2",
           xlim=(Z[:,:,0].min()*1.1, Z[:,:,0].max()*1.1),
           ylim=(Z[:,:,1].min()*1.1, Z[:,:,1].max()*1.1))

    def update(frame):
        scat_t.set_offsets(np.c_[Z[true_idx,frame,0],  Z[true_idx,frame,1]])
        scat_f.set_offsets(np.c_[Z[false_idx,frame,0], Z[false_idx,frame,1]])
        ax.set_title(f"t = {times[frame]:.0f} ms")
        return scat_t, scat_f

    anim = animation.FuncAnimation(fig, update, frames=len(times), blit=True)
    print(f"Saving GIF to {output_path}…")
    anim.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print("✓ GIF saved")

# ---------------- PyQt5 interactive window ---------------------------------
class PCAScatterWindow(QMainWindow):
    def __init__(self, Z, colours, times):
        super().__init__()
        self.Z = Z; self.colours = colours; self.times = times
        self.true_idx  = [i for i,c in enumerate(colours) if c in (COLOUR_MAP["TP"], COLOUR_MAP["TN"])]
        self.false_idx = [i for i,c in enumerate(colours) if c in (COLOUR_MAP["FP"], COLOUR_MAP["FN"])]
        self.frames = Z.shape[1]; self.i = 0

        self.fig, self.ax = plt.subplots(figsize=(6,5))
        self.canvas = FigureCanvas(self.fig)
        ctr = QWidget(); lay = QVBoxLayout(ctr); lay.addWidget(self.canvas); self.setCentralWidget(ctr)

        # initial scatter
        self.s_t = self.ax.scatter(Z[self.true_idx,0,0],  Z[self.true_idx,0,1],  c=[colours[i] for i in self.true_idx],  s=15, alpha=0.3)
        self.s_f = self.ax.scatter(Z[self.false_idx,0,0], Z[self.false_idx,0,1], c=[colours[i] for i in self.false_idx], s=15, alpha=0.8)
        self.ax.set(xlabel="PC‑1", ylabel="PC‑2",
                   xlim=(Z[:,:,0].min()*1.1, Z[:,:,0].max()*1.1),
                   ylim=(Z[:,:,1].min()*1.1, Z[:,:,1].max()*1.1),
                   title=f"t = {times[0]:.0f} ms")

        self.timer = QTimer(self); self.timer.timeout.connect(self.step)
        self.timer.start(1000 // FPS)

    def step(self):
        self.i += 1
        if self.i >= self.frames:
            self.timer.stop(); return
        self.s_t.set_offsets(np.c_[self.Z[self.true_idx,self.i,0],  self.Z[self.true_idx,self.i,1]])
        self.s_f.set_offsets(np.c_[self.Z[self.false_idx,self.i,0], self.Z[self.false_idx,self.i,1]])
        self.ax.set_title(f"t = {self.times[self.i]:.0f} ms")
        self.canvas.draw_idle()

# ---------------- Main ------------------------------------------------------
def main():
    model, hp = load_model(MODEL_DIR)
    h_raw, colours, times = build_dataset(model, hp)
    Z = compute_pca(h_raw)

    # Optional clip for START_MS / END_MS
    mask = np.ones_like(times, dtype=bool)
    if START_MS is not None:
        mask &= times >= START_MS
    if END_MS is not None:
        mask &= times <= END_MS

    Z_sub     = Z[:, mask]
    times_sub = times[mask]

    save_gif(Z_sub, colours, times_sub, OUT_GIF, FPS)

    app = QApplication(sys.argv)
    win = PCAScatterWindow(Z_sub, colours, times_sub)
    win.setWindowTitle(f"Animated PCA scatter (N = {Z_sub.shape[0]})")
    win.resize(800, 600)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
