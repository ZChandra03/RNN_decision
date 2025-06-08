#!/usr/bin/env python3
# divergence.py
# ---------------------------------------------------------------------
# Three linear-decoder curves, aligned on I2 onset (t=0):
#  • blue  → order0 success  vs order0 failure
#  • red   → order0 failure  vs order1 success
#  • green → order0 success  vs order1 success
# Marks:
#   • grey dashed   – I1 offset (order 0)
#   • black solid  – I2 onset (t=0)
#   • grey dotted  – I2 offset (order 0)
# ---------------------------------------------------------------------
import os, json, torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from generate_trials import generate_case_batch

# ─── Paths ────────────────────────────────────────────────────────────
BASE_DIR  = os.path.abspath(os.path.dirname(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "failure_conditions")
MODEL_DIR = os.path.join(BASE_DIR, "models", "easy_trained")
FIG_DIR   = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ─── Load hidden-state bundles ────────────────────────────────────────
succ0 = torch.load(os.path.join(DATA_DIR, "d180_order0_successes.pt"))
fail0 = torch.load(os.path.join(DATA_DIR, "d180_order0_failures.pt"))
succ1 = torch.load(os.path.join(DATA_DIR, "d180_order1_successes.pt"))

h_succ0 = succ0["h"].numpy()   # (100, T0, N)
h_fail0 = fail0["h"].numpy()
h_succ1 = succ1["h"].numpy()   # (100, T1, N)

# ─── Load dt from hp.json ─────────────────────────────────────────────
with open(os.path.join(MODEL_DIR, "hp.json"), "r") as f:
    hp = json.load(f)
dt_ms = hp.get("dt", 10)  # default 10 ms

# ─── Get I1-off / I2-on / I2-off for each order (batch=1) ─────────────
_, _, i1_on0, i1_off0, i2_on0, i2_off0 = generate_case_batch(hp, 180, 0, 1)
_, _, i1_on1, i1_off1, i2_on1, i2_off1 = generate_case_batch(hp, 180, 1, 1)
i1_off0, i2_on0, i2_off0 = map(int, (i1_off0,  i2_on0,  i2_off0))
i1_off1, i2_on1          = map(int, (i1_off1,  i2_on1))

# ─── Helper: minimal-trim decoder accuracy curve ──────────────────────
def acc_curve(A, i2_on_A, B, i2_on_B):
    """
    Align two trial sets on their own I2 onset (t=0), trimming only the
    extra pre- and post-I2 samples so both share the same window.
    Returns (times, acc) where times are in seconds relative to I2 onset.
    """
    T_A, T_B = A.shape[1], B.shape[1]
    pre_A, pre_B   = i2_on_A,      i2_on_B
    post_A, post_B = T_A - i2_on_A, T_B - i2_on_B

    min_pre  = min(pre_A,  pre_B)
    min_post = min(post_A, post_B)

    start_rel = -min_pre            # ≤ 0
    stop_rel  =  min_post - 1       # ≥ 0
    L = stop_rel - start_rel + 1

    start_A, end_A = i2_on_A + start_rel, i2_on_A + stop_rel
    start_B, end_B = i2_on_B + start_rel, i2_on_B + stop_rel
    Xa = A[:, start_A:end_A+1, :]
    Xb = B[:, start_B:end_B+1, :]

    X = np.concatenate([Xa, Xb], axis=0)
    y = np.concatenate([
        np.zeros(Xa.shape[0], dtype=int),
        np.ones (Xb.shape[0], dtype=int),
    ])

    idx = np.arange(X.shape[0])
    tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=42)

    acc = np.zeros(L)
    for t in range(L):
        clf = LogisticRegression(
            solver="liblinear", penalty="l2", max_iter=4000, n_jobs=1
        )
        clf.fit(X[tr, t, :], y[tr])
        acc[t] = accuracy_score(y[te], clf.predict(X[te, t, :]))

    times = (np.arange(start_rel, stop_rel + 1) * dt_ms) / 1000.0
    return times, acc

# ─── Compute curves ───────────────────────────────────────────────────
t_sf, acc_sf = acc_curve(h_succ0, i2_on0, h_fail0, i2_on0)  # o0 succ vs o0 fail
t_fs, acc_fs = acc_curve(h_fail0, i2_on0, h_succ1, i2_on1)  # o0 fail vs o1 succ
t_ss, acc_ss = acc_curve(h_succ0, i2_on0, h_succ1, i2_on1)  # o0 succ vs o1 succ

# ─── Plot ─────────────────────────────────────────────────────────────
plt.figure(figsize=(8,4))
plt.plot(t_sf, acc_sf, lw=2, label="o0 success vs o0 failure", color="C0")
plt.plot(t_fs, acc_fs, lw=2, ls="--", label="o0 failure vs o1 success", color="C1")
plt.plot(t_ss, acc_ss, lw=2, ls=":", label="o0 success vs o1 success", color="C2")

# event markers (relative to order-0 I2 onset)
plt.axvline((i1_off0 - i2_on0)*dt_ms/1000.0,
            color="grey",   ls="--", lw=1.2, label="I1 offset (o0)")
plt.axvline(0,
            color="black",  ls="-",  lw=1.2, label="I2 onset")
plt.axvline((i2_off0 - i2_on0)*dt_ms/1000.0,
            color="grey",   ls=":",  lw=1.2, label="I2 offset (o0)")

plt.xlabel("Time relative to I2 onset (s)")
plt.ylabel("Held-out decoder accuracy")
plt.title("Hidden-state separability  •  d = 180 ms")
plt.ylim(0, 1.05)
plt.legend(fontsize=8, frameon=False)
plt.grid(True, alpha=0.3)
plt.tight_layout()

out_png = os.path.join(FIG_DIR, "decoder_comparison_aligned_I2.png")
plt.savefig(out_png, dpi=300)
print("✓ Saved plot to", out_png)
