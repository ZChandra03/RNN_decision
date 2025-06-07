#!/usr/bin/env python3
# divergence.py  – success-vs-failure decoder accuracy vs time
# now with vertical markers for int-1 / int-2 onset & offset
# --------------------------------------------------------------------
import os, json, torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from generate_trials import generate_case_batch          # <-- new import
# ─── Paths ─────────────────────────────────────────────────────────────
BASE_DIR  = os.path.abspath(os.path.dirname(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "failure_conditions")
MODEL_DIR = os.path.join(BASE_DIR, "models", "easy_trained")
FIG_DIR   = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ─── Load success / failure bundles ───────────────────────────────────
succ = torch.load(os.path.join(DATA_DIR, "d180_order0_successes.pt"))
fail = torch.load(os.path.join(DATA_DIR, "d180_order0_failures.pt"))

h_succ = succ["h"].numpy()
h_fail = fail["h"].numpy()

X = np.concatenate([h_succ, h_fail], axis=0)            # (200, T, N)
y = np.concatenate([np.zeros(h_succ.shape[0], dtype=int),
                    np.ones (h_fail.shape[0],  dtype=int)])

B, T, N = X.shape
print(f"Loaded {B} trials  |  {T} time steps  |  {N} hidden units")

# ─── Train/test split (re-used for every t) ───────────────────────────
train_idx, test_idx = train_test_split(
        np.arange(B), test_size=0.2, stratify=y, random_state=42)

# ─── Decoder sweep over time ──────────────────────────────────────────
acc_vs_time = np.zeros(T)
for t in range(T):
    X_train = X[train_idx, t, :]
    X_test  = X[test_idx,  t, :]

    clf = LogisticRegression(penalty="l2", solver="liblinear",
                             max_iter=4000, n_jobs=1)
    clf.fit(X_train, y[train_idx])
    acc_vs_time[t] = accuracy_score(y[test_idx], clf.predict(X_test))

# ─── Timing conversion & event markers  ───────────────────────────────
with open(os.path.join(MODEL_DIR, "hp.json"), "r") as f:
    hp = json.load(f)
dt_ms = hp.get("dt", 10)
times = np.arange(T) * dt_ms / 1000.0               # seconds

# get int-1 / int-2 boundaries once (batch=1 suffices)
_, _, int1_ons, int1_offs, int2_ons, int2_offs = \
    generate_case_batch(hp, comp_dur_val=180, std_order_val=0, batch_size=1)

markers = {
    "I1_on":  int1_ons.item()  * dt_ms / 1000.0,
    "I1_off": int1_offs.item() * dt_ms / 1000.0,
    "I2_on":  int2_ons.item()  * dt_ms / 1000.0,
    "I2_off": int2_offs.item() * dt_ms / 1000.0,
}

# ─── Plot ─────────────────────────────────────────────────────────────
plt.figure(figsize=(8,4))
plt.plot(times, acc_vs_time, lw=2, label="decoder accuracy")

# interval markers
plt.axvline(markers["I1_on"],  color="black",   lw=1.2, label="I1 on")
plt.axvline(markers["I1_off"], color="black",   lw=1.2, label="I1 off")
plt.axvline(markers["I2_on"],  color="black",  lw=1.2, label="I2 on")
plt.axvline(markers["I2_off"], color="black",  lw=1.2, label="I2 off")

plt.ylim(0, 1.05)
plt.xlabel("Time (s)")
plt.ylabel("Accuracy (success vs failure)")
plt.title("Linear-decoder separability over time\n(d = 180 ms, order = 0)")
plt.legend(loc="upper right", fontsize=8, frameon=False)
plt.grid(True)
plt.tight_layout()

out_png = os.path.join(FIG_DIR, "succ_fail_decoder_accuracy_vs_time.png")
plt.savefig(out_png, dpi=300)
print("✓ Figure saved to", out_png)
