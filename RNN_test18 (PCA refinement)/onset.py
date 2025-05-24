import numpy as np
import torch
import matplotlib.pyplot as plt

import train
from failure_count import generate_case_batch

# ─── Parameters ─────────────────────────────────────────────────────────────
DURS   = [180, 220]      # comp durations (ms)
ORDERS = [0, 1]          # std_order_val values
BATCH  = 1
RULE   = "Interval_Discrim"

# ─── Base hyperparams ───────────────────────────────────────────────────────
HP_base = train.get_default_hp()
HP_base.update({
    "comp_step":    HP_base.get("comp_step", 20),
    "dataset_size": BATCH,
    "rule":         RULE,
})
dt = HP_base["dt"]

# ─── Collect raw series & int2 onsets ───────────────────────────────────────
raw_series = []   # will hold tuples (order, dur, trace)
int2_idxs  = []

for order in ORDERS:
    for dur in DURS:
        HP = HP_base.copy()
        HP["std_order_val"] = order
        HP["comp_dur_val"]  = dur

        x_batch, _, int2 = generate_case_batch(HP, dur, order, BATCH)
        x = x_batch.cpu().numpy()[0, :, 0]   # channel 0
        raw_series.append((order, dur, x))
        int2_idxs.append(int2.item())

# ─── Compute common symmetric window ────────────────────────────────────────
pres   = int2_idxs
posts  = [len(x) - p for (_, _, x), p in zip(raw_series, pres)]
min_pre, min_post = min(pres), min(posts)
frames = min_pre + min_post

# ─── Clip & align all four trials ──────────────────────────────────────────
clipped = []
for (order, dur, x), p in zip(raw_series, pres):
    start = p - min_pre
    end   = p + min_post
    clipped.append((order, dur, x[start:end]))
new_idx = min_pre

# ─── Print onset info ──────────────────────────────────────────────────────
print("Original int2 indices (samples):", pres)
print("Original int2 times   (ms):    ", [round(p*dt,1) for p in pres])
print("Aligned int2 index    (sample):", new_idx)
print("Aligned int2 time     (ms):    ", round(new_idx*dt,1))

# ─── Plot ───────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharey=True)

# Top: Raw series + original onsets
for (order, dur, x), p in zip(raw_series, pres):
    t = np.arange(len(x)) * dt
    ax1.plot(t, x, label=f"order={order}, dur={dur} ms")
    print(len(t))
    ax1.axvline(p*dt, ls='--', label=f"int2 @ {p*dt:.0f} ms")
ax1.set(
    title="All 4 trials: Raw time series",
    xlabel="Time (ms)",
    ylabel="Channel 0 activity"
)
ax1.legend(loc='upper left')

# Bottom: Clipped & aligned + common onset
t_clip = np.arange(frames) * dt
for order, dur, x_cl in clipped:
    ax2.plot(t_clip, x_cl, label=f"order={order}, dur={dur} ms")
ax2.axvline(new_idx*dt, ls='--', color='k', label=f"aligned int2 @ {new_idx*dt:.0f} ms")
ax2.set(
    title="All 4 trials: Clipped & aligned",
    xlabel="Time (ms since clip start)"
)
ax2.legend(loc='upper left')

plt.tight_layout()
plt.show()
