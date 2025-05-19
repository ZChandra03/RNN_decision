import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

from rnn_model import RNNModel
from task import generate_trials

# ─── Configuration ────────────────────────────────────────────────────────────
BASE_DIR    = os.path.abspath(os.path.dirname(__file__))
MODEL_DIR   = os.path.join(BASE_DIR, "models", "easy_trained")
OUTPUT_DIR  = os.path.join(BASE_DIR, "figures")
BATCH_SIZE  = 10000     # trials per batch
THRESHOLD   = 0.4       # decision threshold on output unit
NUM_SAMPLES = 4         # examples per category

os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_dir):
    with open(os.path.join(model_dir, "hp.json"), "r") as f:
        hp = json.load(f)
    model = RNNModel(hp)
    ckpt = torch.load(os.path.join(model_dir, "checkpoint.pt"), map_location="cpu")
    model.load_state_dict(ckpt)
    model.to(DEVICE).eval()
    return model, hp


def make_validation_batch(hp, size):
    hp_val = hp.copy()
    hp_val["dataset_size"] = size
    trial = generate_trials(hp["rule"], hp_val, mode="random", noise_on=True)
    x = torch.tensor(trial.x, dtype=torch.float32)            # (B, T, n_in)
    responds   = torch.tensor(trial.respond, dtype=torch.bool) # (B,)
    int2_ons   = torch.tensor(trial.int2_ons, dtype=torch.long)# (B,)
    return x, responds, int2_ons


def classify_trials(model, x, responds, int2_ons, threshold):
    """Returns TP, FP, FN, TN boolean masks plus the full y_hat (B, T)."""
    with torch.no_grad():
        y_hat = model(x.to(DEVICE))            # (B, T, 1)
    y_hat = y_hat.cpu().squeeze(-1)            # (B, T)
    fired = y_hat > threshold                  # (B, T)

    B, T = fired.shape
    t_idx = torch.arange(T).unsqueeze(0)       # (1, T)
    epoch_mask = t_idx >= int2_ons.unsqueeze(1)  # (B, T)

    fired_in  = (fired & epoch_mask).any(dim=1)     # (B,)
    fired_out = (fired & ~epoch_mask).any(dim=1)    # (B,)
    fired_any = fired.any(dim=1)                    # (B,)

    resp    = responds
    nonresp = ~resp

    TP = resp &  fired_in & ~fired_out
    FN = resp & (~fired_in |  fired_out)
    FP = nonresp & fired_any
    TN = nonresp & ~fired_any

    return TP, FP, FN, TN, y_hat


def collect_samples(chosen, x_batch, y_hat, TP, FP, FN, TN):
    """Fill up to NUM_SAMPLES per category from this batch."""
    masks = {'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN}
    for cat, mask in masks.items():
        idxs = torch.nonzero(mask).squeeze(-1).tolist()
        for idx in idxs:
            if len(chosen[cat]) < NUM_SAMPLES:
                # store the single-trial input (CPU) and output trace (numpy)
                xt     = x_batch[idx:idx+1].clone()      # shape (1, T, n_in)
                y_hat_i= y_hat[idx].cpu().numpy()       # shape (T,)
                chosen[cat].append({'xt': xt, 'y_hat': y_hat_i})
            else:
                break


if __name__ == "__main__":
    model, hp = load_model(MODEL_DIR)

    # 1) Initial batch → confusion & accuracy
    x0, r0, t0 = make_validation_batch(hp, BATCH_SIZE)
    TP0, FP0, FN0, TN0, y0 = classify_trials(model, x0, r0, t0, THRESHOLD)

    n_TP = int(TP0.sum().item())
    n_FP = int(FP0.sum().item())
    n_FN = int(FN0.sum().item())
    n_TN = int(TN0.sum().item())
    acc = (n_TP + n_TN) / BATCH_SIZE

    print(f"Confusion counts: TP={n_TP}, FP={n_FP}, FN={n_FN}, TN={n_TN}")
    print(f"Accuracy: {acc:.4f}")

    # 2) Collect samples until we have 4 of each
    chosen = {cat: [] for cat in ['TP','FP','FN','TN']}
    collect_samples(chosen, x0, y0, TP0, FP0, FN0, TN0)

    # loop further batches if needed
    while any(len(chosen[cat]) < NUM_SAMPLES for cat in chosen):
        x1, r1, t1 = make_validation_batch(hp, BATCH_SIZE)
        TP1, FP1, FN1, TN1, y1 = classify_trials(model, x1, r1, t1, THRESHOLD)
        collect_samples(chosen, x1, y1, TP1, FP1, FN1, TN1)

    # 3) Lock neuron‐sort order to the very first TP sample
    ref_xt = chosen['TP'][0]['xt'].to(DEVICE)  # (1, T, n_in)
    with torch.no_grad():
        h_ref = model.rnn(ref_xt)               # (1, T, N)
    h_arr_ref = h_ref.cpu().numpy()[0].T        # (N, T)
    peaks_ref = h_arr_ref.argmax(axis=1)
    order = np.argsort(peaks_ref)

    # 4) Plot one file per category
    dt_ms = hp.get('dt', 10)
    T = h_arr_ref.shape[1]
    times = np.arange(T) * (dt_ms/1000.)

    for cat, samples in chosen.items():
        n = len(samples)
        fig, axes = plt.subplots(2, n, figsize=(4*n, 4),
                                 gridspec_kw={"height_ratios":[3,1], "wspace":0.3, "hspace":0.3})

        for j, s in enumerate(samples):
            # hidden‐state heatmap
            xt   = s['xt'].to(DEVICE)
            with torch.no_grad():
                h_seq = model.rnn(xt)             # (1, T, N)
            h_arr = h_seq.cpu().numpy()[0].T     # (N, T)
            h_sorted = np.clip(h_arr[order], 0, None)

            im = axes[0,j].imshow(h_sorted, aspect='auto', cmap='viridis',
                                   vmin=0, vmax=1,
                                   extent=[times[0], times[-1], 0, h_sorted.shape[0]])
            axes[0,j].set_title(f"{cat} #{j+1}", fontsize=9)
            axes[0,j].set_xlim(times[[0,-1]])
            if j==0:
                axes[0,j].set_ylabel("Units")

            # output‐trace
            y_hat_i = s['y_hat']
            axes[1,j].plot(times, y_hat_i, linewidth=1.5)
            axes[1,j].axhline(THRESHOLD, linestyle='--', linewidth=1)
            axes[1,j].set_xlim(times[[0,-1]])
            axes[1,j].set_ylim([-0.05,1.05])
            if j==0:
                axes[1,j].set_ylabel("Output")

        # shared colorbar
        cbar = fig.colorbar(im, ax=axes[0,:].tolist(),
                            orientation='vertical', shrink=0.8)
        cbar.set_label("Activity", rotation=270, labelpad=12)

        fig.suptitle(f"{cat} Examples", fontsize=14)
        fig.tight_layout(rect=[0,0,1,0.95])

        out_path = os.path.join(OUTPUT_DIR, f"failure_analysis_{cat}.png")
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"→ saved {out_path}")
