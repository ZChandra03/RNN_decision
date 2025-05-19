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
OUTPUT_DIR  = os.path.join(BASE_DIR, "confusion_analysis_figure")
BATCH_SIZE  = 10000     # total trials to sample for computing TP, FP, etc.
THRESHOLD   = 0.4       # decision threshold on output unit
NUM_SAMPLES = 4         # samples per category

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
    responds = torch.tensor(trial.respond, dtype=torch.bool)   # (B,)
    int2_ons = torch.tensor(trial.int2_ons, dtype=torch.long) # (B,)
    return x, responds, int2_ons


def classify_trials(model, x, responds, int2_ons, threshold):
    with torch.no_grad():
        y_hat = model(x.to(DEVICE))       # (B, T, 1)
    y_hat = y_hat.cpu().squeeze(-1)      # (B, T)
    fired = y_hat > threshold

    B, T = fired.shape
    t_idx = torch.arange(T).unsqueeze(0)
    epoch_mask = t_idx >= int2_ons.unsqueeze(1)

    fired_in = (fired & epoch_mask).any(dim=1)
    fired_out = (fired & ~epoch_mask).any(dim=1)
    fired_any = fired.any(dim=1)

    resp = responds
    nonresp = ~resp

    TP = resp &  fired_in & ~fired_out
    FN = resp & (~fired_in |  fired_out)
    FP = nonresp & fired_any
    TN = nonresp & ~fired_any

    return TP, FP, FN, TN, y_hat


if __name__ == "__main__":
    model, hp = load_model(MODEL_DIR)
    x, responds, int2_ons = make_validation_batch(hp, BATCH_SIZE)
    TP, FP, FN, TN, all_y = classify_trials(model, x, responds, int2_ons, THRESHOLD)

    # pick one sample per category
    rng = np.random.RandomState(42)
    categories = {'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN}
    chosen = {}
    for cat, mask in categories.items():
        idxs = np.where(mask.numpy())[0]
        if len(idxs) > 0:
            chosen_idx = rng.choice(idxs, min(NUM_SAMPLES, len(idxs)), replace=False)[0]
            chosen[cat] = chosen_idx
        else:
            print(f"No examples of {cat} found.")

    # determine fixed neuron order based on TP trial
    tp_idx = chosen.get('TP')
    xt_tp = x[tp_idx:tp_idx+1].to(DEVICE)
    with torch.no_grad():
        h_tp = model.rnn(xt_tp)           # (1, T, N)
    h_arr_tp = h_tp.cpu().numpy()[0].T   # (N, T)
    peaks_tp = h_arr_tp.argmax(axis=1)
    order = np.argsort(peaks_tp)

    # prepare figure
    n_cat = len(chosen)
    dt_ms = hp.get('dt', 10)
    fig, axes = plt.subplots(2, n_cat, figsize=(12, 6),
                             gridspec_kw={"height_ratios": [3, 1], "hspace": 0.25})

    for i, (cat, idx) in enumerate(chosen.items()):
        # hidden activations
        xt = x[idx:idx+1].to(DEVICE)
        with torch.no_grad():
            h_seq = model.rnn(xt)
        h_arr = h_seq.cpu().numpy()[0].T     # (N, T)

        # apply fixed order and clip negatives
        h_sorted = h_arr[order]
        h_clipped = np.clip(h_sorted, 0, None)

        T = h_arr.shape[1]
        times = np.arange(T) * (dt_ms / 1000.0)

        im = axes[0, i].imshow(h_clipped, aspect='auto', cmap='viridis',
                               vmin=0, vmax=1,
                               extent=[times[0], times[-1], 0, h_clipped.shape[0]])
        axes[0, i].set_title(f"{cat}", fontsize=10)
        axes[0, i].set_ylabel("Active Units")
        axes[0, i].set_xlim([times[0], times[-1]])

        # output trace
        y_hat = all_y[idx]
        axes[1, i].plot(times, y_hat, linewidth=2)
        axes[1, i].set_xlabel("Time (s)")
        axes[1, i].set_ylabel("Output")
        axes[1, i].set_ylim([-0.05, 1.05])
        axes[1, i].set_xlim([times[0], times[-1]])
        axes[1, i].axhline(0, color='k', linewidth=0.5)

    # shared colorbar
    cbar = fig.colorbar(im, ax=axes[0, :].tolist(), orientation='vertical', shrink=0.8)
    cbar.set_label("Activity (clipped & saturated)", rotation=270, labelpad=15)

    fig.suptitle("Failure Analysis: Hidden States and Outputs", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(OUTPUT_DIR, "failure_analysis_figure.png")
    fig.savefig(save_path, dpi=300)
    print(f"Figure saved to {save_path}")
    plt.show()
