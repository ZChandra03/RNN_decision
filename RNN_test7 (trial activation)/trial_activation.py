# plot_trial_activation.py
"""Generate single‑trial hidden‑state heatmaps and output traces for two
interval‑discrimination conditions using a trained RNNModel.

Conditions
----------
1. 140 ms (interval‑1) followed by the 200 ms standard  ─ short→standard
2. 260 ms (interval‑1) followed by the 200 ms standard  ─ long→standard

The script reproduces panel‑C of the paper figure the user attached:
  • upper row: hidden‑unit activations (units sorted by peak activity)
  • lower row: network output trace for the same trial

Edit the ``MODEL_DIR`` constant if your trained model lives elsewhere.

Usage (no CLI arguments needed):
    python plot_trial_activation.py
"""

import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

from task import Trial  # The base container class
from rnn_model import RNNModel  # Model definition

# ───────────────────────────────────────────────────────────────────────
# Configuration ── change these if your setup is different
# ───────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models", "easy_trained1")  # folder with hp.json + checkpoint.pt
SAVE_PATH = "trial_activation_figure.png"             # output figure file
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Stimulation parameters (ms)
STD_DUR_MS = 200     # standard interval duration (always second)
COMP_DURS_MS = [140, 260]  # comparison (first interval) durations
DT_MS = 10           # simulation time‑step in ms (must match training)
DELAY_MS = 500      # silent gap between tones (same as training)
TONE_DUR_MS = 20     # duration of tone marker itself (used in dataset generation)

# ───────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────

def load_model(model_dir: str):
    """Load RNN and its hyper‑parameters from *model_dir*."""
    with open(os.path.join(model_dir, "hp.json"), "r") as f:
        hp = json.load(f)
    model = RNNModel(hp)
    ckpt = torch.load(os.path.join(model_dir, "checkpoint.pt"), map_location="cpu")
    model.load_state_dict(ckpt)
    model.to(DEVICE).eval()
    return model, hp


def build_single_trial(hp: dict, comp_ms: int):
    """Create a deterministic single‑trial: *comp_ms* first, then 200 ms standard."""
    dt = hp["dt"]
    std_ms  = STD_DUR_MS
    tone_ms = hp["tone_dur"]

    # ── onset & offset indices (time‑steps) ────────────────────────────
    int1_on  = int(500 / dt)                               # first tone onset at 0.5 s
    int1_off = int1_on + int(comp_ms / dt)
    int2_on  = int1_off + int(tone_ms / dt) + int(DELAY_MS / dt)
    int2_off = int2_on + int(std_ms / dt)

    tdim = int2_off + int(2000 / dt)                      # run 2 s after tone‑2

    trial = Trial(hp, tdim, dataset_size=1)

    # behaviour label (respond if 2nd interval longer)
    trial.respond[0] = std_ms > comp_ms

    # ── stimuli ────────────────────────────────────────────────────────
    trial.add("interval_input", ons=int1_on, offs=int1_off)
    trial.add("interval_input", ons=int2_on, offs=int2_off)

    # target output (ramp from end of second tone to trial end)
    trial.add("discrim", ons=int2_off, offs=tdim)

    # cost mask: ignore first 100 ms; weigh decision window heavily
    post_on  = int2_off + int(50 / dt)
    post_off = int2_off + int(1000 / dt)
    trial.add_c_mask(pre_offs=int2_off, post_ons=post_on, post_offs=post_off)

    # input noise (keeps behaviour realistic)
    trial.add_x_noise()
    return trial


def sort_by_peak(act: np.ndarray):
    """Return activity sorted by the time of each neuron's peak."""
    peaks = act.argmax(axis=1)
    return act[np.argsort(peaks)]


def plot_condition(ax_heat, ax_out, model, hp, comp_ms: int):
    """Run network on one trial and draw heatmap + output."""
    trial = build_single_trial(hp, comp_ms)
    x = torch.tensor(trial.x, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        h = model.rnn(x)               # (1, T, N)
        y_hat = model(x)               # (1, T, 1)

    h = h.cpu().numpy()[0].T          # (N, T)
    y_hat = y_hat.cpu().numpy()[0, :, 0]

    # normalise & sort units for prettier display
    h_norm = np.clip(h, 0, None)
    h_norm /= (h_norm.max() + 1e-6)
    # h_sorted = sort_by_peak(h_norm)  # normalized version
    h_sorted = sort_by_peak(h)         # raw activations

    times = np.arange(h.shape[1]) * (hp["dt"] / 1000.0)  # → seconds

    # heatmap -----------------------------------------------------------
    im = ax_heat.imshow(h_sorted, aspect="auto", cmap="viridis", vmin=0, vmax=1,
                        extent=[times[0], times[-1], 0, h_sorted.shape[0]])
    ax_heat.set_ylabel("Active Units")
    ax_heat.set_title(f"{comp_ms} ms → {STD_DUR_MS} ms", fontsize=10)

    # red markers (tone on/offs)
    markers_ms = (
        500,
        500 + comp_ms,
        500 + comp_ms + TONE_DUR_MS + DELAY_MS,
        500 + comp_ms + TONE_DUR_MS + DELAY_MS + STD_DUR_MS,
    )
    for t_ms in markers_ms:
        ax_heat.axvline(t_ms / 1000.0, color="red", linewidth=1)

    # output trace ------------------------------------------------------
    ax_out.plot(times, y_hat, linewidth=2)
    ax_out.set_xlabel("Time (s)")
    ax_out.set_ylabel("Output")
    ax_out.set_ylim([-0.05, 1.05])
    ax_out.set_xlim([times[0], times[-1]])
    ax_out.axhline(0, color="k", linewidth=0.5)

    # align x‑limits
    ax_heat.set_xlim(ax_out.get_xlim())
    return im


# ───────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model, hp = load_model(MODEL_DIR)

    # make sure timing params match our figure spec
    hp.update({
        "dt": DT_MS,
        "tone_dur": TONE_DUR_MS,
        "std_dur": STD_DUR_MS,
        "delay": DELAY_MS,
    })

    fig, axes = plt.subplots(2, len(COMP_DURS_MS), figsize=(12, 6),
                             gridspec_kw={"height_ratios": [3, 1], "hspace": 0.25})

    for idx, comp in enumerate(COMP_DURS_MS):
        im = plot_condition(axes[0, idx], axes[1, idx], model, hp, comp)

    # shared colour‑bar
    cbar = fig.colorbar(im, ax=axes[0, :].tolist(), orientation="vertical", shrink=0.8)
    cbar.set_label("Normalised activity", rotation=270, labelpad=15)

    fig.suptitle("Single‑trial activations and network outputs", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(SAVE_PATH, dpi=300)
    print(f"Figure saved to {SAVE_PATH}")
    plt.show()
