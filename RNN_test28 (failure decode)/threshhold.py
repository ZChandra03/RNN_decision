# … existing imports …
import matplotlib.pyplot as plt      # ← add
import os
import json
import numpy as np
import torch

from rnn_model import RNNModel

# ─── Configuration ────────────────────────────────────────────────────────────
BASE_DIR    = os.path.abspath(os.path.dirname(__file__))
MODEL_DIR   = os.path.join(BASE_DIR, "models", "easy_trained")
OUTPUT_DIR  = os.path.join(BASE_DIR, "failure_conditions")
from generate_trials import generate_case_batch
BATCH_SIZE  = 1000      # trials per case

os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ─── New configuration ────────────────────────────────────────────────────────
THRESH_RANGE = np.linspace(0.2, 0.8, 20)    # 0.00, 0.01, … , 1.00

def load_model(model_dir):
    with open(os.path.join(model_dir, "hp.json"), "r") as f:
        hp = json.load(f)
    model = RNNModel(hp)
    ckpt = torch.load(os.path.join(model_dir, "checkpoint.pt"), map_location="cpu")
    model.load_state_dict(ckpt)
    model.to(DEVICE).eval()
    return model, hp

def classify_counts(model, x, responds, int2_ons, threshold):
    with torch.no_grad():
        y_hat = model(x.to(DEVICE)).cpu().squeeze(-1)
    fired = y_hat > threshold
    B, T = fired.shape
    t_idx = torch.arange(T).unsqueeze(0)
    epoch_mask = t_idx >= int2_ons.unsqueeze(1)

    fired_in  = (fired & epoch_mask).any(dim=1)
    fired_out = (fired & ~epoch_mask).any(dim=1)
    fired_any = fired.any(dim=1)

    resp    = responds
    nonresp = ~resp

    TP = (resp & fired_in & ~fired_out).sum().item()
    FP = (nonresp & fired_any).sum().item()
    FN = (resp & (~fired_in | fired_out)).sum().item()
    TN = (nonresp & ~fired_any).sum().item()
    return TP, FP, FN, TN

# ─── Helper to accumulate counts across all cases ─────────────────────────────
def run_all_cases(model, hp, durations, orders, batch_size, threshold):
    total_TP = total_FP = total_FN = total_TN = 0
    for d in durations:
        for o in orders:
            x, resp, *_ , int2_ons_t, _  = generate_case_batch(
                    hp, d, o, batch_size)
            TP, FP, FN, TN = classify_counts(
                    model, x, resp, int2_ons_t, threshold)
            total_TP += TP; total_FP += FP
            total_FN += FN; total_TN += TN
    return total_TP, total_FP, total_FN, total_TN

# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model, hp = load_model(MODEL_DIR)

    # comparison durations & orders (unchanged from your script)
    if "comp_step" in hp:
        offsets   = [-4, -3, -2, -1, 1, 2, 3, 4]
        durations = [hp["std_dur"] + o * hp["comp_step"] for o in offsets]
    else:
        durations = [120, 160, 180, 190, 210, 220, 240, 280]
    orders = [0, 1]

    accuracies = []
    for thr in THRESH_RANGE:
        TP, FP, FN, TN = run_all_cases(model, hp, durations, orders,
                                       BATCH_SIZE, thr)
        acc = (TP + TN) / (TP + FP + FN + TN)
        accuracies.append(acc)

    # ── Plot & save ──
    plt.figure()
    plt.plot(THRESH_RANGE, accuracies, linewidth=2)
    plt.xlabel("Decision threshold")
    plt.ylabel("Overall accuracy")
    plt.title("Threshold sweep")
    plt.grid(True)
    sweep_png = os.path.join(OUTPUT_DIR, "threshold_sweep.png")
    plt.savefig(sweep_png, dpi=150, bbox_inches="tight")
    print(f"Sweep complete → plot saved to {sweep_png}")
