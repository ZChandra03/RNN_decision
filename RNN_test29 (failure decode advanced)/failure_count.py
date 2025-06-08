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
THRESHOLD   = 0.4      # decision threshold on output unit

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


if __name__ == "__main__":
    model, hp = load_model(MODEL_DIR)

    # define the 8 comparison durations
    if "comp_step" in hp:
        offsets   = [-1]
        #offsets   = [-1, 1]
        durations = [hp["std_dur"] + o * hp["comp_step"] for o in offsets]
    else:
        durations = [120, 160, 180, 190, 210, 220, 240, 280]
    orders = [0]

    # collect counts for each of the 16 cases
    results = []
    for d in durations:
        for o in orders:
            x, resp, int1_ons_t, int1_offs_t, int2_ons_t, int2_offs_t = generate_case_batch(hp, d, o, BATCH_SIZE)
            counts = classify_counts(model, x, resp, int2_offs_t, THRESHOLD)
            results.append([counts[0], counts[1], counts[2], counts[3]])
            print(f"d={d}, order={o} → TP,FP,FN,TN = {counts}")
