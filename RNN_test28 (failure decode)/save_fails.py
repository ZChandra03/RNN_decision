#!/usr/bin/env python3
# collect_d180_order0.py
# ------------------------------------------------------------------------
# Find 100 failures and 100 successes for d = 180 ms, order = 0
# and store (x, h, y_hat) tensors for later analysis.
# ------------------------------------------------------------------------
import os, json, torch
import numpy as np
from rnn_model import RNNModel                                         
from generate_trials import generate_case_batch                        

# ─── Paths ──────────────────────────────────────────────────────────────
BASE_DIR   = os.path.abspath(os.path.dirname(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "models", "easy_trained")
OUT_DIR    = os.path.join(BASE_DIR, "failure_conditions")
os.makedirs(OUT_DIR, exist_ok=True)

# ─── Hyper-parameters ──────────────────────────────────────────────────
BATCH_SIZE     = 2048      # larger -> faster, adjust for GPU RAM
THRESHOLD      = 0.40      # use the value from your sweep
TARGET_SUCC    = 100
TARGET_FAIL    = 100
COMP_DUR       = 180       # ms
STD_ORDER      = 0         # 0 = standard interval first

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Helper: load model exactly as in failure_count.py ─────────────────
def load_model(model_dir):
    with open(os.path.join(model_dir, "hp.json"), "r") as f:
        hp = json.load(f)
    model = RNNModel(hp)
    ckpt  = torch.load(os.path.join(model_dir, "checkpoint.pt"),
                       map_location="cpu")
    model.load_state_dict(ckpt)
    model.to(DEVICE).eval()
    return model, hp

# ─── Helper: decide if a single trial is correct ───────────────────────
def trial_correct(y_hat_t, respond_flag, int2_off, thr):
    """Return True if model behaviour matches expected response."""
    fired     = (y_hat_t > thr)
    fired_any = fired.any().item()
    # only look during response window (= after int2 ends)
    fired_in  = fired[int2_off:].any().item()
    fired_out = fired[:int2_off].any().item()

    if respond_flag:                       # should fire (positive trial)
        return fired_in and not fired_out  # TP
    else:                                  # should stay silent (negative)
        return not fired_any               # TN

# ─── Main collection routine ───────────────────────────────────────────
def main():
    model, hp = load_model(MODEL_DIR)

    succ_x, succ_h, succ_y = [], [], []
    fail_x, fail_h, fail_y = [], [], []

    scanned = 0
    while len(succ_x) < TARGET_SUCC or len(fail_x) < TARGET_FAIL:
        # 1) build a homogeneous batch for this single condition
        x, respond, _, _, _, int2_off = generate_case_batch(
            hp, COMP_DUR, STD_ORDER, BATCH_SIZE)

        # 2) forward pass ONCE to get hidden and output (noise matches)
        with torch.no_grad():
            h_all   = model.rnn(x.to(DEVICE))               # (B,T,N)
            y_hat   = model.output(h_all).cpu().squeeze(-1) # (B,T)
            h_all   = h_all.cpu()                           # back to CPU

        # 3) examine each trial
        for i in range(BATCH_SIZE):
            ok = trial_correct(
                    y_hat[i],
                    respond[i].item(),
                    int2_off[i].item(),
                    THRESHOLD)

            if ok and len(succ_x) < TARGET_SUCC:
                succ_x.append(x[i].cpu())
                succ_h.append(h_all[i])
                succ_y.append(y_hat[i])
            elif (not ok) and len(fail_x) < TARGET_FAIL:
                fail_x.append(x[i].cpu())
                fail_h.append(h_all[i])
                fail_y.append(y_hat[i])

            if len(succ_x) >= TARGET_SUCC and len(fail_x) >= TARGET_FAIL:
                break

        scanned += BATCH_SIZE
        if scanned % (10 * BATCH_SIZE) == 0:
            print(f"Scanned {scanned} trials … "
                  f"{len(succ_x)} successes, {len(fail_x)} failures captured")

    # 4) stack & save
    torch.save({'x': torch.stack(succ_x),
                'h': torch.stack(succ_h),
                'y_hat': torch.stack(succ_y)},
               os.path.join(OUT_DIR, "d180_order0_successes.pt"))
    torch.save({'x': torch.stack(fail_x),
                'h': torch.stack(fail_h),
                'y_hat': torch.stack(fail_y)},
               os.path.join(OUT_DIR, "d180_order0_failures.pt"))

    print("✓ Collection complete")
    print("  successes →", len(succ_x), "saved",
          "\n  failures  →", len(fail_x), "saved")

# ─── Run as script ─────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
