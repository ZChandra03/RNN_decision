#!/usr/bin/env python3
# save_fails.py  – collect three bundles:
#   • 100 successes  (d=180, order=0)
#   • 100 failures   (d=180, order=0)
#   • 100 successes  (d=180, order=1)
# and save (x, h, y_hat) for each.
# -----------------------------------------------------------------------
import os, json, torch, numpy as np
from rnn_model import RNNModel
from generate_trials import generate_case_batch

# ─── Paths ─────────────────────────────────────────────────────────────
BASE_DIR  = os.path.abspath(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models", "easy_trained")
OUT_DIR   = os.path.join(BASE_DIR, "failure_conditions")
os.makedirs(OUT_DIR, exist_ok=True)

# ─── Hyper-parameters ──────────────────────────────────────────────────
BATCH_SIZE   = 2048     # adjust for your GPU/CPU RAM
THRESHOLD    = 0.40     # from your earlier sweep
TARGET_SUCC  = 100
TARGET_FAIL  = 100
COMP_DUR     = 180      # ms
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Helper: load model ────────────────────────────────────────────────
def load_model(model_dir):
    with open(os.path.join(model_dir, "hp.json"), "r") as f:
        hp = json.load(f)
    model = RNNModel(hp)
    ckpt  = torch.load(os.path.join(model_dir, "checkpoint.pt"),
                       map_location="cpu")
    model.load_state_dict(ckpt)
    model.to(DEVICE).eval()
    return model, hp

# ─── Helper: classification check ──────────────────────────────────────
def trial_correct(y_hat_t, respond_flag, int2_off, thr):
    fired     = (y_hat_t > thr)
    fired_any = fired.any().item()
    fired_in  = fired[int2_off:].any().item()
    fired_out = fired[:int2_off].any().item()

    if respond_flag:                       # should fire
        return fired_in and not fired_out  # TP
    else:                                  # should stay silent
        return not fired_any               # TN

# ─── Core collector ────────────────────────────────────────────────────
def collect(model, hp, order, want_successes, want_failures):
    succ_x, succ_h, succ_y = [], [], []
    fail_x, fail_h, fail_y = [], [], []

    while (len(succ_x) < want_successes) or (len(fail_x) < want_failures):
        x, respond, _, _, _, int2_off = generate_case_batch(
            hp, COMP_DUR, order, BATCH_SIZE)

        with torch.no_grad():
            h_all = model.rnn(x.to(DEVICE))               # (B,T,N)
            y_hat = model.output(h_all).cpu().squeeze(-1) # (B,T)
            h_all = h_all.cpu()

        for i in range(BATCH_SIZE):
            ok = trial_correct(y_hat[i], respond[i].item(),
                               int2_off[i].item(), THRESHOLD)

            if ok and len(succ_x) < want_successes:
                succ_x.append(x[i].cpu())
                succ_h.append(h_all[i])
                succ_y.append(y_hat[i])
            elif (not ok) and len(fail_x) < want_failures:
                fail_x.append(x[i].cpu())
                fail_h.append(h_all[i])
                fail_y.append(y_hat[i])

            if (len(succ_x) >= want_successes and
                len(fail_x) >= want_failures):
                break
    return (succ_x, succ_h, succ_y), (fail_x, fail_h, fail_y)

# ─── Main ──────────────────────────────────────────────────────────────
def main():
    model, hp = load_model(MODEL_DIR)

    # ----- order 0: collect successes & failures ----------------------
    (sx0, sh0, sy0), (fx0, fh0, fy0) = collect(
        model, hp, order=0,
        want_successes=TARGET_SUCC,
        want_failures =TARGET_FAIL)

    torch.save({'x': torch.stack(sx0),
                'h': torch.stack(sh0),
                'y_hat': torch.stack(sy0)},
               os.path.join(OUT_DIR, "d180_order0_successes.pt"))
    torch.save({'x': torch.stack(fx0),
                'h': torch.stack(fh0),
                'y_hat': torch.stack(fy0)},
               os.path.join(OUT_DIR, "d180_order0_failures.pt"))
    print("✓ Saved order-0 successes & failures")

    # ----- order 1: collect successes only ----------------------------
    (sx1, sh1, sy1), _ = collect(
        model, hp, order=1,
        want_successes=TARGET_SUCC,
        want_failures =0)

    torch.save({'x': torch.stack(sx1),
                'h': torch.stack(sh1),
                'y_hat': torch.stack(sy1)},
               os.path.join(OUT_DIR, "d180_order1_successes.pt"))
    print("✓ Saved order-1 successes")

    print("\nSummary:")
    print("  order 0  successes:", len(sx0),
          " | failures:", len(fx0))
    print("  order 1  successes:", len(sx1))

# ─── Run ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
