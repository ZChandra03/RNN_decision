#!/usr/bin/env python3
"""
compare_trials_min.py

Verify that three different code paths create exactly the same respond flag
and int2 onset for the Interval_Discrim task with comp_step = 20 ms:

    1) failure_count.generate_case_batch
    2) task.generate_trials
    3) analysis.make_validation_batch

Run:
    python compare_trials_min.py
"""

import numpy as np
import torch
from pathlib import Path

# ─── Local imports ─────────────────────────────────────────────────────────
import train
from task          import generate_trials as gen_trials_task
from failure_count import generate_case_batch
import analysis

# ─── Globals ───────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RULE   = "Interval_Discrim"
HP     = train.get_default_hp()
HP.update({"comp_step": 20, "dataset_size": 1, "rule": RULE})  # clarity
DT     = HP["dt"]

DURATIONS = [HP["std_dur"] + o * HP["comp_step"]
             for o in (-4, -3, -2, -1, 1, 2, 3, 4)]
ORDERS    = (0, 1)   # 0 = standard first, 1 = comparison first

# ─── MAIN ──────────────────────────────────────────────────────────────────
def main() -> None:
    print("=== TRIAL-GENERATION CONSISTENCY CHECK =============================")
    mismatches = 0

    for dur in DURATIONS:
        for order in ORDERS:
            # ----- failure_count flavour ----------------------------------
            x_fc, resp_fc, int2_fc = generate_case_batch(HP, dur, order, 1)
            resp_fc, int2_fc = bool(resp_fc.item()), int(int2_fc.item())

            # ----- task.generate_trials flavour ---------------------------
            found = False
            for _ in range(1000):
                trial = gen_trials_task(RULE, HP, "random", noise_on=False)
                mask  = (trial.comp_dur == dur) & (trial.std_order == order)
                if mask.any():
                    idx       = int(np.where(mask)[0][0])
                    resp_task = bool(trial.respond[idx])
                    int2_task = int(trial.int2_ons[idx])
                    found = True
                    break
            if not found:
                print(f"[WARN] could not sample (dur={dur}, order={order}) "
                      f"from generate_trials() after 1000 attempts")
                continue

            # Compare FC vs. task ------------------------------------------------
            if resp_fc != resp_task or int2_fc != int2_task:
                mismatches += 1
                print(f"❌ FC vs Task mismatch  dur={dur:3d}  order={order}: "
                      f"FC(resp={resp_fc}, int2={int2_fc})  "
                      f"Task(resp={resp_task}, int2={int2_task})")

            # ----- analysis.make_validation_batch flavour -----------------
            x_an, mask_an, lbl_an, _, _, int2_an = analysis.make_validation_batch(
                HP, size=1, comp_step=20
            )
            resp_an  = bool(lbl_an[0])
            int2_an  = int(int2_an[0])

            # Compare analysis vs. task ----------------------------------------
            if resp_an != resp_task or int2_an != int2_task:
                mismatches += 1
                print(f"❌ Analysis vs Task mismatch  dur={dur:3d}  order={order}: "
                      f"Analysis(resp={resp_an}, int2={int2_an})  "
                      f"Task(resp={resp_task}, int2={int2_task})")

    # ----------------------------------------------------------------------
    if mismatches == 0:
        print("✔ All respond flags and int2 onsets match across "
              "failure_count, task, and analysis.")
    else:
        print(f"⚠ Total mismatches detected: {mismatches}")

# --------------------------------------------------------------------------
if __name__ == "__main__":
    main()
