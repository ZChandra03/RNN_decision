import os
import json
import numpy as np
import torch

from rnn_model import RNNModel
from task import Trial

# ─── Configuration ────────────────────────────────────────────────────────────
BASE_DIR    = os.path.abspath(os.path.dirname(__file__))
MODEL_DIR   = os.path.join(BASE_DIR, "models", "easy_trained")
OUTPUT_DIR  = os.path.join(BASE_DIR, "failure_conditions")
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


def generate_case_batch(hp, comp_dur_val, std_order_val, batch_size):
    """
    Build a batch of trials all with the same comp_dur and std_order.
    Returns x, respond flags, and int2 onset indices.
    """
    dt       = hp["dt"]
    std_dur  = hp["std_dur"]
    tone_dur = hp["tone_dur"]
    delay    = hp["delay"]

    # time‐step calculations
    base_ons    = int(500 / dt)
    tone_steps  = int(tone_dur / dt)
    delay_steps = int(delay / dt)
    std_steps   = int(std_dur / dt)
    comp_steps  = int(comp_dur_val / dt)

    # compute offsets and response logic per order
    if std_order_val == 0:
        #standard first
        int1_offs    = base_ons + std_steps
        int2_ons     = int1_offs + tone_steps + delay_steps
        int2_offs    = int2_ons + comp_steps
        respond_flag = comp_dur_val > std_dur
    else:
        #comparison first
        int1_offs    = base_ons + comp_steps
        int2_ons     = int1_offs + tone_steps + delay_steps
        int2_offs    = int2_ons + std_steps
        respond_flag = std_dur > comp_dur_val

    # determine global tdim
    tdim = int2_offs + tone_steps + int(2000 / dt)

    # instantiate Trial
    config = hp.copy()
    config["dataset_size"] = batch_size
    trial = Trial(config, tdim, batch_size)

    # assign trial parameters
    trial.std_dur   = std_dur
    trial.tone_dur  = tone_dur
    trial.delay     = np.full(batch_size, delay, dtype=int)
    trial.std_order = np.full(batch_size, std_order_val, dtype=int)
    trial.comp_dur  = np.full(batch_size, comp_dur_val, dtype=int)

    trial.int1_ons  = np.full(batch_size, base_ons, dtype=int)
    trial.int1_offs = np.full(batch_size, int1_offs, dtype=int)
    trial.int2_ons  = np.full(batch_size, int2_ons, dtype=int)
    trial.int2_offs = np.full(batch_size, int2_offs, dtype=int)
    trial.respond   = np.full(batch_size, respond_flag, dtype=int)

    # build inputs
    trial.x.fill(0)
    trial.add("interval_input", ons=trial.int1_ons, offs=trial.int1_offs)
    trial.add("interval_input", ons=trial.int2_ons, offs=trial.int2_offs)
    trial.add_x_noise()

    x_tensor      = torch.tensor(trial.x, dtype=torch.float32)
    responds_mask = torch.tensor(trial.respond, dtype=torch.bool)
    int2_ons_t    = torch.tensor(trial.int2_ons, dtype=torch.long)
    int2_offs_t    = torch.tensor(trial.int2_offs, dtype=torch.long)
    return x_tensor, responds_mask, int2_ons_t, int2_offs_t


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
        offsets   = [-4, -3, -2, -1, 1, 2, 3, 4]
        durations = [hp["std_dur"] + o * hp["comp_step"] for o in offsets]
    else:
        durations = [120, 160, 180, 190, 210, 220, 240, 280]
    orders = [0, 1]

    # collect counts for each of the 16 cases
    results = []
    for d in durations:
        for o in orders:
            x, resp, int2 = generate_case_batch(hp, d, o, BATCH_SIZE)
            counts = classify_counts(model, x, resp, int2, THRESHOLD)
            results.append([counts[0], counts[1], counts[2], counts[3]])
            print(f"d={d}, order={o} → TP,FP,FN,TN = {counts}")

    # convert to numpy array and save
    matrix = np.array(results, dtype=int)
    out_path = os.path.join(OUTPUT_DIR, "confusion_counts.npy")
    np.save(out_path, matrix)
    print(f"Saved 16×4 confusion matrix to {out_path}")
