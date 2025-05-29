import numpy as np
import torch

from task import Trial

def generate_case_batch(hp, comp_dur_val, std_order_val, batch_size):
    """
    Build a batch of trials all with the same comp_dur and std_order.
    Returns x, respond flags, and offset/onset indices for intervals.
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
    int1_ons_t    = torch.tensor(trial.int1_ons, dtype=torch.long)
    int1_offs_t    = torch.tensor(trial.int1_offs, dtype=torch.long)
    int2_ons_t    = torch.tensor(trial.int2_ons, dtype=torch.long)
    int2_offs_t    = torch.tensor(trial.int2_offs, dtype=torch.long)
    return x_tensor, responds_mask, int1_ons_t, int1_offs_t, int2_ons_t, int2_offs_t