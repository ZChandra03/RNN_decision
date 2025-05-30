import numpy as np
import torch
from pathlib import Path

# Local imports
import train
from task_original import generate_trials as gen_trials_task_original
# from task import generate_trials as gen_trials_task_original

# Globals
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RULE = 'Interval_Discrim'
HP = train.get_default_hp()
COMP_DUR_VAL = 280  # comparison duration in ms
HP.update({
    'dataset_size': 1,
    'rule': RULE,
    'comp_dur_val': COMP_DUR_VAL,
    'std_order_val': 0,  # standard first
})
DT = HP['dt']

# Generate original task trial
trial = gen_trials_task_original(RULE, HP, 'random', noise_on=True)

# Time axis
times_ms = np.arange(trial.tdim) * trial.dt

import matplotlib.pyplot as plt

# --- Plot input time series, cost mask, and target output in one figure ---
plt.figure(figsize=(10, 6))
plt.plot(times_ms, trial.x[0, :, 0], label='Input (x)', linewidth=2)
#plt.plot(times_ms, trial.c_mask[0, :, 0], label='Cost Mask', linestyle='--', alpha=0.7)
plt.plot(times_ms, trial.y[0, :, 0], label='Target Output (y)', linewidth=2)

plt.xlabel('Time (ms)')
plt.ylabel('Signal / Mask / Output')
plt.title('Input, Cost Mask, and Target Output')
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()
