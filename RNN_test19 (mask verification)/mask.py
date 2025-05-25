import numpy as np
import torch
from pathlib import Path

# Local imports
import train
from task_original import generate_trials as gen_trials_task_original
#from task import generate_trials as gen_trials_task_original
# Globals
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RULE = 'Interval_Discrim'
HP = train.get_default_hp()
COMP_DUR_VAL = 220  # comparison duration in ms
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

# Plot input time series and cost mask
plt.plot(times_ms, trial.x[0, :, 0], label='task_original')
plt.plot(times_ms, trial.c_mask[0, :, 0], label='mask_original', linestyle='--', alpha=0.7)

# Draw Int2 onset
t_int2_on = int(trial.int2_ons[0]) * trial.dt
plt.axvline(t_int2_on, color='C0', linestyle='--', label=f'Int2 onset @ {t_int2_on} ms')

# Draw Int2 offset
t_int2_off = int(trial.int2_offs[0]) * trial.dt
plt.axvline(t_int2_off, color='C1', linestyle='-.', label=f'Int2 offset @ {t_int2_off} ms')

plt.xlabel('Time (ms)')
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()
