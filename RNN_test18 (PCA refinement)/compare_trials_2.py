import numpy as np
import torch
from pathlib import Path

# ─── Local imports ─────────────────────────────────────────────────────────
import train
from task2          import generate_trials as gen_trials_task
from task_original          import generate_trials as gen_trials_task_original
from failure_count import generate_case_batch
import analysis

# ─── Globals ───────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RULE   = "Interval_Discrim"
HP     = train.get_default_hp()
ORDER = 0
COMP_DUR_VAL = 220  # comparison duration in ms
HP.update({"comp_step": 20, "dataset_size": 1, "rule": RULE, "comp_dur_val": COMP_DUR_VAL, 'std_order_val' : ORDER})  # clarity
DT     = HP["dt"]

DURATIONS = [HP["std_dur"] + o * HP["comp_step"]
             for o in (-4, -3, -2, -1, 1, 2, 3, 4)]

trial2 = gen_trials_task(RULE,HP, 'random', noise_on=False)
trial = gen_trials_task_original(RULE, HP, 'random', noise_on=True)
times_ms = np.arange(trial.tdim) * trial.dt

x, resp, int2 = generate_case_batch(HP, COMP_DUR_VAL, ORDER, 1)
#print(x)
print(int2)
print(trial.int2_ons)
print(trial2.int2_ons)
import matplotlib.pyplot as plt

t = np.arange(x.shape[1]) * HP['dt']
plt.plot(t, x[0,:,0].cpu().numpy(), label='failure_count')
plt.plot(t, trial.x[0,:,0],              label='task_original')
plt.plot(t, trial2.x[0,:,0],              label='task')
# ─── draw int2 onsets ────────────────────────────────────────────────
# failure_count int2 is already in time‐steps:
t_int2_fc   = int2.item() * HP['dt']

# for the Trial objects, int2_ons is a step‐index too:
t_int2_orig = int(trial.int2_ons[0])  * trial.dt
t_int2_task = int(trial2.int2_ons[0]) * trial2.dt

plt.axvline(t_int2_fc,   color='C0', linestyle='--',
            label=f'FC int2 @ {t_int2_fc} ms')
plt.axvline(t_int2_orig, color='C1', linestyle='--',
            label=f'Orig int2 @ {t_int2_orig} ms')
plt.axvline(t_int2_task, color='C2', linestyle='--',
            label=f'Task2 int2 @ {t_int2_task} ms')

plt.xlabel('Time (ms)')
plt.legend()
plt.show()