import os
import torch
import numpy as np
from failure_count import load_model        # loads RNNModel + hp.json
from generate_trials import generate_case_batch
from train import get_default_hp
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

BASE_DIR    = os.path.abspath(os.path.dirname(__file__))
MODEL_DIR   = os.path.join(BASE_DIR, "models", "easy_trained")

# 1. Load your pretrained RNN
model, hp = load_model(MODEL_DIR)          # RNNModel with trained weights
rnn = model.rnn                            # BioRNN cell
device = next(rnn.parameters()).device

# 2. Define the 8 comparison durations
if 'comp_step' in hp:
    offsets   = np.array([-4, -3, -2, -1, 1, 2, 3, 4])
    durations = hp['std_dur'] + offsets * hp['comp_step']
else:
    durations = np.array([120, 160, 180, 190, 210, 220, 240, 280])

# 3. Collect hidden‐state features and integer labels
features_list = []
labels_list   = []
trials_per_dur = 500

for label, dur in enumerate(durations):
    # a) Generate a batch of comp‐first trials
    x, resp, int1_ons, int1_offs, int2_ons, int2_offs = \
        generate_case_batch(hp, dur, std_order_val=1, batch_size=trials_per_dur)
    x = x.to(device)

    # b) Run the BioRNN and grab hidden states
    with torch.no_grad():
        h_all = rnn(x)                    # (batch, T, n_rnn)

    # c) Extract h at t = int2_offs for each trial
    t_idx = int2_offs.to(device)         # shape (batch,)
    batch_idx = torch.arange(h_all.size(0), device=device)
    h_final = h_all[batch_idx, t_idx, :] # (batch, n_rnn)

    features_list.append(h_final.cpu().numpy())
    labels_list.append(np.full(trials_per_dur, label, dtype=int))

# d) Stack into feature matrix X and label vector y
X = np.concatenate(features_list, axis=0)
y = np.concatenate(labels_list, axis=0)

# 4. Train‐test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=0
)

# 5. Fit a multinomial logistic‐regression decoder
clf = LogisticRegression(
    solver='lbfgs',
    max_iter=5000
)
clf.fit(X_train, y_train)

# 6. Evaluate
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Decoding accuracy over 8 comparison durations: {acc*100:.1f}%")
