import os
import torch
import numpy as np
from failure_count import load_model        # loads RNNModel + hp.json
from generate_trials import generate_case_batch
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Paths
BASE_DIR      = os.path.abspath(os.path.dirname(__file__))
MODEL_DIR     = os.path.join(BASE_DIR, "models", "easy_trained")

# 1. Load your pretrained RNN
model, hp = load_model(MODEL_DIR)          # RNNModel with trained weights
rnn     = model.rnn                        # BioRNN cell
device  = next(rnn.parameters()).device

# 2. Define the 8 comparison durations
if 'comp_step' in hp:
    offsets   = np.array([-4, -3, -2, -1, 1, 2, 3, 4])
    durations = hp['std_dur'] + offsets * hp['comp_step']
else:
    durations = np.array([120, 160, 180, 190, 210, 220, 240, 280])

# 3. Collect hidden-state features (aligned to interval 1 offset) and labels
features_list  = []
labels_list    = []
trials_per_dur = 500

for label, dur in enumerate(durations):
    # a) Generate comp-first trials
    x, resp, int1_ons, int1_offs, int2_ons, int2_offs = \
        generate_case_batch(hp, dur, std_order_val=1, batch_size=trials_per_dur)
    x = x.to(device)

    # b) Run the BioRNN and get hidden states
    with torch.no_grad():
        h_all = rnn(x)  # (batch, T, n_rnn)

    # c) Extract hidden state at interval 1 offset
    t_idx     = int1_offs.to(device)                  # (batch,)
    batch_idx = torch.arange(h_all.size(0), device=device)
    h_int1    = h_all[batch_idx, t_idx, :]            # (batch, n_rnn)

    features_list.append(h_int1.cpu().numpy())
    labels_list.append(np.full(trials_per_dur, label, dtype=int))

# d) Stack features and labels
X = np.concatenate(features_list, axis=0)
y = np.concatenate(labels_list, axis=0)

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=0
)

# 5. Fit a multinomial logistic-regression decoder
clf = LogisticRegression(
    solver='lbfgs', max_iter=5000
)
clf.fit(X_train, y_train)

# 6. Evaluate overall decoding performance
y_pred = clf.predict(X_test)
acc    = accuracy_score(y_test, y_pred)
print(f"Overall decoding accuracy (interval 1 offset): {acc*100:.1f}%")

# 7. Track success across the 8 trial types
print("\nPer-duration decoding accuracies:")
for label, dur in enumerate(durations):
    mask  = (y_test == label)
    acc_i = accuracy_score(y_test[mask], y_pred[mask]) if np.any(mask) else 0.0
    print(f"  Duration {dur}: {acc_i*100:.1f}%")

# 8. Misclassification details with percent and predictions
print("\nMisclassification details (true -> predicted bins):")
for label, dur in enumerate(durations):
    mask_total = (y_test == label)
    total = np.sum(mask_total)
    mis_mask = mask_total & (y_pred != label)
    mis_count = np.sum(mis_mask)
    mis_percent = (mis_count / total * 100) if total > 0 else 0.0
    mis_preds = y_pred[mis_mask]
    if mis_count > 0:
        print(f"  True {dur}: {mis_count}/{total} misclassified ({mis_percent:.1f}%), predictions: {mis_preds.tolist()}")
    else:
        print(f"  True {dur}: 0/{total} misclassified (0.0%), no misclassifications")
