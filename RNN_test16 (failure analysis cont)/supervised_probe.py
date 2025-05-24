#!/usr/bin/env python3
"""
supervised_probe.py

Train linear probes (logistic regression) on RNN hidden-state averages to separate:
  1) TP vs FN (within 'should-respond' trials)
  2) TN vs FP (within 'should-not-respond' trials)

Key changes to handle class imbalance:
- Use `class_weight='balanced'` for logistic regression.
- Compute and save precision, recall, F1-score, ROC AUC, and balanced accuracy.
- Optionally down-sample majority class for a balanced train/test split demonstration.

Usage:
    python supervised_probe.py

Outputs:
  - probe_metrics.txt: detailed metrics for each probe
  - probe_axis_resp.npy: learned weight vector (TP vs FN)
  - probe_axis_nonresp.npy: learned weight vector (TN vs FP)
  - hist_tp_fn.png: histogram of projection scores for TP/FN
  - hist_tn_fp.png: histogram of projection scores for TN/FP
"""
import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score,
    balanced_accuracy_score, precision_recall_fscore_support
)

from rnn_model import RNNModel
from failure_count import generate_case_batch

# ────────────────────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.abspath(os.path.dirname(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "models", "easy_trained")
OUTPUT_DIR = os.path.join(BASE_DIR, "supervised_probe")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 5000
THRESHOLD  = 0.4
WINDOW_MS  = 200
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────
def load_model(model_dir):
    with open(os.path.join(model_dir, "hp.json"), "r") as f:
        hp = json.load(f)
    model = RNNModel(hp)
    ckpt = torch.load(os.path.join(model_dir, "checkpoint.pt"), map_location="cpu")
    model.load_state_dict(ckpt)
    model.to(DEVICE).eval()
    return model, hp


def average_hidden_window(h_seq, int2_ons, dt, window_ms):
    B, T, N = h_seq.shape
    steps = int(window_ms / dt)
    h_avg = torch.zeros(B, N, device=h_seq.device)
    for i in range(B):
        start = int2_ons[i].item()
        end = min(start + steps, T)
        h_avg[i] = h_seq[i, start:end].mean(dim=0)
    return h_avg.cpu().numpy()


def balance_classes(X, y):
    """Down-sample majority class to balance the dataset."""
    classes, counts = np.unique(y, return_counts=True)
    min_count = counts.min()
    X_bal_list, y_bal_list = [], []
    for cls in classes:
        idx = np.where(y == cls)[0]
        sel = np.random.choice(idx, size=min_count, replace=False)
        X_bal_list.append(X[sel])
        y_bal_list.append(y[sel])
    X_bal = np.vstack(X_bal_list)
    y_bal = np.hstack(y_bal_list)
    return X_bal, y_bal

# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────
def main():
    model, hp = load_model(MODEL_DIR)
    dt = hp.get("dt", 10)

    # durations showing failures
    comp_durs = [180, 220]
    orders    = [0, 1]

    # collect hidden averages and labels
    X_list, should_respond, fired_any, fired_in = [], [], [], []

    for d in comp_durs:
        for o in orders:
            x, responds, int2_ons = generate_case_batch(hp, d, o, BATCH_SIZE)
            with torch.no_grad():
                h_seq = model.rnn(x.to(DEVICE))
            yhat = model(x.to(DEVICE)).cpu().squeeze(-1)
            fired = yhat > THRESHOLD
            fired_any.append(fired.any(1).numpy())
            fired_in.append((fired & (torch.arange(yhat.shape[1])[None] >= int2_ons.unsqueeze(1))).any(1).numpy())
            should_respond.append(responds.numpy())
            h_avg = average_hidden_window(h_seq, int2_ons, dt, WINDOW_MS)
            X_list.append(h_avg)

    X = np.vstack(X_list)
    should_respond = np.concatenate(should_respond)
    fired_any      = np.concatenate(fired_any)
    fired_in       = np.concatenate(fired_in)

    metrics = []

    # --- Probe 1: TP vs FN (respond True) ---
    mask_resp = should_respond.astype(bool)
    X_resp    = X[mask_resp]
    y_resp    = fired_in[mask_resp].astype(int)  # 1=TP, 0=FN

    # Balanced down-sample for demonstration
    X_resp_bal, y_resp_bal = balance_classes(X_resp, y_resp)
    Xtr, Xte, ytr, yte = train_test_split(
        X_resp_bal, y_resp_bal, stratify=y_resp_bal, test_size=0.3, random_state=0
    )
    clf_resp = LogisticRegression(class_weight='balanced', max_iter=200).fit(Xtr, ytr)
    y_pred_resp = clf_resp.predict(Xte)
    scores_resp = clf_resp.decision_function(Xte)

    acc_resp      = clf_resp.score(Xte, yte)
    bal_acc_resp  = balanced_accuracy_score(yte, y_pred_resp)
    roc_auc_resp  = roc_auc_score(yte, scores_resp)
    report_resp   = classification_report(yte, y_pred_resp, target_names=['FN','TP'])

    np.save(os.path.join(OUTPUT_DIR, 'probe_axis_resp.npy'), clf_resp.coef_.flatten())
    metrics.append((
        'TP vs FN', acc_resp, bal_acc_resp, roc_auc_resp
    ))

    # Histogram of decision scores
    plt.figure(figsize=(6,4))
    plt.hist(scores_resp[yte==1], bins=50, alpha=0.7, label='TP')
    plt.hist(scores_resp[yte==0], bins=50, alpha=0.7, label='FN')
    plt.title(f'Probe TP vs FN: acc={acc_resp:.3f}, bal_acc={bal_acc_resp:.3f}, AUC={roc_auc_resp:.3f}')
    plt.xlabel('Decision score'); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'hist_tp_fn.png'), dpi=300)
    plt.close()

    # Save report
    with open(os.path.join(OUTPUT_DIR, 'report_tp_fn.txt'), 'w') as f:
        f.write(report_resp)

    # --- Probe 2: TN vs FP (respond False) ---
    mask_non = ~should_respond.astype(bool)
    X_non    = X[mask_non]
    y_non    = fired_any[mask_non].astype(int)  # 1=FP, 0=TN

    # Balanced down-sample
    X_non_bal, y_non_bal = balance_classes(X_non, y_non)
    Xtr2, Xte2, ytr2, yte2 = train_test_split(
        X_non_bal, y_non_bal, stratify=y_non_bal, test_size=0.3, random_state=0
    )
    clf_non = LogisticRegression(class_weight='balanced', max_iter=200).fit(Xtr2, ytr2)
    y_pred_non = clf_non.predict(Xte2)
    scores_non = clf_non.decision_function(Xte2)

    acc_non      = clf_non.score(Xte2, yte2)
    bal_acc_non  = balanced_accuracy_score(yte2, y_pred_non)
    roc_auc_non  = roc_auc_score(yte2, scores_non)
    report_non   = classification_report(yte2, y_pred_non, target_names=['TN','FP'])

    np.save(os.path.join(OUTPUT_DIR, 'probe_axis_nonresp.npy'), clf_non.coef_.flatten())
    metrics.append((
        'TN vs FP', acc_non, bal_acc_non, roc_auc_non
    ))

    plt.figure(figsize=(6,4))
    plt.hist(scores_non[yte2==0], bins=50, alpha=0.7, label='TN')
    plt.hist(scores_non[yte2==1], bins=50, alpha=0.7, label='FP')
    plt.title(f'Probe TN vs FP: acc={acc_non:.3f}, bal_acc={bal_acc_non:.3f}, AUC={roc_auc_non:.3f}')
    plt.xlabel('Decision score'); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'hist_tn_fp.png'), dpi=300)
    plt.close()

    with open(os.path.join(OUTPUT_DIR, 'report_tn_fp.txt'), 'w') as f:
        f.write(report_non)

    # Save overall metrics
    with open(os.path.join(OUTPUT_DIR, 'probe_metrics.txt'), 'w') as f:
        for name, acc, bal_acc, auc in metrics:
            f.write(f"{name}: acc={acc:.4f}, bal_acc={bal_acc:.4f}, AUC={auc:.4f}\n")

    print(f"Saved probe results and reports to {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
