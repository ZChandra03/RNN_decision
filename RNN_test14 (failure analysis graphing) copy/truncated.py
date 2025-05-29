import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from rnn_model import RNNModel
from task import generate_trials

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Directory to save figures
OUTPUT_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Utility: load a trained RNNModel
# -----------------------------------------------------------------------------
def load_model(model_dir: str):
    hp_path = os.path.join(model_dir, "hp.json")
    ckpt_path = os.path.join(model_dir, "checkpoint.pt")
    with open(hp_path, "r") as f:
        hp = json.load(f)
    model = RNNModel(hp)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model, hp

# -----------------------------------------------------------------------------
# Truncate recurrent weights to rank-k
# -----------------------------------------------------------------------------
def truncate_wrec(model: RNNModel, U, S, Vt, k):
    W_low = (U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]).astype(np.float32)
    with torch.no_grad():
        model.rnn.w_rec.data.copy_(torch.from_numpy(W_low).to(model.rnn.w_rec.device))

# -----------------------------------------------------------------------------
# Generate one validation batch
# -----------------------------------------------------------------------------
def make_validation_batch(hp: dict, size: int, comp_step: int = 20):
    hp_val = hp.copy()
    hp_val["dataset_size"] = size
    hp_val["comp_step"] = comp_step
    trial = generate_trials(hp_val["rule"], hp_val, mode="random", noise_on=True)
    x = torch.tensor(trial.x, dtype=torch.float32).to(DEVICE)
    decision_mask = torch.tensor(trial.c_mask[:, :, 0] == 2, dtype=torch.bool).to(DEVICE)
    labels = torch.tensor(trial.respond, dtype=torch.bool).to(DEVICE)
    return x, decision_mask, labels

# -----------------------------------------------------------------------------
# Main: evaluate accuracy across different truncation ranks and save plots
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    model_dir = os.path.join(BASE_DIR, "models", "easy_trained")
    batch_size = 1000
    comp_step = 20

    # Load model and compute full SVD of original W_rec
    model, hp = load_model(model_dir)
    w_orig = model.rnn.w_rec.detach().cpu().numpy()
    U, S, Vt = np.linalg.svd(w_orig, full_matrices=False)
    N = w_orig.shape[0]

    # Prepare validation data
    x, mask, labels = make_validation_batch(hp, size=batch_size, comp_step=comp_step)

    # Evaluate accuracy for each k
    ks_full = list(range(1, N + 1))
    ks_small = list(range(1, 51))
    accuracies_full = []
    accuracies_small = []

    for k in ks_full:
        if k < N:
            truncate_wrec(model, U, S, Vt, k)
        else:
            model, _ = load_model(model_dir)
        with torch.no_grad():
            outputs = model(x).squeeze(-1)
        fired = outputs > 0.4
        fired_in = (fired & mask).any(dim=1)
        fired_any = fired.any(dim=1)
        resp = labels
        nonresp = ~labels
        TP = int((resp & fired_in).sum())
        TN = int((nonresp & ~fired_any).sum())
        acc = (TP + TN) / batch_size * 100
        accuracies_full.append(acc)
        if k <= 50:
            accuracies_small.append(acc)
        print(f"k={k}: Accuracy={acc:.2f}%")

    # Plot and save grouped accuracy graphs
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(ks_small, accuracies_small, marker='o')
    axes[0].set_xlabel('Truncation rank k')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Accuracy (k=1 to 50)')
    axes[0].set_ylim(0, 100)
    axes[0].grid(True)

    axes[1].plot(ks_full, accuracies_full, marker='o')
    axes[1].set_xlabel('Truncation rank k')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title(f'Accuracy (k=1 to {N})')
    axes[1].set_ylim(95, 100)
    axes[1].grid(True)

    fig.tight_layout()
    acc_fig_path = os.path.join(OUTPUT_DIR, 'accuracy_truncation.png')
    fig.savefig(acc_fig_path, dpi=300)
    plt.close(fig)
    print(f"→ saved accuracy figure to {acc_fig_path}")

    # Plot and save singular values on log scale
    fig2 = plt.figure()
    plt.plot(range(1, len(S) + 1), S, marker='o')
    plt.xlabel('Index of singular value')
    plt.ylabel('Singular value (log scale)')
    plt.yscale('log')
    plt.title('Singular Values of W_rec')
    plt.grid(True)

    sv_fig_path = os.path.join(OUTPUT_DIR, 'singular_values.png')
    fig2.savefig(sv_fig_path, dpi=300)
    plt.close(fig2)
    print(f"→ saved singular values figure to {sv_fig_path}")
