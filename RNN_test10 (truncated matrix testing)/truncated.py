import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from rnn_model import RNNModel
from task import generate_trials

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
def truncate_wrec(model: RNNModel, U: np.ndarray, S: np.ndarray, Vt: np.ndarray, k: int):
    W_low = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    with torch.no_grad():
        model.rnn.w_rec.data.copy_(torch.from_numpy(W_low).to(DEVICE))

# -----------------------------------------------------------------------------
# Generate one validation batch
# -----------------------------------------------------------------------------
def make_validation_batch(hp: dict, size: int = 256, comp_step: int = 20):
    hp_val = hp.copy()
    hp_val["dataset_size"] = size
    hp_val["comp_step"] = comp_step
    trial = generate_trials(hp_val["rule"], hp_val, mode="random", noise_on=True)
    x = torch.tensor(trial.x, dtype=torch.float32).to(DEVICE)
    mask = torch.tensor(trial.c_mask[:, :, 0] == 2, dtype=torch.bool).to(DEVICE)
    labels = torch.tensor(trial.respond, dtype=torch.bool).to(DEVICE)
    return x, mask, labels

# -----------------------------------------------------------------------------
# Main: evaluate accuracy across different truncation ranks
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Configuration
    model_dir = os.path.join(BASE_DIR, "models", "easy_trained")
    batch_size = 1000

    # Load model and compute full SVD of original W_rec
    model, hp = load_model(model_dir)
    w_orig = model.rnn.w_rec.detach().cpu().numpy()
    U, S, Vt = np.linalg.svd(w_orig, full_matrices=False)
    N = w_orig.shape[0]

    # Prepare validation data
    x, mask, labels = make_validation_batch(hp, size=batch_size)

    # Evaluate accuracy for each k
    ks = list(range(30, N + 1))
    accuracies = []

    for k in ks:
        # Reload original weights to model before truncation
        truncate_wrec(model, U, S, Vt, k)

        with torch.no_grad():
            outputs = model(x).squeeze(-1)  # (B, T)

        fired = outputs > 0.4          # (B, T)
        # Any spike during decision window
        fired_in = (fired & mask).any(dim=1)
        # Any spike at any time
        fired_any = fired.any(dim=1)

        resp = labels
        nonresp = ~labels
        TP = int((resp & fired_in).sum())
        FN = int((resp & ~fired_in).sum())
        FP = int((nonresp & fired_any).sum())
        TN = int((nonresp & ~fired_any).sum())

        acc = (TP + TN) / batch_size * 100
        accuracies.append(acc)
        print(f"k={k}: Accuracy={acc:.2f}%")

    # Plot k vs. accuracy
    plt.figure()
    plt.plot(ks, accuracies, marker='o')
    plt.xlabel('Truncation rank k')
    plt.ylabel('Accuracy (%)')
    plt.ylim(95, 100)
    plt.title('Accuracy vs. truncation rank')
    plt.grid(True)
    plt.show()
