import json
import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from rnn_model import RNNModel
from task import generate_trials

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-6

# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def load_model(model_dir: str) -> Tuple[RNNModel, dict]:
    """Return a model on DEVICE and the associated hp dict."""
    with open(os.path.join(model_dir, "hp.json"), "r") as f:
        hp = json.load(f)
    model = RNNModel(hp)
    state = torch.load(os.path.join(model_dir, "checkpoint.pt"), map_location="cpu")
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model, hp


def make_validation_batch(hp: dict, size: int):
    """Generate a fresh batch without input noise and return tensors + meta."""
    hp_val = hp.copy()
    hp_val["dataset_size"] = size
    trial = generate_trials(hp_val["rule"], hp_val, mode="random", noise_on=True)
    x = torch.tensor(trial.x, dtype=torch.float32)
    decision_mask = torch.tensor(trial.c_mask[:, :, 0] == 2, dtype=torch.bool)
    labels = torch.tensor(trial.respond, dtype=torch.bool)
    return x, decision_mask, labels

# -----------------------------------------------------------------------------
# Selectivity & activation computation
# -----------------------------------------------------------------------------

def selectivity_by_decision(model: RNNModel, x: torch.Tensor, mask: torch.BoolTensor,
                             labels: torch.BoolTensor):
    """Return SI per hidden unit and mean activities for each class."""
    with torch.no_grad():
        h = model.rnn(x.to(DEVICE))            # (B, T, N)
        h = h.cpu()

    mask_f = mask.unsqueeze(-1).float()                      # (B, T, 1)
    dec_counts = mask_f.sum(dim=1)                          # (B, 1)
    h_dec = (h * mask_f).sum(dim=1) / (dec_counts + EPS)     # (B, N)

    pos = labels
    neg = ~labels
    mean_pos = h_dec[pos].mean(dim=0)
    mean_neg = h_dec[neg].mean(dim=0)

    si = (mean_pos - mean_neg) / (mean_pos + mean_neg + EPS)
    return si.numpy(), mean_pos.numpy(), mean_neg.numpy()

# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------

def plot_si(si: np.ndarray, out_path: str, model_name: str, top: int = 20):
    idx = np.argsort(-np.abs(si))
    sorted_si = si[idx]
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(sorted_si)), sorted_si, width=1.0)
    plt.xlabel("Neuron (sorted by |SI|)")
    plt.ylabel("Selectivity Index")
    plt.title(f"Hidden-unit decision selectivity ({model_name})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Top {top} selective neurons:")
    for rank in range(top):
        j = idx[rank]
        print(f" #{rank+1:2d} neuron {j:3d}: SI = {si[j]:+.3f}")


def plot_activation_heatmap(mean_pos: np.ndarray, mean_neg: np.ndarray,
                            si: np.ndarray, out_path: str, model_name: str):
    idx = np.argsort(-np.abs(si))
    data = np.vstack([mean_pos[idx], mean_neg[idx]])  # (2, N)

    fig, ax = plt.subplots(figsize=(10, 2))
    vmax = np.max(np.abs(data))
    im = ax.imshow(data, aspect='auto', cmap='seismic', vmin=0, vmax=vmax)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Respond1', 'Respond0'])
    ax.set_xticks([])
    ax.set_title(f'Mean activation during decision window ({model_name})')
    cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.3)
    cbar.set_label('Activation')
    heatmap_path = out_path.replace('.png', f'_{model_name}_activation_heatmap.png')
    fig.tight_layout()
    fig.savefig(heatmap_path, dpi=300)
    plt.close(fig)
    print(f"Heatmap saved to {heatmap_path}")

# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def main():
    model_dir = os.path.join(BASE_DIR, "models", "easy_trained")
    dataset_size = 1000
    top          = 20
    out          = "selectivityEasyTask1.png"
    model_name   = os.path.basename(model_dir.rstrip(os.sep))
    base_name    = os.path.splitext(out)[0]
    out_filename = f"{base_name}_{model_name}.png"
    out_path     = os.path.join(BASE_DIR, out_filename)

    model, hp = load_model(model_dir)
    x, mask, labels = make_validation_batch(hp, dataset_size)

    si, mean_pos, mean_neg = selectivity_by_decision(model, x, mask, labels)

    # ─── Save selectivity & activation arrays for downstream weight analysis ───
    results_fname = f"results_{model_name}.npz"
    results_path  = os.path.join(BASE_DIR, results_fname)
    np.savez(
        results_path,
        selectivity=si,
        activation=(mean_pos + mean_neg) / 2.0    # or save both mean_pos & mean_neg separately if you like
    )
    print(f"[INFO] Saved analysis results to {results_path}")
    
    plot_si(si, out_path, model_name, top=top)
    plot_activation_heatmap(mean_pos, mean_neg, si, out_path, model_name)

    # ─── Print activations for the top‐N selective neurons ───────────────
    idx = np.argsort(-np.abs(si))  # sort by absolute SI descending
    print(f"\nTop {top} neurons with their mean activations and SI:")
    for rank in range(top):
        j = idx[rank]
        print(f" #{rank+1:2d} neuron {j:3d}: "
              f"mean_pos = {mean_pos[j]:.5f}, "
              f"mean_neg = {mean_neg[j]:.5f}, "
              f"SI = {si[j]:+.3f}")
    # ──────────────────────────────────────────────────────────────────────

    # ------------------ Confusion matrix ------------------
    with torch.no_grad():
        y_hat = model(x.to(DEVICE))
    out_tensor = y_hat.cpu().squeeze(-1)
    preds = ((out_tensor > 0) & mask).any(dim=1).numpy()
    trues = labels.numpy()
    TP = np.logical_and(preds,  trues).sum()
    TN = np.logical_and(~preds, ~trues).sum()
    FP = np.logical_and(preds, ~trues).sum()
    FN = np.logical_and(~preds,  trues).sum()
    print("Confusion matrix:")
    print(f"  TP: {TP}  FP: {FP}")
    print(f"  FN: {FN}  TN: {TN}")

    print(f"All plots saved based on base name '{out_path}'")


if __name__ == "__main__":
    main()

