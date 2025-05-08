# analysis.py
"""
Analyze hidden‑unit selectivity in a trained RNNModel for the interval‑discrimination
rule. The script:
  1. Loads hyper‑parameters and checkpoint from a given model directory.
  2. Generates a fresh noise‑free validation set.
  3. Runs the network, harvesting hidden states from the BioRNN core.
  4. Computes a Selectivity Index (SI) per neuron:
        SI_j = (⟨h_j⟩_respond1 − ⟨h_j⟩_respond0) / (⟨h_j⟩_respond1 + ⟨h_j⟩_respond0 + ε)
     where ⟨·⟩ denotes the mean activity in the post‑response window.
  5. Saves a bar plot of all SIs (sorted) and prints the top‑N selective neurons.
  6. Creates a heatmap of the average activations for each decision outcome.
  7. Computes and prints the confusion matrix (TP/FP/FN/TN) for the model’s decisions.

Run:  
    python analysis.py --model_dir models/PT_100_Interval_Discrim_bs_10 \
                       --dataset_size 1000 --top 20 --out selectivity.png
"""
import argparse
import json
import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from rnn_model import RNNModel
from task import generate_trials


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
    trial = generate_trials(hp_val["rule"], hp_val, mode="random", noise_on=False)

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

def plot_si(si: np.ndarray, out_path: str, top: int = 20):
    idx = np.argsort(-np.abs(si))
    sorted_si = si[idx]
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(sorted_si)), sorted_si, width=1.0)
    plt.xlabel("Neuron (sorted by |SI|)")
    plt.ylabel("Selectivity Index")
    plt.title("Hidden‑unit decision selectivity")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Top {top} selective neurons:")
    for rank in range(top):
        j = idx[rank]
        print(f" #{rank+1:2d} neuron {j:3d}: SI = {si[j]:+.3f}")


def plot_activation_heatmap(mean_pos: np.ndarray, mean_neg: np.ndarray,
                            si: np.ndarray, out_path: str):
    idx = np.argsort(-np.abs(si))
    data = np.vstack([mean_pos[idx], mean_neg[idx]])  # (2, N)

    fig, ax = plt.subplots(figsize=(10, 2))
    vmax = np.max(np.abs(data))
    im = ax.imshow(data, aspect='auto', cmap='seismic', vmin=0, vmax=vmax)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Respond1', 'Respond0'])
    ax.set_xticks([])
    ax.set_title('Mean activation during decision window')
    cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.3)
    cbar.set_label('Activation')
    heatmap_path = out_path.replace('.png', '_activation_heatmap.png')
    fig.tight_layout()
    fig.savefig(heatmap_path, dpi=300)
    plt.close(fig)
    print(f"Heatmap saved to {heatmap_path}")


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str,
                        default="models/PT_100_Interval_Discrim_bs_10",
                        help="Folder containing checkpoint.pt and hp.json")
    parser.add_argument("--dataset_size", type=int, default=1000,
                        help="Number of validation trials to probe")
    parser.add_argument("--top", type=int, default=20,
                        help="How many top neurons to print")
    parser.add_argument("--out", type=str, default="selectivity.png",
                        help="Output path for the bar plot")
    args = parser.parse_args()

    model, hp = load_model(args.model_dir)
    x, mask, labels = make_validation_batch(hp, args.dataset_size)

    si, mean_pos, mean_neg = selectivity_by_decision(model, x, mask, labels)

    plot_si(si, args.out, top=args.top)
    plot_activation_heatmap(mean_pos, mean_neg, si, args.out)

    # ------------------ Confusion matrix ------------------
    with torch.no_grad():
        y_hat = model(x.to(DEVICE))                 # (B, T, 1)
    out = y_hat.cpu().squeeze(-1)                   # (B, T)
    # predicted if any output > 0 during decision window
    preds = ((out > 0) & mask).any(dim=1).numpy()
    trues = labels.numpy()
    TP = np.logical_and(preds, trues).sum()
    TN = np.logical_and(~preds, ~trues).sum()
    FP = np.logical_and(preds, ~trues).sum()
    FN = np.logical_and(~preds, trues).sum()
    print("Confusion matrix:")
    print(f"  TP: {TP}  FP: {FP}")
    print(f"  FN: {FN}  TN: {TN}")

    print(f"All plots saved based on base name '{args.out}'")

if __name__ == "__main__":
    main()
