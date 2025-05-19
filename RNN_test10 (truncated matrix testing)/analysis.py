import json
import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from rnn_model import RNNModel
from task import generate_trials
from train import mse_loss_with_mask

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


def make_validation_batch(hp: dict, size: int, comp_step: int = 20):
    """Generate a fresh batch with input noise and return tensors + meta + labels + mask."""
    hp_val = hp.copy()
    hp_val["dataset_size"] = size
    hp_val["comp_step"] = comp_step
    trial = generate_trials(hp_val["rule"], hp_val, mode="random", noise_on=True)
    x = torch.tensor(trial.x, dtype=torch.float32)
    y = torch.tensor(trial.y, dtype=torch.float32)
    c_mask = torch.tensor(trial.c_mask[:, :, 0:1], dtype=torch.float32)
    decision_mask = torch.tensor(trial.c_mask[:, :, 0] == 2, dtype=torch.bool)
    labels = torch.tensor(trial.respond, dtype=torch.bool)
    int2_ons = torch.tensor(trial.int2_ons, dtype=torch.long)
    return x, decision_mask, labels, y, c_mask, int2_ons

# -----------------------------------------------------------------------------
# d-prime selectivity & activation computation
# -----------------------------------------------------------------------------

def selectivity_by_decision(model: RNNModel, x: torch.Tensor, mask: torch.BoolTensor,
                             labels: torch.BoolTensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute d' per hidden unit and mean activations for each class."""
    with torch.no_grad():
        h = model.rnn(x.to(DEVICE))            # (B, T, N)
        h = h.cpu()

    mask_f = mask.unsqueeze(-1).float()                      # (B, T, 1)
    dec_counts = mask_f.sum(dim=1)                          # (B, 1)
    h_dec = (h * mask_f).sum(dim=1) / (dec_counts + EPS)     # (B, N)

    pos = labels
    neg = ~labels
    resp_pos = h_dec[pos]    # (n_pos, N)
    resp_neg = h_dec[neg]    # (n_neg, N)

    mean_pos = resp_pos.mean(dim=0)
    mean_neg = resp_neg.mean(dim=0)
    sigma_pos = resp_pos.std(dim=0, unbiased=True)
    sigma_neg = resp_neg.std(dim=0, unbiased=True)

    pooled_std = torch.sqrt(0.5 * (sigma_pos**2 + sigma_neg**2))
    dprime = (mean_pos - mean_neg) / (pooled_std + EPS)

    return dprime.numpy(), mean_pos.numpy(), mean_neg.numpy()

# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------

def plot_dprime(dp: np.ndarray, out_path: str, model_name: str, top: int = 20):
    idx = np.argsort(-np.abs(dp))
    sorted_dp = dp[idx]
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(sorted_dp)), sorted_dp, width=1.0)
    plt.xlabel("Neuron (sorted by |d′|)")
    plt.ylabel("d′ (d-prime)")
    plt.title(f"Hidden-unit decision d' ({model_name})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_activation_heatmap(mean_pos: np.ndarray, mean_neg: np.ndarray,
                            dp: np.ndarray, out_path: str, model_name: str):
    idx = np.argsort(-np.abs(dp))
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
    comp_step    = 20
    out          = "dprimeEasyTask1.png"
    model_name   = os.path.basename(model_dir.rstrip(os.sep))
    base_name    = os.path.splitext(out)[0]
    out_filename = f"{base_name}_{model_name}.png"
    out_path     = os.path.join(BASE_DIR, out_filename)

    model, hp = load_model(model_dir)
    x, mask, labels, y_batch, c_mask_batch, int2_ons = make_validation_batch(hp, dataset_size, comp_step)

    dp, mean_pos, mean_neg = selectivity_by_decision(model, x, mask, labels)

    # ─── Save d-prime & activation arrays for downstream analysis ───
    results_fname = f"results_{model_name}.npz"
    results_path  = os.path.join(BASE_DIR, results_fname)
    np.savez(
        results_path,
        dprime=dp,
        activation=(mean_pos + mean_neg) / 2.0
    )
    print(f"[INFO] Saved analysis results to {results_path}")

    plot_dprime(dp, out_path, model_name, top=top)
    plot_activation_heatmap(mean_pos, mean_neg, dp, out_path, model_name)

    # ─── Confusion matrix & validation loss ───
    with torch.no_grad():
        y_hat = model(x.to(DEVICE))

    out_tensor = y_hat.cpu().squeeze(-1)
    thresh = 0.4
    fired = out_tensor > thresh
    resp_trials = labels
    nonresp_trials = ~labels
    fired_any = fired.any(dim=1)

    T = fired.size(1)
    time_idx = torch.arange(T, device=fired.device).unsqueeze(0)
    epoch_mask = time_idx >= int2_ons.unsqueeze(1)
    fired_in_epoch  = (fired & epoch_mask).any(dim=1)
    fired_out_epoch = (fired & ~epoch_mask).any(dim=1)

    TP = (resp_trials & fired_in_epoch & ~fired_out_epoch).sum().item()
    FN = (resp_trials & (~fired_in_epoch | fired_out_epoch)).sum().item()
    FP = (nonresp_trials & fired_any).sum().item()
    TN = (nonresp_trials & ~fired_any).sum().item()

    print(f"Confusion matrix:  TP: {TP}  FP: {FP}  FN: {FN}  TN: {TN}")
    percent_correct = ((TP + TN) / dataset_size) * 100
    print(f"Overall accuracy: {percent_correct:.2f}%")

    loss = mse_loss_with_mask(y_hat, y_batch.to(DEVICE), c_mask_batch.to(DEVICE))
    print(f"Validation MSE loss: {loss.item():.6f}")

if __name__ == "__main__":
    main()