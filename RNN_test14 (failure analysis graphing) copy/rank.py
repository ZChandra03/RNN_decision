import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from rnn_model import RNNModel

# -----------------------------------------------------------------------------
# Utility: load a trained RNNModel
# -----------------------------------------------------------------------------
def load_model(model_dir: str):
    """
    Load model and hyperparameters from the given directory.
    Assumes hp.json and checkpoint.pt exist in model_dir.
    """
    hp_path = os.path.join(model_dir, 'hp.json')
    ckpt_path = os.path.join(model_dir, 'checkpoint.pt')
    with open(hp_path, 'r') as f:
        hp = json.load(f)
    model = RNNModel(hp)
    state = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    return model, hp

# -----------------------------------------------------------------------------
# Compute and plot singular values of the recurrent weight matrix
# -----------------------------------------------------------------------------
def plot_singular_values(model):
    # Extract recurrent weights
    w_rec = model.rnn.w_rec.detach().cpu().numpy()  # shape: (n_rec, n_rec)

    # Compute SVD
    U, S, Vt = np.linalg.svd(w_rec, full_matrices=False)

    # Numeric rank
    tol = S.max() * max(w_rec.shape) * np.finfo(S.dtype).eps
    numeric_rank = np.sum(S > tol)
    print(f"Numeric rank: {numeric_rank}/{w_rec.shape[0]}")

    # Effective rank (entropy-based)
    p = S / S.sum()
    eff_rank = np.exp(-np.sum(p * np.log(p + 1e-20)))
    print(f"Effective rank: {eff_rank:.2f}")

    # Plot singular values
    plt.figure()
    plt.plot(np.arange(1, len(S) + 1), S, marker='o')
    plt.title('Singular Values of $W_{\mathrm{rec}}$')
    plt.xlabel('Singular value index')
    plt.ylabel('Singular value')
    plt.yscale('log')
    plt.grid(True)
    plt.show()

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(BASE_DIR, 'models', 'easy_trained')

    print(f"Loading model from: {model_dir}")
    model, hp = load_model(model_dir)
    print(f"Model loaded with {hp.get('n_rnn')} recurrent units.")

    plot_singular_values(model)
