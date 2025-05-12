import os
import json
import numpy as np
import torch
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
# Compute effective recurrent weights with strict Dale enforcement
# -----------------------------------------------------------------------------
def compute_effective_weights(model):
    """
    Apply autapse mask and enforce Dale's law by row-wise sign.
    Returns:
      W_eff (np.ndarray): effective recurrent weight matrix (n_rec x n_rec)
      ei_vec (np.ndarray): vector of +1 (E) or -1 (I) per neuron (length n_rec)
    """
    # Extract parameters
    w_rec = model.rnn.w_rec.detach().cpu().numpy()           # (n_rec, n_rec)
    autapse = model.rnn.autapse_mask.detach().cpu().numpy()  # same shape
    # The ei_mask in code was a diagonal; get the diag vector
    ei_mask_full = model.rnn.ei_mask.detach().cpu().numpy()  # shape (n_rec, n_rec)
    ei_vec = np.diag(ei_mask_full)                           # +1 or -1, length n_rec

    # Apply autapse (zero diag) and then enforce Dale
    w_masked = w_rec * autapse
    # Multiply each row by its sign: excitatory rows stay positive, inhibitory flip sign
    W_eff = w_masked * ei_vec[:, None]
    return W_eff, ei_vec

# -----------------------------------------------------------------------------
# Check for sign violations
# -----------------------------------------------------------------------------
def check_dale_violations(W_eff: np.ndarray, ei_vec: np.ndarray):
    """
    Identify neurons whose outgoing weights violate Dale's law.
    A violation occurs when an excitatory neuron has negative weights or
    an inhibitory neuron has positive weights.

    Returns:
      violations (list of tuples): (neuron_index, ei_sign, num_bad_weights)
    """
    n_rec = W_eff.shape[0]
    violations = []
    for i in range(n_rec):
        sign = ei_vec[i]  # +1 (E) or -1 (I)
        row = W_eff[i, :]
        # bad if row*sign < 0
        bad = (row * sign) < 0
        if np.any(bad):
            violations.append((i, int(sign), int(bad.sum())))
    return violations

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    # -- Configure your model directory here --
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(BASE_DIR, 'models', 'easy_trained')

    print(f"Loading model from: {model_dir}")
    model, hp = load_model(model_dir)
    print(f"Model loaded with {hp.get('n_rnn')} recurrent units.")

    # Compute weights
    W_eff, ei_vec = compute_effective_weights(model)

    # Check violations
    violations = check_dale_violations(W_eff, ei_vec)
    total = len(ei_vec)
    print(f"Total neurons: {total}. Violations found: {len(violations)}")
    if violations:
        print("First few violations:")
        for idx, sign, count in violations[:10]:
            cell_type = 'E' if sign > 0 else 'I'
            print(f" Neuron {idx}: type={cell_type}, bad_weight_count={count}")
    else:
        print("No Dale's law violations detected—masking is perfect.")
