import numpy as np
import torch
import matplotlib.pyplot as plt
import os
# Reuse your existing load_model function from analysis.py
from analysis import load_model

# Directory setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'analysis_outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def effective_rec_weights(model):
    """
    Compute the effective recurrent weight matrix after applying Dale's law and autapse mask.
    Returns a NumPy array of shape (n_rec, n_rec).
    """
    w_rec = model.rnn.w_rec.detach().cpu()
    autapse_mask = model.rnn.autapse_mask.detach().cpu()
    ei_mask = model.rnn.ei_mask.detach().cpu()
    W_eff = ei_mask @ (w_rec * autapse_mask)
    return W_eff.numpy(), ei_mask


def plot_ei_histograms(W_eff, ei_mask, model_name):
    """
    Plot and save separate histograms for excitatory vs. inhibitory outgoing weights.
    """
    ei_vec = ei_mask.diag().numpy()
    ex_idx = np.where(ei_vec > 0)[0]
    inh_idx = np.where(ei_vec < 0)[0]

    w_ex = W_eff[ex_idx, :].flatten()
    w_inh = W_eff[inh_idx, :].flatten()

    plt.figure(figsize=(8, 5))
    plt.hist(w_ex, bins=50, alpha=0.7, label='Excitatory', density=True)
    plt.hist(w_inh, bins=50, alpha=0.7, label='Inhibitory', density=True)
    plt.xlabel('Weight value')
    plt.ylabel('Density')
    plt.title(f'E/I Weight Distributions ({model_name})')
    plt.legend()
    fname = f'ei_histograms_{model_name}.png'
    out_path = os.path.join(OUTPUT_DIR, fname)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f'[INFO] Saved: {out_path}')


def cross_reference_activity(W_eff, results_path, model_name):
    """
    Cross-reference each neuron's connectivity with activity metrics; save scatter plot.
    """
    data = np.load(results_path)
    degree = np.sum(np.abs(W_eff), axis=1)
    plt.figure(figsize=(6, 6))
    if 'selectivity' in data:
        sel = data['selectivity']
        plt.scatter(degree, sel, alpha=0.7, label='Selectivity')
    if 'activation' in data:
        act = data['activation']
        plt.scatter(degree, act, alpha=0.7, label='Activation')
    plt.xlabel('Connectivity Degree')
    plt.legend()
    plt.title(f'Degree vs. Activity ({model_name})')
    fname = f'degree_activity_{model_name}.png'
    out_path = os.path.join(OUTPUT_DIR, fname)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f'[INFO] Saved: {out_path}')


def spectral_properties(W_eff, model_name):
    """
    Compute and save the eigenvalue spectrum of the recurrent weight matrix.
    """
    eigs = np.linalg.eigvals(W_eff)
    spectral_radius = np.max(np.abs(eigs))
    print(f'Spectral radius: {spectral_radius:.4f}')

    plt.figure(figsize=(6, 6))
    plt.scatter(eigs.real, eigs.imag, alpha=0.6)
    theta = np.linspace(0, 2*np.pi, 200)
    plt.plot(np.cos(theta), np.sin(theta), '--', linewidth=1)
    plt.xlabel('Real part')
    plt.ylabel('Imag part')
    plt.title(f'Eigenvalue Spectrum ({model_name})')
    plt.axis('equal')
    fname = f'eigen_spectrum_{model_name}.png'
    out_path = os.path.join(OUTPUT_DIR, fname)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f'[INFO] Saved: {out_path}')


if __name__ == '__main__':
    # Configuration
    model_dir = os.path.join(BASE_DIR, 'models', 'easy_trained')
    results_path = os.path.join(BASE_DIR, "results_easy_trained.npz")
    model_name = os.path.basename(model_dir)

    print(f'[INFO] Loading model from {model_dir}')
    model, hp = load_model(model_dir)
    print(f'[INFO] Model has {hp.get("n_rnn")} recurrent units')

    W_eff, ei_mask = effective_rec_weights(model)

    plot_ei_histograms(W_eff, ei_mask, model_name)

    if results_path:
        cross_reference_activity(W_eff, results_path, model_name)

    spectral_properties(W_eff, model_name)

    print('[INFO] All analyses complete')
