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
    Returns W_eff (n_rec×n_rec) and ei_mask (n_rec×n_rec).
    """
    w_rec = model.rnn.w_rec.detach().cpu()
    autapse_mask = model.rnn.autapse_mask.detach().cpu()
    ei_mask = model.rnn.ei_mask.detach().cpu()
    W_eff = ei_mask @ (w_rec * autapse_mask)
    return W_eff.numpy(), ei_mask.numpy()


def plot_ei_histograms(W_eff, ei_mask, model_name):
    """
    Plot and save histograms for excitatory vs inhibitory weights, and return raw arrays.
    """
    ei_vec = np.diag(ei_mask)
    ex_idx = np.where(ei_vec > 0)[0]
    inh_idx = np.where(ei_vec < 0)[0]
    w_ex = W_eff[ex_idx, :].flatten()
    w_inh = W_eff[inh_idx, :].flatten()
    print("E/I counts:", np.sum(ei_vec>0), "E  vs.", np.sum(ei_vec<0), "I")
    plt.figure(figsize=(8, 5))
    plt.hist(w_ex, bins=50, alpha=0.7, label='Excitatory', density=True)
    plt.hist(w_inh, bins=50, alpha=0.7, label='Inhibitory', density=True)
    plt.xlabel('Weight value')
    plt.ylabel('Density')
    plt.title(f'E/I Weight Distributions ({model_name})')
    plt.legend()
    fname = f'ei_histograms_{model_name}.png'
    out_path = os.path.join(OUTPUT_DIR, fname)
    plt.tight_layout(); plt.savefig(out_path, dpi=300); plt.close()
    print(f'[INFO] Saved: {out_path}')
    return w_ex, w_inh, ei_vec


def cross_reference_activity(W_eff, results_path, model_name):
    """
    Plot degree vs selectivity & activation; return raw data arrays.
    """
    data = np.load(results_path)
    degree = np.sum(np.abs(W_eff), axis=1)
    sel = data.get('selectivity')
    act = data.get('activation')

    plt.figure(figsize=(6, 6))
    if sel is not None:
        plt.scatter(degree, sel, alpha=0.7, label='Selectivity')
    if act is not None:
        plt.scatter(degree, act, alpha=0.7, label='Activation')
    plt.xlabel('Connectivity Degree')
    plt.legend()
    plt.title(f'Degree vs. Activity ({model_name})')
    fname = f'degree_activity_{model_name}.png'
    out_path = os.path.join(OUTPUT_DIR, fname)
    plt.tight_layout(); plt.savefig(out_path, dpi=300); plt.close()
    print(f'[INFO] Saved: {out_path}')
    return degree, sel, act


def spectral_properties(W_eff, model_name):
    """
    Plot eigenvalue spectrum; return eigenvalues.
    """
    eigs = np.linalg.eigvals(W_eff)
    spectral_radius = np.max(np.abs(eigs))
    print(f'Spectral radius: {spectral_radius:.4f}')

    plt.figure(figsize=(6, 6))
    plt.scatter(eigs.real, eigs.imag, alpha=0.6)
    theta = np.linspace(0, 2*np.pi, 200)
    plt.plot(np.cos(theta), np.sin(theta), '--', linewidth=1)
    plt.xlabel('Real part'); plt.ylabel('Imag part')
    plt.title(f'Eigenvalue Spectrum ({model_name})')
    plt.axis('equal')
    fname = f'eigen_spectrum_{model_name}.png'
    out_path = os.path.join(OUTPUT_DIR, fname)
    plt.tight_layout(); plt.savefig(out_path, dpi=300); plt.close()
    print(f'[INFO] Saved: {out_path}')
    return eigs


def save_raw_data(model_name, w_ex, w_inh, ei_vec, degree, sel, act, eigs):
    """
    Save all raw arrays to a single npz file for upload/analysis.
    """
    fname = f'analysis_data_{model_name}.npz'
    out_path = os.path.join(OUTPUT_DIR, fname)
    np.savez(out_path,
             w_ex=w_ex,
             w_inh=w_inh,
             ei_vec=ei_vec,
             degree=degree,
             selectivity=sel,
             activation=act,
             eigenvalues=eigs)
    print(f'[INFO] Raw data saved to {out_path}')
    return out_path


if __name__ == '__main__':
    # Configuration
    model_dir = os.path.join(BASE_DIR, 'models', 'easy_trained')
    results_path = os.path.join(BASE_DIR, "results_easy_trained.npz")
    model_name = os.path.basename(model_dir)

    print(f'[INFO] Loading model from {model_dir}')
    model, hp = load_model(model_dir)
    print(f'[INFO] Model has {hp.get("n_rnn")} recurrent units')

    # Compute weights
    W_eff, ei_mask = effective_rec_weights(model)

    # Plot & collect E/I histograms data
    w_ex, w_inh, ei_vec = plot_ei_histograms(W_eff, ei_mask, model_name)

    # Plot & collect degree vs. activity data
    degree, sel, act = None, None, None
    if results_path and os.path.exists(results_path):
        degree, sel, act = cross_reference_activity(W_eff, results_path, model_name)
    else:
        print(f'[WARNING] results file not found: {results_path}')

    # Plot & collect spectral data
    eigs = spectral_properties(W_eff, model_name)

    # Save all raw arrays
    save_raw_data(model_name, w_ex, w_inh, ei_vec, degree, sel, act, eigs)

    print('[INFO] All analyses complete')
