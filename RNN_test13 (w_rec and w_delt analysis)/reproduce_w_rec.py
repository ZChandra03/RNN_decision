import numpy as np
import torch
from train import get_default_hp
from rnn_model import RNNModel

def get_initial_w_rec(seed, hp_override=None):
    """
    Return the exactly-initialized recurrent weight matrix for a given seed.
    
    Args:
        seed (int): the RNG seed used in BioRNN (hp['seed'])
        hp_override (dict, optional): any hyperparam overrides, 
            e.g. {'n_rnn': 128, 'w_rec_gain': 0.2, ...}
            
    Returns:
        w_rec (np.ndarray): shape (n_rnn, n_rnn)
    """
    # 1) build the hp dict just like train.py does
    hp = get_default_hp()
    if hp_override:
        hp.update(hp_override)
    hp['seed'] = seed
    
    # 2) instantiate the model (this runs np.RandomState(seed) inside BioRNN.__init__)
    model = RNNModel(hp)
    
    # 3) grab the recurrent weight matrix as NumPy
    #    (it's a Parameter, but built entirely from numpy RNG)
    w_rec = model.rnn.w_rec.detach().cpu().numpy()
    return w_rec

if __name__ == "__main__":
    seed = 100
    
    w_rec = get_initial_w_rec(seed)
    print("w_rec.shape =", w_rec.shape)
    # optionally save to .npy
    #np.save(f"w_rec_seed_{seed}.npy", w_rec)
    print(f"Saved w_rec_seed_{seed}.npy")