# rnn_model.py
import torch
import torch.nn as nn
import numpy as np

# --- Hyperparameters ---
N_REC = 256
TAU = 50
DT = 10
ALPHA = DT / TAU
SIGMA_REC = 0.05

class CustomRNN(nn.Module):
    def __init__(self, n_input, n_output):
        super().__init__()
        self.N = N_REC
        self.n_input = n_input
        self.n_output = n_output

        # Input weights (fixed)
        self.W_in = nn.Parameter(torch.randn(self.N, self.n_input) * 0.1, requires_grad=False)

        # Recurrent weights with Dale's Law
        W_rec = self.init_dale_weights()
        self.W_rec = nn.Parameter(W_rec)

        # Output layer
        self.W_out = nn.Linear(self.N, self.n_output, bias=True)

        # Bias
        self.bias = nn.Parameter(torch.zeros(self.N))

    def init_dale_weights(self):
        W_ortho = torch.linalg.qr(torch.randn(self.N, self.N))[0] * 0.1
        D = torch.ones(self.N)
        D[204:] = -4  # 204 excitatory, 52 inhibitory
        A = torch.ones(self.N, self.N) - torch.eye(self.N)
        W_rec = D.view(-1, 1) * A * torch.abs(W_ortho)
        return W_rec

    def forward(self, x):
        """
        x: [batch, tdim, n_input]
        returns: out [batch, tdim, n_output], l2_rates (scalar)
        """
        batch_size, tdim, _ = x.shape
        device = x.device

        # initialize rates to zero
        r = torch.zeros(batch_size, self.N, device=device)

        # pre-sample noise for all time steps
        noise_seq = torch.randn(batch_size, tdim, self.N, device=device) \
                    * SIGMA_REC * np.sqrt(2 / ALPHA)

        outputs = []
        l2_accum = 0.0

        for t in range(tdim):
            noise = noise_seq[:, t, :]
            # input drive
            input_t = x[:, t, :] @ self.W_in.T
            # Euler update with ReLU
            r = (1 - ALPHA) * r + ALPHA * torch.relu(r @ self.W_rec.T + input_t + self.bias + noise)
            # output
            z = self.W_out(r)
            outputs.append(z.unsqueeze(1))
            # accumulate L2 of rates
            l2_accum += (r ** 2).mean()

        out = torch.cat(outputs, dim=1)  # [batch, tdim, n_output]
        l2_rates = l2_accum / tdim  # average over time steps
        return out, l2_rates