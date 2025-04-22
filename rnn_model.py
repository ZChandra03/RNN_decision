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
        super(CustomRNN, self).__init__()
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

    def forward(self, x, return_hidden=False):
        batch_size, tdim, _ = x.shape
        # initialize rates to zero
        r = torch.zeros(batch_size, self.N, device=x.device)
        outputs = []
        hidden_hist = [] if return_hidden else None

        for t in range(tdim):
            # 1) noise term (scaled per discretized OU process)
            noise = torch.randn(batch_size, self.N, device=x.device) \
                    * SIGMA_REC * np.sqrt(2 * ALPHA)

            # 2) input drive at time t
            input_t = torch.matmul(x[:, t, :], self.W_in.T)

            # 3) Euler‐update of rates with ReLU nonlinearity
            r = (1 - ALPHA) * r + ALPHA * torch.relu(
                    torch.matmul(r, self.W_rec.T)
                    + input_t
                    + self.bias
                    + noise
                )

            # 4) linear readout
            z = self.W_out(r)
            outputs.append(z.unsqueeze(1))

            # optionally record hidden state
            if return_hidden:
                hidden_hist.append(r)

        out = torch.cat(outputs, dim=1)  # [batch, tdim, n_output]

        if return_hidden:
            # stack into [batch, tdim, N]
            rates = torch.stack(hidden_hist, dim=1)
            return out, rates
        else:
            return out

