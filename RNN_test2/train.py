# train.py

import torch
import torch.optim as optim
from task import generate_trials
from rnn_model import CustomRNN

# ---- Hyper‑parameters and task config --------------------------------------
config = {
    "dt"          : 10,
    "n_input"     : 1,
    "n_output"    : 1,
    "dataset_size": 400,          # 20 mini‑batches × 50
    "std_dur"     : 200,
    "tone_dur"    : 20,
    "delay"       : 750,
    "loss_type"   : "mean_squared_error",
    "sigma_x"     : 0.005,
    "alpha"       : 0.2,
}

MB_SIZE   = 10                      # trials per mini‑batch
EPOCHS    = 4000
LR        = 3e-4
BETA_L2   = 5e-5                   # L2 coefficient on rates
THRESH    = 0.4                    # decision threshold for “long” class

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Model & optimiser -----------------------------------------------------
model = CustomRNN(
    n_input=config["n_input"],
    n_output=config["n_output"]
).to(device)

optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR
)

# ---- Training loop ---------------------------------------------------------
for epoch in range(EPOCHS):
    # (re)generate a fresh 1000‑trial dataset each epoch
    trial = generate_trials("Interval_Discrim", config,
                            mode="random", noise_on=True)

    x_full = torch.tensor(trial.x,      dtype=torch.float32, device=device)
    y_full = torch.tensor(trial.y,      dtype=torch.float32, device=device)
    m_full = torch.tensor(trial.c_mask, dtype=torch.float32, device=device)

    # split into 20 mini‑batches of 50
    x_batches = torch.split(x_full, MB_SIZE)
    y_batches = torch.split(y_full, MB_SIZE)
    m_batches = torch.split(m_full, MB_SIZE)

    epoch_loss = 0.0
    model.train()
    for x_mb, y_mb, m_mb in zip(x_batches, y_batches, m_batches):
        optimizer.zero_grad()

        out_mb, rates_mb = model(x_mb, return_hidden=True)

        mse = ((out_mb - y_mb) ** 2 * m_mb).mean()
        l2  = BETA_L2 * (rates_mb ** 2).mean()
        loss = mse + l2

        loss.backward()

        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)

        optimizer.step()

        with torch.no_grad():

            model.W_rec.clamp_(-4.0, 4.0)

        epoch_loss += loss.item()

    # average loss over the 20 mini‑batches
    epoch_loss /= len(x_batches)

    # ---- diagnostics every 20 epochs --------------------------------------
    if epoch % 20 == 0:
        model.eval()
        with torch.no_grad():
            out_full, _ = model(x_full, return_hidden=True)

            # ----- response‑window metrics -----
            resp = out_full[:, -20:, 0]                 # last 200 ms (20 time‑steps)
            gt   = (y_full.sum(dim=1) > 0).squeeze(1)   # True for “long” trials
            pred = (resp > THRESH).any(dim=1)           # network decision

            acc        = (pred == gt).float().mean().item()
            mean_resp  = resp.mean().item()
            max_resp   = resp.max().item()

        # largest gradient in the most recent backward pass
        max_grad = max(
            p.grad.abs().max().item() for p in model.parameters()
            if p.grad is not None
        )

        print(
            f"Epoch {epoch:4d} | Loss {epoch_loss:.6f} | Acc {acc:.3f} | "
            f"mean_resp {mean_resp:.3f} | max_resp {max_resp:.3f} | "
            f"max_grad {max_grad:.3f}"
        )
        

print("Training complete.")
