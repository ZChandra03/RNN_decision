# train.py

import torch
import torch.optim as optim
from task import generate_trials
from rnn_model import CustomRNN

# --- Config ---
config = {
    'dt': 10,
    'n_input': 1,
    'n_output': 1,
    'dataset_size': 400,
    'std_dur': 200,
    'tone_dur': 20,
    'delay': 750,
    'loss_type': 'mean_squared_error',
    'sigma_x': 0.005,
    'alpha': 0.2
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Generate Training Data ---
#trial = generate_trials('Interval_Discrim', config, mode='random', noise_on=True)
#x_train = torch.tensor(trial.x, dtype=torch.float32).to(device)
#y_train = torch.tensor(trial.y, dtype=torch.float32).to(device)
#mask = torch.tensor(trial.c_mask, dtype=torch.float32).to(device)

# --- Initialize Model ---
model = CustomRNN(n_input=config['n_input'], n_output=config['n_output']).to(device)
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.001
    )

# --- Training Loop ---
EPOCHS = 4000

beta = 1e-6                     # L2 coeff for rates 
for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()

    # regenerate the 400-trial batch each epoch:
    trial   = generate_trials('Interval_Discrim', config, mode='random', noise_on=True)
    x_train = torch.tensor(trial.x,      dtype=torch.float32).to(device)
    y_train = torch.tensor(trial.y,      dtype=torch.float32).to(device)
    mask    = torch.tensor(trial.c_mask, dtype=torch.float32).to(device)

    # forward, capturing hidden rates
    output, rates = model(x_train, return_hidden=True)

    # masked MSE
    mse_loss = ((output - y_train)**2 * mask).mean()

    # L2 on rates (mean over batch, time, and units)
    l2_rates = beta * (rates**2).mean()

    loss = mse_loss + l2_rates

    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        # --- compute classification accuracy ---
        with torch.no_grad():
            # Ground truth: "long" trials have a nonzero target during the response epoch
            # y_train shape: [batch, time, 1]
            # sum over time to flag any nonzero → True for long trials
            gt = (y_train.sum(dim=1) > 0).squeeze(1)      # shape [batch]

            # Model output shape: [batch, time, 1] → remove last dim
            out = output.squeeze(-1)                     # shape [batch, time]

            # Predict "long" if output ever crosses the 0.5 threshold
            pred = (out > 0.5).any(dim=1)                # shape [batch]

            # Accuracy = fraction of correct predictions
            acc = (pred == gt).float().mean().item()

        print(f"Epoch {epoch}, Loss: {loss.item():.6f}, Acc: {acc:.3f}")

print("Training Complete.")
