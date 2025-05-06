# train.py

import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from rnn_model import CustomRNN
from task import TrialsDataset
from torch.amp import autocast, GradScaler

torch.backends.cudnn.benchmark = True

def main():
    print("[00] Entering main()", flush=True)
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
    'sigma_x': 0.1,
    'alpha': 1.0
    }
    print("[01] Config built", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[02] Using device {device}", flush=True)

    # Build dataset & loader
    print("[03] Building dataset", flush=True)
    dataset = TrialsDataset('Interval_Discrim', config, mode='random', noise_on=True)
    print(f"[04] Dataset shape x:{dataset.x.shape}", flush=True)

    print("[05] Creating DataLoader", flush=True)
    loader = DataLoader(dataset,
                        batch_size=config['dataset_size'],
                        shuffle=True,
                        pin_memory=True,
                        num_workers=2)      # <— try setting num_workers=0 if it hangs here
    print("[06] DataLoader ready", flush=True)

    # Model + optimizer + AMP
    model     = CustomRNN(config['n_input'], config['n_output']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scaler    = GradScaler("cuda")
    print("[07] Model, optimizer, scaler initialized", flush=True)

    # Quick sanity check: fetch one batch
    try:
        print("[08] Fetching first batch …", flush=True)
        xb, yb, mb = next(iter(loader))
        print(f"[09] Got batch shapes {xb.shape}, {yb.shape}, {mb.shape}", flush=True)
    except Exception as e:
        print("[!!] Error fetching batch:", e, flush=True)
        return

    # Training loop
    EPOCHS = 10    # shrink for debug
    beta   = 1e-6

    # before the epoch loop
    total_load_time = 0.0
    total_fw_time   = 0.0
    total_bw_time   = 0.0

    for epoch in range(EPOCHS):
        for x_batch, y_batch, mask in loader:
            t0 = time.perf_counter()
            # move to GPU (this “counts” as data loading cost)
            x = x_batch.to(device, non_blocking=True)
            y = y_batch.to(device, non_blocking=True)
            m = mask.to(device, non_blocking=True)
            t1 = time.perf_counter(); total_load_time += (t1-t0)

            optimizer.zero_grad()
            with autocast("cuda"):
                out, l2 = model(x)
            t2 = time.perf_counter(); total_fw_time += (t2-t1)

            loss = ((out - y)**2 * m).mean() + beta*l2
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            t3 = time.perf_counter(); total_bw_time += (t3-t2)

        print(f"Epoch {epoch} times (s): load {total_load_time:.2f}, "
            f"fw {total_fw_time:.2f}, bw {total_bw_time:.2f}")

if __name__=="__main__":
    import torch.multiprocessing as mp
    mp.freeze_support()
    main()
