import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
import json
import time
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__)))
from data.dataset import GWDataset
from models.tcn_baseline import TCNBaseline
from models.transformer_reconstructor import TransformerReconstructor
from models.source_agnostic_dae import SourceAgnosticDAE

def apply_random_masking(x, mask_ratio=0.15):
    x_masked = x.clone()
    b, c, length = x.shape
    mask_len = int(length * mask_ratio)
    for i in range(b):
        start = np.random.randint(0, length - mask_len)
        x_masked[i, :, start:start+mask_len] = 0.0
    return x_masked

def compute_overlap_batch(pred, target):
    p = pred.squeeze(1)
    t = target.squeeze(1)
    num = torch.sum(p * t, dim=-1)
    den = torch.sqrt(torch.sum(p**2, dim=-1) * torch.sum(t**2, dim=-1)) + 1e-8
    return torch.clamp(num / den, -1.0, 1.0)

def train_and_eval_trial(model, train_loader, val_loader, epochs, lr, device, is_dae=False, mask_ratio=0.10):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_val_overlap = 0.0

    epoch_history = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_overlaps = []

        for noisy, clean in train_loader:
            noisy, clean = noisy.to(device, non_blocking=True), clean.to(device, non_blocking=True)
            
            inp = apply_random_masking(noisy, mask_ratio=mask_ratio) if is_dae else noisy

            optimizer.zero_grad()
            pred = model(inp)
            loss = criterion(pred, clean)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * noisy.size(0)
            with torch.no_grad():
                ov = compute_overlap_batch(pred, clean)
                train_overlaps.extend(ov.cpu().numpy().tolist())

        scheduler.step()
        train_loss /= len(train_loader.dataset)
        train_overlap = float(np.mean(train_overlaps)) * 100

        # Eval
        model.eval()
        val_loss = 0.0
        val_overlaps = []
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(device, non_blocking=True), clean.to(device, non_blocking=True)
                pred = model(noisy)
                loss = criterion(pred, clean)
                val_loss += loss.item() * noisy.size(0)
                ov = compute_overlap_batch(pred, clean)
                val_overlaps.extend(ov.cpu().numpy().tolist())

        val_loss /= len(val_loader.dataset)
        val_overlap = float(np.mean(val_overlaps)) * 100

        if val_loss < best_val_loss:
            best_val_loss = val_loss
        if val_overlap > best_val_overlap:
            best_val_overlap = val_overlap

        epoch_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_overlap': train_overlap,
            'val_overlap': val_overlap
        })

    return best_val_loss, best_val_overlap, epoch_history

def run_hyperparameter_tuning():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 75)
    print("      SYSTEMATIC HYPERPARAMETER TUNING & OPTIMIZATION PIPELINE")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 75)

    train_dataset = GWDataset('data/datasets/dataset_A/train')
    val_dataset = GWDataset('data/datasets/dataset_A/val')

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, pin_memory=True)

    epochs_per_trial = 12
    tuning_results = {
        'metadata': {
            'epochs_per_trial': epochs_per_trial,
            'batch_size': 32,
            'device': str(device),
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        },
        'tcn_trials': [],
        'transformer_trials': [],
        'dae_trials': []
    }

    # =========================================================================
    # 1. TCN Hyperparameter Search (Kernel Size K, Learning Rate LR, Dropout)
    # =========================================================================
    print("\n[PHASE 1/3] Tuning 1D CNN / TCN Baseline Hyperparameters...")
    tcn_configs = [
        {'name': 'TCN-Trial-1 (K=3, Narrow Temporal Filter)', 'kernel_size': 3, 'lr': 1e-3, 'dropout': 0.1},
        {'name': 'TCN-Trial-2 (K=5, Medium Filter)',          'kernel_size': 5, 'lr': 1e-3, 'dropout': 0.1},
        {'name': 'TCN-Trial-3 (K=7, Baseline Sweetspot)',     'kernel_size': 7, 'lr': 1e-3, 'dropout': 0.1},
        {'name': 'TCN-Trial-4 (K=9, Broad Temporal Filter)',  'kernel_size': 9, 'lr': 1e-3, 'dropout': 0.1},
        {'name': 'TCN-Trial-5 (LR=3e-4, Lower Rate)',         'kernel_size': 7, 'lr': 3e-4, 'dropout': 0.1},
        {'name': 'TCN-Trial-6 (Dropout=0.2, High Reg)',       'kernel_size': 7, 'lr': 1e-3, 'dropout': 0.2},
    ]

    for cfg in tcn_configs:
        t0 = time.time()
        model = TCNBaseline(
            channels=[32, 64, 64, 128, 128, 64],
            kernel_size=cfg['kernel_size'],
            dropout=cfg['dropout']
        ).to(device)

        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        best_loss, best_ov, history = train_and_eval_trial(
            model, train_loader, val_loader, epochs=epochs_per_trial, lr=cfg['lr'], device=device
        )
        elapsed = time.time() - t0

        trial_data = {
            'config_name': cfg['name'],
            'params': params,
            'hyperparameters': cfg,
            'best_val_loss': float(best_loss),
            'best_val_overlap': float(best_ov),
            'elapsed_sec': float(elapsed),
            'history': history
        }
        tuning_results['tcn_trials'].append(trial_data)
        print(f"  -> {cfg['name']:<48} | Val Overlap: {best_ov:5.2f}% | MSE: {best_loss:.5f} ({elapsed:.1f}s)")

    # =========================================================================
    # 2. Transformer Search (Patch Size P, d_model, Attention Heads, LR)
    # =========================================================================
    print("\n[PHASE 2/3] Tuning 1D Temporal Transformer Hyperparameters...")
    tf_configs = [
        {'name': 'TF-Trial-1 (P=8, Fine 512 Tokens)',         'patch_size': 8,  'd_model': 64,  'nhead': 2, 'num_layers': 3, 'lr': 1e-3},
        {'name': 'TF-Trial-2 (P=16, d=128, Baseline Sweet)',  'patch_size': 16, 'd_model': 128, 'nhead': 4, 'num_layers': 4, 'lr': 1e-3},
        {'name': 'TF-Trial-3 (P=32, Coarse 128 Tokens)',      'patch_size': 32, 'd_model': 128, 'nhead': 4, 'num_layers': 4, 'lr': 1e-3},
        {'name': 'TF-Trial-4 (Deep 6-Layer, d=128)',          'patch_size': 16, 'd_model': 128, 'nhead': 4, 'num_layers': 6, 'lr': 8e-4},
        {'name': 'TF-Trial-5 (LR=2e-3, High Learning Rate)',   'patch_size': 16, 'd_model': 128, 'nhead': 4, 'num_layers': 4, 'lr': 2e-3},
    ]

    for cfg in tf_configs:
        t0 = time.time()
        model = TransformerReconstructor(
            in_channels=1,
            patch_size=cfg['patch_size'],
            d_model=cfg['d_model'],
            nhead=cfg['nhead'],
            num_layers=cfg['num_layers'],
            dim_feedforward=cfg['d_model'] * 4,
            dropout=0.1
        ).to(device)

        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        best_loss, best_ov, history = train_and_eval_trial(
            model, train_loader, val_loader, epochs=epochs_per_trial, lr=cfg['lr'], device=device
        )
        elapsed = time.time() - t0

        trial_data = {
            'config_name': cfg['name'],
            'params': params,
            'hyperparameters': cfg,
            'best_val_loss': float(best_loss),
            'best_val_overlap': float(best_ov),
            'elapsed_sec': float(elapsed),
            'history': history
        }
        tuning_results['transformer_trials'].append(trial_data)
        print(f"  -> {cfg['name']:<48} | Val Overlap: {best_ov:5.2f}% | MSE: {best_loss:.5f} ({elapsed:.1f}s)")

    # =========================================================================
    # 3. Source-Agnostic DAE Search (Mask Ratio, Learning Rate)
    # =========================================================================
    print("\n[PHASE 3/3] Tuning Source-Agnostic DAE Hyperparameters...")
    dae_configs = [
        {'name': 'DAE-Trial-1 (Mask=5%, Light Perturbation)',  'mask_ratio': 0.05, 'lr': 1e-3},
        {'name': 'DAE-Trial-2 (Mask=10%, Optimal Corruption)', 'mask_ratio': 0.10, 'lr': 1e-3},
        {'name': 'DAE-Trial-3 (Mask=20%, Heavy Perturbation)', 'mask_ratio': 0.20, 'lr': 1e-3},
        {'name': 'DAE-Trial-4 (Mask=30%, Extreme Dropout)',    'mask_ratio': 0.30, 'lr': 1e-3},
        {'name': 'DAE-Trial-5 (LR=5e-4, Conservative Rate)',   'mask_ratio': 0.10, 'lr': 5e-4},
    ]

    for cfg in dae_configs:
        t0 = time.time()
        model = SourceAgnosticDAE(in_channels=1).to(device)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        best_loss, best_ov, history = train_and_eval_trial(
            model, train_loader, val_loader, epochs=epochs_per_trial, lr=cfg['lr'], device=device,
            is_dae=True, mask_ratio=cfg['mask_ratio']
        )
        elapsed = time.time() - t0

        trial_data = {
            'config_name': cfg['name'],
            'params': params,
            'hyperparameters': cfg,
            'best_val_loss': float(best_loss),
            'best_val_overlap': float(best_ov),
            'elapsed_sec': float(elapsed),
            'history': history
        }
        tuning_results['dae_trials'].append(trial_data)
        print(f"  -> {cfg['name']:<48} | Val Overlap: {best_ov:5.2f}% | MSE: {best_loss:.5f} ({elapsed:.1f}s)")

    # Save to JSON
    os.makedirs('results/logs', exist_ok=True)
    out_file = 'results/logs/hyperparameter_tuning_results.json'
    with open(out_file, 'w') as f:
        json.dump(tuning_results, f, indent=4)

    print("\n" + "=" * 75)
    print(f"All 16 Hyperparameter Trials Finished Successfully!")
    print(f"Full tuning report saved to: {out_file}")
    print("=" * 75)

if __name__ == '__main__':
    run_hyperparameter_tuning()
