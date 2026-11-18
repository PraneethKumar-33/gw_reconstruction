import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
import os
import sys
import argparse
import json
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__)))

from data.dataset import GWDataset
from models.source_agnostic_dae import SourceAgnosticDAE
from utils import load_best_hyperparameters

def compute_overlap_batch(pred, target):
    p = pred.squeeze(1)
    t = target.squeeze(1)
    num = torch.sum(p * t, dim=-1)
    den = torch.sqrt(torch.sum(p**2, dim=-1) * torch.sum(t**2, dim=-1)) + 1e-8
    return torch.clamp(num / den, -1.0, 1.0)

def apply_random_masking(x, mask_ratio=0.05):
    """
    Masked Signal Reconstruction: Randomly masks contiguous time segments
    to force the autoencoder to learn intrinsic temporal continuity and signal dynamics.
    x shape: [B, 1, 4096]
    """
    x_masked = x.clone()
    b, c, length = x.shape
    mask_len = int(length * mask_ratio)
    if mask_len <= 0:
        return x_masked
        
    for i in range(b):
        start = np.random.randint(0, length - mask_len)
        x_masked[i, :, start:start+mask_len] = 0.0
    return x_masked

def train_source_agnostic():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--use-tuned', action='store_true', default=True,
                        help='Automatically load optimal hyperparameters from tuning sweep')
    args = parser.parse_args()

    # Load base config
    cfg = {}
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)

    # Automatically load optimal tuned hyperparameters
    tuned_params = load_best_hyperparameters('dae') if args.use_tuned else {}
    mask_ratio = tuned_params.get('mask_ratio', cfg.get('dae', {}).get('mask_ratio', 0.05))
    learning_rate = tuned_params.get('lr', cfg.get('dae', {}).get('learning_rate', 0.001))
    epochs = cfg.get('dae', {}).get('epochs', 35)
    batch_size = cfg.get('dae', {}).get('batch_size', 32)
    weight_decay = cfg.get('dae', {}).get('weight_decay', 1e-5)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 65)
    print("      TRAINING SOURCE-AGNOSTIC DENOISING AUTOENCODER (DAE)")
    print("      Strategy: Denoising Autoencoder + Masked Reconstruction")
    print("=" * 65)
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Hyperparameters: Mask Ratio={mask_ratio*100:.1f}% | LR={learning_rate} (Tuned: {args.use_tuned})")
    print(f"Epochs: {epochs} | Batch Size: {batch_size}")

    model = SourceAgnosticDAE(in_channels=1).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {total_params:,}")
    print("=" * 65)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    train_dataset = GWDataset('data/datasets/dataset_A/train')
    val_dataset = GWDataset('data/datasets/dataset_A/val')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    os.makedirs('results/checkpoints', exist_ok=True)
    os.makedirs('results/logs', exist_ok=True)

    best_val_loss = float('inf')
    best_val_overlap = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_overlap': [], 'val_overlap': []}

    print(f"Starting training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples...\n")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_overlaps = []

        pbar = tqdm(
            train_loader, 
            desc=f"Epoch [{epoch+1:02d}/{epochs:02d}]", 
            ncols=95, 
            unit="batch",
            leave=True
        )

        for noisy, clean in pbar:
            noisy, clean = noisy.to(device, non_blocking=True), clean.to(device, non_blocking=True)

            # Apply optimal masked signal perturbation (5% mask)
            corrupted_input = apply_random_masking(noisy, mask_ratio=mask_ratio)

            optimizer.zero_grad()
            pred = model(corrupted_input)
            loss = criterion(pred, clean)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * noisy.size(0)

            with torch.no_grad():
                batch_ov = compute_overlap_batch(pred, clean)
                train_overlaps.extend(batch_ov.cpu().numpy().tolist())

            pbar.set_postfix({'loss': f"{loss.item():.5f}", 'ov': f"{np.mean(train_overlaps)*100:.1f}%"})

        train_loss /= len(train_dataset)
        train_overlap = np.mean(train_overlaps) * 100

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_overlaps = []
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(device, non_blocking=True), clean.to(device, non_blocking=True)
                pred = model(noisy)
                loss = criterion(pred, clean)
                val_loss += loss.item() * noisy.size(0)
                batch_ov = compute_overlap_batch(pred, clean)
                val_overlaps.extend(batch_ov.cpu().numpy().tolist())

        val_loss /= len(val_dataset)
        val_overlap = np.mean(val_overlaps) * 100
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_overlap'].append(train_overlap)
        history['val_overlap'].append(val_overlap)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_val_overlap = val_overlap
            torch.save(model.state_dict(), 'results/checkpoints/best_source_agnostic_dae.pt')
            tag = " [BEST SAVED]"
        else:
            tag = ""

        print(f"Summary -> Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | Train Overlap: {train_overlap:.1f}% | Val Overlap: {val_overlap:.1f}%{tag}")

    with open('results/logs/source_agnostic_training_history.json', 'w') as f:
        json.dump(history, f, indent=4)

    print("\n" + "=" * 65)
    print(f"Source-Agnostic DAE Training Completed!")
    print(f"Best Val MSE: {best_val_loss:.6f} | Best Val Overlap: {best_val_overlap:.2f}%")
    print(f"Checkpoint: results/checkpoints/best_source_agnostic_dae.pt")
    print("=" * 65)

if __name__ == '__main__':
    train_source_agnostic()
