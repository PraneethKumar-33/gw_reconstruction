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
from models.tcn_baseline import TCNBaseline
from utils import load_best_hyperparameters

def compute_overlap_batch(pred, target):
    # Vectorized overlap (cosine similarity / waveform match) across batch
    # pred, target: [B, 1, 4096]
    p = pred.squeeze(1)
    t = target.squeeze(1)
    num = torch.sum(p * t, dim=-1)
    den = torch.sqrt(torch.sum(p**2, dim=-1) * torch.sum(t**2, dim=-1)) + 1e-8
    overlap = num / den
    return torch.clamp(overlap, -1.0, 1.0)

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--use-tuned', action='store_true', default=True, 
                        help='Automatically load optimal hyperparameters from tuning sweep')
    args = parser.parse_args()

    # Load base config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # Automatically load optimal tuned hyperparameters
    tuned_params = load_best_hyperparameters('tcn') if args.use_tuned else {}
    kernel_size = tuned_params.get('kernel_size', cfg.get('tcn', {}).get('kernel_size', cfg['model']['kernel_size']))
    dropout = tuned_params.get('dropout', cfg.get('tcn', {}).get('dropout', 0.2))
    learning_rate = tuned_params.get('lr', cfg.get('tcn', {}).get('learning_rate', cfg['training']['learning_rate']))
    epochs = cfg.get('tcn', {}).get('epochs', cfg['training']['epochs'])
    batch_size = cfg.get('tcn', {}).get('batch_size', cfg['training']['batch_size'])
    weight_decay = cfg.get('tcn', {}).get('weight_decay', cfg['training']['weight_decay'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 65)
    print("      TRAINING 1D TEMPORAL CONVOLUTIONAL NETWORK (TCN)")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Hyperparameters: Kernel={kernel_size} | Dropout={dropout} | LR={learning_rate} (Tuned: {args.use_tuned})")
    print(f"Epochs: {epochs} | Batch Size: {batch_size}")
    print("=" * 65)

    # Build model with tuned parameters
    model = TCNBaseline(
        in_channels=cfg['model']['in_channels'],
        channels=cfg['model']['channels'],
        kernel_size=kernel_size,
        dropout=dropout
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {total_params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=learning_rate,
        weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    # Load data
    train_dataset = GWDataset('data/datasets/dataset_A/train')
    val_dataset = GWDataset('data/datasets/dataset_A/val')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    os.makedirs('results/checkpoints', exist_ok=True)
    os.makedirs('results/logs', exist_ok=True)

    best_val_loss = float('inf')
    best_val_overlap = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_overlap': [], 'val_overlap': [], 'lr': []}

    print(f"\nStarting training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples...\n")
    
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
            
            optimizer.zero_grad()
            pred = model(noisy)
            loss = criterion(pred, clean)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * noisy.size(0)
            
            with torch.no_grad():
                batch_ov = compute_overlap_batch(pred, clean)
                train_overlaps.extend(batch_ov.cpu().tolist())
                mean_b_ov = float(torch.mean(batch_ov))
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'overlap': f"{mean_b_ov*100:.1f}%"})
            
        train_loss /= len(train_dataset)
        train_mean_overlap = np.mean(train_overlaps) * 100.0  # as percentage
        
        # Validation
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
                val_overlaps.extend(batch_ov.cpu().tolist())
        
        val_loss /= len(val_dataset)
        val_mean_overlap = np.mean(val_overlaps) * 100.0  # as percentage
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_overlap'].append(train_mean_overlap)
        history['val_overlap'].append(val_mean_overlap)
        history['lr'].append(current_lr)
        
        status_msg = (f"  --> Epoch {epoch+1:02d} Summary | "
                      f"Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | "
                      f"Train Match: {train_mean_overlap:.2f}% | Val Match: {val_mean_overlap:.2f}%")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_overlap = val_mean_overlap
            torch.save(model.state_dict(), 'results/checkpoints/best_tcn.pt')
            status_msg += " [Best Checkpoint Saved!]"
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg['training']['early_stopping_patience']:
                print(status_msg)
                print(f"\n[Early Stopping] No improvement in validation loss for {cfg['training']['early_stopping_patience']} epochs.")
                break
                
        print(status_msg)

    with open('results/logs/training_history.json', 'w') as f:
        json.dump(history, f, indent=4)
    print(f"\nTraining complete! Peak Val Match Accuracy: {best_val_overlap:.2f}%")
    print("History saved to results/logs/training_history.json")

if __name__ == '__main__':
    train()
