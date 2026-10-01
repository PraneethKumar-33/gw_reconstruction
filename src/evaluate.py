import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import argparse
import yaml
import os
import json
from tqdm import tqdm

from data.dataset import GWDataset
from models.tcn_baseline import TCNBaseline
from utils import plot_reconstruction, plot_spectrogram_comparison

def compute_overlap(pred, target):
    num = np.sum(pred * target)
    den = np.sqrt(np.sum(pred**2) * np.sum(target**2))
    return float(num / (den + 1e-8))

def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--dataset', type=str, default='data/datasets/dataset_A/test')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = TCNBaseline(
        in_channels=cfg['model']['in_channels'],
        channels=cfg['model']['channels'],
        kernel_size=cfg['model']['kernel_size'],
        dropout=cfg['model']['dropout']
    ).to(device)
    
    model_path = 'results/checkpoints/best_tcn.pt'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded trained checkpoint: {model_path}")
    else:
        print("Warning: No checkpoint found, using uninitialized model.")
        
    model.eval()
    dataset = GWDataset(args.dataset)
    loader = DataLoader(dataset, batch_size=cfg['training']['batch_size'], shuffle=False)
    
    mse_list = []
    overlap_list = []
    
    sample_noisy = None
    sample_clean = None
    sample_pred = None

    print(f"\nEvaluating on {len(dataset)} test samples...")
    with torch.no_grad():
        for noisy, clean in tqdm(loader, desc="Evaluating", ncols=80, unit="batch"):
            noisy, clean_np = noisy.to(device), clean.numpy()
            pred_np = model(noisy).cpu().numpy()
            
            for p, c in zip(pred_np, clean_np):
                p_flat, c_flat = p.squeeze(), c.squeeze()
                mse_list.append(float(np.mean((p_flat - c_flat)**2)))
                overlap_list.append(compute_overlap(p_flat, c_flat))
                
            if sample_noisy is None:
                sample_noisy = noisy[0].cpu().numpy().squeeze()
                sample_clean = clean_np[0].squeeze()
                sample_pred = pred_np[0].squeeze()

    mean_mse = np.mean(mse_list)
    mean_overlap = np.mean(overlap_list)
    std_overlap = np.std(overlap_list)

    print("\n" + "=" * 50)
    print("           FINAL TEST EVALUATION REPORT")
    print("=" * 50)
    print(f"Test Set Size : {len(dataset)} pairs")
    print(f"Mean MSE      : {mean_mse:.6f}")
    print(f"Mean Overlap  : {mean_overlap:.4f} (+/- {std_overlap:.4f})")
    print("=" * 50)

    # Save visualization plots for mid-review presentation
    os.makedirs('results/figures', exist_ok=True)
    plot_reconstruction(sample_noisy, sample_pred, sample_clean, save_path='results/figures/test_reconstruction_sample.png')
    print("Saved sample reconstruction plot to: results/figures/test_reconstruction_sample.png")

if __name__ == '__main__':
    evaluate()
