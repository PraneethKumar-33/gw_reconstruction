import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import yaml
import os
import sys
import json
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__)))

from data.dataset import GWDataset
from utils import compute_overlap, build_tuned_model

def evaluate_all():
    config_path = 'configs/config.yaml'
    cfg = {}
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize model using optimal tuned hyperparameters
    model = build_tuned_model('tcn', device=device)

    model_path = 'results/checkpoints/best_tcn.pt'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Successfully loaded checkpoint: {model_path}\n")

    datasets = {
        'Dataset A (Stationary BBH)': 'data/datasets/dataset_A/test',
        'Dataset B (Non-Stationary)': 'data/datasets/dataset_B/test',
        'Dataset C (Glitches)': 'data/datasets/dataset_C/test',
        'Dataset D (Low SNR / Whispers)': 'data/datasets/dataset_D/test',
        'Dataset E (Binary Neutron Stars)': 'data/datasets/dataset_E/test'
    }

    all_results = {}
    batch_size = cfg.get('tcn', {}).get('batch_size', 32)

    print("=" * 70)
    print("      RUNNING BENCHMARK EVALUATION ACROSS ALL 5 TEST DATASETS")
    print("=" * 70)

    for name, path in datasets.items():
        if not os.path.exists(path):
            print(f"Skipping {name}: path '{path}' does not exist.")
            continue

        dataset = GWDataset(path)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        mse_list = []
        overlap_list = []
        peak_errors = []

        with torch.no_grad():
            for noisy, clean in loader:
                noisy, clean_np = noisy.to(device), clean.numpy()
                pred_np = model(noisy).cpu().numpy()

                for p, c in zip(pred_np, clean_np):
                    p_flat, c_flat = p.squeeze(), c.squeeze()
                    mse = float(np.mean((p_flat - c_flat)**2))
                    overlap = compute_overlap(p_flat, c_flat)
                    
                    # Peak amplitude relative error
                    peak_c = np.max(np.abs(c_flat))
                    peak_p = np.max(np.abs(p_flat))
                    peak_err = float(abs(peak_p - peak_c) / (peak_c + 1e-8))

                    mse_list.append(mse)
                    overlap_list.append(overlap)
                    peak_errors.append(peak_err)

        mean_mse = float(np.mean(mse_list))
        mean_overlap = float(np.mean(overlap_list))
        std_overlap = float(np.std(overlap_list))
        mean_peak_err = float(np.mean(peak_errors))

        all_results[name] = {
            'num_samples': len(dataset),
            'mean_mse': mean_mse,
            'mean_overlap_percent': mean_overlap * 100,
            'std_overlap_percent': std_overlap * 100,
            'mean_peak_error_percent': mean_peak_err * 100
        }

        print(f"-> {name:<35} | Overlap: {mean_overlap*100:6.2f}% (+/- {std_overlap*100:5.2f}%) | MSE: {mean_mse:.6f}")

    print("=" * 70)

    os.makedirs('results/logs', exist_ok=True)
    json_path = 'results/logs/benchmark_results.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4)
    print(f"\nDetailed benchmark results saved to: {json_path}")

    return all_results

if __name__ == '__main__':
    evaluate_all()
