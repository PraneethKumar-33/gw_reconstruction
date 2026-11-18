import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import yaml
import os
import sys
import json
import time

sys.path.append(os.path.join(os.path.dirname(__file__)))

from data.dataset import GWDataset
from utils import compute_overlap, build_tuned_model

def benchmark_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Initialize models with optimal tuned hyperparameters and checkpoints
    models = {}

    tcn_path = 'results/checkpoints/best_tcn.pt'
    if os.path.exists(tcn_path):
        m_tcn = build_tuned_model('tcn', device=device)
        m_tcn.load_state_dict(torch.load(tcn_path, map_location=device))
        m_tcn.eval()
        models['1D CNN / TCN Baseline'] = m_tcn

    tf_path = 'results/checkpoints/best_transformer.pt'
    if os.path.exists(tf_path):
        m_tf = build_tuned_model('transformer', device=device)
        m_tf.load_state_dict(torch.load(tf_path, map_location=device))
        m_tf.eval()
        models['Transformer Reconstructor'] = m_tf

    dae_path = 'results/checkpoints/best_source_agnostic_dae.pt'
    if os.path.exists(dae_path):
        m_dae = build_tuned_model('dae', device=device)
        m_dae.load_state_dict(torch.load(dae_path, map_location=device))
        m_dae.eval()
        models['Source-Agnostic DAE'] = m_dae

    if not models:
        print("No trained checkpoints found!")
        return

    datasets = {
        'Dataset A (Stationary BBH)': 'data/datasets/dataset_A/test',
        'Dataset B (Non-Stationary)': 'data/datasets/dataset_B/test',
        'Dataset C (Glitches)': 'data/datasets/dataset_C/test',
        'Dataset D (Low SNR / Whispers)': 'data/datasets/dataset_D/test',
        'Dataset E (Binary Neutron Stars)': 'data/datasets/dataset_E/test'
    }

    comparison_results = {}

    print("=" * 80)
    print("      MULTI-MODEL BENCHMARK COMPARISON MATRIX (500 SAMPLES EACH)")
    print("=" * 80)

    for model_name, model in models.items():
        params = sum(p.numel() for p in model.parameters())
        comparison_results[model_name] = {'parameters': params, 'datasets': {}}

        # Measure latency on dummy tensor
        dummy = torch.randn(1, 1, 4096).to(device)
        # Warmup
        for _ in range(10): _ = model(dummy)
        t0 = time.perf_counter()
        for _ in range(100): _ = model(dummy)
        latency_ms = ((time.perf_counter() - t0) / 100.0) * 1000.0
        comparison_results[model_name]['latency_ms'] = latency_ms

        print(f"\nEvaluating: {model_name} (Params: {params:,} | Latency: {latency_ms:.2f} ms/sample)")
        print("-" * 75)

        for ds_name, ds_path in datasets.items():
            if not os.path.exists(ds_path):
                continue

            dataset = GWDataset(ds_path)
            loader = DataLoader(dataset, batch_size=32, shuffle=False)

            mse_list = []
            overlap_list = []

            with torch.no_grad():
                for noisy, clean in loader:
                    noisy, clean_np = noisy.to(device), clean.numpy()
                    pred_np = model(noisy).cpu().numpy()

                    for p, c in zip(pred_np, clean_np):
                        p_flat, c_flat = p.squeeze(), c.squeeze()
                        mse = float(np.mean((p_flat - c_flat)**2))
                        overlap = compute_overlap(p_flat, c_flat)
                        mse_list.append(mse)
                        overlap_list.append(overlap)

            mean_mse = float(np.mean(mse_list))
            mean_overlap = float(np.mean(overlap_list)) * 100
            std_overlap = float(np.std(overlap_list)) * 100

            comparison_results[model_name]['datasets'][ds_name] = {
                'overlap_percent': mean_overlap,
                'std_overlap': std_overlap,
                'mse': mean_mse
            }

            print(f"  {ds_name:<34} | Overlap: {mean_overlap:6.2f}% (±{std_overlap:5.2f}%) | MSE: {mean_mse:.6f}")

    # Save to JSON
    os.makedirs('results/logs', exist_ok=True)
    out_json = 'results/logs/all_models_comparison.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(comparison_results, f, indent=4)

    print("\n" + "=" * 80)
    print(f"Comparison report saved to: {out_json}")
    print("=" * 80)

if __name__ == '__main__':
    benchmark_models()
