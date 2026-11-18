import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__)))
from utils import build_tuned_model

def compute_overlap(pred, target):
    num = np.sum(pred * target)
    den = np.sqrt(np.sum(pred**2) * np.sum(target**2))
    return float(num / (den + 1e-8)) * 100

def generate_comparison_plots():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs('results/figures', exist_ok=True)

    # 1. Load models with optimal tuned hyperparameters
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
        models['1D Temporal Transformer'] = m_tf

    dae_path = 'results/checkpoints/best_source_agnostic_dae.pt'
    if os.path.exists(dae_path):
        m_dae = build_tuned_model('dae', device=device)
        m_dae.load_state_dict(torch.load(dae_path, map_location=device))
        m_dae.eval()
        models['Source-Agnostic DAE'] = m_dae

    if not models:
        print("No models available.")
        return

    # Load high-SNR sample #12 from Dataset A
    data = np.load('data/datasets/dataset_A/test/pairs.npy')
    sample_idx = 12
    noisy_input, clean_target = data[sample_idx]

    x = torch.tensor(noisy_input, device=device).unsqueeze(0).unsqueeze(0)
    time_axis = np.linspace(0, 1.0, 4096)

    preds = {}
    overlaps = {}
    for name, m in models.items():
        with torch.no_grad():
            out = m(x).squeeze().cpu().numpy()
            preds[name] = out
            overlaps[name] = compute_overlap(out, clean_target)

    # 2. Plot Multi-Panel Comparison
    n_rows = 2 + len(models)
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 2.3 * n_rows), sharex=True)

    # Input
    axes[0].plot(time_axis, noisy_input, color='#555555', lw=1.0, label='Raw Detector Input')
    axes[0].set_title(f"1. Raw LIGO Detector Strain (Sample #{sample_idx} - Hidden GW + Real Noise)", fontsize=11, fontweight='bold')
    axes[0].set_ylabel("Strain")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="upper right")

    # Models
    colors = ['#0066cc', '#d95f02', '#7570b3']
    for idx, (name, p) in enumerate(preds.items()):
        ax = axes[1 + idx]
        ov = overlaps[name]
        ax.plot(time_axis, p, color=colors[idx % len(colors)], lw=1.5, label=f'{name} (Overlap: {ov:.2f}%)')
        ax.set_title(f"{idx+2}. {name} Reconstruction", fontsize=11, fontweight='bold')
        ax.set_ylabel("Strain")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")

    # Ground Truth Target
    axes[-1].plot(time_axis, clean_target, color='#2ca02c', lw=1.6, label='Ground Truth Clean Signal')
    axes[-1].set_title(f"{n_rows}. Ground Truth Target Waveform (Analytical Injection)", fontsize=11, fontweight='bold')
    axes[-1].set_ylabel("Strain")
    axes[-1].set_xlabel("Time (seconds)", fontsize=10)
    axes[-1].grid(True, alpha=0.25)
    axes[-1].legend(loc="upper right")

    plt.tight_layout()
    out_path = 'results/figures/head_to_head_waveform_comparison.png'
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved head-to-head comparison plot: {out_path}")

if __name__ == '__main__':
    generate_comparison_plots()
