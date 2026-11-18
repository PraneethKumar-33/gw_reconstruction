import os
import json
import matplotlib.pyplot as plt
import numpy as np

def plot_tuning_results():
    json_path = 'results/logs/hyperparameter_tuning_results.json'
    if not os.path.exists(json_path):
        print(f"File {json_path} does not exist.")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    os.makedirs('results/figures', exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. TCN Tuning Plot: Kernel Size vs Overlap
    tcn_trials = data.get('tcn_trials', [])
    if tcn_trials:
        names = [t['config_name'].split('(')[1].split(',')[0] for t in tcn_trials]
        overlaps = [t['best_val_overlap'] for t in tcn_trials]
        
        bars1 = axes[0].bar(range(len(names)), overlaps, color='#1f77b4', alpha=0.85, edgecolor='black', linewidth=0.6)
        axes[0].set_title("1. TCN: Kernel Size & Regularization Sweep", fontsize=11, fontweight='bold')
        axes[0].set_xticks(range(len(names)))
        axes[0].set_xticklabels(names, rotation=35, ha='right', fontsize=9)
        axes[0].set_ylabel("Validation Waveform Overlap (%)", fontsize=10)
        axes[0].set_ylim(0, 100)
        axes[0].grid(axis='y', linestyle='--', alpha=0.3)
        for b in bars1:
            h = b.get_height()
            axes[0].annotate(f"{h:.1f}%", xy=(b.get_x() + b.get_width()/2, h),
                             xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9, fontweight='bold')

    # 2. Transformer Tuning Plot: Patch Size & Depth vs Overlap
    tf_trials = data.get('transformer_trials', [])
    if tf_trials:
        names = [t['config_name'].split('(')[1].split(',')[0] for t in tf_trials]
        overlaps = [t['best_val_overlap'] for t in tf_trials]

        bars2 = axes[1].bar(range(len(names)), overlaps, color='#ff7f0e', alpha=0.85, edgecolor='black', linewidth=0.6)
        axes[1].set_title("2. Transformer: Patch Size & Depth Sweep", fontsize=11, fontweight='bold')
        axes[1].set_xticks(range(len(names)))
        axes[1].set_xticklabels(names, rotation=35, ha='right', fontsize=9)
        axes[1].set_ylabel("Validation Waveform Overlap (%)", fontsize=10)
        axes[1].set_ylim(0, 100)
        axes[1].grid(axis='y', linestyle='--', alpha=0.3)
        for b in bars2:
            h = b.get_height()
            axes[1].annotate(f"{h:.1f}%", xy=(b.get_x() + b.get_width()/2, h),
                             xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9, fontweight='bold')

    # 3. Source-Agnostic DAE: Masking Ratio vs Overlap
    dae_trials = data.get('dae_trials', [])
    if dae_trials:
        names = [t['config_name'].split('(')[1].split(',')[0] for t in dae_trials]
        overlaps = [t['best_val_overlap'] for t in dae_trials]

        bars3 = axes[2].bar(range(len(names)), overlaps, color='#2ca02c', alpha=0.85, edgecolor='black', linewidth=0.6)
        axes[2].set_title("3. DAE: Masked Signal Corruption Ratio Sweep", fontsize=11, fontweight='bold')
        axes[2].set_xticks(range(len(names)))
        axes[2].set_xticklabels(names, rotation=35, ha='right', fontsize=9)
        axes[2].set_ylabel("Validation Waveform Overlap (%)", fontsize=10)
        axes[2].set_ylim(0, 100)
        axes[2].grid(axis='y', linestyle='--', alpha=0.3)
        for b in bars3:
            h = b.get_height()
            axes[2].annotate(f"{h:.1f}%", xy=(b.get_x() + b.get_width()/2, h),
                             xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9, fontweight='bold')

    plt.suptitle("Multi-Model Hyperparameter Optimization Analysis (16 Systematic Trials)", fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    out_fig = 'results/figures/hyperparameter_tuning_analysis.png'
    plt.savefig(out_fig, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved tuning analysis plot to: {out_fig}")

if __name__ == '__main__':
    plot_tuning_results()
