import os
import json
import matplotlib.pyplot as plt
import numpy as np

def plot_benchmark_matrix():
    json_path = 'results/logs/all_models_comparison.json'
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    models = list(data.keys())
    # Shorten names for clean plot
    short_model_names = {
        '1D CNN / TCN Baseline': '1D CNN / TCN',
        'Transformer Reconstructor': '1D Transformer',
        'Source-Agnostic DAE': 'Source-Agnostic DAE'
    }

    datasets = [
        'Dataset A (Stationary BBH)',
        'Dataset B (Non-Stationary)',
        'Dataset C (Glitches)',
        'Dataset D (Low SNR / Whispers)',
        'Dataset E (Binary Neutron Stars)'
    ]
    short_ds_labels = [
        'Dataset A\n(Stationary)',
        'Dataset B\n(Non-Stationary)',
        'Dataset C\n(Glitches)',
        'Dataset D\n(Low SNR)',
        'Dataset E\n(Unseen BNS)'
    ]

    x = np.arange(len(datasets))
    width = 0.25

    fig, ax = plt.subplots(figsize=(13, 6))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for i, model in enumerate(models):
        overlaps = []
        for ds in datasets:
            val = data[model]['datasets'].get(ds, {}).get('overlap_percent', 0.0)
            overlaps.append(val)
        
        offset = (i - 1) * width
        rects = ax.bar(x + offset, overlaps, width, label=short_model_names.get(model, model), color=colors[i % len(colors)], alpha=0.9, edgecolor='black', linewidth=0.5)
        
        # Add values on top of bars
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f'{height:.1f}%',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_ylabel('Waveform Overlap Match (%)', fontsize=12, fontweight='bold')
    ax.set_title('Empirical Robustness Benchmark: Waveform-Supervised vs Source-Agnostic Reconstructors', fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(short_ds_labels, fontsize=10)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

    plt.tight_layout()
    out_path = 'results/figures/multi_model_benchmark_barchart.png'
    os.makedirs('results/figures', exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved benchmark comparison barchart: {out_path}")

if __name__ == '__main__':
    plot_benchmark_matrix()
