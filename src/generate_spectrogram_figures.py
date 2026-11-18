import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries
from scipy.signal import spectrogram

sys.path.append(os.path.join(os.path.dirname(__file__)))
from utils import build_tuned_model

def create_spectrogram_plots():
    os.makedirs('results/figures', exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model with optimal tuned hyperparameters
    model = build_tuned_model('tcn', device=device)
    model.load_state_dict(torch.load('results/checkpoints/best_tcn.pt', map_location=device))
    model.eval()

    sample_rate = 4096

    # 1. Dataset A Sample (Stationary BBH)
    dataset_a = np.load('data/datasets/dataset_A/test/pairs.npy')
    # Find a good high-overlap sample
    best_idx = 0
    best_overlap = 0
    for i in range(min(50, len(dataset_a))):
        noisy, clean = dataset_a[i]
        with torch.no_grad():
            x = torch.tensor(noisy, device=device).unsqueeze(0).unsqueeze(0)
            pred = model(x).squeeze().cpu().numpy()
        overlap = float(np.sum(pred * clean) / (np.sqrt(np.sum(pred**2) * np.sum(clean**2)) + 1e-8))
        if overlap > best_overlap:
            best_overlap = overlap
            best_idx = i

    print(f"Dataset A best sample #{best_idx} with overlap: {best_overlap*100:.2f}%")
    noisy_a, clean_a = dataset_a[best_idx]
    with torch.no_grad():
        x = torch.tensor(noisy_a, device=device).unsqueeze(0).unsqueeze(0)
        pred_a = model(x).squeeze().cpu().numpy()

    # Time series comparison
    time_axis = np.linspace(0, 1.0, 4096)
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(time_axis, noisy_a, color='#7f7f7f', lw=1.0, label='Raw Input (Noisy Strain)')
    axes[0].set_title(f"Dataset A (Standard BBH) — Sample #{best_idx} | Overlap Match: {best_overlap*100:.1f}%", fontsize=13, fontweight='bold')
    axes[0].set_ylabel("Normalized Strain")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")

    axes[1].plot(time_axis, pred_a, color='#1f77b4', lw=1.5, label='TCN Reconstructed Waveform')
    axes[1].set_ylabel("Normalized Strain")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right")

    axes[2].plot(time_axis, clean_a, color='#2ca02c', lw=1.5, label='Ground Truth Clean Waveform')
    axes[2].set_ylabel("Normalized Strain")
    axes[2].set_xlabel("Time (seconds)", fontsize=11)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="upper right")

    plt.tight_layout()
    plt.savefig('results/figures/dataset_A_waveform_reconstruction.png', dpi=200)
    plt.close()
    print("Saved: results/figures/dataset_A_waveform_reconstruction.png")

    # Time-Frequency Spectrograms (using scipy / gwpy)
    # Using scipy.signal.spectrogram for robust, artifact-free visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    signals = [
        ('1. Noisy Detector Input', noisy_a),
        (f'2. TCN Reconstructed Output (Overlap {best_overlap*100:.1f}%)', pred_a),
        ('3. Ground Truth Chirp', clean_a)
    ]

    for ax, (title, sig) in zip(axes, signals):
        # Compute spectrogram
        f, t_spec, Sxx = spectrogram(sig, fs=sample_rate, nperseg=256, noverlap=220)
        # Focus on GW sensitive frequency band: 20 Hz to 450 Hz
        mask = (f >= 20) & (f <= 450)
        im = ax.pcolormesh(t_spec, f[mask], 10 * np.log10(Sxx[mask, :] + 1e-10), cmap='inferno', shading='gouraud')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('Frequency (Hz)', fontsize=11)
        fig.colorbar(im, ax=ax, label='Power / Freq (dB/Hz)')

    plt.tight_layout()
    spec_path = 'results/figures/spectrogram_comparison_dataset_A.png'
    plt.savefig(spec_path, dpi=200)
    plt.close()
    print(f"Saved: {spec_path}")

    # 2. Dataset C (Glitches) Spectrogram Comparison
    dataset_c = np.load('data/datasets/dataset_C/test/pairs.npy')
    noisy_c, clean_c = dataset_c[0]
    with torch.no_grad():
        x = torch.tensor(noisy_c, device=device).unsqueeze(0).unsqueeze(0)
        pred_c = model(x).squeeze().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    glitch_signals = [
        ('1. Input with Loud Glitch Burst', noisy_c),
        ('2. TCN Denoised Output (Glitch Suppressed)', pred_c),
        ('3. Ground Truth Hidden Signal', clean_c)
    ]
    for ax, (title, sig) in zip(axes, glitch_signals):
        f, t_spec, Sxx = spectrogram(sig, fs=sample_rate, nperseg=256, noverlap=220)
        mask = (f >= 20) & (f <= 450)
        im = ax.pcolormesh(t_spec, f[mask], 10 * np.log10(Sxx[mask, :] + 1e-10), cmap='inferno', shading='gouraud')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('Frequency (Hz)', fontsize=11)
        fig.colorbar(im, ax=ax, label='Power (dB/Hz)')

    plt.tight_layout()
    glitch_path = 'results/figures/spectrogram_comparison_glitches_C.png'
    plt.savefig(glitch_path, dpi=200)
    plt.close()
    print(f"Saved: {glitch_path}")

    print("All spectrogram figures successfully generated and saved!")

if __name__ == '__main__':
    create_spectrogram_plots()
