import matplotlib.pyplot as plt
import numpy as np
import os
import json
import yaml
import torch

try:
    from gwpy.timeseries import TimeSeries
except ImportError:
    TimeSeries = None

def compute_overlap(pred, target):
    """
    Computes normalized cross-correlation / waveform match metric:
    Overlap(h_pred, h_true) = <h_pred, h_true> / (||h_pred|| * ||h_true||)
    """
    num = np.sum(pred * target)
    den = np.sqrt(np.sum(pred**2) * np.sum(target**2))
    return float(num / (den + 1e-8))

def load_best_hyperparameters(model_name):
    """
    Automatically loads the top-performing hyperparameters from the 16-trial
    tuning sweep (hyperparameter_tuning_results.json). Falls back to configs/config.yaml.
    """
    model_key = model_name.lower()
    tuning_file = os.path.join(os.path.dirname(__file__), '..', 'results', 'logs', 'hyperparameter_tuning_results.json')
    config_file = os.path.join(os.path.dirname(__file__), '..', 'configs', 'config.yaml')

    if os.path.exists(tuning_file):
        try:
            with open(tuning_file, 'r') as f:
                data = json.load(f)
            
            mapping = {
                'tcn': 'tcn_trials',
                'transformer': 'transformer_trials',
                'dae': 'dae_trials'
            }
            category = mapping.get(model_key)
            if category and category in data:
                best_trial = max(data[category], key=lambda t: t['best_val_overlap'])
                print(f"[Auto-Hyperparameters] Loaded top configuration for {model_name.upper()}: '{best_trial['config_name']}'")
                print(f"                       Best Val Overlap: {best_trial['best_val_overlap']:.2f}% | Val MSE: {best_trial['best_val_loss']:.6f}")
                return best_trial['hyperparameters']
        except Exception as e:
            print(f"[Auto-Hyperparameters] Warning reading tuning results: {e}. Falling back to config.yaml")

    # Fallback to config.yaml
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            cfg = yaml.safe_load(f)
        if model_key in cfg:
            return cfg[model_key]

    # Hardcoded optimal fallbacks if files missing
    optimal_defaults = {
        'tcn': {'kernel_size': 7, 'dropout': 0.2, 'lr': 0.001},
        'transformer': {'patch_size': 16, 'd_model': 128, 'nhead': 4, 'num_layers': 4, 'dim_feedforward': 512, 'dropout': 0.1, 'lr': 0.002},
        'dae': {'mask_ratio': 0.05, 'lr': 0.001}
    }
    return optimal_defaults.get(model_key, {})

def build_tuned_model(model_name, device=None):
    """
    Builds and returns an instantiated model configured with its optimal hyperparameters.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    model_key = model_name.lower()
    params = load_best_hyperparameters(model_key)

    if model_key == 'tcn':
        from models.tcn_baseline import TCNBaseline
        model = TCNBaseline(
            in_channels=1,
            channels=[32, 64, 64, 128, 128, 64],
            kernel_size=params.get('kernel_size', 7),
            dropout=params.get('dropout', 0.2)
        )
    elif model_key == 'transformer':
        from models.transformer_reconstructor import TransformerReconstructor
        model = TransformerReconstructor(
            in_channels=1,
            patch_size=params.get('patch_size', 16),
            d_model=params.get('d_model', 128),
            nhead=params.get('nhead', 4),
            num_layers=params.get('num_layers', 4),
            dim_feedforward=params.get('dim_feedforward', 512),
            dropout=params.get('dropout', 0.1)
        )
    elif model_key == 'dae':
        from models.source_agnostic_dae import SourceAgnosticDAE
        model = SourceAgnosticDAE(in_channels=1)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return model.to(device)

def plot_reconstruction(noisy, predicted, clean, sample_rate=4096, save_path=None):
    t = np.linspace(0, 1, sample_rate)
    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    
    axes[0].plot(t, noisy, color='gray', label='Noisy Input')
    axes[1].plot(t, predicted, color='blue', label='Reconstructed')
    axes[2].plot(t, clean, color='green', label='True Clean GW')
    
    for ax in axes:
        ax.legend(loc='upper right')
        ax.set_ylabel('Strain')
        
    axes[2].set_xlabel('Time (s)')
    plt.suptitle('GW Signal Reconstruction')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()

def plot_spectrogram_comparison(noisy, reconstructed, sample_rate=4096, save_path=None):
    if TimeSeries is None:
        from scipy.signal import spectrogram
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        f1, t1, Sxx1 = spectrogram(noisy, fs=sample_rate, nperseg=256, noverlap=128)
        f2, t2, Sxx2 = spectrogram(reconstructed, fs=sample_rate, nperseg=256, noverlap=128)
        ax1.pcolormesh(t1, f1, 10 * np.log10(Sxx1 + 1e-12), shading='gouraud', cmap='viridis')
        ax2.pcolormesh(t2, f2, 10 * np.log10(Sxx2 + 1e-12), shading='gouraud', cmap='viridis')
        ax1.set_title('Input (Noisy)')
        ax2.set_title('Reconstructed')
        ax1.set_ylabel('Frequency [Hz]')
        ax2.set_ylabel('Frequency [Hz]')
        ax1.set_xlabel('Time [s]')
        ax2.set_xlabel('Time [s]')
        ax1.set_ylim(20, 1000)
        ax2.set_ylim(20, 1000)
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ts_noisy = TimeSeries(noisy, sample_rate=sample_rate)
        ts_recon = TimeSeries(reconstructed, sample_rate=sample_rate)
        ts_noisy.spectrogram(0.05, 0.01).plot(ax=ax1)
        ts_recon.spectrogram(0.05, 0.01).plot(ax=ax2)
        ax1.set_title('Input (Noisy)')
        ax2.set_title('Reconstructed')
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()
