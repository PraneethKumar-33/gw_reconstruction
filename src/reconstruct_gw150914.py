import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries
from gwosc.datasets import event_gps

sys.path.append(os.path.join(os.path.dirname(__file__)))
from models.tcn_baseline import TCNBaseline
from data.preprocessing import apply_bandpass
from utils import build_tuned_model

def reconstruct_real_gw150914():
    raw_path = 'data/raw/GW150914_H1.hdf5'
    if not os.path.exists(raw_path):
        print(f"Error: {raw_path} not found.")
        return

    print("Loading raw GW150914 H1 strain data...")
    ts_full = TimeSeries.read(raw_path, format='hdf5')
    sample_rate = int(ts_full.sample_rate.value)

    # Merger GPS time for GW150914
    gps = event_gps('GW150914')
    print(f"GW150914 Merger GPS: {gps}")

    # Crop a 16-second buffer around the merger for high-quality whitening
    ts_segment = ts_full.crop(gps - 8.0, gps + 8.0)

    # 1. Bandpass filter 20 - 500 Hz
    bp = ts_segment.bandpass(20.0, 500.0)

    # 2. Whiten using 4s FFT, 2s overlap
    whitened = bp.whiten(4, 2)

    # 3. Extract exactly 1.0 second centered on the merger (4096 samples)
    # The chirp merger happens near the end of the window, typically at t ~ 0.75s in 1s windows
    # Window from gps - 0.75s to gps + 0.25s
    crop_start = gps - 0.75
    crop_end = gps + 0.25
    whitened_1s = whitened.crop(crop_start, crop_end)
    
    # Ensure exact 4096 samples
    data_1s = np.array(whitened_1s.value[:4096], dtype=np.float32)

    # Normalize amplitude like training data (unit variance / standard scaling)
    data_std = np.std(data_1s)
    if data_std > 0:
        norm_input = data_1s / data_std
    else:
        norm_input = data_1s

    # 4. Load trained TCN model with optimal tuned hyperparameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_tuned_model('tcn', device=device)
    model.load_state_dict(torch.load('results/checkpoints/best_tcn.pt', map_location=device))
    model.eval()

    # 5. Run inference
    with torch.no_grad():
        x = torch.tensor(norm_input, device=device).unsqueeze(0).unsqueeze(0) # [1, 1, 4096]
        pred_clean = model(x).squeeze().cpu().numpy()

    # 6. Plot Time-Domain Reconstruction
    os.makedirs('results/figures', exist_ok=True)
    t = np.linspace(-0.75, 0.25, 4096)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(t, norm_input, color='#555555', alpha=0.8, linewidth=1.0, label='Raw LIGO Hanford (H1) Whitened Strain')
    axes[0].set_title("Actual LIGO Hanford (H1) Detector Recording Around GW150914 Merger", fontsize=13, fontweight='bold')
    axes[0].set_ylabel("Normalized Strain", fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")

    axes[1].plot(t, pred_clean, color='#0066cc', linewidth=1.8, label='TCN Reconstructed Waveform (Zero-Shot on Real Data)')
    axes[1].axvline(0.0, color='red', linestyle='--', alpha=0.7, label='Merger Time (GPS 1126259462.42)')
    axes[1].set_title("Reconstructed Gravitational Wave Signal (TCN Denoising)", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("Time Relative to Merger [s]", fontsize=11)
    axes[1].set_ylabel("Reconstructed Amplitude", fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    time_fig_path = 'results/figures/gw150914_reconstruction.png'
    plt.savefig(time_fig_path, dpi=200)
    plt.close()
    print(f"Saved real event waveform plot: {time_fig_path}")

    # 7. Generate Q-Transform Spectrogram comparison
    try:
        raw_ts = TimeSeries(norm_input, sample_rate=sample_rate)
        recon_ts = TimeSeries(pred_clean, sample_rate=sample_rate)

        q_raw = raw_ts.q_transform(frange=(20, 500), qrange=(4, 64))
        q_recon = recon_ts.q_transform(frange=(20, 500), qrange=(4, 64))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        q_raw.plot(ax=ax1)
        ax1.set_title("Raw Detector Q-Transform (LIGO H1 Noise)", fontsize=12, fontweight='bold')
        ax1.set_ylabel("Frequency [Hz]")
        ax1.set_xlabel("Time [s]")

        q_recon.plot(ax=ax2)
        ax2.set_title("TCN Reconstructed Q-Transform (Chirp Signal)", fontsize=12, fontweight='bold')
        ax2.set_ylabel("Frequency [Hz]")
        ax2.set_xlabel("Time [s]")

        spec_fig_path = 'results/figures/gw150914_spectrogram.png'
        plt.tight_layout()
        plt.savefig(spec_fig_path, dpi=200)
        plt.close()
        print(f"Saved real event spectrogram: {spec_fig_path}")
    except Exception as e:
        print(f"Note on spectrogram generation: {e}")

    print("Real GW150914 event reconstruction completed successfully!")

if __name__ == '__main__':
    reconstruct_real_gw150914()
