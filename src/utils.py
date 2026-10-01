import matplotlib.pyplot as plt
import numpy as np
import os
from gwpy.timeseries import TimeSeries

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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ts_noisy = TimeSeries(noisy, sample_rate=sample_rate)
    ts_recon = TimeSeries(reconstructed, sample_rate=sample_rate)
    
    # Q-transform requires some buffer, but we can do a simple spectrogram
    plot1 = ts_noisy.spectrogram(0.05, 0.01).plot(ax=ax1)
    plot2 = ts_recon.spectrogram(0.05, 0.01).plot(ax=ax2)
    
    ax1.set_title('Input (Noisy)')
    ax2.set_title('Reconstructed')
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()
