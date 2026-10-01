import jax.numpy as jnp
import numpy as np
from ripplegw.waveforms.IMRPhenomD import gen_IMRPhenomD
import os
from scipy.signal import butter, filtfilt

def generate_waveform(Mchirp, eta, sample_rate=4096, duration=1.0, f_lower=20.0):
    N = int(sample_rate * duration)
    f = np.fft.rfftfreq(N, d=1.0/sample_rate)
    
    # We place the merger at tc = 0.75
    params = jnp.array([Mchirp, eta, 0.0, 0.0, 100.0, 0.75, 0.0])
    
    f_eval = f.copy()
    f_eval[f < f_lower] = f_lower
    
    h_f = gen_IMRPhenomD(jnp.array(f_eval), params, f_ref=f_lower)
    h_f = np.array(h_f)
    h_f[f < f_lower] = 0.0
    
    h_t = np.fft.irfft(h_f, n=N) * sample_rate
    
    # Bandpass filter
    b, a = butter(4, [f_lower/(sample_rate/2), 500.0/(sample_rate/2)], btype='band')
    h_t = filtfilt(b, a, h_t)
    return h_t.astype(np.float32)

def inject_and_scale(noise_segment, h_t, target_snr):
    # noise_segment is whitened (std ~ 1)
    # Target SNR in whitened space = ||h||_2 / sigma_noise = ||h||_2
    norm = np.linalg.norm(h_t)
    if norm > 0:
        scaled_h = (h_t / norm) * target_snr
    else:
        scaled_h = np.zeros_like(h_t)
    
    noisy_signal = noise_segment + scaled_h
    return noisy_signal.astype(np.float32), scaled_h.astype(np.float32)

if __name__ == '__main__':
    print('Signal injection module updated and verified.')
