import numpy as np
from gwpy.timeseries import TimeSeries
from scipy.signal import butter, filtfilt

def apply_bandpass(data, sample_rate, flow=20.0, fhigh=500.0, order=4):
    nyq = 0.5 * sample_rate
    low = flow / nyq
    high = fhigh / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def preprocess_segment(ts, flow=20.0, fhigh=500.0):
    # 1. Bandpass filter
    bp = ts.bandpass(flow, fhigh)
    
    # 2. Whiten (ASD estimation using Welch's method)
    # Using 4s FFT window, 2s overlap as standard
    whitened = bp.whiten(4, 2)
    return whitened.value

def extract_noise_segments(raw_hdf5_path, segment_len=4096, overlap=0.5):
    ts = TimeSeries.read(raw_hdf5_path, format='hdf5')
    
    # Check if raw data contains NaNs (detector was unlocking/off)
    # If so, extract only contiguous valid chunks
    valid_mask = ~np.isnan(ts.value)
    if not np.all(valid_mask):
        # Find contiguous slices without NaNs
        slices = []
        in_slice = False
        start_idx = 0
        for i, val in enumerate(valid_mask):
            if val and not in_slice:
                start_idx = i
                in_slice = True
            elif not val and in_slice:
                slices.append((start_idx, i))
                in_slice = False
        if in_slice:
            slices.append((start_idx, len(valid_mask)))
    else:
        slices = [(0, len(ts))]

    sr = int(ts.sample_rate.value)
    # Need chunk to be at least 16 seconds long for reliable whitening PSD (4s FFT)
    min_len = 16 * sr
    
    all_segments = []
    step = int(segment_len * (1 - overlap))
    
    for s_start, s_end in slices:
        if (s_end - s_start) < min_len:
            continue
        sub_ts = ts[s_start:s_end]
        try:
            processed = preprocess_segment(sub_ts)
            # Drop filter edges (2 seconds)
            processed = processed[2*sr : -2*sr]
            
            for start in range(0, len(processed) - segment_len, step):
                seg = processed[start:start+segment_len]
                if not np.isnan(seg).any() and not np.isinf(seg).any():
                    all_segments.append(seg)
        except Exception:
            continue

    if len(all_segments) == 0:
        return np.empty((0, segment_len), dtype=np.float32)
    return np.array(all_segments, dtype=np.float32)

if __name__ == '__main__':
    print("Preprocessing module updated.")
