import os
import numpy as np
from tqdm import tqdm
import yaml

from preprocessing import extract_noise_segments
from inject_signals import generate_waveform, inject_and_scale

def build_dataset(noise_segments, num_samples, snr_range, out_dir, mass_range=(15.0, 50.0)):
    os.makedirs(out_dir, exist_ok=True)
    pairs = []
    
    pbar = tqdm(total=num_samples, desc=f"Building {os.path.basename(out_dir):<5}", ncols=80, unit="pair")
    for i in range(num_samples):
        # 1. Random noise segment from real LIGO data
        noise = noise_segments[np.random.randint(0, len(noise_segments))]
        
        # 2. Random BBH parameters
        Mchirp = np.random.uniform(mass_range[0], mass_range[1])
        eta = np.random.uniform(0.15, 0.25)
        snr = np.random.uniform(snr_range[0], snr_range[1])
        
        # 3. Generate waveform & inject into noise
        h_t = generate_waveform(Mchirp, eta)
        shift = np.random.randint(-1000, 1000)
        h_t = np.roll(h_t, shift)
        
        noisy, clean = inject_and_scale(noise, h_t, snr)
        pairs.append((noisy, clean))
        pbar.update(1)
        
    pbar.close()
    pairs = np.array(pairs, dtype=np.float32)
    np.save(os.path.join(out_dir, 'pairs.npy'), pairs)
    print(f"Saved {num_samples} pairs to {out_dir}/pairs.npy")

if __name__ == '__main__':
    config_path = os.path.join('..', '..', 'configs', 'config.yaml')
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
        
    raw_dir = os.path.join('..', '..', 'data', 'raw')
    files = [os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.endswith('.hdf5')]
    
    print(f"Extracting real LIGO noise segments from {len(files)} detector files...")
    all_segments = []
    for f in files:
        segs = extract_noise_segments(f)
        all_segments.append(segs)
    noise_segments = np.vstack(all_segments)
    print(f"Total real noise segments available: {len(noise_segments)}")

    # For mid-review: 1000 total pairs (800 train, 100 val, 100 test)
    # Generates in ~6 minutes, trains in ~2 minutes with high accuracy!
    total = cfg['dataset'].get('total_samples', 1000)
    train_split = cfg['dataset'].get('train_split', 0.8)
    val_split = cfg['dataset'].get('val_split', 0.1)
    test_split = cfg['dataset'].get('test_split', 0.1)

    n_train = int(total * train_split)
    n_val = int(total * val_split)
    n_test = total - n_train - n_val

    snr_range = cfg['data']['snr_range']
    base_out = os.path.join('..', '..', 'data', 'datasets', 'dataset_A')
    
    print(f"\n--- Generating Controlled Dataset A (Total: {total} pairs) ---")
    print(f"Train: {n_train} | Val: {n_val} | Test: {n_test} | SNR Range: {snr_range}\n")
    
    build_dataset(noise_segments, n_train, snr_range, os.path.join(base_out, 'train'))
    build_dataset(noise_segments, n_val, snr_range, os.path.join(base_out, 'val'))
    build_dataset(noise_segments, n_test, snr_range, os.path.join(base_out, 'test'))
    print("\nDataset generation complete!")
