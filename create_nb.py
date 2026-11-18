import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

text_1 = '''# Phase 3: Gravitational Wave Spectrogram Analysis
This notebook visualizes the TCN model's ability to extract Gravitational Wave chirps from noisy background strain using Q-Transform spectrograms.
'''

code_1 = '''import torch
import numpy as np
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries
import sys
import os

sys.path.append(os.path.join('src'))
from models.tcn_baseline import TCNBaseline

# 1. Load the Model
model = TCNBaseline(num_channels=[32, 64, 64, 128, 128, 64], kernel_size=7, dropout=0.1)
model.load_state_dict(torch.load('results/checkpoints/best_tcn.pt', map_location='cpu'))
model.eval()
print("Model loaded successfully!")
'''

code_2 = '''# 2. Load a sample from our Benchmark Datasets (e.g., Dataset C - Glitches)
dataset_path = 'data/datasets/dataset_C/test/pairs.npy'
data = np.load(dataset_path)

# Pick a random sample
sample_idx = np.random.randint(0, len(data))
noisy_input, clean_target = data[sample_idx]

# Run inference
with torch.no_grad():
    x = torch.tensor(noisy_input).unsqueeze(0).unsqueeze(0)
    reconstruction = model(x).squeeze().numpy()
print("Inference complete!")
'''

code_3 = '''# 3. Generate Q-Transform Spectrograms
# The Q-transform is standard in GW astronomy for time-frequency visualization

def plot_qtransform(ts_data, title, ax):
    ts = TimeSeries(ts_data, sample_rate=4096)
    q = ts.q_transform(frange=(20, 500), qrange=(4, 64))
    im = ax.imshow(q.value, aspect='auto', origin='lower', 
                   extent=[0, 1.0, 20, 500], cmap='viridis')
    ax.set_title(title)
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Time [sec]')
    return im

fig, axs = plt.subplots(1, 3, figsize=(18, 5))

plot_qtransform(noisy_input, '1. Raw LIGO Noise (Input)', axs[0])
plot_qtransform(reconstruction, '2. AI Reconstruction (Output)', axs[1])
plot_qtransform(clean_target, '3. True GW Chirp (Target)', axs[2])

plt.tight_layout()
plt.show()
'''

nb['cells'] = [
    nbf.v4.new_markdown_cell(text_1),
    nbf.v4.new_code_cell(code_1),
    nbf.v4.new_code_cell(code_2),
    nbf.v4.new_code_cell(code_3)
]

with open('visualize_results.ipynb', 'w') as f:
    nbf.write(nb, f)
print("Notebook created successfully!")
