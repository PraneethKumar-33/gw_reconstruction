<div align="center">

# 🌌 Gravitational Wave Signal Reconstruction
### Template-Free Deep Learning Detection Suite for Gravitational-Wave Astrophysics

Denoise and reconstruct unmodeled gravitational wave signals from chaotic LIGO detector noise using **1D Temporal Convolutional Networks (TCN)**, **1D Temporal Transformers**, and **Source-Agnostic Denoising Autoencoders (DAE)**.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Status](https://img.shields.io/badge/Status-Fully%20Implemented%20%26%20Benchmarked-success?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)

</div>

---

## 🎯 Implementation Scope & Architecture Overview

Classical matched filtering relies on millions of pre-computed analytical waveform templates. This repository implements an end-to-end template-free deep learning reconstruction suite covering both **Waveform-Supervised** and **Source-Agnostic** paradigms:

```
                                  ┌────────────────────────────────────────────────────────┐
                                  │           Raw LIGO Strain Input [Batch, 1, 4096]       │
                                  └───────────────────────────┬────────────────────────────┘
                                                              │
                       ┌──────────────────────────────────────┼─────────────────────────────────────┐
                       ▼                                      ▼                                     ▼
      ┌─────────────────────────────────┐   ┌─────────────────────────────────┐   ┌─────────────────────────────────┐
      │     1D CNN / TCN Baseline       │   │    1D Temporal Transformer      │   │       Source-Agnostic DAE       │
      ├─────────────────────────────────┤   ├─────────────────────────────────┤   ├─────────────────────────────────┤
      │ • Waveform-Supervised           │   │ • Waveform-Supervised           │   │ • Unsupervised / Self-Supervised│
      │ • Dilated Causal Conv1D         │   │ • Multi-Head Self-Attention     │   │ • 1D Convolutional Bottleneck   │
      │ • 6 Residual Blocks (d=1..32)   │   │ • 16-sample Patch Tokenizer     │   │ • Masked Signal Reconstruction  │
      │ • Parameters: ~615K             │   │ • Parameters: ~941K             │   │ • Parameters: ~1.65M            │
      │ • Latency: 2.80 ms (Fastest)    │   │ • Latency: 10.85 ms             │   │ • Latency: 6.02 ms              │
      └────────────────┬────────────────┘   └────────────────┬────────────────┘   └────────────────┬────────────────┘
                       │                                     │                                     │
                       └─────────────────────────────────────┼─────────────────────────────────────┘
                                                             ▼
                                  ┌────────────────────────────────────────────────────────┐
                                  │      Reconstructed Clean Waveform [Batch, 1, 4096]     │
                                  └────────────────────────────────────────────────────────┘
```

---

## 📊 Comprehensive Multi-Model Benchmark Comparison (500 Samples Each)

All models were evaluated across the **5 controlled benchmark datasets** representing diverse noise environments and signal morphologies:

| Benchmark Dataset | Condition Description | 1D CNN / TCN Baseline (Overlap % \| MSE) | 1D Temporal Transformer (Overlap % \| MSE) | Source-Agnostic DAE (Overlap % \| MSE) | Winning Model |
|:---|:---|:---:|:---:|:---:|:---:|
| **Dataset A** | Stationary Noise + BBH Chirp | 78.09% ± 18.8% \| 0.01468 | 84.91% ± 14.9% \| 0.01066 | **85.52% ± 16.6% \| 0.00968** | **Source-Agnostic DAE** |
| **Dataset B** | Non-Stationary Noise (Amplitude Drift) | 66.93% ± 20.1% \| 0.03142 | 55.13% ± 23.1% \| 0.06166 | **68.97% ± 25.9% \| 0.03728** | **Source-Agnostic DAE** |
| **Dataset C** | Transient Glitches (Sine-Gaussian Bursts) | **66.72% ± 19.9% \| 0.04134** | 49.82% ± 22.7% \| 0.05861 | 58.66% ± 25.9% \| 0.05886 | **1D CNN / TCN Baseline** |
| **Dataset D** | Sub-Threshold Whispers (Low SNR 4–7) | 35.48% ± 15.5% \| 0.00737 | **51.60% ± 17.5% \| 0.00594** | 44.22% ± 16.5% \| 0.01134 | **1D Transformer** |
| **Dataset E** | Binary Neutron Stars ($1.1 - 1.4 M_\odot$) | 42.38% ± 20.2% \| 0.04418 | **52.76% ± 24.1% \| 0.03533** | 48.52% ± 22.4% \| 0.04119 | **1D Transformer** |

### 🔬 Key Scientific & Architectural Takeaways:
1. **The Glitch Dilemma (Why TCN Wins on Glitches):**  
   TCN maintains **66.72% overlap** on glitch-contaminated data compared to Transformer's **49.82%**. Because TCN uses local dilated convolutions, high-energy instrument glitches remain strictly localized. In contrast, Transformer self-attention queries attend across the whole token sequence, causing glitch energy to pollute other temporal regions.
2. **Sub-Threshold Detection (Why Transformer Wins on Low SNR):**  
   On faint signals (Dataset D), the 1D Transformer excels (**51.60% vs 35.48%**) by pooling global phase coherence across all 256 patches.
3. **Template-Free Generalization (Source-Agnostic DAE):**  
   Trained via masked signal reconstruction without analytical chirp priors, the DAE achieves the overall highest overlap (**85.52%**) on standard data and strong invariance to non-stationary detector drift.
4. **Latency Trade-Off:**  
   TCN executes in **2.80 ms/sample** (~3.8x faster than Transformer at 10.85 ms), making it the optimal choice for real-time low-latency multimessenger astronomy triggers.

---

## 🎛️ Systematic Hyperparameter Tuning & Ablation Studies (16 Trials)

To prevent arbitrary heuristic selection, we conducted a **16-trial systematic hyperparameter ablation sweep** across all 3 architectures on an NVIDIA RTX 3050 GPU.

### 1. 1D CNN / TCN Baseline (Kernel Size & Regularization Sweep)
* **Kernel Size ($K$)**: Tested $K \in \{3, 5, 7, 9\}$. 
  * $K=3$ (Narrow): **59.53% overlap** (misses long-period early inspiral cycles).
  * $K=7$ (Standard): **67.34% overlap**.
  * $K=9$ (Broad): **71.68% overlap** (broader temporal window captures lower frequencies with higher fidelity).
* **Learning Rate ($\eta$)**: $\eta = 3\times 10^{-4}$ achieved **72.00% overlap**, demonstrating smoother gradient convergence.
* **Dropout ($p$)**: $p = 0.2$ increased overlap to **72.38%**, proving higher dropout prevents the network from memorizing detector noise fluctuations.

### 2. 1D Temporal Transformer (Patch Tokenization & Depth Sweep)
* **Patch Size ($P$)**: Tested $P \in \{8, 16, 32\}$.
  * $P=8$ (512 tokens): **68.29% overlap** (excessive sequence length dilutes attention).
  * $P=16$ (256 tokens): **76.50% overlap** (optimal temporal resolution for GW chirps).
  * $P=32$ (128 tokens): **73.99% overlap** (lacks resolution near the merger peak).
* **Depth & LR**: 4 layers at $\eta = 2\times 10^{-3}$ reached **78.76% overlap**; scaling to 6 layers caused slight overfitting (**72.08%**).

### 3. Source-Agnostic DAE (Masked Signal Corruption Sweep)
* **Temporal Masking Ratio ($r_{\text{mask}}$)**: Tested $r_{\text{mask}} \in \{0.05, 0.10, 0.20, 0.30\}$ (5% to 30% random signal dropout).
  * $r_{\text{mask}} = 0.05$ (5%): **86.10% overlap** (MSE: 0.00904 — highest accuracy).
  * $r_{\text{mask}} = 0.10$ (10%): **85.54% overlap**.
  * $r_{\text{mask}} = 0.30$ (30%): **84.45% overlap** (still remarkably stable despite 30% random signal dropout).

*Saved Visualization: `results/figures/hyperparameter_tuning_analysis.png`*
*Full Log: `results/logs/hyperparameter_tuning_results.json`*

---

## 🔭 Zero-Shot Inference on Real Cosmic Discovery: GW150914

The models were evaluated directly on the actual raw LIGO Hanford (H1) detector data from humanity's first gravitational wave observation (**GW150914**, September 14, 2015).
- Raw data loaded from `data/raw/GW150914_H1.hdf5`.
- Suppresses seismic vibrations and reconstructs the characteristic merger peak at $t = 0.0$ s without matched filtering.
- Visualized in `results/figures/gw150914_reconstruction.png`.

---

## 🖥️ Interactive Streamlit Dashboard

Launch the unified dashboard:
```bash
conda activate gw_project
streamlit run app.py
```

### Dashboard Tabs:
1. **Single-Model Interactive Denoising**: Select model (TCN, Transformer, DAE) and condition, with live STFT spectrograms and **Audio Chirp Playback**.
2. **⚔️ Side-by-Side Model Comparison**: Overlay all three models on the exact same detector noise sample.
3. **Real Historical Event (GW150914)**: Inspect zero-shot reconstruction on real cosmic data.
4. **Robustness Benchmark Matrix**: Full metric breakdown and barchart comparison.
5. **Architectures & Theory**: Mathematical formulations for Dilated Convolutions, Multi-Head Attention, and Masked Autoencoders.

---

## 📂 Project Directory Structure

```text
gw_reconstruction/
├── app.py                             # Interactive Streamlit 5-tab application
├── configs/
│   └── config.yaml                    # Hyperparameter & architecture settings
├── data/
│   ├── raw/                           # Raw LIGO HDF5 strain files (GW150914, etc.)
│   ├── noise_segments/                # Extracted detector background noise
│   └── datasets/                      # Datasets A, B, C, D, E (Train, Val, Test)
├── results/
│   ├── checkpoints/
│   │   ├── best_tcn.pt                # TCN weights (Overlap: 78.09%)
│   │   ├── best_transformer.pt        # Transformer weights (Overlap: 84.91%)
│   │   └── best_source_agnostic_dae.pt# DAE weights (Overlap: 85.52%)
│   ├── figures/                       # Head-to-head overlays & benchmark charts
│   └── logs/
│       ├── training_history.json
│       ├── transformer_training_history.json
│       ├── source_agnostic_training_history.json
│       ├── benchmark_results.json
│       └── all_models_comparison.json # 3-model comparative metrics
├── src/
│   ├── data/
│   │   ├── download_data.py           # GWOSC data fetcher
│   │   ├── preprocessing.py           # Bandpass filtering & whitening
│   │   ├── inject_signals.py          # PyCBC injection engine
│   │   ├── generate_datasets.py       # Dataset A pipeline
│   │   ├── generate_benchmarks.py     # Datasets B, C, D, E generator
│   │   └── dataset.py                 # PyTorch Dataset loader
│   ├── models/
│   │   ├── tcn_baseline.py            # Dilated Causal 1D CNN
│   │   ├── transformer_reconstructor.py# 1D Temporal Transformer
│   │   └── source_agnostic_dae.py     # Denoising Autoencoder
│   ├── train.py                       # TCN training loop
│   ├── train_transformer.py           # Transformer training loop
│   ├── train_source_agnostic.py       # DAE masked training loop
│   ├── evaluate_all_models.py         # Multi-model benchmark evaluator
│   ├── generate_comparison_figures.py # Head-to-head waveform plots
│   └── plot_benchmark_comparison.py   # Benchmark bar chart generator
└── README.md
```

---

## ⚡ Execution Commands

### 1. Run Complete Benchmark Suite (All 3 Models × All 5 Datasets)
```bash
conda run -n gw_project python src/evaluate_all_models.py
```

### 2. Generate Comparison Figures
```bash
conda run -n gw_project python src/generate_comparison_figures.py
conda run -n gw_project python src/plot_benchmark_comparison.py
```

---

## 👥 Contributors (B.Tech AI Capstone / Amazon ML Hackathon)
- **Member 1 (Hemavardhan)**: LIGO Data Acquisition, Preprocessing (Whitening & Bandpass), Literature Review.
- **Member 2 (Pamu Praneeth Kumar)**: Deep Learning Architectures (TCN, Transformer, DAE), Waveform Injection Pipeline, Benchmark Stress-Test Framework.
- **Member 3 (Nikhith)**: Training Execution, Optimization & Cosine Schedulers, Evaluation Metrics & Benchmarks.
