# 🌌 Gravitational Wave Signal Reconstruction — Mid-Review Handout
## Final Equal Team Division & Master File Tree

---

### ⚖️ The Perfect Equal Division (Summary)

```
┌─────────────────────────────────┬─────────────────────────────────┬─────────────────────────────────┐
│        NIKHITH (1st)            │      PRANEETH (Middle / You)    │        HEMAVARDHAN (Last)       │
├─────────────────────────────────┼─────────────────────────────────┼─────────────────────────────────┤
│ • Model: 1D Causal Dilated TCN  │ • Model: 1D Temporal Transformer│ • Model: 1D Conv U-Net (DAE)    │
│ • Training: src/train.py        │ • Training: train_transformer.py│ • Training: train_source_agn.py │
│ • Checkpoint: best_tcn.pt       │ • Checkpoint: best_transformer  │ • Checkpoint: best_source_agn   │
│ • Dataset A (Baseline 5,000)    │ • Dataset B (Amplitude Drift)   │ • Dataset D (Low SNR Whispers)  │
│ • Signal Injection Engine       │ • Dataset C (Transient Glitches)│ • Dataset E (Unseen BNS Proxies)│
└─────────────────────────────────┴─────────────────────────────────┴─────────────────────────────────┘

🤝 SHARED & COLLABORATIVE (All 3 Team Members):
• 16-Trial Hyperparameter Tuning: src/tune_hyperparameters.py & src/plot_tuning_analysis.py
• Evaluation Pipeline: src/evaluate_benchmarks.py, evaluate_all_models.py & reconstruct_gw150914.py
• Figures & Evidence: results/figures/* (Bar chart, Waveform overlays, Tuning plot, Real event)
• Logs & Metrics: results/logs/* (all_models_comparison.json, benchmark_results.json)
• Presentation & UI: Gravitational_Wave_Midreview_Theme_Final.pptx & app.py
```

---

### 🌳 Master Project File Tree (Annotated by Owner)

```text
gw_reconstruction/
│
├── 📁 src/
│   ├── 📁 models/                             → 🧠 3 MODELS (1 PER MEMBER)
│   │   ├── 🟢 tcn_baseline.py                → [Nikhith] 1D Causal Dilated ConvNet (615K params).
│   │   ├── 🟡 transformer_reconstructor.py   → [Praneeth] 1D Temporal Transformer (941K params).
│   │   └── 🟣 source_agnostic_dae.py         → [Hemavardhan] 1D Convolutional U-Net Autoencoder (1.65M params).
│   │
│   ├── 🏋️ TRAINING LOOPS (1 PER MEMBER)
│   │   ├── 🟢 train.py                       → [Nikhith] TCN training execution with cosine annealing.
│   │   ├── 🟡 train_transformer.py           → [Praneeth] Transformer training execution.
│   │   └── 🟣 train_source_agnostic.py       → [Hemavardhan] DAE masked signal training execution.
│   │
│   ├── 📡 DATASETS & INJECTIONS (DIVIDED FAIRLY)
│   │   ├── 🟢 generate_datasets.py           → [Nikhith] Dataset A (4,000 train, 500 val, 500 test).
│   │   ├── 🟢 inject_signals.py              → [Nikhith] IMRPhenomD waveform injection & SNR scaling.
│   │   ├── 🟡 generate_benchmarks.py (Part 1)→ [Praneeth] Dataset B (Drift) & Dataset C (Glitches).
│   │   ├── 🟣 generate_benchmarks.py (Part 2)→ [Hemavardhan] Dataset D (Low SNR) & Dataset E (BNS).
│   │   ├── 🤝 preprocessing.py               → [Shared] 20–500 Hz Butterworth bandpass & Welch whitening.
│   │   ├── 🤝 download_data.py               → [Shared] Fetches raw LIGO Hanford HDF5 detector data.
│   │   └── 🤝 dataset.py                     → [Shared] PyTorch Dataset loader class.
│   │
│   ├── ⚙️ HYPERPARAMETER TUNING (SHARED COLLABORATION)
│   │   ├── 🤝 tune_hyperparameters.py        → [Shared] 16-trial sweep across TCN, Transformer, and DAE.
│   │   └── 🤝 plot_tuning_analysis.py        → [Shared] Plots the 3-model tuning curves.
│   │
│   └── 📊 EVALUATION & METRICS (SHARED COLLABORATION)
│       ├── 🤝 evaluate_benchmarks.py         → [Shared] Evaluates all 3 models across Datasets A through E.
│       ├── 🤝 evaluate_all_models.py         → [Shared] Computes head-to-head overlap match on Dataset A.
│       ├── 🤝 plot_benchmark_comparison.py   → [Shared] Plots the 5-dataset robustness bar chart.
│       ├── 🤝 generate_comparison_figures.py → [Shared] Plots 3-model waveform overlay comparisons.
│       ├── 🤝 generate_spectrogram_figures.py→ [Shared] Generates Q-transform spectrogram comparisons.
│       ├── 🤝 reconstruct_gw150914.py        → [Shared] Zero-shot inference on real GW150914 event.
│       └── 🤝 utils.py                       → [Shared] Math utility for waveform cosine overlap similarity.
│
├── 📁 results/
│   ├── 📁 checkpoints/                        → 💾 MODEL CHECKPOINTS (1 PER MEMBER)
│   │   ├── 🟢 best_tcn.pt                    → [Nikhith] Trained TCN weights.
│   │   ├── 🟡 best_transformer.pt            → [Praneeth] Trained Transformer weights.
│   │   └── 🟣 best_source_agnostic_dae.pt    → [Hemavardhan] Trained Autoencoder weights.
│   │
│   ├── 📁 figures/                            → 🖼️ SHARED EVIDENCE & PLOTS
│   │   ├── 🤝 multi_model_benchmark_barchart.png   → [Slide 10] Robustness comparison across A–E.
│   │   ├── 🤝 head_to_head_waveform_comparison.png → [Slide 11] Single-sample true vs predicted waveforms.
│   │   ├── 🤝 hyperparameter_tuning_analysis.png   → [Slide 12] 16-trial optimization analysis.
│   │   └── 🤝 gw150914_reconstruction.png          → [Slide 12] GW150914 real detection extraction.
│   │
│   └── 📁 logs/                               → 📝 SHARED LOGS
│       ├── 🤝 benchmark_results.json         → Full evaluation numbers for Datasets A–E.
│       ├── 🤝 hyperparameter_tuning_results.json → Trial logs for the 16 parameter sweeps.
│       └── 🤝 training histories (.json)     → Loss histories for each model.
│
├── ⚙️ configs/config.yaml                     → [Shared] Master project parameters configuration.
├── 📱 app.py                                  → [Shared] Streamlit interactive web dashboard & audio demo.
└── 📊 Gravitational_Wave_Midreview_Theme_Final.pptx → [Shared] The 13-slide PowerPoint presentation deck.
```
