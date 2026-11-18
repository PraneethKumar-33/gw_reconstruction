import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from scipy.io import wavfile
import io
import os
import sys
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from models.tcn_baseline import TCNBaseline
from models.transformer_reconstructor import TransformerReconstructor
from models.source_agnostic_dae import SourceAgnosticDAE

st.set_page_config(
    page_title="Gravitational Wave AI Reconstructor Suite",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

from utils import build_tuned_model

# ----------------- Helper Functions -----------------
@st.cache_resource
def load_tcn_model():
    path = 'results/checkpoints/best_tcn.pt'
    if os.path.exists(path):
        model = build_tuned_model('tcn', device='cpu')
        model.load_state_dict(torch.load(path, map_location='cpu'))
        model.eval()
        return model
    return None

@st.cache_resource
def load_transformer_model():
    path = 'results/checkpoints/best_transformer.pt'
    if os.path.exists(path):
        model = build_tuned_model('transformer', device='cpu')
        model.load_state_dict(torch.load(path, map_location='cpu'))
        model.eval()
        return model
    return None

@st.cache_resource
def load_dae_model():
    path = 'results/checkpoints/best_source_agnostic_dae.pt'
    if os.path.exists(path):
        model = build_tuned_model('dae', device='cpu')
        model.load_state_dict(torch.load(path, map_location='cpu'))
        model.eval()
        return model
    return None

@st.cache_data
def load_dataset(dataset_folder):
    path = os.path.join('data', 'datasets', dataset_folder, 'test', 'pairs.npy')
    if os.path.exists(path):
        return np.load(path)
    return None

@st.cache_data
def load_benchmark_metrics():
    path = os.path.join('results', 'logs', 'benchmark_results.json')
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None

@st.cache_data
def load_all_models_comparison():
    path = os.path.join('results', 'logs', 'all_models_comparison.json')
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None

def compute_overlap(pred, target):
    num = np.sum(pred * target)
    den = np.sqrt(np.sum(pred**2) * np.sum(target**2))
    return float(num / (den + 1e-8)) * 100

def to_audio_bytes(signal, sample_rate=4096):
    norm = signal / (np.max(np.abs(signal)) + 1e-8)
    audio_data = (norm * 32767).astype(np.int16)
    buf = io.BytesIO()
    wavfile.write(buf, sample_rate, audio_data)
    return buf.getvalue()

# ----------------- UI Header -----------------
st.title("🌌 Gravitational Wave AI Reconstructor Suite")
st.markdown("""
Deep Learning signal reconstruction and denoising engine comparing **1D CNN / TCN**, **1D Temporal Transformer**, and **Source-Agnostic DAE** on LIGO detector strain without matched-filter templates.
""")

# ----------------- Sidebar Controls -----------------
st.sidebar.header("🕹️ Experiment Controls")

model_choice = st.sidebar.selectbox(
    "1. Select Active AI Model:",
    ["1D CNN / TCN Baseline", "1D Temporal Transformer", "Source-Agnostic DAE"]
)

# Load selected model
active_model = None
if model_choice == "1D CNN / TCN Baseline":
    active_model = load_tcn_model()
elif model_choice == "1D Temporal Transformer":
    active_model = load_transformer_model()
elif model_choice == "Source-Agnostic DAE":
    active_model = load_dae_model()

dataset_choice = st.sidebar.selectbox(
    "2. Select Testing Condition:", 
    [
        "Dataset A (Standard Stationary BBH)", 
        "Dataset B (Non-Stationary Noise Ramp)", 
        "Dataset C (Transient Sine-Gaussian Glitches)", 
        "Dataset D (Low SNR / Whispers: 4-7)", 
        "Dataset E (Unseen Binary Neutron Stars)"
    ]
)

folder_map = {
    "Dataset A (Standard Stationary BBH)": "dataset_A",
    "Dataset B (Non-Stationary Noise Ramp)": "dataset_B",
    "Dataset C (Transient Sine-Gaussian Glitches)": "dataset_C",
    "Dataset D (Low SNR / Whispers: 4-7)": "dataset_D",
    "Dataset E (Unseen Binary Neutron Stars)": "dataset_E"
}
selected_folder = folder_map[dataset_choice]

col_btn1, col_btn2 = st.sidebar.columns(2)
if col_btn1.button("🎲 Random Sample", use_container_width=True):
    st.session_state['sample_idx'] = np.random.randint(0, 500)
if col_btn2.button("⭐ Sample #12", use_container_width=True):
    st.session_state['sample_idx'] = 12

if 'sample_idx' not in st.session_state:
    st.session_state['sample_idx'] = 12

# ----------------- Navigation Tabs -----------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📡 Single-Model Interactive Denoising",
    "⚔️ Side-by-Side Model Comparison", 
    "🌟 Real Historical Event (GW150914)", 
    "📊 Empirical Benchmark Matrix", 
    "🧠 Architectures & Theory",
    "🎛️ Hyperparameter Optimization"
])

# =========================================================================
# TAB 1: Single-Model Interactive Denoising
# =========================================================================
with tab1:
    data = load_dataset(selected_folder)
    if data is None:
        st.error(f"Dataset '{selected_folder}' not found. Please verify dataset files.")
    elif active_model is None:
        st.warning(f"Checkpoint for '{model_choice}' is currently training or not yet saved. Please select a trained model from the sidebar.")
    else:
        idx = st.session_state['sample_idx']
        noisy_input, clean_target = data[idx]

        with torch.no_grad():
            x = torch.tensor(noisy_input).unsqueeze(0).unsqueeze(0)
            pred_output = active_model(x).squeeze().numpy()

        overlap = compute_overlap(pred_output, clean_target)
        mse = float(np.mean((pred_output - clean_target)**2))

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Active Architecture", model_choice.split('/')[0].strip())
        m2.metric("Testing Condition", dataset_choice.split('(')[1].split(')')[0])
        m3.metric("Waveform Overlap Match", f"{overlap:.2f}%")
        m4.metric("Mean Squared Error (MSE)", f"{mse:.6f}")

        st.divider()

        time_axis = np.linspace(0, 1.0, 4096)
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(13, 7.5), sharex=True)

        ax1.plot(time_axis, noisy_input, color='#666666', lw=1.0, label='Raw LIGO Detector Strain (X)')
        ax1.set_title("Input: Raw LIGO Strain (Hidden Waveform + Detector Noise)", fontsize=11, fontweight='bold')
        ax1.set_ylabel("Strain")
        ax1.grid(True, alpha=0.25)
        ax1.legend(loc="upper right")

        ax2.plot(time_axis, pred_output, color='#0066cc', lw=1.5, label=f'{model_choice} Output')
        ax2.set_title(f"Output: AI Denoised Waveform Reconstruction ({model_choice})", fontsize=11, fontweight='bold')
        ax2.set_ylabel("Strain")
        ax2.grid(True, alpha=0.25)
        ax2.legend(loc="upper right")

        ax3.plot(time_axis, clean_target, color='#2e7d32', lw=1.5, label='Ground Truth Target (Y_true)')
        ax3.set_title("Ground Truth: Target Gravitational Wave Chirp", fontsize=11, fontweight='bold')
        ax3.set_ylabel("Strain")
        ax3.set_xlabel("Time (seconds)", fontsize=10)
        ax3.grid(True, alpha=0.25)
        ax3.legend(loc="upper right")

        plt.tight_layout()
        st.pyplot(fig)

        col_audio, col_spec_btn = st.columns([1, 1])
        with col_audio:
            st.markdown("#### 🎧 Audio Chirp Synthesis")
            st.caption("Gravitational waves vibrate in the human audio range (30 - 300 Hz). Hear the difference:")
            a_sub1, a_sub2 = st.columns(2)
            with a_sub1:
                st.caption("Input (Noise)")
                st.audio(to_audio_bytes(noisy_input), format='audio/wav')
            with a_sub2:
                st.caption(f"Clean Chirp ({model_choice.split()[0]})")
                st.audio(to_audio_bytes(pred_output), format='audio/wav')

        with st.expander("📈 View Time-Frequency Spectrogram (STFT)", expanded=False):
            fig_spec, axes_spec = plt.subplots(1, 3, figsize=(16, 4.5))
            spec_signals = [
                ("1. Raw Detector Input", noisy_input),
                (f"2. {model_choice} Output ({overlap:.1f}%)", pred_output),
                ("3. Ground Truth Signal", clean_target)
            ]
            for ax, (title, sig) in zip(axes_spec, spec_signals):
                f, t_s, Sxx = spectrogram(sig, fs=4096, nperseg=256, noverlap=220)
                mask = (f >= 20) & (f <= 450)
                im = ax.pcolormesh(t_s, f[mask], 10 * np.log10(Sxx[mask, :] + 1e-10), cmap='inferno', shading='gouraud')
                ax.set_title(title, fontsize=11, fontweight='bold')
                ax.set_xlabel('Time [s]')
                ax.set_ylabel('Freq [Hz]')
                fig_spec.colorbar(im, ax=ax, label='Power (dB)')
            plt.tight_layout()
            st.pyplot(fig_spec)

# =========================================================================
# TAB 2: Side-by-Side Model Comparison
# =========================================================================
with tab2:
    st.subheader("⚔️ Multi-Architecture Head-to-Head Comparison")
    st.markdown("Compare predictions from the **1D CNN/TCN Baseline**, **Temporal Transformer**, and **Source-Agnostic DAE** on the exact same noisy detector sample:")

    m_tcn = load_tcn_model()
    m_tf = load_transformer_model()
    m_dae = load_dae_model()

    data = load_dataset(selected_folder)
    if data is not None:
        idx = st.session_state['sample_idx']
        noisy_input, clean_target = data[idx]

        x_tensor = torch.tensor(noisy_input).unsqueeze(0).unsqueeze(0)
        time_axis = np.linspace(0, 1.0, 4096)

        preds = {}
        if m_tcn:
            with torch.no_grad():
                preds["1D CNN / TCN"] = m_tcn(x_tensor).squeeze().numpy()
        if m_tf:
            with torch.no_grad():
                preds["1D Transformer"] = m_tf(x_tensor).squeeze().numpy()
        if m_dae:
            with torch.no_grad():
                preds["Source-Agnostic DAE"] = m_dae(x_tensor).squeeze().numpy()

        if preds:
            cols = st.columns(len(preds))
            for col, (name, p) in zip(cols, preds.items()):
                ov = compute_overlap(p, clean_target)
                mse_val = float(np.mean((p - clean_target)**2))
                col.metric(name, f"{ov:.1f}% Overlap", f"MSE: {mse_val:.5f}")

            # Plot overlay
            fig_cmp, ax_cmp = plt.subplots(figsize=(14, 5))
            ax_cmp.plot(time_axis, clean_target, color='green', lw=2.0, alpha=0.9, label='Target Clean GW')
            colors = {'1D CNN / TCN': '#0066cc', '1D Transformer': '#ff7f0e', 'Source-Agnostic DAE': '#9467bd'}
            for name, p in preds.items():
                ax_cmp.plot(time_axis, p, color=colors.get(name, 'blue'), lw=1.5, alpha=0.8, linestyle='--', label=f'{name}')
            ax_cmp.set_title(f"Model Reconstruction Overlay on {dataset_choice} (Sample #{idx})", fontsize=12, fontweight='bold')
            ax_cmp.set_xlabel("Time (seconds)")
            ax_cmp.set_ylabel("Strain")
            ax_cmp.grid(True, alpha=0.3)
            ax_cmp.legend(loc="upper right")
            plt.tight_layout()
            st.pyplot(fig_cmp)
        else:
            st.info("Train the models to view head-to-head overlays.")

# =========================================================================
# TAB 3: Real Historical Event (GW150914)
# =========================================================================
with tab3:
    st.subheader("Zero-Shot Reconstruction on Real Cosmic Data: GW150914")
    st.markdown("""
    On **September 14, 2015 at 09:50:45 UTC**, LIGO Hanford (H1) recorded humanity's first direct gravitational wave detection from two colliding black holes.
    Here, the trained models perform zero-shot inference directly on the real un-injected detector recording (`data/raw/GW150914_H1.hdf5`) without any matched-filter templates.
    """)

    real_img_path = 'results/figures/gw150914_reconstruction.png'
    if os.path.exists(real_img_path):
        st.image(real_img_path, caption="Zero-Shot Denoising of GW150914 Recorded at LIGO Hanford (H1)", use_container_width=True)

# =========================================================================
# TAB 4: Robustness Benchmark Matrix
# =========================================================================
with tab4:
    st.subheader("Empirical Multi-Model Robustness Benchmark Matrix (500 Samples Each)")
    st.markdown("Comparative performance of **Waveform-Supervised (TCN, Transformer)** and **Source-Agnostic (DAE)** reconstructors:")

    barchart_path = 'results/figures/multi_model_benchmark_barchart.png'
    if os.path.exists(barchart_path):
        st.image(barchart_path, caption="Waveform Overlap Comparison Across All 5 Conditions", use_container_width=True)

    cmp_data = load_all_models_comparison()
    if cmp_data:
        datasets_list = [
            'Dataset A (Stationary BBH)',
            'Dataset B (Non-Stationary)',
            'Dataset C (Glitches)',
            'Dataset D (Low SNR / Whispers)',
            'Dataset E (Binary Neutron Stars)'
        ]
        
        table_rows = []
        for ds in datasets_list:
            row = {"Benchmark Condition": ds}
            for model_name, m_info in cmp_data.items():
                short_name = model_name.replace(" Baseline", "").replace(" Reconstructor", "")
                ds_info = m_info['datasets'].get(ds, {})
                ov = ds_info.get('overlap_percent', 0.0)
                std_ov = ds_info.get('std_overlap', 0.0)
                mse_val = ds_info.get('mse', 0.0)
                row[f"{short_name} Match (%)"] = f"{ov:.2f}% (±{std_ov:.1f}%)"
                row[f"{short_name} MSE"] = f"{mse_val:.5f}"
            table_rows.append(row)
        
        st.table(table_rows)

        st.markdown("### 🏆 Architectural Trade-Offs & Insights:")
        st.markdown("""
        * **1. Stationary Noise (Dataset A)**: **Source-Agnostic DAE leads (85.52%)**, followed closely by the **1D Transformer (84.91%)** and **TCN (78.09%)**. Deep bottleneck feature learning captures clean oscillatory structure with minimal error.
        * **2. Transient Glitches (Dataset C)**: **TCN Baseline wins decisively (66.72%)** over Transformer (49.82%). Dilated causal convolutions remain strictly localized, preventing glitch energy from contaminating distant time samples, whereas Transformer self-attention queries attend to the high-energy burst across all tokens!
        * **3. Sub-Threshold Faint Signals (Dataset D)**: **1D Transformer outperforms (51.60% vs 35.48%)**. Global multi-head self-attention pools weak signal energy across the entire 1-second duration.
        * **4. Inference Speed**: **TCN is ~3.8x faster (2.80 ms)** than Transformer (10.85 ms), making TCN ideal for low-latency gravitational wave alerts.
        """)

# =========================================================================
# TAB 5: Architectures & Theory
# =========================================================================
with tab5:
    st.subheader("Theoretical Formulation of Implemented Reconstructors")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(r"""
        ### 1. 1D CNN / TCN Baseline
        - **Paradigm**: Waveform-Supervised
        - **Core Layer**: Dilated Causal Conv1D
        - **Dilations**: $2^i \in \{1, 2, 4, 8, 16, 32\}$
        - **Receptive Field**: 757 time samples (~0.185 s)
        - **Parameters**: ~594K
        - **Key Strength**: Ultra-fast GPU inference, deterministic temporal causality.
        """)

    with c2:
        st.markdown("""
        ### 2. 1D Temporal Transformer
        - **Paradigm**: Waveform-Supervised
        - **Core Layer**: Multi-Head Self-Attention
        - **Tokenization**: 16-sample non-overlapping patches (256 tokens)
        - **Attention Heads**: 4 heads, $d_{model}=128$
        - **Parameters**: ~941K
        - **Key Strength**: Global receptive field capturing long-range frequency chirp drift.
        """)

    with c3:
        st.markdown("""
        ### 3. Source-Agnostic DAE
        - **Paradigm**: Source-Agnostic / Unsupervised
        - **Core Strategy**: Denoising Autoencoder + Masked Reconstruction
        - **Architecture**: 1D U-Net Convolutional Bottleneck
        - **Parameters**: ~1.65M
        - **Key Strength**: Does not require analytic binary chirp templates; handles arbitrary burst morphologies.
        """)

# =========================================================================
# TAB 6: Hyperparameter Optimization & Ablation Studies
# =========================================================================
with tab6:
    st.subheader("🎛️ Systematic Multi-Model Hyperparameter Optimization")
    st.markdown("""
    To eliminate heuristic guessing and achieve optimal signal recovery, we performed a 
    **16-trial systematic hyperparameter ablation sweep** across all 3 architectures:
    """)

    tuning_plot_path = 'results/figures/hyperparameter_tuning_analysis.png'
    if os.path.exists(tuning_plot_path):
        st.image(tuning_plot_path, caption="Hyperparameter Sensitivity & Overlap Ablation Across Architectures", use_container_width=True)

    tuning_json_path = 'results/logs/hyperparameter_tuning_results.json'
    if os.path.exists(tuning_json_path):
        with open(tuning_json_path, 'r') as f:
            tdata = json.load(f)

        col_t1, col_t2 = st.columns([1, 1])

        with col_t1:
            st.markdown("#### 1. 1D CNN / TCN Tuning (Kernel Size & Regularization)")
            tcn_rows = []
            for t in tdata.get('tcn_trials', []):
                tcn_rows.append({
                    "Trial Configuration": t['config_name'],
                    "Kernel Size (K)": t['hyperparameters']['kernel_size'],
                    "Learning Rate": t['hyperparameters']['lr'],
                    "Dropout": t['hyperparameters']['dropout'],
                    "Val Overlap (%)": f"{t['best_val_overlap']:.2f}%",
                    "Val MSE": f"{t['best_val_loss']:.5f}"
                })
            st.table(tcn_rows)

            st.markdown("#### 2. Source-Agnostic DAE Tuning (Mask Corruption Ratio)")
            dae_rows = []
            for t in tdata.get('dae_trials', []):
                dae_rows.append({
                    "Trial Configuration": t['config_name'],
                    "Mask Ratio": f"{t['hyperparameters']['mask_ratio']*100:.0f}%",
                    "Learning Rate": t['hyperparameters']['lr'],
                    "Val Overlap (%)": f"{t['best_val_overlap']:.2f}%",
                    "Val MSE": f"{t['best_val_loss']:.5f}"
                })
            st.table(dae_rows)

        with col_t2:
            st.markdown("#### 3. 1D Temporal Transformer Tuning (Patch Tokenizer & Depth)")
            tf_rows = []
            for t in tdata.get('transformer_trials', []):
                tf_rows.append({
                    "Trial Configuration": t['config_name'],
                    "Patch Size (P)": t['hyperparameters']['patch_size'],
                    "Tokens": 4096 // t['hyperparameters']['patch_size'],
                    "Layers": t['hyperparameters']['num_layers'],
                    "Val Overlap (%)": f"{t['best_val_overlap']:.2f}%",
                    "Val MSE": f"{t['best_val_loss']:.5f}"
                })
            st.table(tf_rows)

            st.success("""
            **Optimal Discovered Configurations:**
            - **TCN:** Kernel $K=7$, LR $1\\times 10^{-3}$, Dropout $0.1$
            - **Transformer:** Patch $P=16$ (256 tokens), $d=128$, 4 Layers, LR $1\\times 10^{-3}$
            - **DAE:** Mask Ratio $10\\%$, Bottleneck $256$, LR $1\\times 10^{-3}$
            """)
    else:
        st.info("Hyperparameter tuning sweep is currently executing in the background. Please wait for completion.")
