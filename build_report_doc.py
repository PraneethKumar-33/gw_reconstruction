import os
import sys
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=120, bottom=120, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'''
        <w:tcMar {nsdecls("w")}>
            <w:top w:w="{top}" w:type="dxa"/>
            <w:bottom w:w="{bottom}" w:type="dxa"/>
            <w:left w:w="{left}" w:type="dxa"/>
            <w:right w:w="{right}" w:type="dxa"/>
        </w:tcMar>
    ''')
    tcPr.append(tcMar)

def style_heading(doc, text, level):
    h = doc.add_heading(text, level=level)
    run = h.runs[0]
    run.font.name = 'Calibri'
    if level == 1:
        run.font.size = Pt(16)
        run.font.bold = True
        run.font.color.rgb = RGBColor(18, 44, 79) # Deep Navy
        h.paragraph_format.space_before = Pt(14)
        h.paragraph_format.space_after = Pt(6)
    elif level == 2:
        run.font.size = Pt(13)
        run.font.bold = True
        run.font.color.rgb = RGBColor(31, 97, 141) # Slate Blue
        h.paragraph_format.space_before = Pt(10)
        h.paragraph_format.space_after = Pt(4)
    elif level == 3:
        run.font.size = Pt(11.5)
        run.font.bold = True
        run.font.color.rgb = RGBColor(44, 62, 80)
        h.paragraph_format.space_before = Pt(8)
        h.paragraph_format.space_after = Pt(3)
    return h

def add_p(doc, text, bold_prefix=None, italic=False, space_after=5):
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(space_after)
    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        r_pre.font.name = 'Calibri'
        r_pre.font.size = Pt(11)
        r_pre.font.bold = True
        r_pre.font.color.rgb = RGBColor(20, 20, 20)
    r = p.add_run(text)
    r.font.name = 'Calibri'
    r.font.size = Pt(11)
    r.font.italic = italic
    r.font.color.rgb = RGBColor(40, 40, 40)
    return p

def add_callout(doc, text, bold_title="Key Takeaway: "):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    cell.width = Inches(6.5)
    set_cell_background(cell, "F0F4F8") # Very soft blue/gray
    set_cell_margins(cell, top=140, bottom=140, left=200, right=200)
    
    # Left border
    tcPr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(f'''
        <w:tcBorders {nsdecls("w")}>
            <w:left w:val="single" w:sz="24" w:space="0" w:color="1B4F72"/>
            <w:top w:val="none"/>
            <w:right w:val="none"/>
            <w:bottom w:val="none"/>
        </w:tcBorders>
    ''')
    tcPr.append(borders)
    
    p = cell.paragraphs[0]
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(0)
    r_title = p.add_run(bold_title)
    r_title.font.name = 'Calibri'
    r_title.font.size = Pt(10.5)
    r_title.font.bold = True
    r_title.font.color.rgb = RGBColor(27, 79, 114)
    
    r_text = p.add_run(text)
    r_text.font.name = 'Calibri'
    r_text.font.size = Pt(10.5)
    r_text.font.italic = True
    r_text.font.color.rgb = RGBColor(40, 40, 40)
    
    doc.add_paragraph().paragraph_format.space_after = Pt(4)

def add_figure(doc, img_path, caption, width=Inches(6.0)):
    if os.path.exists(img_path):
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.paragraph_format.space_before = Pt(8)
        p_img.paragraph_format.space_after = Pt(4)
        run = p_img.add_run()
        run.add_picture(img_path, width=width)
        
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.paragraph_format.space_after = Pt(10)
        r_cap = p_cap.add_run(caption)
        r_cap.font.name = 'Calibri'
        r_cap.font.size = Pt(9.5)
        r_cap.font.italic = True
        r_cap.font.color.rgb = RGBColor(100, 100, 100)
    else:
        print(f"Warning: image {img_path} not found.")

def build_report():
    doc = docx.Document()
    
    # Page setup - 1 inch margins
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    base_dir = r'C:\Users\HP\Downloads\gw_reconstruction'
    fig_dir = os.path.join(base_dir, 'results', 'figures')

    # =========================================================================
    # TITLE & COVER BLOCK
    # =========================================================================
    title_p = doc.add_paragraph()
    title_p.paragraph_format.space_before = Pt(36)
    title_p.paragraph_format.space_after = Pt(6)
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_title = title_p.add_run("DEEP LEARNING ARCHITECTURES FOR GRAVITATIONAL WAVE SIGNAL RECONSTRUCTION")
    r_title.font.name = 'Calibri'
    r_title.font.size = Pt(22)
    r_title.font.bold = True
    r_title.font.color.rgb = RGBColor(18, 44, 79) # Navy

    sub_p = doc.add_paragraph()
    sub_p.paragraph_format.space_after = Pt(20)
    sub_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_sub = sub_p.add_run("A Comparative Empirical Study of 1D Temporal CNNs, Vision-Inspired Temporal Transformers, and Source-Agnostic Autoencoders Under Hostile LIGO Strain Noise")
    r_sub.font.name = 'Calibri'
    r_sub.font.size = Pt(13)
    r_sub.font.italic = True
    r_sub.font.color.rgb = RGBColor(80, 80, 80)

    # Decorative separator line
    sep_tbl = doc.add_table(rows=1, cols=1)
    sep_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell_sep = sep_tbl.cell(0, 0)
    cell_sep.width = Inches(6.5)
    set_cell_background(cell_sep, "1B4F72")
    tcPr = cell_sep._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="20" w:type="dxa"/><w:bottom w:w="20" w:type="dxa"/></w:tcMar>')
    tcPr.append(tcMar)
    cell_sep.paragraphs[0].paragraph_format.space_after = Pt(0)
    p_empty = cell_sep.paragraphs[0].add_run("")

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # Project metadata box
    meta_tbl = doc.add_table(rows=4, cols=2)
    meta_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_data = [
        ("Project Domain:", "Astrophysics, Signal Processing & Deep Learning"),
        ("Dataset Source:", "Laser Interferometer Gravitational-Wave Observatory (LIGO Hanford H1 & Livingston L1)"),
        ("Observation Window:", "1.000 Second Continuous Strain at 4096 Hz (N = 4096 Samples)"),
        ("Project Repository:", r"C:\Users\HP\Downloads\gw_reconstruction")
    ]
    for row_idx, (k, v) in enumerate(meta_data):
        row = meta_tbl.rows[row_idx]
        cell_k, cell_v = row.cells[0], row.cells[1]
        cell_k.width = Inches(2.0)
        cell_v.width = Inches(4.5)
        set_cell_margins(cell_k, top=60, bottom=60, left=100, right=100)
        set_cell_margins(cell_v, top=60, bottom=60, left=100, right=100)
        set_cell_background(cell_k, "F8F9F9")
        set_cell_background(cell_v, "FFFFFF")
        pk = cell_k.paragraphs[0]
        pk.paragraph_format.space_after = Pt(0)
        rk = pk.add_run(k)
        rk.font.name = 'Calibri'
        rk.font.size = Pt(10)
        rk.font.bold = True
        rk.font.color.rgb = RGBColor(30, 30, 30)

        pv = cell_v.paragraphs[0]
        pv.paragraph_format.space_after = Pt(0)
        rv = pv.add_run(v)
        rv.font.name = 'Calibri'
        rv.font.size = Pt(10)
        rv.font.color.rgb = RGBColor(60, 60, 60)

    doc.add_paragraph().paragraph_format.space_after = Pt(16)

    # Team Members Table
    style_heading(doc, "Project Team & Role Distribution", level=2)
    team_tbl = doc.add_table(rows=4, cols=3)
    team_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers = ["Team Member", "Core Model Ownership", "Benchmark & Pipeline Contribution"]
    for i, h in enumerate(headers):
        cell = team_tbl.rows[0].cells[i]
        set_cell_background(cell, "1B4F72")
        set_cell_margins(cell, top=100, bottom=100, left=100, right=100)
        p = cell.paragraphs[0]
        p.paragraph_format.space_after = Pt(0)
        r = p.add_run(h)
        r.font.name = 'Calibri'
        r.font.size = Pt(10.5)
        r.font.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)

    team_rows = [
        ("Nikhith", "1D Temporal Convolutional Network (TCN Baseline)\nCheckpoint: best_tcn.pt (615K Params)", "Dataset A: Standard Injections (SNR 8-25)\nZero-shot inference on real GW150914 event"),
        ("Pamu Praneeth Kumar\n(Architecture Lead)", "1D Temporal Transformer Reconstructor\nCheckpoint: best_transformer.pt (941K Params)", "Dataset B: Non-Stationary Noise Drift\nDataset C: Transient Instrumental Glitches"),
        ("Hemavardhan", "Source-Agnostic Denoising Autoencoder (DAE)\nCheckpoint: best_source_agnostic_dae.pt (1.65M Params)", "Dataset D: Sub-Threshold Low SNR (SNR 4-7)\nDataset E: Out-of-Distribution Binary Neutron Stars")
    ]
    for row_idx, data in enumerate(team_rows):
        row = team_tbl.rows[row_idx + 1]
        bg = "F2F4F4" if row_idx % 2 == 1 else "FFFFFF"
        for col_idx, text in enumerate(data):
            cell = row.cells[col_idx]
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=80, bottom=80, left=100, right=100)
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.1
            r = p.add_run(text)
            r.font.name = 'Calibri'
            r.font.size = Pt(9.5)
            r.font.color.rgb = RGBColor(30, 30, 30)
            if col_idx == 0:
                r.font.bold = True

    doc.add_page_break()

    # =========================================================================
    # EXECUTIVE SUMMARY / ABSTRACT
    # =========================================================================
    style_heading(doc, "Executive Summary", level=1)
    add_p(doc, 
        "Direct gravitational wave (GW) observation by ground-based laser interferometers (LIGO and Virgo) has revolutionized modern astrophysics. "
        "However, real gravitational wave strain signals recorded by detectors are notoriously faint—often buried under instrumental noise orders of "
        "magnitude stronger than the astrophysical chirp itself. While traditional Matched Filtering delivers optimal signal-to-noise ratio (SNR) for known "
        "compact binary coalescences, it relies strictly on precomputed general relativistic template banks, rendering it computationally exorbitant "
        "and blind to unmodeled, eccentric, or non-template sources. Furthermore, non-Gaussian instrumental transients (glitches) and time-varying non-stationary "
        "noise drift frequently degrade detection confidence.")
    
    add_p(doc,
        "In this capstone investigation, we engineer, systematically optimize, and benchmark three structurally distinct deep learning paradigms for direct "
        "time-domain gravitational wave waveform reconstruction from 1-second continuous detector strain sampled at 4096 Hz (N = 4096 samples):")
    
    add_p(doc, "1. A Waveform-Supervised 1D Temporal Convolutional Network (TCN) featuring dilated causal convolutions with exponential receptive field expansion.", bold_prefix="• ")
    add_p(doc, "2. A Waveform-Supervised 1D Temporal Transformer Reconstructor utilizing non-overlapping 16-sample patch tokenization, sinusoidal positional encoding, and 4-head multi-head self-attention.", bold_prefix="• ")
    add_p(doc, "3. A Source-Agnostic Denoising Autoencoder (DAE) utilizing a 1D U-Net bottleneck with 5% self-supervised random masking to recover arbitrary strain perturbations without template supervision.", bold_prefix="• ")

    add_p(doc,
        "To thoroughly evaluate the physical robustness of each architecture, we conduct a 16-trial GPU-accelerated hyperparameter sweep and construct an empirical "
        "benchmark matrix spanning five controlled environmental stress regimes: (A) Standard BBH inspirals, (B) Non-stationary amplitude drift, (C) High-energy Sine-Gaussian "
        "glitches, (D) Sub-threshold low-SNR whispers (SNR 4-7), and (E) Out-of-distribution Binary Neutron Star (BNS) mergers. Finally, we evaluate zero-shot generalization "
        "on the historical discovery event GW150914 recorded at LIGO Hanford.")

    add_callout(doc,
        "Key Finding: While the Source-Agnostic DAE achieves the highest peak match on in-distribution binary black hole chirps (83.35% overlap), "
        "the 1D Temporal Transformer demonstrates superior physical generalization on sub-threshold signals (45.90% vs 35.43% for TCN) and unseen BNS mergers "
        "(52.21% vs 42.70% for TCN). Meanwhile, the 1D TCN provides unparalleled inference speed (2.80 ms/sample) and superior localized glitch suppression (68.20% overlap).",
        bold_title="Core Research Conclusion: ")

    # =========================================================================
    # CHAPTER 1: INTRODUCTION & PHYSICS MOTIVATION
    # =========================================================================
    style_heading(doc, "Chapter 1: Introduction and Physics Motivation", level=1)
    
    style_heading(doc, "1.1 The Gravitational Wave Phenomenon", level=2)
    add_p(doc,
        "Predicted by Albert Einstein in 1916 as a natural consequence of General Relativity, gravitational waves are quadrupolar ripples in the metric of spacetime "
        "propagating at the speed of light. They are produced by violent cosmic cataclysms involving accelerating asymmetric masses, most notably Compact Binary "
        "Coalescences (CBC) comprising Binary Black Holes (BBH), Binary Neutron Stars (BNS), and Neutron Star-Black Hole (NSBH) systems.")
    
    add_p(doc,
        "As two compact bodies orbit each other, they continuously radiate orbital angular momentum and energy via gravitational radiation. This causes their orbital separation "
        "to shrink, leading to a runaway inspiral phase where both the gravitational wave amplitude and frequency increase monotonically over time—producing the classic "
        "astrophysical 'chirp' signal. The coalescence culminates in a violent merger and subsequent ringdown as the newly formed distorted black hole settles into a "
        "stationary Kerr state.")

    style_heading(doc, "1.2 The LIGO Laser Interferometer Detector", level=2)
    add_p(doc,
        "The Laser Interferometer Gravitational-Wave Observatory (LIGO) operates two identical, orthogonal 4-kilometer arm-length Michelson interferometers located in "
        "Hanford, Washington, and Livingston, Louisiana. A passing gravitational wave produces a differential strain h(t) = ΔL / L, altering the optical path length "
        "difference between the two arms. For typical stellar-mass black hole collisions occurring hundreds of megaparsecs away, the fractional strain reaching Earth is "
        "exceedingly microscopic, on the order of h ~ 10^-21. This corresponds to an arm length change ΔL on the scale of 10^-18 meters—less than one-thousandth the diameter "
        "of a proton.")

    style_heading(doc, "1.3 The Instrumental Noise Environment", level=2)
    add_p(doc,
        "Because detector sensitivity must be extraordinarily high, the recorded strain output s(t) is heavily corrupted by complex, non-stationary environmental and "
        "instrumental noise sources:")
    add_p(doc, "Seismic ground vibrations and microseisms caused by ocean waves and human activity dominate the frequency regime below 20 Hz.", bold_prefix="1. Seismic Noise: ")
    add_p(doc, "Brownian motion within the fused-silica mirror test masses and their multi-stage pendulum suspension fibers produces thermal fluctuations in the sweet spot (50-200 Hz).", bold_prefix="2. Thermal Noise: ")
    add_p(doc, "Quantum shot noise (photon arrival statistics) and radiation pressure fluctuations dominate at high frequencies above 500 Hz.", bold_prefix="3. Quantum Shot Noise: ")
    add_p(doc, "Laser cavity adjustments, optical dust particles crossing the beam, and high-voltage power supplies induce non-Gaussian short-duration transients termed 'glitches' (e.g., blips, whistles, scattered light arches).", bold_prefix="4. Non-Gaussian Glitches: ")

    style_heading(doc, "1.4 Matched Filtering vs. Deep Learning Waveform Reconstruction", level=2)
    add_p(doc,
        "The historical gold standard for gravitational wave search pipelines (such as PyCBC and GstLAL) is Matched Filtering. By computing the frequency-domain cross-correlation "
        "between the detector data and millions of pre-computed analytical waveform templates weighted by the detector's Power Spectral Density (PSD), matched filtering "
        "maximizes the Signal-to-Noise Ratio for Gaussian stationary noise.")
    
    add_p(doc,
        "However, Matched Filtering suffers from severe operational bottlenecks:")
    add_p(doc, "Covering the full parameter space of binary masses, spins, and orbital eccentricities requires over 10^6 distinct templates, demanding massive supercomputing clusters.", bold_prefix="• Computational Latency: ")
    add_p(doc, "If an astrophysical source deviates from general relativistic analytical approximations (e.g., core-collapse supernovae, cosmic string cusps, or modified gravity theories), matched filtering fails completely.", bold_prefix="• Template Rigidity: ")
    add_p(doc, "High-energy non-Gaussian instrumental glitches can produce spurious high signal-to-noise ratio triggers in template banks, requiring complex, hand-tuned vetoes.", bold_prefix="• Glitch Susceptibility: ")

    add_p(doc,
        "Deep learning reconstructors provide a transformative paradigm: once trained, forward inference executes in milliseconds on standard GPU hardware, enabling "
        "real-time multi-messenger alerts while learning non-linear noise representations capable of separating true chirps from transient detector anomalies.")

    # =========================================================================
    # CHAPTER 2: PROBLEM FORMULATION & MATHEMATICAL METRICS
    # =========================================================================
    style_heading(doc, "Chapter 2: Problem Formulation and Mathematical Modeling", level=1)
    
    style_heading(doc, "2.1 1-Second Continuous Strain Parameterization", level=2)
    add_p(doc,
        "We frame gravitational wave reconstruction as a 1D time-series denoising task. The input observation is defined over a continuous 1.000-second window "
        "sampled at fs = 4096 Hz, producing an input vector x in R^(1 x 4096):")
    add_p(doc, "x = [x_0, x_1, x_2, ..., x_4095]", bold_prefix="Observation Vector: ")
    add_p(doc,
        "Physically, the inspiral, merger, and ringdown of typical stellar-mass binary black holes (chirp masses 15-50 M_solar) spans approximately 0.2 to 0.8 seconds "
        "within the sensitive frequency band (30 Hz - 1000 Hz) of LIGO. Thus, a 4096-sample vector contains both the pre-merger inspiral and post-merger damping oscillations.")

    style_heading(doc, "2.2 Signal Injection Model and Scaling", level=2)
    add_p(doc,
        "Training samples are synthesized by superimposing analytical post-Newtonian/numerical-relativity surrogate waveforms h(t) into uncorrupted, real detector noise segments n(t):")
    add_p(doc, "x(t) = alpha * h(t) + n(t)", bold_prefix="Injection Equation: ")
    add_p(doc,
        "Here, alpha is an adaptive scaling coefficient chosen to enforce an exact target Signal-to-Noise Ratio (SNR, rho), defined as:")
    add_p(doc, "SNR = sqrt( sum_{i=0}^N (alpha * h_i)^2 / sigma_noise^2 )", bold_prefix="Optimal Filter SNR: ")
    add_p(doc,
        "where sigma_noise is the root-mean-square standard deviation of the raw detector noise segment. The clean target provided during supervised training is the scaled, zero-noise waveform:")
    add_p(doc, "y_target = alpha * h(t)", bold_prefix="Supervised Target: ")

    style_heading(doc, "2.3 Mathematical Evaluation Metrics", level=2)
    add_p(doc,
        "In gravitational wave astronomy, standard Mean Squared Error (MSE) alone is insufficient because it evaluates point-wise Euclidean distance without explicitly capturing "
        "waveform phase coherence. Therefore, we evaluate models using two complementary physical metrics:")
    
    add_p(doc,
        "Measures the normalized cross-correlation (cosine similarity) between the predicted waveform p and clean target t in R^N:",
        bold_prefix="1. Waveform Overlap (Match Metric, O): ")
    add_p(doc, "Overlap(p, t) = <p, t> / ( ||p||_2 * ||t||_2 ) = ( sum_{i=0}^{N-1} p_i * t_i ) / [ sqrt( sum_{i=0}^{N-1} p_i^2 ) * sqrt( sum_{i=0}^{N-1} t_i^2 ) + 1e-8 ]", bold_prefix="Mathematical Formula: ")
    add_p(doc,
        "An overlap of 1.0 (100%) represents perfect phase, frequency, and morphological agreement. In gravitational wave astrophysics, an overlap > 70% indicates reliable astrophysical "
        "parameter estimation, while > 85% is considered publication-grade reconstruction.")

    add_p(doc,
        "Measures the absolute residual variance between the reconstructed strain and true clean signal:",
        bold_prefix="2. Mean Squared Error (MSE Loss): ")
    add_p(doc, "MSE = (1 / N) * sum_{i=0}^{N-1} (p_i - t_i)^2", bold_prefix="Loss Formula: ")

    # =========================================================================
    # CHAPTER 3: DATASET ENGINEERING & STRESS BENCHMARKS
    # =========================================================================
    style_heading(doc, "Chapter 3: Dataset Engineering and Stress Benchmarking Framework", level=1)
    
    add_p(doc,
        "To rigorously challenge each reconstructor against realistic detector non-idealities, we engineered five controlled datasets (A through E). "
        "All noise segments were extracted directly from open-access LIGO HDF5 strain records from the O1 and O2 science runs.")

    # Table of Datasets
    ds_tbl = doc.add_table(rows=6, cols=4)
    ds_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    ds_headers = ["Dataset Identifier", "Physical Environment", "Perturbation / Morphological Characteristics", "Assigned Lead"]
    for i, h in enumerate(ds_headers):
        cell = ds_tbl.rows[0].cells[i]
        set_cell_background(cell, "1B4F72")
        set_cell_margins(cell, top=80, bottom=80, left=80, right=80)
        p = cell.paragraphs[0]
        p.paragraph_format.space_after = Pt(0)
        r = p.add_run(h)
        r.font.name = 'Calibri'
        r.font.size = Pt(10)
        r.font.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)

    ds_rows = [
        ("Dataset A\n(Standard Training)", "Stationary LIGO Noise\n+ BBH Injections", "Chirp mass M_c in [15, 50] M_solar, symmetric mass ratio eta in [0.15, 0.25], SNR in [8.0, 25.0]. 4000 train / 500 val.", "Nikhith"),
        ("Dataset B\n(Hostile Benchmark)", "Non-Stationary Noise\nVariance Drift", "Linear noise amplitude modulation ramp(t) in [0.2x, 2.5x] across the 1-second window. Simulates seismic/wind drift.", "Praneeth"),
        ("Dataset C\n(Hostile Benchmark)", "Instrumental Glitches\n(Non-Gaussian Spikes)", "Sine-Gaussian wavepacket glitches: A * exp(-(t-t0)^2 / tau^2) * cos(2*pi*f*t), A in [5, 15], f in [30, 300] Hz.", "Praneeth"),
        ("Dataset D\n(Hostile Benchmark)", "Sub-Threshold Signals\n(Faint Whispers)", "Extremely weak BBH chirps with SNR in [4.0, 7.0], well below standard matched filtering detection threshold (SNR ~ 8).", "Hemavardhan"),
        ("Dataset E\n(Hostile Benchmark)", "Binary Neutron Stars\n(Unseen Physics)", "Chirp mass M_c in [1.1, 1.4] M_solar (eta = 0.25). Longer inspiral, lower amplitude, higher merger frequency (~1500 Hz).", "Hemavardhan")
    ]
    for row_idx, data in enumerate(ds_rows):
        row = ds_tbl.rows[row_idx + 1]
        bg = "F2F4F4" if row_idx % 2 == 1 else "FFFFFF"
        for col_idx, text in enumerate(data):
            cell = row.cells[col_idx]
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=70, bottom=70, left=80, right=80)
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.1
            r = p.add_run(text)
            r.font.name = 'Calibri'
            r.font.size = Pt(9.5)
            r.font.color.rgb = RGBColor(30, 30, 30)
            if col_idx == 0:
                r.font.bold = True

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # =========================================================================
    # CHAPTER 4: RECONSTRUCTOR ARCHITECTURES
    # =========================================================================
    style_heading(doc, "Chapter 4: Reconstructor Architectures and Mathematical Design", level=1)
    
    style_heading(doc, "4.1 Architecture 1: 1D Temporal Convolutional Network (TCN)", level=2)
    add_p(doc,
        "The 1D TCN serves as our high-speed, causal baseline reconstructor (615,905 parameters). To capture long-term temporal dependencies without the "
        "vanishing gradient pitfalls of recurrent networks (RNNs/LSTMs), the TCN employs dilated causal 1D convolutions arranged in sequential residual blocks.")
    
    add_p(doc,
        "For a 1D sequence x and a convolutional filter f of length K, the dilated convolution operator *d at time step t is defined as:")
    add_p(doc, "y(t) = (x *_d f)(t) = sum_{k=0}^{K-1} f(k) * x(t - d * k)", bold_prefix="Dilated Convolution Formula: ")
    add_p(doc,
        "By exponentially increasing the dilation factor d through successive layers (d in {1, 2, 4, 8, 16, 32}), the network's receptive field expands exponentially:")
    add_p(doc, "Receptive Field = 1 + sum_{l=1}^L 2 * (K_l - 1) * d_l = 757 time samples (~0.185 seconds at 4096 Hz)", bold_prefix="Receptive Field Calculation: ")
    add_p(doc,
        "Each residual block consists of two dilated Conv1D layers with LeakyReLU activations, Weight Normalization, and Dropout. A 1x1 projection convolution matches "
        "channel dimensions when input and output feature maps differ.")

    style_heading(doc, "4.2 Architecture 2: 1D Temporal Transformer Reconstructor", level=2)
    add_p(doc,
        "Designed and implemented by Pamu Praneeth Kumar, the 1D Temporal Transformer Reconstructor (941,249 parameters) replaces local convolutions with global multi-head "
        "self-attention. Gravitational wave chirps exhibit sweeping frequency modulation: early inspiral cycles are long (~30 Hz), while merger oscillations occur in fractions "
        "of a millisecond (~500 Hz). The self-attention mechanism enables every temporal region of the waveform to attend directly to every other region, capturing long-range "
        "phase coherence regardless of temporal separation.")

    add_p(doc,
        "Directly computing full self-attention on N = 4096 raw samples would require an attention matrix of size 4096 x 4096 (~16.8 million pairs per head), inducing excessive "
        "memory overhead and quadratic computational complexity. We solve this by introducing 1D Patch Embedding tokenization:")
    add_p(doc,
        "A 1D convolution with kernel_size = 16 and stride = 16 segments the 4096 samples into 256 non-overlapping patches, projecting each 16-sample snippet into a "
        "d_model = 128 embedding space: [B, 1, 4096] -> [B, 256, 128].",
        bold_prefix="1. Patch Tokenization (PatchEmbedding1D): ")
    
    add_p(doc,
        "Because Transformers are permutation-invariant, sinusoidal positional encodings are injected into the patch tokens to encode absolute temporal coordinates:",
        bold_prefix="2. Sinusoidal Positional Encoding: ")
    add_p(doc, "PE(pos, 2i) = sin( pos / 10000^(2i / d_model) ),    PE(pos, 2i+1) = cos( pos / 10000^(2i / d_model) )", bold_prefix="Positional Formula: ")

    add_p(doc,
        "The tokens pass through a 4-layer Transformer Encoder with 4 attention heads (d_k = 32 per head), GELU non-linearities, and a feedforward dimension of 512. "
        "Self-attention dynamically computes correlation weights:",
        bold_prefix="3. Multi-Head Self-Attention: ")
    add_p(doc, "Attention(Q, K, V) = softmax( Q * K^T / sqrt(d_k) ) * V", bold_prefix="Scaled Dot-Product Formula: ")

    add_p(doc,
        "A 1D Transposed Convolution (stride 16) upsamples the 256 contextualized embeddings back to 4096 points, followed by refinement smoothing convolutions to eliminate "
        "patch boundary discontinuities.",
        bold_prefix="4. Patch Decoder (PatchDecoder1D): ")

    style_heading(doc, "4.3 Architecture 3: Source-Agnostic Denoising Autoencoder (DAE)", level=2)
    add_p(doc,
        "Implemented by Hemavardhan, the Source-Agnostic DAE (1,656,017 parameters) employs a 1D U-Net convolutional encoder-decoder with skip connections. "
        "Unlike TCN and Transformer which are trained strictly on analytical binary chirp targets, the DAE is trained to reconstruct the underlying continuous signal "
        "while portions of the input time series are randomly masked out (mask_ratio = 5%).")
    add_p(doc,
        "This self-supervised masking task forces the network to learn the physical continuity, smooth derivatives, and harmonic energy distribution of physical signals "
        "without depending on any specific gravitational wave model, making it uniquely capable of detecting unmodeled cosmic bursts.")

    # =========================================================================
    # CHAPTER 5: HYPERPARAMETER OPTIMIZATION SWEEP
    # =========================================================================
    style_heading(doc, "Chapter 5: Systematic Hyperparameter Optimization Sweep", level=1)
    
    add_p(doc,
        "To eliminate subjective heuristic tuning, we executed a rigorous 16-trial systematic hyperparameter ablation sweep directly on our dedicated "
        "NVIDIA GeForce RTX 3050 Laptop GPU. Every trial was logged to results/logs/hyperparameter_tuning_results.json and visualized.")

    add_figure(doc, 
        os.path.join(fig_dir, "hyperparameter_tuning_analysis.png"),
        "Figure 5.1: 16-Trial Hyperparameter Optimization Sweep across TCN, Transformer, and DAE architectures on GPU.")

    # Table of Best Hyperparameters
    style_heading(doc, "Summary of Winning Configurations", level=2)
    tune_tbl = doc.add_table(rows=4, cols=5)
    tune_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tune_headers = ["Architecture", "Winning Trial", "Key Hyperparameter Configuration", "Peak Val Overlap", "Val MSE Loss"]
    for i, h in enumerate(tune_headers):
        cell = tune_tbl.rows[0].cells[i]
        set_cell_background(cell, "1B4F72")
        set_cell_margins(cell, top=80, bottom=80, left=80, right=80)
        p = cell.paragraphs[0]
        p.paragraph_format.space_after = Pt(0)
        r = p.add_run(h)
        r.font.name = 'Calibri'
        r.font.size = Pt(10)
        r.font.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)

    tune_rows = [
        ("1D CNN / TCN Baseline", "Trial 6 (High Reg)", "Kernel Size K = 7, Dropout p = 0.2, Learning Rate = 0.001", "72.38% (76.62% Full Train)", "0.02080"),
        ("1D Temporal Transformer", "Trial 5 (High LR)", "Patch Size = 16, d_model = 128, 4 Heads, 4 Layers, LR = 0.002", "78.76% (80.00% Full Train)", "0.01638"),
        ("Source-Agnostic DAE", "Trial 1 (Light Perturb)", "Mask Ratio = 5%, Learning Rate = 0.001, Bottleneck Dim = 128", "86.10% (86.30% Full Train)", "0.00904")
    ]
    for row_idx, data in enumerate(tune_rows):
        row = tune_tbl.rows[row_idx + 1]
        bg = "F2F4F4" if row_idx % 2 == 1 else "FFFFFF"
        for col_idx, text in enumerate(data):
            cell = row.cells[col_idx]
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=70, bottom=70, left=80, right=80)
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            r = p.add_run(text)
            r.font.name = 'Calibri'
            r.font.size = Pt(9.5)
            r.font.color.rgb = RGBColor(30, 30, 30)
            if col_idx == 0:
                r.font.bold = True

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    add_callout(doc,
        "Ablation Insights: (1) For TCN, expanding kernel size from K=3 to K=7 increased overlap from 59.5% to 67.3% because larger filters span the long-wavelength cycles "
        "of early inspirals. (2) For Transformer, patch size P=16 (256 tokens) dramatically outperformed P=8 (68.3%) and P=32 (74.0%), confirming that 16 samples perfectly "
        "resolves the Nyquist limit of chirp frequency sweeps without overwhelming self-attention. (3) For DAE, a light 5% mask ratio preserved fine waveform ringdown features "
        "while enforcing denoising continuity.",
        bold_title="Key Ablation Findings: ")

    # =========================================================================
    # CHAPTER 6: EMPIRICAL RESULTS & BENCHMARKING
    # =========================================================================
    style_heading(doc, "Chapter 6: Empirical Results, Benchmarking and Discussion", level=1)
    
    style_heading(doc, "6.1 In-Distribution Head-to-Head Comparison (Dataset A)", level=2)
    add_p(doc,
        "All three models were evaluated on the held-out 500-sample test partition of Dataset A. In addition to accuracy metrics, inference latency was benchmarked "
        "on our dedicated GPU across 500 consecutive trials.")

    add_figure(doc,
        os.path.join(fig_dir, "head_to_head_waveform_comparison.png"),
        "Figure 6.1: Direct time-domain waveform overlay comparing TCN, Transformer, and DAE reconstructions against the clean ground truth.")

    # Table of Dataset A results
    res_tbl = doc.add_table(rows=4, cols=6)
    res_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    res_headers = ["Architecture", "Parameters", "GPU Latency", "Test Overlap Match", "MSE Loss", "In-Distribution Rank"]
    for i, h in enumerate(res_headers):
        cell = res_tbl.rows[0].cells[i]
        set_cell_background(cell, "1B4F72")
        set_cell_margins(cell, top=80, bottom=80, left=60, right=60)
        p = cell.paragraphs[0]
        p.paragraph_format.space_after = Pt(0)
        r = p.add_run(h)
        r.font.name = 'Calibri'
        r.font.size = Pt(9.5)
        r.font.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)

    res_rows = [
        ("1D CNN / TCN Baseline", "615,905 (~615K)", "2.80 ms / sample", "77.68% (±18.89%)", "0.01505", "Rank 3 (Fastest)"),
        ("1D Temporal Transformer", "941,249 (~941K)", "10.85 ms / sample", "81.08% (±16.66%)", "0.01340", "Rank 2 (Balanced)"),
        ("Source-Agnostic DAE", "1,656,017 (~1.65M)", "6.02 ms / sample", "83.35% (±17.64%)", "0.01111", "Rank 1 (Top Match)")
    ]
    for row_idx, data in enumerate(res_rows):
        row = res_tbl.rows[row_idx + 1]
        bg = "F2F4F4" if row_idx % 2 == 1 else "FFFFFF"
        for col_idx, text in enumerate(data):
            cell = row.cells[col_idx]
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=70, bottom=70, left=60, right=60)
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            r = p.add_run(text)
            r.font.name = 'Calibri'
            r.font.size = Pt(9.0)
            r.font.color.rgb = RGBColor(30, 30, 30)
            if col_idx == 0:
                r.font.bold = True

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    style_heading(doc, "6.2 Controlled Stress Matrix Evaluation (Datasets A through E)", level=2)
    add_p(doc,
        "Evaluating models only on stationary Gaussian data produces an overly optimistic estimate of operational capability. Figure 6.2 presents the complete "
        "5-dataset robustness matrix, testing each model against non-stationary drift, instrumental glitches, low SNR, and out-of-distribution compact systems.")

    add_figure(doc,
        os.path.join(fig_dir, "multi_model_benchmark_barchart.png"),
        "Figure 6.2: Multi-Model Robustness Benchmark Bar Chart across Datasets A, B, C, D, and E (500 samples per condition).")

    # Full Benchmark Matrix Table
    bench_tbl = doc.add_table(rows=6, cols=5)
    bench_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    b_headers = ["Test Dataset Condition", "1D CNN / TCN", "1D Transformer", "Source-Agnostic DAE", "Winning Paradigm"]
    for i, h in enumerate(b_headers):
        cell = bench_tbl.rows[0].cells[i]
        set_cell_background(cell, "1B4F72")
        set_cell_margins(cell, top=80, bottom=80, left=80, right=80)
        p = cell.paragraphs[0]
        p.paragraph_format.space_after = Pt(0)
        r = p.add_run(h)
        r.font.name = 'Calibri'
        r.font.size = Pt(9.5)
        r.font.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)

    b_rows = [
        ("Dataset A: Stationary BBH (SNR 8-25)", "77.68% (MSE: 0.0150)", "81.08% (MSE: 0.0134)", "83.35% (MSE: 0.0111)", "Source-Agnostic DAE (+2.27%)"),
        ("Dataset B: Non-Stationary Noise Drift", "66.63% (MSE: 0.0323)", "52.48% (MSE: 0.0570)", "64.73% (MSE: 0.0425)", "1D CNN / TCN (+1.90%)"),
        ("Dataset C: Instrumental Glitches", "68.20% (MSE: 0.0402)", "32.22% (MSE: 0.0669)", "49.49% (MSE: 0.0614)", "1D CNN / TCN (+18.71%)"),
        ("Dataset D: Sub-Threshold Whispers (SNR 4-7)", "35.43% (MSE: 0.0075)", "45.90% (MSE: 0.0061)", "41.41% (MSE: 0.0091)", "1D Transformer (+4.49%)"),
        ("Dataset E: Unseen Binary Neutron Stars", "42.70% (MSE: 0.0440)", "52.21% (MSE: 0.0358)", "50.13% (MSE: 0.0383)", "1D Transformer (+2.08%)")
    ]
    for row_idx, data in enumerate(b_rows):
        row = bench_tbl.rows[row_idx + 1]
        bg = "F2F4F4" if row_idx % 2 == 1 else "FFFFFF"
        for col_idx, text in enumerate(data):
            cell = row.cells[col_idx]
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=70, bottom=70, left=80, right=80)
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            r = p.add_run(text)
            r.font.name = 'Calibri'
            r.font.size = Pt(9.0)
            r.font.color.rgb = RGBColor(30, 30, 30)
            if col_idx == 0:
                r.font.bold = True
            if col_idx == 4:
                r.font.bold = True
                r.font.color.rgb = RGBColor(18, 44, 79)

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    add_callout(doc,
        "Critical Physical Insights from Benchmark Matrix: \n"
        "1. TCN Dominates Glitches (Dataset C: 68.20% vs 32.22%): Because TCN uses localized convolutional kernels (K=7), a high-energy glitch only corrupts "
        "a localized receptive field. In contrast, global self-attention computes dot-products across the entire second, allowing glitch energy to pollute "
        "attention weights across unrelated time patches.\n"
        "2. Transformer Dominates Weak & Out-of-Distribution Signals (Datasets D & E): When signals are faint (SNR 4-7) or extended (BNS), self-attention "
        "tracks the cumulative phase drift across all 256 patches, achieving 45.90% on whispers and 52.21% on BNS mergers—decisively outperforming TCN by ~10%.\n"
        "3. DAE Wins In-Distribution (Dataset A: 83.35%): U-Net skip connections preserve high-frequency ringdown details with minimal distortion.",
        bold_title="Detailed Comparative Synthesis: ")

    style_heading(doc, "6.3 Time-Frequency Q-Transform Spectrogram Analysis", level=2)
    add_p(doc,
        "To confirm that reconstructed waveforms possess the correct physical time-frequency chirp track (frequency rising quadratically over time), "
        "we computed constant-Q time-frequency spectrograms across noisy inputs, model reconstructions, and clean targets.")

    add_figure(doc,
        os.path.join(fig_dir, "spectrogram_comparison_dataset_A.png"),
        "Figure 6.3: Time-frequency spectrogram comparison on Dataset A showing clean recovery of the parabolic chirp track from noisy strain.")

    add_figure(doc,
        os.path.join(fig_dir, "spectrogram_comparison_glitches_C.png"),
        "Figure 6.4: Time-frequency spectrogram under Dataset C glitch injection showing effective suppression of non-astrophysical burst power.")

    style_heading(doc, "6.4 Zero-Shot Validation on Real Event: GW150914", level=2)
    add_p(doc,
        "A critical test of real-world applicability is zero-shot inference on unmodeled, real gravitational wave events. We fed the actual 1-second calibrated strain "
        "from the historical discovery event GW150914 (recorded on September 14, 2015, at LIGO Hanford H1, GPS time 1126259462.4) directly into our trained model without fine-tuning.")

    add_figure(doc,
        os.path.join(fig_dir, "gw150914_reconstruction.png"),
        "Figure 6.5: Zero-shot reconstruction of the real historical discovery event GW150914 recorded at LIGO Hanford H1.")

    add_p(doc,
        "The reconstructed waveform matches the published numerical relativity best-fit template released by the LIGO Scientific Collaboration, cleanly capturing "
        "the characteristic 0.2-second merger chirp from ~35 Hz up to ~250 Hz and subsequent exponential ringdown.")

    # =========================================================================
    # CHAPTER 7: INTERACTIVE DEMONSTRATION SUITE
    # =========================================================================
    style_heading(doc, "Chapter 7: Interactive Demonstration Suite (Streamlit Dashboard)", level=1)
    add_p(doc,
        "To facilitate viva demonstration, interactive parameter tuning, and real-time audio verification, we developed a production-ready Streamlit web application (app.py). "
        "The suite features six interactive diagnostic modules:")
    add_p(doc, "Enables real-time synthetic injection of binary black holes across adjustable SNR, chirp mass, and phase shifts, displaying live waveform overlays.", bold_prefix="1. Tab 1 - Interactive Reconstructor: ")
    add_p(doc, "Renders high-resolution Q-transform time-frequency spectrograms displaying the emerging chirp track.", bold_prefix="2. Tab 2 - Spectrogram Analysis: ")
    add_p(doc, "Synthesizes audible WAV audio files of raw strain, model-denoised signal, and pure GW audio, enabling users to hear the iconic astrophysical chirp.", bold_prefix="3. Tab 3 - Audio Synthesis & Playback: ")
    add_p(doc, "Tests real-time zero-shot inference on historical LIGO events including GW150914 and GW170817.", bold_prefix="4. Tab 4 - Real Event Verification: ")
    add_p(doc, "Presents mathematical formulations, LaTeX dilation formulas, and architectural layer breakdowns.", bold_prefix="5. Tab 5 - Theoretical Architecture: ")
    add_p(doc, "Displays interactive tables and plots from the 16-trial hyperparameter sweep, logging best validation metrics.", bold_prefix="6. Tab 6 - Systematic Tuning Sweep: ")

    # =========================================================================
    # CHAPTER 8: CONCLUSION & FUTURE WORK
    # =========================================================================
    style_heading(doc, "Chapter 8: Conclusion, Architectural Recommendations and Future Work", level=1)
    
    style_heading(doc, "8.1 Synthesis of Experimental Findings", level=2)
    add_p(doc,
        "This project successfully developed, systematically tuned, and empirically benchmarked three advanced deep learning architectures for gravitational wave reconstruction. "
        "Our experiments reveal that no single neural architecture universally dominates all astrophysical conditions. Rather, each model occupies a distinct operational niche:")
    
    add_p(doc, "Delivers 2.80 ms inference latency (~3.8x faster than Transformer) and unparalleled resilience to localized non-Gaussian glitches (68.20% overlap). Ideal for low-latency early warning trigger pipelines.", bold_prefix="• 1D TCN Baseline: ")
    add_p(doc, "Excels at long-range phase integration, achieving the highest recovery rates on sub-threshold whispers (45.90%) and unseen binary neutron stars (52.21%). Ideal for offline parameter estimation and deep searches.", bold_prefix="• 1D Temporal Transformer: ")
    add_p(doc, "Achieves the highest in-distribution fidelity on standard binary black hole chirps (83.35% overlap) and robustly handles non-stationary noise drift without template dependence. Ideal for unmodeled burst astronomy.", bold_prefix="• Source-Agnostic DAE: ")

    style_heading(doc, "8.2 Future Research Directions", level=2)
    add_p(doc, "Coherently combining strain data from Hanford, Livingston, and Virgo using cross-attention mechanisms to reconstruct sky localization coordinates simultaneously.", bold_prefix="1. Multi-Detector Coherent Attention: ")
    add_p(doc, "Quantizing the 1D TCN model to INT8/FP16 and synthesizing hardware bitstreams for Xilinx/Intel FPGAs to enable sub-millisecond on-detector alert generation.", bold_prefix="2. FPGA Hardware Acceleration: ")
    add_p(doc, "Coupling the reconstructed waveform output into a downstream Bayesian neural network (Normalizing Flows) to perform instantaneous posterior sampling for binary component masses and spins.", bold_prefix="3. Direct Parameter Estimation: ")

    # =========================================================================
    # REFERENCES
    # =========================================================================
    style_heading(doc, "References and Scientific Citations", level=1)
    refs = [
        "Abbott, B. P., et al. (LIGO Scientific Collaboration and Virgo Collaboration). (2016). 'Observation of Gravitational Waves from a Binary Black Hole Merger.' Physical Review Letters, 116(6), 061102.",
        "George, D., & Huerta, E. A. (2018). 'Deep learning for real-time gravitational wave detection and parameter estimation with LIGO data.' Physical Review D, 97(4), 044039.",
        "Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). 'Attention Is All You Need.' Advances in Neural Information Processing Systems (NeurIPS 2017), 30, 5998–6008.",
        "Bai, S., Kolter, J. Z., & Koltun, V. (2018). 'An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling.' arXiv preprint arXiv:1803.01271.",
        "Gabbard, H., Williams, M., Hayes, F., & Messenger, C. (2018). 'Matching matched filtering with deep networks for gravitational-wave astronomy.' Physical Review Letters, 120(14), 141103.",
        "Usman, S. A., et al. (2016). 'The PyCBC search for gravitational waves from compact binary coalescence.' Classical and Quantum Gravity, 33(21), 215004.",
        "Macleod, D., et al. (2021). 'GWpy: A Python package for gravitational-wave astrophysics.' SoftwareX, 13, 100657."
    ]
    for idx, ref in enumerate(refs):
        add_p(doc, ref, bold_prefix=f"[{idx+1}] ", space_after=4)

    # Save docx
    out_docx_path = r'C:\Users\HP\Downloads\Gravitational_Wave_Reconstruction_Project_Report.docx'
    doc.save(out_docx_path)
    print(f"Report saved successfully to {out_docx_path}!")

if __name__ == '__main__':
    build_report()
