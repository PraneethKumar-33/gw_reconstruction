"""
Robust PPTX generator for GW Reconstruction project.
Uses only manual shapes (no placeholder lookups) to avoid KeyError issues.
"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pathlib import Path
import os

# â”€â”€ Paths â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
FIGURES = Path(r"C:\Users\HP\Downloads\gw_reconstruction\results\figures")
OUTPUT  = Path(r"C:\Users\HP\Downloads\gw_reconstruction\results\presentation\GW_Reconstruction_Presentation.pptx")

# â”€â”€ Theme colours â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
BG       = RGBColor(12, 12, 24)       # deep space navy
WHITE    = RGBColor(255, 255, 255)
LIGHT    = RGBColor(210, 210, 220)
ACCENT1  = RGBColor(0, 188, 212)      # teal / cyan
ACCENT2  = RGBColor(255, 152, 0)      # amber / orange
ACCENT3  = RGBColor(76, 175, 80)      # green
DIM      = RGBColor(140, 140, 160)
TABLE_H  = RGBColor(30, 30, 55)       # table header bg
TABLE_R1 = RGBColor(22, 22, 40)       # table row bg 1
TABLE_R2 = RGBColor(28, 28, 48)       # table row bg 2

SLIDE_W = Inches(13.333)   # widescreen 16:9
SLIDE_H = Inches(7.5)

# â”€â”€ Helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def set_bg(slide, color=BG):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color

def add_textbox(slide, left, top, width, height):
    return slide.shapes.add_textbox(left, top, width, height)

def set_para(tf, text, size=18, color=LIGHT, bold=False, alignment=PP_ALIGN.LEFT):
    """Append a paragraph to a text-frame."""
    if len(tf.paragraphs) == 1 and tf.paragraphs[0].text == "":
        p = tf.paragraphs[0]
    else:
        p = tf.add_paragraph()
    p.text = text
    p.font.size = Pt(size)
    p.font.color.rgb = color
    p.font.bold = bold
    p.alignment = alignment
    p.space_after = Pt(4)
    return p

def add_accent_line(slide, left, top, width, color=ACCENT1):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, left, top, width, Pt(3)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    return shape

def add_image_safe(slide, path, left, top, width=None, height=None):
    """Add image only if file exists."""
    p = Path(path)
    if p.exists():
        kwargs = {}
        if width:  kwargs["width"]  = width
        if height: kwargs["height"] = height
        slide.shapes.add_picture(str(p), left, top, **kwargs)
        return True
    else:
        print(f"  âš   Image not found: {p}")
        return False

# â”€â”€ Build presentation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
prs = Presentation()
prs.slide_width  = SLIDE_W
prs.slide_height = SLIDE_H

# =====================================================================
# SLIDE 1 â€” TITLE
# =====================================================================
sl = prs.slides.add_slide(prs.slide_layouts[6])  # blank
set_bg(sl)

# Decorative accent bar at top
add_accent_line(sl, Inches(0), Inches(0), SLIDE_W, ACCENT1)

# Title
tb = add_textbox(sl, Inches(0.8), Inches(1.0), Inches(11.7), Inches(1.6))
set_para(tb.text_frame, "Template-Free Gravitational Wave", size=40, color=WHITE, bold=True, alignment=PP_ALIGN.CENTER)
set_para(tb.text_frame, "Signal Reconstruction Using Deep Learning", size=40, color=WHITE, bold=True, alignment=PP_ALIGN.CENTER)

# Subtitle
tb2 = add_textbox(sl, Inches(1.5), Inches(2.8), Inches(10.3), Inches(0.8))
set_para(tb2.text_frame, "A Comparative Study of 1D TCN, 1D Transformer, and Source-Agnostic Autoencoders on LIGO Strain",
         size=18, color=ACCENT1, alignment=PP_ALIGN.CENTER)

# Team
tb3 = add_textbox(sl, Inches(1.5), Inches(4.0), Inches(10.3), Inches(2.0))
set_para(tb3.text_frame, "Team Members", size=20, color=ACCENT2, bold=True, alignment=PP_ALIGN.CENTER)
set_para(tb3.text_frame, "Hemavardhan â€” Data Ingestion, Signal Preprocessing, Literature Review", size=16, color=LIGHT, alignment=PP_ALIGN.CENTER)
set_para(tb3.text_frame, "Pamu Praneeth Kumar â€” Architecture Engineering (TCN, Transformer, DAE), Benchmark Design", size=16, color=LIGHT, alignment=PP_ALIGN.CENTER)
set_para(tb3.text_frame, "Nikhith â€” Training Execution, Hyperparameter Optimization, Evaluation & Metrics", size=16, color=LIGHT, alignment=PP_ALIGN.CENTER)

# Institution
tb4 = add_textbox(sl, Inches(1.5), Inches(6.2), Inches(10.3), Inches(0.8))
set_para(tb4.text_frame, "Department of CSE (AI) â€¢ Amrita School of Computing â€¢ Amrita Vishwa Vidyapeetham", size=14, color=DIM, alignment=PP_ALIGN.CENTER)
set_para(tb4.text_frame, "Domain: Deep Learning for Multimessenger Time-Domain Astrophysics", size=13, color=DIM, alignment=PP_ALIGN.CENTER)

print("[OK] Slide 1: Title")

# =====================================================================
# SLIDE 2 â€” PROBLEM STATEMENT & MOTIVATION
# =====================================================================
sl = prs.slides.add_slide(prs.slide_layouts[6])
set_bg(sl)
add_accent_line(sl, Inches(0.6), Inches(0.95), Inches(5), ACCENT1)

tb = add_textbox(sl, Inches(0.6), Inches(0.3), Inches(12), Inches(0.7))
set_para(tb.text_frame, "Problem Statement & Motivation", size=36, color=WHITE, bold=True)

tb = add_textbox(sl, Inches(0.6), Inches(1.3), Inches(11.8), Inches(5.8))
tf = tb.text_frame
tf.word_wrap = True

set_para(tf, "The Problem", size=22, color=ACCENT1, bold=True)
set_para(tf, "â€¢ Gravitational waves (GWs) produce infinitesimal strain perturbations (h ~ 10â»Â²Â¹) buried within", size=16, color=LIGHT)
set_para(tf, "  non-stationary, non-Gaussian detector noise at LIGO.", size=16, color=LIGHT)
set_para(tf, "", size=10, color=LIGHT)

set_para(tf, "Why It Matters", size=22, color=ACCENT1, bold=True)
set_para(tf, "â€¢ Classical detection relies on Matched Filtering â€” correlating noisy strain against millions", size=16, color=LIGHT)
set_para(tf, "  of pre-computed theoretical waveform templates.", size=16, color=LIGHT)
set_para(tf, "â€¢ The Bottleneck: Requires prohibitive computational resources, scales poorly, and is completely", size=16, color=LIGHT)
set_para(tf, "  blind to unmodeled or unexpected astrophysical phenomena.", size=16, color=LIGHT)
set_para(tf, "", size=10, color=LIGHT)

set_para(tf, "Research Gap", size=22, color=ACCENT2, bold=True)
set_para(tf, "â€¢ Existing ML approaches focus on binary detection or 2D spectrogram conversion,", size=16, color=LIGHT)
set_para(tf, "  discarding critical time-domain phase coherence and failing to evaluate robustness", size=16, color=LIGHT)
set_para(tf, "  against realistic detector glitches and non-stationary amplitude drifts.", size=16, color=LIGHT)

print("[OK] Slide 2: Problem Statement")

# =====================================================================
# SLIDE 3 â€” OBJECTIVES
# =====================================================================
sl = prs.slides.add_slide(prs.slide_layouts[6])
set_bg(sl)
add_accent_line(sl, Inches(0.6), Inches(0.95), Inches(3.5), ACCENT1)

tb = add_textbox(sl, Inches(0.6), Inches(0.3), Inches(12), Inches(0.7))
set_para(tb.text_frame, "Objectives", size=36, color=WHITE, bold=True)

tb = add_textbox(sl, Inches(0.6), Inches(1.3), Inches(11.8), Inches(5.8))
tf = tb.text_frame
tf.word_wrap = True

set_para(tf, "Primary Objective", size=22, color=ACCENT1, bold=True)
set_para(tf, "Design, implement, and benchmark an end-to-end, template-free deep learning pipeline", size=16, color=LIGHT)
set_para(tf, "capable of dynamically reconstructing 1D time-series GW waveforms from noisy detector strain.", size=16, color=LIGHT)
set_para(tf, "", size=10, color=LIGHT)

set_para(tf, "Specific Goals", size=22, color=ACCENT1, bold=True)
set_para(tf, "1. Multi-Architecture Implementation â€” Custom 1D Dilated Causal TCN, 1D Temporal", size=16, color=LIGHT)
set_para(tf, "   Transformer, and Source-Agnostic Denoising Autoencoder (DAE) in PyTorch.", size=16, color=LIGHT)
set_para(tf, "2. 5-Condition Controlled Benchmarking â€” 5 specialized stress-test datasets", size=16, color=LIGHT)
set_para(tf, "   (Datasets Aâ€“E, 500 samples each) isolating different physical challenges.", size=16, color=LIGHT)
set_para(tf, "3. 16-Trial Hyperparameter Sweep â€” Kernel sizes, patch tokenizers, masking ratios, LRs.", size=16, color=LIGHT)
set_para(tf, "4. Real Cosmic Validation â€” Zero-shot inference on actual GW150914 event.", size=16, color=LIGHT)
set_para(tf, "5. Interactive Streamlit Dashboard â€” Live reconstruction, spectrograms, chirp synthesis.", size=16, color=LIGHT)

print("âœ“ Slide 3: Objectives")

# =====================================================================
# SLIDE 4 â€” LITERATURE REVIEW
# =====================================================================
sl = prs.slides.add_slide(prs.slide_layouts[6])
set_bg(sl)
add_accent_line(sl, Inches(0.6), Inches(0.95), Inches(6.5), ACCENT1)

tb = add_textbox(sl, Inches(0.6), Inches(0.3), Inches(12), Inches(0.7))
set_para(tb.text_frame, "Literature Review & Research Gap", size=36, color=WHITE, bold=True)

# Build a manual table using shapes
rows_data = [
    ["Author & Year", "Methodology", "Limitations"],
    ["Abbott et al. (2016)", "Classical Matched Filtering\n(PyCBC / LALSuite)", "Computationally prohibitive;\nfails on unmodeled bursts"],
    ["George & Huerta (2018)", "2D CNN on Q-Transform\nSpectrograms", "Loses sub-sample phase\naccuracy; 2D FFT overhead"],
    ["Gabbard et al. (2018)", "1D Convolutional Neural\nNetworks", "Binary classification only;\nno waveform reconstruction"],
    ["Shen et al. (2019)", "1D Denoising Autoencoders", "Tested only on idealized\nGaussian noise"],
]

tbl_left = Inches(0.6)
tbl_top = Inches(1.2)
col_widths = [Inches(2.8), Inches(3.8), Inches(4.8)]
row_h = Inches(0.85)
header_h = Inches(0.55)

table_shape = sl.shapes.add_table(len(rows_data), 3, tbl_left, tbl_top,
                                   sum(w for w in col_widths), header_h + row_h * (len(rows_data)-1))
table = table_shape.table

# Set column widths
for i, w in enumerate(col_widths):
    table.columns[i].width = w

for r_idx, row in enumerate(rows_data):
    for c_idx, cell_text in enumerate(row):
        cell = table.cell(r_idx, c_idx)
        cell.text = cell_text
        for para in cell.text_frame.paragraphs:
            para.font.size = Pt(13)
            if r_idx == 0:
                para.font.bold = True
                para.font.color.rgb = WHITE
            else:
                para.font.color.rgb = LIGHT
            para.alignment = PP_ALIGN.LEFT
        # Cell fill
        fill = cell.fill
        fill.solid()
        if r_idx == 0:
            fill.fore_color.rgb = TABLE_H
        elif r_idx % 2 == 0:
            fill.fore_color.rgb = TABLE_R2
        else:
            fill.fore_color.rgb = TABLE_R1

# Research gap note
tb = add_textbox(sl, Inches(0.6), Inches(5.8), Inches(11.8), Inches(1.2))
tf = tb.text_frame
tf.word_wrap = True
set_para(tf, "Research Gap Addressed:", size=18, color=ACCENT2, bold=True)
set_para(tf, "Bridging template-free extraction and real observatory conditions through a 3-way architectural", size=15, color=LIGHT)
set_para(tf, "comparison evaluated across non-stationary drift, glitches, and sub-threshold SNR limits.", size=15, color=LIGHT)

print("âœ“ Slide 4: Literature Review")

# =====================================================================
# SLIDE 5 â€” NOVELTY & CONTRIBUTION
# =====================================================================
sl = prs.slides.add_slide(prs.slide_layouts[6])
set_bg(sl)
add_accent_line(sl, Inches(0.6), Inches(0.95), Inches(5.5), ACCENT1)

tb = add_textbox(sl, Inches(0.6), Inches(0.3), Inches(12), Inches(0.7))
set_para(tb.text_frame, "Novelty & Contribution", size=36, color=WHITE, bold=True)

tb = add_textbox(sl, Inches(0.6), Inches(1.3), Inches(11.8), Inches(5.8))
tf = tb.text_frame
tf.word_wrap = True

set_para(tf, "1. Paradigm-Level Architectural Comparison", size=18, color=ACCENT1, bold=True)
set_para(tf, "   First systematic comparison: Waveform-Supervised (TCN + Transformer)", size=15, color=LIGHT)
set_para(tf, "   vs. Source-Agnostic Self-Supervised (DAE via Masked Signal Reconstruction).", size=15, color=LIGHT)
set_para(tf, "", size=8, color=LIGHT)

set_para(tf, "2. Controlled 5-Condition Stress-Testing Framework", size=18, color=ACCENT1, bold=True)
set_para(tf, "   5 standardized benchmark datasets isolating physical challenge vectors:", size=15, color=LIGHT)
set_para(tf, "   Baseline (A), Non-Stationary Drift (B), Transient Glitches (C),", size=15, color=LIGHT)
set_para(tf, "   Sub-Threshold Whispers (D), Unseen BNS (E).", size=15, color=LIGHT)
set_para(tf, "", size=8, color=LIGHT)

set_para(tf, "3. Discovery of \"The Glitch Dilemma\"", size=18, color=ACCENT2, bold=True)
set_para(tf, "   TCNs win decisively on glitch-contaminated data (66.72% vs 49.82% overlap)", size=15, color=LIGHT)
set_para(tf, "   due to localized causal receptive fields preventing glitch energy leakage.", size=15, color=LIGHT)
set_para(tf, "", size=8, color=LIGHT)

set_para(tf, "4. Zero-Shot Real Event Validation + Interactive Dashboard", size=18, color=ACCENT1, bold=True)
set_para(tf, "   Zero pre-trained weights; custom PyTorch architectures from scratch.", size=15, color=LIGHT)
set_para(tf, "   Successful zero-shot reconstruction of GW150914.", size=15, color=LIGHT)
set_para(tf, "   Real-time Streamlit dashboard with auditory chirp synthesis.", size=15, color=LIGHT)

print("âœ“ Slide 5: Novelty")

# =====================================================================
# SLIDE 6 â€” DATASET DETAILS
# =====================================================================
sl = prs.slides.add_slide(prs.slide_layouts[6])
set_bg(sl)
add_accent_line(sl, Inches(0.6), Inches(0.95), Inches(4.5), ACCENT1)

tb = add_textbox(sl, Inches(0.6), Inches(0.3), Inches(12), Inches(0.7))
set_para(tb.text_frame, "Dataset Details", size=36, color=WHITE, bold=True)

# Left column
tb = add_textbox(sl, Inches(0.6), Inches(1.3), Inches(5.8), Inches(5.8))
tf = tb.text_frame
tf.word_wrap = True

set_para(tf, "Source & Repository", size=20, color=ACCENT1, bold=True)
set_para(tf, "â€¢ GWOSC (O1/O2/O3 observing runs)", size=15, color=LIGHT)
set_para(tf, "â€¢ PyCBC waveform simulations (IMRPhenomD)", size=15, color=LIGHT)
set_para(tf, "", size=8, color=LIGHT)

set_para(tf, "Data Scale", size=20, color=ACCENT1, bold=True)
set_para(tf, "â€¢ Dataset A: 5,000 pairs (4K train / 500 val / 500 test)", size=15, color=LIGHT)
set_para(tf, "â€¢ Datasets Bâ€“E: 500 test pairs each", size=15, color=LIGHT)
set_para(tf, "â€¢ Total: 7,000 samples", size=15, color=LIGHT)
set_para(tf, "", size=8, color=LIGHT)

set_para(tf, "Sampling & Dimensions", size=20, color=ACCENT1, bold=True)
set_para(tf, "â€¢ 4096 Hz sampling rate", size=15, color=LIGHT)
set_para(tf, "â€¢ 1.0 s window â†’ 4096 samples per segment", size=15, color=LIGHT)
set_para(tf, "â€¢ Input X: [Batch, 1, 4096] â€” Noisy Strain", size=15, color=LIGHT)
set_para(tf, "â€¢ Target Y: [Batch, 1, 4096] â€” Clean Chirp", size=15, color=LIGHT)

# Right column
tb = add_textbox(sl, Inches(6.8), Inches(1.3), Inches(5.8), Inches(5.8))
tf = tb.text_frame
tf.word_wrap = True

set_para(tf, "Parameter Ranges", size=20, color=ACCENT2, bold=True)
set_para(tf, "â€¢ BBH masses: mâ‚, mâ‚‚ âˆˆ [10, 80] Mâ˜‰", size=15, color=LIGHT)
set_para(tf, "â€¢ BNS masses (Dataset E): [1.1, 1.4] Mâ˜‰", size=15, color=LIGHT)
set_para(tf, "â€¢ Spins: Ï‡â‚z, Ï‡â‚‚z âˆˆ [âˆ’0.9, 0.9]", size=15, color=LIGHT)
set_para(tf, "", size=8, color=LIGHT)

set_para(tf, "SNR Ranges", size=20, color=ACCENT2, bold=True)
set_para(tf, "â€¢ Datasets A, B, C: SNR 5.0 â€“ 25.0", size=15, color=LIGHT)
set_para(tf, "â€¢ Dataset D (Whispers): SNR 4.0 â€“ 7.0", size=15, color=LIGHT)
set_para(tf, "", size=8, color=LIGHT)

set_para(tf, "Benchmark Conditions", size=20, color=ACCENT2, bold=True)
set_para(tf, "â€¢ A: Stationary noise + standard BBH", size=15, color=LIGHT)
set_para(tf, "â€¢ B: Non-stationary amplitude drift", size=15, color=LIGHT)
set_para(tf, "â€¢ C: Transient sine-Gaussian glitches", size=15, color=LIGHT)
set_para(tf, "â€¢ D: Sub-threshold faint signals", size=15, color=LIGHT)
set_para(tf, "â€¢ E: Unseen BNS morphologies", size=15, color=LIGHT)

print("âœ“ Slide 6: Dataset Details")

# =====================================================================
# SLIDE 7 â€” DATA PREPROCESSING
# =====================================================================
sl = prs.slides.add_slide(prs.slide_layouts[6])
set_bg(sl)
add_accent_line(sl, Inches(0.6), Inches(0.95), Inches(5.5), ACCENT1)

tb = add_textbox(sl, Inches(0.6), Inches(0.3), Inches(12), Inches(0.7))
set_para(tb.text_frame, "Data Preprocessing & Leakage Prevention", size=36, color=WHITE, bold=True)

tb = add_textbox(sl, Inches(0.6), Inches(1.3), Inches(11.8), Inches(5.8))
tf = tb.text_frame
tf.word_wrap = True

steps = [
    ("1. Contiguous Slice Extraction",
     "Scanned raw HDF5 files for continuous lock segments,\n   eliminating detector unlock gaps and NaNs."),
    ("2. 4th-Order Butterworth Bandpass Filter",
     "20 Hz â€“ 500 Hz passband removes seismic vibrations\n   and high-frequency thermal harmonics."),
    ("3. Frequency-Domain Strain Whitening",
     "PSD via Welch's method (4 s FFT, 2 s overlap);\n   divide in Fourier domain â†’ zero-mean, unit-variance Gaussian background."),
    ("4. Filter Edge Truncation",
     "Drop 2 s boundary data on both edges to prevent\n   impulse response ringing artifacts."),
    ("5. Strict Data Leakage Prevention",
     "Train / Val / Test from non-overlapping GPS intervals\n   separated by buffer zones â†’ zero temporal contamination."),
]
for title, desc in steps:
    set_para(tf, title, size=18, color=ACCENT1, bold=True)
    for line in desc.split("\n"):
        set_para(tf, "   " + line.strip(), size=15, color=LIGHT)
    set_para(tf, "", size=6, color=LIGHT)

print("âœ“ Slide 7: Preprocessing")

# =====================================================================
# SLIDE 8 â€” SYSTEM ARCHITECTURE
# =====================================================================
sl = prs.slides.add_slide(prs.slide_layouts[6])
set_bg(sl)
add_accent_line(sl, Inches(0.6), Inches(0.95), Inches(6), ACCENT1)

tb = add_textbox(sl, Inches(0.6), Inches(0.3), Inches(12), Inches(0.7))
set_para(tb.text_frame, "Proposed System Architecture", size=36, color=WHITE, bold=True)

# Draw a simplified pipeline using textboxes + shapes
def add_box(slide, left, top, w, h, text, bg_color, font_size=13, font_color=WHITE):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, w, h)
    shape.fill.solid()
    shape.fill.fore_color.rgb = bg_color
    shape.line.color.rgb = ACCENT1
    shape.line.width = Pt(1.5)
    tf = shape.text_frame
    tf.word_wrap = True
    tf.paragraphs[0].alignment = PP_ALIGN.CENTER
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.color.rgb = font_color
    p.font.bold = True
    return shape

def add_arrow(slide, x, y1, y2, color=ACCENT1):
    """Vertical connector line."""
    connector = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y1, Pt(3), y2 - y1)
    connector.fill.solid()
    connector.fill.fore_color.rgb = color
    connector.line.fill.background()
    # arrowhead triangle
    tri = slide.shapes.add_shape(MSO_SHAPE.ISOSCELES_TRIANGLE, x - Pt(6), y2, Pt(15), Pt(12))
    tri.fill.solid()
    tri.fill.fore_color.rgb = color
    tri.line.fill.background()

box_w = Inches(3.5)
box_h = Inches(0.6)
center_x = Inches(4.9)

y = Inches(1.2)
add_box(sl, center_x, y, box_w, box_h, "Raw LIGO HDF5 Strain", RGBColor(40, 40, 70))
add_arrow(sl, center_x + box_w // 2, y + box_h, y + box_h + Inches(0.25))

y += Inches(1.0)
add_box(sl, center_x, y, box_w, box_h, "Bandpass 20â€“500 Hz + Whitening", RGBColor(40, 40, 70))
add_arrow(sl, center_x + box_w // 2, y + box_h, y + box_h + Inches(0.25))

y += Inches(1.0)
add_box(sl, center_x, y, box_w, box_h, "PyCBC Waveform Injection Engine", RGBColor(40, 40, 70))

# Branch left and right
y_branch = y + box_h + Inches(0.4)

# Left: Training
add_box(sl, Inches(1.0), y_branch, Inches(3.0), Inches(0.55),
        "Training Set (Dataset A)\n4,000 pairs", RGBColor(0, 80, 90), font_size=12)

# Right: Benchmark
add_box(sl, Inches(9.0), y_branch, Inches(3.5), Inches(0.55),
        "Benchmark Suites (B, C, D, E)\n2,000 test pairs", RGBColor(80, 60, 0), font_size=12)

# Models box (wide, center)
y_models = y_branch + Inches(0.9)
model_box = add_box(sl, Inches(1.5), y_models, Inches(10.3), Inches(0.9),
                     "Deep Learning Reconstructors\n1D TCN  â€¢  1D Transformer  â€¢  Source-Agnostic DAE",
                     RGBColor(20, 60, 80), font_size=14)

# Evaluation
y_eval = y_models + Inches(1.2)
add_box(sl, Inches(2.5), y_eval, Inches(8.3), Inches(0.7),
        "Evaluation: Overlap Match  â€¢  MSE  â€¢  Latency Benchmark",
        RGBColor(40, 40, 70), font_size=13)

# Dashboard
y_dash = y_eval + Inches(0.9)
add_box(sl, Inches(3.5), y_dash, Inches(6.3), Inches(0.6),
        "Streamlit Dashboard â€¢ GW150914 â€¢ Spectrograms â€¢ Chirp Audio",
        RGBColor(50, 25, 60), font_size=12)

print("âœ“ Slide 8: Architecture")

# =====================================================================
# SLIDE 9 â€” MODEL METHODOLOGY
# =====================================================================
sl = prs.slides.add_slide(prs.slide_layouts[6])
set_bg(sl)
add_accent_line(sl, Inches(0.6), Inches(0.95), Inches(7), ACCENT1)

tb = add_textbox(sl, Inches(0.6), Inches(0.3), Inches(12), Inches(0.7))
set_para(tb.text_frame, "Deep Learning Models & Methodology", size=36, color=WHITE, bold=True)

# Three columns
col_data = [
    {
        "title": "1D TCN Baseline",
        "color": ACCENT1,
        "lines": [
            "6 Residual Temporal Blocks",
            "Dilations: {1,2,4,8,16,32}",
            "Kernel Size K=7",
            "Weight Norm + Dropout 0.1",
            "Receptive Field: 757 samples",
            "Parameters: ~615K",
            "",
            "âœ¦ Preserves temporal causality",
            "âœ¦ 2.80 ms inference latency",
        ],
    },
    {
        "title": "1D Transformer",
        "color": ACCENT2,
        "lines": [
            "Patch Tokenizer P=16",
            "4096 â†’ 256 tokens, d=128",
            "Sinusoidal Positional Encoding",
            "4 MHSA Layers (4 heads)",
            "FFN=512, GELU activation",
            "Parameters: ~941K",
            "",
            "âœ¦ Global phase coherence",
            "âœ¦ Best on faint signals",
        ],
    },
    {
        "title": "Source-Agnostic DAE",
        "color": ACCENT3,
        "lines": [
            "1D Conv U-Net architecture",
            "4096â†’2048â†’1024â†’512â†’256",
            "Residual skip connections",
            "Instance Normalization",
            "10% masked reconstruction",
            "Parameters: ~1.65M",
            "",
            "âœ¦ Template-free training",
            "âœ¦ Highest baseline overlap",
        ],
    },
]

for i, col in enumerate(col_data):
    left = Inches(0.5 + i * 4.2)
    tb = add_textbox(sl, left, Inches(1.2), Inches(3.8), Inches(4.5))
    tf = tb.text_frame
    tf.word_wrap = True
    set_para(tf, col["title"], size=20, color=col["color"], bold=True)
    for line in col["lines"]:
        set_para(tf, line, size=14, color=LIGHT)

# Training specs at bottom
tb = add_textbox(sl, Inches(0.6), Inches(6.0), Inches(12), Inches(1.2))
tf = tb.text_frame
tf.word_wrap = True
set_para(tf, "Training:  Adam (lr=10â»Â³, wd=10â»âµ)  â€¢  Cosine Annealing LR  â€¢  Loss: MSE  â€¢  Hardware: NVIDIA RTX 3050 (CUDA 12.1)",
         size=14, color=DIM, alignment=PP_ALIGN.CENTER)

print("âœ“ Slide 9: Models")

# =====================================================================
# SLIDE 10 â€” RESULTS: MULTI-MODEL PERFORMANCE
# =====================================================================
sl = prs.slides.add_slide(prs.slide_layouts[6])
set_bg(sl)
add_accent_line(sl, Inches(0.6), Inches(0.95), Inches(8), ACCENT1)

tb = add_textbox(sl, Inches(0.6), Inches(0.3), Inches(12), Inches(0.7))
set_para(tb.text_frame, "Experimental Results â€” Multi-Model Performance", size=32, color=WHITE, bold=True)

# Results table
rows = [
    ["Metric",          "TCN",           "Transformer",   "DAE",            "Best"],
    ["Mean Overlap",    "78.09%",        "84.91%",        "85.52% âœ“",       "DAE (+7.43%)"],
    ["Peak Overlap",    "96.98%",        "97.82% âœ“",      "97.45%",         "Transformer"],
    ["Mean MSE",        "0.014681",      "0.010660",      "0.009679 âœ“",     "DAE (âˆ’34.1%)"],
    ["Inference Latency","2.80 ms âœ“",   "10.85 ms",      "6.02 ms",        "TCN (3.8Ã— faster)"],
    ["Parameters",      "615K âœ“",        "941K",          "1.65M",          "TCN (lightest)"],
]

tbl_shape = sl.shapes.add_table(len(rows), 5, Inches(0.5), Inches(1.15),
                                 Inches(12.3), Inches(2.8))
tbl = tbl_shape.table
col_ws = [Inches(2.2), Inches(2.2), Inches(2.2), Inches(2.2), Inches(3.5)]
for i, w in enumerate(col_ws):
    tbl.columns[i].width = w

for r, row in enumerate(rows):
    for c, val in enumerate(row):
        cell = tbl.cell(r, c)
        cell.text = val
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(13)
            p.font.color.rgb = WHITE if r == 0 else LIGHT
            p.font.bold = (r == 0)
            p.alignment = PP_ALIGN.CENTER
        cell.fill.solid()
        if r == 0:
            cell.fill.fore_color.rgb = TABLE_H
        elif r % 2 == 0:
            cell.fill.fore_color.rgb = TABLE_R2
        else:
            cell.fill.fore_color.rgb = TABLE_R1

# Embed figures
add_image_safe(sl, FIGURES / "head_to_head_waveform_comparison.png",
               Inches(0.3), Inches(4.1), width=Inches(6.3))
add_image_safe(sl, FIGURES / "multi_model_benchmark_barchart.png",
               Inches(6.8), Inches(4.1), width=Inches(6.0))

print("âœ“ Slide 10: Multi-Model Results")

# =====================================================================
# SLIDE 11 â€” ROBUSTNESS BENCHMARK
# =====================================================================
sl = prs.slides.add_slide(prs.slide_layouts[6])
set_bg(sl)
add_accent_line(sl, Inches(0.6), Inches(0.95), Inches(8), ACCENT1)

tb = add_textbox(sl, Inches(0.6), Inches(0.3), Inches(12), Inches(0.7))
set_para(tb.text_frame, "Robustness Benchmark Matrix (5 Conditions Ã— 3 Models)", size=32, color=WHITE, bold=True)

rows = [
    ["Condition",             "Challenge",                    "TCN",    "Transformer", "DAE",    "Winner"],
    ["Dataset A",             "Stationary Noise + BBH",       "78.09%", "84.91%",      "85.52%", "DAE"],
    ["Dataset B",             "Non-Stationary Drift",         "66.93%", "55.13%",      "68.97%", "DAE"],
    ["Dataset C",             "Transient Glitches",           "66.72%", "49.82%",      "58.66%", "TCN âœ“"],
    ["Dataset D",             "Sub-Threshold (SNR 4â€“7)",      "35.48%", "51.60%",      "44.22%", "Transformer âœ“"],
    ["Dataset E",             "Unseen BNS Morphologies",      "42.38%", "52.76%",      "48.52%", "Transformer âœ“"],
]

tbl_shape = sl.shapes.add_table(len(rows), 6, Inches(0.3), Inches(1.15),
                                 Inches(12.7), Inches(2.8))
tbl = tbl_shape.table
col_ws2 = [Inches(1.6), Inches(3.0), Inches(1.6), Inches(1.8), Inches(1.6), Inches(2.2)]
for i, w in enumerate(col_ws2):
    tbl.columns[i].width = w

for r, row in enumerate(rows):
    for c, val in enumerate(row):
        cell = tbl.cell(r, c)
        cell.text = val
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(12)
            p.font.color.rgb = WHITE if r == 0 else LIGHT
            p.font.bold = (r == 0)
            p.alignment = PP_ALIGN.CENTER
        cell.fill.solid()
        if r == 0:
            cell.fill.fore_color.rgb = TABLE_H
        elif r % 2 == 0:
            cell.fill.fore_color.rgb = TABLE_R2
        else:
            cell.fill.fore_color.rgb = TABLE_R1

# Key findings
tb = add_textbox(sl, Inches(0.5), Inches(4.2), Inches(12.3), Inches(3.0))
tf = tb.text_frame
tf.word_wrap = True

set_para(tf, "Critical Physical Findings", size=20, color=ACCENT2, bold=True)
set_para(tf, "", size=4, color=LIGHT)
set_para(tf, "Glitch Resilience (TCN):  66.72% overlap vs Transformer's 49.82%.", size=15, color=LIGHT)
set_para(tf, "   â†’ Localized causal convolutions restrict glitch energy to bounded receptive-field windows.", size=14, color=DIM)
set_para(tf, "", size=4, color=LIGHT)
set_para(tf, "Sub-Threshold Coherence (Transformer):  51.60% overlap vs TCN's 35.48%.", size=15, color=LIGHT)
set_para(tf, "   â†’ Global multi-head self-attention pools distributed phase coherence across all 256 tokens.", size=14, color=DIM)
set_para(tf, "", size=4, color=LIGHT)
set_para(tf, "Low-Latency Suitability:  TCN at 2.80 ms/sample easily meets real-time trigger alert requirements.", size=15, color=LIGHT)

print("âœ“ Slide 11: Robustness Benchmark")

# =====================================================================
# SLIDE 12 â€” HYPERPARAMETER TUNING + GW150914
# =====================================================================
sl = prs.slides.add_slide(prs.slide_layouts[6])
set_bg(sl)
add_accent_line(sl, Inches(0.6), Inches(0.95), Inches(9), ACCENT1)

tb = add_textbox(sl, Inches(0.6), Inches(0.3), Inches(12), Inches(0.7))
set_para(tb.text_frame, "Hyperparameter Tuning & GW150914 Validation", size=32, color=WHITE, bold=True)

# Left: tuning text
tb = add_textbox(sl, Inches(0.5), Inches(1.15), Inches(6.0), Inches(2.8))
tf = tb.text_frame
tf.word_wrap = True

set_para(tf, "16-Trial Systematic Optimization", size=18, color=ACCENT1, bold=True)
set_para(tf, "TCN (6 trials): K=9 â†’ 71.68%; dropout 0.2 â†’ 72.38%", size=14, color=LIGHT)
set_para(tf, "Transformer (5 trials): P=16 optimal (76.50%);", size=14, color=LIGHT)
set_para(tf, "   lr=2Ã—10â»Â³ â†’ 78.76%", size=14, color=LIGHT)
set_para(tf, "DAE (5 trials): 5â€“10% mask â†’ 86.10% overlap;", size=14, color=LIGHT)
set_para(tf, "   stable even at 30% masking (84.45%)", size=14, color=LIGHT)

# Right: GW150914 text
tb = add_textbox(sl, Inches(6.8), Inches(1.15), Inches(6.0), Inches(2.8))
tf = tb.text_frame
tf.word_wrap = True

set_para(tf, "Zero-Shot GW150914 Validation", size=18, color=ACCENT2, bold=True)
set_para(tf, "Fed raw LIGO Hanford (H1) recording", size=14, color=LIGHT)
set_para(tf, "directly into models â€” no templates.", size=14, color=LIGHT)
set_para(tf, "", size=6, color=LIGHT)
set_para(tf, "Result: Successfully suppressed seismic", size=14, color=LIGHT)
set_para(tf, "noise and reconstructed the historic", size=14, color=LIGHT)
set_para(tf, "36 Mâ˜‰ + 29 Mâ˜‰ BBH merger peak at t = 0.0 s.", size=14, color=LIGHT)

# Embed figures below
add_image_safe(sl, FIGURES / "hyperparameter_tuning_analysis.png",
               Inches(0.3), Inches(4.0), width=Inches(6.3))
add_image_safe(sl, FIGURES / "gw150914_reconstruction.png",
               Inches(6.8), Inches(4.0), width=Inches(6.0))

print("âœ“ Slide 12: Hyperparameter Tuning + GW150914")

# =====================================================================
# SLIDE 13 â€” LIMITATIONS, FUTURE SCOPE, CONCLUSION
# =====================================================================
sl = prs.slides.add_slide(prs.slide_layouts[6])
set_bg(sl)
add_accent_line(sl, Inches(0.6), Inches(0.95), Inches(8), ACCENT1)

tb = add_textbox(sl, Inches(0.6), Inches(0.3), Inches(12), Inches(0.7))
set_para(tb.text_frame, "Limitations, Future Scope & Conclusion", size=36, color=WHITE, bold=True)

# Left column: Limitations + Future Scope
tb = add_textbox(sl, Inches(0.6), Inches(1.3), Inches(5.8), Inches(5.5))
tf = tb.text_frame
tf.word_wrap = True

set_para(tf, "Current Limitations", size=20, color=ACCENT2, bold=True)
set_para(tf, "â€¢ Single-detector input â€” no spatial coherence", size=15, color=LIGHT)
set_para(tf, "  between H1 and L1 detectors.", size=15, color=LIGHT)
set_para(tf, "â€¢ Extended BNS inspirals (>30 s) exceed the", size=15, color=LIGHT)
set_para(tf, "  1.0-second input window.", size=15, color=LIGHT)
set_para(tf, "", size=10, color=LIGHT)

set_para(tf, "Future Scope", size=20, color=ACCENT1, bold=True)
set_para(tf, "â€¢ Dual-Detector Coherent Ingestion:", size=15, color=LIGHT)
set_para(tf, "  Expand to [Batch, 2, 4096] for spatial", size=15, color=LIGHT)
set_para(tf, "  coincidence and glitch rejection.", size=15, color=LIGHT)
set_para(tf, "â€¢ Physics-Informed Spectral Loss:", size=15, color=LIGHT)
set_para(tf, "  Augment MSE with Fourier magnitude +", size=15, color=LIGHT)
set_para(tf, "  phase penalties.", size=15, color=LIGHT)
set_para(tf, "â€¢ Edge Deployment: Quantized TCN on FPGA /", size=15, color=LIGHT)
set_para(tf, "  Jetson for sub-ms edge alerts at LIGO.", size=15, color=LIGHT)

# Right column: Conclusion
tb = add_textbox(sl, Inches(6.8), Inches(1.3), Inches(5.8), Inches(5.5))
tf = tb.text_frame
tf.word_wrap = True

set_para(tf, "Conclusion", size=20, color=ACCENT3, bold=True)
set_para(tf, "", size=6, color=LIGHT)
set_para(tf, "âœ“ Successfully developed an end-to-end,", size=15, color=LIGHT)
set_para(tf, "  template-free deep learning reconstruction", size=15, color=LIGHT)
set_para(tf, "  framework.", size=15, color=LIGHT)
set_para(tf, "", size=6, color=LIGHT)
set_para(tf, "âœ“ Transformers excel at faint sub-threshold", size=15, color=LIGHT)
set_para(tf, "  signals (51.6% overlap).", size=15, color=LIGHT)
set_para(tf, "", size=6, color=LIGHT)
set_para(tf, "âœ“ TCNs provide unmatched glitch resistance", size=15, color=LIGHT)
set_para(tf, "  (66.7% overlap) and 3.8Ã— faster inference", size=15, color=LIGHT)
set_para(tf, "  (2.8 ms).", size=15, color=LIGHT)
set_para(tf, "", size=6, color=LIGHT)
set_para(tf, "âœ“ Deep learning is a viable, template-free", size=15, color=LIGHT)
set_para(tf, "  alternative for real-time gravitational", size=15, color=LIGHT)
set_para(tf, "  wave astronomy.", size=15, color=LIGHT)

# Bottom accent
add_accent_line(sl, Inches(0.6), Inches(7.1), Inches(12), ACCENT1)

tb = add_textbox(sl, Inches(2), Inches(7.15), Inches(9.3), Inches(0.3))
set_para(tb.text_frame, "Thank You  â€¢  Questions?", size=18, color=WHITE, bold=True, alignment=PP_ALIGN.CENTER)

print("âœ“ Slide 13: Conclusion")

# â”€â”€ Save â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
prs.save(str(OUTPUT))
print(f"\nâœ… Presentation saved successfully to:\n   {OUTPUT}")
print(f"   Total slides: {len(prs.slides)}")
