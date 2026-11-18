import os
import re
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

# Paths
markdown_path = Path(r"C:/Users/HP/.gemini/antigravity/brain/6633e6fa-42e3-46cc-9624-7ba472c4caaa/GW_Final_Presentation_Deck_Content.md")
figures_dir = Path(r"C:/Users/HP/Downloads/gw_reconstruction/results/figures")
output_path = Path(r"C:/Users/HP/Downloads/gw_reconstruction/results/presentation/GW_Reconstruction_Presentation.pptx")

# Read markdown
with markdown_path.open("r", encoding="utf-8") as f:
    md_text = f.read()

# Split into slides based on "### Slide" headings
slides_raw = re.split(r"^### Slide \d+:", md_text, flags=re.MULTILINE)
# The first split part before first slide header is irrelevant
slides_raw = slides_raw[1:]

# Helper to create a slide with dark background
def add_slide(prs, title_text, body_text="", layout_idx=1):
    slide_layout = prs.slide_layouts[layout_idx]  # Title and Content layout
    slide = prs.slides.add_slide(slide_layout)
    # Set dark background
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = RGBColor(20, 20, 30)  # dark navy
    # Title
    title = slide.shapes.title
    title.text = title_text
    title.text_frame.paragraphs[0].font.size = Pt(44)
    title.text_frame.paragraphs[0].font.color.rgb = RGBColor(255, 255, 255)
    # Body/content
    if body_text:
        tf = slide.placeholders[1].text_frame
        tf.clear()
        for line in body_text.splitlines():
            p = tf.add_paragraph()
            p.text = line.strip()
            p.font.size = Pt(24)
            p.font.color.rgb = RGBColor(230, 230, 230)
            p.level = 0
    return slide

prs = Presentation()

# Process each slide
for raw in slides_raw:
    # The first line after split is the title line (up to newline)
    lines = raw.strip().splitlines()
    if not lines:
        continue
    title_line = lines[0].strip()
    # The rest is body
    body = "\n".join(lines[1:]).strip()
    # Clean markdown symbols (bullet stars, tables etc.) – keep simple text
    # Remove markdown tables delimiters
    body = re.sub(r"\|.*\|", "", body)
    # Remove any residual markdown formatting characters
    body = body.replace("*", "")
    body = body.replace("- ", "• ")
    # Add slide
    slide = add_slide(prs, title_line, body)

    # Insert specific images based on slide title keywords
    title_lower = title_line.lower()
    if "experimental results — multi-model performance" in title_lower:
        img1 = figures_dir / "head_to_head_waveform_comparison.png"
        img2 = figures_dir / "multi_model_benchmark_barchart.png"
        if img1.exists():
            slide.shapes.add_picture(str(img1), Inches(0.5), Inches(4.5), width=Inches(4))
        if img2.exists():
            slide.shapes.add_picture(str(img2), Inches(5.5), Inches(4.5), width=Inches(4))
    elif "experimental results — robustness benchmark matrix" in title_lower:
        img = figures_dir / "multi_model_benchmark_barchart.png"
        if img.exists():
            slide.shapes.add_picture(str(img), Inches(1), Inches(2), width=Inches(8))
    elif "hyperparameter tuning" in title_lower:
        img1 = figures_dir / "hyperparameter_tuning_analysis.png"
        img2 = figures_dir / "gw150914_reconstruction.png"
        if img1.exists():
            slide.shapes.add_picture(str(img1), Inches(0.5), Inches(2.5), width=Inches(4.5))
        if img2.exists():
            slide.shapes.add_picture(str(img2), Inches(5), Inches(2.5), width=Inches(4.5))
    elif "proposed system architecture" in title_lower:
        # Placeholder rectangle if no architecture image is present
        # Could add a simple shape later
        pass

# Save presentation
output_path.parent.mkdir(parents=True, exist_ok=True)
prs.save(str(output_path))
print(f"Presentation saved to {output_path}")
