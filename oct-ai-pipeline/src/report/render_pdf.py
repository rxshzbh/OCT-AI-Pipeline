"""
report/render_pdf.py — Stage 7b: lay out the report dict as a one-page PDF.

Layout priority, top to bottom, matches how a doctor actually scans a
document: the headline diagnosis first, then the visual evidence (so they
can eyeball-verify it immediately), then the biomarker checklist last as
the detailed backup. Keep this order if you change the layout — burying
the diagnosis below images defeats the "<60 second read" goal.
"""

from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


def render_pdf(report: dict, output_dir: str = "data/reports") -> str:
    out_path = Path(output_dir) / f"{report['scan_id']}_report.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleBig", parent=styles["Title"], fontSize=18)
    warn_style = ParagraphStyle("Warn", parent=styles["Normal"], textColor=colors.red)

    doc = SimpleDocTemplate(str(out_path), pagesize=A4,
                             topMargin=15 * mm, bottomMargin=15 * mm)
    story = []

    # --- Headline diagnosis ---
    dx = report["diagnosis"]
    story.append(Paragraph(f"Diagnosis: {dx['label']}", title_style))
    story.append(Paragraph(f"Confidence: {dx['confidence']:.0%}", styles["Normal"]))
    if dx["is_atypical"]:
        story.append(Paragraph(
            f"⚠ Atypical scan — findings do not clearly match known disease classes "
            f"(anomaly score {dx['anomaly_score']:.2f}). Recommend specialist review.",
            warn_style,
        ))
    if dx["differentials"]:
        diffs = ", ".join(f"{d['label']} ({d['confidence']:.0%})" for d in dx["differentials"])
        story.append(Paragraph(f"Differentials considered: {diffs}", styles["Normal"]))
    story.append(Spacer(1, 8 * mm))

    # --- Visual evidence: segmentation + Grad-CAM side by side ---
    ev = report["evidence"]
    img_row = []
    for path, caption in [
        (ev["segmentation_overlay"], "Segmentation — regions identified"),
        (ev["gradcam_overlay"], "Model attention (Grad-CAM)"),
    ]:
        if Path(path).exists():
            img_row.append(RLImage(path, width=80 * mm, height=80 * mm))
        else:
            img_row.append(Paragraph(f"[missing: {path}]", styles["Normal"]))
    story.append(Table([img_row], colWidths=[85 * mm, 85 * mm]))
    story.append(Paragraph(
        "Segmentation — regions identified" + " " * 20 + "Model attention (Grad-CAM)",
        styles["Normal"],
    ))

    if ev["low_confidence_explanation_warning"]:
        story.append(Paragraph(
            f"⚠ Segmentation and model-attention maps show low spatial agreement "
            f"({ev['xai_segmentation_agreement']:.0%}) — treat this explanation with caution.",
            warn_style,
        ))
    story.append(Spacer(1, 8 * mm))

    # --- Measurements ---
    if ev["measurements"]:
        rows = [["Measurement", "Value"]] + [
            [k.replace("_", " "), f"{v:.1f}" if v is not None else "n/a"]
            for k, v in ev["measurements"].items()
        ]
        t = Table(rows, colWidths=[90 * mm, 40 * mm])
        t.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ]))
        story.append(t)
        story.append(Spacer(1, 8 * mm))

    # --- Biomarker checklist (full, sorted by confidence) ---
    story.append(Paragraph("Biomarker checklist", styles["Heading2"]))
    bio_rows = [["Biomarker", "Confidence", "Flagged"]] + [
        [b["name"].replace("_", " "), f"{b['confidence']:.0%}", "✓" if b["flagged"] else ""]
        for b in report["biomarkers"]["all"]
    ]
    bio_table = Table(bio_rows, colWidths=[90 * mm, 30 * mm, 20 * mm])
    bio_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
    ]))
    story.append(bio_table)
    story.append(Spacer(1, 6 * mm))

    story.append(Paragraph(report["disclaimer"], ParagraphStyle(
        "Disclaimer", parent=styles["Normal"], fontSize=8, textColor=colors.grey,
    )))

    doc.build(story)
    return str(out_path)
