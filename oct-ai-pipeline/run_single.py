"""
run_single.py — run the full pipeline on ONE image and produce a report.

Usage:
    python run_single.py --image path/to/scan.dcm

This is deliberately simple: no server, no queue, just "take one image,
run every stage, write a JSON + PDF report." Get this working end-to-end
with your trained models before building anything more complex on top of
it — a correct single-image run is the foundation everything else
(batch processing, an API, a folder-watcher for hospital use) builds on
later without changing any of the stage logic itself.
"""

import argparse
import json
from pathlib import Path

from src.ingest.dicom_loader import load_scan
from src.ingest.quality_check import check_quality
from src.preprocess.transforms import letterbox_resize, normalize_intensity
from src.report.build_report_json import build_report, Diagnosis, Evidence
from src.report.render_pdf import render_pdf

# Model wrappers — implement/import these once Stage 2+ backbones are trained.
# Left as explicit imports here (rather than hidden inside OCTPipeline) so
# you can run this script stage-by-stage while models are still being
# trained, commenting out whichever stage isn't ready yet.
from src.models.retfound_wrapper import RETFoundEncoder
from src.models.internimage_wrapper import InternImageEncoder
from src.models.medvit_wrapper import MedViTEncoder
from src.models.nnunet_wrapper import NNUNetSegmenter
from src.models.biomarker_head import BiomarkerHead, predict_biomarkers
from src.models.anomaly_head import AnomalyScorer
from src.models.fusion import FusionHead
from src.xai.gradcam import GradCAM
from src.xai.consistency_check import spatial_agreement


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to a single OCT scan (.dcm/.png/.jpg)")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--out-dir", default="data/reports")
    args = parser.parse_args()

    scan_id = Path(args.image).stem

    # --- Stage 0: ingest + QC ---
    loaded = load_scan(args.image)
    ok, reason = check_quality(loaded.pixels)
    if not ok:
        print(f"REJECTED: {reason}")
        return

    # --- Stage 1: preprocess (scale-aware) ---
    img, resize_record = letterbox_resize(
        loaded.pixels, target_size=(512, 512),
        axial_mm_per_px=loaded.meta.axial_mm_per_px,
        lateral_mm_per_px=loaded.meta.lateral_mm_per_px,
    )
    img = normalize_intensity(img, method="zscore")

    # --- Stage 2: embeddings ---
    retfound = RETFoundEncoder.from_pretrained("checkpoints/retfound.pt")
    internimage = InternImageEncoder.from_pretrained("checkpoints/internimage.pt")
    medvit = MedViTEncoder.from_pretrained("checkpoints/medvit.pt")
    emb_a, emb_b, emb_c = retfound(img), internimage(img), medvit(img)

    # --- Stage 3: segmentation ---
    nnunet = NNUNetSegmenter.from_pretrained("checkpoints/nnunet.pt")
    seg_mask = nnunet(img)

    # --- Stage 3.5: biomarkers (second pair of eyes) ---
    biomarker_head = BiomarkerHead.from_pretrained("checkpoints/biomarker_head.pt")  # add this classmethod alongside your training script
    biomarker_logits = biomarker_head(emb_a)  # or a fused embedding, per your fusion.py design
    biomarkers = predict_biomarkers(biomarker_logits, threshold=0.5)

    # --- Stage 4: diagnosis (closed-set + open-set) ---
    anomaly = AnomalyScorer.from_pretrained("checkpoints/anomaly_head.pt")
    anomaly_score = anomaly.score([emb_a, emb_b, emb_c])
    fusion = FusionHead.from_pretrained("checkpoints/fusion_head.pt")
    dx_label, dx_conf, differentials = fusion.classify([emb_a, emb_b, emb_c], seg_mask)

    diagnosis = Diagnosis(
        label=dx_label,
        confidence=dx_conf,
        differentials=differentials,
        is_atypical=anomaly_score > anomaly.threshold,
        anomaly_score=float(anomaly_score),
    )

    # --- Stage 5: XAI — the "proof" ---
    gradcam = GradCAM(internimage)
    cam = gradcam.explain(img)
    agreement = spatial_agreement(cam, seg_mask)

    seg_overlay_path = f"{args.out_dir}/{scan_id}_segmentation.png"
    cam_overlay_path = f"{args.out_dir}/{scan_id}_gradcam.png"
    # save_overlay(img, seg_mask, seg_overlay_path)   # implement in xai/ or preprocess/ using matplotlib
    # save_overlay(img, cam, cam_overlay_path)

    evidence = Evidence(
        segmentation_overlay_path=seg_overlay_path,
        gradcam_overlay_path=cam_overlay_path,
        xai_segmentation_agreement=agreement,
        measurements={
            "central_subfield_thickness_um": None,  # fill in from seg_mask + resize_record.pixels_to_mm(...)
        },
    )

    # --- Stage 7: report ---
    report = build_report(scan_id, diagnosis, biomarkers, evidence)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{args.out_dir}/{scan_id}_report.json", "w") as f:
        json.dump(report, f, indent=2)
    pdf_path = render_pdf(report, output_dir=args.out_dir)

    print(f"Diagnosis: {report['diagnosis']['label']} ({report['diagnosis']['confidence']:.0%})")
    print(f"Flagged biomarkers: {[b['name'] for b in report['biomarkers']['flagged']]}")
    print(f"Report saved: {pdf_path}")


if __name__ == "__main__":
    main()
