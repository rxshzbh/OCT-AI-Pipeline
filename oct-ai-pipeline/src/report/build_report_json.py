"""
report/build_report_json.py — Stage 7a: assemble the structured report.

This is the single source of truth for what goes in front of a doctor. It
does NOT render pixels — render_pdf.py consumes this dict and lays it out.
Keeping them separate means you can also serve this JSON directly (e.g. to
a hospital EHR integration) without needing the PDF at all.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from src.models.biomarker_head import BiomarkerResult


@dataclass
class Diagnosis:
    label: str
    confidence: float
    differentials: List[Dict[str, float]] = field(default_factory=list)  # [{"label": ..., "confidence": ...}]
    is_atypical: bool = False          # True if anomaly head flagged this as outside known classes
    anomaly_score: Optional[float] = None


@dataclass
class Evidence:
    """The three-part 'proof of what it saw', kept as separate artifacts
    rather than one merged image, so each can be shown/toggled individually
    in the report and each can be independently sanity-checked."""
    segmentation_overlay_path: str      # PNG: original scan + segmented regions
    gradcam_overlay_path: str           # PNG: classifier attention heatmap
    xai_segmentation_agreement: float   # 0-1, how much the heatmap and mask spatially overlap
    measurements: Dict[str, Optional[float]] = field(default_factory=dict)  # e.g. {"central_subfield_thickness_um": 312.4}


def build_report(
    scan_id: str,
    diagnosis: Diagnosis,
    biomarkers: List[BiomarkerResult],
    evidence: Evidence,
    biomarker_threshold: float = 0.5,
) -> dict:
    flagged = [b for b in biomarkers if b.confidence >= biomarker_threshold]
    borderline = [
        b for b in biomarkers
        if biomarker_threshold - 0.15 <= b.confidence < biomarker_threshold
    ]  # near-miss findings worth a second look, shown separately from clearly-negative ones

    low_confidence_explanation = evidence.xai_segmentation_agreement < 0.4

    return {
        "scan_id": scan_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "diagnosis": {
            "label": diagnosis.label,
            "confidence": round(diagnosis.confidence, 3),
            "differentials": diagnosis.differentials,
            "is_atypical": diagnosis.is_atypical,
            "anomaly_score": diagnosis.anomaly_score,
        },
        "biomarkers": {
            "flagged": [_biomarker_dict(b) for b in flagged],
            "borderline": [_biomarker_dict(b) for b in borderline],
            "all": [_biomarker_dict(b) for b in biomarkers],  # full checklist, sorted by confidence already
        },
        "evidence": {
            "segmentation_overlay": evidence.segmentation_overlay_path,
            "gradcam_overlay": evidence.gradcam_overlay_path,
            "xai_segmentation_agreement": round(evidence.xai_segmentation_agreement, 3),
            "low_confidence_explanation_warning": low_confidence_explanation,
            "measurements": evidence.measurements,
        },
        "disclaimer": "Decision-support output only. Not a standalone diagnostic device. Requires clinician review.",
    }


def _biomarker_dict(b: BiomarkerResult) -> dict:
    return {"name": b.name, "confidence": round(b.confidence, 3), "flagged": b.flagged}
