"""
biomarker_head.py — Stage 3.5: biomarker identification.

Deliberately separate from the diagnosis head (models/fusion.py). Runs on
the same fused embedding but is trained independently with multi-label
loss, so it can flag a biomarker even when it doesn't change the final
diagnosis. This is the "second pair of eyes" component — its whole job is
to surface small/subtle findings, not to agree with the diagnosis head.

Trained on OLIVES (9,408 biomarker-labeled OCT images, 16 expert-graded
biomarkers). Swap BIOMARKER_NAMES / NUM_BIOMARKERS if your OLIVES export
uses a different subset.
"""

from dataclasses import dataclass
from typing import Dict, List
import torch
import torch.nn as nn

BIOMARKER_NAMES: List[str] = [
    "intraretinal_fluid",
    "subretinal_fluid",
    "intraretinal_hyperreflective_foci",
    "partially_attached_vitreous_face",
    "fully_attached_vitreous_face",
    "vitreous_debris",
    "diffuse_retinal_thickening",
    "disrupted_ellipsoid_zone",
    "drusen",
    "epiretinal_membrane",
    "subretinal_hyperreflective_material",
    "pigment_epithelial_detachment",
    "atrophy",
    "fibrosis",
    "disorganization_of_retinal_inner_layers",
    "cystoid_macular_edema",
]  # replace with the exact 16 from your OLIVES export if names differ
NUM_BIOMARKERS = len(BIOMARKER_NAMES)


class BiomarkerHead(nn.Module):
    """Multi-label sigmoid head over a fused embedding vector."""

    def __init__(self, embedding_dim: int, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, NUM_BIOMARKERS),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """Returns raw logits, shape (batch, NUM_BIOMARKERS). Apply sigmoid
        outside the model (e.g. in loss/inference) to keep BCEWithLogitsLoss
        numerically stable during training."""
        return self.net(embedding)


class BiomarkerLoss(nn.Module):
    """
    Per-biomarker weighted BCE. Pass pos_weight computed from OLIVES label
    frequencies (rarer biomarkers get a higher weight) so the model doesn't
    just learn to always predict "absent" for the long-tail biomarkers.
    """

    def __init__(self, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(logits, targets)


@dataclass
class BiomarkerResult:
    name: str
    confidence: float
    flagged: bool  # confidence >= threshold


def predict_biomarkers(
    logits: torch.Tensor, threshold: float = 0.5
) -> List[BiomarkerResult]:
    """
    Converts raw logits into a full, sorted checklist — including
    below-threshold biomarkers. The report should show ALL of these, not
    just the flagged ones (see report/build_report_json.py), since a
    near-threshold score is often exactly what's worth a doctor's second
    look.
    """
    probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()
    results = [
        BiomarkerResult(name=name, confidence=float(p), flagged=bool(p >= threshold))
        for name, p in zip(BIOMARKER_NAMES, probs)
    ]
    return sorted(results, key=lambda r: r.confidence, reverse=True)


def compute_pos_weight(label_counts: Dict[str, int], total_samples: int) -> torch.Tensor:
    """
    label_counts: {biomarker_name: number_of_positive_samples_in_training_set}
    Standard inverse-frequency weighting for BCEWithLogitsLoss's pos_weight.
    """
    weights = []
    for name in BIOMARKER_NAMES:
        pos = max(label_counts.get(name, 1), 1)
        neg = max(total_samples - pos, 1)
        weights.append(neg / pos)
    return torch.tensor(weights, dtype=torch.float32)
