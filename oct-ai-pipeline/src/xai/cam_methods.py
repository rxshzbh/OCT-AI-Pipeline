"""
xai/cam_methods.py — Stage 5: explainability heatmaps.

Built on `pytorch-grad-cam` (pip install grad-cam) rather than reimplementing
CAM math — it already provides correct, tested implementations of every
method below.

Default choice: EigenCAM. It's gradient-free (PCA on activations, single
forward pass) and scored competitively against Grad-CAM/Grad-CAM++/Score-CAM
across several recent medical-imaging benchmarks, while being the cheapest
of the group computationally — the right tradeoff given limited GPU budget.

Grad-CAM++ is kept as a cheap second opinion (see xai_consistency_check.py
for how these get cross-referenced against each other AND against the
segmentation mask).

Score-CAM is included but flagged as expensive: it requires one forward
pass PER activation channel, which is a real cost multiplier on any GPU
you're paying for by the hour.
"""

from dataclasses import dataclass
from typing import Literal
import numpy as np
import torch

from pytorch_grad_cam import EigenCAM, GradCAMPlusPlus, ScoreCAM, LayerCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

CAMMethod = Literal["eigencam", "gradcam++", "scorecam", "layercam"]

_METHOD_REGISTRY = {
    "eigencam": EigenCAM,       # default — cheap, gradient-free
    "gradcam++": GradCAMPlusPlus,  # cheap second opinion
    "layercam": LayerCAM,       # sharper fine-boundary detail, multi-layer
    "scorecam": ScoreCAM,       # expensive — one forward pass per channel; use sparingly
}


@dataclass
class CAMResult:
    method: str
    heatmap: np.ndarray  # normalized 0-1, same H/W as input


def compute_cam(
    model: torch.nn.Module,
    target_layers: list,
    input_tensor: torch.Tensor,
    target_class: int,
    method: CAMMethod = "eigencam",
) -> CAMResult:
    """
    target_layers: the conv/attention layers to hook (e.g. the last
    residual block of InternImage-L). Pick these once per backbone and
    reuse — this is architecture-specific, not something to guess per call.
    """
    cam_cls = _METHOD_REGISTRY[method]
    with cam_cls(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=[ClassifierOutputTarget(target_class)],
        )[0]  # (H, W), already normalized 0-1
    return CAMResult(method=method, heatmap=grayscale_cam)


def compute_cam_ensemble(
    model: torch.nn.Module,
    target_layers: list,
    input_tensor: torch.Tensor,
    target_class: int,
    methods: list[CAMMethod] = ("eigencam", "gradcam++"),
) -> list[CAMResult]:
    """
    Run several cheap methods and return all of them — used by the
    consistency check to see whether independent methods agree on WHERE
    the model is looking, which is a stronger trust signal than any single
    method's heatmap on its own. Leave scorecam out of the default list
    unless you've budgeted for the extra compute.
    """
    return [
        compute_cam(model, target_layers, input_tensor, target_class, method=m)
        for m in methods
    ]
