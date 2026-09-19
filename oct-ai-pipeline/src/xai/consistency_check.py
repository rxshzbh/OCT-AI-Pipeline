"""
xai/consistency_check.py — Stage 5 continued: does the explanation agree
with itself?

Two separate checks, both cheap and both worth reporting:
1. heatmap vs segmentation mask agreement (does model attention line up
   with the region it segmented as abnormal?)
2. cross-method agreement (do EigenCAM and Grad-CAM++ point at roughly
   the same place?) — independent methods agreeing is a stronger trust
   signal than either one alone.

Both return a 0-1 score. Low scores should surface as a visible warning
in the report (see render_pdf.py) rather than being silently included as
if the explanation were reliable.
"""

import numpy as np


def spatial_agreement(heatmap: np.ndarray, mask: np.ndarray, threshold: float = 0.5) -> float:
    """
    IoU between the thresholded heatmap and the segmentation mask.
    Both inputs should already be the same H/W (i.e. mapped back through
    the same ResizeRecord — see preprocess/transforms.py — before calling
    this, not compared at two different resolutions).
    """
    heat_binary = heatmap >= threshold
    mask_binary = mask.astype(bool)

    intersection = np.logical_and(heat_binary, mask_binary).sum()
    union = np.logical_or(heat_binary, mask_binary).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def cross_method_agreement(heatmap_a: np.ndarray, heatmap_b: np.ndarray) -> float:
    """
    Pearson correlation between two CAM methods' raw (non-thresholded)
    heatmaps. High correlation = independent methods are pointing at the
    same evidence, which is more convincing than one method's word alone.
    """
    a, b = heatmap_a.flatten(), heatmap_b.flatten()
    if a.std() == 0 or b.std() == 0:
        return 0.0
    corr = np.corrcoef(a, b)[0, 1]
    return float(np.clip(corr, 0.0, 1.0))  # negative correlation isn't meaningful here, clip to 0
