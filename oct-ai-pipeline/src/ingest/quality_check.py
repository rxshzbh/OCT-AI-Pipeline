"""
quality_check.py — Stage 0 continued: reject scans that would silently
produce a garbage diagnosis. Cheap, explainable checks only — this is a
gate, not a model.
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class QualityThresholds:
    min_signal_std: float = 5.0        # near-blank/black frames
    blur_laplacian_var_min: float = 15.0  # below this = likely defocused
    max_saturation_fraction: float = 0.15  # fraction of pixels clipped at max


def check_quality(pixels: np.ndarray, thresholds: QualityThresholds = QualityThresholds()) -> Tuple[bool, str]:
    """
    Returns (ok, reason). If ok is False, reason explains why — this string
    goes straight into the report/logs so a rejected scan is diagnosable,
    not just silently dropped.
    """
    img = pixels if pixels.ndim == 2 else pixels[len(pixels) // 2]  # mid-slice for volumes

    # 1. Signal check — near-uniform frames (blank, occluded, wrong export)
    if img.std() < thresholds.min_signal_std:
        return False, f"low_signal (std={img.std():.2f} < {thresholds.min_signal_std})"

    # 2. Blur check via Laplacian variance — cheap, no extra model needed
    lap_var = _laplacian_variance(img)
    if lap_var < thresholds.blur_laplacian_var_min:
        return False, f"likely_defocused (laplacian_var={lap_var:.2f})"

    # 3. Saturation/clipping check — overexposed scans lose layer boundaries
    max_val = img.max()
    sat_fraction = float((img >= max_val * 0.98).mean())
    if sat_fraction > thresholds.max_saturation_fraction:
        return False, f"overexposed (saturated_fraction={sat_fraction:.2f})"

    return True, "ok"


def _laplacian_variance(img: np.ndarray) -> float:
    # Manual 2D Laplacian to avoid an OpenCV dependency for this one check.
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    padded = np.pad(img, 1, mode="edge")
    lap = (
        padded[:-2, 1:-1] * kernel[0, 1]
        + padded[1:-1, :-2] * kernel[1, 0]
        + padded[1:-1, 1:-1] * kernel[1, 1]
        + padded[1:-1, 2:] * kernel[1, 2]
        + padded[2:, 1:-1] * kernel[2, 1]
    )
    return float(lap.var())
