"""
transforms.py — Stage 1: preprocessing.

Core rule: resize for the network, never for measurement. Any quantitative
output (thickness, area, volume) must be computed by mapping a prediction
back through `ResizeRecord.to_original(...)`, not by measuring on the
resized image.
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class ResizeRecord:
    """Everything needed to undo the resize for measurement purposes."""
    original_shape: Tuple[int, int]     # (H, W) before any resizing
    resized_shape: Tuple[int, int]      # (H, W) fed to the network
    pad_top: int
    pad_left: int
    scale: float                         # single scalar since aspect ratio is preserved
    axial_mm_per_px: float | None
    lateral_mm_per_px: float | None

    def to_original_coords(self, row: np.ndarray, col: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Map pixel coordinates in the resized/padded image back to the original."""
        orig_row = (row - self.pad_top) / self.scale
        orig_col = (col - self.pad_left) / self.scale
        return orig_row, orig_col

    def pixels_to_mm(self, n_pixels_axial: float, n_pixels_lateral: float) -> Tuple[float | None, float | None]:
        """Convert a measurement made in ORIGINAL pixel units to mm, if calibrated."""
        axial_mm = n_pixels_axial * self.axial_mm_per_px if self.axial_mm_per_px else None
        lateral_mm = n_pixels_lateral * self.lateral_mm_per_px if self.lateral_mm_per_px else None
        return axial_mm, lateral_mm


def letterbox_resize(
    img: np.ndarray,
    target_size: Tuple[int, int],
    axial_mm_per_px: float | None = None,
    lateral_mm_per_px: float | None = None,
) -> Tuple[np.ndarray, ResizeRecord]:
    """
    Aspect-ratio-preserving resize with padding (letterbox), instead of a
    naive stretch to target_size. This keeps axial:lateral proportions
    intact — a plain cv2.resize(img, target_size) does NOT, because it
    scales height and width independently to hit an exact target shape.
    """
    h, w = img.shape[:2]
    target_h, target_w = target_size

    scale = min(target_h / h, target_w / w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))

    resized = _resize_2d(img, (new_h, new_w))

    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left

    padded = np.pad(
        resized,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
        constant_values=0,
    )

    record = ResizeRecord(
        original_shape=(h, w),
        resized_shape=target_size,
        pad_top=pad_top,
        pad_left=pad_left,
        scale=scale,
        axial_mm_per_px=axial_mm_per_px,
        lateral_mm_per_px=lateral_mm_per_px,
    )
    return padded, record


def _resize_2d(img: np.ndarray, new_shape: Tuple[int, int]) -> np.ndarray:
    """Bilinear resize without an OpenCV/PIL dependency (swap in cv2.resize
    for speed once you're off the free-tier laptop constraint)."""
    from scipy.ndimage import zoom  # pip install scipy

    zoom_factors = (new_shape[0] / img.shape[0], new_shape[1] / img.shape[1])
    return zoom(img, zoom_factors, order=1)


def normalize_intensity(img: np.ndarray, method: str = "zscore") -> np.ndarray:
    """Per-scan intensity normalization. z-score is a safer default than
    min-max for OCT because speckle noise creates a long tail that min-max
    normalization would compress the useful signal against."""
    if method == "zscore":
        mean, std = img.mean(), img.std() + 1e-8
        return (img - mean) / std
    elif method == "minmax":
        lo, hi = img.min(), img.max()
        return (img - lo) / (hi - lo + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")
