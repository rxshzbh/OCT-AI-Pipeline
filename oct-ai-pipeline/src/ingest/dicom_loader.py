"""
dicom_loader.py — Stage 0: ingestion.

Loads an OCT scan (DICOM, or a plain PNG/JPG export) and returns both the
pixel array AND the physical scale metadata (mm-per-pixel in axial and
lateral directions). This metadata must travel with the image through every
later stage — it is what lets us convert a segmentation mask back into a
real thickness/volume measurement after the image has been resized for the
network.

If a plain PNG/JPG has no embedded scale (common with de-identified export
datasets like Kermany/OCT2017), scale is left as None and flagged — any
downstream quantitative measurement on that scan should be marked
"relative units only, not calibrated" in the report.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np


@dataclass
class ScanMeta:
    axial_mm_per_px: Optional[float]     # depth direction (rows)
    lateral_mm_per_px: Optional[float]    # scan-width direction (cols)
    manufacturer: Optional[str] = None
    modality: Optional[str] = None
    is_calibrated: bool = False


@dataclass
class LoadedScan:
    pixels: np.ndarray          # (H, W) or (D, H, W) for a volume
    meta: ScanMeta
    source_path: str


def load_scan(path: str) -> LoadedScan:
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix in (".dcm", ".dicom"):
        return _load_dicom(p)
    elif suffix in (".png", ".jpg", ".jpeg", ".tiff", ".bmp"):
        return _load_plain_image(p)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def _load_dicom(p: Path) -> LoadedScan:
    import pydicom  # pip install pydicom

    ds = pydicom.dcmread(str(p))
    pixels = ds.pixel_array.astype(np.float32)

    # Pixel spacing tags vary by manufacturer; check the common locations.
    axial_mm, lateral_mm = None, None
    if hasattr(ds, "PixelSpacing"):
        # DICOM PixelSpacing is [row_spacing, col_spacing] in mm
        row_spacing, col_spacing = ds.PixelSpacing
        axial_mm, lateral_mm = float(row_spacing), float(col_spacing)
    elif hasattr(ds, "SharedFunctionalGroupsSequence"):
        # Some OCT DICOMs (e.g. Heidelberg, Topcon) nest spacing here
        try:
            seq = ds.SharedFunctionalGroupsSequence[0]
            spacing = seq.PixelMeasuresSequence[0].PixelSpacing
            axial_mm, lateral_mm = float(spacing[0]), float(spacing[1])
        except (AttributeError, IndexError):
            pass

    meta = ScanMeta(
        axial_mm_per_px=axial_mm,
        lateral_mm_per_px=lateral_mm,
        manufacturer=getattr(ds, "Manufacturer", None),
        modality=getattr(ds, "Modality", None),
        is_calibrated=axial_mm is not None and lateral_mm is not None,
    )
    return LoadedScan(pixels=pixels, meta=meta, source_path=str(p))


def _load_plain_image(p: Path) -> LoadedScan:
    import imageio.v3 as iio  # pip install imageio

    pixels = iio.imread(str(p)).astype(np.float32)
    if pixels.ndim == 3:
        # collapse RGB exports to grayscale (OCT is inherently grayscale)
        pixels = pixels.mean(axis=-1)

    meta = ScanMeta(
        axial_mm_per_px=None,
        lateral_mm_per_px=None,
        is_calibrated=False,
    )
    return LoadedScan(pixels=pixels, meta=meta, source_path=str(p))
