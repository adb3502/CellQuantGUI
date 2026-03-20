"""Load and validate dark-frame and flat-field calibration images."""

from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
from skimage import io as skio


def load_dark_frames(paths: List[Union[str, Path]]) -> np.ndarray:
    """Load one or more dark-frame TIFFs and compute the master dark.

    Uses the pixel-wise median across frames (robust to cosmic rays and
    hot-pixel transients).  A single frame is returned unchanged.

    Args:
        paths: Paths to dark-frame TIFF files (must all share the same shape).

    Returns:
        2-D master dark image (float64).
    """
    if not paths:
        raise ValueError("At least one dark-frame path is required")

    frames = []
    ref_shape = None
    for p in paths:
        img = skio.imread(str(p)).astype(np.float64)
        if img.ndim > 2:
            img = img[0] if img.shape[0] <= 5 else img[:, :, 0]
        if ref_shape is None:
            ref_shape = img.shape
        elif img.shape != ref_shape:
            raise ValueError(
                f"Dark frame {p} shape {img.shape} does not match "
                f"reference shape {ref_shape}"
            )
        frames.append(img)

    if len(frames) == 1:
        return frames[0]
    return np.median(np.stack(frames, axis=0), axis=0)


def load_flat_field(paths: List[Union[str, Path]]) -> np.ndarray:
    """Load flat-field image(s) and compute the normalised flat.

    The result is scaled so that its median equals 1.0.  A floor of 1 %
    of the median is applied to avoid division-by-zero artefacts.

    Args:
        paths: Paths to flat-field TIFF files.

    Returns:
        2-D normalised flat-field image (float64, median ≈ 1.0).
    """
    if not paths:
        raise ValueError("At least one flat-field path is required")

    frames = []
    ref_shape = None
    for p in paths:
        img = skio.imread(str(p)).astype(np.float64)
        if img.ndim > 2:
            img = img[0] if img.shape[0] <= 5 else img[:, :, 0]
        if ref_shape is None:
            ref_shape = img.shape
        elif img.shape != ref_shape:
            raise ValueError(
                f"Flat-field {p} shape {img.shape} does not match "
                f"reference shape {ref_shape}"
            )
        frames.append(img)

    flat = frames[0] if len(frames) == 1 else np.median(np.stack(frames, axis=0), axis=0)

    med = np.median(flat)
    if med <= 0:
        raise ValueError("Flat-field median is <= 0; the image appears blank")

    flat_norm = flat / med
    # Floor at 1% of median to avoid division-by-near-zero
    flat_norm = np.maximum(flat_norm, 0.01)
    return flat_norm


def validate_calibration_frame(
    calibration: np.ndarray,
    reference_shape: Tuple[int, int],
    frame_type: str = "dark",
) -> List[str]:
    """Check that a calibration frame is compatible with science images.

    Args:
        calibration: Master dark or normalised flat array.
        reference_shape: (H, W) of the science images.
        frame_type: ``"dark"`` or ``"flat"`` (for error messages).

    Returns:
        List of warning/error strings.  Empty means valid.
    """
    warnings: List[str] = []

    if calibration.ndim != 2:
        warnings.append(f"{frame_type} frame must be 2-D, got {calibration.ndim}-D")
        return warnings

    if calibration.shape != reference_shape:
        warnings.append(
            f"{frame_type} frame shape {calibration.shape} does not match "
            f"science image shape {reference_shape}"
        )

    if frame_type == "flat":
        med = np.median(calibration)
        if abs(med - 1.0) > 0.05:
            warnings.append(
                f"Flat-field median is {med:.3f} (expected ~1.0). "
                "Was the flat normalised correctly?"
            )
        low_frac = np.mean(calibration < 0.5)
        if low_frac > 0.1:
            warnings.append(
                f"{low_frac*100:.1f}% of flat-field pixels are below 0.5. "
                "Check for vignetting or a bad flat-field acquisition."
            )

    return warnings
