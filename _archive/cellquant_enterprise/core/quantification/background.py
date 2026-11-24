"""
Background estimation methods for fluorescence microscopy.

Provides multiple strategies for estimating background intensity
from regions outside cells.
"""

from typing import Literal, Optional, Tuple
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter


BackgroundMethod = Literal['median', 'percentile5', 'mean', 'mode', 'rolling_ball']


def estimate_background(
    image: np.ndarray,
    masks: np.ndarray,
    method: BackgroundMethod = 'median'
) -> float:
    """
    Estimate background intensity from non-cell regions.

    Args:
        image: Fluorescence intensity image
        masks: Label mask (0 = background, >0 = cells)
        method: Estimation method
            - 'median': Median of background pixels (default, robust)
            - 'percentile5': 5th percentile (for images with some noise)
            - 'mean': Mean of background pixels (sensitive to outliers)
            - 'mode': Most common value (good for uniform backgrounds)

    Returns:
        Estimated background intensity

    Example:
        >>> bg = estimate_background(marker_img, masks, method='median')
        >>> print(f"Background: {bg:.2f}")
    """
    # Get background pixels (where mask is 0)
    background_pixels = image[masks == 0]

    # Handle case with very few background pixels
    if len(background_pixels) < (image.size * 0.01):
        # Less than 1% background - use 5th percentile of whole image
        print(f"Warning: Limited background pixels ({len(background_pixels)}), "
              "using 5th percentile of entire image")
        return float(np.percentile(image, 5))

    if method == 'median':
        return float(np.median(background_pixels))

    elif method == 'percentile5':
        return float(np.percentile(background_pixels, 5))

    elif method == 'mean':
        return float(np.mean(background_pixels))

    elif method == 'mode':
        # Approximate mode using histogram
        hist, bin_edges = np.histogram(background_pixels, bins=256)
        mode_idx = np.argmax(hist)
        return float((bin_edges[mode_idx] + bin_edges[mode_idx + 1]) / 2)

    elif method == 'rolling_ball':
        # Rolling ball background subtraction (approximation)
        # Use large median filter as proxy
        bg_image = median_filter(image.astype(np.float32), size=50)
        return float(np.median(bg_image[masks == 0]))

    else:
        raise ValueError(f"Unknown method: {method}")


def estimate_background_per_cell(
    image: np.ndarray,
    masks: np.ndarray,
    neighborhood_size: int = 50
) -> np.ndarray:
    """
    Estimate local background for each cell.

    For images with non-uniform background (e.g., illumination gradients),
    this provides per-cell background estimates based on nearby regions.

    Args:
        image: Fluorescence intensity image
        masks: Label mask
        neighborhood_size: Size of local neighborhood for estimation

    Returns:
        Array of background values, one per cell label
    """
    from scipy.ndimage import uniform_filter

    # Create smoothed background estimate
    bg_estimate = uniform_filter(image.astype(np.float32), size=neighborhood_size)

    # Get unique cell labels
    labels = np.unique(masks)
    labels = labels[labels != 0]

    # For each cell, get the local background from the smoothed image
    backgrounds = np.zeros(len(labels), dtype=np.float64)

    for i, label in enumerate(labels):
        cell_mask = masks == label
        # Get background from the boundary of the cell
        from scipy.ndimage import binary_dilation
        dilated = binary_dilation(cell_mask, iterations=5)
        boundary = dilated & ~cell_mask & (masks == 0)

        if np.any(boundary):
            backgrounds[i] = np.median(image[boundary])
        else:
            # Fall back to smoothed estimate at cell centroid
            y, x = np.where(cell_mask)
            cy, cx = int(y.mean()), int(x.mean())
            backgrounds[i] = bg_estimate[cy, cx]

    return backgrounds


def subtract_background(
    image: np.ndarray,
    background: float,
    clip_negative: bool = True
) -> np.ndarray:
    """
    Subtract background from image.

    Args:
        image: Input image
        background: Background value to subtract
        clip_negative: Whether to clip negative values to 0

    Returns:
        Background-subtracted image
    """
    result = image.astype(np.float64) - background

    if clip_negative:
        result = np.maximum(result, 0)

    return result


def estimate_background_from_corners(
    image: np.ndarray,
    corner_fraction: float = 0.1
) -> float:
    """
    Estimate background from image corners.

    Useful when cells cover most of the field of view,
    assuming corners are more likely to be background.

    Args:
        image: Input image
        corner_fraction: Fraction of image size for corner regions

    Returns:
        Estimated background
    """
    h, w = image.shape[:2]
    ch = int(h * corner_fraction)
    cw = int(w * corner_fraction)

    corners = [
        image[:ch, :cw],           # Top-left
        image[:ch, -cw:],          # Top-right
        image[-ch:, :cw],          # Bottom-left
        image[-ch:, -cw:],         # Bottom-right
    ]

    corner_pixels = np.concatenate([c.flatten() for c in corners])
    return float(np.median(corner_pixels))


def auto_detect_background_method(
    image: np.ndarray,
    masks: np.ndarray
) -> Tuple[BackgroundMethod, str]:
    """
    Automatically select best background method based on image characteristics.

    Args:
        image: Input image
        masks: Label mask

    Returns:
        Tuple of (recommended method, reasoning)
    """
    background_pixels = image[masks == 0]

    if len(background_pixels) < (image.size * 0.05):
        return 'percentile5', "Limited background area (<5%), using 5th percentile"

    # Check for uniform background vs. gradient
    bg_std = np.std(background_pixels)
    bg_mean = np.mean(background_pixels)
    cv = bg_std / (bg_mean + 1e-8)  # Coefficient of variation

    if cv < 0.1:
        return 'median', "Uniform background (CV < 0.1), using median"
    elif cv < 0.3:
        return 'median', "Moderately uniform background, using median"
    else:
        return 'percentile5', f"Variable background (CV={cv:.2f}), using 5th percentile"
