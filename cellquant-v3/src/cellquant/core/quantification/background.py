"""
Background estimation for fluorescence microscopy — seven methods.

Each method produces a ``BackgroundResult`` carrying:
- a single global scalar (backward-compatible),
- optional per-cell values (for local methods),
- background standard deviation (for error propagation in CTCF), and
- traceability fields (``method_used``, ``method_reason``).

The ``'auto'`` selector routes to the best method based on cell
coverage and background spatial uniformity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
from scipy.ndimage import (
    labeled_comprehension,
    maximum_filter,
    distance_transform_edt,
)

BackgroundMethod = Literal[
    "annular_ring",
    "masked_annular_ring",
    "voronoi",
    "rolling_ball",
    "white_tophat",
    "polynomial_surface",
    "global_roi",
    "auto",
]


@dataclass
class BackgroundResult:
    """Output of any background estimation method."""

    global_value: float
    per_cell_values: Optional[np.ndarray] = None
    method_used: str = ""
    method_reason: str = ""
    background_std: float = 0.0
    low_confidence: bool = False


# ── Helpers ────────────────────────────────────────────────────────


def _labels_from_masks(masks: np.ndarray) -> np.ndarray:
    labels = np.unique(masks)
    return labels[labels != 0]


def _cell_coverage(masks: np.ndarray) -> float:
    return float(np.count_nonzero(masks)) / masks.size


def _bg_cv(image: np.ndarray, masks: np.ndarray) -> float:
    bg = image[masks == 0].astype(np.float64)
    if len(bg) < 100:
        return 999.0
    mean = bg.mean()
    if mean < 1e-8:
        return 999.0
    return float(bg.std() / mean)


# ── Method (a): Annular ring per cell ──────────────────────────────


def _background_annular_ring(
    image: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray,
    ring_width: int = 5,
    gap: int = 2,
) -> Tuple[np.ndarray, float]:
    """Batch annular ring via ``maximum_filter`` on the label image.

    1.  Dilate all labels simultaneously using ``maximum_filter``
        with *size = 2*(gap+ring_width)+1*.
    2.  Create the "ring" as pixels that are labelled by the dilation
        but are background in the original mask.
    3.  For pixels that fall inside the gap (dilated by gap only),
        exclude them to produce the actual ring band.
    4.  ``labeled_comprehension`` on the ring gives per-cell medians.

    Complexity: O(image_size) — no per-cell Python loops.
    """
    outer_radius = gap + ring_width
    outer_size = 2 * outer_radius + 1
    inner_size = 2 * gap + 1

    outer_dilated = maximum_filter(masks, size=outer_size)
    inner_dilated = maximum_filter(masks, size=inner_size)

    # Ring = outer-dilated labels that are true background AND outside
    # the inner dilation zone of ANY cell
    bg_mask = masks == 0
    ring_labels = np.where(bg_mask & (outer_dilated > 0), outer_dilated, 0)
    # Remove the gap zone (pixels too close to any cell)
    ring_labels = np.where(inner_dilated > 0, 0, ring_labels)

    per_cell = labeled_comprehension(
        image.astype(np.float64), ring_labels, labels, np.median, np.float64, np.nan
    )
    # Replace NaN (cells with no ring pixels) with global background median
    global_bg = image[bg_mask].astype(np.float64)
    fallback = float(np.median(global_bg)) if global_bg.size > 0 else 0.0
    per_cell = np.where(np.isnan(per_cell), fallback, per_cell)

    bg_std = float(np.std(global_bg)) if global_bg.size > 0 else 0.0
    return per_cell, bg_std


# ── Method (b): Masked annular ring ────────────────────────────────


def _background_masked_annular_ring(
    image: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray,
    ring_width: int = 5,
    gap: int = 2,
) -> Tuple[np.ndarray, float]:
    """Like annular ring but the ring is intersected with ``masks == 0``
    so neighbouring cells are excluded.  The batch ``maximum_filter``
    approach naturally handles this because we already filter on
    ``bg_mask = (masks == 0)``."""
    # Identical to annular ring — the bg_mask intersection already
    # excludes all cell pixels.  Kept as separate entry for semantic
    # clarity (and possible future refinement of neighbour weighting).
    return _background_annular_ring(image, masks, labels, ring_width, gap)


# ── Method (c): Voronoi partitioned intercellular space ────────────


def _background_voronoi(
    image: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray,
    min_distance: int = 10,
) -> Tuple[np.ndarray, float]:
    """Partition background into Voronoi regions per cell.

    Uses ``distance_transform_edt`` on the cell mask to find the
    nearest cell for every background pixel in one vectorised call.
    Then ``labeled_comprehension`` computes per-cell median background.

    Args:
        min_distance: Minimum pixel distance from any cell boundary.
            Pixels closer than this are excluded to avoid halo/scatter
            contamination from neighbouring cells.  Falls back to all
            background pixels if fewer than 100 distant pixels exist.
    """
    bg_mask = masks == 0
    if not np.any(bg_mask):
        return np.zeros(len(labels), dtype=np.float64), 0.0

    # distance_transform_edt with return_indices gives coordinates of
    # the nearest foreground (cell) pixel for every pixel in the image.
    dist, nearest_idx = distance_transform_edt(bg_mask, return_indices=True)

    # Only use background pixels far enough from cells to avoid
    # halo / scattered-light contamination
    if min_distance > 0:
        usable_bg = bg_mask & (dist >= min_distance)
        # Progressive fallback if too few distant pixels
        if np.sum(usable_bg) < 100:
            for d in (5, 3, 1):
                usable_bg = bg_mask & (dist >= d)
                if np.sum(usable_bg) >= 100:
                    break
            else:
                usable_bg = bg_mask
    else:
        usable_bg = bg_mask

    # Map each usable background pixel to the label of its nearest cell
    voronoi_labels = np.zeros_like(masks)
    voronoi_labels[usable_bg] = masks[nearest_idx[0][usable_bg], nearest_idx[1][usable_bg]]

    per_cell = labeled_comprehension(
        image.astype(np.float64), voronoi_labels, labels, np.median, np.float64, np.nan
    )
    usable_pixels = image[usable_bg].astype(np.float64)
    fallback = float(np.median(usable_pixels)) if usable_pixels.size > 0 else 0.0
    per_cell = np.where(np.isnan(per_cell), fallback, per_cell)

    bg_std = float(np.std(usable_pixels)) if usable_pixels.size > 0 else 0.0
    return per_cell, bg_std


# ── Method (d): Rolling ball (morphological opening) ───────────────


def _background_rolling_ball(
    image: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray,
    radius: int = 50,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """True rolling ball via morphological opening with a disk SE.

    Returns per-cell background values extracted from the surface,
    the full background surface image, and std.
    """
    from skimage.morphology import disk, opening

    img_f = image.astype(np.float64)
    selem = disk(radius)
    bg_surface = opening(img_f, selem).astype(np.float64)

    per_cell = labeled_comprehension(
        bg_surface, masks, labels, np.median, np.float64, 0.0
    )
    bg_std = float(np.std(bg_surface[masks == 0])) if np.any(masks == 0) else 0.0
    return per_cell, bg_surface, bg_std


# ── Method (e): White top-hat ──────────────────────────────────────


def _background_white_tophat(
    image: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray,
    radius: int = 50,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """White top-hat: background = image − top_hat(image).

    The top-hat extracts bright features; the residual is the
    background surface.
    """
    from skimage.morphology import disk, white_tophat

    img_f = image.astype(np.float64)
    selem = disk(radius)
    tophat = white_tophat(img_f, selem).astype(np.float64)
    bg_surface = img_f - tophat

    per_cell = labeled_comprehension(
        bg_surface, masks, labels, np.median, np.float64, 0.0
    )
    bg_std = float(np.std(bg_surface[masks == 0])) if np.any(masks == 0) else 0.0
    return per_cell, bg_surface, bg_std


# ── Method (f): Polynomial surface fitting ─────────────────────────


def _background_polynomial_surface(
    image: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray,
    order: int = 3,
    max_samples: int = 10_000,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Fit a 2-D polynomial (degree 2–4) to background pixels.

    Subsamples background pixels onto a grid for speed, fits via
    least-squares, then evaluates over the full image.
    """
    bg_mask = masks == 0
    bg_ys, bg_xs = np.where(bg_mask)

    if len(bg_ys) < 50:
        # Not enough background pixels — fall back to global median
        med = float(np.median(image[bg_mask])) if len(bg_ys) > 0 else 0.0
        per_cell = np.full(len(labels), med, dtype=np.float64)
        bg_surface = np.full_like(image, med, dtype=np.float64)
        return per_cell, bg_surface, 0.0

    # Subsample for speed
    if len(bg_ys) > max_samples:
        idx = np.random.default_rng(42).choice(len(bg_ys), max_samples, replace=False)
        bg_ys_s, bg_xs_s = bg_ys[idx], bg_xs[idx]
    else:
        bg_ys_s, bg_xs_s = bg_ys, bg_xs

    # Normalise coordinates to [0, 1] for numerical stability
    h, w = image.shape[:2]
    yn = bg_ys_s.astype(np.float64) / max(h - 1, 1)
    xn = bg_xs_s.astype(np.float64) / max(w - 1, 1)

    # Build Vandermonde matrix for 2-D polynomial
    cols: list[np.ndarray] = []
    for py in range(order + 1):
        for px in range(order + 1 - py):
            cols.append((yn ** py) * (xn ** px))
    A = np.column_stack(cols)
    b = image[bg_ys_s, bg_xs_s].astype(np.float64)

    # Least-squares fit
    coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    # Evaluate surface over the full image
    yy, xx = np.mgrid[0:h, 0:w]
    yn_full = yy.astype(np.float64) / max(h - 1, 1)
    xn_full = xx.astype(np.float64) / max(w - 1, 1)

    bg_surface = np.zeros((h, w), dtype=np.float64)
    idx = 0
    for py in range(order + 1):
        for px in range(order + 1 - py):
            bg_surface += coeffs[idx] * (yn_full ** py) * (xn_full ** px)
            idx += 1

    per_cell = labeled_comprehension(
        bg_surface, masks, labels, np.median, np.float64, 0.0
    )
    residuals = image[bg_mask].astype(np.float64) - bg_surface[bg_mask]
    bg_std = float(np.std(residuals)) if residuals.size > 0 else 0.0
    return per_cell, bg_surface, bg_std


# ── Method (g): Global ROI (current behaviour) ────────────────────


def _background_global_roi(
    image: np.ndarray,
    masks: np.ndarray,
    statistic: str = "median",
) -> Tuple[float, float]:
    """Global background from all cell-free pixels.

    Returns (background_value, background_std).
    """
    bg_pixels = image[masks == 0].astype(np.float64)

    if bg_pixels.size == 0:
        return 0.0, 0.0

    if statistic == "median":
        val = float(np.median(bg_pixels))
    elif statistic == "mean":
        val = float(np.mean(bg_pixels))
    elif statistic == "mode":
        hist, edges = np.histogram(bg_pixels, bins=256)
        peak = np.argmax(hist)
        val = float((edges[peak] + edges[peak + 1]) / 2)
    elif statistic == "percentile5":
        val = float(np.percentile(bg_pixels, 5))
    else:
        val = float(np.median(bg_pixels))

    return val, float(np.std(bg_pixels))


# ── Auto-detection ─────────────────────────────────────────────────


def auto_detect_method(
    image: np.ndarray,
    masks: np.ndarray,
) -> Tuple[BackgroundMethod, str]:
    """Select the best background method based on image characteristics.

    Decision tree:
    - Cell coverage > 80 % → ``masked_annular_ring``
    - Background CV > 0.30 → ``polynomial_surface``
    - Background CV > 0.15 → ``voronoi``
    - Otherwise            → ``voronoi`` (per-cell local estimation)
    """
    coverage = _cell_coverage(masks)

    if coverage > 0.80:
        return "masked_annular_ring", (
            f"High cell coverage ({coverage:.0%}); using masked annular ring "
            "to sample intercellular gaps"
        )

    cv = _bg_cv(image, masks)

    if cv > 0.30:
        return "polynomial_surface", (
            f"High background non-uniformity (CV={cv:.2f}); "
            "using polynomial surface fit"
        )
    if cv > 0.15:
        return "voronoi", (
            f"Moderate background non-uniformity (CV={cv:.2f}); "
            "using Voronoi-partitioned local background"
        )

    return "voronoi", (
        f"Uniform background (CV={cv:.2f}); using Voronoi-partitioned local background"
    )


# ── Main entry point ───────────────────────────────────────────────


def estimate_background(
    image: np.ndarray,
    masks: np.ndarray,
    method: BackgroundMethod = "auto",
    *,
    negative_control: Optional[np.ndarray] = None,
    manual_bg: Optional[float] = None,
    ring_width: int = 5,
    ring_gap: int = 2,
    ball_radius: int = 50,
    poly_order: int = 3,
    global_statistic: str = "median",
) -> BackgroundResult:
    """Estimate background intensity using the specified method.

    Args:
        image: Fluorescence intensity image (2-D).
        masks: Label mask (0 = background, >0 = cells).
        method: One of the seven methods, or ``'auto'``.
        negative_control: Optional cell-free reference image (same acquisition).
        manual_bg: Optional manually measured background value.
        ring_width: Width of annular ring (methods a, b).
        ring_gap: Gap between cell boundary and ring (methods a, b).
        ball_radius: Radius for rolling ball / top-hat (methods d, e).
        poly_order: Polynomial degree (method f).
        global_statistic: Statistic for global ROI (method g).

    Returns:
        ``BackgroundResult`` with global and/or per-cell values.
    """
    labels = _labels_from_masks(masks)
    reason = ""
    low_confidence = False

    # ── Manual override ───────────────────────────────────────────
    if manual_bg is not None:
        return BackgroundResult(
            global_value=manual_bg,
            method_used="manual",
            method_reason="User-supplied manual background value",
            background_std=0.0,
        )

    # ── Negative control override ─────────────────────────────────
    if negative_control is not None:
        val = float(np.median(negative_control.astype(np.float64)))
        std = float(np.std(negative_control.astype(np.float64)))
        return BackgroundResult(
            global_value=val,
            method_used="negative_control",
            method_reason="Background from negative-control reference image",
            background_std=std,
        )

    # ── Auto-select ───────────────────────────────────────────────
    if method == "auto":
        method, reason = auto_detect_method(image, masks)

    # ── Insufficient background check ─────────────────────────────
    coverage = _cell_coverage(masks)
    bg_pixel_count = int(np.sum(masks == 0))

    if coverage > 0.95 and bg_pixel_count < 50 * max(len(labels), 1):
        # Not enough background for reliable estimation
        low_confidence = True
        reason += (
            " | WARNING: <5% background area — using 1st percentile "
            "as approximation. Provide a negative control for accuracy."
        )
        val = float(np.percentile(image.astype(np.float64), 1))
        bg = image[masks == 0].astype(np.float64)
        std = float(np.std(bg)) if bg.size > 0 else 0.0
        return BackgroundResult(
            global_value=val,
            method_used=f"{method}_fallback_p1",
            method_reason=reason,
            background_std=std,
            low_confidence=True,
        )

    # ── Dispatch to method ────────────────────────────────────────
    if method == "annular_ring":
        per_cell, bg_std = _background_annular_ring(
            image, masks, labels, ring_width, ring_gap
        )
        return BackgroundResult(
            global_value=float(np.median(per_cell)),
            per_cell_values=per_cell,
            method_used="annular_ring",
            method_reason=reason or "Annular ring per cell",
            background_std=bg_std,
        )

    if method == "masked_annular_ring":
        per_cell, bg_std = _background_masked_annular_ring(
            image, masks, labels, ring_width, ring_gap
        )
        return BackgroundResult(
            global_value=float(np.median(per_cell)),
            per_cell_values=per_cell,
            method_used="masked_annular_ring",
            method_reason=reason or "Masked annular ring (neighbours excluded)",
            background_std=bg_std,
        )

    if method == "voronoi":
        per_cell, bg_std = _background_voronoi(image, masks, labels)
        return BackgroundResult(
            global_value=float(np.median(per_cell)),
            per_cell_values=per_cell,
            method_used="voronoi",
            method_reason=reason or "Voronoi-partitioned intercellular space",
            background_std=bg_std,
        )

    if method == "rolling_ball":
        per_cell, _, bg_std = _background_rolling_ball(
            image, masks, labels, ball_radius
        )
        return BackgroundResult(
            global_value=float(np.median(per_cell)),
            per_cell_values=per_cell,
            method_used="rolling_ball",
            method_reason=reason or f"Rolling ball (radius={ball_radius})",
            background_std=bg_std,
        )

    if method == "white_tophat":
        per_cell, _, bg_std = _background_white_tophat(
            image, masks, labels, ball_radius
        )
        return BackgroundResult(
            global_value=float(np.median(per_cell)),
            per_cell_values=per_cell,
            method_used="white_tophat",
            method_reason=reason or f"White top-hat (radius={ball_radius})",
            background_std=bg_std,
        )

    if method == "polynomial_surface":
        per_cell, _, bg_std = _background_polynomial_surface(
            image, masks, labels, poly_order
        )
        return BackgroundResult(
            global_value=float(np.median(per_cell)),
            per_cell_values=per_cell,
            method_used="polynomial_surface",
            method_reason=reason or f"Polynomial surface (order={poly_order})",
            background_std=bg_std,
        )

    if method == "global_roi":
        val, bg_std = _background_global_roi(image, masks, global_statistic)
        return BackgroundResult(
            global_value=val,
            method_used="global_roi",
            method_reason=reason or f"Global ROI ({global_statistic})",
            background_std=bg_std,
        )

    raise ValueError(f"Unknown background method: {method}")


# ── Legacy helper ──────────────────────────────────────────────────


def subtract_background(
    image: np.ndarray,
    background: float,
    clip_negative: bool = True,
) -> np.ndarray:
    """Subtract a scalar background from an image."""
    result = image.astype(np.float64) - background
    if clip_negative:
        np.maximum(result, 0.0, out=result)
    return result
