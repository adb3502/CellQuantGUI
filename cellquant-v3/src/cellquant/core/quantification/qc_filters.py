"""
Post-segmentation quality-control filters.

Applies configurable morphological quality gates (border removal, area,
solidity, eccentricity, circularity, aspect ratio) to a label mask
**before** quantification.  All heavy lifting is delegated to
``skimage.measure.regionprops_table`` which runs in C — no per-cell
Python loops.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class QCFilterConfig:
    """Thresholds for post-segmentation quality filtering.

    Set any threshold to ``None`` to disable that filter.
    Defaults are suitable for adherent epithelial cell cultures.
    """

    remove_border_objects: bool = True
    min_area: Optional[int] = None
    max_area: Optional[int] = None
    area_iqr_factor: float = 1.5  # Only used when min/max area explicitly enabled
    min_solidity: Optional[float] = None
    max_eccentricity: Optional[float] = None
    min_circularity: Optional[float] = None
    max_aspect_ratio: Optional[float] = None


@dataclass
class CellMorphology:
    """Morphological measurements for every cell in a mask.

    All arrays are parallel (same length, same cell ordering).
    """

    labels: np.ndarray
    areas: np.ndarray
    centroids_y: np.ndarray
    centroids_x: np.ndarray
    perimeters: np.ndarray
    solidities: np.ndarray
    eccentricities: np.ndarray
    circularities: np.ndarray
    aspect_ratios: np.ndarray
    is_border: np.ndarray  # bool


def compute_cell_morphology(masks: np.ndarray) -> CellMorphology:
    """Compute morphological properties for all cells.

    Uses ``skimage.measure.regionprops_table`` (vectorised, runs in C).

    Args:
        masks: 2-D integer label mask (0 = background).

    Returns:
        ``CellMorphology`` with parallel arrays for each metric.
    """
    from skimage.measure import regionprops_table
    from skimage.segmentation import clear_border

    props = regionprops_table(
        masks,
        properties=(
            "label",
            "area",
            "centroid",
            "perimeter",
            "solidity",
            "eccentricity",
            "major_axis_length",
            "minor_axis_length",
        ),
    )

    labels = props["label"].astype(np.int32)
    areas = props["area"].astype(np.float64)
    centroids_y = props["centroid-0"].astype(np.float64)
    centroids_x = props["centroid-1"].astype(np.float64)
    perimeters = props["perimeter"].astype(np.float64)
    solidities = props["solidity"].astype(np.float64)
    eccentricities = props["eccentricity"].astype(np.float64)

    # Circularity = 4 * pi * area / perimeter^2
    with np.errstate(divide="ignore", invalid="ignore"):
        circularities = np.where(
            perimeters > 0,
            4.0 * np.pi * areas / (perimeters ** 2),
            0.0,
        )
    circularities = np.clip(circularities, 0.0, 1.0)

    # Aspect ratio = major / minor axis
    major = props["major_axis_length"].astype(np.float64)
    minor = props["minor_axis_length"].astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        aspect_ratios = np.where(minor > 0, major / minor, 999.0)

    # Border detection: cells present in original but absent after
    # clear_border are border objects.
    cleared = clear_border(masks)
    border_set = set(labels.tolist()) - set(np.unique(cleared).tolist())
    is_border = np.array([lb in border_set for lb in labels], dtype=bool)

    return CellMorphology(
        labels=labels,
        areas=areas,
        centroids_y=centroids_y,
        centroids_x=centroids_x,
        perimeters=perimeters,
        solidities=solidities,
        eccentricities=eccentricities,
        circularities=circularities,
        aspect_ratios=aspect_ratios,
        is_border=is_border,
    )


def auto_area_thresholds(
    areas: np.ndarray,
    iqr_factor: float = 1.5,
) -> Tuple[float, float]:
    """Compute area thresholds from IQR-based outlier fences.

    Returns (min_area, max_area).
    """
    q1 = float(np.percentile(areas, 25))
    q3 = float(np.percentile(areas, 75))
    iqr = q3 - q1
    return max(q1 - iqr_factor * iqr, 1.0), q3 + iqr_factor * iqr


def apply_qc_filters(
    masks: np.ndarray,
    config: QCFilterConfig,
    morphology: Optional[CellMorphology] = None,
) -> Tuple[np.ndarray, CellMorphology, Dict[str, int]]:
    """Apply QC filters and return cleaned masks.

    Args:
        masks: Input label mask (2-D int array).
        config: Filter configuration.
        morphology: Pre-computed morphology (computed if ``None``).

    Returns:
        Tuple of:
        - **filtered_masks**: Copy with rejected cells zeroed out.
        - **morphology**: ``CellMorphology`` for the *original* cells.
        - **rejection_counts**: ``{filter_name: n_rejected}`` dict.
    """
    if morphology is None:
        morphology = compute_cell_morphology(masks)

    n = len(morphology.labels)
    keep = np.ones(n, dtype=bool)
    rejection_counts: Dict[str, int] = {}

    # ── Border objects ────────────────────────────────────────────
    if config.remove_border_objects:
        border_mask = morphology.is_border
        rejection_counts["border"] = int(border_mask.sum())
        keep &= ~border_mask

    # ── Area (only applied when explicitly set) ─────────────────
    if config.min_area is not None:
        too_small = morphology.areas < config.min_area
    else:
        too_small = np.zeros(n, dtype=bool)

    if config.max_area is not None:
        too_large = morphology.areas > config.max_area
    else:
        too_large = np.zeros(n, dtype=bool)

    rejection_counts["area_small"] = int(too_small.sum())
    rejection_counts["area_large"] = int(too_large.sum())
    keep &= ~too_small & ~too_large

    # ── Solidity ──────────────────────────────────────────────────
    if config.min_solidity is not None:
        low_solidity = morphology.solidities < config.min_solidity
        rejection_counts["solidity"] = int(low_solidity.sum())
        keep &= ~low_solidity

    # ── Eccentricity ──────────────────────────────────────────────
    if config.max_eccentricity is not None:
        high_ecc = morphology.eccentricities > config.max_eccentricity
        rejection_counts["eccentricity"] = int(high_ecc.sum())
        keep &= ~high_ecc

    # ── Circularity ───────────────────────────────────────────────
    if config.min_circularity is not None:
        low_circ = morphology.circularities < config.min_circularity
        rejection_counts["circularity"] = int(low_circ.sum())
        keep &= ~low_circ

    # ── Aspect ratio ──────────────────────────────────────────────
    if config.max_aspect_ratio is not None:
        high_ar = morphology.aspect_ratios > config.max_aspect_ratio
        rejection_counts["aspect_ratio"] = int(high_ar.sum())
        keep &= ~high_ar

    # ── Build filtered mask ───────────────────────────────────────
    rejected_labels = set(morphology.labels[~keep].tolist())
    if rejected_labels:
        filtered_masks = masks.copy()
        # Vectorised: zero out all rejected labels in one pass
        reject_lut = np.ones(masks.max() + 1, dtype=bool)
        for lb in rejected_labels:
            reject_lut[lb] = False
        filtered_masks = np.where(reject_lut[masks], masks, 0)
    else:
        filtered_masks = masks

    rejection_counts["total_rejected"] = int((~keep).sum())
    rejection_counts["total_kept"] = int(keep.sum())

    return filtered_masks, morphology, rejection_counts
