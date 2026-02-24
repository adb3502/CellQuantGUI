"""
Vectorised CTCF (Corrected Total Cell Fluorescence) calculation.

Enhanced pipeline with:
- Per-cell or global background support
- Error propagation:  σ_CTCF = √(A·σ²_I + A²·σ²_bg)
- Mean Corrected Fluorescence (MCF = CTCF / Area)
- Min / max / std intensity per cell
- Quality flags: is_saturated, is_dim
- Full output table with morphology columns

All heavy computation uses ``scipy.ndimage.labeled_comprehension``
which processes every cell in a single C-level pass through the image.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
from scipy.ndimage import labeled_comprehension

if TYPE_CHECKING:
    import pandas as pd
    from cellquant.core.quantification.qc_filters import CellMorphology


# ── Result container ───────────────────────────────────────────────


@dataclass
class QuantificationResult:
    """Results from CTCF quantification for a single marker.

    All arrays are parallel (same length, indexed by cell position in *labels*).
    """

    marker_name: str
    cell_ids: np.ndarray
    areas: np.ndarray
    integrated_densities: np.ndarray
    mean_intensities: np.ndarray
    ctcf: np.ndarray
    background: float  # global scalar (backward compat)

    # ── New fields ────────────────────────────────────────────────
    per_cell_backgrounds: Optional[np.ndarray] = None
    background_std: float = 0.0
    ctcf_sigma: Optional[np.ndarray] = None
    mcf: Optional[np.ndarray] = None
    min_intensities: Optional[np.ndarray] = None
    max_intensities: Optional[np.ndarray] = None
    std_intensities: Optional[np.ndarray] = None
    is_saturated: Optional[np.ndarray] = None
    is_dim: Optional[np.ndarray] = None

    # ── Legacy ────────────────────────────────────────────────────
    nuclear_signal: Optional[np.ndarray] = None
    is_mitochondrial: bool = False

    @property
    def n_cells(self) -> int:
        return len(self.cell_ids)

    def to_dict(self) -> Dict[str, np.ndarray]:
        """Flat dict of per-marker columns for DataFrame construction."""
        mk = self.marker_name
        d: Dict[str, np.ndarray] = {
            f"{mk}_CTCF": self.ctcf,
            f"{mk}_IntegratedDensity": self.integrated_densities,
            f"{mk}_MeanIntensity": self.mean_intensities,
            f"{mk}_Background": (
                self.per_cell_backgrounds
                if self.per_cell_backgrounds is not None
                else np.full_like(self.ctcf, self.background)
            ),
        }
        if self.mcf is not None:
            d[f"{mk}_MCF"] = self.mcf
        if self.ctcf_sigma is not None:
            d[f"{mk}_CTCF_sigma"] = self.ctcf_sigma
        if self.min_intensities is not None:
            d[f"{mk}_MinIntensity"] = self.min_intensities
        if self.max_intensities is not None:
            d[f"{mk}_MaxIntensity"] = self.max_intensities
        if self.std_intensities is not None:
            d[f"{mk}_StdIntensity"] = self.std_intensities
        return d


# ── Core calculation ───────────────────────────────────────────────

_EMPTY_F64 = np.array([], dtype=np.float64)
_EMPTY_I32 = np.array([], dtype=np.int32)
_EMPTY_BOOL = np.array([], dtype=bool)


def calculate_ctcf_vectorized(
    marker_image: np.ndarray,
    masks: np.ndarray,
    background: float,
    marker_name: str = "Marker",
    *,
    per_cell_backgrounds: Optional[np.ndarray] = None,
    background_std: float = 0.0,
    compute_error: bool = True,
    saturation_threshold: Optional[float] = None,
    dim_threshold_percentile: float = 5.0,
    nuclear_masks: Optional[np.ndarray] = None,
    is_mitochondrial: bool = False,
) -> QuantificationResult:
    """Calculate CTCF for all cells in a single vectorised pass.

    Args:
        marker_image: 2-D fluorescence image.
        masks: 2-D integer label mask (0 = background).
        background: Global background scalar.
        marker_name: Column-name prefix.
        per_cell_backgrounds: Optional per-cell background from local methods.
        background_std: Std dev of background (for error propagation).
        compute_error: Whether to compute σ_CTCF.
        saturation_threshold: Auto-detected from dtype if ``None``.
        dim_threshold_percentile: Percentile threshold for dim flag.
        nuclear_masks: For mitochondrial correction.
        is_mitochondrial: Apply mitochondrial correction.

    Returns:
        ``QuantificationResult`` with all metrics.
    """
    labels = np.unique(masks)
    labels = labels[labels != 0]

    if len(labels) == 0:
        return QuantificationResult(
            marker_name=marker_name,
            cell_ids=_EMPTY_I32,
            areas=_EMPTY_F64,
            integrated_densities=_EMPTY_F64,
            mean_intensities=_EMPTY_F64,
            ctcf=_EMPTY_F64,
            background=background,
            is_mitochondrial=is_mitochondrial,
        )

    marker_f = marker_image.astype(np.float64)

    # ── Vectorised per-cell metrics (single pass each) ────────────

    ones = np.ones(masks.shape, dtype=np.float64)
    areas = labeled_comprehension(ones, masks, labels, np.sum, np.float64, 0.0)

    integrated_densities = labeled_comprehension(
        marker_f, masks, labels, np.sum, np.float64, 0.0
    )
    mean_intensities = labeled_comprehension(
        marker_f, masks, labels, np.mean, np.float64, 0.0
    )
    std_intensities = labeled_comprehension(
        marker_f, masks, labels, np.std, np.float64, 0.0
    )
    min_intensities = labeled_comprehension(
        marker_f, masks, labels, np.min, np.float64, 0.0
    )
    max_intensities = labeled_comprehension(
        marker_f, masks, labels, np.max, np.float64, 0.0
    )

    # ── CTCF ──────────────────────────────────────────────────────

    if per_cell_backgrounds is not None:
        # Local: CTCF_i = IntDen_i − Area_i × bg_i
        ctcf_raw = integrated_densities - (areas * per_cell_backgrounds)
    else:
        ctcf_raw = integrated_densities - (areas * background)

    # Mitochondrial correction
    nuclear_signal = None
    if is_mitochondrial and nuclear_masks is not None:
        nuclear_signal = _calculate_nuclear_signal(
            marker_f, masks, nuclear_masks, labels
        )
        ctcf_raw = ctcf_raw - nuclear_signal

    ctcf = np.maximum(0.0, ctcf_raw)

    # ── MCF ───────────────────────────────────────────────────────

    mcf = np.where(areas > 0, ctcf / areas, 0.0)

    # ── Error propagation ─────────────────────────────────────────
    # σ_CTCF = √( A · σ²_I  +  A² · σ²_bg )
    # where σ_I is the per-cell intensity std

    ctcf_sigma = None
    if compute_error and background_std > 0:
        variance_intensity = areas * (std_intensities ** 2)
        variance_background = (areas ** 2) * (background_std ** 2)
        ctcf_sigma = np.sqrt(variance_intensity + variance_background)

    # ── Quality flags ─────────────────────────────────────────────

    if saturation_threshold is None:
        if marker_image.dtype == np.uint16:
            saturation_threshold = 65535.0 * 0.99
        elif marker_image.dtype == np.uint8:
            saturation_threshold = 255.0 * 0.99
        else:
            saturation_threshold = float(np.max(marker_image)) * 0.99

    is_saturated = max_intensities >= saturation_threshold

    if len(mean_intensities) > 10:
        dim_cutoff = float(np.percentile(mean_intensities, dim_threshold_percentile))
        is_dim = mean_intensities < dim_cutoff
    else:
        is_dim = np.zeros(len(labels), dtype=bool)

    return QuantificationResult(
        marker_name=marker_name,
        cell_ids=labels.astype(np.int32),
        areas=areas,
        integrated_densities=integrated_densities,
        mean_intensities=mean_intensities,
        ctcf=ctcf,
        background=background,
        per_cell_backgrounds=per_cell_backgrounds,
        background_std=background_std,
        ctcf_sigma=ctcf_sigma,
        mcf=mcf,
        min_intensities=min_intensities,
        max_intensities=max_intensities,
        std_intensities=std_intensities,
        is_saturated=is_saturated,
        is_dim=is_dim,
        nuclear_signal=nuclear_signal,
        is_mitochondrial=is_mitochondrial,
    )


# ── Nuclear signal helper ──────────────────────────────────────────


def _calculate_nuclear_signal(
    marker_image: np.ndarray,
    cell_masks: np.ndarray,
    nuclear_masks: np.ndarray,
    cell_labels: np.ndarray,
) -> np.ndarray:
    """Marker signal within nuclear region for each cell."""
    nuclear_signals = np.zeros(len(cell_labels), dtype=np.float64)
    for i, cell_id in enumerate(cell_labels):
        cell_region = cell_masks == cell_id
        nuclear_in_cell = cell_region & (nuclear_masks > 0)
        if np.any(nuclear_in_cell):
            nuclear_signals[i] = marker_image[nuclear_in_cell].sum()
    return nuclear_signals


# ── Multi-marker quantification ────────────────────────────────────


def quantify_multiple_markers(
    marker_images: Dict[str, np.ndarray],
    masks: np.ndarray,
    backgrounds: Dict[str, float],
    *,
    per_cell_backgrounds_map: Optional[Dict[str, np.ndarray]] = None,
    background_stds: Optional[Dict[str, float]] = None,
    nuclear_masks: Optional[np.ndarray] = None,
    mitochondrial_markers: Optional[List[str]] = None,
    parallel: bool = True,
    n_workers: int = 4,
) -> Dict[str, QuantificationResult]:
    """Quantify multiple markers, optionally in parallel.

    Args:
        marker_images: marker name → image array.
        masks: Label mask.
        backgrounds: marker name → global background scalar.
        per_cell_backgrounds_map: marker name → per-cell background array.
        background_stds: marker name → background std dev.
        nuclear_masks: For mitochondrial correction.
        mitochondrial_markers: Which markers are mitochondrial.
        parallel: Use threading.
        n_workers: Thread pool size.
    """
    if mitochondrial_markers is None:
        mitochondrial_markers = []
    if per_cell_backgrounds_map is None:
        per_cell_backgrounds_map = {}
    if background_stds is None:
        background_stds = {}

    def quantify_single(name: str) -> QuantificationResult:
        return calculate_ctcf_vectorized(
            marker_image=marker_images[name],
            masks=masks,
            background=backgrounds[name],
            marker_name=name,
            per_cell_backgrounds=per_cell_backgrounds_map.get(name),
            background_std=background_stds.get(name, 0.0),
            nuclear_masks=nuclear_masks,
            is_mitochondrial=name in mitochondrial_markers,
        )

    if parallel and len(marker_images) > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            results = list(ex.map(quantify_single, marker_images.keys()))
        return {r.marker_name: r for r in results}
    return {name: quantify_single(name) for name in marker_images}


# ── DataFrame output ───────────────────────────────────────────────


def results_to_dataframe(
    results: Dict[str, QuantificationResult],
    condition: str = "",
    image_set: str = "",
    segmentation_type: str = "cellular",
    *,
    morphology: Optional[CellMorphology] = None,
    pixel_size_um: Optional[float] = None,
) -> "pd.DataFrame":
    """Convert quantification results to a rich DataFrame.

    Columns produced:
    - Base: Condition, SegmentationType, ImageSet, CellID, Area
    - If *morphology*: x_centroid, y_centroid, circularity, solidity,
      eccentricity, perimeter, aspect_ratio, is_border
    - If *pixel_size_um*: area_um2
    - Per marker (from ``QuantificationResult.to_dict()``):
      _CTCF, _IntegratedDensity, _MeanIntensity, _Background,
      _MCF, _CTCF_sigma, _MinIntensity, _MaxIntensity, _StdIntensity
    - Quality flags: is_saturated, is_dim (from first marker)
    """
    import pandas as pd

    if not results:
        return pd.DataFrame()

    first = next(iter(results.values()))
    n = first.n_cells
    if n == 0:
        return pd.DataFrame()

    data: Dict[str, object] = {
        "Condition": [condition] * n,
        "SegmentationType": [segmentation_type] * n,
        "ImageSet": [image_set] * n,
        "CellID": first.cell_ids,
        "Area": first.areas,
    }

    # ── Morphology columns ────────────────────────────────────────
    if morphology is not None:
        # morphology may cover the *original* (pre-filter) cells; align
        # to the labels in the results by index mapping.
        morph_idx = {int(lb): i for i, lb in enumerate(morphology.labels)}
        result_labels = first.cell_ids
        idxs = np.array([morph_idx.get(int(lb), -1) for lb in result_labels])
        valid = idxs >= 0

        def _aligned(arr: np.ndarray, default: float = 0.0) -> np.ndarray:
            out = np.full(n, default, dtype=arr.dtype)
            out[valid] = arr[idxs[valid]]
            return out

        data["x_centroid"] = _aligned(morphology.centroids_x)
        data["y_centroid"] = _aligned(morphology.centroids_y)
        data["circularity"] = _aligned(morphology.circularities)
        data["solidity"] = _aligned(morphology.solidities)
        data["eccentricity"] = _aligned(morphology.eccentricities)
        data["perimeter"] = _aligned(morphology.perimeters)
        data["aspect_ratio"] = _aligned(morphology.aspect_ratios)
        data["is_border"] = _aligned(morphology.is_border, default=False)

    if pixel_size_um is not None:
        data["area_um2"] = first.areas * (pixel_size_um ** 2)

    # ── Per-marker columns ────────────────────────────────────────
    for result in results.values():
        data.update(result.to_dict())

    # ── Quality flags from first marker ───────────────────────────
    if first.is_saturated is not None:
        data["is_saturated"] = first.is_saturated
    if first.is_dim is not None:
        data["is_dim"] = first.is_dim

    return pd.DataFrame(data)
