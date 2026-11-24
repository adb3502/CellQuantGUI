"""SA-beta-gal brightfield colorimetric quantification.

Deliberately does NOT use calculate_ctcf_vectorized: that function clamps at
zero (ctcf.py:185), which would destroy absorptive signal, since a chromogen
is darker than its background rather than brighter.

NOTE: the default stain vectors are the scikit-image HED values, fitted for
haematoxylin-DAB rather than X-gal blue. They are configurable and MUST be
calibrated against real controls before results are reported. See spec 9.2.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

_EPS = 1e-6


@dataclass
class SABetaGalConfig:
    white_percentile: float = 95.0
    # HED channel used as the chromogen proxy. Index 0 is haematoxylin, itself
    # a blue-purple stain, so it is the closest stock proxy for X-gal blue:
    # measured response to a blue precipitate is 0.0053 -> 0.1806 on this
    # channel, while DAB (index 2) drops to zero. Still a proxy, not a
    # calibrated X-gal vector - see spec 9.2.
    chromogen_index: int = 0
    threshold: Optional[float] = None  # None -> Otsu on the per-cell OD values
    stain_vectors: Optional[List[List[float]]] = field(default=None)
    min_area: int = 10


def estimate_white_reference(rgb: np.ndarray, percentile: float = 95.0) -> np.ndarray:
    """Per-channel illumination level of the unstained background."""
    rgb = np.asarray(rgb, dtype=np.float64)
    return np.percentile(rgb.reshape(-1, rgb.shape[-1]), percentile, axis=0)


def compute_optical_density(rgb: np.ndarray, white: np.ndarray) -> np.ndarray:
    """OD = -log10(I / I_white), guarded against log10(0) and averaged over channels."""
    rgb = np.asarray(rgb, dtype=np.float64)
    white = np.asarray(white, dtype=np.float64)
    white = np.where(white <= 0, 1.0, white)
    ratio = np.clip(rgb / white, _EPS, 1.0)
    return (-np.log10(ratio)).mean(axis=-1)


def deconvolve_chromogen(rgb: np.ndarray, config: SABetaGalConfig) -> np.ndarray:
    """Separate the chromogen contribution via colour deconvolution."""
    from skimage.color import rgb2hed

    arr = np.asarray(rgb, dtype=np.float64)
    if arr.max() > 1.0:
        arr = arr / 255.0
    arr = np.clip(arr, _EPS, 1.0)

    if config.stain_vectors is not None:
        from skimage.color import separate_stains
        matrix = np.linalg.inv(np.asarray(config.stain_vectors, dtype=np.float64))
        separated = separate_stains(arr, matrix)
    else:
        separated = rgb2hed(arr)

    idx = int(config.chromogen_index)
    channel = separated[:, :, idx]
    return np.nan_to_num(channel, nan=0.0, posinf=0.0, neginf=0.0)


def otsu_threshold(values: np.ndarray) -> float:
    """Otsu cut on a 1-D set of per-cell values.

    WARNING: Otsu assumes the values are bimodal. A real plate with many
    exactly-unstained cells plus a few intensely stained ones violates
    that assumption, and Otsu can maximise between-class variance INSIDE
    the stained group - splitting positives from positives and
    under-reporting percent-positive. Set config.threshold explicitly
    from a negative control for any result you intend to report.
    See spec 9.2 and test_otsu_mis_splits_a_skewed_od_distribution.
    """
    from skimage.filters import threshold_otsu

    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 4:
        return float("nan")
    if np.allclose(values, values[0]):
        return float("nan")

    t = float(threshold_otsu(values))

    # Otsu always returns a cut, even for unimodal data: on an all-negative
    # control it splits noise in half and reports ~50% positive. Require real
    # separation between the two groups before trusting it.
    lo, hi = values[values <= t], values[values > t]
    if lo.size == 0 or hi.size == 0:
        return float("nan")
    spread = float(values.max() - values.min())
    separation = float(hi.mean() - lo.mean())
    # Measured: unimodal noise gives a separation/spread ratio near 0.35,
    # a genuinely bimodal distribution near 1.0. 0.6 sits between them.
    if spread <= 0 or separation < 0.6 * spread:
        # Not convincingly bimodal - refuse rather than fabricate positives.
        return float("nan")
    return t


def quantify_sabetagal(rgb: np.ndarray, masks: np.ndarray,
                       config: SABetaGalConfig,
                       condition: str = "", image_set: str = "") -> pd.DataFrame:
    """Per-cell optical density and senescence-positive classification."""
    from scipy import ndimage

    rgb = np.asarray(rgb)
    masks = np.asarray(masks)

    white = estimate_white_reference(rgb, config.white_percentile)
    # Colour deconvolution isolates the chromogen; total-darkness OD would
    # also count dark nuclei, debris and out-of-focus material as stain.
    # Fall back to total OD only if deconvolution is unavailable.
    try:
        od = deconvolve_chromogen(rgb, config)
    except Exception as exc:
        print(f"Colour deconvolution failed ({exc}); using total optical density")
        od = compute_optical_density(rgb, white)

    labels = np.unique(masks)
    labels = labels[labels > 0]
    if labels.size == 0:
        return pd.DataFrame(columns=["Condition", "ImageSet", "CellID", "Area",
                                     "mean_OD", "integrated_OD", "is_positive"])

    mean_od = np.atleast_1d(ndimage.mean(od, labels=masks, index=labels))
    areas = np.atleast_1d(ndimage.sum(np.ones_like(od), labels=masks, index=labels))

    if config.threshold is not None:
        threshold = float(config.threshold)
        threshold_source = "user"
    else:
        threshold = otsu_threshold(mean_od)
        threshold_source = "otsu"

    if not np.isfinite(threshold):
        # No defensible cut: classify nothing rather than invent positives.
        # The caller sees threshold_source == "undecidable".
        is_positive = np.zeros(len(labels), dtype=bool)
        threshold_source = "undecidable"
    else:
        is_positive = mean_od > threshold

    df = pd.DataFrame({
        "Condition": condition,
        "ImageSet": image_set,
        "CellID": labels.astype(int),
        "Area": areas.astype(float),
        "mean_OD": mean_od.astype(float),
        "integrated_OD": (mean_od * areas).astype(float),
        "is_positive": is_positive,
        "threshold_used": threshold,
        "threshold_source": threshold_source,
    })
    df.attrs["threshold"] = float(threshold)
    return df[df["Area"] >= config.min_area].reset_index(drop=True)


def summarize_positivity(df: pd.DataFrame) -> pd.DataFrame:
    """Percent-positive per condition - the standard senescence readout."""
    if df.empty:
        return pd.DataFrame(columns=["Condition", "n_cells", "n_positive",
                                     "percent_positive"])
    grouped = df.groupby("Condition")["is_positive"].agg(["count", "sum"])
    grouped.columns = ["n_cells", "n_positive"]
    grouped["percent_positive"] = 100.0 * grouped["n_positive"] / grouped["n_cells"]
    return grouped.reset_index()
