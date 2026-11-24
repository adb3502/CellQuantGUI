"""Reading single frames of a recording for display.

Kept apart from the analysis loaders because display has different needs:
one frame at a time, fast, and with contrast that is FIXED across the whole
recording. Auto-contrasting each frame on its own stretches every frame to
the same brightness range, which hides the response the viewer exists to show.

Also renders the pseudocolour activity overlay (dF/F0 per cell or per pixel,
like SlideBook's ratio view) on top of those frames.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np


def frame_count(path: Union[str, Path]) -> int:
    import tifffile

    with tifffile.TiffFile(str(path)) as tif:
        shape = tif.series[0].shape
        return int(shape[0]) if len(shape) >= 3 else 1


def read_frame(path: Union[str, Path], index: int) -> np.ndarray:
    """Decode ONE frame. A 120-frame SlideBook file is 63 MB and cannot be
    memory-mapped, so reading the whole stack per frame would make scrubbing
    read gigabytes."""
    import tifffile

    with tifffile.TiffFile(str(path)) as tif:
        series = tif.series[0]
        n = int(series.shape[0]) if len(series.shape) >= 3 else 1
        if index < 0 or index >= n:
            raise IndexError(f"Frame {index} out of range (0-{n - 1})")
        if n == 1:
            return series.asarray()
        # One page per frame is the common layout; fall back to slicing the
        # series for files that pack frames differently.
        if len(tif.pages) == n:
            return tif.pages[index].asarray()
        return series.asarray()[index]


@lru_cache(maxsize=64)
def _limits_cached(path: str, mtime: float) -> Tuple[float, float]:
    n = frame_count(path)
    # Sample frames across the recording rather than decoding all of them.
    idx = sorted(set(np.linspace(0, n - 1, num=min(n, 12)).astype(int)))
    sample = np.stack([read_frame(path, int(i)) for i in idx]).astype(np.float64)
    lo, hi = np.percentile(sample, (0.35, 99.9))
    if hi <= lo:
        lo, hi = float(sample.min()), float(sample.max())
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def display_limits(path: Union[str, Path]) -> Tuple[float, float]:
    """One (low, high) pair for the WHOLE recording, so a cell that brightens
    is drawn brighter. Cached per file and invalidated if the file changes."""
    p = Path(path)
    return _limits_cached(str(p), p.stat().st_mtime)


def to_uint8(img: np.ndarray, lo: float, hi: float) -> np.ndarray:
    scaled = (np.asarray(img, np.float64) - lo) / (hi - lo)
    return (np.clip(scaled, 0.0, 1.0) * 255).astype(np.uint8)


def to_rgb(gray: np.ndarray) -> np.ndarray:
    g = np.asarray(gray, np.uint8)
    return np.stack([g, g, g], axis=-1)


def draw_outlines(rgb: np.ndarray, masks: np.ndarray,
                  colour: Tuple[int, int, int] = (255, 40, 40)) -> np.ndarray:
    """Draw ROI boundaries onto an RGB image WITHOUT touching its contrast.

    The segmentation page's outline helper re-stretches the image to its own
    min/max first, which silently undid the fixed contrast of every video
    frame shown with ROIs. This only colours boundary pixels.
    """
    masks = np.asarray(masks)
    out = np.array(rgb, dtype=np.uint8, copy=True)
    if masks.max() == 0:
        return out
    p = np.pad(masks, 1, mode="constant", constant_values=0)
    centre = p[1:-1, 1:-1]
    edge = (centre > 0) & (
        (p[:-2, 1:-1] != centre) | (p[2:, 1:-1] != centre)
        | (p[1:-1, :-2] != centre) | (p[1:-1, 2:] != centre)
    )
    out[edge] = colour
    return out


# ── Colour map ────────────────────────────────────────────────────────────

def turbo(x: np.ndarray) -> np.ndarray:
    """Google's Turbo colour map (polynomial fit), x in [0, 1] -> uint8 RGB.

    A rainbow map without jet's false bands; low values blue, high red. The
    frontend uses the same map so a colour means the same dF/F0 in the
    overlay and in the heatmap chart.
    """
    x = np.clip(np.asarray(x, np.float64), 0.0, 1.0)
    r = 0.13572138 + x * (4.61539260 + x * (-42.66032258 + x * (132.13108234 + x * (-152.94239396 + x * 59.28637943))))
    g = 0.09140261 + x * (2.19418839 + x * (4.84296658 + x * (-14.18503333 + x * (4.27729857 + x * 2.82956604))))
    b = 0.10667330 + x * (12.64194608 + x * (-60.58204836 + x * (110.36276771 + x * (-89.90310912 + x * 27.34824973))))
    rgb = np.stack([r, g, b], axis=-1)
    return (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)


# Turbo's bottom end (x < ~0.1) is near-black purple. Blended over a grey
# cell that reads as "no cell", so the overlay starts here: the lowest value
# shows as clear blue. The frontend colour bar samples the same range.
TURBO_START = 0.1


def blend_colour(rgb: np.ndarray, values: np.ndarray, lo: float, hi: float,
                 alpha: float = 0.6) -> np.ndarray:
    """Tint pixels by value. NaN pixels are left untouched (stay grey)."""
    out = np.array(rgb, dtype=np.float64, copy=True)
    values = np.asarray(values, np.float64)
    ok = np.isfinite(values)
    if not ok.any():
        return out.astype(np.uint8)
    norm = (values[ok] - lo) / (hi - lo if hi > lo else 1.0)
    x = TURBO_START + (1.0 - TURBO_START) * np.clip(norm, 0.0, 1.0)
    out[ok] = (1.0 - alpha) * out[ok] + alpha * turbo(x).astype(np.float64)
    return np.clip(out, 0, 255).astype(np.uint8)


def _robust_limits(values: np.ndarray) -> Tuple[float, float]:
    """2nd-98th percentile, always including 0 so 'no change' has a colour.
    A few very bright cells must not compress everyone else into one hue."""
    v = np.asarray(values, np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0, 1.0
    lo = min(0.0, float(np.percentile(v, 2)))
    hi = float(np.percentile(v, 98))
    if hi <= lo:
        hi = lo + 1e-6
    return lo, hi


# ── Per-cell activity (the analysed traces) ───────────────────────────────

_cell_cache: dict = {}


def cell_activity_matrix(traces_df, condition: str, image_set: str):
    """(label -> row index lookup, matrix[cell_row, frame]) of dF/F0.

    Built from the run's own traces table, so the overlay shows exactly the
    numbers in the results - not a re-computation that could disagree.
    """
    key = (id(traces_df), len(traces_df), condition, image_set)
    hit = _cell_cache.get(key)
    if hit is not None:
        return hit
    sub = traces_df[(traces_df["Condition"] == condition)
                    & (traces_df["ImageSet"] == image_set)]
    if sub.empty:
        raise KeyError(f"No traces for {condition}/{image_set}")
    wide = sub.pivot_table(index="CellID", columns="Frame", values="dF_F0")
    ids = np.asarray(wide.index, dtype=np.int64)
    matrix = wide.to_numpy(dtype=np.float64)
    if len(_cell_cache) > 32:
        _cell_cache.clear()
    _cell_cache[key] = (ids, matrix)
    return ids, matrix


def cell_values_for_frame(traces_df, condition: str, image_set: str,
                          frame: int, max_label: int) -> np.ndarray:
    """Array indexed by mask label giving that cell's dF/F0 at `frame`."""
    ids, matrix = cell_activity_matrix(traces_df, condition, image_set)
    lut = np.full(max(int(max_label), int(ids.max(initial=0))) + 1, np.nan)
    if 0 <= frame < matrix.shape[1]:
        lut[ids] = matrix[:, frame]
    lut[0] = np.nan  # background is never coloured
    return lut


def cell_activity_limits(traces_df, condition: str, image_set: str) -> Tuple[float, float]:
    _, matrix = cell_activity_matrix(traces_df, condition, image_set)
    return _robust_limits(matrix)


# ── Per-pixel activity ────────────────────────────────────────────────────

def baseline_frame_count(n_frames: int, params: Optional[dict]) -> int:
    """How many leading frames form F0, mirroring the analysis.

    'fraction' and 'stim_frame' match temporal._baseline_window exactly.
    'auto' picks an onset per cell, which has no per-pixel equivalent; its
    leading reference window (5% of frames, at least 3) always precedes any
    detected onset, so that window is used.
    """
    params = params or {}
    mode = params.get("baseline_mode", "fraction")
    if mode == "stim_frame" and params.get("stim_frame") is not None:
        return max(1, min(int(params["stim_frame"]), n_frames))
    if mode == "auto":
        return max(1, min(n_frames, max(3, int(round(n_frames * 0.05)))))
    frac = float(params.get("baseline_fraction", 0.10))
    return max(1, int(round(n_frames * frac)))


def _frame_background(frame: np.ndarray, masks: np.ndarray) -> float:
    # Same estimator as the analysis: median of pixels outside every ROI.
    bg = frame[masks == 0]
    return float(np.median(bg)) if bg.size else 0.0


# Single pixels carry the full shot noise of one photon count, and their
# ratio to an equally noisy baseline amplifies it into speckle. A 1-pixel
# Gaussian averages ~9 neighbours: noise drops ~3x while cell-scale structure
# (tens of pixels) is untouched. Numerator and baseline are smoothed alike,
# so a uniform response keeps its exact value.
_SMOOTH_SIGMA = 1.0

# A pixel whose baseline is below this fraction of its own cell's median
# baseline sits near background - typically the dim rim of a cell. Dividing
# by it turns noise into extreme colours, so it is left uncoloured.
_DIM_FRACTION = 0.25


def _smooth(img: np.ndarray) -> np.ndarray:
    from scipy.ndimage import gaussian_filter

    return gaussian_filter(img, sigma=_SMOOTH_SIGMA)


def _dim_floor(f0: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Per-pixel floor: _DIM_FRACTION of the median baseline of its cell."""
    from scipy import ndimage

    labels = np.unique(masks)
    labels = labels[labels > 0]
    floor = np.full(int(masks.max()) + 1, np.inf)
    if labels.size:
        med = ndimage.median(f0, labels=masks, index=labels)
        floor[labels] = _DIM_FRACTION * np.asarray(med, dtype=np.float64)
    out = floor[np.asarray(masks, dtype=np.int64)]
    # Never accept a non-positive baseline, whatever the cell median.
    return np.maximum(out, np.finfo(np.float64).tiny)


@lru_cache(maxsize=16)
def _pixel_f0_cached(path: str, mtime: float, n_baseline: int,
                     masks_key: Tuple[int, int, int, int], masks_bytes: bytes,
                     shape: Tuple[int, int]) -> np.ndarray:
    masks = np.frombuffer(masks_bytes, dtype=np.int32).reshape(shape)
    corrected = []
    for i in range(n_baseline):
        f = read_frame(path, i).astype(np.float64)
        corrected.append(_smooth(f - _frame_background(f, masks)))
    return np.median(np.stack(corrected), axis=0)


def _pixel_f0(path, masks: np.ndarray, n_baseline: int) -> np.ndarray:
    p = Path(path)
    m = np.ascontiguousarray(masks, dtype=np.int32)
    key = (m.shape[0], m.shape[1], int(m.max()), int(m.sum()))
    return _pixel_f0_cached(str(p), p.stat().st_mtime, int(n_baseline),
                            key, m.tobytes(), m.shape)


def pixel_dff_frame(path, frame: int, masks: np.ndarray, n_baseline: int) -> np.ndarray:
    """Per-pixel dF/F0 inside the ROIs; NaN everywhere else.

    Same formula as the per-cell traces - background-subtracted, baseline is
    the median of the baseline window - applied to each pixel. Outside the
    ROIs F0 is background noise near zero, and dividing by it would paint
    the whole field with meaningless extreme colours.
    """
    masks = np.asarray(masks)
    f = read_frame(path, frame).astype(np.float64)
    corrected = _smooth(f - _frame_background(f, masks))
    f0 = _pixel_f0(path, masks, n_baseline)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = (corrected - f0) / f0
    d[(masks == 0) | (f0 < _dim_floor(f0, masks))] = np.nan
    return d


def pixel_activity_limits(path, masks: np.ndarray, n_baseline: int) -> Tuple[float, float]:
    n = frame_count(path)
    idx = sorted(set(np.linspace(0, n - 1, num=min(n, 12)).astype(int)))
    vals = np.concatenate([pixel_dff_frame(path, int(i), masks, n_baseline).ravel()
                           for i in idx])
    return _robust_limits(vals)
