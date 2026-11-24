"""Reading single frames of a recording for display.

Kept apart from the analysis loaders because display has different needs:
one frame at a time, fast, and with contrast that is FIXED across the whole
recording. Auto-contrasting each frame on its own stretches every frame to
the same brightness range, which hides the response the viewer exists to show.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Tuple, Union

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
