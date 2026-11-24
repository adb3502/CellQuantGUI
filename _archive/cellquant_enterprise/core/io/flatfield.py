"""
Flat-field and dark current correction for fluorescence microscopy.

Corrects two major sources of systematic bias before CTCF measurement:
  - Dark current / electronic offset: additive, subtracted first
  - Illumination non-uniformity: multiplicative, divided out via flat-field

Formula:  I_corrected = (I_raw - dark) / flat_norm
          where flat_norm = (flat_avg - dark) / mean(flat_avg - dark)

Skipping flat-field correction introduces 10–40% systematic bias in CTCF
across the field of view (Peng et al., Nature Communications 8:14836, 2017).
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
from skimage import io as skio


class FlatFieldCorrector:
    """
    Per-channel flat-field and dark current corrector.

    Typical usage with dedicated calibration images:
        corrector = FlatFieldCorrector()
        corrector.load_dark_frames(dark_paths)          # acquired with light blocked
        corrector.load_flat_frames(flat_paths, "C0")    # one set per channel
        corrector.load_flat_frames(flat_paths, "C1")
        corrected = corrector.correct(raw_image, channel="C0")

    Usage without a dark frame (flat only):
        corrector = FlatFieldCorrector()
        corrector.load_flat_frames(flat_paths, "C0")
        corrected = corrector.correct(raw_image, channel="C0")

    Usage with a single pre-averaged flat image:
        corrector = FlatFieldCorrector()
        corrector.load_single_flat("flat_C0.tif", "C0")
        corrected = corrector.correct(raw_image, channel="C0")
    """

    def __init__(self):
        self._dark: Optional[np.ndarray] = None
        self._flats: Dict[str, np.ndarray] = {}  # channel_upper -> normalized flat

    # ------------------------------------------------------------------ #
    # Loading calibration frames                                           #
    # ------------------------------------------------------------------ #

    def load_dark_frames(self, paths: List[Union[str, Path]]):
        """
        Average N dark frames into a master dark.

        Acquire with light blocked, identical exposure/gain to science frames.
        N >= 20 recommended; N = 100 reduces master dark noise to 10% of
        single-frame noise (σ_master = σ_single / √N).

        Args:
            paths: Paths to dark frame TIFFs
        """
        if not paths:
            raise ValueError("No dark frames provided")
        frames = [_load_raw(p) for p in paths]
        self._dark = np.mean(np.stack(frames, axis=0), axis=0)

    def load_flat_frames(self, paths: List[Union[str, Path]], channel: str):
        """
        Median-project and normalize flat-field frames for one channel.

        Acquire on a uniformly fluorescent sample (concentrated fluorescein,
        or a commercial fluorescent slide). Median projection rejects dust/debris.
        Must be acquired separately for each filter set / channel.

        Args:
            paths: Paths to flat-field TIFFs (N >= 9 positions recommended)
            channel: Channel suffix matching ChannelConfig (e.g. "C0", "C1")
        """
        if not paths:
            raise ValueError(f"No flat frames provided for channel {channel}")
        frames = [_load_raw(p) for p in paths]
        flat_avg = np.median(np.stack(frames, axis=0), axis=0)
        self._store_flat(flat_avg, channel)

    def load_single_flat(self, path: Union[str, Path], channel: str):
        """Load a single pre-averaged flat-field image."""
        self.load_flat_frames([path], channel)

    def set_dark(self, dark_array: np.ndarray):
        """Set dark frame directly from a numpy array."""
        self._dark = dark_array.astype(np.float64)

    def set_flat(self, flat_array: np.ndarray, channel: str):
        """Set flat-field frame directly from a numpy array."""
        self._store_flat(flat_array.astype(np.float64), channel)

    # ------------------------------------------------------------------ #
    # Correction                                                           #
    # ------------------------------------------------------------------ #

    def correct(
        self,
        image: np.ndarray,
        channel: Optional[str] = None
    ) -> np.ndarray:
        """
        Apply flat-field correction to a single image.

        Applies: I_corrected = (I_raw - dark) / flat_norm
        Result is clipped to [0, ∞) and returned as float32.

        Args:
            image: Raw 2D image array (any numeric dtype)
            channel: Channel suffix (e.g. "C0"). If None and only one flat
                     is loaded, that flat is used. If None and no flat is
                     loaded, only dark subtraction is applied.

        Returns:
            Corrected float32 image, same spatial dimensions as input
        """
        img = image.astype(np.float64)

        if self._dark is not None:
            img = img - self._dark

        flat = self._get_flat(channel)
        if flat is not None:
            img = img / np.maximum(flat, 1e-6)

        return np.maximum(img, 0).astype(np.float32)

    def correct_dict(
        self,
        images: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Correct a dict of {channel_suffix: image} in one call.

        Channels without a loaded flat-field receive dark subtraction only.

        Args:
            images: Mapping of channel suffix -> raw image

        Returns:
            Mapping of channel suffix -> corrected image
        """
        return {ch: self.correct(img, channel=ch) for ch, img in images.items()}

    # ------------------------------------------------------------------ #
    # Introspection / diagnostics                                          #
    # ------------------------------------------------------------------ #

    @property
    def has_dark(self) -> bool:
        return self._dark is not None

    @property
    def loaded_channels(self) -> List[str]:
        return list(self._flats.keys())

    @property
    def is_active(self) -> bool:
        """True if any correction will be applied to images."""
        return self._dark is not None or bool(self._flats)

    def illumination_uniformity(self, channel: Optional[str] = None) -> Optional[float]:
        """
        Return the peak-to-mean ratio of the flat-field for a channel.

        Values significantly above 1.0 indicate severe illumination non-uniformity.
        A ratio of 1.4 means edge pixels receive 40% less light than the centre.

        Returns None if no flat is loaded for that channel.
        """
        flat = self._get_flat(channel)
        if flat is None:
            return None
        return float(flat.max() / flat.mean())

    def summary(self) -> str:
        parts = []
        if self._dark is not None:
            parts.append(f"dark loaded (shape={self._dark.shape})")
        else:
            parts.append("no dark frame")
        if self._flats:
            parts.append(f"flats for channels: {list(self._flats.keys())}")
        else:
            parts.append("no flat-field images")
        return "; ".join(parts)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _store_flat(self, flat_avg: np.ndarray, channel: str):
        """Dark-subtract (if available) and mean-normalize a flat, then store it."""
        flat_f = flat_avg.astype(np.float64)
        if self._dark is not None:
            flat_f = flat_f - self._dark
        flat_f = np.maximum(flat_f, 0)

        norm = np.mean(flat_f)
        if norm <= 0:
            raise ValueError(
                f"Flat-field mean is <= 0 for channel {channel}. "
                "Check that flat images are not blank."
            )
        self._flats[channel.upper()] = flat_f / norm

    def _get_flat(self, channel: Optional[str]) -> Optional[np.ndarray]:
        if not self._flats:
            return None
        if channel is None:
            return next(iter(self._flats.values()))
        return self._flats.get(channel.upper())


# ------------------------------------------------------------------ #
# Standalone utilities                                                #
# ------------------------------------------------------------------ #

def estimate_shading_from_images(images: List[np.ndarray]) -> np.ndarray:
    """
    Estimate a flat-field correction map directly from science images.

    Uses the BaSiC-inspired approach: median-project intensity-normalized
    images. Assumes cells are randomly distributed across FOVs so the
    median projection approximates the smooth illumination envelope.

    Requires at least 8 images; 20+ gives more stable estimates.

    Args:
        images: List of raw 2D fluorescence images (same channel and settings)

    Returns:
        Normalized shading map (float64, mean=1). Divide raw images by this
        map to correct for illumination non-uniformity.

    Raises:
        ValueError: If fewer than 8 images are provided
    """
    if len(images) < 8:
        raise ValueError(
            f"At least 8 images needed for shading estimation, got {len(images)}. "
            "Consider using a dedicated flat-field slide instead."
        )

    normalized = []
    for img in images:
        img_f = img.astype(np.float64)
        m = np.mean(img_f)
        if m > 0:
            normalized.append(img_f / m)

    stack = np.stack(normalized, axis=0)
    shading = np.median(stack, axis=0)

    mean_val = np.mean(shading)
    return shading / mean_val if mean_val > 0 else shading


def _load_raw(path: Union[str, Path]) -> np.ndarray:
    """Load image preserving full dynamic range as float64."""
    img = skio.imread(str(path))
    if img.ndim == 3:
        img = img[0] if img.shape[0] <= 5 else img[img.shape[0] // 2]
    return img.astype(np.float64)
