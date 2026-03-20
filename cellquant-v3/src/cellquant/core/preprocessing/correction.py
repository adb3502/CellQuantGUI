"""Apply dark subtraction and flat-field correction to raw images."""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class PreprocessingConfig:
    """Calibration data for image correction.

    Both fields are optional — if only one is provided, only that
    correction step is applied.  If neither is provided,
    ``correct_image`` returns the input unchanged.
    """

    dark_master: Optional[np.ndarray] = None
    flat_norm: Optional[np.ndarray] = None
    clip_negative: bool = True


def correct_image(
    raw_image: np.ndarray,
    config: PreprocessingConfig,
) -> np.ndarray:
    """Apply dark subtraction and/or flat-field correction.

    Formula::

        I_corrected = (I_raw − Dark_master) / Flat_norm

    If only dark is provided, only subtraction is applied.
    If only flat is provided, only division is applied.
    If neither, the input is returned as float64 (no copy).

    Args:
        raw_image: Raw fluorescence image (any dtype).
        config: Preprocessing configuration with optional calibration data.

    Returns:
        Corrected image as float64.
    """
    if config.dark_master is None and config.flat_norm is None:
        return raw_image.astype(np.float64)

    result = raw_image.astype(np.float64)

    if config.dark_master is not None:
        result -= config.dark_master

    if config.flat_norm is not None:
        result /= config.flat_norm

    if config.clip_negative:
        np.maximum(result, 0.0, out=result)

    return result


def correct_image_batch(
    images: Dict[str, np.ndarray],
    config: PreprocessingConfig,
) -> Dict[str, np.ndarray]:
    """Apply correction to a dictionary of marker images.

    Args:
        images: Mapping of marker name → raw image array.
        config: Preprocessing configuration.

    Returns:
        Mapping of marker name → corrected image.
    """
    return {name: correct_image(img, config) for name, img in images.items()}
