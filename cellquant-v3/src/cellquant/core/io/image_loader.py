"""
Image loading and normalization utilities.

Optimized for high-throughput microscopy image processing with:
- Multi-dimensional TIFF support (Z-stacks, time series, channels)
- Memory mapping for large files (>1GB)
- Robust percentile-based normalization
"""

from pathlib import Path
from typing import Union, Tuple, Optional, List
import numpy as np
from dataclasses import dataclass

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False

from skimage import io
from skimage.exposure import rescale_intensity


@dataclass
class ImageMetadata:
    """Metadata about a loaded image."""
    path: Path
    shape: Tuple[int, ...]
    dtype: np.dtype
    n_channels: int
    is_memmap: bool
    size_bytes: int


def load_image(
    image_path: Union[str, Path],
    channel: Optional[int] = None
) -> np.ndarray:
    """
    Load an image with robust multi-dimensional handling.

    Automatically extracts 2D data from multi-dimensional arrays (Z-stacks,
    time series, multi-channel). For explicit channel selection, use the
    channel parameter.

    Args:
        image_path: Path to image file (TIFF, PNG, etc.)
        channel: Optional channel index to extract. If None, auto-selects.

    Returns:
        2D numpy array as uint16 or float32

    Example:
        >>> img = load_image("sample_C0.tif")
        >>> img.shape
        (512, 512)
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        img = io.imread(str(image_path))
    except Exception as e:
        raise IOError(f"Failed to load image {image_path}: {e}")

    # Handle multi-dimensional arrays
    if img.ndim > 2:
        if channel is not None:
            # Explicit channel selection
            if img.shape[0] <= 5:  # Channels in first dimension (C, Y, X)
                img = img[channel, :, :]
            elif img.shape[-1] <= 5:  # Channels in last dimension (Y, X, C)
                img = img[:, :, channel]
            else:
                raise ValueError(f"Cannot extract channel {channel} from shape {img.shape}")
        else:
            # Auto-select first channel/slice
            if img.shape[0] > 0 and img.shape[0] <= 5:
                img = img[0, :, :]
            elif img.shape[-1] > 0 and img.shape[-1] <= 5:
                img = img[:, :, 0]
            elif img.ndim == 3:
                # Could be (Z, Y, X) - take middle slice
                img = img[img.shape[0] // 2, :, :]

    # Preserve uint16 for quantification, otherwise use float32
    if img.dtype == np.uint16:
        return img.astype(np.uint16)
    else:
        return img.astype(np.float32)


def load_image_memmap(
    image_path: Union[str, Path],
    mode: str = 'r'
) -> Tuple[np.ndarray, ImageMetadata]:
    """
    Load large image as memory-mapped array for memory efficiency.

    For images >1GB, this avoids loading the entire file into RAM.
    The array is read on-demand as regions are accessed.

    Args:
        image_path: Path to TIFF file
        mode: Memory map mode ('r' for read-only, 'r+' for read-write)

    Returns:
        Tuple of (memory-mapped array, metadata)

    Raises:
        ImportError: If tifffile is not installed

    Example:
        >>> img, meta = load_image_memmap("large_stack.tif")
        >>> print(f"Shape: {meta.shape}, Size: {meta.size_bytes / 1e9:.2f} GB")
    """
    if not HAS_TIFFFILE:
        raise ImportError("tifffile required for memory mapping. Install with: pip install tifffile")

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Memory-map the TIFF
    img = tifffile.memmap(str(image_path), mode=mode)

    metadata = ImageMetadata(
        path=image_path,
        shape=img.shape,
        dtype=img.dtype,
        n_channels=img.shape[0] if img.ndim > 2 and img.shape[0] <= 5 else 1,
        is_memmap=True,
        size_bytes=image_path.stat().st_size
    )

    return img, metadata


def load_multichannel_tiff(
    image_path: Union[str, Path]
) -> Tuple[np.ndarray, int]:
    """
    Load a multi-channel TIFF and return all channels.

    Args:
        image_path: Path to multi-channel TIFF

    Returns:
        Tuple of (array with shape (C, Y, X), number of channels)
    """
    image_path = Path(image_path)

    if HAS_TIFFFILE:
        img = tifffile.imread(str(image_path))
    else:
        img = io.imread(str(image_path))

    # Ensure channels are in first dimension
    if img.ndim == 2:
        img = img[np.newaxis, :, :]
        n_channels = 1
    elif img.ndim == 3:
        if img.shape[-1] <= 5:  # (Y, X, C) -> (C, Y, X)
            img = np.moveaxis(img, -1, 0)
        n_channels = img.shape[0]
    else:
        raise ValueError(f"Unexpected image dimensions: {img.shape}")

    return img, n_channels


def normalize_image(
    img: np.ndarray,
    perc_low: float = 1.0,
    perc_high: float = 99.0,
    clip: bool = True
) -> np.ndarray:
    """
    Percentile-based image normalization with robust edge case handling.

    Normalizes image to [0, 1] range using percentile values to handle
    outliers and varying bit depths gracefully.

    Args:
        img: Input image array
        perc_low: Lower percentile for normalization (default 1.0)
        perc_high: Upper percentile for normalization (default 99.0)
        clip: Whether to clip values outside [0, 1] (default True)

    Returns:
        Normalized image as float32 in range [0, 1]

    Example:
        >>> raw = load_image("sample.tif")  # uint16 [0, 65535]
        >>> norm = normalize_image(raw)
        >>> norm.min(), norm.max()
        (0.0, 1.0)
    """
    img_float = img.astype(np.float32)

    # Handle edge case: uniform image
    min_val, max_val = np.min(img_float), np.max(img_float)
    if min_val == max_val:
        return np.zeros_like(img_float) if min_val == 0 else np.ones_like(img_float) * 0.5

    # Calculate percentiles
    p_low, p_high = np.percentile(img_float, (perc_low, perc_high))

    # Handle edge case: percentiles are equal
    if p_high <= p_low:
        p_low = min_val
        p_high = max_val

    # Final safeguard
    if p_high <= p_low:
        return (img_float - p_low) / max(1e-8, p_high - p_low)

    # Use scikit-image's rescale_intensity for robust normalization
    normalized = rescale_intensity(
        img_float,
        in_range=(p_low, p_high),
        out_range=(0.0, 1.0)
    )

    if clip:
        normalized = np.clip(normalized, 0.0, 1.0)

    return normalized.astype(np.float32)


def create_composite(
    channels: List[np.ndarray],
    colors: Optional[List[Tuple[float, float, float]]] = None,
    normalize_each: bool = True
) -> np.ndarray:
    """
    Create RGB composite from multiple channels.

    Args:
        channels: List of 2D arrays (grayscale channels)
        colors: List of RGB tuples for each channel. Defaults to cyan, green, magenta.
        normalize_each: Whether to normalize each channel individually

    Returns:
        RGB composite as uint8 array (Y, X, 3)
    """
    if not channels:
        raise ValueError("At least one channel required")

    # Default fluorescent colors
    default_colors = [
        (0.0, 1.0, 1.0),    # Cyan (DAPI, nuclear)
        (0.0, 1.0, 0.0),    # Green (cytoplasm)
        (1.0, 0.0, 0.6),    # Magenta (marker)
        (1.0, 1.0, 0.0),    # Yellow (marker 2)
        (1.0, 0.4, 0.0),    # Orange (marker 3)
    ]

    if colors is None:
        colors = default_colors[:len(channels)]

    # Initialize composite
    shape = channels[0].shape
    composite = np.zeros((*shape, 3), dtype=np.float32)

    for ch, color in zip(channels, colors):
        if normalize_each:
            ch_norm = normalize_image(ch)
        else:
            ch_norm = ch.astype(np.float32)
            if ch_norm.max() > 1.0:
                ch_norm = ch_norm / ch_norm.max()

        for i, c in enumerate(color):
            composite[:, :, i] += ch_norm * c

    # Clip and convert to uint8
    composite = np.clip(composite, 0, 1)
    return (composite * 255).astype(np.uint8)


def find_images_by_suffix(
    folder: Union[str, Path],
    suffixes: List[str],
    extensions: List[str] = ['.tif', '.tiff', '.png']
) -> dict:
    """
    Find and group images by channel suffix.

    Scans a folder for images matching the given suffixes and groups them
    by base name for multi-channel analysis.

    Args:
        folder: Path to folder containing images
        suffixes: List of channel suffixes (e.g., ['C0', 'C1', 'C2'])
        extensions: Allowed file extensions

    Returns:
        Dictionary mapping base_name -> {suffix: path}

    Example:
        >>> images = find_images_by_suffix("data/", ["C0", "C1"])
        >>> images
        {'sample1': {'C0': Path('sample1_C0.tif'), 'C1': Path('sample1_C1.tif')}}
    """
    import re

    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder}")

    # Build regex pattern for suffix extraction
    # Handles patterns like: sample_C0.tif, sample_w1_C0.tif, Sample_XY123_C0.tiff
    suffix_pattern = re.compile(
        r'^(.+?)_(' + '|'.join(re.escape(s) for s in suffixes) + r')(?:_.*)?\.(?:tif|tiff|png)$',
        re.IGNORECASE
    )

    grouped = {}

    for file in folder.iterdir():
        if file.suffix.lower() not in extensions:
            continue

        match = suffix_pattern.match(file.name)
        if match:
            base_name = match.group(1)
            suffix = match.group(2)

            if base_name not in grouped:
                grouped[base_name] = {}
            grouped[base_name][suffix.upper()] = file

    return grouped
