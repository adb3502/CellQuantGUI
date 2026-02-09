"""
Mask I/O utilities for loading and saving segmentation masks.

Supports multiple formats:
- TIFF (16-bit label images)
- NumPy arrays (.npy, .npz)
- Zarr arrays (for large datasets)
"""

from pathlib import Path
from typing import Union, Tuple, Optional, Dict, Any
import numpy as np
from dataclasses import dataclass, field

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False

from skimage import io


@dataclass
class MaskMetadata:
    """Metadata about a loaded mask."""
    path: Optional[Path]
    shape: Tuple[int, ...]
    dtype: np.dtype
    n_cells: int
    unique_labels: list
    min_area: float
    max_area: float
    mean_area: float
    issues: list = field(default_factory=list)
    is_valid: bool = True


def load_mask(
    path: Union[str, Path],
    validate: bool = True
) -> Tuple[np.ndarray, MaskMetadata]:
    """
    Load segmentation mask from file with optional validation.

    Supports TIFF, NumPy, and Zarr formats. Returns mask as int32 label array
    where 0 = background and positive integers = cell IDs.

    Args:
        path: Path to mask file
        validate: Whether to validate mask integrity

    Returns:
        Tuple of (mask array, metadata)

    Example:
        >>> mask, meta = load_mask("segmentation_masks.tif")
        >>> print(f"Found {meta.n_cells} cells")
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Mask file not found: {path}")

    # Load based on extension
    suffix = path.suffix.lower()

    if suffix in ['.npy']:
        mask = np.load(path)
    elif suffix in ['.npz']:
        data = np.load(path)
        # Try common key names
        for key in ['masks', 'mask', 'labels', 'arr_0']:
            if key in data:
                mask = data[key]
                break
        else:
            raise KeyError(f"Could not find mask array in {path}. Keys: {list(data.keys())}")
    elif suffix in ['.zarr']:
        try:
            import zarr
            mask = zarr.open(str(path), mode='r')[:]
        except ImportError:
            raise ImportError("zarr required for .zarr files. Install with: pip install zarr")
    else:
        # Assume TIFF or other image format
        if HAS_TIFFFILE:
            mask = tifffile.imread(str(path))
        else:
            mask = io.imread(str(path))

    # Ensure proper dtype for label masks
    mask = mask.astype(np.int32)

    # Handle multi-dimensional masks (take first frame/slice if needed)
    if mask.ndim > 2:
        mask = mask[0] if mask.shape[0] <= mask.shape[-1] else mask[..., 0]

    # Validate and compute metadata
    if validate:
        metadata = validate_mask(mask, path)
    else:
        unique = np.unique(mask)
        metadata = MaskMetadata(
            path=path,
            shape=mask.shape,
            dtype=mask.dtype,
            n_cells=len(unique) - 1 if 0 in unique else len(unique),
            unique_labels=unique.tolist(),
            min_area=0,
            max_area=0,
            mean_area=0,
            issues=[],
            is_valid=True
        )

    return mask, metadata


def save_mask(
    mask: np.ndarray,
    path: Union[str, Path],
    format: str = 'tiff',
    compress: bool = True
) -> Path:
    """
    Save segmentation mask to file.

    Args:
        mask: Label mask array (2D, integer)
        path: Output path
        format: Output format ('tiff', 'npy', 'npz')
        compress: Whether to compress (for tiff and npz)

    Returns:
        Path to saved file
    """
    path = Path(path)

    # Ensure mask is proper label format
    mask = mask.astype(np.uint16 if mask.max() < 65535 else np.int32)

    if format == 'tiff' or path.suffix.lower() in ['.tif', '.tiff']:
        if HAS_TIFFFILE:
            compression = 'zlib' if compress else None
            tifffile.imwrite(str(path), mask, compression=compression)
        else:
            io.imsave(str(path), mask)

    elif format == 'npy' or path.suffix.lower() == '.npy':
        np.save(path, mask)

    elif format == 'npz' or path.suffix.lower() == '.npz':
        if compress:
            np.savez_compressed(path, masks=mask)
        else:
            np.savez(path, masks=mask)

    else:
        raise ValueError(f"Unsupported format: {format}")

    return path


def validate_mask(
    mask: np.ndarray,
    path: Optional[Path] = None
) -> MaskMetadata:
    """
    Validate mask integrity and compute statistics.

    Checks for common issues like:
    - Very small cells (noise)
    - Discontinuous cells (fragmented)
    - Missing labels (gaps in numbering)

    Args:
        mask: Label mask array
        path: Optional path for metadata

    Returns:
        MaskMetadata with validation results
    """
    from scipy.ndimage import sum as ndi_sum, label as ndi_label

    issues = []

    # Get unique labels (excluding background)
    unique_labels = np.unique(mask)
    background_present = 0 in unique_labels

    if background_present:
        cell_labels = unique_labels[unique_labels != 0]
    else:
        cell_labels = unique_labels

    n_cells = len(cell_labels)

    if n_cells == 0:
        return MaskMetadata(
            path=path,
            shape=mask.shape,
            dtype=mask.dtype,
            n_cells=0,
            unique_labels=unique_labels.tolist(),
            min_area=0,
            max_area=0,
            mean_area=0,
            issues=["No cells found in mask"],
            is_valid=False
        )

    # Calculate areas for all cells
    ones = np.ones_like(mask)
    areas = ndi_sum(ones, mask, cell_labels)

    # Check for very small cells
    min_area_threshold = 50
    small_cells = np.sum(areas < min_area_threshold)
    if small_cells > 0:
        issues.append(f"{small_cells} cells with area < {min_area_threshold} pixels (possible noise)")

    # Check for very large cells (potential merging issues)
    max_area_threshold = np.median(areas) * 10
    large_cells = np.sum(areas > max_area_threshold)
    if large_cells > 0:
        issues.append(f"{large_cells} cells with area > 10x median (possible merged cells)")

    # Check for gaps in label numbering
    if background_present and n_cells > 0:
        expected_max = n_cells
        actual_max = cell_labels.max()
        if actual_max > expected_max:
            issues.append(f"Gap in label numbering: {n_cells} cells but max label is {actual_max}")

    # Check for discontinuous cells (same label in multiple components)
    # This is expensive, so only do for small masks
    if mask.size < 10_000_000:  # ~3000x3000
        from scipy.ndimage import binary_erosion
        eroded = binary_erosion(mask > 0)
        _, n_components = ndi_label(eroded)
        if n_components > n_cells * 1.2:  # Allow 20% tolerance
            issues.append(f"Possible touching/fragmented cells: {n_cells} labels but ~{n_components} components")

    return MaskMetadata(
        path=path,
        shape=mask.shape,
        dtype=mask.dtype,
        n_cells=n_cells,
        unique_labels=unique_labels.tolist(),
        min_area=float(areas.min()),
        max_area=float(areas.max()),
        mean_area=float(areas.mean()),
        issues=issues,
        is_valid=len(issues) == 0
    )


def relabel_sequential(mask: np.ndarray) -> np.ndarray:
    """
    Relabel mask to have sequential labels starting from 1.

    Useful after removing cells or when labels have gaps.

    Args:
        mask: Label mask with potentially non-sequential labels

    Returns:
        Relabeled mask with sequential integers
    """
    from skimage.segmentation import relabel_sequential as skimage_relabel

    relabeled, _, _ = skimage_relabel(mask)
    return relabeled


def merge_masks(
    mask1: np.ndarray,
    mask2: np.ndarray,
    mode: str = 'union'
) -> np.ndarray:
    """
    Merge two masks together.

    Args:
        mask1: First mask
        mask2: Second mask (labels will be renumbered)
        mode: 'union' (combine all), 'intersection' (only overlapping), 'mask1_priority'

    Returns:
        Merged mask with relabeled cells
    """
    if mask1.shape != mask2.shape:
        raise ValueError(f"Mask shapes must match: {mask1.shape} vs {mask2.shape}")

    if mode == 'union':
        # Renumber mask2 to not conflict with mask1
        max_label = mask1.max()
        mask2_shifted = np.where(mask2 > 0, mask2 + max_label, 0)
        merged = np.where(mask1 > 0, mask1, mask2_shifted)

    elif mode == 'intersection':
        # Only keep cells that overlap in both masks
        overlap = (mask1 > 0) & (mask2 > 0)
        merged = np.where(overlap, mask1, 0)

    elif mode == 'mask1_priority':
        # mask1 takes priority, fill gaps with mask2
        max_label = mask1.max()
        mask2_shifted = np.where(mask2 > 0, mask2 + max_label, 0)
        merged = np.where(mask1 > 0, mask1, mask2_shifted)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return relabel_sequential(merged)
