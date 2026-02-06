"""
Mask manipulation utilities for semi-supervised ROI editing.

Provides operations for adding, removing, merging, and splitting cells
in segmentation masks.
"""

from typing import List, Tuple, Optional
import numpy as np
from skimage.draw import polygon
from skimage.segmentation import watershed, relabel_sequential
from skimage.morphology import remove_small_objects, binary_dilation, binary_erosion
from scipy.ndimage import binary_fill_holes, label as ndi_label, distance_transform_edt


def add_cell(
    masks: np.ndarray,
    polygon_coords: np.ndarray,
    overwrite_existing: bool = False
) -> np.ndarray:
    """
    Add a new cell from polygon coordinates.

    Args:
        masks: Existing label mask
        polygon_coords: Nx2 array of (row, col) coordinates defining the polygon
        overwrite_existing: If True, overwrite existing cells in the polygon area

    Returns:
        Updated mask with new cell added

    Example:
        >>> coords = np.array([[100, 100], [100, 150], [150, 150], [150, 100]])
        >>> new_masks = add_cell(masks, coords)
    """
    new_masks = masks.copy()
    new_label = int(masks.max()) + 1

    # Create polygon mask
    rr, cc = polygon(polygon_coords[:, 0], polygon_coords[:, 1], shape=masks.shape)

    # Ensure coordinates are within bounds
    valid = (rr >= 0) & (rr < masks.shape[0]) & (cc >= 0) & (cc < masks.shape[1])
    rr, cc = rr[valid], cc[valid]

    if overwrite_existing:
        new_masks[rr, cc] = new_label
    else:
        # Only write to background pixels
        background = new_masks[rr, cc] == 0
        new_masks[rr[background], cc[background]] = new_label

    return new_masks


def add_cell_from_click(
    masks: np.ndarray,
    image: np.ndarray,
    click_point: Tuple[int, int],
    expansion_threshold: float = 0.3
) -> np.ndarray:
    """
    Add a new cell by flood-filling from a click point.

    Uses intensity-based region growing from the click point.

    Args:
        masks: Existing label mask
        image: Intensity image for region growing
        click_point: (row, col) click position
        expansion_threshold: Intensity threshold for expansion (fraction of local intensity)

    Returns:
        Updated mask with new cell added
    """
    from skimage.segmentation import flood

    row, col = click_point

    # Only add if clicking on background
    if masks[row, col] != 0:
        return masks

    new_masks = masks.copy()
    new_label = int(masks.max()) + 1

    # Get local intensity at click point
    local_intensity = image[row, col]

    # Flood fill based on intensity similarity
    tolerance = local_intensity * expansion_threshold
    filled = flood(
        image,
        (row, col),
        tolerance=tolerance,
        connectivity=1
    )

    # Only fill background regions
    filled = filled & (masks == 0)

    if np.any(filled):
        new_masks[filled] = new_label

    return new_masks


def remove_cell(
    masks: np.ndarray,
    cell_id: int
) -> np.ndarray:
    """
    Remove a cell by setting its pixels to background.

    Args:
        masks: Label mask
        cell_id: ID of cell to remove

    Returns:
        Updated mask with cell removed
    """
    new_masks = masks.copy()
    new_masks[new_masks == cell_id] = 0
    return new_masks


def remove_cell_at_position(
    masks: np.ndarray,
    position: Tuple[int, int]
) -> np.ndarray:
    """
    Remove the cell at the given position.

    Args:
        masks: Label mask
        position: (row, col) position

    Returns:
        Updated mask with cell removed
    """
    row, col = position
    cell_id = masks[row, col]

    if cell_id == 0:
        return masks  # No cell at this position

    return remove_cell(masks, cell_id)


def merge_cells(
    masks: np.ndarray,
    cell_ids: List[int]
) -> np.ndarray:
    """
    Merge multiple cells into one.

    All specified cells will be relabeled to the smallest ID.

    Args:
        masks: Label mask
        cell_ids: List of cell IDs to merge

    Returns:
        Updated mask with cells merged
    """
    if len(cell_ids) < 2:
        return masks

    new_masks = masks.copy()
    target_id = min(cell_ids)

    for cell_id in cell_ids:
        if cell_id != target_id:
            new_masks[new_masks == cell_id] = target_id

    return new_masks


def split_cell(
    masks: np.ndarray,
    image: np.ndarray,
    cell_id: int,
    seed_points: np.ndarray
) -> np.ndarray:
    """
    Split a cell using watershed from seed points.

    Args:
        masks: Label mask
        image: Intensity image (used for watershed gradient)
        cell_id: ID of cell to split
        seed_points: Nx2 array of (row, col) seed positions for new cells

    Returns:
        Updated mask with cell split
    """
    from scipy.ndimage import sobel

    new_masks = masks.copy()
    cell_mask = masks == cell_id

    if not np.any(cell_mask):
        return masks

    # Create markers from seed points
    markers = np.zeros_like(masks)
    next_label = int(masks.max()) + 1

    for i, (row, col) in enumerate(seed_points):
        if cell_mask[row, col]:  # Ensure seed is inside the cell
            markers[row, col] = next_label + i

    if markers.max() == 0:
        return masks  # No valid seed points

    # Use image gradient as elevation map for watershed
    gradient = np.abs(sobel(image.astype(float), axis=0)) + np.abs(sobel(image.astype(float), axis=1))

    # Apply watershed within cell region
    ws_result = watershed(gradient, markers, mask=cell_mask)

    # Update masks
    new_masks[cell_mask] = 0
    new_masks[ws_result > 0] = ws_result[ws_result > 0]

    return new_masks


def split_touching_cells(
    masks: np.ndarray,
    image: np.ndarray,
    min_distance: int = 10
) -> np.ndarray:
    """
    Automatically split touching/merged cells using distance transform watershed.

    Args:
        masks: Label mask with potentially merged cells
        image: Intensity image
        min_distance: Minimum distance between cell centers

    Returns:
        Updated mask with split cells
    """
    from skimage.feature import peak_local_max

    new_masks = np.zeros_like(masks)
    next_label = 1

    for cell_id in np.unique(masks):
        if cell_id == 0:
            continue

        cell_mask = masks == cell_id

        # Distance transform
        distance = distance_transform_edt(cell_mask)

        # Find peaks (cell centers)
        peaks = peak_local_max(
            distance,
            min_distance=min_distance,
            labels=cell_mask.astype(int)
        )

        if len(peaks) <= 1:
            # No split needed
            new_masks[cell_mask] = next_label
            next_label += 1
        else:
            # Multiple peaks - watershed split
            markers = np.zeros_like(masks)
            for i, (r, c) in enumerate(peaks):
                markers[r, c] = i + 1

            ws = watershed(-distance, markers, mask=cell_mask)

            for ws_label in np.unique(ws):
                if ws_label == 0:
                    continue
                new_masks[ws == ws_label] = next_label
                next_label += 1

    return new_masks


def clean_small_objects(
    masks: np.ndarray,
    min_size: int = 50
) -> np.ndarray:
    """
    Remove cells smaller than min_size.

    Args:
        masks: Label mask
        min_size: Minimum cell area in pixels

    Returns:
        Cleaned mask
    """
    # Convert to binary, clean, then relabel
    binary = masks > 0
    cleaned = remove_small_objects(binary, min_size=min_size)

    # Keep only cells that survived cleaning
    new_masks = masks.copy()
    new_masks[~cleaned] = 0

    # Relabel to sequential
    relabeled, _, _ = relabel_sequential(new_masks)
    return relabeled


def fill_holes(
    masks: np.ndarray,
    max_hole_size: Optional[int] = None
) -> np.ndarray:
    """
    Fill holes within cells.

    Args:
        masks: Label mask
        max_hole_size: Maximum hole size to fill (None = fill all)

    Returns:
        Mask with holes filled
    """
    new_masks = masks.copy()

    for cell_id in np.unique(masks):
        if cell_id == 0:
            continue

        cell_mask = masks == cell_id
        filled = binary_fill_holes(cell_mask)

        if max_hole_size is not None:
            # Only fill small holes
            holes = filled & ~cell_mask
            if holes.sum() > max_hole_size:
                continue

        new_masks[filled] = cell_id

    return new_masks


def dilate_masks(
    masks: np.ndarray,
    iterations: int = 1
) -> np.ndarray:
    """
    Dilate all cell masks.

    Args:
        masks: Label mask
        iterations: Number of dilation iterations

    Returns:
        Dilated masks
    """
    from skimage.morphology import disk

    new_masks = masks.copy()
    selem = disk(1)

    for cell_id in np.unique(masks):
        if cell_id == 0:
            continue

        cell_mask = masks == cell_id

        for _ in range(iterations):
            dilated = binary_dilation(cell_mask, selem)
            # Only expand into background
            expansion = dilated & (new_masks == 0)
            new_masks[expansion] = cell_id
            cell_mask = new_masks == cell_id

    return new_masks


def erode_masks(
    masks: np.ndarray,
    iterations: int = 1
) -> np.ndarray:
    """
    Erode all cell masks.

    Args:
        masks: Label mask
        iterations: Number of erosion iterations

    Returns:
        Eroded masks
    """
    from skimage.morphology import disk

    new_masks = np.zeros_like(masks)
    selem = disk(1)

    for cell_id in np.unique(masks):
        if cell_id == 0:
            continue

        cell_mask = masks == cell_id

        for _ in range(iterations):
            cell_mask = binary_erosion(cell_mask, selem)

        new_masks[cell_mask] = cell_id

    return new_masks


def smooth_boundaries(
    masks: np.ndarray,
    sigma: float = 1.0
) -> np.ndarray:
    """
    Smooth cell boundaries using morphological operations.

    Args:
        masks: Label mask
        sigma: Smoothing strength

    Returns:
        Mask with smoothed boundaries
    """
    from scipy.ndimage import gaussian_filter

    new_masks = np.zeros_like(masks)
    iterations = max(1, int(sigma))

    for cell_id in np.unique(masks):
        if cell_id == 0:
            continue

        cell_mask = masks == cell_id

        # Close then open for smoothing
        closed = binary_dilation(cell_mask, iterations=iterations)
        closed = binary_erosion(closed, iterations=iterations)
        smoothed = binary_erosion(closed, iterations=iterations)
        smoothed = binary_dilation(smoothed, iterations=iterations)

        new_masks[smoothed] = cell_id

    return new_masks
