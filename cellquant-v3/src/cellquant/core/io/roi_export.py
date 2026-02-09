"""
ROI export utilities for ImageJ-compatible ROI files.

Generates ROI zip files that can be imported directly into ImageJ/Fiji,
with Cell_XXX naming to match CellID in results CSV.
"""

from pathlib import Path
from typing import Union, List, Tuple, Optional
import numpy as np
import struct
import zipfile
import io as iomodule
from dataclasses import dataclass


@dataclass
class ROIData:
    """Container for ROI information."""
    cell_id: int
    name: str
    coordinates: np.ndarray
    area: int
    centroid: Tuple[float, float]


def create_imagej_roi_bytes(
    coordinates: np.ndarray,
    roi_type: int = 7
) -> Optional[bytes]:
    """
    Create ImageJ ROI binary data from coordinates.

    Based on ImageJ ROI format specification:
    https://imagej.nih.gov/ij/developer/source/ij/io/RoiDecoder.java.html

    Args:
        coordinates: Nx2 array of (y, x) coordinates
        roi_type: ROI type (0=polygon, 7=freehand)

    Returns:
        Binary ROI data or None if coordinates are empty
    """
    n_coords = len(coordinates)
    if n_coords == 0:
        return None

    # Get bounding box
    y_coords = coordinates[:, 0]
    x_coords = coordinates[:, 1]
    top = int(np.min(y_coords))
    left = int(np.min(x_coords))
    bottom = int(np.max(y_coords))
    right = int(np.max(x_coords))

    # Build header (64 bytes)
    header = bytearray(64)
    header[0:4] = b'Iout'  # Magic number
    header[4:6] = struct.pack('>h', 228)  # Version
    header[6:8] = struct.pack('>h', roi_type)  # Type
    header[8:10] = struct.pack('>h', top)  # Top
    header[10:12] = struct.pack('>h', left)  # Left
    header[12:14] = struct.pack('>h', bottom)  # Bottom
    header[14:16] = struct.pack('>h', right)  # Right
    header[16:18] = struct.pack('>h', n_coords)  # N coordinates

    # Build coordinate data (relative to bounding box)
    coord_data = bytearray()
    for y, x in coordinates:
        coord_data.extend(struct.pack('>h', int(x - left)))
        coord_data.extend(struct.pack('>h', int(y - top)))

    return bytes(header + coord_data)


def extract_cell_contours(
    masks: np.ndarray,
    cell_ids: Optional[List[int]] = None
) -> List[ROIData]:
    """
    Extract contours for all cells in a mask.

    Args:
        masks: Label mask array
        cell_ids: Optional list of specific cell IDs to extract

    Returns:
        List of ROIData objects
    """
    from skimage.measure import find_contours, regionprops

    if cell_ids is None:
        cell_ids = np.unique(masks)
        cell_ids = cell_ids[cell_ids != 0]  # Exclude background

    rois = []

    # Get region properties for centroids
    props = regionprops(masks)
    centroid_map = {p.label: p.centroid for p in props}
    area_map = {p.label: p.area for p in props}

    for cell_id in cell_ids:
        cell_id = int(cell_id)

        # Get binary mask for this cell
        cell_mask = (masks == cell_id).astype(np.uint8)

        # Find contours
        contours = find_contours(cell_mask, 0.5)

        if len(contours) > 0:
            # Use the longest contour (main boundary)
            contour = max(contours, key=len)

            rois.append(ROIData(
                cell_id=cell_id,
                name=f"Cell_{cell_id:03d}",
                coordinates=contour,
                area=area_map.get(cell_id, 0),
                centroid=centroid_map.get(cell_id, (0, 0))
            ))

    return rois


def save_rois_imagej(
    masks: np.ndarray,
    save_path: Union[str, Path],
    cell_ids: Optional[List[int]] = None
) -> Tuple[Path, int]:
    """
    Save ROIs to ImageJ-compatible zip file.

    Creates a zip file containing .roi files with Cell_XXX naming
    that matches CellID in the results CSV.

    Args:
        masks: Label mask array
        save_path: Path for output zip file
        cell_ids: Optional list of specific cell IDs to export

    Returns:
        Tuple of (output path, number of ROIs saved)

    Example:
        >>> path, n = save_rois_imagej(masks, "rois.zip")
        >>> print(f"Saved {n} ROIs to {path}")
    """
    save_path = Path(save_path)

    # Extract contours
    rois = extract_cell_contours(masks, cell_ids)

    # Write directly to zip in memory (faster than temp files)
    with zipfile.ZipFile(save_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for roi in rois:
            roi_data = create_imagej_roi_bytes(roi.coordinates)
            if roi_data:
                roi_name = f"{roi.name}.roi"
                zipf.writestr(roi_name, roi_data)

    return save_path, len(rois)


def export_rois_zip(
    masks: np.ndarray,
    output_dir: Optional[Union[str, Path]] = None,
    base_name: str = "rois",
    include_metadata: bool = True
) -> Path:
    """
    Export ROIs with optional metadata CSV.

    Args:
        masks: Label mask array
        output_dir: Output directory (defaults to current)
        base_name: Base name for output files
        include_metadata: Whether to include metadata CSV in zip

    Returns:
        Path to output zip file
    """
    import csv

    output_dir = Path(output_dir) if output_dir else Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)

    zip_path = output_dir / f"{base_name}.zip"

    # Extract contours
    rois = extract_cell_contours(masks)

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add ROI files
        for roi in rois:
            roi_data = create_imagej_roi_bytes(roi.coordinates)
            if roi_data:
                zipf.writestr(f"{roi.name}.roi", roi_data)

        # Add metadata CSV
        if include_metadata and rois:
            csv_buffer = iomodule.StringIO()
            writer = csv.writer(csv_buffer)
            writer.writerow(['CellID', 'ROI_Name', 'Area', 'Centroid_Y', 'Centroid_X'])
            for roi in rois:
                writer.writerow([
                    roi.cell_id,
                    roi.name,
                    roi.area,
                    f"{roi.centroid[0]:.2f}",
                    f"{roi.centroid[1]:.2f}"
                ])
            zipf.writestr('roi_metadata.csv', csv_buffer.getvalue())

    return zip_path


def load_imagej_roi(roi_data: bytes) -> Optional[np.ndarray]:
    """
    Parse ImageJ ROI binary data.

    Args:
        roi_data: Binary ROI data

    Returns:
        Nx2 array of (y, x) coordinates or None
    """
    if len(roi_data) < 64:
        return None

    # Check magic number
    if roi_data[0:4] != b'Iout':
        return None

    # Parse header
    roi_type = struct.unpack('>h', roi_data[6:8])[0]
    top = struct.unpack('>h', roi_data[8:10])[0]
    left = struct.unpack('>h', roi_data[10:12])[0]
    n_coords = struct.unpack('>h', roi_data[16:18])[0]

    # Parse coordinates
    coords = []
    offset = 64
    for _ in range(n_coords):
        if offset + 4 > len(roi_data):
            break
        x = struct.unpack('>h', roi_data[offset:offset+2])[0] + left
        y = struct.unpack('>h', roi_data[offset+2:offset+4])[0] + top
        coords.append([y, x])
        offset += 4

    return np.array(coords) if coords else None


def load_rois_from_zip(
    zip_path: Union[str, Path]
) -> dict:
    """
    Load ROIs from ImageJ zip file.

    Args:
        zip_path: Path to ROI zip file

    Returns:
        Dictionary mapping ROI name -> coordinates array
    """
    zip_path = Path(zip_path)
    rois = {}

    with zipfile.ZipFile(zip_path, 'r') as zipf:
        for name in zipf.namelist():
            if name.endswith('.roi'):
                roi_data = zipf.read(name)
                coords = load_imagej_roi(roi_data)
                if coords is not None:
                    roi_name = Path(name).stem
                    rois[roi_name] = coords

    return rois
