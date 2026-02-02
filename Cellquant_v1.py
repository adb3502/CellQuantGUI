import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog
import numpy as np
import pandas as pd
from skimage import io
from skimage.exposure import rescale_intensity
from skimage.color import gray2rgb 
from cellpose import models
from cellpose import utils as cellpose_utils 
from cellpose import io as cellpose_io 
from cellpose import plot as cellpose_plot
import os
from pathlib import Path
import re
import torch 
import copy 

# --- Configuration (Defaults for UI) ---
DEFAULT_NUCLEAR_SUFFIX = "C0"
DEFAULT_CYTO_SUFFIX = "C1"
DEFAULT_MARKER_SUFFIXES = "C2" 
DEFAULT_MARKER_NAMES = "Marker1"
DEFAULT_CELLPOSE_MIN_SIZE = 15 
DEFAULT_CELLPOSE_MODEL = "cpsam" 
DEFAULT_CELLPOSE_DIAMETER = 30.0
DEFAULT_CELLPOSE_FLOW_THRESH = 0.4
DEFAULT_SEGMENTATION_TARGET = "Cells"  # "Cells", "Nuclei", "Both"

# --- Helper Functions ---
def load_image(image_path):
    """Load an image with robust multi-dimensional handling"""
    try:
        img = io.imread(str(image_path))
        if img.ndim > 2: 
            if img.shape[0] > 0 and img.shape[0] <= 5 : 
                 img = img[0, :, :]
            elif img.shape[-1] > 0 and img.shape[-1] <= 5: 
                 img = img[:, :, 0]
        return img.astype(np.uint16) if img.dtype == np.uint16 else img.astype(np.float32)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def normalize_image(img, perc_low=1.0, perc_high=99.0):
    """Enhanced normalization with better edge case handling"""
    img_float = img.astype(np.float32)
    min_val, max_val = np.min(img_float), np.max(img_float)
    if min_val == max_val: 
        return np.zeros_like(img_float) if min_val == 0 else (img_float - min_val) 

    p_low, p_high = np.percentile(img_float, (perc_low, perc_high))
    if p_high <= p_low: 
        p_low = min_val
        p_high = max_val
    if p_high <= p_low:
        return (img_float - p_low) / (1e-6 if (p_high - p_low) == 0 else (p_high - p_low)) 
    return rescale_intensity(img_float, in_range=(p_low, p_high), out_range=(0.0, 1.0))

def estimate_background_enhanced(image_data, cell_masks, method='median'):
    """Enhanced background estimation with multiple methods"""
    background_pixels = image_data[cell_masks == 0]
    
    # Improved: Use 5th percentile if background region is too small
    if len(background_pixels) < (image_data.size * 0.01): 
        print(f"    Warning: Limited background pixels ({len(background_pixels)}), using 5th percentile")
        return np.percentile(image_data, 5) 
    
    if method == 'median':
        return np.median(background_pixels)
    elif method == 'percentile5':
        return np.percentile(background_pixels, 5)
    elif method == 'mean':
        return np.mean(background_pixels)
    else:
        return np.median(background_pixels)

def calculate_ctcf(roi_pixels_intensity, area, background_intensity, nuclear_pixels_intensity=None, is_mitochondrial=False):
    """Calculate CTCF with enhanced metrics and optional nuclear subtraction for mitochondrial markers"""
    integrated_density = np.sum(roi_pixels_intensity)
    ctcf_raw = integrated_density - (area * background_intensity)

    # For mitochondrial markers, subtract nuclear signal from the whole cell signal
    if is_mitochondrial and nuclear_pixels_intensity is not None:
        nuclear_signal = np.sum(nuclear_pixels_intensity)
        ctcf_raw -= nuclear_signal

    ctcf = max(0, ctcf_raw)
    mean_intensity = integrated_density / area if area > 0 else 0
    return ctcf, integrated_density, mean_intensity

def create_colorful_segmentation_overlay(image, masks):
    """Create colorful segmentation overlay like Cellpose GUI"""
    try:
        # Convert image to grayscale if needed for overlay
        if image.ndim == 3:
            display_img = np.mean(image, axis=0 if image.shape[0] <= 3 else -1)
        else:
            display_img = image
            
        # Use Cellpose's built-in mask overlay function
        colored_overlay = cellpose_plot.mask_overlay(display_img, masks)
        return colored_overlay
    except Exception as e:
        print(f"    Warning: Could not create colorful overlay: {e}")
        # Fallback to simple colored masks
        return create_simple_colored_overlay(image, masks)

def find_file_by_suffix(base_name, target_suffix, base_names_map, file_suffix_map):
    """
    Find a file that matches the base_name and target_suffix.
    Returns the file path if found, None otherwise.
    """
    if base_name not in base_names_map:
        return None
    
    # Look through all files for this base_name
    for file_path in base_names_map[base_name]:
        extracted_suffix = file_suffix_map.get(file_path, "")
        
        # Case-insensitive comparison
        if extracted_suffix.lower() == target_suffix.lower():
            return file_path
    
    return None

def create_simple_colored_overlay(image, masks):
    """Fallback method for colored overlay"""
    if image.ndim == 3:
        display_img = np.mean(image, axis=0 if image.shape[0] <= 3 else -1)
    else:
        display_img = image

    # Normalize display image
    display_img_norm = (display_img - np.min(display_img)) / (np.max(display_img) - np.min(display_img) + 1e-8)

    # Create RGB overlay
    overlay = np.stack([display_img_norm, display_img_norm, display_img_norm], axis=2)

    # Add random colors for each mask
    unique_masks = np.unique(masks)[1:]  # Skip background
    np.random.seed(42)  # For reproducible colors

    for mask_id in unique_masks:
        color = np.random.rand(3)
        mask_pixels = masks == mask_id
        overlay[mask_pixels] = color * 0.7 + overlay[mask_pixels] * 0.3

    return (overlay * 255).astype(np.uint8)

def save_rois_with_cell_ids(masks, save_path):
    """
    Save ROIs to a zip file with explicit Cell_XXX naming that matches CellID in CSV.

    This ensures that ROI "Cell_042" in ImageJ corresponds to CellID 42 in the results CSV,
    making it easy to find and edit specific cells.
    """
    import zipfile
    from skimage import measure
    import struct

    def create_imagej_roi(coordinates, name):
        """Create an ImageJ ROI in binary format for polygon/freehand ROI"""
        # ImageJ ROI format header
        # Based on https://imagej.nih.gov/ij/developer/source/ij/io/RoiDecoder.java.html

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
        width = right - left + 1
        height = bottom - top + 1

        # ROI type: 0=polygon, 7=freehand
        roi_type = 7  # freehand

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

        # Build coordinate data
        coord_data = bytearray()
        for y, x in coordinates:
            # Coordinates relative to bounding box
            coord_data.extend(struct.pack('>h', int(x - left)))
            coord_data.extend(struct.pack('>h', int(y - top)))

        return bytes(header + coord_data)

    # Create temporary directory for ROI files
    import tempfile
    temp_dir = tempfile.mkdtemp()
    roi_files = []

    try:
        unique_cell_ids = np.unique(masks)[1:]  # Skip background (0)

        for cell_id in unique_cell_ids:
            # Get the mask for this cell
            cell_mask = (masks == cell_id).astype(np.uint8)

            # Find contours for this cell
            contours = measure.find_contours(cell_mask, 0.5)

            if len(contours) > 0:
                # Use the longest contour (main boundary)
                contour = max(contours, key=len)

                # Create ROI name that matches CellID
                roi_name = f"Cell_{int(cell_id):03d}.roi"
                roi_path = os.path.join(temp_dir, roi_name)

                # Create ImageJ ROI binary data
                roi_data = create_imagej_roi(contour, roi_name)

                if roi_data:
                    # Write ROI file
                    with open(roi_path, 'wb') as f:
                        f.write(roi_data)
                    roi_files.append((roi_name, roi_path))

        # Create zip file with all ROIs
        with zipfile.ZipFile(save_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for roi_name, roi_path in roi_files:
                zipf.write(roi_path, arcname=roi_name)

        print(f"      Saved {len(roi_files)} ROIs to {os.path.basename(save_path)}")

    finally:
        # Clean up temporary files
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

# --- Core Analysis Logic ---
def process_condition(condition_name, folder_path, 
                      channel_settings, 
                      cellpose_model_instance, 
                      main_output_folder,
                      cellpose_diameter=DEFAULT_CELLPOSE_DIAMETER, 
                      cellpose_flow_thresh=DEFAULT_CELLPOSE_FLOW_THRESH,
                      cellpose_min_size=DEFAULT_CELLPOSE_MIN_SIZE,
                      segmentation_target="Cells"):
    
    nuclear_suffix = channel_settings['nuclear_suffix']
    cyto_suffix = channel_settings['cyto_suffix']
    marker_suffixes = channel_settings['marker_suffixes']
    marker_names = channel_settings['marker_names']
    
    use_nuc_seg = channel_settings.get('use_nuc_seg', True)
    use_cyto_seg = channel_settings.get('use_cyto_seg', True)
    third_seg_channel_name = channel_settings.get('third_seg_channel_name', "None")

    print(f"\nProcessing condition: {condition_name} from {folder_path}")
    print(f"  Segmentation Target: {segmentation_target}")
    print(f"  Seg Channels: Nuc={use_nuc_seg}({nuclear_suffix}), Cyto={use_cyto_seg}({cyto_suffix}), Third='{third_seg_channel_name}'")
    print(f"  Quant Markers: {list(zip(marker_suffixes, marker_names))}")
    
    # Determine what segmentations to run
    run_cellular = segmentation_target in ["Cells", "Both"]
    run_nuclear = segmentation_target in ["Nuclei", "Both"]
    
    results_dict = {}  # Will store results for each segmentation type
    
    if run_cellular:
        print(f"  Running cellular segmentation...")
        cellular_results = _run_segmentation_analysis(
            condition_name, folder_path, channel_settings, cellpose_model_instance,
            main_output_folder, cellpose_diameter, cellpose_flow_thresh, cellpose_min_size,
            "cellular", use_nuc_seg, use_cyto_seg, third_seg_channel_name
        )
        results_dict["cellular"] = cellular_results
    
    if run_nuclear:
        print(f"  Running nuclear-only segmentation...")
        nuclear_results = _run_segmentation_analysis(
            condition_name, folder_path, channel_settings, cellpose_model_instance,
            main_output_folder, cellpose_diameter, cellpose_flow_thresh, cellpose_min_size,
            "nuclear", True, False, "None"  # Only nuclear channel for nuclear segmentation
        )
        results_dict["nuclear"] = nuclear_results
    
    return results_dict

def _run_segmentation_analysis(condition_name, folder_path, channel_settings, cellpose_model_instance,
                              main_output_folder, cellpose_diameter, cellpose_flow_thresh, cellpose_min_size,
                              seg_type, use_nuc_seg, use_cyto_seg, third_seg_channel_name):
    """Run segmentation and analysis for a specific segmentation type"""
    
    nuclear_suffix = channel_settings['nuclear_suffix']
    cyto_suffix = channel_settings['cyto_suffix']
    marker_suffixes = channel_settings['marker_suffixes']
    marker_names = channel_settings['marker_names']
    
    condition_results_list = [] 
    
    # Create output path for this segmentation type
    seg_folder_name = f"{condition_name}_{seg_type}" if seg_type != "cellular" else condition_name
    condition_output_path = Path(main_output_folder) / seg_folder_name
    condition_output_path.mkdir(parents=True, exist_ok=True)
    
    tiff_files = list(Path(folder_path).glob('*.tif*')) 
    base_names_map = {}  # Maps base_name -> list of file paths
    file_suffix_map = {}  # Maps file_path -> extracted_suffix
    
    for f_path in tiff_files:
        # Enhanced pattern to handle complex naming patterns and extract the actual suffix
        # Pattern captures the suffix at the end, handling various formats:
        # "basename_suffix.tif", "basename_w1_t00_z00.tif", "basename_C02_sx_10_sy_9_w3.tif"
        
        # Try multiple patterns to handle different naming conventions
        patterns = [
            # Pattern 1: Handle complex suffixes with time/z-stack info (w1_t00_z00, w2_t00_z00, etc.)
            r"^(.+?)_([A-Za-z]\d+(?:_[a-z]\d+)*)\.tif(?:f)?$",  # Captures: sample_w1_t00_z00.tif
            # Pattern 2: Handle XY/Z/T metadata patterns - extract base before XY coordinates
            r"^(.+?)_XY\d+.*?_([A-Za-z]\d+)\.tif(?:f)?$",  # Capture 1_XY946668190_Z0_T0_C2.tiff
            # Pattern 3: Handle sx_/sy_ coordinate patterns with suffix at end
            r"^(.+?)_([A-Za-z]\d+)_sx_\d+_sy_\d+.*?\.tif(?:f)?$",  # Hacat SA JC1 VC OM_C02_sx_1_sy_1_w2.tif
            # Pattern 4: Standard underscore separation with suffix at end (any suffix)
            r"^(.+?)_([^_\s]+)\.tif(?:f)?$",
        ]
        
        matched = False
        for pattern in patterns:
            match = re.match(pattern, f_path.name, re.IGNORECASE)
            if match:
                if len(match.groups()) == 2:
                    base_name = match.group(1).strip()
                    extracted_suffix = match.group(2).strip()
                    
                    # Store the file path and its extracted suffix
                    file_suffix_map[f_path] = extracted_suffix
                    
                    if base_name not in base_names_map:
                        base_names_map[base_name] = []
                    base_names_map[base_name].append(f_path)
                    matched = True
                    break
        
        # If no pattern matched, try a fallback approach
        if not matched:
            # Simple fallback: assume everything before last underscore is base, after is suffix
            parts = f_path.stem.rsplit('_', 1)
            if len(parts) == 2:
                base_name = parts[0].strip()
                extracted_suffix = parts[1].strip()
                file_suffix_map[f_path] = extracted_suffix
                
                if base_name not in base_names_map:
                    base_names_map[base_name] = []
                base_names_map[base_name].append(f_path)
            
    if not base_names_map:
        print(f"  No valid image sets found for condition {condition_name}")
        return []

    print(f"    Preparing {len(base_names_map)} image sets for segmentation...")
    for base_name in sorted(list(base_names_map.keys())):
        segmentation_input_stack_list = []
        loaded_channels_for_seg_display = [] 
        raw_channels_for_composite = {}  # Store raw images for composite

        # Load channels for segmentation based on parameters
        if use_nuc_seg:
            nuclear_path = find_file_by_suffix(base_name, nuclear_suffix, base_names_map, file_suffix_map)
            if nuclear_path:
                nuclear_img_raw = load_image(nuclear_path)
                if nuclear_img_raw is not None:
                    nuclear_norm = normalize_image(nuclear_img_raw)
                    segmentation_input_stack_list.append(nuclear_norm)
                    loaded_channels_for_seg_display.append(nuclear_norm)
                    raw_channels_for_composite[f'nuclear_{nuclear_suffix}'] = nuclear_img_raw
                else: 
                    print(f"    Warning: Could not load nuclear seg channel for {base_name}")
            else: 
                print(f"    Warning: Nuclear seg channel file not found for {base_name} (Suffix: {nuclear_suffix})")
        
        if use_cyto_seg:
            cyto_path = find_file_by_suffix(base_name, cyto_suffix, base_names_map, file_suffix_map)
            if cyto_path:
                cyto_img_raw = load_image(cyto_path)
                if cyto_img_raw is not None:
                    cyto_norm = normalize_image(cyto_img_raw)
                    segmentation_input_stack_list.append(cyto_norm)
                    loaded_channels_for_seg_display.append(cyto_norm)
                    raw_channels_for_composite[f'cyto_{cyto_suffix}'] = cyto_img_raw
                else: 
                    print(f"    Warning: Could not load cyto seg channel for {base_name}")
            else: 
                print(f"    Warning: Cyto seg channel file not found for {base_name} (Suffix: {cyto_suffix})")

        if third_seg_channel_name != "None" and third_seg_channel_name:
            try:
                marker_idx_for_seg = marker_names.index(third_seg_channel_name)
                third_seg_suffix = marker_suffixes[marker_idx_for_seg]
                third_seg_path = find_file_by_suffix(base_name, third_seg_suffix, base_names_map, file_suffix_map)
                if third_seg_path:
                    third_img_raw = load_image(third_seg_path)
                    if third_img_raw is not None:
                        third_norm = normalize_image(third_img_raw)
                        segmentation_input_stack_list.append(third_norm)
                        loaded_channels_for_seg_display.append(third_norm)
                        raw_channels_for_composite[f'third_{third_seg_suffix}'] = third_img_raw
                    else: 
                        print(f"    Warning: Could not load third seg channel '{third_seg_channel_name}' for {base_name}")
                else: 
                    print(f"    Warning: Third seg channel file '{third_seg_channel_name}' (Suffix: {third_seg_suffix}) not found for {base_name}")
            except (ValueError, IndexError):
                print(f"    Warning: Marker name '{third_seg_channel_name}' for third seg channel not in defined marker names for this condition.")

        # Load all marker channels for composite (even if not used for segmentation)
        for marker_idx, marker_suffix in enumerate(marker_suffixes):
            marker_name = marker_names[marker_idx]
            marker_path = find_file_by_suffix(base_name, marker_suffix, base_names_map, file_suffix_map)
            if marker_path:
                marker_img_raw = load_image(marker_path)
                if marker_img_raw is not None:
                    raw_channels_for_composite[f'marker_{marker_suffix}_{marker_name}'] = marker_img_raw

        if not segmentation_input_stack_list:
            print(f"    Skipping {base_name}: No valid channels selected or loaded for segmentation.")
            continue
        
        # Limit to 3 channels for Cellpose-SAM
        if len(segmentation_input_stack_list) > 3:
            print(f"    Warning: More than 3 channels selected for segmentation for {base_name}. Using first 3.")
            segmentation_input_stack_list = segmentation_input_stack_list[:3]
            loaded_channels_for_seg_display = loaded_channels_for_seg_display[:3]

        final_stacked_for_cellpose = np.stack(segmentation_input_stack_list, axis=0)
        
        # Create visual composite for overlay
        if loaded_channels_for_seg_display:
            visual_sum = np.sum(np.stack(loaded_channels_for_seg_display, axis=0), axis=0)
            visual_composite_uint8 = (normalize_image(visual_sum, 0, 100) * 255).astype(np.uint8)
        else: 
            visual_composite_uint8 = np.zeros_like(segmentation_input_stack_list[0], dtype=np.uint8) 

        # Run Cellpose on this image set
        print(f"    Running Cellpose on {base_name}...")
        try:
            eval_args = {
                "diameter": cellpose_diameter,
                "flow_threshold": cellpose_flow_thresh,
                "min_size": cellpose_min_size
            }
            
            # Enhanced channel handling for Cellpose-SAM
            if final_stacked_for_cellpose.shape[0] == 1:
                eval_args["channels"] = [0,0]  # Single channel
            # For 2-3 channels, Cellpose-SAM handles automatically - no channels parameter needed
            
            masks, flows, styles = cellpose_model_instance.eval(
                [final_stacked_for_cellpose],  # List with single image
                **eval_args
            )
            masks = masks[0]  # Extract from list
            
        except Exception as e:
            print(f"    ERROR during Cellpose segmentation for {base_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Process results
        image_set_output_path = condition_output_path / base_name
        image_set_output_path.mkdir(parents=True, exist_ok=True)

        if np.max(masks) == 0:
            print(f"      No cells detected by Cellpose in {base_name}.")
            continue

        # Save standard outputs
        masks_save_path = image_set_output_path / f"{base_name}_masks_{seg_type}.tif"
        io.imsave(str(masks_save_path), masks.astype(np.uint16), check_contrast=False)

        # Save ROIs with explicit Cell_XXX naming that matches CellID in CSV
        rois_zip_path = image_set_output_path / f"{base_name}_rois_{seg_type}.zip"
        save_rois_with_cell_ids(masks, str(rois_zip_path))
        
        # Create standard outline overlay
        outlines = cellpose_utils.masks_to_outlines(masks)
        overlay_display_img = visual_composite_uint8
        
        if overlay_display_img.ndim == 2:
            overlay_display_img_rgb = gray2rgb(overlay_display_img)
        else: 
            overlay_display_img_rgb = overlay_display_img 
        
        overlay_img_manual = overlay_display_img_rgb.copy()
        overlay_img_manual[outlines, 0] = 255
        overlay_img_manual[outlines, 1] = 0
        overlay_img_manual[outlines, 2] = 0

        overlay_save_path = image_set_output_path / f"{base_name}_outline_overlay_{seg_type}.png"
        io.imsave(str(overlay_save_path), overlay_img_manual, check_contrast=False)
        
        # NEW: Create colorful segmentation overlay like Cellpose GUI
        try:
            colorful_overlay = create_colorful_segmentation_overlay(final_stacked_for_cellpose, masks)
            colorful_overlay_path = image_set_output_path / f"{base_name}_colorful_segmentation_{seg_type}.png"
            io.imsave(str(colorful_overlay_path), colorful_overlay, check_contrast=False)
        except Exception as e:
            print(f"    Warning: Could not create colorful overlay for {base_name}: {e}")
        
        # NEW: Create multi-channel composite TIFF
        try:
            if raw_channels_for_composite:
                composite_stack = []
                composite_names = []
                
                # Ensure all images have same dimensions
                reference_shape = None
                for name, img in raw_channels_for_composite.items():
                    if reference_shape is None:
                        reference_shape = img.shape
                    elif img.shape == reference_shape:
                        composite_stack.append(img)
                        composite_names.append(name)
                    else:
                        print(f"    Warning: Skipping {name} from composite - dimension mismatch")
                
                if len(composite_stack) > 1:
                    composite_array = np.stack(composite_stack, axis=0)
                    composite_path = image_set_output_path / f"{base_name}_composite_all_channels_{seg_type}.tif"
                    io.imsave(str(composite_path), composite_array.astype(np.uint16), check_contrast=False)
                    
                    # Save channel info
                    info_path = image_set_output_path / f"{base_name}_channel_info_{seg_type}.txt"
                    with open(info_path, 'w') as f:
                        f.write(f"Channel order for {base_name}_composite_all_channels_{seg_type}.tif:\n")
                        for i, name in enumerate(composite_names):
                            f.write(f"Channel {i}: {name}\n")
        except Exception as e:
            print(f"    Warning: Could not create composite TIFF for {base_name}: {e}")
        
        # Quantify markers - FIXED: One row per cell with all markers as columns
        print(f"    Starting quantification for {len(marker_suffixes)} markers: {marker_names}")

        # Load nuclear mask if needed for mitochondrial marker quantification
        nuclear_masks = None
        mitochondrial_markers_list = channel_settings.get('mitochondrial_markers', [])
        if mitochondrial_markers_list:
            nuc_mask_path = find_file_by_suffix(base_name, nuclear_suffix, base_names_map, file_suffix_map)
            if nuc_mask_path:
                try:
                    nuc_mask_temp = load_image(nuc_mask_path)
                    if nuc_mask_temp is not None:
                        # Run Cellpose on nuclear channel to get nuclear masks
                        nuc_norm = normalize_image(nuc_mask_temp)
                        masks_nuc, _, _ = cellpose_model_instance.eval(
                            [np.stack([nuc_norm], axis=0)],
                            diameter=cellpose_diameter,
                            flow_threshold=cellpose_flow_thresh,
                            min_size=cellpose_min_size,
                            channels=[0, 0]
                        )
                        nuclear_masks = masks_nuc[0]
                except Exception as e:
                    print(f"    Warning: Could not generate nuclear masks for mitochondrial calculation: {e}")

        # First, get all unique cells and create base entries
        unique_cell_ids = np.unique(masks)[1:]
        cell_data_dict = {}  # Will store data by cell_id

        print(f"    Found {len(unique_cell_ids)} cells to quantify")

        # Initialize data structure for each cell
        for cell_id in unique_cell_ids:
            cell_coords_yx = np.where(masks == cell_id)

            valid_y = cell_coords_yx[0] < masks.shape[0]
            valid_x = cell_coords_yx[1] < masks.shape[1]
            valid_coords = np.logical_and(valid_y, valid_x)

            y_coords_final = cell_coords_yx[0][valid_coords]
            x_coords_final = cell_coords_yx[1][valid_coords]
            area = len(y_coords_final)

            if area > 0:
                cell_data_dict[cell_id] = {
                    "Condition": condition_name,
                    "SegmentationType": seg_type,
                    "ImageSet": base_name,
                    "CellID": cell_id,
                    "Area": area
                }
        
        # Now process each marker and add data to existing cell entries
        for marker_idx, marker_suffix in enumerate(marker_suffixes):
            marker_name = marker_names[marker_idx]
            is_mitochondrial_marker = marker_name in mitochondrial_markers_list
            print(f"      Processing marker {marker_idx+1}/{len(marker_suffixes)}: {marker_name} (suffix: {marker_suffix}){' [MITOCHONDRIAL]' if is_mitochondrial_marker else ''}")

            marker_path = find_file_by_suffix(base_name, marker_suffix, base_names_map, file_suffix_map)

            if not marker_path:
                print(f"      ERROR: Skipping marker {marker_name} for {base_name}: File for suffix '{marker_suffix}' not found.")
                print(f"        Looking for suffix: {marker_suffix}")
                if base_name in base_names_map:
                    available_suffixes = [file_suffix_map.get(f, "unknown") for f in base_names_map[base_name]]
                    print(f"        Available suffixes for {base_name}: {available_suffixes}")
                else:
                    print(f"        Base name '{base_name}' not found in any files.")

                # Add empty values for missing marker
                for cell_id in cell_data_dict:
                    cell_data_dict[cell_id][f"{marker_name}_CTCF"] = 0
                    cell_data_dict[cell_id][f"{marker_name}_IntegratedDensity"] = 0
                    cell_data_dict[cell_id][f"{marker_name}_MeanIntensity"] = 0
                    cell_data_dict[cell_id][f"{marker_name}_Background"] = 0
                continue

            print(f"      Found file: {marker_path.name}")
            marker_img = load_image(marker_path)
            if marker_img is None:
                print(f"      ERROR: Skipping marker {marker_name} for {base_name}: Error loading image.")
                # Add empty values for failed marker
                for cell_id in cell_data_dict:
                    cell_data_dict[cell_id][f"{marker_name}_CTCF"] = 0
                    cell_data_dict[cell_id][f"{marker_name}_IntegratedDensity"] = 0
                    cell_data_dict[cell_id][f"{marker_name}_MeanIntensity"] = 0
                    cell_data_dict[cell_id][f"{marker_name}_Background"] = 0
                continue

            try:
                # Enhanced background estimation
                background_val = estimate_background_enhanced(marker_img, masks, method='median')
                print(f"      Background value: {background_val:.2f}")

                cells_processed = 0
                for cell_id in cell_data_dict:
                    cell_coords_yx = np.where(masks == cell_id)

                    valid_y = cell_coords_yx[0] < marker_img.shape[0]
                    valid_x = cell_coords_yx[1] < marker_img.shape[1]
                    valid_coords = np.logical_and(valid_y, valid_x)

                    y_coords_final = cell_coords_yx[0][valid_coords]
                    x_coords_final = cell_coords_yx[1][valid_coords]

                    if len(y_coords_final) == 0:
                        # No valid pixels for this cell
                        cell_data_dict[cell_id][f"{marker_name}_CTCF"] = 0
                        cell_data_dict[cell_id][f"{marker_name}_IntegratedDensity"] = 0
                        cell_data_dict[cell_id][f"{marker_name}_MeanIntensity"] = 0
                        cell_data_dict[cell_id][f"{marker_name}_Background"] = background_val
                        continue

                    roi_pixels = marker_img[y_coords_final, x_coords_final]
                    area = len(roi_pixels)

                    # For mitochondrial markers, extract nuclear pixels
                    nuclear_pixels = None
                    if is_mitochondrial_marker and nuclear_masks is not None:
                        try:
                            nuclear_coords_yx = np.where(nuclear_masks > 0)
                            # Get pixels within the cell that are also in the nucleus
                            nuclear_pixels = marker_img[nuclear_coords_yx[0], nuclear_coords_yx[1]]
                        except Exception as e:
                            print(f"      Warning: Could not extract nuclear pixels for {marker_name}: {e}")

                    ctcf_val, integrated_density, mean_intensity = calculate_ctcf(
                        roi_pixels, area, background_val,
                        nuclear_pixels_intensity=nuclear_pixels,
                        is_mitochondrial=is_mitochondrial_marker
                    )

                    # Add marker data to existing cell entry
                    cell_data_dict[cell_id][f"{marker_name}_CTCF"] = ctcf_val
                    cell_data_dict[cell_id][f"{marker_name}_IntegratedDensity"] = integrated_density
                    cell_data_dict[cell_id][f"{marker_name}_MeanIntensity"] = mean_intensity
                    cell_data_dict[cell_id][f"{marker_name}_Background"] = background_val

                    cells_processed += 1

                print(f"      Successfully processed {cells_processed} cells for marker {marker_name}")
                
            except Exception as e:
                print(f"      ERROR: Exception while processing marker {marker_name}: {e}")
                import traceback
                traceback.print_exc()
                # Add empty values for failed marker
                for cell_id in cell_data_dict:
                    cell_data_dict[cell_id][f"{marker_name}_CTCF"] = 0
                    cell_data_dict[cell_id][f"{marker_name}_IntegratedDensity"] = 0
                    cell_data_dict[cell_id][f"{marker_name}_MeanIntensity"] = 0
                    cell_data_dict[cell_id][f"{marker_name}_Background"] = 0
                continue
        
        # Convert cell_data_dict to list for CSV output
        for cell_data in cell_data_dict.values():
            condition_results_list.append(cell_data)
                
        print(f"    Finished quantifying {base_name}, found {np.max(masks)} cells/nuclei. Output in: {image_set_output_path.name}")
    
    return condition_results_list

# --- Tkinter UI ---
class EditConditionDialog(simpledialog.Dialog):
    def __init__(self, parent, title=None, initial_condition_data=None):
        self.initial_data = initial_condition_data if initial_condition_data else {}
        self.marker_names_for_dialog = self.initial_data.get("marker_names", [])
        if isinstance(self.marker_names_for_dialog, str): 
            self.marker_names_for_dialog = [n.strip() for n in self.marker_names_for_dialog.split(',') if n.strip()]

        super().__init__(parent, title)

    def body(self, master):
        row_idx = 0
        tk.Label(master, text="Condition Name:").grid(row=row_idx, column=0, sticky=tk.W, padx=5, pady=2)
        self.name_var = tk.StringVar(master, value=self.initial_data.get("name", ""))
        tk.Entry(master, textvariable=self.name_var, width=40, state='readonly').grid(row=row_idx, column=1, columnspan=2, padx=5, pady=2)
        row_idx += 1

        tk.Label(master, text="Folder Path:").grid(row=row_idx, column=0, sticky=tk.W, padx=5, pady=2)
        self.path_var = tk.StringVar(master, value=str(self.initial_data.get("path", "")))
        tk.Entry(master, textvariable=self.path_var, width=40, state='readonly').grid(row=row_idx, column=1, columnspan=2, padx=5, pady=2)
        row_idx += 1

        tk.Label(master, text="--- Channels for Quantification ---", font=('Helvetica', 10, 'bold')).grid(row=row_idx, column=0, columnspan=3, sticky=tk.W, pady=(10,2))
        row_idx += 1

        tk.Label(master, text="Nuclear Suffix (Quant):").grid(row=row_idx, column=0, sticky=tk.W, padx=5, pady=2)
        self.nuc_quant_var = tk.StringVar(master, value=self.initial_data.get("nuclear_suffix", DEFAULT_NUCLEAR_SUFFIX))
        self.nuc_quant_entry_widget = tk.Entry(master, textvariable=self.nuc_quant_var, width=20)
        self.nuc_quant_entry_widget.grid(row=row_idx, column=1, padx=5, pady=2)
        row_idx += 1

        tk.Label(master, text="Cytosolic Suffix (Quant):").grid(row=row_idx, column=0, sticky=tk.W, padx=5, pady=2)
        self.cyto_quant_var = tk.StringVar(master, value=self.initial_data.get("cyto_suffix", DEFAULT_CYTO_SUFFIX))
        tk.Entry(master, textvariable=self.cyto_quant_var, width=20).grid(row=row_idx, column=1, padx=5, pady=2)
        row_idx += 1

        tk.Label(master, text="Fluorescent Marker Suffixes (comma-sep):").grid(row=row_idx, column=0, sticky=tk.W, padx=5, pady=2)
        self.marker_s_quant_var = tk.StringVar(master, value=self.initial_data.get("marker_suffixes_str", DEFAULT_MARKER_SUFFIXES))
        self.marker_s_quant_entry = tk.Entry(master, textvariable=self.marker_s_quant_var, width=40)
        self.marker_s_quant_entry.grid(row=row_idx, column=1, columnspan=2, padx=5, pady=2)
        self.marker_s_quant_var.trace_add("write", self._update_marker_options_for_seg_dialog)
        row_idx += 1

        tk.Label(master, text="Fluorescent Marker Names (comma-sep):").grid(row=row_idx, column=0, sticky=tk.W, padx=5, pady=2)
        self.marker_n_quant_var = tk.StringVar(master, value=self.initial_data.get("marker_names_str", DEFAULT_MARKER_NAMES))
        self.marker_n_quant_entry = tk.Entry(master, textvariable=self.marker_n_quant_var, width=40)
        self.marker_n_quant_entry.grid(row=row_idx, column=1, columnspan=2, padx=5, pady=2)
        self.marker_n_quant_var.trace_add("write", self._update_marker_options_for_seg_dialog)
        row_idx += 1

        tk.Label(master, text="Mitochondrial Markers (comma-sep marker names):").grid(row=row_idx, column=0, sticky=tk.W, padx=5, pady=2)
        self.mitochondrial_markers_var = tk.StringVar(master, value=self.initial_data.get("mitochondrial_markers_str", ""))
        self.mitochondrial_markers_entry = tk.Entry(master, textvariable=self.mitochondrial_markers_var, width=40)
        self.mitochondrial_markers_entry.grid(row=row_idx, column=1, columnspan=2, padx=5, pady=2)
        row_idx += 1

        tk.Label(master, text="--- Channels for Cellpose Segmentation (Max 3) ---", font=('Helvetica', 10, 'bold')).grid(row=row_idx, column=0, columnspan=3, sticky=tk.W, pady=(10,2))
        row_idx += 1

        self.use_nuc_seg_var = tk.BooleanVar(master, value=self.initial_data.get("use_nuc_seg", True))
        tk.Checkbutton(master, text="Use Nuclear for Seg (from Quant Suffix above)", variable=self.use_nuc_seg_var).grid(row=row_idx, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)
        row_idx += 1

        self.use_cyto_seg_var = tk.BooleanVar(master, value=self.initial_data.get("use_cyto_seg", True))
        tk.Checkbutton(master, text="Use Cytoplasmic for Seg (from Quant Suffix above)", variable=self.use_cyto_seg_var).grid(row=row_idx, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)
        row_idx += 1
        
        tk.Label(master, text="Optional 3rd Seg Channel:").grid(row=row_idx, column=0, sticky=tk.W, padx=5, pady=2)
        self.third_seg_channel_name_var = tk.StringVar(master, value=self.initial_data.get("third_seg_channel_name", "None"))
        
        self.marker_options_for_seg_dialog_list = ["None"] + self.marker_names_for_dialog
        self.third_seg_channel_dropdown = ttk.Combobox(master, textvariable=self.third_seg_channel_name_var, 
                                                       values=self.marker_options_for_seg_dialog_list, state="readonly", width=37)
        if self.third_seg_channel_name_var.get() not in self.marker_options_for_seg_dialog_list:
            self.third_seg_channel_name_var.set("None")
        self.third_seg_channel_dropdown.grid(row=row_idx, column=1, columnspan=2, padx=5, pady=2)
        row_idx += 1
        
        return self.nuc_quant_entry_widget

    def _update_marker_options_for_seg_dialog(self, *args):
        """Dynamically update the marker options for 3rd seg channel in the dialog."""
        marker_names_str = self.marker_n_quant_var.get()
        new_marker_names = [n.strip() for n in marker_names_str.split(',') if n.strip()]
        
        self.marker_names_for_dialog = new_marker_names
        new_options = ["None"] + self.marker_names_for_dialog
        
        current_selection = self.third_seg_channel_name_var.get()
        self.third_seg_channel_dropdown['values'] = new_options
        
        if current_selection not in new_options:
            self.third_seg_channel_name_var.set("None")

    def apply(self):
        nuc_suffix = self.nuc_quant_var.get().strip()
        cyto_suffix = self.cyto_quant_var.get().strip()
        marker_suffixes_str = self.marker_s_quant_var.get().strip()
        marker_names_str = self.marker_n_quant_var.get().strip()
        mitochondrial_markers_str = self.mitochondrial_markers_var.get().strip()

        use_nuc_seg = self.use_nuc_seg_var.get()
        use_cyto_seg = self.use_cyto_seg_var.get()
        third_seg_channel_name = self.third_seg_channel_name_var.get()
        if third_seg_channel_name == "None":
            third_seg_channel_name = "None"

        if not all([nuc_suffix, cyto_suffix, marker_suffixes_str, marker_names_str]):
            messagebox.showerror("Validation Error", "All quantification channel suffix/name fields are required.", parent=self)
            self.result = None
            return

        marker_suffixes = [s.strip() for s in marker_suffixes_str.split(',')]
        marker_names = [n.strip() for n in marker_names_str.split(',')]

        if len(marker_suffixes) != len(marker_names):
            messagebox.showerror("Validation Error", "Number of marker suffixes must match number of marker names.", parent=self)
            self.result = None
            return

        # Parse mitochondrial markers list
        mitochondrial_markers = [m.strip() for m in mitochondrial_markers_str.split(',') if m.strip()]

        # Validate that all mitochondrial markers are in the marker names list
        for mito_marker in mitochondrial_markers:
            if mito_marker not in marker_names:
                messagebox.showerror("Validation Error", f"Mitochondrial marker '{mito_marker}' is not in the Marker Names list.", parent=self)
                self.result = None
                return

        selected_seg_count = use_nuc_seg + use_cyto_seg + (1 if third_seg_channel_name != "None" else 0)
        if selected_seg_count == 0:
            messagebox.showerror("Validation Error", "At least one channel must be selected for segmentation.", parent=self)
            self.result = None
            return

        self.result = self.initial_data
        self.result.update({
            "nuclear_suffix": nuc_suffix, "cyto_suffix": cyto_suffix,
            "marker_suffixes": marker_suffixes, "marker_names": marker_names,
            "marker_suffixes_str": marker_suffixes_str, "marker_names_str": marker_names_str,
            "mitochondrial_markers": mitochondrial_markers, "mitochondrial_markers_str": mitochondrial_markers_str,
            "use_nuc_seg": use_nuc_seg, "use_cyto_seg": use_cyto_seg,
            "third_seg_channel_name": third_seg_channel_name
        })

class BioVisionCTCFProApp:
    def __init__(self, master):
        self.master = master
        master.title("BioVision CTCF Pro - Advanced Cellular Analysis Suite")
        master.geometry("900x900") 

        self.conditions_data_list = [] 

        # Experiment Setup Frame
        exp_setup_frame = tk.LabelFrame(master, text="Experiment Setup", padx=10, pady=10)
        exp_setup_frame.grid(row=0, column=0, columnspan=3, sticky=tk.EW, padx=5, pady=5)

        tk.Label(exp_setup_frame, text="Main Output Folder:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.output_folder_var = tk.StringVar()
        self.output_entry = tk.Entry(exp_setup_frame, textvariable=self.output_folder_var, width=60)
        self.output_entry.grid(row=0, column=1, pady=2, sticky=tk.EW)
        tk.Button(exp_setup_frame, text="Browse...", command=self.select_output_folder).grid(row=0, column=2, padx=5, pady=2)

        tk.Label(exp_setup_frame, text="Experiment Folder (contains condition subfolders):").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.exp_folder_var = tk.StringVar()
        self.exp_folder_entry = tk.Entry(exp_setup_frame, textvariable=self.exp_folder_var, width=60)
        self.exp_folder_entry.grid(row=1, column=1, pady=2, sticky=tk.EW)
        tk.Button(exp_setup_frame, text="Browse...", command=self.select_experiment_folder).grid(row=1, column=2, padx=5, pady=2)

        # Default Channel Settings Frame
        default_channel_frame = tk.LabelFrame(master, text="Default Channel Settings (Applied when loading & via 'Apply Defaults')", padx=10, pady=10)
        default_channel_frame.grid(row=1, column=0, columnspan=3, sticky=tk.EW, padx=5, pady=5)
        
        tk.Label(default_channel_frame, text="Nuclear Suffix (Quant):").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.default_nuc_suffix_var = tk.StringVar(value=DEFAULT_NUCLEAR_SUFFIX)
        tk.Entry(default_channel_frame, textvariable=self.default_nuc_suffix_var, width=25).grid(row=0, column=1, sticky=tk.W, pady=2)

        tk.Label(default_channel_frame, text="Cytosolic Suffix (Quant):").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.default_cyto_suffix_var = tk.StringVar(value=DEFAULT_CYTO_SUFFIX)
        tk.Entry(default_channel_frame, textvariable=self.default_cyto_suffix_var, width=25).grid(row=1, column=1, sticky=tk.W, pady=2)

        tk.Label(default_channel_frame, text="Fluorescent Marker Suffixes (comma-sep):").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.default_marker_s_var = tk.StringVar(value=DEFAULT_MARKER_SUFFIXES)
        self.default_marker_s_entry = tk.Entry(default_channel_frame, textvariable=self.default_marker_s_var, width=25)
        self.default_marker_s_entry.grid(row=2, column=1, sticky=tk.W, pady=2)
        
        tk.Label(default_channel_frame, text="Fluorescent Marker Names (comma-sep):").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.default_marker_n_var = tk.StringVar(value=DEFAULT_MARKER_NAMES)
        self.default_marker_n_entry = tk.Entry(default_channel_frame, textvariable=self.default_marker_n_var, width=25)
        self.default_marker_n_entry.grid(row=3, column=1, sticky=tk.W, pady=2)
        self.default_marker_n_var.trace_add("write", self._update_marker_mitochondrial_checkboxes)

        # Mitochondrial Markers Section
        tk.Label(default_channel_frame, text="--- Mark Mitochondrial Markers (Optional) ---", font=('Helvetica', 9, 'italic')).grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=(5,0))

        # Frame to hold mitochondrial marker checkboxes
        self.mito_checkboxes_frame = tk.Frame(default_channel_frame)
        self.mito_checkboxes_frame.grid(row=5, column=0, columnspan=2, sticky=tk.W, padx=0, pady=2)

        self.default_mito_marker_vars = {}  # Will store BooleanVar for each marker

        # NEW: Segmentation Target Selection
        tk.Label(default_channel_frame, text="--- Segmentation & Analysis Target ---", font=('Helvetica', 9, 'italic')).grid(row=6, column=0, columnspan=2, sticky=tk.W, pady=(5,0))

        tk.Label(default_channel_frame, text="Segmentation Target:").grid(row=7, column=0, sticky=tk.W, pady=2)
        self.default_seg_target_var = tk.StringVar(value=DEFAULT_SEGMENTATION_TARGET)
        self.seg_target_dropdown = ttk.Combobox(default_channel_frame, textvariable=self.default_seg_target_var,
                                               values=["Cells", "Nuclei", "Both"], state="readonly", width=22)
        self.seg_target_dropdown.grid(row=7, column=1, sticky=tk.W, pady=2)

        tk.Label(default_channel_frame, text="--- Default Segmentation Channels ---", font=('Helvetica', 9, 'italic')).grid(row=8, column=0, columnspan=2, sticky=tk.W, pady=(5,0))
        self.default_use_nuc_seg_var = tk.BooleanVar(value=True)
        tk.Checkbutton(default_channel_frame, text="Use Nuclear for Seg", variable=self.default_use_nuc_seg_var).grid(row=9, column=0, sticky=tk.W)
        self.default_use_cyto_seg_var = tk.BooleanVar(value=True)
        tk.Checkbutton(default_channel_frame, text="Use Cytoplasmic for Seg", variable=self.default_use_cyto_seg_var).grid(row=9, column=1, sticky=tk.W)

        tk.Label(default_channel_frame, text="Default Optional 3rd Seg Channel:").grid(row=10, column=0, sticky=tk.W, pady=2)
        self.default_third_seg_channel_name_var = tk.StringVar(value="None")
        self.default_third_seg_channel_dropdown = ttk.Combobox(default_channel_frame,
                                                              textvariable=self.default_third_seg_channel_name_var,
                                                              values=["None"], state="readonly", width=22)
        self.default_third_seg_channel_dropdown.grid(row=10, column=1, sticky=tk.W, pady=2)
        self._update_default_third_seg_channel_options_mainui()

        # Now initialize mitochondrial checkboxes after all widgets are created
        self._update_marker_mitochondrial_checkboxes()

        tk.Button(default_channel_frame, text="Apply These Defaults to All Loaded Conditions", command=self.apply_defaults_to_all_loaded).grid(row=11, column=0, columnspan=3, pady=(5,2), sticky=tk.W)

        # Cellpose Settings Frame
        cp_frame = tk.LabelFrame(master, text="Cellpose-SAM Settings", padx=10, pady=10)
        cp_frame.grid(row=2, column=0, columnspan=3, sticky=tk.EW, padx=5, pady=5)
        
        tk.Label(cp_frame, text="Model:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.cellpose_model_var = tk.StringVar(value=DEFAULT_CELLPOSE_MODEL) 
        self.model_options = ["cpsam", "cyto", "cyto2", "cyto3", "cyto4", "nuclei", "livecell"] 
        self.model_dropdown = ttk.Combobox(cp_frame, textvariable=self.cellpose_model_var, values=self.model_options, state="readonly", width=20)
        self.model_dropdown.grid(row=0, column=1, sticky=tk.W, pady=2)
        tk.Label(cp_frame, text="(Cellpose-SAM automatically handles 1-3 input channels)").grid(row=0, column=2, columnspan=2, sticky=tk.W, padx=5, pady=2, ipadx=5)

        tk.Label(cp_frame, text="Diameter (pixels, 0 for auto):").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.diameter_var = tk.DoubleVar(value=DEFAULT_CELLPOSE_DIAMETER)
        tk.Entry(cp_frame, textvariable=self.diameter_var, width=10).grid(row=1, column=1, sticky=tk.W, pady=2)

        tk.Label(cp_frame, text="Flow Threshold:").grid(row=1, column=2, sticky=tk.W, padx=10, pady=2)
        self.flow_thresh_var = tk.DoubleVar(value=DEFAULT_CELLPOSE_FLOW_THRESH)
        tk.Entry(cp_frame, textvariable=self.flow_thresh_var, width=10).grid(row=1, column=3, sticky=tk.W, pady=2)

        tk.Label(cp_frame, text="Min Size (pixels):").grid(row=2, column=0, sticky=tk.W, pady=2) 
        self.min_size_var = tk.IntVar(value=DEFAULT_CELLPOSE_MIN_SIZE) 
        tk.Entry(cp_frame, textvariable=self.min_size_var, width=10).grid(row=2, column=1, sticky=tk.W, pady=2)

        # Conditions Management Frame
        cond_list_frame = tk.LabelFrame(master, text="Experiment Conditions", padx=10, pady=10)
        cond_list_frame.grid(row=3, column=0, columnspan=3, sticky=tk.NSEW, padx=5, pady=5)
        master.grid_rowconfigure(3, weight=1) 
        cond_list_frame.grid_columnconfigure(0, weight=1) 

        button_bar_cond = tk.Frame(cond_list_frame)
        button_bar_cond.grid(row=0, column=0, pady=(0,5), sticky=tk.W)
        tk.Button(button_bar_cond, text="Load/Refresh Conditions from Folder", command=self.load_conditions_from_folder).pack(side=tk.LEFT, padx=(0,5))
        tk.Button(button_bar_cond, text="Edit Selected Condition Settings", command=self.edit_selected_condition).pack(side=tk.LEFT, padx=(0,5))
        tk.Button(button_bar_cond, text="Remove Selected Condition", command=self.remove_selected_condition).pack(side=tk.LEFT)
        
        self.conditions_listbox = tk.Listbox(cond_list_frame, height=8)
        self.conditions_listbox.grid(row=1, column=0, sticky=tk.NSEW, padx=5, pady=5)
        cond_list_frame.grid_rowconfigure(1, weight=1)

        # Run & Status Frame
        run_status_frame = tk.Frame(master, padx=5, pady=5)
        run_status_frame.grid(row=4, column=0, columnspan=3, sticky=tk.EW)

        self.status_var = tk.StringVar(value="Ready. Select folders and load conditions.")
        tk.Label(run_status_frame, textvariable=self.status_var, wraplength=500, justify=tk.LEFT).pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        tk.Button(run_status_frame, text="Run Analysis", command=self.run_analysis_clicked, bg="lightgreen", height=2, width=15).pack(side=tk.RIGHT, padx=10)
        
        # Configure grid weights
        exp_setup_frame.grid_columnconfigure(1, weight=1) 
        default_channel_frame.grid_columnconfigure(1, weight=1)
        cp_frame.grid_columnconfigure(1, weight=1)
        cp_frame.grid_columnconfigure(3, weight=1)

    def _update_marker_mitochondrial_checkboxes(self, *args):
        """Dynamically update mitochondrial marker checkboxes based on marker names."""
        marker_names_str = self.default_marker_n_var.get()
        new_marker_names = [n.strip() for n in marker_names_str.split(',') if n.strip()]

        # Clear existing checkboxes
        for widget in self.mito_checkboxes_frame.winfo_children():
            widget.destroy()

        self.default_mito_marker_vars = {}

        # Create checkboxes for each marker
        for marker_name in new_marker_names:
            if marker_name not in self.default_mito_marker_vars:
                self.default_mito_marker_vars[marker_name] = tk.BooleanVar(value=False)

            cb = tk.Checkbutton(
                self.mito_checkboxes_frame,
                text=marker_name,
                variable=self.default_mito_marker_vars[marker_name]
            )
            cb.pack(side=tk.LEFT, padx=5)

        # Also update the 3rd seg channel dropdown
        self._update_default_third_seg_channel_options_mainui()

    def _update_default_third_seg_channel_options_mainui(self, *args):
        """Dynamically update the marker options for default 3rd seg channel in main UI."""
        marker_names_str = self.default_marker_n_var.get()
        new_marker_names = [n.strip() for n in marker_names_str.split(',') if n.strip()]

        new_options = ["None"] + new_marker_names

        current_selection = self.default_third_seg_channel_name_var.get()
        self.default_third_seg_channel_dropdown['values'] = new_options

        if current_selection not in new_options:
            self.default_third_seg_channel_name_var.set("None")

    def select_output_folder(self):
        folder = filedialog.askdirectory(title="Select Main Output Folder")
        if folder:
            self.output_folder_var.set(folder)

    def select_experiment_folder(self):
        folder = filedialog.askdirectory(title="Select Experiment Folder (containing condition subfolders)")
        if folder:
            self.exp_folder_var.set(folder)
            self.load_conditions_from_folder() 

    def _update_conditions_listbox(self):
        self.conditions_listbox.delete(0, tk.END)
        for i, cond_data in enumerate(self.conditions_data_list):
            quant_marker_str = ", ".join([f"{s}={n}" for s, n in zip(cond_data['marker_suffixes'], cond_data['marker_names'])])
            
            seg_chans_display = []
            if cond_data.get('use_nuc_seg'): seg_chans_display.append(f"N({cond_data['nuclear_suffix']})")
            if cond_data.get('use_cyto_seg'): seg_chans_display.append(f"C({cond_data['cyto_suffix']})")
            third_seg = cond_data.get('third_seg_channel_name', "None")
            if third_seg != "None" and third_seg:
                try:
                    third_seg_idx = cond_data['marker_names'].index(third_seg)
                    third_seg_suffix = cond_data['marker_suffixes'][third_seg_idx]
                    seg_chans_display.append(f"M({third_seg_suffix}={third_seg})")
                except (ValueError, IndexError): 
                    seg_chans_display.append(f"M({third_seg}-?)")

            seg_str = ", ".join(seg_chans_display) if seg_chans_display else "None"

            display_text = (f"{i+1}. {cond_data['name']} "
                            f"[Markers:{quant_marker_str}] " 
                            f"[Seg: {seg_str}]")
            self.conditions_listbox.insert(tk.END, display_text)

    def load_conditions_from_folder(self):
        exp_folder_str = self.exp_folder_var.get()
        if not exp_folder_str or not os.path.isdir(exp_folder_str):
            messagebox.showwarning("Warning", "Please select a valid experiment folder first.", parent=self.master)
            return

        default_nuc = self.default_nuc_suffix_var.get().strip()
        default_cyto = self.default_cyto_suffix_var.get().strip()
        default_marker_s_str = self.default_marker_s_var.get().strip()
        default_marker_n_str = self.default_marker_n_var.get().strip()

        default_use_nuc_seg = self.default_use_nuc_seg_var.get()
        default_use_cyto_seg = self.default_use_cyto_seg_var.get()
        default_third_seg = self.default_third_seg_channel_name_var.get()

        if not all([default_nuc, default_cyto, default_marker_s_str, default_marker_n_str]):
            messagebox.showerror("Error", "Please set all Default Channel Settings (for quantification) before loading conditions.", parent=self.master)
            return

        default_marker_s = [s.strip() for s in default_marker_s_str.split(',')]
        default_marker_n = [n.strip() for n in default_marker_n_str.split(',')]

        if len(default_marker_s) != len(default_marker_n):
            messagebox.showerror("Error", "Default Marker Suffixes count must match Default Marker Names count.", parent=self.master)
            return

        if default_third_seg != "None" and default_third_seg not in default_marker_n:
            messagebox.showwarning("Warning",
                                   f"The 'Default Optional 3rd Seg Channel' ('{default_third_seg}') "
                                   f"is not in the 'Default Marker Names' list. It will be set to 'None' for loaded conditions.",
                                   parent=self.master)
            default_third_seg = "None"

        # Collect mitochondrial markers from checkboxes
        default_mitochondrial_markers = [name for name, var in self.default_mito_marker_vars.items() if var.get()]
        default_mitochondrial_markers_str = ", ".join(default_mitochondrial_markers)

        exp_folder = Path(exp_folder_str)
        self.conditions_data_list = []

        found_conditions_count = 0
        for item in exp_folder.iterdir():
            if item.is_dir():
                if any(item.glob('*.tif*')):
                    condition_entry = {
                        "name": item.name,
                        "path": item,
                        "nuclear_suffix": default_nuc,
                        "cyto_suffix": default_cyto,
                        "marker_suffixes": copy.deepcopy(default_marker_s),
                        "marker_names": copy.deepcopy(default_marker_n),
                        "marker_suffixes_str": default_marker_s_str,
                        "marker_names_str": default_marker_n_str,
                        "mitochondrial_markers": copy.deepcopy(default_mitochondrial_markers),
                        "mitochondrial_markers_str": default_mitochondrial_markers_str,
                        "use_nuc_seg": default_use_nuc_seg,
                        "use_cyto_seg": default_use_cyto_seg,
                        "third_seg_channel_name": default_third_seg
                    }
                    self.conditions_data_list.append(condition_entry)
                    found_conditions_count += 1

        self._update_conditions_listbox()
        if found_conditions_count > 0:
            self.status_var.set(f"Found {found_conditions_count} condition(s). Applied default settings. Edit if needed.")
        else:
            self.status_var.set(f"No subfolders with TIFF images found in '{exp_folder.name}'.")

    def apply_defaults_to_all_loaded(self):
        if not self.conditions_data_list:
            messagebox.showinfo("Info", "No conditions loaded to apply defaults to.", parent=self.master)
            return

        default_nuc = self.default_nuc_suffix_var.get().strip()
        default_cyto = self.default_cyto_suffix_var.get().strip()
        default_marker_s_str = self.default_marker_s_var.get().strip()
        default_marker_n_str = self.default_marker_n_var.get().strip()
        
        default_use_nuc_seg = self.default_use_nuc_seg_var.get()
        default_use_cyto_seg = self.default_use_cyto_seg_var.get()
        default_third_seg = self.default_third_seg_channel_name_var.get()

        if not all([default_nuc, default_cyto, default_marker_s_str, default_marker_n_str]):
            messagebox.showerror("Error", "Please ensure all Default Channel Settings (for quantification) are filled before applying.", parent=self.master)
            return
        
        default_marker_s = [s.strip() for s in default_marker_s_str.split(',')]
        default_marker_n = [n.strip() for n in default_marker_n_str.split(',')]

        if len(default_marker_s) != len(default_marker_n):
            messagebox.showerror("Error", "Default Marker Suffixes count must match Default Marker Names count.", parent=self.master)
            return
        
        if default_third_seg != "None" and default_third_seg not in default_marker_n:
            messagebox.showwarning("Warning", 
                                   f"The 'Default Optional 3rd Seg Channel' ('{default_third_seg}') "
                                   f"is not in the 'Default Marker Names' list. It will be set to 'None' when applying defaults.",
                                   parent=self.master)
            effective_default_third_seg = "None"
        else:
            effective_default_third_seg = default_third_seg

        # Collect mitochondrial markers from checkboxes
        default_mitochondrial_markers = [name for name, var in self.default_mito_marker_vars.items() if var.get()]
        default_mitochondrial_markers_str = ", ".join(default_mitochondrial_markers)

        if messagebox.askyesno("Confirm Apply Defaults",
                               "This will overwrite settings for ALL currently loaded conditions with the values from the 'Default Channel Settings' section. Are you sure?",
                               parent=self.master):
            for cond_data in self.conditions_data_list:
                cond_data["nuclear_suffix"] = default_nuc
                cond_data["cyto_suffix"] = default_cyto
                cond_data["marker_suffixes"] = copy.deepcopy(default_marker_s)
                cond_data["marker_names"] = copy.deepcopy(default_marker_n)
                cond_data["marker_suffixes_str"] = default_marker_s_str
                cond_data["marker_names_str"] = default_marker_n_str
                cond_data["mitochondrial_markers"] = copy.deepcopy(default_mitochondrial_markers)
                cond_data["mitochondrial_markers_str"] = default_mitochondrial_markers_str
                cond_data["use_nuc_seg"] = default_use_nuc_seg
                cond_data["use_cyto_seg"] = default_use_cyto_seg
                cond_data["third_seg_channel_name"] = effective_default_third_seg

            self._update_conditions_listbox()
            self.status_var.set("Applied default channel settings to all loaded conditions.")

    def edit_selected_condition(self):
        selected_indices = self.conditions_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("No Selection", "Please select a condition from the list to edit its settings.", parent=self.master)
            return
        
        idx = selected_indices[0]
        condition_to_edit = self.conditions_data_list[idx]
        
        dialog = EditConditionDialog(self.master, f"Edit Settings for: {condition_to_edit['name']}", 
                                     initial_condition_data=copy.deepcopy(condition_to_edit))
        
        if dialog.result: 
            self.conditions_data_list[idx] = dialog.result 
            self._update_conditions_listbox()
            self.status_var.set(f"Settings updated for condition: {condition_to_edit['name']}")
        else:
            self.status_var.set(f"Editing cancelled for condition: {condition_to_edit['name']}")

    def remove_selected_condition(self):
        selected_indices = self.conditions_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("No Selection", "Please select a condition to remove.", parent=self.master)
            return
        
        idx = selected_indices[0]
        condition_name = self.conditions_data_list[idx]['name']
        if messagebox.askyesno("Confirm Delete", f"Are you sure you want to remove condition '{condition_name}' from the analysis list?", parent=self.master):
            del self.conditions_data_list[idx]
            self._update_conditions_listbox()
            self.status_var.set(f"Removed condition: {condition_name}")

    def run_analysis_clicked(self):
        output_folder = self.output_folder_var.get()
        if not output_folder or not os.path.isdir(output_folder):
            messagebox.showerror("Error", "Please select a valid Main Output Folder.", parent=self.master)
            return
        if not self.conditions_data_list:
            messagebox.showerror("Error", "No conditions loaded. Please select an Experiment Folder and click 'Load/Refresh Conditions'.", parent=self.master)
            return
        
        if not messagebox.askokcancel("Confirm Analysis", "Please ensure channel configurations (for quantification AND segmentation) are correctly set for ALL conditions (edit if necessary).\n\nProceed with analysis?", parent=self.master):
            return
        
        cp_model_type = self.cellpose_model_var.get()
        cp_diameter_val = self.diameter_var.get()
        cp_diameter = None if cp_diameter_val <= 0 else float(cp_diameter_val) 
        cp_flow_thresh = self.flow_thresh_var.get()
        cp_min_size = self.min_size_var.get() 
        segmentation_target = self.default_seg_target_var.get()

        self.status_var.set("Initializing Cellpose...")
        self.master.update_idletasks()
        
        use_gpu = torch.cuda.is_available()
        print(f"Attempting to use GPU: {use_gpu}")
        
        try:
            cellpose_model_instance = models.CellposeModel(gpu=use_gpu, model_type=cp_model_type) 
            print(f"Cellpose model '{cp_model_type}' initialized (GPU: {use_gpu}).")
        except Exception as e:
            messagebox.showerror("Cellpose Error", f"Failed to initialize Cellpose model: {e}\nMake sure Cellpose is installed correctly and check GPU compatibility if using GPU.", parent=self.master)
            self.status_var.set(f"Error: Cellpose init failed.")
            print(f"Cellpose init error: {e}")
            return

        self.status_var.set("Running analysis... This may take a while.")
        self.master.update_idletasks()
        
        all_results_data_dict = {}  # Will store results by segmentation type
        
        try:
            for i, cond_data in enumerate(self.conditions_data_list):
                self.status_var.set(f"Processing Condition {i+1}/{len(self.conditions_data_list)}: {cond_data['name']}")
                self.master.update_idletasks()
                
                results = process_condition(
                    cond_data['name'],
                    cond_data['path'], 
                    cond_data,
                    cellpose_model_instance, 
                    output_folder,
                    cellpose_diameter=cp_diameter,
                    cellpose_flow_thresh=cp_flow_thresh,
                    cellpose_min_size=cp_min_size,
                    segmentation_target=segmentation_target
                )
                
                # Merge results into main dictionary
                for seg_type, result_list in results.items():
                    if seg_type not in all_results_data_dict:
                        all_results_data_dict[seg_type] = []
                    all_results_data_dict[seg_type].extend(result_list)
            
            if not any(all_results_data_dict.values()):
                messagebox.showinfo("Info", "Analysis complete, but no cell data was generated across all conditions.", parent=self.master)
                self.status_var.set("Analysis complete. No data generated.")
                return

            # Save results to separate CSVs for each segmentation type
            exp_folder_name_part = Path(self.exp_folder_var.get()).name if self.exp_folder_var.get() else "experiment"
            timestamp = exp_folder_name_part + "_" + pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            
            saved_files = []
            for seg_type, result_list in all_results_data_dict.items():
                if result_list:
                    results_df = pd.DataFrame(result_list)
                    output_csv_path = Path(output_folder) / f"ctcf_analysis_{seg_type}_{timestamp}.csv"
                    results_df.to_csv(output_csv_path, index=False)
                    saved_files.append(str(output_csv_path))
            
            files_str = "\n".join(saved_files)
            messagebox.showinfo("Success", f"Analysis complete! Results CSV(s) saved to:\n{files_str}\n\nMasks, ROIs, overlays, and composite images saved in condition-specific subfolders.", parent=self.master)
            self.status_var.set(f"Analysis complete. {len(saved_files)} CSV file(s) generated.")

        except Exception as e:
            messagebox.showerror("Analysis Error", f"An error occurred during analysis: {e}", parent=self.master)
            self.status_var.set(f"Error during analysis: {e}")
            print(f"Analysis runtime error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    root = tk.Tk()
    app = BioVisionCTCFProApp(root)
    root.mainloop()