"""
Cellquant v2 - Timelapse Extension
===================================
Extends the existing Cellquant_v1.py with timelapse tracking capabilities.

Maintains full compatibility with existing workflow:
- Same channel configuration system
- Same CTCF calculation
- Same output structure
- Adds: tracking, lineage trees, per-cell intensity profiles

Author: Claude (for Longevity India radiation biology)
Date: 2025
"""

import numpy as np
import pandas as pd
from skimage import io
from skimage.exposure import rescale_intensity
from skimage.measure import regionprops_table
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, List, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

# Try to import existing Cellquant functions
try:
    from Cellquant_v1 import (
        load_image, 
        normalize_image, 
        estimate_background_enhanced,
        calculate_ctcf,
        create_colorful_segmentation_overlay,
        DEFAULT_CELLPOSE_MODEL,
        DEFAULT_CELLPOSE_DIAMETER,
        DEFAULT_CELLPOSE_FLOW_THRESH,
        DEFAULT_CELLPOSE_MIN_SIZE,
    )
    CELLQUANT_AVAILABLE = True
    print("✓ Cellquant_v1 functions imported successfully")
except ImportError:
    CELLQUANT_AVAILABLE = False
    print("⚠ Cellquant_v1 not found, using built-in functions")
    
    # Built-in versions of core functions
    DEFAULT_CELLPOSE_MODEL = "cpsam"
    DEFAULT_CELLPOSE_DIAMETER = 30.0
    DEFAULT_CELLPOSE_FLOW_THRESH = 0.4
    DEFAULT_CELLPOSE_MIN_SIZE = 15
    
    def load_image(image_path):
        """Load an image with robust multi-dimensional handling"""
        try:
            img = io.imread(str(image_path))
            if img.ndim > 2: 
                if img.shape[0] > 0 and img.shape[0] <= 5: 
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
            p_low, p_high = min_val, max_val
        if p_high <= p_low:
            return (img_float - p_low) / max(1e-6, p_high - p_low)
        return rescale_intensity(img_float, in_range=(p_low, p_high), out_range=(0.0, 1.0))

    def estimate_background_enhanced(image_data, cell_masks, method='median'):
        """Enhanced background estimation"""
        background_pixels = image_data[cell_masks == 0]
        if len(background_pixels) < (image_data.size * 0.01):
            return np.percentile(image_data, 5)
        if method == 'median':
            return np.median(background_pixels)
        elif method == 'percentile5':
            return np.percentile(background_pixels, 5)
        return np.median(background_pixels)

    def calculate_ctcf(roi_pixels_intensity, area, background_intensity):
        """Calculate CTCF with enhanced metrics"""
        integrated_density = np.sum(roi_pixels_intensity)
        ctcf_raw = integrated_density - (area * background_intensity)
        ctcf = max(0, ctcf_raw)
        mean_intensity = integrated_density / area if area > 0 else 0
        return ctcf, integrated_density, mean_intensity


# =============================================================================
# SEGMENTATION - Cellpose-SAM wrapper
# =============================================================================

class TimelapseSegmenter:
    """
    Segmenter for timelapse data using Cellpose-SAM.
    Compatible with Cellquant channel configuration.
    """
    
    def __init__(
        self,
        model_type: str = DEFAULT_CELLPOSE_MODEL,
        diameter: Optional[float] = DEFAULT_CELLPOSE_DIAMETER,
        flow_threshold: float = DEFAULT_CELLPOSE_FLOW_THRESH,
        min_size: int = DEFAULT_CELLPOSE_MIN_SIZE,
        gpu: bool = True,
    ):
        self.model_type = model_type
        self.diameter = diameter
        self.flow_threshold = flow_threshold
        self.min_size = min_size
        self.gpu = gpu
        self.model = None
        
    def initialize(self):
        """Lazy initialization of Cellpose model."""
        if self.model is not None:
            return
            
        try:
            from cellpose import models
            import torch
            
            use_gpu = self.gpu and torch.cuda.is_available()
            
            # Try CellposeModel first (supports cpsam)
            try:
                self.model = models.CellposeModel(
                    gpu=use_gpu,
                    model_type=self.model_type
                )
                print(f"✓ Initialized Cellpose model '{self.model_type}' (GPU: {use_gpu})")
            except Exception as e:
                # Fallback to standard Cellpose
                print(f"⚠ CellposeModel failed ({e}), trying standard Cellpose")
                self.model = models.Cellpose(
                    gpu=use_gpu,
                    model_type='cyto2'
                )
                print(f"✓ Initialized fallback Cellpose model (GPU: {use_gpu})")
                
        except ImportError:
            raise ImportError("Cellpose not installed. Run: pip install cellpose")
    
    def segment_frame(
        self,
        image: np.ndarray,
        channels: List[int] = [0, 0],
    ) -> np.ndarray:
        """
        Segment a single frame.
        
        Parameters
        ----------
        image : np.ndarray
            Single frame, shape (Y, X) or (C, Y, X)
        channels : list
            Cellpose channel configuration [cytoplasm, nucleus]
            
        Returns
        -------
        masks : np.ndarray
            Integer label mask
        """
        self.initialize()
        
        eval_kwargs = {
            'diameter': self.diameter,
            'flow_threshold': self.flow_threshold,
            'min_size': self.min_size,
        }
        
        # Handle channel configuration
        if image.ndim == 2:
            eval_kwargs['channels'] = [0, 0]
        elif image.ndim == 3 and image.shape[0] == 1:
            image = image[0]
            eval_kwargs['channels'] = [0, 0]
        # For multi-channel, Cellpose-SAM handles automatically
        
        masks, flows, styles = self.model.eval([image], **eval_kwargs)
        return masks[0]
    
    def segment_timelapse(
        self,
        images: np.ndarray,
        channels: List[int] = [0, 0],
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Segment all frames in a timelapse.
        
        Parameters
        ----------
        images : np.ndarray
            Timelapse stack, shape (T, Y, X) or (T, C, Y, X)
            
        Returns
        -------
        masks : np.ndarray
            Shape (T, Y, X) with integer labels
        """
        self.initialize()
        
        n_frames = images.shape[0]
        
        # Determine output shape
        if images.ndim == 4:
            mask_shape = (n_frames, images.shape[2], images.shape[3])
        else:
            mask_shape = images.shape
            
        masks = np.zeros(mask_shape, dtype=np.uint16)
        
        iterator = tqdm(range(n_frames), desc="Segmenting frames") if show_progress else range(n_frames)
        
        for t in iterator:
            frame = images[t]
            masks[t] = self.segment_frame(frame, channels)
            
        return masks


# =============================================================================
# TRACKING - btrack integration
# =============================================================================

class TimelapseTracker:
    """
    Cell tracker using btrack for division-aware tracking.
    """
    
    def __init__(
        self,
        max_search_radius: float = 50,
        track_sigma: float = 15.0,
        lambda_time: float = 5.0,
        lambda_dist: float = 3.0,
        lambda_link: float = 10.0,
        lambda_branch: float = 50.0,  # Controls division detection sensitivity
    ):
        self.max_search_radius = max_search_radius
        self.track_sigma = track_sigma
        self.lambda_time = lambda_time
        self.lambda_dist = lambda_dist
        self.lambda_link = lambda_link
        self.lambda_branch = lambda_branch
        
    def track(
        self,
        masks: np.ndarray,
        intensity_images: Optional[np.ndarray] = None,
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Track cells across frames with division detection.
        
        Parameters
        ----------
        masks : np.ndarray
            Label masks from segmentation (T, Y, X)
        intensity_images : np.ndarray, optional
            Original intensity images for measuring fluorescence
            
        Returns
        -------
        tracks_df : pd.DataFrame
            Tracking results with Cellquant-compatible columns
        lineage_tree : dict
            Dictionary mapping track_id to parent_id
        """
        try:
            import btrack
            from btrack import BayesianTracker
            from btrack.utils import segmentation_to_objects
        except ImportError:
            print("⚠ btrack not installed, using simple centroid tracking")
            return self._simple_centroid_tracking(masks, intensity_images)
        
        print("Converting segmentation masks to trackable objects...")
        
        # Convert masks to btrack objects
        properties = ['centroid', 'area', 'label']
        if intensity_images is not None:
            properties.append('mean_intensity')
            
        objects = segmentation_to_objects(
            masks,
            intensity_image=intensity_images,
            properties=tuple(properties),
        )
        
        print(f"Found {len(objects)} objects across {masks.shape[0]} frames")
        
        if len(objects) == 0:
            return pd.DataFrame(), {}
        
        # Configure and run tracker
        with BayesianTracker() as tracker:
            # Use cell config as base
            tracker.configure_from_file(btrack.config.cell_config())
            
            # Adjust parameters
            tracker.max_search_radius = self.max_search_radius
            
            # Add objects and track
            tracker.append(objects)
            
            print("Running Bayesian tracking...")
            tracker.track()
            
            print("Optimizing (resolving divisions, merges)...")
            tracker.optimize()
            
            tracks = tracker.tracks
        
        # Convert to DataFrame (Cellquant-compatible format)
        return self._tracks_to_dataframe(tracks, masks.shape[0])
    
    def _tracks_to_dataframe(
        self,
        tracks,
        n_frames: int,
    ) -> Tuple[pd.DataFrame, Dict]:
        """Convert btrack tracks to Cellquant-compatible DataFrame."""
        
        tracks_data = []
        lineage_tree = {}
        
        for track in tracks:
            track_id = track.ID
            parent_id = track.parent if hasattr(track, 'parent') else None
            root_id = track.root if hasattr(track, 'root') else track_id
            generation = track.generation if hasattr(track, 'generation') else 0
            
            if parent_id is not None and parent_id != track_id:
                lineage_tree[track_id] = parent_id
            
            for i, t in enumerate(track.t):
                row = {
                    'TrackID': track_id,
                    'Frame': int(t),
                    'CentroidX': track.x[i],
                    'CentroidY': track.y[i],
                    'ParentID': parent_id if parent_id != track_id else None,
                    'RootID': root_id,
                    'Generation': generation,
                    'IsDivision': (parent_id is not None and parent_id != track_id and i == 0),
                }
                
                # Add area and intensity if available
                if hasattr(track, 'properties'):
                    if 'area' in track.properties:
                        row['Area'] = track.properties['area'][i]
                    if 'mean_intensity' in track.properties:
                        row['MeanIntensity'] = track.properties['mean_intensity'][i]
                    if 'label' in track.properties:
                        row['SegmentationLabel'] = track.properties['label'][i]
                        
                tracks_data.append(row)
        
        tracks_df = pd.DataFrame(tracks_data)
        
        if len(tracks_df) > 0:
            tracks_df = tracks_df.sort_values(['TrackID', 'Frame']).reset_index(drop=True)
            
        print(f"✓ Tracking complete: {tracks_df['TrackID'].nunique()} tracks, "
              f"{len(lineage_tree)} division events")
        
        return tracks_df, lineage_tree
    
    def _simple_centroid_tracking(
        self,
        masks: np.ndarray,
        intensity_images: Optional[np.ndarray] = None,
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Simple centroid-based tracking fallback when btrack is unavailable.
        Uses Hungarian algorithm for frame-to-frame matching.
        """
        from scipy.optimize import linear_sum_assignment
        
        print("Using simple centroid tracking (btrack unavailable)...")
        
        n_frames = masks.shape[0]
        all_data = []
        next_track_id = 0
        prev_centroids = {}  # track_id -> (y, x)
        
        for t in tqdm(range(n_frames), desc="Tracking"):
            frame_mask = masks[t]
            
            if frame_mask.max() == 0:
                continue
                
            # Get properties for current frame
            props = regionprops_table(
                frame_mask,
                intensity_image=intensity_images[t] if intensity_images is not None else None,
                properties=['label', 'centroid', 'area'] + 
                          (['mean_intensity'] if intensity_images is not None else [])
            )
            
            current_labels = props['label']
            current_centroids = np.column_stack([props['centroid-0'], props['centroid-1']])
            
            if t == 0 or len(prev_centroids) == 0:
                # First frame or no previous tracks - create new tracks
                for i, label in enumerate(current_labels):
                    track_id = next_track_id
                    next_track_id += 1
                    
                    row = {
                        'TrackID': track_id,
                        'Frame': t,
                        'CentroidX': current_centroids[i, 1],
                        'CentroidY': current_centroids[i, 0],
                        'Area': props['area'][i],
                        'SegmentationLabel': label,
                        'ParentID': None,
                        'RootID': track_id,
                        'Generation': 0,
                        'IsDivision': False,
                    }
                    if intensity_images is not None:
                        row['MeanIntensity'] = props['mean_intensity'][i]
                    
                    all_data.append(row)
                    prev_centroids[track_id] = current_centroids[i]
            else:
                # Match current cells to previous tracks
                prev_ids = list(prev_centroids.keys())
                prev_positions = np.array([prev_centroids[tid] for tid in prev_ids])
                
                # Compute cost matrix (Euclidean distance)
                cost_matrix = np.zeros((len(current_centroids), len(prev_positions)))
                for i, curr_pos in enumerate(current_centroids):
                    for j, prev_pos in enumerate(prev_positions):
                        dist = np.sqrt(np.sum((curr_pos - prev_pos) ** 2))
                        cost_matrix[i, j] = dist if dist < self.max_search_radius else 1e6
                
                # Hungarian algorithm
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                
                matched_current = set()
                matched_prev = set()
                new_prev_centroids = {}
                
                for i, j in zip(row_ind, col_ind):
                    if cost_matrix[i, j] < self.max_search_radius:
                        track_id = prev_ids[j]
                        matched_current.add(i)
                        matched_prev.add(j)
                        
                        row = {
                            'TrackID': track_id,
                            'Frame': t,
                            'CentroidX': current_centroids[i, 1],
                            'CentroidY': current_centroids[i, 0],
                            'Area': props['area'][i],
                            'SegmentationLabel': current_labels[i],
                            'ParentID': None,
                            'RootID': track_id,
                            'Generation': 0,
                            'IsDivision': False,
                        }
                        if intensity_images is not None:
                            row['MeanIntensity'] = props['mean_intensity'][i]
                        
                        all_data.append(row)
                        new_prev_centroids[track_id] = current_centroids[i]
                
                # Unmatched current cells -> new tracks
                for i in range(len(current_centroids)):
                    if i not in matched_current:
                        track_id = next_track_id
                        next_track_id += 1
                        
                        row = {
                            'TrackID': track_id,
                            'Frame': t,
                            'CentroidX': current_centroids[i, 1],
                            'CentroidY': current_centroids[i, 0],
                            'Area': props['area'][i],
                            'SegmentationLabel': current_labels[i],
                            'ParentID': None,
                            'RootID': track_id,
                            'Generation': 0,
                            'IsDivision': False,
                        }
                        if intensity_images is not None:
                            row['MeanIntensity'] = props['mean_intensity'][i]
                        
                        all_data.append(row)
                        new_prev_centroids[track_id] = current_centroids[i]
                
                prev_centroids = new_prev_centroids
        
        tracks_df = pd.DataFrame(all_data)
        if len(tracks_df) > 0:
            tracks_df = tracks_df.sort_values(['TrackID', 'Frame']).reset_index(drop=True)
            
        print(f"✓ Simple tracking complete: {tracks_df['TrackID'].nunique()} tracks")
        return tracks_df, {}


# =============================================================================
# CTCF EXTRACTION - Compatible with Cellquant output format
# =============================================================================

class TimelapseCTCFExtractor:
    """
    Extract CTCF values for tracked cells across timelapse.
    Output format matches Cellquant_v1 CSV structure.
    """
    
    def __init__(
        self,
        background_method: str = 'median',
    ):
        self.background_method = background_method
        
    def extract_ctcf_for_tracks(
        self,
        tracks_df: pd.DataFrame,
        masks: np.ndarray,
        marker_images: Dict[str, np.ndarray],
        marker_names: List[str],
        condition_name: str = "Condition",
    ) -> pd.DataFrame:
        """
        Extract CTCF values for all tracked cells at all timepoints.
        
        Parameters
        ----------
        tracks_df : pd.DataFrame
            Tracking results with TrackID, Frame, SegmentationLabel columns
        masks : np.ndarray
            Segmentation masks (T, Y, X)
        marker_images : dict
            Dictionary mapping marker suffix to timelapse array (T, Y, X)
        marker_names : list
            List of marker names corresponding to marker_images keys
        condition_name : str
            Name for the Condition column
            
        Returns
        -------
        ctcf_df : pd.DataFrame
            DataFrame with Cellquant-compatible format
        """
        
        results = []
        n_frames = masks.shape[0]
        
        # Get unique frames from tracks
        frames_in_tracks = tracks_df['Frame'].unique()
        
        for frame in tqdm(sorted(frames_in_tracks), desc="Extracting CTCF"):
            frame_mask = masks[frame]
            frame_tracks = tracks_df[tracks_df['Frame'] == frame]
            
            if frame_mask.max() == 0 or len(frame_tracks) == 0:
                continue
            
            # Process each marker
            marker_data = {}
            for marker_suffix, marker_name in zip(marker_images.keys(), marker_names):
                marker_stack = marker_images[marker_suffix]
                
                if frame >= len(marker_stack):
                    continue
                    
                marker_frame = marker_stack[frame]
                background = estimate_background_enhanced(
                    marker_frame, frame_mask, self.background_method
                )
                
                marker_data[marker_name] = {
                    'image': marker_frame,
                    'background': background,
                }
            
            # Process each tracked cell in this frame
            for _, track_row in frame_tracks.iterrows():
                track_id = track_row['TrackID']
                seg_label = track_row.get('SegmentationLabel', None)
                
                # Find cell mask - either by label or by centroid
                if seg_label is not None and seg_label in np.unique(frame_mask):
                    cell_mask = frame_mask == seg_label
                else:
                    # Fallback: find nearest cell to centroid
                    y, x = int(track_row['CentroidY']), int(track_row['CentroidX'])
                    if 0 <= y < frame_mask.shape[0] and 0 <= x < frame_mask.shape[1]:
                        seg_label = frame_mask[y, x]
                        if seg_label > 0:
                            cell_mask = frame_mask == seg_label
                        else:
                            continue
                    else:
                        continue
                
                # Get cell coordinates
                cell_coords = np.where(cell_mask)
                area = len(cell_coords[0])
                
                if area == 0:
                    continue
                
                # Base row data (Cellquant-compatible)
                row = {
                    'Condition': condition_name,
                    'Frame': frame,
                    'TrackID': track_id,
                    'CellID': seg_label,  # Cellquant uses CellID
                    'Area': area,
                    'CentroidX': track_row['CentroidX'],
                    'CentroidY': track_row['CentroidY'],
                    'ParentID': track_row.get('ParentID'),
                    'RootID': track_row.get('RootID'),
                    'Generation': track_row.get('Generation', 0),
                    'IsDivision': track_row.get('IsDivision', False),
                }
                
                # Calculate CTCF for each marker
                for marker_name, mdata in marker_data.items():
                    roi_pixels = mdata['image'][cell_coords]
                    background = mdata['background']
                    
                    ctcf, integrated_density, mean_intensity = calculate_ctcf(
                        roi_pixels, area, background
                    )
                    
                    # Cellquant column naming convention
                    row[f'{marker_name}_CTCF'] = ctcf
                    row[f'{marker_name}_IntegratedDensity'] = integrated_density
                    row[f'{marker_name}_MeanIntensity'] = mean_intensity
                    row[f'{marker_name}_Background'] = background
                
                results.append(row)
        
        ctcf_df = pd.DataFrame(results)
        
        if len(ctcf_df) > 0:
            ctcf_df = ctcf_df.sort_values(['TrackID', 'Frame']).reset_index(drop=True)
            
        return ctcf_df


# =============================================================================
# MAIN PIPELINE - Timelapse version of Cellquant workflow
# =============================================================================

class CellquantTimelapse:
    """
    Main pipeline for timelapse CTCF analysis.
    
    Maintains compatibility with Cellquant_v1:
    - Same channel configuration
    - Same output CSV format
    - Same CTCF calculation
    - Adds: tracking, lineage, per-cell profiles
    
    Example usage:
    
        pipeline = CellquantTimelapse(
            timelapse_path='p53mStrawberry_0Gy.tif',
            output_dir='results',
            condition_name='0Gy',
        )
        
        # Configure channels (same as Cellquant GUI)
        pipeline.configure_channels(
            nuclear_suffix='C0',
            cyto_suffix='C1',
            marker_suffixes=['C2'],
            marker_names=['p53-mStrawberry'],
        )
        
        # Run
        results = pipeline.run()
    """
    
    def __init__(
        self,
        timelapse_path: Union[str, Path],
        output_dir: Union[str, Path] = 'output',
        condition_name: str = 'Condition',
        # Cellpose parameters (matching Cellquant defaults)
        cellpose_model: str = DEFAULT_CELLPOSE_MODEL,
        cellpose_diameter: Optional[float] = DEFAULT_CELLPOSE_DIAMETER,
        cellpose_flow_thresh: float = DEFAULT_CELLPOSE_FLOW_THRESH,
        cellpose_min_size: int = DEFAULT_CELLPOSE_MIN_SIZE,
        gpu: bool = True,
        # Tracking parameters
        max_search_radius: float = 50,
    ):
        self.timelapse_path = Path(timelapse_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.condition_name = condition_name
        
        # Initialize components
        self.segmenter = TimelapseSegmenter(
            model_type=cellpose_model,
            diameter=cellpose_diameter,
            flow_threshold=cellpose_flow_thresh,
            min_size=cellpose_min_size,
            gpu=gpu,
        )
        
        self.tracker = TimelapseTracker(
            max_search_radius=max_search_radius,
        )
        
        self.ctcf_extractor = TimelapseCTCFExtractor()
        
        # Channel configuration (set via configure_channels)
        self.channel_config = None
        
        # Data containers
        self.images = None
        self.masks = None
        self.tracks_df = None
        self.lineage_tree = None
        self.ctcf_df = None
        
    def configure_channels(
        self,
        nuclear_suffix: str = 'C0',
        cyto_suffix: str = 'C1',
        marker_suffixes: List[str] = ['C2'],
        marker_names: List[str] = ['Marker1'],
        use_nuc_seg: bool = True,
        use_cyto_seg: bool = True,
    ):
        """
        Configure channel settings (same as Cellquant GUI).
        
        For single-channel timelapse (already the marker), use:
            marker_suffixes=[''], marker_names=['p53-mStrawberry']
        """
        self.channel_config = {
            'nuclear_suffix': nuclear_suffix,
            'cyto_suffix': cyto_suffix,
            'marker_suffixes': marker_suffixes,
            'marker_names': marker_names,
            'use_nuc_seg': use_nuc_seg,
            'use_cyto_seg': use_cyto_seg,
        }
        print(f"✓ Channel configuration set: {len(marker_names)} marker(s)")
        
    def load_timelapse(
        self,
        channel: Optional[int] = None,
    ) -> np.ndarray:
        """
        Load timelapse from file or folder.
        
        Parameters
        ----------
        channel : int, optional
            For multi-channel data, which channel to extract
        """
        print(f"Loading timelapse from {self.timelapse_path}...")
        
        if self.timelapse_path.is_dir():
            # Load from folder of images
            image_files = sorted(
                list(self.timelapse_path.glob('*.tif')) + 
                list(self.timelapse_path.glob('*.tiff')) +
                list(self.timelapse_path.glob('*.png'))
            )
            if not image_files:
                raise ValueError(f"No image files found in {self.timelapse_path}")
            self.images = np.stack([load_image(f) for f in image_files])
        else:
            # Load from single stack
            self.images = io.imread(str(self.timelapse_path))
        
        # Handle dimensions
        if self.images.ndim == 4:
            if channel is not None:
                # Extract specific channel
                if self.images.shape[-1] < self.images.shape[1]:
                    self.images = self.images[..., channel]
                else:
                    self.images = self.images[:, channel]
            else:
                print(f"⚠ Multi-channel timelapse: {self.images.shape}")
                print("  Specify channel parameter to extract single channel")
        
        print(f"✓ Loaded: {self.images.shape}, dtype={self.images.dtype}")
        return self.images
    
    def load_separate_channels(
        self,
        channel_paths: Dict[str, Union[str, Path]],
    ) -> Dict[str, np.ndarray]:
        """
        Load multiple channel timelapses separately.
        
        Parameters
        ----------
        channel_paths : dict
            Mapping of channel suffix to file path
            e.g., {'C0': 'nuclear.tif', 'C1': 'cyto.tif', 'C2': 'marker.tif'}
        """
        self.channel_images = {}
        
        for suffix, path in channel_paths.items():
            print(f"Loading channel {suffix} from {path}...")
            img = io.imread(str(path))
            self.channel_images[suffix] = img
            print(f"  Shape: {img.shape}")
        
        # Use first channel as reference for segmentation
        first_key = list(self.channel_images.keys())[0]
        self.images = self.channel_images[first_key]
        
        return self.channel_images
    
    def segment(self) -> np.ndarray:
        """Run segmentation on all frames."""
        if self.images is None:
            raise ValueError("Load timelapse first")
            
        print("\nRunning Cellpose segmentation...")
        
        # Prepare input based on channel config
        if hasattr(self, 'channel_images') and self.channel_config:
            # Multi-channel input
            seg_channels = []
            if self.channel_config['use_nuc_seg']:
                nuc_suffix = self.channel_config['nuclear_suffix']
                if nuc_suffix in self.channel_images:
                    seg_channels.append(normalize_image(self.channel_images[nuc_suffix]))
            if self.channel_config['use_cyto_seg']:
                cyto_suffix = self.channel_config['cyto_suffix']
                if cyto_suffix in self.channel_images:
                    seg_channels.append(normalize_image(self.channel_images[cyto_suffix]))
            
            if seg_channels:
                # Stack channels: (T, C, Y, X)
                seg_input = np.stack([normalize_image(c) for c in seg_channels], axis=1)
            else:
                seg_input = np.stack([normalize_image(f) for f in self.images])
        else:
            # Single channel
            seg_input = np.stack([normalize_image(f) for f in self.images])
        
        self.masks = self.segmenter.segment_timelapse(seg_input)
        
        # Save masks
        mask_path = self.output_dir / f'{self.condition_name}_masks.tif'
        io.imsave(str(mask_path), self.masks.astype(np.uint16), check_contrast=False)
        print(f"✓ Saved masks to {mask_path}")
        
        return self.masks
    
    def track(self) -> pd.DataFrame:
        """Track cells across frames."""
        if self.masks is None:
            raise ValueError("Run segmentation first")
            
        print("\nRunning cell tracking...")
        
        self.tracks_df, self.lineage_tree = self.tracker.track(
            self.masks,
            intensity_images=self.images,
        )
        
        # Save tracks
        tracks_path = self.output_dir / f'{self.condition_name}_tracks.csv'
        self.tracks_df.to_csv(tracks_path, index=False)
        print(f"✓ Saved tracks to {tracks_path}")
        
        # Save lineage
        if self.lineage_tree:
            lineage_path = self.output_dir / f'{self.condition_name}_lineage.csv'
            lineage_df = pd.DataFrame([
                {'DaughterID': k, 'ParentID': v}
                for k, v in self.lineage_tree.items()
            ])
            lineage_df.to_csv(lineage_path, index=False)
            print(f"✓ Saved lineage to {lineage_path}")
        
        return self.tracks_df
    
    def extract_ctcf(self) -> pd.DataFrame:
        """Extract CTCF values for all tracked cells."""
        if self.tracks_df is None:
            raise ValueError("Run tracking first")
        if self.channel_config is None:
            raise ValueError("Configure channels first")
            
        print("\nExtracting CTCF values...")
        
        # Prepare marker images
        if hasattr(self, 'channel_images'):
            marker_images = {
                suffix: self.channel_images[suffix]
                for suffix in self.channel_config['marker_suffixes']
                if suffix in self.channel_images
            }
        else:
            # Single channel case - use main images as marker
            marker_images = {'': self.images}
        
        self.ctcf_df = self.ctcf_extractor.extract_ctcf_for_tracks(
            self.tracks_df,
            self.masks,
            marker_images,
            self.channel_config['marker_names'],
            self.condition_name,
        )
        
        # Save CTCF results (Cellquant-compatible format)
        ctcf_path = self.output_dir / f'ctcf_analysis_timelapse_{self.condition_name}.csv'
        self.ctcf_df.to_csv(ctcf_path, index=False)
        print(f"✓ Saved CTCF data to {ctcf_path}")
        
        return self.ctcf_df
    
    def generate_plots(self):
        """Generate analysis plots."""
        if self.ctcf_df is None or len(self.ctcf_df) == 0:
            print("No CTCF data for plotting")
            return
            
        print("\nGenerating plots...")
        
        # Get marker columns
        marker_names = self.channel_config['marker_names']
        
        for marker_name in marker_names:
            intensity_col = f'{marker_name}_MeanIntensity'
            ctcf_col = f'{marker_name}_CTCF'
            
            if intensity_col not in self.ctcf_df.columns:
                continue
            
            # 1. Intensity traces
            fig, ax = plt.subplots(figsize=(12, 6))
            for track_id, track_data in self.ctcf_df.groupby('TrackID'):
                track_data = track_data.sort_values('Frame')
                ax.plot(track_data['Frame'], track_data[intensity_col], alpha=0.5, linewidth=1)
                
                # Mark divisions
                divisions = track_data[track_data['IsDivision'] == True]
                if len(divisions) > 0:
                    ax.scatter(
                        divisions['Frame'], 
                        divisions[intensity_col],
                        c='red', s=50, marker='*', zorder=5
                    )
            
            ax.set_xlabel('Frame')
            ax.set_ylabel(f'{marker_name} Mean Intensity')
            ax.set_title(f'{self.condition_name} - {marker_name} Intensity Profiles\n'
                        f'({self.ctcf_df["TrackID"].nunique()} tracks, red stars = divisions)')
            
            trace_path = self.output_dir / f'{self.condition_name}_{marker_name}_traces.png'
            plt.savefig(str(trace_path), dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved {trace_path.name}")
            
            # 2. Heatmap
            pivot_df = self.ctcf_df.pivot_table(
                index='TrackID',
                columns='Frame',
                values=intensity_col,
                aggfunc='mean'
            )
            
            if len(pivot_df) > 0:
                # Sort by mean intensity
                pivot_df = pivot_df.loc[pivot_df.mean(axis=1).sort_values(ascending=False).index]
                
                fig, ax = plt.subplots(figsize=(14, 8))
                sns.heatmap(pivot_df, cmap='viridis', ax=ax, cbar_kws={'label': intensity_col})
                ax.set_xlabel('Frame')
                ax.set_ylabel(f'Cell Track (n={len(pivot_df)})')
                ax.set_title(f'{self.condition_name} - {marker_name} Intensity Heatmap')
                
                heatmap_path = self.output_dir / f'{self.condition_name}_{marker_name}_heatmap.png'
                plt.savefig(str(heatmap_path), dpi=150, bbox_inches='tight')
                plt.close()
                print(f"  Saved {heatmap_path.name}")
        
        # 3. Cell count over time
        fig, ax = plt.subplots(figsize=(10, 5))
        cells_per_frame = self.ctcf_df.groupby('Frame')['TrackID'].nunique()
        ax.plot(cells_per_frame.index, cells_per_frame.values, 'b-', linewidth=2)
        ax.set_xlabel('Frame')
        ax.set_ylabel('Number of Cells')
        ax.set_title(f'{self.condition_name} - Cell Count Over Time')
        
        count_path = self.output_dir / f'{self.condition_name}_cell_count.png'
        plt.savefig(str(count_path), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved {count_path.name}")
        
        # 4. Division timeline
        if self.lineage_tree:
            divisions = self.ctcf_df[self.ctcf_df['IsDivision'] == True]
            if len(divisions) > 0:
                fig, ax = plt.subplots(figsize=(10, 5))
                division_counts = divisions.groupby('Frame').size()
                ax.bar(division_counts.index, division_counts.values, color='red', alpha=0.7)
                ax.set_xlabel('Frame')
                ax.set_ylabel('Number of Divisions')
                ax.set_title(f'{self.condition_name} - Division Events Over Time')
                
                div_path = self.output_dir / f'{self.condition_name}_divisions.png'
                plt.savefig(str(div_path), dpi=150, bbox_inches='tight')
                plt.close()
                print(f"  Saved {div_path.name}")
    
    def run(
        self,
        channel: Optional[int] = None,
        generate_plots: bool = True,
    ) -> Dict:
        """
        Run complete pipeline.
        
        Parameters
        ----------
        channel : int, optional
            For multi-channel input, which channel to use for segmentation
        generate_plots : bool
            Whether to generate analysis plots
        """
        # Set default channel config if not set
        if self.channel_config is None:
            self.configure_channels(
                marker_suffixes=[''],
                marker_names=['Intensity'],
            )
        
        self.load_timelapse(channel=channel)
        self.segment()
        self.track()
        self.extract_ctcf()
        
        if generate_plots:
            self.generate_plots()
        
        print(f"\n{'='*60}")
        print(f"✓ Pipeline complete for {self.condition_name}")
        print(f"  Frames: {len(self.images)}")
        print(f"  Cells tracked: {self.ctcf_df['TrackID'].nunique()}")
        print(f"  Division events: {len(self.lineage_tree)}")
        print(f"  Output: {self.output_dir}")
        print(f"{'='*60}")
        
        return {
            'images': self.images,
            'masks': self.masks,
            'tracks': self.tracks_df,
            'lineage': self.lineage_tree,
            'ctcf': self.ctcf_df,
        }


# =============================================================================
# BATCH PROCESSING - Multiple conditions (like Cellquant GUI)
# =============================================================================

def batch_process_timelapse_experiment(
    conditions: Dict[str, str],
    output_dir: str = 'batch_results',
    marker_names: List[str] = ['p53-mStrawberry'],
    **pipeline_kwargs,
) -> pd.DataFrame:
    """
    Process multiple conditions (e.g., radiation doses).
    
    Parameters
    ----------
    conditions : dict
        Mapping of condition name to timelapse file path
        e.g., {'0Gy': 'data/0Gy.tif', '2Gy': 'data/2Gy.tif', ...}
    output_dir : str
        Base output directory
    marker_names : list
        Names of markers being quantified
    **pipeline_kwargs
        Additional arguments passed to CellquantTimelapse
        
    Returns
    -------
    combined_df : pd.DataFrame
        Combined CTCF data from all conditions
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    for condition_name, timelapse_path in conditions.items():
        print(f"\n{'#'*60}")
        print(f"# Processing: {condition_name}")
        print(f"{'#'*60}")
        
        if not Path(timelapse_path).exists():
            print(f"⚠ File not found: {timelapse_path}, skipping")
            continue
        
        try:
            pipeline = CellquantTimelapse(
                timelapse_path=timelapse_path,
                output_dir=str(output_dir / condition_name),
                condition_name=condition_name,
                **pipeline_kwargs,
            )
            
            pipeline.configure_channels(
                marker_suffixes=[''],
                marker_names=marker_names,
            )
            
            results = pipeline.run()
            
            if results['ctcf'] is not None and len(results['ctcf']) > 0:
                all_results.append(results['ctcf'])
                
        except Exception as e:
            print(f"✗ Error processing {condition_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not all_results:
        print("No results to combine")
        return pd.DataFrame()
    
    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Save combined results
    combined_path = output_dir / 'all_conditions_ctcf.csv'
    combined_df.to_csv(combined_path, index=False)
    print(f"\n✓ Saved combined results to {combined_path}")
    
    # Generate comparison plots
    _generate_comparison_plots(combined_df, output_dir, marker_names)
    
    return combined_df


def _generate_comparison_plots(
    df: pd.DataFrame,
    output_dir: Path,
    marker_names: List[str],
):
    """Generate cross-condition comparison plots."""
    
    for marker_name in marker_names:
        intensity_col = f'{marker_name}_MeanIntensity'
        
        if intensity_col not in df.columns:
            continue
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Mean intensity over time by condition
        ax = axes[0, 0]
        for condition, group in df.groupby('Condition'):
            mean_by_frame = group.groupby('Frame')[intensity_col].mean()
            ax.plot(mean_by_frame.index, mean_by_frame.values, label=condition, linewidth=2)
        ax.set_xlabel('Frame')
        ax.set_ylabel(f'Mean {marker_name} Intensity')
        ax.set_title('Intensity Dynamics by Condition')
        ax.legend()
        
        # 2. Cell count over time
        ax = axes[0, 1]
        for condition, group in df.groupby('Condition'):
            cells_by_frame = group.groupby('Frame')['TrackID'].nunique()
            ax.plot(cells_by_frame.index, cells_by_frame.values, label=condition, linewidth=2)
        ax.set_xlabel('Frame')
        ax.set_ylabel('Number of Cells')
        ax.set_title('Cell Count Over Time')
        ax.legend()
        
        # 3. Peak intensity distribution
        ax = axes[1, 0]
        peak_intensities = df.groupby(['Condition', 'TrackID'])[intensity_col].max().reset_index()
        if len(peak_intensities) > 0:
            sns.boxplot(data=peak_intensities, x='Condition', y=intensity_col, ax=ax)
            ax.set_xlabel('Condition')
            ax.set_ylabel(f'Peak {marker_name} Intensity')
            ax.set_title('Distribution of Peak Intensities')
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 4. Division events
        ax = axes[1, 1]
        divisions_per_condition = df[df['IsDivision'] == True].groupby('Condition').size()
        if len(divisions_per_condition) > 0:
            divisions_per_condition.plot(kind='bar', ax=ax, color='coral')
            ax.set_xlabel('Condition')
            ax.set_ylabel('Number of Divisions')
            ax.set_title('Division Events by Condition')
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        comparison_path = output_dir / f'comparison_{marker_name}.png'
        plt.savefig(str(comparison_path), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved comparison plot to {comparison_path}")


# =============================================================================
# NAPARI VISUALIZATION
# =============================================================================

def visualize_in_napari(
    images: np.ndarray,
    masks: np.ndarray,
    tracks_df: pd.DataFrame,
    ctcf_df: Optional[pd.DataFrame] = None,
):
    """
    Open results in Napari for interactive visualization.
    """
    try:
        import napari
    except ImportError:
        print("Napari not installed. Run: pip install napari[all]")
        return None
    
    viewer = napari.Viewer()
    
    # Add intensity images
    viewer.add_image(
        images,
        name='Intensity',
        colormap='magma',
    )
    
    # Add segmentation masks
    viewer.add_labels(
        masks,
        name='Segmentation',
        opacity=0.3,
    )
    
    # Add tracks
    if len(tracks_df) > 0:
        # Napari track format: (ID, T, Y, X)
        tracks_array = tracks_df[['TrackID', 'Frame', 'CentroidY', 'CentroidX']].values
        
        viewer.add_tracks(
            tracks_array,
            name='Tracks',
            colormap='turbo',
            tail_length=20,
        )
    
    napari.run()
    return viewer


# =============================================================================
# TKINTER GUI - Matching Cellquant v1 Style
# =============================================================================

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import queue
import os


class CellquantTimelapseGUI:
    """
    Tkinter GUI for Cellquant Timelapse Analysis.
    Matches the look and feel of Cellquant_v1.
    """
    
    def __init__(self, master):
        self.master = master
        self.master.title("Cellquant Timelapse v2 - CTCF Analysis with Tracking")
        self.master.geometry("800x900")
        self.master.minsize(700, 800)
        
        # Data storage
        self.timelapse_files = []  # List of (condition_name, file_path) tuples
        self.results = {}
        self.processing_queue = queue.Queue()
        
        # Create main UI
        self._create_widgets()
        
    def _create_widgets(self):
        """Create all GUI widgets."""
        
        # Main container with scrollbar
        main_frame = ttk.Frame(self.master, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # =====================================================================
        # SECTION 1: Timelapse Files
        # =====================================================================
        files_frame = ttk.LabelFrame(main_frame, text="Timelapse Files", padding="10")
        files_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Buttons row
        btn_frame = ttk.Frame(files_frame)
        btn_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Button(btn_frame, text="Add Timelapse File(s)", 
                   command=self._add_timelapse_files).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_frame, text="Add Folder of Timelapses", 
                   command=self._add_timelapse_folder).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_frame, text="Remove Selected", 
                   command=self._remove_selected).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_frame, text="Clear All", 
                   command=self._clear_all).pack(side=tk.LEFT)
        
        # File listbox with scrollbar
        list_frame = ttk.Frame(files_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        self.file_listbox = tk.Listbox(list_frame, height=6, selectmode=tk.EXTENDED)
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.file_listbox.yview)
        self.file_listbox.configure(yscrollcommand=scrollbar.set)
        
        self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # =====================================================================
        # SECTION 2: Output Settings
        # =====================================================================
        output_frame = ttk.LabelFrame(main_frame, text="Output Settings", padding="10")
        output_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(output_frame, text="Output Folder:").grid(row=0, column=0, sticky=tk.W)
        self.output_var = tk.StringVar(value=os.path.expanduser("~/cellquant_timelapse_output"))
        ttk.Entry(output_frame, textvariable=self.output_var, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(output_frame, text="Browse", command=self._browse_output).grid(row=0, column=2)
        
        # =====================================================================
        # SECTION 3: Channel Configuration (matching Cellquant v1)
        # =====================================================================
        channel_frame = ttk.LabelFrame(main_frame, text="Channel Configuration", padding="10")
        channel_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Row 0: Nuclear channel
        ttk.Label(channel_frame, text="Nuclear Suffix:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.nuclear_suffix_var = tk.StringVar(value="C0")
        ttk.Entry(channel_frame, textvariable=self.nuclear_suffix_var, width=15).grid(row=0, column=1, padx=5, pady=2)
        
        self.use_nuc_seg_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(channel_frame, text="Use for Segmentation", 
                        variable=self.use_nuc_seg_var).grid(row=0, column=2, padx=5)
        
        # Row 1: Cytoplasm channel
        ttk.Label(channel_frame, text="Cytoplasm Suffix:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.cyto_suffix_var = tk.StringVar(value="C1")
        ttk.Entry(channel_frame, textvariable=self.cyto_suffix_var, width=15).grid(row=1, column=1, padx=5, pady=2)
        
        self.use_cyto_seg_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(channel_frame, text="Use for Segmentation", 
                        variable=self.use_cyto_seg_var).grid(row=1, column=2, padx=5)
        
        # Row 2: Marker channels
        ttk.Label(channel_frame, text="Marker Suffix(es):").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.marker_suffix_var = tk.StringVar(value="C2")
        ttk.Entry(channel_frame, textvariable=self.marker_suffix_var, width=30).grid(row=2, column=1, columnspan=2, sticky=tk.W, padx=5, pady=2)
        ttk.Label(channel_frame, text="(comma-separated for multiple)").grid(row=2, column=3, sticky=tk.W)
        
        # Row 3: Marker names
        ttk.Label(channel_frame, text="Marker Name(s):").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.marker_names_var = tk.StringVar(value="p53-mStrawberry")
        ttk.Entry(channel_frame, textvariable=self.marker_names_var, width=30).grid(row=3, column=1, columnspan=2, sticky=tk.W, padx=5, pady=2)
        ttk.Label(channel_frame, text="(comma-separated, must match suffixes)").grid(row=3, column=3, sticky=tk.W)
        
        # Single channel mode checkbox
        self.single_channel_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(channel_frame, text="Single-channel timelapse (marker channel only)", 
                        variable=self.single_channel_var,
                        command=self._toggle_single_channel).grid(row=4, column=0, columnspan=4, sticky=tk.W, pady=5)
        
        # =====================================================================
        # SECTION 4: Cellpose Settings (matching Cellquant v1)
        # =====================================================================
        cellpose_frame = ttk.LabelFrame(main_frame, text="Cellpose Settings", padding="10")
        cellpose_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Row 0: Model selection
        ttk.Label(cellpose_frame, text="Model:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.model_var = tk.StringVar(value="cpsam")
        model_combo = ttk.Combobox(cellpose_frame, textvariable=self.model_var, width=15,
                                   values=["cpsam", "cyto3", "cyto2", "cyto", "nuclei"])
        model_combo.grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)
        
        # Row 0: Diameter
        ttk.Label(cellpose_frame, text="Diameter:").grid(row=0, column=2, sticky=tk.W, padx=(20, 0), pady=2)
        self.diameter_var = tk.DoubleVar(value=30.0)
        ttk.Spinbox(cellpose_frame, textvariable=self.diameter_var, from_=0, to=500, 
                    width=8).grid(row=0, column=3, padx=5, pady=2, sticky=tk.W)
        ttk.Label(cellpose_frame, text="(0 = auto)").grid(row=0, column=4, sticky=tk.W)
        
        # Row 1: Flow threshold
        ttk.Label(cellpose_frame, text="Flow Threshold:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.flow_thresh_var = tk.DoubleVar(value=0.4)
        ttk.Spinbox(cellpose_frame, textvariable=self.flow_thresh_var, from_=0, to=1, 
                    increment=0.1, width=8).grid(row=1, column=1, padx=5, pady=2, sticky=tk.W)
        
        # Row 1: Min size
        ttk.Label(cellpose_frame, text="Min Size:").grid(row=1, column=2, sticky=tk.W, padx=(20, 0), pady=2)
        self.min_size_var = tk.IntVar(value=15)
        ttk.Spinbox(cellpose_frame, textvariable=self.min_size_var, from_=0, to=1000, 
                    width=8).grid(row=1, column=3, padx=5, pady=2, sticky=tk.W)
        
        # =====================================================================
        # SECTION 5: Tracking Settings
        # =====================================================================
        tracking_frame = ttk.LabelFrame(main_frame, text="Tracking Settings", padding="10")
        tracking_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Row 0: Search radius
        ttk.Label(tracking_frame, text="Max Search Radius (px):").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.search_radius_var = tk.DoubleVar(value=50.0)
        ttk.Spinbox(tracking_frame, textvariable=self.search_radius_var, from_=10, to=200, 
                    width=8).grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)
        ttk.Label(tracking_frame, text="(max distance cell can move between frames)").grid(row=0, column=2, sticky=tk.W)
        
        # Row 1: Min track length
        ttk.Label(tracking_frame, text="Min Track Length:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.min_track_length_var = tk.IntVar(value=10)
        ttk.Spinbox(tracking_frame, textvariable=self.min_track_length_var, from_=1, to=100, 
                    width=8).grid(row=1, column=1, padx=5, pady=2, sticky=tk.W)
        ttk.Label(tracking_frame, text="(frames, for filtering short tracks)").grid(row=1, column=2, sticky=tk.W)
        
        # =====================================================================
        # SECTION 6: Analysis Options
        # =====================================================================
        options_frame = ttk.LabelFrame(main_frame, text="Analysis Options", padding="10")
        options_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.generate_plots_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Generate analysis plots", 
                        variable=self.generate_plots_var).grid(row=0, column=0, sticky=tk.W)
        
        self.open_napari_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="Open results in Napari viewer", 
                        variable=self.open_napari_var).grid(row=0, column=1, sticky=tk.W, padx=(20, 0))
        
        self.save_masks_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Save segmentation masks", 
                        variable=self.save_masks_var).grid(row=1, column=0, sticky=tk.W)
        
        self.use_gpu_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Use GPU (if available)", 
                        variable=self.use_gpu_var).grid(row=1, column=1, sticky=tk.W, padx=(20, 0))
        
        # =====================================================================
        # SECTION 7: Run Button and Progress
        # =====================================================================
        run_frame = ttk.Frame(main_frame)
        run_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.run_button = ttk.Button(run_frame, text="▶ Run Analysis", 
                                     command=self._run_analysis, style='Accent.TButton')
        self.run_button.pack(pady=10)
        
        # Progress bar
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(run_frame, variable=self.progress_var, 
                                            maximum=100, length=400)
        self.progress_bar.pack(pady=5)
        
        # Status label
        self.status_var = tk.StringVar(value="Ready. Add timelapse files to begin.")
        ttk.Label(run_frame, textvariable=self.status_var, 
                  font=('TkDefaultFont', 9, 'italic')).pack(pady=5)
        
        # =====================================================================
        # SECTION 8: Log Output
        # =====================================================================
        log_frame = ttk.LabelFrame(main_frame, text="Log", padding="5")
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        self.log_text = tk.Text(log_frame, height=10, wrap=tk.WORD, state=tk.DISABLED)
        log_scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Initial state
        self._toggle_single_channel()
        
    def _log(self, message: str):
        """Add message to log window."""
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)
        self.master.update_idletasks()
        
    def _toggle_single_channel(self):
        """Toggle between single and multi-channel modes."""
        if self.single_channel_var.get():
            # Single channel mode - disable nuclear/cyto settings
            self.nuclear_suffix_var.set("")
            self.cyto_suffix_var.set("")
            self.use_nuc_seg_var.set(False)
            self.use_cyto_seg_var.set(False)
        else:
            # Multi-channel mode - restore defaults
            self.nuclear_suffix_var.set("C0")
            self.cyto_suffix_var.set("C1")
            self.use_nuc_seg_var.set(True)
            self.use_cyto_seg_var.set(True)
    
    def _add_timelapse_files(self):
        """Add timelapse files via file dialog."""
        files = filedialog.askopenfilenames(
            title="Select Timelapse File(s)",
            filetypes=[
                ("TIFF files", "*.tif *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        for filepath in files:
            # Extract condition name from filename
            filename = os.path.basename(filepath)
            condition_name = os.path.splitext(filename)[0]
            
            # Check for duplicates
            existing_paths = [f[1] for f in self.timelapse_files]
            if filepath not in existing_paths:
                self.timelapse_files.append((condition_name, filepath))
                self.file_listbox.insert(tk.END, f"{condition_name}: {filepath}")
        
        self._update_status()
    
    def _add_timelapse_folder(self):
        """Add all timelapse files from a folder."""
        folder = filedialog.askdirectory(title="Select Folder Containing Timelapses")
        
        if folder:
            tif_files = list(Path(folder).glob("*.tif")) + list(Path(folder).glob("*.tiff"))
            
            for filepath in sorted(tif_files):
                condition_name = filepath.stem
                filepath_str = str(filepath)
                
                existing_paths = [f[1] for f in self.timelapse_files]
                if filepath_str not in existing_paths:
                    self.timelapse_files.append((condition_name, filepath_str))
                    self.file_listbox.insert(tk.END, f"{condition_name}: {filepath_str}")
            
            self._log(f"Added {len(tif_files)} timelapse files from {folder}")
            self._update_status()
    
    def _remove_selected(self):
        """Remove selected files from list."""
        selected = self.file_listbox.curselection()
        for idx in reversed(selected):
            self.file_listbox.delete(idx)
            del self.timelapse_files[idx]
        self._update_status()
    
    def _clear_all(self):
        """Clear all files from list."""
        self.file_listbox.delete(0, tk.END)
        self.timelapse_files = []
        self._update_status()
    
    def _browse_output(self):
        """Browse for output folder."""
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self.output_var.set(folder)
    
    def _update_status(self):
        """Update status bar."""
        n_files = len(self.timelapse_files)
        if n_files == 0:
            self.status_var.set("Ready. Add timelapse files to begin.")
        else:
            self.status_var.set(f"{n_files} timelapse file(s) loaded. Click 'Run Analysis' to process.")
    
    def _get_channel_config(self) -> dict:
        """Get channel configuration from GUI."""
        marker_suffixes = [s.strip() for s in self.marker_suffix_var.get().split(',') if s.strip()]
        marker_names = [n.strip() for n in self.marker_names_var.get().split(',') if n.strip()]
        
        # For single channel mode, use empty suffix
        if self.single_channel_var.get():
            marker_suffixes = ['']
        
        return {
            'nuclear_suffix': self.nuclear_suffix_var.get(),
            'cyto_suffix': self.cyto_suffix_var.get(),
            'marker_suffixes': marker_suffixes,
            'marker_names': marker_names,
            'use_nuc_seg': self.use_nuc_seg_var.get(),
            'use_cyto_seg': self.use_cyto_seg_var.get(),
        }
    
    def _run_analysis(self):
        """Run the analysis pipeline."""
        if not self.timelapse_files:
            messagebox.showerror("Error", "No timelapse files loaded. Please add files first.")
            return
        
        output_dir = self.output_var.get()
        if not output_dir:
            messagebox.showerror("Error", "Please select an output folder.")
            return
        
        # Validate marker configuration
        channel_config = self._get_channel_config()
        if len(channel_config['marker_suffixes']) != len(channel_config['marker_names']):
            messagebox.showerror("Error", 
                "Number of marker suffixes must match number of marker names.")
            return
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Disable run button during processing
        self.run_button.configure(state=tk.DISABLED)
        self.progress_var.set(0)
        
        # Run in separate thread to keep GUI responsive
        thread = threading.Thread(target=self._analysis_thread, args=(output_dir, channel_config))
        thread.start()
        
        # Start checking for updates
        self.master.after(100, self._check_progress)
    
    def _analysis_thread(self, output_dir: str, channel_config: dict):
        """Analysis thread - runs processing in background."""
        try:
            n_files = len(self.timelapse_files)
            all_results = []
            
            for i, (condition_name, filepath) in enumerate(self.timelapse_files):
                self.processing_queue.put(('status', f"Processing {condition_name} ({i+1}/{n_files})..."))
                self.processing_queue.put(('log', f"\n{'='*50}"))
                self.processing_queue.put(('log', f"Processing: {condition_name}"))
                self.processing_queue.put(('log', f"File: {filepath}"))
                self.processing_queue.put(('log', f"{'='*50}"))
                
                try:
                    # Create pipeline
                    pipeline = CellquantTimelapse(
                        timelapse_path=filepath,
                        output_dir=str(Path(output_dir) / condition_name),
                        condition_name=condition_name,
                        cellpose_model=self.model_var.get(),
                        cellpose_diameter=self.diameter_var.get() if self.diameter_var.get() > 0 else None,
                        cellpose_flow_thresh=self.flow_thresh_var.get(),
                        cellpose_min_size=self.min_size_var.get(),
                        gpu=self.use_gpu_var.get(),
                        max_search_radius=self.search_radius_var.get(),
                    )
                    
                    # Configure channels
                    pipeline.configure_channels(**channel_config)
                    
                    # Run pipeline
                    results = pipeline.run(generate_plots=self.generate_plots_var.get())
                    
                    if results['ctcf'] is not None and len(results['ctcf']) > 0:
                        all_results.append(results['ctcf'])
                        self.processing_queue.put(('log', 
                            f"✓ {condition_name}: {results['ctcf']['TrackID'].nunique()} tracks, "
                            f"{len(results['lineage'])} divisions"))
                    
                    # Open in Napari if requested (only for last file)
                    if self.open_napari_var.get() and i == n_files - 1:
                        self.processing_queue.put(('log', "Opening Napari viewer..."))
                        visualize_in_napari(
                            results['images'],
                            results['masks'],
                            results['tracks'],
                            results['ctcf'],
                        )
                    
                except Exception as e:
                    self.processing_queue.put(('log', f"✗ Error processing {condition_name}: {e}"))
                    import traceback
                    self.processing_queue.put(('log', traceback.format_exc()))
                
                # Update progress
                progress = ((i + 1) / n_files) * 100
                self.processing_queue.put(('progress', progress))
            
            # Save combined results
            if all_results:
                combined_df = pd.concat(all_results, ignore_index=True)
                combined_path = Path(output_dir) / 'all_conditions_ctcf.csv'
                combined_df.to_csv(combined_path, index=False)
                self.processing_queue.put(('log', f"\n✓ Saved combined results to {combined_path}"))
                
                # Generate comparison plots
                if self.generate_plots_var.get() and len(all_results) > 1:
                    self.processing_queue.put(('log', "Generating comparison plots..."))
                    _generate_comparison_plots(
                        combined_df, 
                        Path(output_dir),
                        channel_config['marker_names']
                    )
            
            self.processing_queue.put(('done', f"Analysis complete! Results saved to {output_dir}"))
            
        except Exception as e:
            self.processing_queue.put(('error', str(e)))
    
    def _check_progress(self):
        """Check for updates from processing thread."""
        try:
            while True:
                msg_type, msg_data = self.processing_queue.get_nowait()
                
                if msg_type == 'status':
                    self.status_var.set(msg_data)
                elif msg_type == 'progress':
                    self.progress_var.set(msg_data)
                elif msg_type == 'log':
                    self._log(msg_data)
                elif msg_type == 'done':
                    self.status_var.set(msg_data)
                    self.progress_var.set(100)
                    self.run_button.configure(state=tk.NORMAL)
                    messagebox.showinfo("Complete", msg_data)
                    return
                elif msg_type == 'error':
                    self.status_var.set(f"Error: {msg_data}")
                    self.run_button.configure(state=tk.NORMAL)
                    messagebox.showerror("Error", msg_data)
                    return
                    
        except queue.Empty:
            pass
        
        # Continue checking
        self.master.after(100, self._check_progress)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    import sys
    
    # Check if CLI arguments provided
    if len(sys.argv) > 1 and not sys.argv[1].startswith('-h'):
        # CLI mode
        import argparse
        
        parser = argparse.ArgumentParser(
            description='Cellquant Timelapse - CTCF analysis with tracking'
        )
        parser.add_argument(
            'input',
            type=str,
            help='Path to timelapse file or folder'
        )
        parser.add_argument(
            '-o', '--output',
            type=str,
            default='output',
            help='Output directory'
        )
        parser.add_argument(
            '-n', '--name',
            type=str,
            default='Condition',
            help='Condition name'
        )
        parser.add_argument(
            '--marker',
            type=str,
            default='Marker',
            help='Marker name for quantification'
        )
        parser.add_argument(
            '--diameter',
            type=float,
            default=DEFAULT_CELLPOSE_DIAMETER,
            help='Cell diameter for Cellpose'
        )
        parser.add_argument(
            '--model',
            type=str,
            default=DEFAULT_CELLPOSE_MODEL,
            help='Cellpose model type (default: cpsam)'
        )
        parser.add_argument(
            '--no-gpu',
            action='store_true',
            help='Disable GPU'
        )
        parser.add_argument(
            '--search-radius',
            type=float,
            default=50,
            help='Max tracking search radius'
        )
        parser.add_argument(
            '--channel',
            type=int,
            default=None,
            help='Channel to extract from multi-channel data'
        )
        parser.add_argument(
            '--napari',
            action='store_true',
            help='Open results in Napari viewer'
        )
        
        args = parser.parse_args()
        
        pipeline = CellquantTimelapse(
            timelapse_path=args.input,
            output_dir=args.output,
            condition_name=args.name,
            cellpose_model=args.model,
            cellpose_diameter=args.diameter,
            gpu=not args.no_gpu,
            max_search_radius=args.search_radius,
        )
        
        pipeline.configure_channels(
            marker_suffixes=[''],
            marker_names=[args.marker],
        )
        
        results = pipeline.run(channel=args.channel)
        
        if args.napari:
            visualize_in_napari(
                results['images'],
                results['masks'],
                results['tracks'],
                results['ctcf'],
            )
    else:
        # GUI mode (default)
        root = tk.Tk()
        app = CellquantTimelapseGUI(root)
        root.mainloop()