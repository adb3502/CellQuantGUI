# Core Analysis Pipeline
# Handles image processing, segmentation, and quantification

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Image processing imports
try:
    from skimage import io, measure, filters, morphology, segmentation
    from skimage.feature import peak_local_maxima
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    logging.warning("scikit-image not available. Install with: pip install scikit-image")

try:
    from cellpose import models, utils
    CELLPOSE_AVAILABLE = True
except ImportError:
    CELLPOSE_AVAILABLE = False
    logging.warning("Cellpose not available. Install with: pip install cellpose")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logging.warning("PIL not available. Install with: pip install pillow")

class AnalysisPipeline:
    """Core analysis pipeline for quantitative microscopy"""
    
    def __init__(self, experiment_config: Dict, progress_callback=None):
        self.config = experiment_config
        self.progress_callback = progress_callback
        self.logger = logging.getLogger(__name__)
        
        # Analysis results storage
        self.results = {
            'experiment_info': {},
            'conditions': {},
            'summary_statistics': {},
            'processing_log': []
        }
        
        # Initialize Cellpose model
        self.cellpose_model = None
        self._initialize_cellpose()
        
        # Thread safety
        self.lock = threading.Lock()
        
    def _initialize_cellpose(self):
        """Initialize Cellpose model"""
        if not CELLPOSE_AVAILABLE:
            raise ImportError("Cellpose is required for segmentation")
        
        try:
            model_type = self.config.get('cellpose_model', 'cyto2')
            gpu = self.config.get('use_gpu', False)
            
            self.cellpose_model = models.Cellpose(gpu=gpu, model_type=model_type)
            self.logger.info(f"Initialized Cellpose model: {model_type} (GPU: {gpu})")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Cellpose: {e}")
            raise
    
    def run_analysis(self) -> Dict:
        """Run the complete analysis pipeline"""
        
        self.logger.info("Starting analysis pipeline")
        start_time = datetime.now()
        
        try:
            # Phase 1: Setup and validation
            self._update_progress("Validating configuration...", 0)
            self._validate_configuration()
            
            # Phase 2: Process each condition
            total_conditions = len(self.config['conditions'])
            
            for i, condition in enumerate(self.config['conditions']):
                condition_progress = int((i / total_conditions) * 80)  # 80% for condition processing
                
                self._update_progress(f"Processing condition: {condition['name']}", condition_progress)
                self._process_condition(condition)
            
            # Phase 3: Generate summary statistics
            self._update_progress("Generating summary statistics...", 85)
            self._generate_summary_statistics()
            
            # Phase 4: Export results
            self._update_progress("Exporting results...", 95)
            self._export_results()
            
            # Complete
            end_time = datetime.now()
            duration = end_time - start_time
            
            self.results['experiment_info'] = {
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': duration.total_seconds(),
                'total_conditions': total_conditions,
                'total_images_processed': sum(len(cond_results.get('images', [])) 
                                            for cond_results in self.results['conditions'].values())
            }
            
            self._update_progress("Analysis complete!", 100)
            self.logger.info(f"Analysis completed in {duration}")
            
            return self.results
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            self._update_progress(f"Analysis failed: {str(e)}", -1)
            raise
    
    def _validate_configuration(self):
        """Validate analysis configuration"""
        
        required_fields = ['conditions', 'output_directory']
        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required configuration field: {field}")
        
        if not self.config['conditions']:
            raise ValueError("No conditions specified for analysis")
        
        # Validate each condition
        for condition in self.config['conditions']:
            self._validate_condition(condition)
    
    def _validate_condition(self, condition: Dict):
        """Validate a single condition configuration"""
        
        required_fields = ['name', 'directory', 'channels']
        for field in required_fields:
            if field not in condition:
                raise ValueError(f"Condition missing required field: {field}")
        
        directory = Path(condition['directory'])
        if not directory.exists():
            raise FileNotFoundError(f"Condition directory not found: {directory}")
        
        # Validate channels
        if not condition['channels']:
            raise ValueError(f"No channels configured for condition: {condition['name']}")
        
        # Check for at least one quantification channel
        quant_channels = [ch for ch in condition['channels'] if ch.get('quantify', False)]
        if not quant_channels:
            self.logger.warning(f"No quantification channels in condition: {condition['name']}")
    
    def _process_condition(self, condition: Dict):
        """Process a single experimental condition"""
        
        condition_name = condition['name']
        self.logger.info(f"Processing condition: {condition_name}")
        
        # Initialize condition results
        condition_results = {
            'name': condition_name,
            'directory': str(condition['directory']),
            'channels': condition['channels'],
            'images': [],
            'cell_data': [],
            'summary': {}
        }
        
        # Discover images in the condition directory
        images = self._discover_images(Path(condition['directory']))
        if not images:
            self.logger.warning(f"No images found in condition: {condition_name}")
            return
        
        self.logger.info(f"Found {len(images)} image groups in {condition_name}")
        
        # Process each image (or image group)
        for i, image_group in enumerate(images):
            try:
                image_name = self._get_image_group_name(image_group)
                self.logger.info(f"Processing image: {image_name}")
                
                # Load and process the image
                image_data = self._load_image_group(image_group, condition['channels'])
                
                # Perform segmentation
                masks, cell_rois = self._segment_cells(image_data, condition)
                
                # Quantify fluorescence
                cell_measurements = self._quantify_fluorescence(image_data, masks, cell_rois, condition)
                
                # Store results
                image_results = {
                    'name': image_name,
                    'files': [str(f) for f in image_group] if isinstance(image_group, list) else [str(image_group)],
                    'cell_count': len(cell_rois),
                    'measurements': cell_measurements
                }
                
                condition_results['images'].append(image_results)
                condition_results['cell_data'].extend(cell_measurements)
                
            except Exception as e:
                self.logger.error(f"Error processing image {i} in condition {condition_name}: {e}")
                continue
        
        # Generate condition summary
        condition_results['summary'] = self._summarize_condition_data(condition_results['cell_data'])
        
        # Store condition results
        with self.lock:
            self.results['conditions'][condition_name] = condition_results
    
    def _discover_images(self, directory: Path) -> List:
        """Discover and group images in directory"""
        
        supported_extensions = {'.tif', '.tiff', '.png', '.jpg', '.jpeg'}
        all_images = []
        
        for ext in supported_extensions:
            all_images.extend(directory.glob(f'*{ext}'))
            all_images.extend(directory.glob(f'*{ext.upper()}'))
        
        if not all_images:
            return []
        
        # Check if we have multichannel TIFFs or need to group single-channel images
        if len(all_images) == 1 and self._is_multichannel_tiff(all_images[0]):
            return [all_images[0]]  # Single multichannel TIFF
        else:
            return self._group_single_channel_images(all_images)  # Multiple single-channel images
    
    def _is_multichannel_tiff(self, filepath: Path) -> bool:
        """Check if TIFF file has multiple channels"""
        try:
            if not PIL_AVAILABLE:
                return False
                
            with Image.open(filepath) as img:
                return hasattr(img, 'n_frames') and img.n_frames > 1
        except Exception:
            return False
    
    def _group_single_channel_images(self, image_paths: List[Path]) -> List[List[Path]]:
        """Group single-channel images by base name"""
        
        import re
        
        groups = {}
        for path in image_paths:
            # Extract base name (remove channel identifiers)
            base_name = re.sub(r'_[cC]\d+|_[cC][hH]\d+|_channel\d+', '', path.stem)
            
            if base_name not in groups:
                groups[base_name] = []
            groups[base_name].append(path)
        
        # Return grouped images
        return list(groups.values())
    
    def _get_image_group_name(self, image_group) -> str:
        """Get display name for image group"""
        if isinstance(image_group, list):
            return Path(image_group[0]).stem
        else:
            return Path(image_group).stem
    
    def _load_image_group(self, image_group, channels: List[Dict]) -> Dict[str, np.ndarray]:
        """Load image data according to channel configuration"""
        
        image_data = {}
        
        if isinstance(image_group, list):
            # Multiple single-channel images
            for i, channel in enumerate(channels):
                if channel.get('source') == 'single_file' and 'filepath' in channel:
                    # Find matching file in the group
                    channel_file = None
                    for filepath in image_group:
                        if filepath == Path(channel['filepath']).name or filepath.name == Path(channel['filepath']).name:
                            channel_file = filepath
                            break
                    
                    if channel_file:
                        image_data[channel['name']] = self._load_single_image(channel_file)
                    else:
                        self.logger.warning(f"Could not find file for channel: {channel['name']}")
        else:
            # Single multichannel TIFF
            multichannel_image = self._load_multichannel_tiff(image_group)
            
            for channel in channels:
                if channel.get('source') == 'multichannel' and 'index' in channel:
                    channel_idx = channel['index']
                    if channel_idx < multichannel_image.shape[0]:
                        image_data[channel['name']] = multichannel_image[channel_idx]
                    else:
                        self.logger.warning(f"Channel index {channel_idx} out of range for {channel['name']}")
        
        return image_data
    
    def _load_single_image(self, filepath: Path) -> np.ndarray:
        """Load a single image file"""
        try:
            if SKIMAGE_AVAILABLE:
                return io.imread(str(filepath))
            elif PIL_AVAILABLE:
                with Image.open(filepath) as img:
                    return np.array(img)
            else:
                raise ImportError("No image loading library available")
        except Exception as e:
            self.logger.error(f"Error loading image {filepath}: {e}")
            raise
    
    def _load_multichannel_tiff(self, filepath: Path) -> np.ndarray:
        """Load multichannel TIFF file"""
        try:
            if SKIMAGE_AVAILABLE:
                # Use skimage for multichannel TIFF
                image = io.imread(str(filepath))
                if image.ndim == 3:
                    return image
                elif image.ndim == 2:
                    return image[np.newaxis, ...]  # Add channel dimension
                else:
                    raise ValueError(f"Unexpected image dimensions: {image.shape}")
            else:
                raise ImportError("scikit-image required for multichannel TIFF loading")
        except Exception as e:
            self.logger.error(f"Error loading multichannel TIFF {filepath}: {e}")
            raise
    
    def _segment_cells(self, image_data: Dict[str, np.ndarray], condition: Dict) -> Tuple[np.ndarray, List[Dict]]:
        """Perform cell segmentation using Cellpose"""
        
        # Find the best channel for segmentation
        segmentation_image = self._select_segmentation_channel(image_data, condition['channels'])
        
        if segmentation_image is None:
            raise ValueError("No suitable channel found for segmentation")
        
        # Cellpose parameters
        diameter = self.config.get('cell_diameter', None)
        if diameter is not None:
            diameter = float(diameter)
        
        # Run Cellpose
        try:
            masks, flows, styles, diams = self.cellpose_model.eval(
                segmentation_image,
                diameter=diameter,
                channels=[0, 0],  # Grayscale
                flow_threshold=self.config.get('flow_threshold', 0.4),
                cellprob_threshold=self.config.get('cellprob_threshold', 0.0)
            )
            
            # Convert masks to ROI list
            cell_rois = self._masks_to_rois(masks)
            
            self.logger.info(f"Segmented {len(cell_rois)} cells")
            return masks, cell_rois
            
        except Exception as e:
            self.logger.error(f"Cellpose segmentation failed: {e}")
            raise
    
    def _select_segmentation_channel(self, image_data: Dict[str, np.ndarray], channels: List[Dict]) -> Optional[np.ndarray]:
        """Select the best channel for segmentation"""
        
        # Look for channels designated for segmentation
        seg_channels = [ch for ch in channels if ch.get('purpose') == 'segmentation']
        
        if seg_channels:
            channel_name = seg_channels[0]['name']
            if channel_name in image_data:
                return image_data[channel_name]
        
        # Fall back to nuclear channels
        nuclear_channels = [ch for ch in channels if ch.get('type') == 'nuclear']
        if nuclear_channels:
            channel_name = nuclear_channels[0]['name']
            if channel_name in image_data:
                return image_data[channel_name]
        
        # Use first available channel
        if image_data:
            return list(image_data.values())[0]
        
        return None
    
    def _masks_to_rois(self, masks: np.ndarray) -> List[Dict]:
        """Convert segmentation masks to ROI dictionaries"""
        
        if not SKIMAGE_AVAILABLE:
            raise ImportError("scikit-image required for ROI extraction")
        
        rois = []
        for region in measure.regionprops(masks):
            roi = {
                'label': region.label,
                'area': region.area,
                'centroid': region.centroid,
                'bbox': region.bbox,
                'coords': region.coords,
                'perimeter': region.perimeter,
                'eccentricity': region.eccentricity,
                'solidity': region.solidity
            }
            rois.append(roi)
        
        return rois
    
    def _quantify_fluorescence(self, image_data: Dict[str, np.ndarray], masks: np.ndarray, 
                             cell_rois: List[Dict], condition: Dict) -> List[Dict]:
        """Quantify fluorescence for each cell"""
        
        measurements = []
        
        # Get channels to quantify
        quant_channels = [ch for ch in condition['channels'] if ch.get('quantify', False)]
        
        for roi in cell_rois:
            cell_id = roi['label']
            coords = roi['coords']
            
            cell_measurements = {
                'cell_id': cell_id,
                'condition': condition['name'],
                'area': roi['area'],
                'centroid_y': roi['centroid'][0],
                'centroid_x': roi['centroid'][1],
                'perimeter': roi.get('perimeter', 0),
                'eccentricity': roi.get('eccentricity', 0),
                'solidity': roi.get('solidity', 0)
            }
            
            # Quantify each channel
            for channel in quant_channels:
                channel_name = channel['name']
                
                if channel_name not in image_data:
                    continue
                
                channel_image = image_data[channel_name]
                
                # Calculate background
                background = self._estimate_background(channel_image, masks)
                
                # Get measurements for this channel
                channel_measurements = self._calculate_ctcf(
                    channel_image, coords, background, channel.get('nuclear_only', False), masks
                )
                
                # Add channel prefix to measurement names
                for key, value in channel_measurements.items():
                    cell_measurements[f"{channel_name}_{key}"] = value
            
            measurements.append(cell_measurements)
        
        return measurements
    
    def _estimate_background(self, image: np.ndarray, masks: np.ndarray) -> float:
        """Estimate background fluorescence"""
        
        method = self.config.get('background_method', 'mode')
        
        # Get background pixels (not covered by any cell)
        background_mask = masks == 0
        background_pixels = image[background_mask]
        
        if len(background_pixels) == 0:
            self.logger.warning("No background pixels found, using image minimum")
            return float(np.min(image))
        
        if method == 'mode':
            # Use histogram mode
            hist, bins = np.histogram(background_pixels, bins=50)
            mode_idx = np.argmax(hist)
            background = (bins[mode_idx] + bins[mode_idx + 1]) / 2
        elif method == 'median':
            background = np.median(background_pixels)
        elif method == 'mean':
            background = np.mean(background_pixels)
        else:
            background = np.median(background_pixels)  # Default fallback
        
        return float(background)
    
    def _calculate_ctcf(self, image: np.ndarray, coords: np.ndarray, background: float, 
                       nuclear_only: bool, masks: np.ndarray) -> Dict[str, float]:
        """Calculate Corrected Total Cell Fluorescence"""
        
        # Get pixel values at ROI coordinates
        roi_pixels = image[coords[:, 0], coords[:, 1]]
        
        integrated_density = np.sum(roi_pixels)
        area = len(roi_pixels)
        mean_intensity = np.mean(roi_pixels)
        max_intensity = np.max(roi_pixels)
        min_intensity = np.min(roi_pixels)
        std_intensity = np.std(roi_pixels)
        
        # Calculate CTCF
        ctcf = integrated_density - (area * background)
        
        measurements = {
            'integrated_density': float(integrated_density),
            'area': float(area),
            'mean_intensity': float(mean_intensity),
            'max_intensity': float(max_intensity),
            'min_intensity': float(min_intensity),
            'std_intensity': float(std_intensity),
            'background': float(background),
            'ctcf': float(ctcf)
        }
        
        # If nuclear_only flag is set, also calculate nuclear-specific measurements
        if nuclear_only:
            # This would require nuclear segmentation - simplified implementation
            nuclear_measurements = self._calculate_nuclear_measurements(image, coords, background)
            for key, value in nuclear_measurements.items():
                measurements[f"nuclear_{key}"] = value
        
        return measurements
    
    def _calculate_nuclear_measurements(self, image: np.ndarray, coords: np.ndarray, background: float) -> Dict[str, float]:
        """Calculate nuclear-specific measurements (simplified)"""
        
        # This is a simplified implementation
        # In practice, you would need proper nuclear segmentation
        
        # For now, assume nuclear region is the brightest central portion
        roi_pixels = image[coords[:, 0], coords[:, 1]]
        threshold = np.percentile(roi_pixels, 75)  # Top 25% of pixels
        nuclear_pixels = roi_pixels[roi_pixels >= threshold]
        
        if len(nuclear_pixels) == 0:
            nuclear_pixels = roi_pixels
        
        nuclear_integrated_density = np.sum(nuclear_pixels)
        nuclear_area = len(nuclear_pixels)
        nuclear_mean_intensity = np.mean(nuclear_pixels)
        nuclear_ctcf = nuclear_integrated_density - (nuclear_area * background)
        
        return {
            'integrated_density': float(nuclear_integrated_density),
            'area': float(nuclear_area),
            'mean_intensity': float(nuclear_mean_intensity),
            'ctcf': float(nuclear_ctcf)
        }
    
    def _summarize_condition_data(self, cell_data: List[Dict]) -> Dict:
        """Generate summary statistics for a condition"""
        
        if not cell_data:
            return {}
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(cell_data)
        
        summary = {
            'total_cells': len(df),
            'total_images': len(df.groupby('cell_id')),
            'mean_cell_area': float(df['area'].mean()),
            'std_cell_area': float(df['area'].std()),
        }
        
        # Summarize each quantified channel
        for column in df.columns:
            if column.endswith('_ctcf'):
                channel_name = column.replace('_ctcf', '')
                summary[f'{channel_name}_mean_ctcf'] = float(df[column].mean())
                summary[f'{channel_name}_std_ctcf'] = float(df[column].std())
                summary[f'{channel_name}_median_ctcf'] = float(df[column].median())
        
        return summary
    
    def _generate_summary_statistics(self):
        """Generate overall experiment summary statistics"""
        
        all_conditions = list(self.results['conditions'].keys())
        
        summary = {
            'total_conditions': len(all_conditions),
            'total_cells': sum(len(cond['cell_data']) for cond in self.results['conditions'].values()),
            'total_images': sum(len(cond['images']) for cond in self.results['conditions'].values()),
            'conditions': all_conditions
        }
        
        # Compare conditions if there are multiple
        if len(all_conditions) > 1:
            summary['condition_comparisons'] = self._compare_conditions()
        
        self.results['summary_statistics'] = summary
    
    def _compare_conditions(self) -> Dict:
        """Compare measurements between conditions"""
        
        comparisons = {}
        
        # Get all CTCF measurements across conditions
        for condition_name, condition_data in self.results['conditions'].items():
            if not condition_data['cell_data']:
                continue
                
            df = pd.DataFrame(condition_data['cell_data'])
            ctcf_columns = [col for col in df.columns if col.endswith('_ctcf')]
            
            for ctcf_col in ctcf_columns:
                channel_name = ctcf_col.replace('_ctcf', '')
                
                if channel_name not in comparisons:
                    comparisons[channel_name] = {}
                
                comparisons[channel_name][condition_name] = {
                    'mean': float(df[ctcf_col].mean()),
                    'std': float(df[ctcf_col].std()),
                    'median': float(df[ctcf_col].median()),
                    'n_cells': len(df)
                }
        
        return comparisons
    
    def _export_results(self):
        """Export analysis results"""
        
        output_dir = Path(self.config['output_directory'])
        output_dir.mkdir(exist_ok=True)
        
        # Export summary JSON
        summary_file = output_dir / 'analysis_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Export detailed cell data CSV
        self._export_cell_data_csv(output_dir)
        
        # Export condition summaries
        self._export_condition_summaries(output_dir)
        
        self.logger.info(f"Results exported to: {output_dir}")
    
    def _export_cell_data_csv(self, output_dir: Path):
        """Export detailed cell data as CSV"""
        
        all_cell_data = []
        for condition_data in self.results['conditions'].values():
            all_cell_data.extend(condition_data['cell_data'])
        
        if all_cell_data:
            df = pd.DataFrame(all_cell_data)
            csv_file = output_dir / 'cell_data.csv'
            df.to_csv(csv_file, index=False)
            self.logger.info(f"Cell data exported to: {csv_file}")
    
    def _export_condition_summaries(self, output_dir: Path):
        """Export condition summaries as CSV"""
        
        summaries = []
        for condition_name, condition_data in self.results['conditions'].items():
            summary = condition_data['summary'].copy()
            summary['condition'] = condition_name
            summaries.append(summary)
        
        if summaries:
            df = pd.DataFrame(summaries)
            csv_file = output_dir / 'condition_summaries.csv'
            df.to_csv(csv_file, index=False)
            self.logger.info(f"Condition summaries exported to: {csv_file}")
    
    def _update_progress(self, message: str, percentage: int):
        """Update progress callback"""
        if self.progress_callback:
            self.progress_callback(message, percentage)
        
        self.logger.info(f"Progress: {percentage}% - {message}")


class SegmentationEngine:
    """Cellpose integration for cell segmentation - Updated for v4.0.4+"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._check_cellpose_installation()
    
    def _check_cellpose_installation(self):
        """Verify Cellpose is available"""
        try:
            import cellpose
            from cellpose import models
            self.cellpose_available = True
            self.cellpose_version = cellpose.__version__
            self.logger.info(f"Cellpose {self.cellpose_version} successfully imported")
        except ImportError:
            self.cellpose_available = False
            self.logger.warning("Cellpose not available. Install with: pip install cellpose")
    
    def segment_cells(self, image: np.ndarray, model_type: str = 'cyto2', 
                     diameter: Optional[float] = None) -> Tuple[np.ndarray, List]:
        """Perform cell segmentation using Cellpose v4.0.4+"""
        if not self.cellpose_available:
            raise ImportError("Cellpose not installed")
        
        try:
            # Import the new API structure
            from cellpose import models
            
            # For Cellpose 4.x, the API has changed
            if hasattr(models, 'CellposeModel'):
                # New API (v4.0+)
                model = models.CellposeModel(gpu=False, model_type=model_type)
                
                # Ensure image is in correct format
                if image.ndim == 2:
                    # Add channel dimension for grayscale
                    image_input = image[np.newaxis, ...]
                else:
                    image_input = image
                
                # Run segmentation
                masks, flows, styles = model.eval(
                    image_input,
                    diameter=diameter,
                    channels=[0, 0],  # grayscale
                    flow_threshold=0.4,
                    cellprob_threshold=0.0
                )
                
            elif hasattr(models, 'Cellpose'):
                # Legacy API (v2.x-3.x)
                model = models.Cellpose(gpu=False, model_type=model_type)
                
                masks, flows, styles, diams = model.eval(
                    image, 
                    diameter=diameter,
                    channels=[0, 0]
                )
            
            else:
                # Try the most basic approach
                model = models.CellposeModel(model_type=model_type)
                masks = model.eval(image, diameter=diameter)[0]
                flows = None
                styles = None
            
            # Extract cell ROIs
            rois = self._masks_to_rois(masks)
            
            self.logger.info(f"Segmented {len(rois)} cells using Cellpose {self.cellpose_version}")
            return masks, rois
            
        except Exception as e:
            self.logger.error(f"Cellpose segmentation failed: {e}")
            self.logger.info("Trying fallback segmentation method...")
            
            # Fallback to simple threshold-based segmentation
            return self._fallback_segmentation(image)
    
    def _fallback_segmentation(self, image: np.ndarray) -> Tuple[np.ndarray, List]:
        """Fallback segmentation when Cellpose fails"""
        try:
            from skimage import filters, measure, morphology, segmentation
            
            # Simple threshold-based segmentation
            # Apply Gaussian filter to reduce noise
            filtered = filters.gaussian(image, sigma=1)
            
            # Otsu thresholding
            threshold = filters.threshold_otsu(filtered)
            binary = filtered > threshold
            
            # Clean up the binary image
            binary = morphology.remove_small_objects(binary, min_size=50)
            binary = morphology.remove_small_holes(binary, area_threshold=20)
            
            # Watershed segmentation to separate touching cells
            distance = morphology.distance_transform_edt(binary)
            local_maxima = filters.rank.maximum(distance, morphology.disk(10)) == distance
            local_maxima = local_maxima & (distance > 5)
            markers = measure.label(local_maxima)
            
            masks = segmentation.watershed(-distance, markers, mask=binary)
            
            # Extract ROIs
            rois = self._masks_to_rois(masks)
            
            self.logger.info(f"Fallback segmentation completed: {len(rois)} cells detected")
            return masks, rois
            
        except Exception as e:
            self.logger.error(f"Fallback segmentation also failed: {e}")
            # Return empty results
            return np.zeros_like(image), []
    
    def _masks_to_rois(self, masks: np.ndarray) -> List[Dict]:
        """Convert segmentation masks to ROI coordinates"""
        try:
            from skimage import measure
            
            rois = []
            for region in measure.regionprops(masks):
                roi = {
                    'label': region.label,
                    'area': region.area,
                    'centroid': region.centroid,
                    'bbox': region.bbox,
                    'coords': region.coords
                }
                
                # Add additional properties if available
                try:
                    roi['perimeter'] = region.perimeter
                    roi['eccentricity'] = region.eccentricity
                    roi['solidity'] = region.solidity
                except:
                    pass
                
                rois.append(roi)
            
            return rois
            
        except Exception as e:
            self.logger.error(f"Error converting masks to ROIs: {e}")
            return []


# Utility class for managing analysis parameters
class AnalysisParameters:
    """Default analysis parameters and validation"""
    
    DEFAULT_PARAMS = {
        'cellpose_model': 'cyto2',
        'cell_diameter': None,
        'use_gpu': False,
        'flow_threshold': 0.4,
        'cellprob_threshold': 0.0,
        'background_method': 'mode',  # 'mode', 'median', 'mean'
        'min_cell_area': 50,
        'max_cell_area': 5000,
        'export_images': True,
        'export_rois': True
    }
    
    @classmethod
    def get_default_config(cls) -> Dict:
        """Get default analysis configuration"""
        return cls.DEFAULT_PARAMS.copy()
    
    @classmethod
    def validate_parameters(cls, params: Dict) -> Dict:
        """Validate and fill in missing parameters"""
        validated = cls.DEFAULT_PARAMS.copy()
        validated.update(params)
        
        # Type conversions and validations
        if validated['cell_diameter'] is not None:
            validated['cell_diameter'] = float(validated['cell_diameter'])
        
        validated['flow_threshold'] = float(validated['flow_threshold'])
        validated['cellprob_threshold'] = float(validated['cellprob_threshold'])
        validated['min_cell_area'] = int(validated['min_cell_area'])
        validated['max_cell_area'] = int(validated['max_cell_area'])
        
        return validated