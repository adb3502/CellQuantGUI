"""
Batch processing pipeline for high-throughput cell quantification.

Orchestrates folder scanning, segmentation, and quantification for
entire experiments with progress tracking and parallel processing.
"""

from typing import Optional, Dict, List, Any, Callable, Tuple, Generator
from pathlib import Path
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import re
import time
import threading

from cellquant_enterprise.core.io.image_loader import (
    load_image, normalize_image, find_images_by_suffix
)
from cellquant_enterprise.core.io.mask_io import load_mask, save_mask
from cellquant_enterprise.core.io.roi_export import save_rois_imagej
from cellquant_enterprise.core.segmentation.cellpose_engine import (
    CellposeEngine, SegmentationParams, SegmentationResult
)
from cellquant_enterprise.core.quantification.ctcf import (
    calculate_ctcf_vectorized, quantify_multiple_markers, results_to_dataframe
)
from cellquant_enterprise.core.quantification.background import estimate_background


@dataclass
class ChannelConfig:
    """Configuration for channel assignment."""
    nuclear_suffix: str = "C0"
    cyto_suffix: str = "C1"
    marker_suffixes: List[str] = field(default_factory=lambda: ["C2"])
    marker_names: List[str] = field(default_factory=lambda: ["Marker1"])
    mitochondrial_markers: List[str] = field(default_factory=list)
    segmentation_channels: List[str] = field(default_factory=lambda: ["nuclear", "cyto"])


@dataclass
class ExperimentCondition:
    """A single experimental condition with its images."""
    name: str
    path: Path
    image_sets: Dict[str, Dict[str, Path]] = field(default_factory=dict)
    channel_config: Optional[ChannelConfig] = None
    n_images: int = 0

    def __post_init__(self):
        self.n_images = len(self.image_sets)


@dataclass
class ProcessingProgress:
    """Progress tracking for pipeline operations."""
    stage: str
    current: int
    total: int
    condition: str = ""
    image_set: str = ""
    message: str = ""
    elapsed_seconds: float = 0.0

    @property
    def percent(self) -> float:
        return 100 * self.current / self.total if self.total > 0 else 0


class BatchPipeline:
    """
    Batch processing pipeline for cell quantification experiments.

    Handles:
    - Experiment folder scanning and condition detection
    - Batch GPU segmentation with progress
    - Parallel marker quantification
    - Combined results generation

    Example:
        >>> pipeline = BatchPipeline()
        >>> conditions = pipeline.scan_experiment_folder("path/to/experiment")
        >>> results = pipeline.run_analysis(conditions, seg_params, channel_config)
    """

    def __init__(
        self,
        n_workers: int = 4,
        use_gpu: bool = True,
        batch_size: int = 4
    ):
        """
        Initialize the pipeline.

        Args:
            n_workers: Number of parallel workers for CPU operations
            use_gpu: Whether to use GPU for segmentation
            batch_size: Batch size for GPU segmentation
        """
        self.n_workers = n_workers
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self._cancel_flag = threading.Event()
        self._engine = None

    @property
    def engine(self) -> CellposeEngine:
        """Lazy-load the segmentation engine."""
        if self._engine is None:
            self._engine = CellposeEngine(
                use_gpu=self.use_gpu,
                batch_size=self.batch_size
            )
        return self._engine

    def set_model(self, model_type: str):
        """Set the Cellpose model type."""
        self._engine = CellposeEngine(
            model_type=model_type,
            use_gpu=self.use_gpu,
            batch_size=self.batch_size
        )

    def cancel(self):
        """Cancel ongoing processing."""
        self._cancel_flag.set()

    def reset_cancel(self):
        """Reset the cancel flag."""
        self._cancel_flag.clear()

    def scan_experiment_folder(
        self,
        folder_path: Path,
        channel_config: ChannelConfig
    ) -> List[ExperimentCondition]:
        """
        Scan experiment folder for conditions and image sets.

        Expects structure:
            experiment_folder/
                condition1/
                    image1_C0.tif, image1_C1.tif, image1_C2.tif
                    image2_C0.tif, image2_C1.tif, image2_C2.tif
                condition2/
                    ...

        Args:
            folder_path: Path to experiment folder
            channel_config: Channel configuration

        Returns:
            List of ExperimentCondition objects
        """
        folder_path = Path(folder_path)

        if not folder_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {folder_path}")

        conditions = []

        # Get all suffixes to search for
        all_suffixes = [
            channel_config.nuclear_suffix,
            channel_config.cyto_suffix,
        ] + channel_config.marker_suffixes

        # Scan subdirectories as conditions
        for subdir in sorted(folder_path.iterdir()):
            if not subdir.is_dir():
                continue

            # Skip hidden directories
            if subdir.name.startswith('.'):
                continue

            # Find images grouped by suffix
            image_sets = find_images_by_suffix(subdir, all_suffixes)

            if image_sets:
                conditions.append(ExperimentCondition(
                    name=subdir.name,
                    path=subdir,
                    image_sets=image_sets,
                    channel_config=channel_config,
                    n_images=len(image_sets)
                ))

        return conditions

    def run_analysis(
        self,
        conditions: List[ExperimentCondition],
        seg_params: SegmentationParams,
        output_folder: Path,
        progress_callback: Optional[Callable[[ProcessingProgress], None]] = None,
        save_outputs: bool = True
    ) -> pd.DataFrame:
        """
        Run full analysis pipeline on all conditions.

        Args:
            conditions: List of conditions to process
            seg_params: Segmentation parameters
            output_folder: Output folder for results
            progress_callback: Optional callback for progress updates
            save_outputs: Whether to save intermediate outputs (masks, overlays, ROIs)

        Returns:
            Combined DataFrame with results for all conditions
        """
        self.reset_cancel()
        start_time = time.time()

        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        all_results = []
        total_images = sum(c.n_images for c in conditions)
        processed_images = 0

        for condition in conditions:
            if self._cancel_flag.is_set():
                break

            condition_output = output_folder / condition.name
            condition_output.mkdir(parents=True, exist_ok=True)

            # Process each image set in the condition
            for base_name, image_paths in condition.image_sets.items():
                if self._cancel_flag.is_set():
                    break

                # Update progress
                if progress_callback:
                    progress_callback(ProcessingProgress(
                        stage="Processing",
                        current=processed_images,
                        total=total_images,
                        condition=condition.name,
                        image_set=base_name,
                        message=f"Processing {base_name}...",
                        elapsed_seconds=time.time() - start_time
                    ))

                try:
                    # Process single image set
                    result_df = self._process_image_set(
                        condition=condition,
                        base_name=base_name,
                        image_paths=image_paths,
                        seg_params=seg_params,
                        output_folder=condition_output,
                        save_outputs=save_outputs
                    )

                    if result_df is not None and len(result_df) > 0:
                        all_results.append(result_df)

                except Exception as e:
                    print(f"Error processing {condition.name}/{base_name}: {e}")

                processed_images += 1

        # Combine all results
        if all_results:
            combined_df = pd.concat(all_results, ignore_index=True)
        else:
            combined_df = pd.DataFrame()

        # Save combined results
        if len(combined_df) > 0:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            csv_path = output_folder / f"ctcf_analysis_{timestamp}.csv"
            combined_df.to_csv(csv_path, index=False)

        # Final progress update
        if progress_callback:
            progress_callback(ProcessingProgress(
                stage="Complete",
                current=total_images,
                total=total_images,
                message=f"Processed {total_images} images, {len(combined_df)} cells",
                elapsed_seconds=time.time() - start_time
            ))

        return combined_df

    def _process_image_set(
        self,
        condition: ExperimentCondition,
        base_name: str,
        image_paths: Dict[str, Path],
        seg_params: SegmentationParams,
        output_folder: Path,
        save_outputs: bool
    ) -> Optional[pd.DataFrame]:
        """Process a single image set (one FOV with all channels)."""
        config = condition.channel_config

        # Load nuclear and cytoplasm channels
        nuclear_path = image_paths.get(config.nuclear_suffix.upper())
        cyto_path = image_paths.get(config.cyto_suffix.upper())

        if not nuclear_path:
            print(f"  Warning: No nuclear channel for {base_name}")
            return None

        nuclear_img = load_image(nuclear_path)
        nuclear_norm = normalize_image(nuclear_img)

        cyto_img = load_image(cyto_path) if cyto_path else None
        cyto_norm = normalize_image(cyto_img) if cyto_img is not None else None

        # Stack channels for segmentation
        if cyto_norm is not None:
            seg_input = np.stack([nuclear_norm, cyto_norm], axis=0)
        else:
            seg_input = nuclear_norm

        # Run segmentation
        result = self.engine.segment_single(
            seg_input,
            diameter=seg_params.diameter,
            flow_threshold=seg_params.flow_threshold,
            min_size=seg_params.min_size,
            channels=seg_params.channels
        )
        masks = result.masks

        if masks.max() == 0:
            print(f"  Warning: No cells detected in {base_name}")
            return None

        # Save outputs
        if save_outputs:
            set_output = output_folder / base_name
            set_output.mkdir(parents=True, exist_ok=True)

            # Save masks
            save_mask(masks, set_output / f"{base_name}_masks.tif")

            # Save overlay
            overlay = CellposeEngine.create_overlay(seg_input, masks)
            from skimage import io
            io.imsave(str(set_output / f"{base_name}_overlay.png"), overlay)

            # Save ROIs
            save_rois_imagej(masks, set_output / f"{base_name}_rois.zip")

        # Load marker images and quantify
        marker_images = {}
        for suffix, name in zip(config.marker_suffixes, config.marker_names):
            marker_path = image_paths.get(suffix.upper())
            if marker_path:
                marker_images[name] = load_image(marker_path)

        if not marker_images:
            print(f"  Warning: No marker channels for {base_name}")
            return None

        # Estimate backgrounds
        backgrounds = {
            name: estimate_background(img, masks)
            for name, img in marker_images.items()
        }

        # Quantify all markers (parallel)
        results = quantify_multiple_markers(
            marker_images=marker_images,
            masks=masks,
            backgrounds=backgrounds,
            mitochondrial_markers=config.mitochondrial_markers,
            parallel=True,
            n_workers=self.n_workers
        )

        # Convert to DataFrame
        df = results_to_dataframe(
            results=results,
            condition=condition.name,
            image_set=base_name,
            segmentation_type="cellular"
        )

        return df

    def run_segmentation_only(
        self,
        conditions: List[ExperimentCondition],
        seg_params: SegmentationParams,
        output_folder: Path,
        progress_callback: Optional[Callable[[ProcessingProgress], None]] = None
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Run segmentation only (no quantification).

        Returns dict mapping condition -> {image_set: masks}
        """
        self.reset_cancel()
        start_time = time.time()

        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        all_masks = {}
        total_images = sum(c.n_images for c in conditions)
        processed = 0

        for condition in conditions:
            if self._cancel_flag.is_set():
                break

            config = condition.channel_config
            condition_masks = {}

            # Collect all images for batch processing
            images_to_segment = []
            image_names = []

            for base_name, image_paths in condition.image_sets.items():
                nuclear_path = image_paths.get(config.nuclear_suffix.upper())
                cyto_path = image_paths.get(config.cyto_suffix.upper())

                if not nuclear_path:
                    continue

                nuclear_img = load_image(nuclear_path)
                nuclear_norm = normalize_image(nuclear_img)

                if cyto_path:
                    cyto_img = load_image(cyto_path)
                    cyto_norm = normalize_image(cyto_img)
                    seg_input = np.stack([nuclear_norm, cyto_norm], axis=0)
                else:
                    seg_input = nuclear_norm

                images_to_segment.append(seg_input)
                image_names.append(base_name)

            # Batch segment
            if images_to_segment:
                def batch_progress(current, total):
                    if progress_callback:
                        progress_callback(ProcessingProgress(
                            stage="Segmenting",
                            current=processed + current,
                            total=total_images,
                            condition=condition.name,
                            message=f"Batch segmenting {condition.name}...",
                            elapsed_seconds=time.time() - start_time
                        ))

                results = self.engine.segment_batch(
                    images_to_segment,
                    diameter=seg_params.diameter,
                    flow_threshold=seg_params.flow_threshold,
                    min_size=seg_params.min_size,
                    channels=seg_params.channels,
                    progress_callback=batch_progress
                )

                for name, result in zip(image_names, results):
                    condition_masks[name] = result.masks

                    # Save masks
                    condition_output = output_folder / condition.name
                    condition_output.mkdir(parents=True, exist_ok=True)
                    save_mask(result.masks, condition_output / f"{name}_masks.tif")

            all_masks[condition.name] = condition_masks
            processed += len(images_to_segment)

        return all_masks

    def run_quantification_only(
        self,
        conditions: List[ExperimentCondition],
        masks_dict: Dict[str, Dict[str, np.ndarray]],
        output_folder: Path,
        progress_callback: Optional[Callable[[ProcessingProgress], None]] = None
    ) -> pd.DataFrame:
        """
        Run quantification on pre-computed masks.

        Args:
            conditions: List of conditions
            masks_dict: Pre-computed masks {condition: {image_set: masks}}
            output_folder: Output folder
            progress_callback: Progress callback

        Returns:
            Combined results DataFrame
        """
        self.reset_cancel()
        start_time = time.time()

        all_results = []
        total = sum(len(masks_dict.get(c.name, {})) for c in conditions)
        processed = 0

        for condition in conditions:
            if self._cancel_flag.is_set():
                break

            config = condition.channel_config
            condition_masks = masks_dict.get(condition.name, {})

            for base_name, masks in condition_masks.items():
                if self._cancel_flag.is_set():
                    break

                if progress_callback:
                    progress_callback(ProcessingProgress(
                        stage="Quantifying",
                        current=processed,
                        total=total,
                        condition=condition.name,
                        image_set=base_name,
                        elapsed_seconds=time.time() - start_time
                    ))

                image_paths = condition.image_sets.get(base_name, {})

                # Load marker images
                marker_images = {}
                for suffix, name in zip(config.marker_suffixes, config.marker_names):
                    marker_path = image_paths.get(suffix.upper())
                    if marker_path:
                        marker_images[name] = load_image(marker_path)

                if not marker_images:
                    processed += 1
                    continue

                # Estimate backgrounds
                backgrounds = {
                    name: estimate_background(img, masks)
                    for name, img in marker_images.items()
                }

                # Quantify
                results = quantify_multiple_markers(
                    marker_images=marker_images,
                    masks=masks,
                    backgrounds=backgrounds,
                    mitochondrial_markers=config.mitochondrial_markers,
                    parallel=True,
                    n_workers=self.n_workers
                )

                df = results_to_dataframe(
                    results=results,
                    condition=condition.name,
                    image_set=base_name
                )

                if len(df) > 0:
                    all_results.append(df)

                processed += 1

        if all_results:
            combined = pd.concat(all_results, ignore_index=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            combined.to_csv(output_folder / f"ctcf_analysis_{timestamp}.csv", index=False)
            return combined

        return pd.DataFrame()


def create_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create summary statistics from results DataFrame.

    Args:
        df: Results DataFrame

    Returns:
        Summary DataFrame with per-condition statistics
    """
    if df.empty:
        return pd.DataFrame()

    # Find CTCF columns
    ctcf_cols = [c for c in df.columns if c.endswith('_CTCF')]

    summary_data = []

    for condition in df['Condition'].unique():
        cond_df = df[df['Condition'] == condition]

        row = {
            'Condition': condition,
            'N_Cells': len(cond_df),
            'Mean_Area': cond_df['Area'].mean(),
            'Std_Area': cond_df['Area'].std(),
        }

        for col in ctcf_cols:
            marker = col.replace('_CTCF', '')
            row[f'{marker}_Mean_CTCF'] = cond_df[col].mean()
            row[f'{marker}_Std_CTCF'] = cond_df[col].std()
            row[f'{marker}_Median_CTCF'] = cond_df[col].median()

        summary_data.append(row)

    return pd.DataFrame(summary_data)
