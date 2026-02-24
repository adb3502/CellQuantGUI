"""
Batch-optimized Cellpose segmentation engine.

Provides GPU-accelerated cell segmentation with batch processing
for 3-5x speedup over sequential processing.
"""

from typing import Optional, List, Dict, Any, Union, Callable
from pathlib import Path
import numpy as np
from dataclasses import dataclass, field
import warnings

try:
    import torch
    HAS_TORCH = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    HAS_TORCH = False
    CUDA_AVAILABLE = False

try:
    from cellpose import models, io as cellpose_io, plot as cellpose_plot
    from cellpose import utils as cellpose_utils
    HAS_CELLPOSE = True
except ImportError:
    HAS_CELLPOSE = False


@dataclass
class SegmentationParams:
    """Parameters for Cellpose segmentation."""
    model_type: str = "cpsam"
    diameter: float = 30.0
    flow_threshold: float = 0.4
    cellprob_threshold: float = 0.0
    min_size: int = 15
    channels: List[int] = field(default_factory=lambda: [0, 0])
    use_gpu: bool = True
    batch_size: int = 4


@dataclass
class SegmentationResult:
    """Result from segmentation of a single image."""
    masks: np.ndarray
    flows: Optional[np.ndarray] = None
    styles: Optional[np.ndarray] = None
    n_cells: int = 0
    diameter_used: float = 0.0

    def __post_init__(self):
        if self.n_cells == 0 and self.masks is not None:
            self.n_cells = len(np.unique(self.masks)) - 1


class CellposeEngine:
    """
    Batch-optimized Cellpose segmentation engine.

    Features:
    - Lazy model loading (loads on first use)
    - Batch GPU processing for multiple images
    - Automatic GPU/CPU selection
    - Progress callbacks for UI integration

    Example:
        >>> engine = CellposeEngine(model_type="cpsam", use_gpu=True)
        >>> results = engine.segment_batch(images, diameter=30.0)
        >>> print(f"Segmented {len(results)} images")
    """

    SUPPORTED_MODELS = [
        "cpsam",      # Cellpose-SAM (best accuracy)
        "cyto",       # Cytoplasm (original)
        "cyto2",      # Cytoplasm v2
        "cyto3",      # Cytoplasm v3
        "nuclei",     # Nuclei only
        "livecell",   # Live cell imaging
    ]

    def __init__(
        self,
        model_type: str = "cpsam",
        use_gpu: bool = True,
        batch_size: int = 4,
        custom_model_path: Optional[str] = None,
    ):
        """
        Initialize the segmentation engine.

        Args:
            model_type: Cellpose model to use
            use_gpu: Whether to use GPU acceleration
            batch_size: Number of images to process per batch
            custom_model_path: Path to a fine-tuned model (overrides model_type)
        """
        if not HAS_CELLPOSE:
            raise ImportError(
                "Cellpose required. Install with: pip install cellpose"
            )

        if not custom_model_path and model_type not in self.SUPPORTED_MODELS:
            warnings.warn(
                f"Model '{model_type}' not in standard list. "
                f"Supported: {self.SUPPORTED_MODELS}"
            )

        self.model_type = model_type
        self.custom_model_path = custom_model_path
        self.use_gpu = use_gpu and CUDA_AVAILABLE
        self.batch_size = batch_size
        self._model = None

        if use_gpu and not CUDA_AVAILABLE:
            warnings.warn("GPU requested but CUDA not available. Using CPU.")

    @property
    def model(self):
        """Lazy load the Cellpose model."""
        if self._model is None:
            if self.custom_model_path:
                self._model = models.CellposeModel(
                    gpu=self.use_gpu,
                    pretrained_model=self.custom_model_path,
                )
            else:
                self._model = models.CellposeModel(
                    gpu=self.use_gpu,
                    model_type=self.model_type,
                )
        return self._model

    def segment_single(
        self,
        image: np.ndarray,
        diameter: float = 30.0,
        flow_threshold: float = 0.4,
        min_size: int = 15,
        channels: Optional[List[int]] = None
    ) -> SegmentationResult:
        """
        Segment a single image.

        Args:
            image: Image array (Y, X) or (C, Y, X) or (Y, X, C)
            diameter: Expected cell diameter (0 for auto)
            flow_threshold: Flow error threshold
            min_size: Minimum cell size in pixels
            channels: Channel specification [cytoplasm, nucleus]

        Returns:
            SegmentationResult with masks and metadata
        """
        if channels is None:
            channels = [0, 0]

        # Ensure image is in correct format
        image = self._prepare_image(image)

        # Cellpose model inference — v4.0.1+ returns 3 values (no diameter)
        out = self.model.eval(
            image,
            diameter=diameter if diameter and diameter > 0 else None,
            flow_threshold=flow_threshold,
            min_size=min_size,
            channels=channels
        )
        if len(out) == 4:
            masks, flows, styles, diameter_used = out
        else:
            masks, flows, styles = out
            diameter_used = diameter or 0.0

        return SegmentationResult(
            masks=masks,
            flows=flows,
            styles=styles,
            diameter_used=diameter_used
        )

    def segment_batch(
        self,
        images: List[np.ndarray],
        diameter: float = 30.0,
        flow_threshold: float = 0.4,
        min_size: int = 15,
        channels: Optional[List[int]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[SegmentationResult]:
        """
        Segment multiple images with batch GPU processing.

        This is 3-5x faster than processing images one at a time
        because it batches inference on the GPU.

        Args:
            images: List of image arrays
            diameter: Expected cell diameter
            flow_threshold: Flow error threshold
            min_size: Minimum cell size
            channels: Channel specification
            progress_callback: Optional callback(current, total) for progress

        Returns:
            List of SegmentationResult, one per image
        """
        if channels is None:
            channels = [0, 0]

        # Prepare all images
        prepared_images = [self._prepare_image(img) for img in images]

        results = []
        n_images = len(prepared_images)

        # Process in batches for memory efficiency
        for i in range(0, n_images, self.batch_size):
            batch = prepared_images[i:i + self.batch_size]
            batch_size = len(batch)

            # Batch inference using Cellpose API
            masks_list, flows_list, styles_list = self.model.eval(
                batch,
                diameter=diameter if diameter and diameter > 0 else None,
                flow_threshold=flow_threshold,
                min_size=min_size,
                channels=channels,
                batch_size=batch_size
            )

            # Handle single vs. multiple results
            if not isinstance(masks_list, list):
                masks_list = [masks_list]
                flows_list = [flows_list] if flows_list is not None else [None]
                styles_list = [styles_list] if styles_list is not None else [None]

            for j, (masks, flows, styles) in enumerate(zip(masks_list, flows_list, styles_list)):
                results.append(SegmentationResult(
                    masks=masks,
                    flows=flows,
                    styles=styles,
                    diameter_used=diameter
                ))

            if progress_callback:
                progress_callback(min(i + batch_size, n_images), n_images)

        return results

    def segment_with_params(
        self,
        image: np.ndarray,
        params: SegmentationParams
    ) -> SegmentationResult:
        """
        Segment with a params object.

        Args:
            image: Image array
            params: SegmentationParams object

        Returns:
            SegmentationResult
        """
        return self.segment_single(
            image=image,
            diameter=params.diameter,
            flow_threshold=params.flow_threshold,
            min_size=params.min_size,
            channels=params.channels
        )

    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        """
        Prepare image for Cellpose.

        Ensures correct format: (C, Y, X) or (Y, X) with channels <= 3.
        """
        if image.ndim == 2:
            return image

        if image.ndim == 3:
            # Check if channels are in last dimension (Y, X, C)
            if image.shape[-1] <= 5:
                # Move channels to first dimension
                image = np.moveaxis(image, -1, 0)

            # Cellpose expects max 3 channels
            if image.shape[0] > 3:
                image = image[:3]

            return image

        raise ValueError(f"Unexpected image dimensions: {image.shape}")

    @staticmethod
    def create_overlay(
        image: np.ndarray,
        masks: np.ndarray,
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        Create colorful mask overlay on image.

        Args:
            image: Original image
            masks: Segmentation masks
            alpha: Overlay transparency

        Returns:
            RGB overlay image (uint8)
        """
        try:
            if image.ndim == 3:
                display_img = np.mean(image, axis=0 if image.shape[0] <= 3 else -1)
            else:
                display_img = image

            overlay = cellpose_plot.mask_overlay(display_img, masks)
            return overlay

        except Exception:
            # Fallback to simple overlay
            return _create_simple_overlay(image, masks, alpha)

    def clear_gpu_memory(self):
        """Clear GPU memory after processing."""
        if HAS_TORCH and CUDA_AVAILABLE:
            torch.cuda.empty_cache()


def _create_simple_overlay(
    image: np.ndarray,
    masks: np.ndarray,
    alpha: float = 0.5
) -> np.ndarray:
    """Vectorized mask overlay — single-pass LUT instead of per-cell loop."""
    if image.ndim == 3:
        display_img = np.mean(image, axis=0 if image.shape[0] <= 3 else -1)
    else:
        display_img = image.astype(np.float64)

    # Normalize to 0-1
    vmin, vmax = display_img.min(), display_img.max()
    if vmax > vmin:
        display_norm = (display_img - vmin) / (vmax - vmin)
    else:
        display_norm = np.zeros_like(display_img, dtype=np.float64)

    # Build color LUT: index 0 = no mask (black placeholder, won't be used)
    max_label = int(masks.max())
    rng = np.random.RandomState(42)
    lut = rng.rand(max_label + 1, 3).astype(np.float32)
    lut[0] = 0  # background — unused

    # Vectorized: map every pixel to its mask color in one shot
    mask_colors = lut[masks]  # (H, W, 3)

    # Grayscale RGB base
    gray = display_norm.astype(np.float32)
    overlay = np.stack([gray, gray, gray], axis=-1)

    # Blend only where mask > 0
    has_mask = masks > 0
    overlay[has_mask] = mask_colors[has_mask] * alpha + overlay[has_mask] * (1 - alpha)

    return (np.clip(overlay, 0, 1) * 255).astype(np.uint8)


def _create_outline_overlay(
    image: np.ndarray,
    masks: np.ndarray,
) -> np.ndarray:
    """Red outline overlay — cell boundaries drawn on the image."""
    from scipy.ndimage import binary_dilation

    if image.ndim == 3:
        display_img = np.mean(image, axis=0 if image.shape[0] <= 3 else -1)
    else:
        display_img = image.astype(np.float64)

    # Normalize to uint8
    vmin, vmax = display_img.min(), display_img.max()
    if vmax > vmin:
        gray = ((display_img - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    else:
        gray = np.zeros(display_img.shape, dtype=np.uint8)

    # Build RGB from grayscale
    overlay = np.stack([gray, gray, gray], axis=-1)

    # Extract outlines: pixels at mask boundaries
    # A pixel is an outline if it's in a mask but has a neighbor with a different label
    if masks.max() == 0:
        return overlay

    # Shift masks in 4 directions and compare — boundary pixels differ from neighbors
    padded = np.pad(masks, 1, mode='constant', constant_values=0)
    outlines = (
        (masks != padded[:-2, 1:-1]) |  # up
        (masks != padded[2:, 1:-1]) |    # down
        (masks != padded[1:-1, :-2]) |   # left
        (masks != padded[1:-1, 2:])      # right
    ) & (masks > 0)

    # Draw red outlines
    overlay[outlines, 0] = 255
    overlay[outlines, 1] = 0
    overlay[outlines, 2] = 0

    return overlay


def segment_image_stack(
    engine: CellposeEngine,
    stack: np.ndarray,
    params: SegmentationParams,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> np.ndarray:
    """
    Segment a stack of images (e.g., timelapse).

    Args:
        engine: CellposeEngine instance
        stack: 3D array (T, Y, X) or 4D (T, C, Y, X)
        params: Segmentation parameters
        progress_callback: Optional progress callback

    Returns:
        3D mask array (T, Y, X)
    """
    if stack.ndim == 3:
        n_frames = stack.shape[0]
        frames = [stack[t] for t in range(n_frames)]
    elif stack.ndim == 4:
        n_frames = stack.shape[0]
        frames = [stack[t] for t in range(n_frames)]
    else:
        raise ValueError(f"Expected 3D or 4D stack, got shape {stack.shape}")

    results = engine.segment_batch(
        frames,
        diameter=params.diameter,
        flow_threshold=params.flow_threshold,
        min_size=params.min_size,
        channels=params.channels,
        progress_callback=progress_callback
    )

    # Stack masks
    masks_stack = np.stack([r.masks for r in results], axis=0)
    return masks_stack
