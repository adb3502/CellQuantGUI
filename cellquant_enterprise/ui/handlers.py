"""
Backend handlers for Gradio UI components.

Connects UI actions to the core processing pipeline.
"""

from typing import Optional, Dict, List, Tuple, Any, Generator
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import threading
import time

from cellquant_enterprise.core.pipeline import (
    BatchPipeline, ChannelConfig, ExperimentCondition, ProcessingProgress,
    create_summary_statistics
)
from cellquant_enterprise.core.segmentation.cellpose_engine import (
    CellposeEngine, SegmentationParams
)
from cellquant_enterprise.core.io.image_loader import load_image, normalize_image, create_composite
from cellquant_enterprise.core.io.mask_io import load_mask, save_mask
from cellquant_enterprise.core.napari_bridge import NapariBridge, launch_napari_editor


class SessionState:
    """
    Session state management for the Gradio app.

    Stores experiment data, processing results, and UI state.
    """

    def __init__(self):
        self.experiment_path: Optional[Path] = None
        self.conditions: List[ExperimentCondition] = []
        self.channel_config: Optional[ChannelConfig] = None
        self.masks: Dict[str, Dict[str, np.ndarray]] = {}
        self.results: Optional[pd.DataFrame] = None
        self.current_preview: Optional[np.ndarray] = None
        self.current_masks_preview: Optional[np.ndarray] = None
        self.is_processing: bool = False
        self.pipeline: Optional[BatchPipeline] = None
        self.napari_bridge: Optional[NapariBridge] = None
        self.edited_masks: Dict[str, np.ndarray] = {}

    def to_dict(self) -> dict:
        """Convert to dictionary for Gradio State."""
        return {
            'experiment_path': str(self.experiment_path) if self.experiment_path else None,
            'n_conditions': len(self.conditions),
            'n_results': len(self.results) if self.results is not None else 0,
            'is_processing': self.is_processing,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'SessionState':
        """Create from Gradio State dictionary."""
        state = cls()
        if data.get('experiment_path'):
            state.experiment_path = Path(data['experiment_path'])
        return state


# Global state (will be replaced by Gradio State in production)
_global_state = SessionState()


def get_pipeline() -> BatchPipeline:
    """Get or create the batch pipeline."""
    if _global_state.pipeline is None:
        _global_state.pipeline = BatchPipeline()
    return _global_state.pipeline


# =============================================================================
# TAB 1: LOAD IMAGES HANDLERS
# =============================================================================

def scan_experiment_folder(
    folder_path: str,
    nuclear_suffix: str,
    cyto_suffix: str,
    marker_config: pd.DataFrame
) -> Tuple[pd.DataFrame, List[np.ndarray], str]:
    """
    Scan experiment folder and return conditions table and preview images.

    Returns:
        Tuple of (conditions_table, preview_images, status_message)
    """
    if not folder_path or not Path(folder_path).exists():
        return pd.DataFrame(), [], "Invalid folder path"

    folder = Path(folder_path)

    # Build channel config from UI inputs
    marker_suffixes = []
    marker_names = []

    if marker_config is not None and len(marker_config) > 0:
        for _, row in marker_config.iterrows():
            if row.iloc[0]:  # Has suffix
                marker_suffixes.append(str(row.iloc[0]))
                marker_names.append(str(row.iloc[1]) if row.iloc[1] else f"Marker{len(marker_suffixes)}")

    channel_config = ChannelConfig(
        nuclear_suffix=nuclear_suffix,
        cyto_suffix=cyto_suffix,
        marker_suffixes=marker_suffixes or ["C2"],
        marker_names=marker_names or ["Marker1"]
    )

    _global_state.channel_config = channel_config
    _global_state.experiment_path = folder

    # Scan for conditions
    pipeline = get_pipeline()
    conditions = pipeline.scan_experiment_folder(folder, channel_config)
    _global_state.conditions = conditions

    # Build conditions table
    table_data = []
    for cond in conditions:
        table_data.append({
            'Condition': cond.name,
            'Images': cond.n_images,
            'Status': 'Ready'
        })

    conditions_df = pd.DataFrame(table_data)

    # Generate preview from first condition's first image
    preview_images = []
    if conditions and conditions[0].image_sets:
        first_cond = conditions[0]
        first_set = list(first_cond.image_sets.values())[0]

        # Load each channel
        for suffix in [nuclear_suffix, cyto_suffix] + marker_suffixes:
            path = first_set.get(suffix.upper())
            if path and path.exists():
                img = load_image(path)
                img_norm = normalize_image(img)
                # Convert to uint8 for display
                img_display = (img_norm * 255).astype(np.uint8)
                preview_images.append(img_display)

    status = f"Found {len(conditions)} conditions with {sum(c.n_images for c in conditions)} total image sets"

    return conditions_df, preview_images, status


# =============================================================================
# TAB 2: SEGMENTATION HANDLERS
# =============================================================================

def run_segmentation(
    model_type: str,
    seg_target: str,
    diameter: float,
    flow_thresh: float,
    min_size: int,
    use_gpu: bool,
    output_folder: str,
    progress=None
) -> Tuple[np.ndarray, pd.DataFrame, str]:
    """
    Run batch segmentation on all conditions.

    Returns:
        Tuple of (preview_image, stats_table, status_message)
    """
    if not _global_state.conditions:
        return None, pd.DataFrame(), "No experiment loaded"

    _global_state.is_processing = True

    pipeline = get_pipeline()
    pipeline.set_model(model_type)

    seg_params = SegmentationParams(
        model_type=model_type,
        diameter=diameter,
        flow_threshold=flow_thresh,
        min_size=min_size,
        use_gpu=use_gpu
    )

    output_path = Path(output_folder) if output_folder else _global_state.experiment_path / "output"

    # Progress callback
    def progress_callback(prog: ProcessingProgress):
        if progress:
            progress(prog.percent / 100, desc=prog.message)

    try:
        masks_dict = pipeline.run_segmentation_only(
            conditions=_global_state.conditions,
            seg_params=seg_params,
            output_folder=output_path,
            progress_callback=progress_callback
        )

        _global_state.masks = masks_dict

        # Generate preview from first result
        preview = None
        if masks_dict:
            first_cond = list(masks_dict.keys())[0]
            first_masks = masks_dict[first_cond]
            if first_masks:
                first_set = list(first_masks.keys())[0]
                masks = first_masks[first_set]

                # Get corresponding image for overlay
                cond = next(c for c in _global_state.conditions if c.name == first_cond)
                img_paths = cond.image_sets.get(first_set, {})
                nuc_path = img_paths.get(_global_state.channel_config.nuclear_suffix.upper())

                if nuc_path:
                    img = load_image(nuc_path)
                    preview = CellposeEngine.create_overlay(img, masks)

        # Build stats table
        stats_data = []
        total_cells = 0
        for cond_name, cond_masks in masks_dict.items():
            cond_cells = sum(m.max() for m in cond_masks.values())
            n_images = len(cond_masks)
            stats_data.append({
                'Condition': cond_name,
                'Images': n_images,
                'Cells Detected': cond_cells,
                'Avg Cells/Image': cond_cells / n_images if n_images > 0 else 0
            })
            total_cells += cond_cells

        stats_df = pd.DataFrame(stats_data)
        status = f"Segmentation complete: {total_cells} cells detected"

    except Exception as e:
        preview = None
        stats_df = pd.DataFrame()
        status = f"Error: {str(e)}"

    finally:
        _global_state.is_processing = False

    return preview, stats_df, status


def load_existing_masks(
    mask_folder: str,
    mask_pattern: str
) -> Tuple[pd.DataFrame, str]:
    """
    Load existing masks from folder.

    Returns:
        Tuple of (stats_table, status_message)
    """
    if not mask_folder or not Path(mask_folder).exists():
        return pd.DataFrame(), "Invalid mask folder"

    import glob

    mask_folder = Path(mask_folder)
    mask_files = list(mask_folder.glob(mask_pattern))

    if not mask_files:
        return pd.DataFrame(), f"No masks found matching pattern: {mask_pattern}"

    masks_dict = {}
    total_cells = 0

    for mask_file in mask_files:
        masks, meta = load_mask(mask_file)
        # Extract condition and image set from filename
        # Assumes format: condition_imageset_masks.tif
        name_parts = mask_file.stem.replace('_masks', '').rsplit('_', 1)
        if len(name_parts) == 2:
            cond, img_set = name_parts
        else:
            cond = "default"
            img_set = name_parts[0]

        if cond not in masks_dict:
            masks_dict[cond] = {}
        masks_dict[cond][img_set] = masks
        total_cells += meta.n_cells

    _global_state.masks = masks_dict

    # Build stats
    stats_data = []
    for cond_name, cond_masks in masks_dict.items():
        cond_cells = sum(m.max() for m in cond_masks.values())
        stats_data.append({
            'Condition': cond_name,
            'Images': len(cond_masks),
            'Cells': cond_cells
        })

    return pd.DataFrame(stats_data), f"Loaded {len(mask_files)} masks with {total_cells} cells"


# =============================================================================
# TAB 3: ROI EDITOR HANDLERS
# =============================================================================

def get_image_set_choices() -> List[str]:
    """Get available image sets for ROI editing."""
    choices = []
    for cond in _global_state.conditions:
        for img_set in cond.image_sets.keys():
            choices.append(f"{cond.name}/{img_set}")
    return choices


def launch_napari_for_editing(
    image_set_selection: str
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, str]:
    """
    Launch Napari for ROI editing.

    Returns:
        Tuple of (original_preview, edited_preview, stats_table, status)
    """
    if not image_set_selection:
        return None, None, pd.DataFrame(), "No image set selected"

    # Parse selection
    parts = image_set_selection.split('/')
    if len(parts) != 2:
        return None, None, pd.DataFrame(), "Invalid selection format"

    cond_name, img_set_name = parts

    # Get masks
    masks = _global_state.masks.get(cond_name, {}).get(img_set_name)
    if masks is None:
        return None, None, pd.DataFrame(), "No masks found for this image set"

    # Get image
    cond = next((c for c in _global_state.conditions if c.name == cond_name), None)
    if cond is None:
        return None, None, pd.DataFrame(), "Condition not found"

    img_paths = cond.image_sets.get(img_set_name, {})
    nuc_path = img_paths.get(_global_state.channel_config.nuclear_suffix.upper())

    if not nuc_path:
        return None, None, pd.DataFrame(), "Nuclear image not found"

    image = load_image(nuc_path)

    # Store original for comparison
    original_preview = CellposeEngine.create_overlay(image, masks)
    _global_state.current_preview = image
    _global_state.current_masks_preview = masks.copy()

    # Launch Napari (this will block until closed)
    try:
        edited_masks = launch_napari_editor(
            image=image,
            masks=masks,
            blocking=True
        )

        # Store edited masks
        _global_state.edited_masks[f"{cond_name}/{img_set_name}"] = edited_masks

        # Create edited preview
        edited_preview = CellposeEngine.create_overlay(image, edited_masks)

        # Calculate stats
        orig_cells = len(np.unique(masks)) - 1
        edit_cells = len(np.unique(edited_masks)) - 1
        changed_pixels = np.sum(masks != edited_masks)

        stats_df = pd.DataFrame([
            {'Metric': 'Original Cells', 'Original': orig_cells, 'Edited': orig_cells, 'Change': 0},
            {'Metric': 'Edited Cells', 'Original': orig_cells, 'Edited': edit_cells, 'Change': edit_cells - orig_cells},
            {'Metric': 'Pixels Changed', 'Original': 0, 'Edited': changed_pixels, 'Change': changed_pixels},
        ])

        status = f"Editing complete: {orig_cells} -> {edit_cells} cells"

    except Exception as e:
        edited_preview = original_preview
        stats_df = pd.DataFrame()
        status = f"Napari error: {str(e)}"

    return original_preview, edited_preview, stats_df, status


def save_edited_masks(image_set_selection: str) -> str:
    """Save edited masks back to the masks dict."""
    key = image_set_selection
    if key in _global_state.edited_masks:
        parts = key.split('/')
        if len(parts) == 2:
            cond_name, img_set_name = parts
            if cond_name not in _global_state.masks:
                _global_state.masks[cond_name] = {}
            _global_state.masks[cond_name][img_set_name] = _global_state.edited_masks[key]
            return f"Saved edits for {key}"
    return "No edits to save"


# =============================================================================
# TAB 4: QUANTIFICATION HANDLERS
# =============================================================================

def run_quantification(
    background_method: str,
    parallel_workers: int,
    use_vectorized: bool,
    output_folder: str,
    save_overlays: bool,
    save_rois: bool,
    progress=None
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Run quantification on segmented masks.

    Returns:
        Tuple of (condition_progress_df, live_results_df, status)
    """
    if not _global_state.masks:
        return pd.DataFrame(), pd.DataFrame(), "No masks available. Run segmentation first."

    if not _global_state.conditions:
        return pd.DataFrame(), pd.DataFrame(), "No experiment loaded."

    _global_state.is_processing = True

    pipeline = get_pipeline()
    pipeline.n_workers = parallel_workers

    output_path = Path(output_folder) if output_folder else _global_state.experiment_path / "output"

    def progress_callback(prog: ProcessingProgress):
        if progress:
            progress(prog.percent / 100, desc=f"{prog.condition}: {prog.message}")

    try:
        results_df = pipeline.run_quantification_only(
            conditions=_global_state.conditions,
            masks_dict=_global_state.masks,
            output_folder=output_path,
            progress_callback=progress_callback
        )

        _global_state.results = results_df

        # Build progress table
        progress_data = []
        for cond in _global_state.conditions:
            cond_results = results_df[results_df['Condition'] == cond.name] if len(results_df) > 0 else pd.DataFrame()
            progress_data.append({
                'Condition': cond.name,
                'Progress': '100%',
                'Cells': len(cond_results),
                'Time': 'Complete'
            })

        progress_df = pd.DataFrame(progress_data)

        # Preview of results (first 100 rows)
        preview_df = results_df.head(100) if len(results_df) > 0 else pd.DataFrame()

        status = f"Quantification complete: {len(results_df)} cells analyzed"

    except Exception as e:
        progress_df = pd.DataFrame()
        preview_df = pd.DataFrame()
        status = f"Error: {str(e)}"

    finally:
        _global_state.is_processing = False

    return progress_df, preview_df, status


# =============================================================================
# TAB 5: RESULTS HANDLERS
# =============================================================================

def get_results_table(
    condition_filter: List[str],
    marker_filter: List[str],
    min_area: float
) -> pd.DataFrame:
    """Get filtered results table."""
    if _global_state.results is None:
        return pd.DataFrame()

    df = _global_state.results.copy()

    # Apply condition filter
    if condition_filter and 'All' not in condition_filter:
        df = df[df['Condition'].isin(condition_filter)]

    # Apply area filter
    df = df[df['Area'] >= min_area]

    # Apply marker filter (select only certain columns)
    if marker_filter and 'All' not in marker_filter:
        base_cols = ['Condition', 'SegmentationType', 'ImageSet', 'CellID', 'Area']
        marker_cols = []
        for marker in marker_filter:
            marker_cols.extend([c for c in df.columns if c.startswith(marker)])
        df = df[base_cols + marker_cols]

    return df


def create_box_plot(marker: str) -> go.Figure:
    """Create box plot of CTCF by condition."""
    if _global_state.results is None or not marker:
        return go.Figure()

    df = _global_state.results
    col = f"{marker}_CTCF"

    if col not in df.columns:
        return go.Figure()

    fig = px.box(
        df,
        x='Condition',
        y=col,
        color='Condition',
        title=f'{marker} CTCF by Condition'
    )

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    return fig


def create_scatter_plot(x_var: str, y_var: str, color_by: str) -> go.Figure:
    """Create scatter plot of two variables."""
    if _global_state.results is None or not x_var or not y_var:
        return go.Figure()

    df = _global_state.results

    if x_var not in df.columns or y_var not in df.columns:
        return go.Figure()

    fig = px.scatter(
        df,
        x=x_var,
        y=y_var,
        color=color_by if color_by in df.columns else 'Condition',
        title=f'{y_var} vs {x_var}',
        opacity=0.6
    )

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    return fig


def create_heatmap() -> go.Figure:
    """Create correlation heatmap."""
    if _global_state.results is None:
        return go.Figure()

    df = _global_state.results

    # Get numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Remove ID columns
    numeric_cols = [c for c in numeric_cols if 'ID' not in c]

    if len(numeric_cols) < 2:
        return go.Figure()

    corr = df[numeric_cols].corr()

    fig = px.imshow(
        corr,
        title='Correlation Heatmap',
        color_continuous_scale='RdBu_r',
        zmin=-1,
        zmax=1
    )

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    return fig


def export_results_csv() -> str:
    """Export results to CSV and return file path."""
    if _global_state.results is None:
        return None

    output_path = _global_state.experiment_path / "output" if _global_state.experiment_path else Path(".")
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = output_path / f"cellquant_results_{timestamp}.csv"

    _global_state.results.to_csv(csv_path, index=False)

    return str(csv_path)


def export_results_excel() -> str:
    """Export results to Excel and return file path."""
    if _global_state.results is None:
        return None

    output_path = _global_state.experiment_path / "output" if _global_state.experiment_path else Path(".")
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    xlsx_path = output_path / f"cellquant_results_{timestamp}.xlsx"

    with pd.ExcelWriter(xlsx_path) as writer:
        _global_state.results.to_excel(writer, sheet_name='Results', index=False)

        # Add summary sheet
        summary = create_summary_statistics(_global_state.results)
        if len(summary) > 0:
            summary.to_excel(writer, sheet_name='Summary', index=False)

    return str(xlsx_path)


def get_marker_choices() -> List[str]:
    """Get available marker names from results."""
    if _global_state.results is None:
        return []

    ctcf_cols = [c for c in _global_state.results.columns if c.endswith('_CTCF')]
    return [c.replace('_CTCF', '') for c in ctcf_cols]


def get_condition_choices() -> List[str]:
    """Get available condition names."""
    if _global_state.results is None:
        return ['All']

    return ['All'] + _global_state.results['Condition'].unique().tolist()


def get_numeric_column_choices() -> List[str]:
    """Get numeric columns for plotting."""
    if _global_state.results is None:
        return []

    return _global_state.results.select_dtypes(include=[np.number]).columns.tolist()
