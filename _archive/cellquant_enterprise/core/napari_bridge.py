"""
Napari integration for interactive ROI editing.

Provides a bridge between the Gradio web UI and Napari's powerful
label editing tools for semi-supervised mask refinement.
"""

from typing import Optional, Dict, List, Tuple, Any, Callable
from pathlib import Path
import numpy as np
import threading
import warnings

try:
    import napari
    from napari.layers import Labels, Image
    HAS_NAPARI = True
except ImportError:
    HAS_NAPARI = False


class NapariBridge:
    """
    Bridge for launching and managing Napari editing sessions.

    Provides methods to:
    - Launch Napari with images and masks
    - Track edits made in Napari
    - Import edited masks back to the application

    Example:
        >>> bridge = NapariBridge()
        >>> edited_masks = bridge.edit_masks(image, masks)
    """

    def __init__(self):
        if not HAS_NAPARI:
            raise ImportError(
                "Napari required for ROI editing. Install with: pip install napari[all]"
            )

        self._viewer = None
        self._original_masks = None
        self._edited_masks = None
        self._edit_history = []
        self._is_editing = False
        self._on_close_callback = None

    @property
    def is_editing(self) -> bool:
        """Check if an editing session is active."""
        return self._is_editing and self._viewer is not None

    def launch_editor(
        self,
        image: np.ndarray,
        masks: np.ndarray,
        channel_names: Optional[List[str]] = None,
        markers: Optional[Dict[str, np.ndarray]] = None,
        title: str = "CellQuant ROI Editor",
        on_close: Optional[Callable[[np.ndarray], None]] = None
    ) -> None:
        """
        Launch Napari viewer with image and editable masks.

        Args:
            image: Image array (Y, X) or (C, Y, X)
            masks: Label mask to edit
            channel_names: Optional names for each channel
            markers: Optional dict of additional marker images to display
            title: Viewer window title
            on_close: Callback when viewer closes, receives edited masks
        """
        if self.is_editing:
            warnings.warn("Editing session already active. Close it first.")
            return

        self._original_masks = masks.copy()
        self._edited_masks = masks.copy()
        self._edit_history = []
        self._on_close_callback = on_close
        self._is_editing = True

        # Create viewer
        self._viewer = napari.Viewer(title=title)

        # Add image layers
        self._add_image_layers(image, channel_names, markers)

        # Add editable labels layer
        self._labels_layer = self._viewer.add_labels(
            masks,
            name="Cell Masks (Editable)",
            opacity=0.5
        )

        # Configure labels layer for editing
        self._labels_layer.mode = 'paint'
        self._labels_layer.brush_size = 10

        # Connect to layer events for tracking changes
        self._labels_layer.events.data.connect(self._on_masks_changed)

        # Connect to viewer close event
        self._viewer.window._qt_window.destroyed.connect(self._on_viewer_closed)

        # Add custom keybindings
        self._add_keybindings()

        # Show instructions
        self._show_instructions()

    def _add_image_layers(
        self,
        image: np.ndarray,
        channel_names: Optional[List[str]],
        markers: Optional[Dict[str, np.ndarray]]
    ):
        """Add image layers to the viewer."""
        # Define fluorescent colormaps
        colormaps = ['cyan', 'green', 'magenta', 'yellow', 'red']

        if image.ndim == 2:
            self._viewer.add_image(image, name="Image", colormap='gray')

        elif image.ndim == 3:
            # Multi-channel image
            n_channels = image.shape[0]
            if channel_names is None:
                channel_names = [f"Channel {i}" for i in range(n_channels)]

            for i in range(n_channels):
                cmap = colormaps[i % len(colormaps)]
                name = channel_names[i] if i < len(channel_names) else f"Channel {i}"
                self._viewer.add_image(
                    image[i],
                    name=name,
                    colormap=cmap,
                    blending='additive',
                    visible=(i < 3)  # Show first 3 channels by default
                )

        # Add additional marker images
        if markers:
            for name, marker_img in markers.items():
                self._viewer.add_image(
                    marker_img,
                    name=name,
                    colormap='magenta',
                    blending='additive',
                    visible=False
                )

    def _add_keybindings(self):
        """Add custom keybindings for common operations."""

        @self._viewer.bind_key('s')
        def save_and_close(viewer):
            """Save edits and close viewer."""
            self._save_and_close()

        @self._viewer.bind_key('Escape')
        def cancel_and_close(viewer):
            """Discard edits and close viewer."""
            self._cancel_and_close()

        @self._viewer.bind_key('r')
        def reset_masks(viewer):
            """Reset to original masks."""
            self._labels_layer.data = self._original_masks.copy()
            self._edited_masks = self._original_masks.copy()

        @self._viewer.bind_key('f')
        def fill_mode(viewer):
            """Switch to fill mode."""
            self._labels_layer.mode = 'fill'

        @self._viewer.bind_key('p')
        def paint_mode(viewer):
            """Switch to paint mode."""
            self._labels_layer.mode = 'paint'

        @self._viewer.bind_key('e')
        def erase_mode(viewer):
            """Switch to erase mode."""
            self._labels_layer.mode = 'erase'

        @self._viewer.bind_key('k')
        def pick_mode(viewer):
            """Switch to pick mode (select existing label)."""
            self._labels_layer.mode = 'pick'

    def _show_instructions(self):
        """Show editing instructions in status bar."""
        instructions = (
            "ROI Editor | P: Paint | F: Fill | E: Erase | K: Pick Label | "
            "S: Save & Close | Esc: Cancel | R: Reset | Ctrl+Z: Undo"
        )
        self._viewer.status = instructions

    def _on_masks_changed(self, event):
        """Handle mask edit events."""
        self._edited_masks = self._labels_layer.data.copy()
        self._edit_history.append(('edit', self._edited_masks.copy()))

    def _on_viewer_closed(self):
        """Handle viewer close event."""
        self._is_editing = False

        if self._on_close_callback and self._edited_masks is not None:
            self._on_close_callback(self._edited_masks)

        self._viewer = None

    def _save_and_close(self):
        """Save edits and close viewer."""
        self._edited_masks = self._labels_layer.data.copy()
        if self._viewer:
            self._viewer.close()

    def _cancel_and_close(self):
        """Discard edits and close viewer."""
        self._edited_masks = self._original_masks.copy()
        if self._viewer:
            self._viewer.close()

    def get_edited_masks(self) -> Optional[np.ndarray]:
        """
        Get the edited masks.

        Returns:
            Edited mask array or None if no edits
        """
        return self._edited_masks

    def get_edit_summary(self) -> Dict[str, Any]:
        """
        Get summary of edits made.

        Returns:
            Dictionary with edit statistics
        """
        if self._original_masks is None or self._edited_masks is None:
            return {}

        orig_cells = len(np.unique(self._original_masks)) - 1
        edit_cells = len(np.unique(self._edited_masks)) - 1

        # Calculate changed pixels
        changed = np.sum(self._original_masks != self._edited_masks)
        total = self._original_masks.size

        return {
            'original_cells': orig_cells,
            'edited_cells': edit_cells,
            'cells_added': max(0, edit_cells - orig_cells),
            'cells_removed': max(0, orig_cells - edit_cells),
            'pixels_changed': changed,
            'percent_changed': 100 * changed / total,
            'n_edits': len(self._edit_history)
        }


def launch_napari_editor(
    image: np.ndarray,
    masks: np.ndarray,
    channel_names: Optional[List[str]] = None,
    markers: Optional[Dict[str, np.ndarray]] = None,
    blocking: bool = True
) -> np.ndarray:
    """
    Convenience function to launch Napari editor and get edited masks.

    Args:
        image: Image array
        masks: Label mask to edit
        channel_names: Optional channel names
        markers: Optional additional marker images
        blocking: Whether to block until editing is complete

    Returns:
        Edited mask array
    """
    result = {'masks': masks.copy()}

    def on_close(edited_masks):
        result['masks'] = edited_masks

    bridge = NapariBridge()
    bridge.launch_editor(
        image=image,
        masks=masks,
        channel_names=channel_names,
        markers=markers,
        on_close=on_close
    )

    if blocking:
        napari.run()

    return result['masks']


def save_napari_session(
    masks: np.ndarray,
    output_path: Path,
    include_overlay: bool = False,
    image: Optional[np.ndarray] = None
) -> Path:
    """
    Save edited masks from Napari session.

    Args:
        masks: Edited mask array
        output_path: Output path for mask file
        include_overlay: Whether to save overlay image
        image: Original image (required if include_overlay=True)

    Returns:
        Path to saved file
    """
    from cellquant_enterprise.core.io.mask_io import save_mask

    output_path = Path(output_path)
    save_mask(masks, output_path)

    if include_overlay and image is not None:
        from cellquant_enterprise.core.segmentation.cellpose_engine import CellposeEngine
        overlay = CellposeEngine.create_overlay(image, masks)

        overlay_path = output_path.with_suffix('.overlay.png')
        from skimage import io
        io.imsave(str(overlay_path), overlay)

    return output_path


def load_napari_labels(path: Path) -> np.ndarray:
    """
    Load masks saved from Napari.

    Args:
        path: Path to mask file

    Returns:
        Mask array
    """
    from cellquant_enterprise.core.io.mask_io import load_mask

    masks, _ = load_mask(path)
    return masks


class NapariEditorThread(threading.Thread):
    """
    Thread wrapper for running Napari in background.

    Useful for integrating with Gradio without blocking the main thread.
    """

    def __init__(
        self,
        image: np.ndarray,
        masks: np.ndarray,
        channel_names: Optional[List[str]] = None,
        on_complete: Optional[Callable[[np.ndarray], None]] = None
    ):
        super().__init__(daemon=True)
        self.image = image
        self.masks = masks
        self.channel_names = channel_names
        self.on_complete = on_complete
        self.result_masks = None

    def run(self):
        """Run the editor in a separate thread."""
        def on_close(edited_masks):
            self.result_masks = edited_masks
            if self.on_complete:
                self.on_complete(edited_masks)

        bridge = NapariBridge()
        bridge.launch_editor(
            image=self.image,
            masks=self.masks,
            channel_names=self.channel_names,
            on_close=on_close
        )

        napari.run()
