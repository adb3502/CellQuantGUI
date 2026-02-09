"""Trackastra wrapper for cell tracking."""

from typing import Optional, Tuple
import numpy as np


class CellTracker:
    """
    Wrapper around Trackastra for cell tracking in timelapse data.

    Loads the pretrained model once and provides a simple API
    for tracking segmented cells across frames.

    Example:
        >>> tracker = CellTracker(model="general_2d")
        >>> tracked_masks, graph = tracker.track(images, masks, mode="greedy")
    """

    def __init__(self, model: str = "general_2d", device: str = "automatic"):
        try:
            from trackastra.model import Trackastra
        except ImportError:
            raise ImportError(
                "trackastra required for cell tracking. "
                "Install with: pip install trackastra"
            )

        self._model = Trackastra.from_pretrained(model, device=device)
        self.model_name = model

    def track(
        self,
        images: np.ndarray,
        masks: np.ndarray,
        mode: str = "greedy",
    ) -> Tuple[object, np.ndarray]:
        """
        Track cells across a timelapse.

        Args:
            images: Timelapse array (T, Y, X)
            masks: Segmentation masks (T, Y, X)
            mode: Linking mode - "greedy", "greedy_nodiv", or "ilp"

        Returns:
            Tuple of (track_graph, tracked_masks)
        """
        track_graph, masks_tracked = self._model.track(images, masks, mode=mode)
        return track_graph, masks_tracked

    def track_and_export(
        self,
        images: np.ndarray,
        masks: np.ndarray,
        output_dir: str,
        mode: str = "greedy",
    ) -> Tuple[object, np.ndarray]:
        """Track and export results in CTC format."""
        from trackastra.tracking import graph_to_ctc

        track_graph, masks_tracked = self.track(images, masks, mode=mode)
        ctc_tracks, ctc_masks = graph_to_ctc(
            track_graph, masks_tracked, outdir=output_dir
        )
        return track_graph, masks_tracked
