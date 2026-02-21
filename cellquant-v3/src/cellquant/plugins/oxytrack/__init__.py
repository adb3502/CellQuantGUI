"""OxyTrack – Oxysterol-Senescence Experiment Tracker.

Tracks oxysterol treatment experiments linked to cellular senescence,
managing treatment conditions, time-courses, marker panels, and
connecting CellQuant quantification results to experiment metadata.
"""

from cellquant.plugins.oxytrack.plugin import OxyTrackPlugin

__all__ = ["OxyTrackPlugin"]
