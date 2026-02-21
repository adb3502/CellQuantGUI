"""
OxyTrack plugin entry point.
"""

from typing import Dict

from cellquant_enterprise.plugins.base import FusionPlugin
from cellquant_enterprise.plugins.oxytrack.storage import OxyTrackStore
from cellquant_enterprise.plugins.oxytrack.ui import create_oxytrack_tab


class OxyTrackPlugin(FusionPlugin):
    """Oxysterol-Senescence Experiment Tracker."""

    def __init__(self):
        self._store: OxyTrackStore | None = None

    @property
    def name(self) -> str:
        return "oxytrack"

    @property
    def display_name(self) -> str:
        return "OxyTrack"

    @property
    def description(self) -> str:
        return "Track oxysterol treatments, senescence markers, and link CellQuant results to experiments."

    @property
    def store(self) -> OxyTrackStore:
        if self._store is None:
            self._store = OxyTrackStore()
        return self._store

    def create_tab(self) -> Dict:
        return create_oxytrack_tab(self.store)

    def on_load(self) -> None:
        # Pre-warm the store so the JSON is loaded once at startup
        _ = self.store
