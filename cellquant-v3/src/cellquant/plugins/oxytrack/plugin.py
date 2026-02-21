"""OxyTrack FUSION plugin entry-point."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter

from cellquant.plugins.base import FusionPlugin, PluginMetadata
from cellquant.plugins.oxytrack.router import build_router, set_store
from cellquant.plugins.oxytrack.store import OxyTrackStore


class OxyTrackPlugin(FusionPlugin):
    """Oxysterol-Senescence Experiment Tracker plugin."""

    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="OxyTrack",
            slug="oxytrack",
            version="1.0.0",
            description=(
                "Track oxysterol treatment experiments linked to cellular "
                "senescence – manage conditions, time-courses, marker panels, "
                "and bridge CellQuant quantification results."
            ),
        )

    def create_router(self) -> APIRouter:
        return build_router()

    def activate(self, data_dir: Path) -> None:
        store = OxyTrackStore(data_dir)
        set_store(store)
