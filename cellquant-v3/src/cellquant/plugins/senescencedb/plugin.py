"""SenescenceDB FUSION plugin entry-point."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter

from cellquant.plugins.base import FusionPlugin, PluginMetadata
from cellquant.plugins.senescencedb.router import build_router, set_store
from cellquant.plugins.senescencedb.store import SenescenceDBStore


class SenescenceDBPlugin(FusionPlugin):
    """Personal Research Knowledge Base for senescence research."""

    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="SenescenceDB",
            slug="senescencedb",
            version="1.0.0",
            description=(
                "Structured knowledge base for senescence research – papers, "
                "markers, pathways, protocols, and notes with full-text search "
                "and cross-referencing."
            ),
        )

    def create_router(self) -> APIRouter:
        return build_router()

    def activate(self, data_dir: Path) -> None:
        store = SenescenceDBStore(data_dir)
        set_store(store)
