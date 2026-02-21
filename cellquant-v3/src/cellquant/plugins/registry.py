"""Plugin registry – discovers, loads, and registers FUSION plugins."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI

from cellquant.plugins.base import FusionPlugin, PluginMetadata


class PluginRegistry:
    """Central registry that manages the lifecycle of FUSION plugins."""

    def __init__(self) -> None:
        self._plugins: Dict[str, FusionPlugin] = {}
        self._base_data_dir = Path.home() / ".cellquant" / "plugins"

    # -- registration --------------------------------------------------------

    def register(self, plugin: FusionPlugin) -> None:
        """Register a plugin instance."""
        meta = plugin.metadata()
        if meta.slug in self._plugins:
            raise ValueError(f"Plugin '{meta.slug}' is already registered")
        self._plugins[meta.slug] = plugin

    # -- mounting ------------------------------------------------------------

    def mount_all(self, app: FastAPI, prefix: str = "/api/v1/plugins") -> None:
        """Mount every registered plugin's router on *app* and activate it."""
        for slug, plugin in self._plugins.items():
            meta = plugin.metadata()
            router = plugin.create_router()
            app.include_router(router, prefix=f"{prefix}/{slug}", tags=[meta.name])

            data_dir = self._base_data_dir / slug
            data_dir.mkdir(parents=True, exist_ok=True)
            plugin.activate(data_dir)

    def deactivate_all(self) -> None:
        """Gracefully shut down every plugin."""
        for plugin in self._plugins.values():
            plugin.deactivate()

    # -- introspection -------------------------------------------------------

    def list_plugins(self) -> List[PluginMetadata]:
        """Return metadata for all registered plugins."""
        return [p.metadata() for p in self._plugins.values()]

    def get(self, slug: str) -> FusionPlugin | None:
        return self._plugins.get(slug)
