"""
Base classes for the FUSION plugin system.

All FUSION plugins inherit from FusionPlugin and register via PluginRegistry.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Type


class FusionPlugin(ABC):
    """Base class for all FUSION plugins."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for the plugin (e.g. 'oxytrack')."""

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable name shown in the UI tab."""

    @property
    @abstractmethod
    def description(self) -> str:
        """One-line description of the plugin."""

    @abstractmethod
    def create_tab(self) -> Dict:
        """
        Build the Gradio UI components for this plugin's tab.

        Called inside a `gr.Tab(...)` context. Should create all Gradio
        components, wire up event handlers, and return a dict of the
        key components for external reference.
        """

    def on_load(self) -> None:
        """Optional hook called when the plugin is first loaded."""

    def on_unload(self) -> None:
        """Optional hook called when the plugin is being unloaded."""


class PluginRegistry:
    """Registry that discovers and manages FUSION plugins."""

    def __init__(self):
        self._plugins: Dict[str, FusionPlugin] = {}

    def register(self, plugin: FusionPlugin) -> None:
        """Register a plugin instance."""
        self._plugins[plugin.name] = plugin
        plugin.on_load()

    def unregister(self, name: str) -> None:
        """Unregister a plugin by name."""
        plugin = self._plugins.pop(name, None)
        if plugin:
            plugin.on_unload()

    def get(self, name: str) -> Optional[FusionPlugin]:
        """Get a registered plugin by name."""
        return self._plugins.get(name)

    @property
    def plugins(self) -> List[FusionPlugin]:
        """Return all registered plugins in registration order."""
        return list(self._plugins.values())

    def create_all_tabs(self) -> Dict[str, Dict]:
        """
        Create Gradio tabs for every registered plugin.

        Returns a dict mapping plugin name -> component dict.
        Must be called inside a gr.Tabs() context.
        """
        import gradio as gr

        components = {}
        for plugin in self._plugins.values():
            with gr.Tab(plugin.display_name, id=f"tab-fusion-{plugin.name}"):
                components[plugin.name] = plugin.create_tab()
        return components


# Global registry instance
registry = PluginRegistry()


def get_registry() -> PluginRegistry:
    """Return the global plugin registry."""
    return registry
