"""FUSION plugin system for CellQuant.

Plugins extend CellQuant with domain-specific experiment tracking,
knowledge management, and analysis capabilities. Each plugin provides
its own FastAPI router, data models, and storage layer.
"""

from cellquant.plugins.base import FusionPlugin, PluginMetadata
from cellquant.plugins.registry import PluginRegistry

__all__ = ["FusionPlugin", "PluginMetadata", "PluginRegistry"]
