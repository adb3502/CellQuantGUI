"""Base class and metadata for FUSION plugins."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from fastapi import APIRouter


@dataclass
class PluginMetadata:
    """Descriptive metadata for a FUSION plugin."""

    name: str
    slug: str  # URL-safe identifier, e.g. "oxytrack"
    version: str
    description: str
    author: str = "CellQuant Contributors"


class FusionPlugin(ABC):
    """Abstract base for all FUSION plugin modules.

    Subclasses must implement ``metadata`` and ``create_router`` at minimum.
    The plugin registry calls ``activate`` after the router is mounted.
    """

    @abstractmethod
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""

    @abstractmethod
    def create_router(self) -> APIRouter:
        """Return a FastAPI router with the plugin's endpoints."""

    def activate(self, data_dir: Path) -> None:
        """Called once after the router is mounted.

        Use this to initialise databases, create storage directories, etc.
        ``data_dir`` is the plugin-specific data directory inside
        ``~/.cellquant/plugins/<slug>/``.
        """

    def deactivate(self) -> None:
        """Called during graceful shutdown (optional)."""
