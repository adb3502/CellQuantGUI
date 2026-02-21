"""
FUSION Plugin System for CellQuant Enterprise.

FUSION (Flexible, User-extensible, Scientific Integration and Operations Network)
provides a plugin architecture for extending CellQuant with domain-specific tools.

Available plugins:
- OxyTrack: Oxysterol-Senescence Experiment Tracker
- SenescenceDB: Personal Research Knowledge Base
"""

from cellquant_enterprise.plugins.base import FusionPlugin, PluginRegistry

__all__ = ["FusionPlugin", "PluginRegistry"]
