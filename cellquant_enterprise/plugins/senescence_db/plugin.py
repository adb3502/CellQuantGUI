"""
SenescenceDB plugin entry point.
"""

from typing import Dict

from cellquant_enterprise.plugins.base import FusionPlugin
from cellquant_enterprise.plugins.senescence_db.storage import SenescenceDBStore
from cellquant_enterprise.plugins.senescence_db.ui import create_senescence_db_tab


class SenescenceDBPlugin(FusionPlugin):
    """Personal Research Knowledge Base for senescence research."""

    def __init__(self):
        self._store: SenescenceDBStore | None = None

    @property
    def name(self) -> str:
        return "senescence_db"

    @property
    def display_name(self) -> str:
        return "SenescenceDB"

    @property
    def description(self) -> str:
        return "Organize papers, protocols, and findings for senescence research."

    @property
    def store(self) -> SenescenceDBStore:
        if self._store is None:
            self._store = SenescenceDBStore()
        return self._store

    def create_tab(self) -> Dict:
        return create_senescence_db_tab(self.store)

    def on_load(self) -> None:
        _ = self.store
