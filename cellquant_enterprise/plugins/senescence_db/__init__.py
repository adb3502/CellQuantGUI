"""
SenescenceDB - Personal Research Knowledge Base

A FUSION plugin for organizing senescence research knowledge:
papers, protocols, findings, and their links to experiments.
"""


def __getattr__(name):
    if name == "SenescenceDBPlugin":
        from cellquant_enterprise.plugins.senescence_db.plugin import SenescenceDBPlugin
        return SenescenceDBPlugin
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["SenescenceDBPlugin"]
