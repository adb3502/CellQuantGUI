"""
OxyTrack - Oxysterol-Senescence Experiment Tracker

A FUSION plugin for managing oxysterol treatment experiments,
tracking senescence markers, and linking quantification results
to experimental conditions.
"""


def __getattr__(name):
    if name == "OxyTrackPlugin":
        from cellquant_enterprise.plugins.oxytrack.plugin import OxyTrackPlugin
        return OxyTrackPlugin
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["OxyTrackPlugin"]
