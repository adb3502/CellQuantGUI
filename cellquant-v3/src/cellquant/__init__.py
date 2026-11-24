"""CellQuant - High-throughput cell quantification and tracking for microscopy."""

# Single source of truth is pyproject.toml. This was a hardcoded literal and
# drifted: bumping pyproject to 3.1.0 left /api/health still reporting 3.0.0,
# because that endpoint reads THIS value, not the package metadata.
try:
    from importlib.metadata import version as _version

    __version__ = _version("cellquant")
except Exception:  # not installed (e.g. running from a source checkout)
    __version__ = "3.1.0"
