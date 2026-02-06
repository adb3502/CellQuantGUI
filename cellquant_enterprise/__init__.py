"""
CellQuant - High-throughput cell quantification platform

Fast, semi-supervised cell analysis with:
- Vectorized CTCF quantification
- Batch GPU segmentation via Cellpose
- Napari integration for ROI editing
- Modern Gradio web interface
"""

__version__ = "2.0.0"
__author__ = "CellQuant Team"

# Lazy imports to avoid import errors when dependencies aren't installed
def __getattr__(name):
    if name == "BatchPipeline":
        from cellquant_enterprise.core.pipeline import BatchPipeline
        return BatchPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["BatchPipeline", "__version__"]
