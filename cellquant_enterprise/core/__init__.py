"""Core engine modules for CellQuant Enterprise."""

# Use lazy imports to allow importing without all dependencies
__all__ = [
    "load_image",
    "normalize_image",
    "load_mask",
    "save_mask",
    "calculate_ctcf_vectorized",
    "estimate_background",
    "CellposeEngine",
]

def __getattr__(name):
    if name in ("load_image", "normalize_image"):
        from cellquant_enterprise.core.io.image_loader import load_image, normalize_image
        return locals()[name]
    elif name in ("load_mask", "save_mask"):
        from cellquant_enterprise.core.io.mask_io import load_mask, save_mask
        return locals()[name]
    elif name == "calculate_ctcf_vectorized":
        from cellquant_enterprise.core.quantification.ctcf import calculate_ctcf_vectorized
        return calculate_ctcf_vectorized
    elif name == "estimate_background":
        from cellquant_enterprise.core.quantification.background import estimate_background
        return estimate_background
    elif name == "CellposeEngine":
        from cellquant_enterprise.core.segmentation.cellpose_engine import CellposeEngine
        return CellposeEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
