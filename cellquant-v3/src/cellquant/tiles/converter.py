"""Convert TIFF images to DZI tile pyramids for web viewing."""

import math
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


def generate_tiles_for_image(
    image_path: Path,
    output_dir: Path,
    tile_size: int = 256,
    fmt: str = "png",
) -> dict:
    """
    Convert a TIFF to a DZI tile pyramid.

    Args:
        image_path: Path to source TIFF
        output_dir: Directory for tile output
        tile_size: Tile dimensions (default 256)
        fmt: Output format (png or jpg)

    Returns:
        Metadata dict with width, height, tile_size, max_level
    """
    from cellquant.core.io.image_loader import load_image, normalize_image

    img = load_image(image_path)
    img_norm = normalize_image(img)
    img_uint8 = (img_norm * 255).astype(np.uint8)

    height, width = img_uint8.shape[:2]
    max_level = math.ceil(math.log2(max(width, height, 1)))

    for level in range(max_level + 1):
        scale = 2 ** (max_level - level)
        level_w = max(1, width // scale)
        level_h = max(1, height // scale)

        # Resize for this level
        pil_img = Image.fromarray(img_uint8)
        level_img = pil_img.resize((level_w, level_h), Image.LANCZOS)
        level_arr = np.array(level_img)

        level_dir = output_dir / str(level)
        level_dir.mkdir(parents=True, exist_ok=True)

        # Cut into tiles
        for ty, y in enumerate(range(0, level_h, tile_size)):
            for tx, x in enumerate(range(0, level_w, tile_size)):
                tile = level_arr[y : y + tile_size, x : x + tile_size]
                tile_path = level_dir / f"{tx}_{ty}.{fmt}"
                Image.fromarray(tile).save(tile_path)

    return {
        "width": width,
        "height": height,
        "tile_size": tile_size,
        "max_level": max_level,
    }


def generate_mask_tiles(
    masks: np.ndarray,
    output_dir: Path,
    tile_size: int = 256,
    alpha: int = 128,
) -> dict:
    """
    Generate RGBA tile pyramid from a label mask.

    Each cell gets a deterministic color with the given alpha.

    Args:
        masks: 2D label mask (int32)
        output_dir: Directory for tile output
        tile_size: Tile dimensions
        alpha: Alpha value for cell pixels (0-255)

    Returns:
        Metadata dict
    """
    height, width = masks.shape
    max_level = math.ceil(math.log2(max(width, height, 1)))

    # Create RGBA image
    rgba = _masks_to_rgba(masks, alpha)

    for level in range(max_level + 1):
        scale = 2 ** (max_level - level)
        level_w = max(1, width // scale)
        level_h = max(1, height // scale)

        pil_img = Image.fromarray(rgba)
        level_img = pil_img.resize((level_w, level_h), Image.NEAREST)
        level_arr = np.array(level_img)

        level_dir = output_dir / str(level)
        level_dir.mkdir(parents=True, exist_ok=True)

        for ty, y in enumerate(range(0, level_h, tile_size)):
            for tx, x in enumerate(range(0, level_w, tile_size)):
                tile = level_arr[y : y + tile_size, x : x + tile_size]
                tile_path = level_dir / f"{tx}_{ty}.png"
                Image.fromarray(tile).save(tile_path)

    return {"width": width, "height": height, "tile_size": tile_size, "max_level": max_level}


def _masks_to_rgba(masks: np.ndarray, alpha: int = 128) -> np.ndarray:
    """Convert label mask to RGBA array with deterministic per-cell colors."""
    rgba = np.zeros((*masks.shape, 4), dtype=np.uint8)

    for cell_id in np.unique(masks):
        if cell_id == 0:
            continue
        r = (cell_id * 67 + 13) % 256
        g = (cell_id * 137 + 43) % 256
        b = (cell_id * 209 + 97) % 256
        pixels = masks == cell_id
        rgba[pixels] = [r, g, b, alpha]

    return rgba
