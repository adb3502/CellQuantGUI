"""Thumbnail generation for image grid."""

from pathlib import Path
import numpy as np
from PIL import Image


def generate_thumbnail(
    image_path: Path,
    output_path: Path,
    size: int = 256,
    quality: int = 85,
):
    """
    Generate a JPEG thumbnail from a TIFF image.

    Args:
        image_path: Source TIFF path
        output_path: Output JPEG path
        size: Thumbnail size (square, default 256)
        quality: JPEG quality (default 85)
    """
    from cellquant.core.io.image_loader import load_image, normalize_image

    img = load_image(image_path)
    img_norm = normalize_image(img)
    img_uint8 = (img_norm * 255).astype(np.uint8)

    pil_img = Image.fromarray(img_uint8)
    pil_img.thumbnail((size, size), Image.LANCZOS)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pil_img.save(output_path, "JPEG", quality=quality)
