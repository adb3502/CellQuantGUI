"""Server-side image rendering with ImageJ-quality auto-contrast."""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import threading
import numpy as np
from PIL import Image


def imagej_auto_contrast(img: np.ndarray) -> np.ndarray:
    """
    ImageJ-style auto brightness/contrast.

    Uses 0.35% saturation on each side — same algorithm as
    ImageJ's "Auto" button in Brightness/Contrast dialog.
    Returns uint8 ready for display.
    """
    img_float = img.astype(np.float64)

    p_low, p_high = np.percentile(img_float, (0.35, 99.65))

    if p_high <= p_low:
        p_low = img_float.min()
        p_high = img_float.max()
    if p_high <= p_low:
        return np.zeros(img.shape, dtype=np.uint8)

    normalized = (img_float - p_low) / (p_high - p_low)
    return (np.clip(normalized, 0.0, 1.0) * 255).astype(np.uint8)


def _apply_lut(gray_uint8: np.ndarray, hex_color: str) -> np.ndarray:
    """Apply a false-color LUT to a grayscale uint8 image.

    Returns an (H, W, 3) uint8 RGB array.
    """
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    f = gray_uint8.astype(np.float32) / 255.0
    rgb = np.stack([
        (f * r).astype(np.uint8),
        (f * g).astype(np.uint8),
        (f * b).astype(np.uint8),
    ], axis=-1)
    return rgb


def render_image(
    image_path: Path,
    output_path: Path,
    max_size: int = 0,
    quality: int = 95,
    color: str | None = None,
) -> Path:
    """
    Render a TIFF with ImageJ-identical auto-contrast.

    Args:
        max_size: If > 0, resize to fit within this many px (preview mode).
                  If 0, render at native resolution (full-res mode).
        quality: JPEG quality (95 for preview, 100 for full-res).
        color: Hex color (e.g. '#00FF00') for false-color LUT.
               None or '#FFFFFF' renders grayscale.
    """
    if output_path.exists():
        return output_path

    from cellquant.core.io.image_loader import load_image

    img = load_image(image_path)
    img_uint8 = imagej_auto_contrast(img)

    # Apply false-color LUT if requested (skip for white/gray = standard grayscale)
    if color and color.lstrip('#').upper() not in ('FFFFFF', 'FFF', 'CCCCCC'):
        rgb = _apply_lut(img_uint8, color)
        pil_img = Image.fromarray(rgb, mode='RGB')
    else:
        pil_img = Image.fromarray(img_uint8, mode='L')

    if max_size > 0 and max(pil_img.size) > max_size:
        pil_img.thumbnail((max_size, max_size), Image.LANCZOS)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pil_img.save(output_path, "JPEG", quality=quality)
    return output_path


# Background pre-renderer
_prerender_executor = ThreadPoolExecutor(max_workers=4)


def _render_one(src_path: str, dst_path: str, max_size: int, quality: int):
    """Worker function for background pre-rendering."""
    try:
        render_image(Path(src_path), Path(dst_path), max_size=max_size, quality=quality)
    except Exception as e:
        print(f"Pre-render failed {src_path}: {e}")


def prerender_all(session, max_size: int = 800, quality: int = 95) -> None:
    """
    Kick off background pre-rendering of all images in a session.

    Uses a thread pool with 4 workers. Non-blocking — returns
    immediately. Renders display-size WebP previews (~30-60KB each)
    for instant browsing. On-demand render is the fallback for any
    that aren't ready yet.
    """
    tasks = []
    for cond_name, cond_data in session.conditions.items():
        image_sets = cond_data.get("image_sets", {})
        for base_name, channels in image_sets.items():
            cache_dir = session.directory / "renders" / cond_name / base_name
            cache_dir.mkdir(parents=True, exist_ok=True)
            for suffix, file_path in channels.items():
                cache_path = cache_dir / f"{suffix}_{max_size}_gray.jpg"
                if not cache_path.exists():
                    tasks.append((str(file_path), str(cache_path)))

    if not tasks:
        return

    print(f"Pre-rendering {len(tasks)} images in background (4 workers)...")

    def _run():
        for src, dst in tasks:
            _prerender_executor.submit(_render_one, src, dst, max_size, quality)

    threading.Thread(target=_run, daemon=True).start()
