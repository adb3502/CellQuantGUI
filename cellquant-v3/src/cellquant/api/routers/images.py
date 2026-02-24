"""Image tile serving and server-side rendering."""

from pathlib import Path
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from cellquant.api.dependencies import get_session

router = APIRouter(prefix="/images", tags=["images"])


def _resolve_image_path(session, condition: str, base_name: str, channel: str) -> Path:
    """Resolve a channel to its file path on disk."""
    cond_data = session.conditions.get(condition, {})
    image_sets = cond_data.get("image_sets", {})
    channels = image_sets.get(base_name, {})
    image_path = channels.get(channel) or channels.get(channel.upper())

    if not image_path or not Path(image_path).exists():
        raise HTTPException(404, f"Image not found: {condition}/{base_name}/{channel}")
    return Path(image_path)


@router.get("/{session_id}/{condition}/{base_name}/{channel}/render")
async def render_preview(
    session_id: str,
    condition: str,
    base_name: str,
    channel: str,
    size: int = Query(default=800, ge=0, le=4096),
    color: str | None = Query(default=None, description="Hex color for LUT, e.g. #00FF00"),
):
    """
    Render a TIFF with ImageJ-identical auto-contrast.

    size=800 (default): display-size preview, ~30-60KB, instant browsing.
    size=0: full native resolution, lossless.
    color: optional hex color for false-color LUT rendering.
    """
    session = get_session(session_id)
    src = _resolve_image_path(session, condition, base_name, channel)

    cache_dir = session.directory / "renders" / condition / base_name
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Include color in cache key so different LUTs get separate files
    color_tag = color.lstrip('#') if color else "gray"
    cache_path = cache_dir / f"{channel}_{size}_{color_tag}.jpg"

    if not cache_path.exists():
        from cellquant.tiles.thumbnail import render_image
        render_image(src, cache_path, max_size=size if size > 0 else 0, color=color)

    return FileResponse(
        cache_path,
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@router.get("/{session_id}/{condition}/{base_name}/{channel}/tile/{level}/{col}_{row}.png")
async def get_tile(
    session_id: str,
    condition: str,
    base_name: str,
    channel: str,
    level: int,
    col: int,
    row: int,
):
    """Serve a DZI tile. Generates on demand if not cached."""
    session = get_session(session_id)
    tile_dir = session.get_tile_dir(condition, base_name, channel)
    tile_path = tile_dir / str(level) / f"{col}_{row}.png"

    if not tile_path.exists():
        from cellquant.tiles.converter import generate_tiles_for_image
        src = _resolve_image_path(session, condition, base_name, channel)
        generate_tiles_for_image(src, tile_dir)

    if tile_path.exists():
        return FileResponse(tile_path, media_type="image/png")
    raise HTTPException(404, "Tile not found")


@router.get("/{session_id}/{condition}/{base_name}/{channel}/metadata")
async def get_image_metadata(
    session_id: str, condition: str, base_name: str, channel: str
):
    """Get image dimensions and metadata."""
    session = get_session(session_id)
    src = _resolve_image_path(session, condition, base_name, channel)

    from cellquant.core.io.image_loader import load_image
    img = load_image(src)

    return {
        "width": img.shape[1],
        "height": img.shape[0],
        "dtype": str(img.dtype),
        "tile_size": 256,
    }


# Keep legacy thumbnail routes for backward compatibility
@router.get("/{session_id}/{condition}/{base_name}/thumbnail")
async def get_thumbnail(session_id: str, condition: str, base_name: str):
    """Legacy: redirect to render endpoint using first channel."""
    session = get_session(session_id)
    cond_data = session.conditions.get(condition, {})
    channels_data = cond_data.get("image_sets", {}).get(base_name, {})
    if not channels_data:
        raise HTTPException(404, "Image set not found")
    first_channel = next(iter(channels_data.keys()))
    return await render_preview(session_id, condition, base_name, first_channel, size=512)
