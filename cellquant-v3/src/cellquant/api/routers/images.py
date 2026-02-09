"""Image tile serving and thumbnails."""

from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response

from cellquant.api.dependencies import get_session

router = APIRouter(prefix="/images", tags=["images"])


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
        # Generate tiles on demand
        from cellquant.tiles.converter import generate_tiles_for_image

        cond_data = session.conditions.get(condition, {})
        image_sets = cond_data.get("image_sets", {})
        channels = image_sets.get(base_name, {})
        image_path = channels.get(channel) or channels.get(channel.upper())

        if not image_path or not Path(image_path).exists():
            raise HTTPException(404, f"Image not found for {condition}/{base_name}/{channel}")

        generate_tiles_for_image(Path(image_path), tile_dir)

    if tile_path.exists():
        return FileResponse(tile_path, media_type="image/png")
    raise HTTPException(404, "Tile not found")


@router.get("/{session_id}/{condition}/{base_name}/{channel}/metadata")
async def get_image_metadata(
    session_id: str, condition: str, base_name: str, channel: str
):
    """Get image dimensions and metadata."""
    session = get_session(session_id)
    cond_data = session.conditions.get(condition, {})
    image_sets = cond_data.get("image_sets", {})
    channels = image_sets.get(base_name, {})
    image_path = channels.get(channel) or channels.get(channel.upper())

    if not image_path or not Path(image_path).exists():
        raise HTTPException(404, "Image not found")

    from cellquant.core.io.image_loader import load_image
    img = load_image(image_path)

    return {
        "width": img.shape[1],
        "height": img.shape[0],
        "dtype": str(img.dtype),
        "tile_size": 256,
    }


@router.get("/{session_id}/{condition}/{base_name}/thumbnail")
async def get_thumbnail(session_id: str, condition: str, base_name: str):
    """Get 256px JPEG thumbnail for an image set."""
    session = get_session(session_id)
    cond_data = session.conditions.get(condition, {})
    channels_data = cond_data.get("image_sets", {}).get(base_name, {})

    if not channels_data:
        raise HTTPException(404, "Image set not found")

    # Use first available channel for thumbnail
    first_channel = next(iter(channels_data.keys()))
    thumb_path = session.get_thumbnail_path(condition, base_name, first_channel)

    if not thumb_path.exists():
        from cellquant.tiles.thumbnail import generate_thumbnail

        image_path = channels_data[first_channel]
        generate_thumbnail(Path(image_path), thumb_path)

    if thumb_path.exists():
        return FileResponse(thumb_path, media_type="image/jpeg")
    raise HTTPException(404, "Thumbnail generation failed")
