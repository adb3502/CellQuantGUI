"""Mask viewing and editing endpoints."""

from pathlib import Path
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from cellquant.api.dependencies import get_session
from cellquant.api.schemas.masks import (
    DeleteCellRequest,
    MergeCellsRequest,
    MaskStatsResponse,
)

router = APIRouter(prefix="/masks", tags=["masks"])


@router.get("/{session_id}/{condition}/{base_name}/stats", response_model=MaskStatsResponse)
async def get_mask_stats(session_id: str, condition: str, base_name: str):
    """Get mask statistics (cell count, areas)."""
    session = get_session(session_id)
    masks = _get_masks(session, condition, base_name)

    import numpy as np
    from scipy.ndimage import sum as ndi_sum

    labels = np.unique(masks)
    labels = labels[labels != 0]
    n_cells = len(labels)

    if n_cells == 0:
        return MaskStatsResponse(n_cells=0, min_area=0, max_area=0, mean_area=0)

    ones = np.ones_like(masks)
    areas = ndi_sum(ones, masks, labels)

    return MaskStatsResponse(
        n_cells=n_cells,
        min_area=float(np.min(areas)),
        max_area=float(np.max(areas)),
        mean_area=float(np.mean(areas)),
    )


@router.get("/{session_id}/{condition}/{base_name}/render")
async def render_mask_overlay(
    session_id: str,
    condition: str,
    base_name: str,
    size: int = Query(default=800, ge=100, le=4096),
    style: str = Query(default="filled", pattern="^(filled|outline)$"),
    bg: str = Query(default="composite", pattern="^(composite|cyto)$"),
):
    """Render mask overlay on a background image.

    style=filled: colored fill overlay (default)
    style=outline: red cell boundary outlines
    bg=composite: all channels summed (default)
    bg=cyto: cytoplasm channel only
    """
    import numpy as np
    from PIL import Image
    from cellquant.tiles.thumbnail import imagej_auto_contrast
    from cellquant.core.segmentation.cellpose_engine import (
        _create_simple_overlay, _create_outline_overlay,
    )
    from cellquant.core.io.image_loader import load_image

    session = get_session(session_id)
    masks = _get_masks(session, condition, base_name)

    cond_data = session.conditions.get(condition, {})
    image_sets = cond_data.get("image_sets", {})
    channels = image_sets.get(base_name, {})
    if not channels:
        raise HTTPException(404, f"No channels found: {condition}/{base_name}")

    # Check cache (keyed by style + bg + size)
    cache_dir = session.directory / "renders" / condition / base_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"mask_{style}_{bg}_{size}.jpg"

    if not cache_path.exists():
        if bg == "cyto":
            # Single cyto channel
            cyto_suffix = cond_data.get("cyto_suffix", "C1")
            cyto_path = channels.get(cyto_suffix) or channels.get(cyto_suffix.upper())
            if not cyto_path or not Path(cyto_path).exists():
                raise HTTPException(404, f"Cyto channel not found: {condition}/{base_name}")
            bg_img = load_image(cyto_path)
        else:
            # Composite: sum all channels
            loaded = []
            for suffix, ch_path in channels.items():
                if ch_path and Path(ch_path).exists():
                    loaded.append(load_image(ch_path).astype(np.float64))
            if not loaded:
                raise HTTPException(404, f"No channel images found: {condition}/{base_name}")
            bg_img = loaded[0] if len(loaded) == 1 else np.sum(np.stack(loaded, axis=0), axis=0)

        bg_uint8 = imagej_auto_contrast(bg_img)

        if style == "outline":
            overlay = _create_outline_overlay(bg_uint8, masks)
        else:
            overlay = _create_simple_overlay(bg_uint8, masks, alpha=0.5)
        pil_img = Image.fromarray(overlay, mode="RGB")
        if max(pil_img.size) > size:
            pil_img.thumbnail((size, size), Image.LANCZOS)
        pil_img.save(cache_path, "JPEG", quality=90)

    return FileResponse(
        cache_path,
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=60"},
    )


@router.get("/{session_id}/{condition}/{base_name}/tile/{level}/{col}_{row}.png")
async def get_mask_tile(
    session_id: str, condition: str, base_name: str, level: int, col: int, row: int
):
    """Serve a mask overlay tile (RGBA PNG)."""
    session = get_session(session_id)
    tile_dir = session.get_tile_dir(condition, base_name, "_mask")
    tile_path = tile_dir / str(level) / f"{col}_{row}.png"

    if not tile_path.exists():
        masks = _get_masks(session, condition, base_name)
        from cellquant.tiles.converter import generate_mask_tiles
        generate_mask_tiles(masks, tile_dir)

    if tile_path.exists():
        return FileResponse(tile_path, media_type="image/png")
    raise HTTPException(404, "Mask tile not found")


@router.put("/{session_id}/{condition}/{base_name}/delete-cell")
async def delete_cell(session_id: str, condition: str, base_name: str, req: DeleteCellRequest):
    """Delete a cell from the mask."""
    session = get_session(session_id)
    masks = _get_masks(session, condition, base_name)

    from cellquant.core.segmentation.mask_utils import remove_cell
    import numpy as np

    new_masks = remove_cell(masks, req.cell_id)
    session.masks[condition][base_name] = new_masks
    np.save(session.get_mask_path(condition, base_name), new_masks)

    # Invalidate mask tile cache
    _invalidate_mask_tiles(session, condition, base_name)

    n_cells = len(np.unique(new_masks)) - 1
    return {"success": True, "n_cells": n_cells}


@router.put("/{session_id}/{condition}/{base_name}/merge-cells")
async def merge_cells(session_id: str, condition: str, base_name: str, req: MergeCellsRequest):
    """Merge multiple cells into one."""
    session = get_session(session_id)
    masks = _get_masks(session, condition, base_name)

    from cellquant.core.segmentation.mask_utils import merge_cells as merge_fn
    import numpy as np

    new_masks = merge_fn(masks, req.cell_ids)
    session.masks[condition][base_name] = new_masks
    np.save(session.get_mask_path(condition, base_name), new_masks)

    _invalidate_mask_tiles(session, condition, base_name)

    n_cells = len(np.unique(new_masks)) - 1
    return {"success": True, "n_cells": n_cells}


@router.get("/{session_id}/{condition}/{base_name}/cell-at/{row}/{col}")
async def get_cell_at(session_id: str, condition: str, base_name: str, row: int, col: int):
    """Get cell ID at a given pixel coordinate."""
    session = get_session(session_id)
    masks = _get_masks(session, condition, base_name)

    if 0 <= row < masks.shape[0] and 0 <= col < masks.shape[1]:
        cell_id = int(masks[row, col])
        return {"cell_id": cell_id}
    return {"cell_id": 0}


def _get_masks(session, condition: str, base_name: str):
    """Load masks from session memory or disk."""
    import numpy as np

    if condition in session.masks and base_name in session.masks[condition]:
        return session.masks[condition][base_name]

    mask_path = session.get_mask_path(condition, base_name)
    if mask_path.exists():
        masks = np.load(mask_path)
        if condition not in session.masks:
            session.masks[condition] = {}
        session.masks[condition][base_name] = masks
        return masks

    raise HTTPException(404, f"No masks for {condition}/{base_name}")


def _invalidate_mask_tiles(session, condition: str, base_name: str):
    """Delete cached mask tiles after an edit."""
    import shutil
    tile_dir = session.get_tile_dir(condition, base_name, "_mask")
    if tile_dir.exists():
        shutil.rmtree(tile_dir, ignore_errors=True)
