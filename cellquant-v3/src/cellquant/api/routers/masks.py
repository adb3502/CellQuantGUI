"""Mask viewing and editing endpoints."""

import io
from pathlib import Path
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, StreamingResponse

from cellquant.api.dependencies import get_session
from cellquant.api.schemas.masks import (
    DeleteCellRequest,
    MergeCellsRequest,
    AddCellPolygonRequest,
    AddCellFloodRequest,
    DilateCellsRequest,
    ErodeCellsRequest,
    SmoothRequest,
    FillHolesRequest,
    CleanSmallRequest,
    MaskStatsResponse,
    MaskEditResponse,
)

router = APIRouter(prefix="/masks", tags=["masks"])


# ── Read endpoints ────────────────────────────────────────


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


@router.get("/{session_id}/{condition}/{base_name}/tiff")
async def get_mask_tiff(session_id: str, condition: str, base_name: str):
    """Serve the mask as a 16-bit TIFF for ImageJ.JS."""
    import numpy as np
    from PIL import Image

    session = get_session(session_id)
    masks = _get_masks(session, condition, base_name)

    # Convert to 16-bit for ImageJ compatibility
    mask_16 = masks.astype(np.uint16)
    pil_img = Image.fromarray(mask_16)

    buf = io.BytesIO()
    pil_img.save(buf, format="TIFF")
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="image/tiff",
        headers={"Cache-Control": "no-cache"},
    )


@router.get("/{session_id}/{condition}/{base_name}/png")
async def get_mask_png(session_id: str, condition: str, base_name: str):
    """Serve the mask as a 16-bit PNG for ImageJ.JS."""
    import numpy as np
    from PIL import Image

    session = get_session(session_id)
    masks = _get_masks(session, condition, base_name)

    mask_16 = masks.astype(np.uint16)
    pil_img = Image.fromarray(mask_16)

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="image/png",
        headers={"Cache-Control": "no-cache"},
    )


@router.put("/{session_id}/{condition}/{base_name}/tiff")
async def put_mask_tiff(session_id: str, condition: str, base_name: str, request: Request):
    """Upload an edited mask TIFF from ImageJ.JS."""
    import numpy as np
    from PIL import Image

    session = get_session(session_id)
    body = await request.body()
    buf = io.BytesIO(body)
    pil_img = Image.open(buf)
    new_masks = np.array(pil_img).astype(np.int32)

    _save_masks(session, condition, base_name, new_masks, "imagej_edit")

    n_cells = len(np.unique(new_masks)) - 1
    return {"success": True, "n_cells": n_cells}


@router.get("/{session_id}/{condition}/{base_name}/render")
async def render_mask_overlay(
    session_id: str,
    condition: str,
    base_name: str,
    size: int = Query(default=800, ge=100, le=4096),
    style: str = Query(default="filled", pattern="^(filled|outline)$"),
    bg: str = Query(default=""),
):
    """Render mask overlay on a background image.

    bg: channel suffix to use as background (e.g. 'C0', 'C1', 'C2').
         If empty, uses the first available channel.
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

    # Sanitize bg for cache key
    bg_key = bg if bg else "default"
    cache_dir = session.directory / "renders" / condition / base_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Outlines are saved as PNG (lossless) — JPEG destroys 1-px red lines via DCT compression
    cache_ext = "png" if style == "outline" else "jpg"
    cache_path = cache_dir / f"mask_{style}_{bg_key}_{size}.{cache_ext}"

    if not cache_path.exists():
        if bg:
            # Load a specific channel by suffix
            ch_path = channels.get(bg) or channels.get(bg.upper()) or channels.get(bg.lower())
            if not ch_path or not Path(ch_path).exists():
                raise HTTPException(404, f"Channel '{bg}' not found: {condition}/{base_name}")
            bg_img = load_image(ch_path)
        else:
            # Fallback: first available channel
            for suffix, ch_path in channels.items():
                if ch_path and Path(ch_path).exists():
                    bg_img = load_image(ch_path)
                    break
            else:
                raise HTTPException(404, f"No channel images found: {condition}/{base_name}")

        bg_uint8 = imagej_auto_contrast(bg_img)

        if style == "outline":
            overlay = _create_outline_overlay(bg_uint8, masks)
        else:
            overlay = _create_simple_overlay(bg_uint8, masks, alpha=0.5)
        pil_img = Image.fromarray(overlay, mode="RGB")
        if max(pil_img.size) > size:
            pil_img.thumbnail((size, size), Image.LANCZOS)
        if style == "outline":
            pil_img.save(cache_path, "PNG")
        else:
            pil_img.save(cache_path, "JPEG", quality=90)

    media_type = "image/png" if style == "outline" else "image/jpeg"
    return FileResponse(
        cache_path,
        media_type=media_type,
        headers={"Cache-Control": "public, max-age=60"},
    )


@router.get("/{session_id}/{condition}/{base_name}/tile/{level}/{col}_{row}.png")
async def get_mask_tile(
    session_id: str, condition: str, base_name: str, level: int, col: int, row: int
):
    """Serve a mask overlay tile (RGBA PNG)."""
    # OpenLayers may send negative y coords; convert to positive row index
    if row < 0:
        row = -(row + 1)
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


@router.get("/{session_id}/{condition}/{base_name}/cell-at/{row}/{col}")
async def get_cell_at(session_id: str, condition: str, base_name: str, row: int, col: int):
    """Get cell ID at a given pixel coordinate."""
    session = get_session(session_id)
    masks = _get_masks(session, condition, base_name)

    if 0 <= row < masks.shape[0] and 0 <= col < masks.shape[1]:
        cell_id = int(masks[row, col])
        return {"cell_id": cell_id}
    return {"cell_id": 0}


# ── Edit endpoints ────────────────────────────────────────


@router.put("/{session_id}/{condition}/{base_name}/delete-cell", response_model=MaskEditResponse)
async def delete_cell(session_id: str, condition: str, base_name: str, req: DeleteCellRequest):
    """Delete a cell from the mask."""
    session = get_session(session_id)
    masks = _get_masks(session, condition, base_name)

    from cellquant.core.segmentation.mask_utils import remove_cell
    import numpy as np

    new_masks = remove_cell(masks, req.cell_id)
    _save_masks(session, condition, base_name, new_masks, "delete_cell")

    n_cells = len(np.unique(new_masks)) - 1
    return MaskEditResponse(success=True, n_cells=n_cells)


@router.put("/{session_id}/{condition}/{base_name}/merge-cells", response_model=MaskEditResponse)
async def merge_cells(session_id: str, condition: str, base_name: str, req: MergeCellsRequest):
    """Merge multiple cells into one."""
    session = get_session(session_id)
    masks = _get_masks(session, condition, base_name)

    from cellquant.core.segmentation.mask_utils import merge_cells as merge_fn
    import numpy as np

    new_masks = merge_fn(masks, req.cell_ids)
    _save_masks(session, condition, base_name, new_masks, "merge_cells")

    n_cells = len(np.unique(new_masks)) - 1
    return MaskEditResponse(success=True, n_cells=n_cells)


@router.put("/{session_id}/{condition}/{base_name}/add-cell", response_model=MaskEditResponse)
async def add_cell(session_id: str, condition: str, base_name: str, req: AddCellPolygonRequest):
    """Add a new cell from polygon coordinates."""
    session = get_session(session_id)
    masks = _get_masks(session, condition, base_name)

    from cellquant.core.segmentation.mask_utils import add_cell as add_cell_fn
    import numpy as np

    coords = np.array(req.polygon_coords)
    new_masks = add_cell_fn(masks, coords, overwrite_existing=req.overwrite)
    _save_masks(session, condition, base_name, new_masks, "add_cell")

    n_cells = len(np.unique(new_masks)) - 1
    return MaskEditResponse(success=True, n_cells=n_cells)


@router.put("/{session_id}/{condition}/{base_name}/add-cell-flood", response_model=MaskEditResponse)
async def add_cell_flood(session_id: str, condition: str, base_name: str, req: AddCellFloodRequest):
    """Add a new cell by flood-filling from a click point."""
    session = get_session(session_id)
    masks = _get_masks(session, condition, base_name)

    from cellquant.core.segmentation.mask_utils import add_cell_from_click
    import numpy as np

    image = _load_intensity_image(session, condition, base_name)
    new_masks = add_cell_from_click(masks, image, (req.row, req.col), req.threshold)
    _save_masks(session, condition, base_name, new_masks, "add_cell_flood")

    n_cells = len(np.unique(new_masks)) - 1
    return MaskEditResponse(success=True, n_cells=n_cells)


@router.put("/{session_id}/{condition}/{base_name}/dilate", response_model=MaskEditResponse)
async def dilate_cells(session_id: str, condition: str, base_name: str, req: DilateCellsRequest):
    """Dilate selected cells."""
    session = get_session(session_id)
    masks = _get_masks(session, condition, base_name)

    from cellquant.core.segmentation.mask_utils import dilate_masks
    import numpy as np

    if req.cell_ids:
        # Dilate only selected cells: extract subset, dilate, merge back
        subset = np.where(np.isin(masks, req.cell_ids), masks, 0)
        dilated_subset = dilate_masks(subset, iterations=req.iterations)
        new_masks = masks.copy()
        expanded = (dilated_subset > 0) & (masks == 0)
        new_masks[expanded] = dilated_subset[expanded]
    else:
        new_masks = dilate_masks(masks, iterations=req.iterations)

    _save_masks(session, condition, base_name, new_masks, "dilate")

    n_cells = len(np.unique(new_masks)) - 1
    return MaskEditResponse(success=True, n_cells=n_cells)


@router.put("/{session_id}/{condition}/{base_name}/erode", response_model=MaskEditResponse)
async def erode_cells(session_id: str, condition: str, base_name: str, req: ErodeCellsRequest):
    """Erode selected cells."""
    session = get_session(session_id)
    masks = _get_masks(session, condition, base_name)

    from cellquant.core.segmentation.mask_utils import erode_masks
    import numpy as np

    if req.cell_ids:
        subset = np.where(np.isin(masks, req.cell_ids), masks, 0)
        eroded_subset = erode_masks(subset, iterations=req.iterations)
        new_masks = masks.copy()
        for cid in req.cell_ids:
            was_cell = masks == cid
            still_cell = eroded_subset == cid
            new_masks[was_cell & ~still_cell] = 0
    else:
        new_masks = erode_masks(masks, iterations=req.iterations)

    _save_masks(session, condition, base_name, new_masks, "erode")

    n_cells = len(np.unique(new_masks)) - 1
    return MaskEditResponse(success=True, n_cells=n_cells)


@router.put("/{session_id}/{condition}/{base_name}/smooth", response_model=MaskEditResponse)
async def smooth_masks(session_id: str, condition: str, base_name: str, req: SmoothRequest):
    """Smooth cell boundaries."""
    session = get_session(session_id)
    masks = _get_masks(session, condition, base_name)

    from cellquant.core.segmentation.mask_utils import smooth_boundaries
    import numpy as np

    if req.cell_ids:
        subset = np.where(np.isin(masks, req.cell_ids), masks, 0)
        smoothed = smooth_boundaries(subset, sigma=req.sigma)
        new_masks = masks.copy()
        for cid in req.cell_ids:
            new_masks[masks == cid] = 0
        new_masks[smoothed > 0] = smoothed[smoothed > 0]
    else:
        new_masks = smooth_boundaries(masks, sigma=req.sigma)

    _save_masks(session, condition, base_name, new_masks, "smooth")

    n_cells = len(np.unique(new_masks)) - 1
    return MaskEditResponse(success=True, n_cells=n_cells)


@router.put("/{session_id}/{condition}/{base_name}/fill-holes", response_model=MaskEditResponse)
async def fill_holes_endpoint(session_id: str, condition: str, base_name: str, req: FillHolesRequest):
    """Fill holes within cells."""
    session = get_session(session_id)
    masks = _get_masks(session, condition, base_name)

    from cellquant.core.segmentation.mask_utils import fill_holes
    import numpy as np

    new_masks = fill_holes(masks, max_hole_size=req.max_hole_size)
    _save_masks(session, condition, base_name, new_masks, "fill_holes")

    n_cells = len(np.unique(new_masks)) - 1
    return MaskEditResponse(success=True, n_cells=n_cells)


@router.put("/{session_id}/{condition}/{base_name}/clean-small", response_model=MaskEditResponse)
async def clean_small(session_id: str, condition: str, base_name: str, req: CleanSmallRequest):
    """Remove cells smaller than min_size."""
    session = get_session(session_id)
    masks = _get_masks(session, condition, base_name)

    from cellquant.core.segmentation.mask_utils import clean_small_objects
    import numpy as np

    new_masks = clean_small_objects(masks, min_size=req.min_size)
    _save_masks(session, condition, base_name, new_masks, "clean_small")

    n_cells = len(np.unique(new_masks)) - 1
    return MaskEditResponse(success=True, n_cells=n_cells)


# ── Undo endpoints ────────────────────────────────────────


@router.post("/{session_id}/{condition}/{base_name}/undo", response_model=MaskEditResponse)
async def undo_edit(session_id: str, condition: str, base_name: str):
    """Undo the last mask edit for this image."""
    session = get_session(session_id)
    import numpy as np

    previous = session.edit_history.pop(condition, base_name)
    if previous is None:
        raise HTTPException(400, "Nothing to undo")

    session.masks[condition][base_name] = previous
    np.save(session.get_mask_path(condition, base_name), previous)
    _invalidate_mask_tiles(session, condition, base_name)
    _invalidate_render_cache(session, condition, base_name)

    n_cells = len(np.unique(previous)) - 1
    return MaskEditResponse(success=True, n_cells=n_cells)


@router.get("/{session_id}/edit-status")
async def get_edit_status(session_id: str):
    """Get which images have been edited in this session."""
    session = get_session(session_id)
    return {
        "edited_images": session.edit_history.edited_images(),
    }


@router.get("/{session_id}/{condition}/{base_name}/undo-depth")
async def get_undo_depth(session_id: str, condition: str, base_name: str):
    """Get the number of undoable operations for this image."""
    session = get_session(session_id)
    return {
        "depth": session.edit_history.undo_depth(condition, base_name),
        "can_undo": session.edit_history.can_undo(condition, base_name),
    }


# ── Helpers ───────────────────────────────────────────────


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


def _save_masks(session, condition: str, base_name: str, new_masks,
                operation: str = "edit"):
    """Push undo snapshot, save masks to session memory and disk, invalidate caches."""
    import numpy as np

    # Push undo snapshot BEFORE overwriting
    old_masks = session.masks.get(condition, {}).get(base_name)
    if old_masks is not None:
        session.edit_history.push(condition, base_name, operation, old_masks)

    session.masks[condition][base_name] = new_masks
    np.save(session.get_mask_path(condition, base_name), new_masks)
    _invalidate_mask_tiles(session, condition, base_name)
    _invalidate_render_cache(session, condition, base_name)


def _invalidate_mask_tiles(session, condition: str, base_name: str):
    """Delete cached mask tiles after an edit."""
    import shutil
    tile_dir = session.get_tile_dir(condition, base_name, "_mask")
    if tile_dir.exists():
        shutil.rmtree(tile_dir, ignore_errors=True)


def _invalidate_render_cache(session, condition: str, base_name: str):
    """Delete cached mask renders after an edit."""
    import shutil
    render_dir = session.directory / "renders" / condition / base_name
    if render_dir.exists():
        shutil.rmtree(render_dir, ignore_errors=True)


def _load_intensity_image(session, condition: str, base_name: str):
    """Load the intensity image for flood-fill operations."""
    import numpy as np
    from cellquant.core.io.image_loader import load_image

    cond_data = session.conditions.get(condition, {})
    image_sets = cond_data.get("image_sets", {})
    channels = image_sets.get(base_name, {})
    if not channels:
        raise HTTPException(404, f"No channels found: {condition}/{base_name}")

    # Use the first available channel as intensity reference
    for suffix, ch_path in channels.items():
        if ch_path and Path(ch_path).exists():
            return load_image(ch_path)

    raise HTTPException(404, f"No intensity image found: {condition}/{base_name}")
