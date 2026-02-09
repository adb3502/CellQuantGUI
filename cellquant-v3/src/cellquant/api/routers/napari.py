"""Napari integration endpoint (desktop mode only)."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from cellquant.api.dependencies import get_session

router = APIRouter(prefix="/napari", tags=["napari"])


class NapariLaunchRequest(BaseModel):
    session_id: str
    condition: str
    base_name: str


@router.post("/launch")
async def launch_napari(req: NapariLaunchRequest):
    """Launch Napari editor for a specific image set (desktop only)."""
    try:
        from cellquant.core.napari_bridge import NapariBridge
    except ImportError:
        raise HTTPException(501, "Napari not available. Install with: pip install napari[all]")

    session = get_session(req.session_id)

    # Load image and masks
    cond_data = session.conditions.get(req.condition, {})
    image_sets = cond_data.get("image_sets", {})
    channels = image_sets.get(req.base_name, {})

    if not channels:
        raise HTTPException(404, "Image set not found")

    import numpy as np
    from cellquant.core.io.image_loader import load_image, normalize_image

    # Load first channel as display image
    first_key = next(iter(channels.keys()))
    image = load_image(channels[first_key])

    # Load masks
    masks = None
    if req.condition in session.masks and req.base_name in session.masks[req.condition]:
        masks = session.masks[req.condition][req.base_name]

    if masks is None:
        mask_path = session.get_mask_path(req.condition, req.base_name)
        if mask_path.exists():
            masks = np.load(mask_path)

    if masks is None:
        raise HTTPException(404, "No masks to edit. Run segmentation first.")

    def on_close(edited_masks):
        session.masks.setdefault(req.condition, {})[req.base_name] = edited_masks
        np.save(session.get_mask_path(req.condition, req.base_name), edited_masks)

    from cellquant.core.napari_bridge import NapariEditorThread

    editor = NapariEditorThread(image=image, masks=masks, on_complete=on_close)
    editor.start()

    return {"status": "launched", "message": "Napari editor opened. Close it to save edits."}
