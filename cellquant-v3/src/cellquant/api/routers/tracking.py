"""Tracking endpoints for Trackastra cell tracking."""

from fastapi import APIRouter

from cellquant.api.dependencies import get_session, get_task_queue
from cellquant.api.schemas.tracking import TrackingRequest, TrackingResponse
from cellquant.tasks.worker import run_tracking_task

router = APIRouter(prefix="/tracking", tags=["tracking"])


@router.post("/run", response_model=TrackingResponse)
async def start_tracking(req: TrackingRequest):
    """Start Trackastra tracking as a background task."""
    session = get_session(req.session_id)
    queue = get_task_queue()

    tracking_params = {
        "model": req.model,
        "mode": req.mode,
        "condition": req.condition,
        "device": req.device,
    }

    task_id = queue.submit(
        session_id=req.session_id,
        task_type="tracking",
        fn=run_tracking_task,
        queue=queue,
        session=session,
        tracking_params=tracking_params,
    )

    return TrackingResponse(task_id=task_id)


@router.get("/tracks/{session_id}/{condition}")
async def get_tracks(session_id: str, condition: str):
    """Get tracking results for a condition."""
    session = get_session(session_id)
    tracked = session.tracked_masks.get(condition, {})

    if not tracked:
        from fastapi import HTTPException
        raise HTTPException(404, "No tracking results for this condition")

    # Return track summary
    import numpy as np
    all_ids = set()
    for masks in tracked.values():
        all_ids.update(np.unique(masks[masks > 0]).tolist())

    return {
        "condition": condition,
        "n_frames": len(tracked),
        "n_tracks": len(all_ids),
        "frame_names": sorted(tracked.keys()),
    }
