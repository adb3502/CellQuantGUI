"""Segmentation endpoints."""

from fastapi import APIRouter

from cellquant.api.dependencies import get_session, get_task_queue
from cellquant.api.schemas.segmentation import (
    SegmentationRequest,
    SegmentationStatusResponse,
    TaskResponse,
)
from cellquant.tasks.worker import run_segmentation_task

router = APIRouter(prefix="/segmentation", tags=["segmentation"])


@router.post("/run", response_model=TaskResponse)
async def start_segmentation(req: SegmentationRequest):
    """Start batch segmentation as a background task."""
    session = get_session(req.session_id)
    queue = get_task_queue()

    seg_params = {
        "model_type": req.model_type,
        "diameter": req.diameter,
        "flow_threshold": req.flow_threshold,
        "cellprob_threshold": req.cellprob_threshold,
        "min_size": req.min_size,
        "channels": req.channels,
        "use_gpu": req.use_gpu,
        "batch_size": req.batch_size,
    }

    task_id = queue.submit(
        session_id=req.session_id,
        task_type="segmentation",
        fn=run_segmentation_task,
        queue=queue,
        session=session,
        seg_params=seg_params,
    )

    return TaskResponse(task_id=task_id)


@router.get("/status/{task_id}", response_model=SegmentationStatusResponse)
async def get_status(task_id: str):
    """Poll segmentation task status."""
    queue = get_task_queue()
    task = queue.get_task(task_id)
    if not task:
        from fastapi import HTTPException
        raise HTTPException(404, "Task not found")

    return SegmentationStatusResponse(
        task_id=task.id,
        status=task.status,
        progress=task.progress,
        message=task.message,
        elapsed_seconds=task.elapsed,
        result=task.result,
    )


@router.post("/cancel/{task_id}")
async def cancel_segmentation(task_id: str):
    queue = get_task_queue()
    cancelled = queue.cancel_task(task_id)
    return {"cancelled": cancelled}
