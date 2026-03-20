"""Segmentation endpoints."""

from pathlib import Path

from fastapi import APIRouter, HTTPException

from cellquant.api.dependencies import get_session, get_task_queue
from cellquant.api.schemas.segmentation import (
    SegmentationRequest,
    SegmentationStatusResponse,
    TaskResponse,
    ConditionMaskStatus,
    MaskStatusResponse,
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
        "skip_existing": req.skip_existing,
        "segmentation_suffixes": req.segmentation_suffixes,
        "condition_overrides": {
            k: v.model_dump(exclude_none=True)
            for k, v in (req.condition_overrides or {}).items()
        },
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


@router.get("/masks/status/{session_id}", response_model=MaskStatusResponse)
async def get_mask_status(session_id: str):
    """Check which masks exist on disk for a session."""
    session = get_session(session_id)

    masks_root = session.directory / "masks"
    conditions_status = []
    total_masks = 0

    expected_total = sum(
        len(cond.get("image_sets", {}))
        for cond in session.conditions.values()
    )

    if masks_root.exists():
        for cond_name in session.conditions:
            cond_dir = masks_root / cond_name
            if cond_dir.exists():
                npy_files = list(cond_dir.glob("*_masks.npy"))
                base_names = [f.stem.replace("_masks", "") for f in npy_files]
                conditions_status.append(ConditionMaskStatus(
                    name=cond_name,
                    mask_count=len(npy_files),
                    base_names=sorted(base_names),
                ))
                total_masks += len(npy_files)
            else:
                conditions_status.append(ConditionMaskStatus(
                    name=cond_name,
                    mask_count=0,
                    base_names=[],
                ))

    has_nuclear = any(
        list(cond_dir.glob("*_nuclear_masks.npy"))
        for cond_dir in [masks_root / cn for cn in session.conditions]
        if cond_dir.exists()
    ) if masks_root.exists() else False

    # Check for existing results
    has_results = False
    results_n_cells = 0
    results_path = session.get_results_path()
    if results_path.exists():
        has_results = True
        if session.results_df is not None:
            results_n_cells = len(session.results_df)
        else:
            try:
                import pandas as pd
                df = pd.read_parquet(results_path)
                results_n_cells = len(df)
            except Exception:
                pass

    return MaskStatusResponse(
        conditions=conditions_status,
        total_masks=total_masks,
        expected_total=expected_total,
        is_complete=(total_masks >= expected_total and expected_total > 0),
        has_results=has_results,
        results_n_cells=results_n_cells,
        has_nuclear=has_nuclear,
    )


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
