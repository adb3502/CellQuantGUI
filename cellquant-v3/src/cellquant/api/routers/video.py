"""Video analytics endpoints."""
from fastapi import APIRouter, HTTPException

from cellquant.api.dependencies import get_session, get_task_queue
from cellquant.api.schemas.video import (
    DoseTableResponse,
    ParsedDoseOut,
    VideoRunRequest,
    VideoRunResponse,
)
from cellquant.tasks.worker import run_video_task

router = APIRouter(prefix="/video", tags=["video"])


@router.post("/run", response_model=VideoRunResponse)
async def start_video(req: VideoRunRequest):
    session = get_session(req.session_id)
    queue = get_task_queue()

    task_id = queue.submit(
        session_id=req.session_id,
        task_type="video",
        fn=run_video_task,
        queue=queue,
        session=session,
        video_params=req.model_dump(exclude={"session_id"}),
    )
    return VideoRunResponse(task_id=task_id)


@router.get("/summary/{session_id}")
async def get_video_summary(session_id: str):
    session = get_session(session_id)
    df = session.trace_summary_df
    if df is None or df.empty:
        raise HTTPException(404, "No video results for this session")
    return {"rows": df.to_dict(orient="records"), "n_cells": len(df)}


@router.get("/traces/{session_id}/{condition}")
async def get_traces(session_id: str, condition: str):
    session = get_session(session_id)
    df = session.traces_df
    if df is None or df.empty:
        raise HTTPException(404, "No traces for this session")
    subset = df[df["Condition"] == condition]
    if subset.empty:
        raise HTTPException(404, f"No traces for condition {condition}")
    return {"rows": subset.to_dict(orient="records")}


@router.get("/parse-doses/{session_id}", response_model=DoseTableResponse)
async def parse_doses(session_id: str):
    """Parsed dose table for user confirmation before any EC50 fit."""
    from cellquant.core.quantification.dose_response import parse_dose_table

    session = get_session(session_id)
    names = []
    for cond in session.conditions.values():
        names.extend(cond.get("image_sets", {}).keys())

    doses = parse_dose_table(names)
    out = [
        ParsedDoseOut(
            source=d.source, parsed=d.parsed, molar=d.molar, unit=d.unit,
            volume_ul=d.volume_ul, ambiguous=d.ambiguous, note=d.note,
            group_key=d.group_key,
        )
        for d in doses
    ]
    return DoseTableResponse(
        doses=out,
        needs_confirmation=any((not d.parsed) or d.ambiguous for d in doses),
    )
