"""SA-beta-gal analytics endpoints."""
from fastapi import APIRouter, HTTPException

from cellquant.api.dependencies import get_session, get_task_queue
from cellquant.api.schemas.sabetagal import SABetaGalRunRequest, SABetaGalRunResponse
from cellquant.tasks.worker import run_sabetagal_task

router = APIRouter(prefix="/sabetagal", tags=["sabetagal"])


@router.post("/run", response_model=SABetaGalRunResponse)
async def start_sabetagal(req: SABetaGalRunRequest):
    session = get_session(req.session_id)
    queue = get_task_queue()

    task_id = queue.submit(
        session_id=req.session_id,
        task_type="sabetagal",
        fn=run_sabetagal_task,
        queue=queue,
        session=session,
        sabetagal_params=req.model_dump(exclude={"session_id"}),
    )
    return SABetaGalRunResponse(task_id=task_id)


@router.get("/summary/{session_id}")
async def get_sabetagal_summary(session_id: str):
    from cellquant.core.quantification.sabetagal import summarize_positivity

    session = get_session(session_id)
    df = session.sabetagal_df
    if df is None or df.empty:
        raise HTTPException(404, "No SA-beta-gal results for this session")

    return {
        "summary": summarize_positivity(df).to_dict(orient="records"),
        "n_cells": len(df),
        "provisional": True,
    }


@router.get("/results/{session_id}")
async def get_sabetagal_results(session_id: str):
    session = get_session(session_id)
    df = session.sabetagal_df
    if df is None or df.empty:
        raise HTTPException(404, "No SA-beta-gal results for this session")
    return {"rows": df.to_dict(orient="records")}
