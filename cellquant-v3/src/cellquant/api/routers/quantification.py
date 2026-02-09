"""Quantification endpoints."""

import math
from fastapi import APIRouter, HTTPException

from cellquant.api.dependencies import get_session, get_task_queue
from cellquant.api.schemas.segmentation import TaskResponse
from cellquant.api.schemas.quantification import (
    QuantificationRequest,
    ResultsPageResponse,
    SummaryStatsResponse,
)
from cellquant.tasks.worker import run_quantification_task

router = APIRouter(prefix="/quantification", tags=["quantification"])


@router.post("/run", response_model=TaskResponse)
async def start_quantification(req: QuantificationRequest):
    """Start quantification as a background task."""
    session = get_session(req.session_id)
    queue = get_task_queue()

    quant_params = {
        "background_method": req.background_method,
        "marker_suffixes": req.marker_suffixes,
        "marker_names": req.marker_names,
        "mitochondrial_markers": req.mitochondrial_markers,
    }

    task_id = queue.submit(
        session_id=req.session_id,
        task_type="quantification",
        fn=run_quantification_task,
        queue=queue,
        session=session,
        quant_params=quant_params,
    )

    return TaskResponse(task_id=task_id)


@router.get("/results/{session_id}/page/{page}", response_model=ResultsPageResponse)
async def get_results_page(session_id: str, page: int = 0, per_page: int = 1000):
    """Get paginated quantification results."""
    session = get_session(session_id)

    if session.results_df is None:
        session.load_results()

    if session.results_df is None or len(session.results_df) == 0:
        return ResultsPageResponse(
            page=0, per_page=per_page, total_rows=0, total_pages=0, columns=[], data=[]
        )

    df = session.results_df
    total_rows = len(df)
    total_pages = math.ceil(total_rows / per_page)

    start = page * per_page
    end = min(start + per_page, total_rows)
    page_df = df.iloc[start:end]

    return ResultsPageResponse(
        page=page,
        per_page=per_page,
        total_rows=total_rows,
        total_pages=total_pages,
        columns=list(df.columns),
        data=page_df.to_dict(orient="records"),
    )


@router.get("/summary/{session_id}", response_model=SummaryStatsResponse)
async def get_summary(session_id: str):
    """Get summary statistics."""
    session = get_session(session_id)

    if session.results_df is None:
        session.load_results()

    if session.results_df is None or len(session.results_df) == 0:
        return SummaryStatsResponse(
            total_cells=0, n_conditions=0, n_image_sets=0, per_condition=[]
        )

    df = session.results_df
    per_condition = []

    for cond in df["Condition"].unique():
        cond_df = df[df["Condition"] == cond]
        ctcf_cols = [c for c in df.columns if c.endswith("_CTCF")]

        stats = {
            "condition": cond,
            "n_cells": len(cond_df),
            "mean_area": float(cond_df["Area"].mean()),
        }
        for col in ctcf_cols:
            marker = col.replace("_CTCF", "")
            stats[f"{marker}_mean"] = float(cond_df[col].mean())
            stats[f"{marker}_std"] = float(cond_df[col].std())
            stats[f"{marker}_median"] = float(cond_df[col].median())

        per_condition.append(stats)

    return SummaryStatsResponse(
        total_cells=len(df),
        n_conditions=df["Condition"].nunique(),
        n_image_sets=df["ImageSet"].nunique() if "ImageSet" in df.columns else 0,
        per_condition=per_condition,
    )
