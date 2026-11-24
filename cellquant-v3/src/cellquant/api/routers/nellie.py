"""Nellie organelle analysis endpoints."""

import math
import pandas as pd
from fastapi import APIRouter, HTTPException
from pathlib import Path

from cellquant.api.dependencies import get_session, get_task_queue
from cellquant.api.schemas.segmentation import TaskResponse
from cellquant.api.schemas.nellie import (
    NellieRequest,
    NellieResultsResponse,
    NellieSummaryResponse,
)
from cellquant.tasks.worker import run_nellie_task

router = APIRouter(prefix="/nellie", tags=["nellie"])


def _get_nellie_df(session) -> pd.DataFrame | None:
    """Return the nellie results DataFrame from session or disk."""
    # Prefer standalone nellie_df if it exists
    nellie_df = getattr(session, "nellie_df", None)
    if nellie_df is not None and len(nellie_df) > 0:
        return nellie_df

    # Try loading from disk
    parquet_path = session.directory / "nellie_output" / "nellie_results.parquet"
    csv_path = session.directory / "nellie_output" / "nellie_results.csv"
    if parquet_path.exists():
        df = pd.read_parquet(str(parquet_path))
        session.nellie_df = df
        return df
    if csv_path.exists():
        df = pd.read_csv(str(csv_path))
        session.nellie_df = df
        return df

    # Fall back to main results_df if it contains nellie columns
    if session.results_df is not None:
        nellie_cols = [c for c in session.results_df.columns if c.startswith("nellie_")]
        if nellie_cols:
            return session.results_df

    return None


@router.post("/run", response_model=TaskResponse)
async def start_nellie(req: NellieRequest):
    """Start nellie organelle analysis as a background task."""
    session = get_session(req.session_id)
    queue = get_task_queue()

    nellie_params = {
        "nellie_params": req.nellie_params.model_dump(),
    }

    task_id = queue.submit(
        session_id=req.session_id,
        task_type="nellie",
        fn=run_nellie_task,
        queue=queue,
        session=session,
        nellie_params=nellie_params,
    )

    return TaskResponse(task_id=task_id)


@router.post("/cancel/{task_id}")
async def cancel_nellie(task_id: str):
    """Cancel a running nellie task."""
    queue = get_task_queue()
    cancelled = queue.cancel_task(task_id)
    return {"cancelled": cancelled}


@router.get("/results/{session_id}/page/{page}", response_model=NellieResultsResponse)
async def get_nellie_results_page(session_id: str, page: int = 0, per_page: int = 1000):
    """Return paginated nellie results."""
    session = get_session(session_id)
    df = _get_nellie_df(session)

    if df is None or len(df) == 0:
        return NellieResultsResponse(
            page=0, per_page=per_page, total_rows=0, total_pages=0, columns=[], data=[]
        )

    total_rows = len(df)
    total_pages = math.ceil(total_rows / per_page)
    start = page * per_page
    end = min(start + per_page, total_rows)
    page_df = df.iloc[start:end]

    return NellieResultsResponse(
        page=page,
        per_page=per_page,
        total_rows=total_rows,
        total_pages=total_pages,
        columns=list(df.columns),
        data=page_df.to_dict(orient="records"),
    )


@router.get("/summary/{session_id}", response_model=NellieSummaryResponse)
async def get_nellie_summary(session_id: str):
    """Return per-condition summary of nellie organelle features."""
    session = get_session(session_id)
    df = _get_nellie_df(session)

    if df is None or len(df) == 0:
        return NellieSummaryResponse(
            total_cells=0, n_conditions=0, n_image_sets=0, per_condition=[]
        )

    per_condition = []
    nellie_cols = [c for c in df.columns if c.startswith("nellie_") and c != "nellie_cell_level"]

    for cond in df["Condition"].unique():
        cond_df = df[df["Condition"] == cond]
        stats: dict = {"condition": cond, "n_cells": len(cond_df)}
        if "nellie_n_organelles" in cond_df.columns:
            stats["mean_n_organelles"] = float(cond_df["nellie_n_organelles"].mean())
            stats["total_organelles"] = float(cond_df["nellie_n_organelles"].sum())
        for col in nellie_cols:
            if col == "nellie_n_organelles":
                continue
            try:
                stats[f"{col}_mean"] = float(cond_df[col].mean())
            except Exception:
                pass
        per_condition.append(stats)

    return NellieSummaryResponse(
        total_cells=len(df),
        n_conditions=df["Condition"].nunique(),
        n_image_sets=df["ImageSet"].nunique() if "ImageSet" in df.columns else 0,
        per_condition=per_condition,
    )


@router.get("/columns/{session_id}")
async def get_nellie_columns(session_id: str):
    """Return list of available nellie feature column names."""
    session = get_session(session_id)
    df = _get_nellie_df(session)
    if df is None:
        return {"columns": []}
    nellie_cols = [c for c in df.columns if c.startswith("nellie_")]
    return {"columns": nellie_cols}
