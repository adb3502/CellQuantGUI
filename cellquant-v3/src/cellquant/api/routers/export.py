"""Export endpoints for CSV, Excel, ROIs."""

import time
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from cellquant.api.dependencies import get_session

router = APIRouter(prefix="/export", tags=["export"])


@router.post("/csv/{session_id}")
async def export_csv(session_id: str):
    """Export results as CSV."""
    session = get_session(session_id)

    if session.results_df is None:
        session.load_results()
    if session.results_df is None or len(session.results_df) == 0:
        raise HTTPException(404, "No results to export")

    export_dir = session.get_export_dir()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = export_dir / f"ctcf_results_{timestamp}.csv"
    session.results_df.to_csv(csv_path, index=False)

    return FileResponse(
        csv_path,
        media_type="text/csv",
        filename=csv_path.name,
    )


@router.post("/excel/{session_id}")
async def export_excel(session_id: str):
    """Export results as Excel."""
    session = get_session(session_id)

    if session.results_df is None:
        session.load_results()
    if session.results_df is None or len(session.results_df) == 0:
        raise HTTPException(404, "No results to export")

    export_dir = session.get_export_dir()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    xlsx_path = export_dir / f"ctcf_results_{timestamp}.xlsx"
    session.results_df.to_excel(xlsx_path, index=False)

    return FileResponse(
        xlsx_path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=xlsx_path.name,
    )


@router.post("/rois/{session_id}/{condition}/{base_name}")
async def export_rois(session_id: str, condition: str, base_name: str):
    """Export ROIs as ImageJ-compatible zip."""
    session = get_session(session_id)

    # Get masks
    import numpy as np
    masks = None
    if condition in session.masks and base_name in session.masks[condition]:
        masks = session.masks[condition][base_name]
    else:
        mask_path = session.get_mask_path(condition, base_name)
        if mask_path.exists():
            masks = np.load(mask_path)

    if masks is None:
        raise HTTPException(404, "No masks found")

    from cellquant.core.io.roi_export import save_rois_imagej

    export_dir = session.get_export_dir()
    roi_path = export_dir / f"{base_name}_rois.zip"
    save_rois_imagej(masks, roi_path)

    return FileResponse(
        roi_path,
        media_type="application/zip",
        filename=roi_path.name,
    )
