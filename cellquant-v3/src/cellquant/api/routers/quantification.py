"""Quantification endpoints."""

import io
import math
import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from cellquant.api.dependencies import get_session, get_task_queue
from cellquant.api.schemas.segmentation import TaskResponse
from cellquant.api.schemas.quantification import (
    QuantificationRequest,
    ResultsPageResponse,
    SummaryStatsResponse,
    QCSummaryResponse,
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
        "qc_filters": req.qc_filters.model_dump(),
        "negative_control_path": req.negative_control_path,
        "manual_background_value": req.manual_background_value,
        "outlier_threshold": req.outlier_threshold,
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


@router.get("/chart-data/{session_id}")
async def get_chart_data(session_id: str):
    """Return lightweight data for charts — all rows but only chart-relevant columns."""
    session = get_session(session_id)

    if session.results_df is None:
        session.load_results()

    if session.results_df is None or len(session.results_df) == 0:
        return {"columns": [], "data": [], "total_rows": 0}

    df = session.results_df

    # Select only columns needed for charts
    keep = []
    for col in df.columns:
        if col in ("Condition", "ImageSet", "CellID", "Area",
                    "x_centroid", "y_centroid"):
            keep.append(col)
        elif col.endswith("_CTCF") or col.endswith("_Background"):
            keep.append(col)
        elif col.startswith("is_outlier_") or col in ("is_saturated", "is_dim"):
            keep.append(col)

    chart_df = df[keep] if keep else df

    return {
        "columns": list(chart_df.columns),
        "data": chart_df.to_dict(orient="records"),
        "total_rows": len(chart_df),
    }


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


@router.get("/qc-summary/{session_id}", response_model=QCSummaryResponse)
async def get_qc_summary(session_id: str):
    """Get hierarchical QC summary (cells -> FOVs -> conditions)."""
    from cellquant.core.quantification.outliers import (
        per_fov_aggregation,
        hierarchical_summary,
    )

    session = get_session(session_id)

    if session.results_df is None:
        session.load_results()

    if session.results_df is None or len(session.results_df) == 0:
        return QCSummaryResponse(summary=[], fov_data=[])

    df = session.results_df
    fov_df = per_fov_aggregation(df)
    summary_df = hierarchical_summary(df, fov_df)

    return QCSummaryResponse(
        summary=summary_df.to_dict(orient="records") if len(summary_df) > 0 else [],
        fov_data=fov_df.to_dict(orient="records") if len(fov_df) > 0 else [],
    )


@router.get("/heatmap-image/{session_id}")
async def get_heatmap_image(
    session_id: str,
    condition: str = Query(...),
    image_set: str = Query(...),
    channel: str = Query(...),
    metric: str = Query(default="mean"),  # mean | ctcf | background
):
    """
    Render a PNG heatmap where each cell mask region is filled with a
    color-mapped quantification value.

    metric=mean       → per-cell mean intensity (viridis)
    metric=ctcf       → CTCF value (viridis)
    metric=background → per-cell background used (RdYlBu)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    from PIL import Image

    session = get_session(session_id)

    # ── Load mask ──────────────────────────────────────────────────
    mask_path = session.get_mask_path(condition, image_set)
    if not mask_path.exists():
        # Try loading from in-memory session masks
        masks_for_cond = session.masks.get(condition, {})
        masks = masks_for_cond.get(image_set)
        if masks is None:
            raise HTTPException(404, f"Mask not found: {condition}/{image_set}")
    else:
        masks = np.load(str(mask_path))

    # ── Load results ───────────────────────────────────────────────
    if session.results_df is None:
        session.load_results()
    if session.results_df is None:
        raise HTTPException(404, "No quantification results — run quantification first")

    df = session.results_df
    subset = df[(df["Condition"] == condition) & (df["ImageSet"] == image_set)]
    if len(subset) == 0:
        raise HTTPException(404, f"No results for {condition}/{image_set}")

    # ── Choose column ──────────────────────────────────────────────
    col_map = {
        "mean": f"{channel}_MeanIntensity",
        "ctcf": f"{channel}_CTCF",
        "background": f"{channel}_Background",
    }
    col = col_map.get(metric, f"{channel}_MeanIntensity")
    if col not in subset.columns:
        available = [c for c in subset.columns if c.startswith(channel + "_")]
        raise HTTPException(
            404,
            f"Column '{col}' not found. Available for this channel: {available}"
        )

    # ── Build vectorised LUT: label → value ───────────────────────
    cell_ids = subset["CellID"].astype(int).values
    values = subset[col].astype(float).values

    max_label = int(masks.max())
    lut_values = np.zeros(max_label + 1, dtype=np.float64)
    lut_valid = np.zeros(max_label + 1, dtype=bool)
    for cid, val in zip(cell_ids, values):
        if 0 <= cid <= max_label:
            lut_values[cid] = val
            lut_valid[cid] = True

    # Paint: each pixel gets its cell's value (0 where no cell)
    value_img = lut_values[masks]   # (H, W) float
    has_cell = lut_valid[masks]     # (H, W) bool

    # ── Normalize and apply colormap ───────────────────────────────
    valid_vals = value_img[has_cell]
    if len(valid_vals) == 0:
        raise HTTPException(404, "No valid cell values to render")

    vmin, vmax = float(valid_vals.min()), float(valid_vals.max())
    if vmax == vmin:
        vmax = vmin + 1.0

    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap_name = "RdYlBu_r" if metric == "background" else "viridis"
    cmap = cm.get_cmap(cmap_name)

    rgba = cmap(norm(value_img))                    # (H, W, 4) float [0,1]
    rgba[~has_cell] = [0.08, 0.08, 0.08, 1.0]      # dark background

    img_uint8 = (rgba[:, :, :3] * 255).astype(np.uint8)

    # ── Encode as PNG ──────────────────────────────────────────────
    pil_img = Image.fromarray(img_uint8)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG", optimize=False)
    buf.seek(0)

    return Response(
        content=buf.read(),
        media_type="image/png",
        headers={
            "Cache-Control": "no-cache",
            "X-Vmin": str(round(vmin, 4)),
            "X-Vmax": str(round(vmax, 4)),
        },
    )


@router.get("/mask-id-map/{session_id}")
async def get_mask_id_map(
    session_id: str,
    condition: str = Query(...),
    image_set: str = Query(...),
):
    """
    Return the segmentation mask as a lossless RGB PNG where each pixel
    encodes the integer cell label:  cell_id = (R << 8) | G.
    Background pixels are (0, 0, 0).  Supports up to 65 535 cells per image.
    """
    from PIL import Image

    session = get_session(session_id)

    mask_path = session.get_mask_path(condition, image_set)
    if not mask_path.exists():
        masks_mem = session.masks.get(condition, {}).get(image_set)
        if masks_mem is None:
            raise HTTPException(404, f"Mask not found: {condition}/{image_set}")
        masks = masks_mem
    else:
        masks = np.load(str(mask_path))

    masks_u16 = masks.astype(np.uint16)
    r = (masks_u16 >> 8).astype(np.uint8)
    g = (masks_u16 & 0xFF).astype(np.uint8)
    b = np.zeros_like(r)
    rgb = np.stack([r, g, b], axis=-1)

    pil_img = Image.fromarray(rgb, mode="RGB")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)

    return Response(
        content=buf.read(),
        media_type="image/png",
        headers={"Cache-Control": "no-cache"},
    )


@router.get("/cell-data/{session_id}")
async def get_cell_data(
    session_id: str,
    condition: str = Query(...),
    image_set: str = Query(...),
):
    """
    Return per-cell quantification data for a specific image set as a JSON
    dict keyed by cell_id (string).  Values include all available metrics.
    """
    session = get_session(session_id)

    if session.results_df is None:
        session.load_results()
    if session.results_df is None:
        raise HTTPException(404, "No quantification results found")

    df = session.results_df
    subset = df[
        (df["Condition"] == condition) & (df["ImageSet"] == image_set)
    ]

    if len(subset) == 0:
        raise HTTPException(404, f"No results for {condition}/{image_set}")

    # Return all numeric columns, keyed by cell_id
    cells: dict[str, dict] = {}
    for _, row in subset.iterrows():
        cell_id = int(row["CellID"])
        record: dict = {}
        for col in subset.columns:
            if col in ("Condition", "ImageSet", "SegmentationType"):
                continue
            val = row[col]
            if isinstance(val, (int, float, np.integer, np.floating)):
                record[col] = None if (isinstance(val, float) and math.isnan(val)) else float(val)
            elif isinstance(val, bool):
                record[col] = bool(val)
            else:
                record[col] = str(val)
        cells[str(cell_id)] = record

    return {"condition": condition, "image_set": image_set, "cells": cells}


@router.get("/background-regions/{session_id}")
async def get_background_regions(
    session_id: str,
    condition: str = Query(...),
    image_set: str = Query(...),
    channel: str = Query(...),
):
    """
    Render the background SAMPLING REGIONS used for each cell.

    - Background pixels are colored by the background value assigned to
      the nearest cell (Voronoi partition, same geometry as the voronoi
      background method).
    - Cell interiors are shown as dark grey.
    - Cell outlines are drawn in white so individual cells are distinguishable.
    - Colorscale: RdYlBu_r (blue = low background, red = high).

    This makes it visually clear which background area was attributed to
    each cell and whether background varies spatially across the field.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    from PIL import Image
    from scipy.ndimage import distance_transform_edt

    session = get_session(session_id)

    # ── Load mask ──────────────────────────────────────────────────
    mask_path = session.get_mask_path(condition, image_set)
    if not mask_path.exists():
        masks_mem = session.masks.get(condition, {}).get(image_set)
        if masks_mem is None:
            raise HTTPException(404, f"Mask not found: {condition}/{image_set}")
        masks = masks_mem
    else:
        masks = np.load(str(mask_path))

    # ── Load per-cell background values ───────────────────────────
    if session.results_df is None:
        session.load_results()
    if session.results_df is None:
        raise HTTPException(404, "No quantification results — run quantification first")

    df = session.results_df
    subset = df[(df["Condition"] == condition) & (df["ImageSet"] == image_set)]
    if len(subset) == 0:
        raise HTTPException(404, f"No results for {condition}/{image_set}")

    bg_col = f"{channel}_Background"
    if bg_col not in subset.columns:
        available = [c for c in subset.columns if c.endswith("_Background")]
        raise HTTPException(404, f"Column '{bg_col}' not found. Available: {available}")

    cell_ids = subset["CellID"].astype(int).values
    bg_values = subset[bg_col].astype(float).values

    # ── Voronoi partition of background pixels ────────────────────
    # For every background pixel, find the nearest cell.
    bg_mask = masks == 0          # True where no cell
    _, nearest_idx = distance_transform_edt(bg_mask, return_indices=True)
    # nearest_idx[0] = row of nearest cell pixel, [1] = col
    # The label at that position is the nearest cell id
    voronoi_labels = np.zeros_like(masks)
    voronoi_labels[bg_mask] = masks[nearest_idx[0][bg_mask], nearest_idx[1][bg_mask]]

    # ── Build LUT: cell_id → background value ─────────────────────
    max_label = int(masks.max())
    lut_bg = np.zeros(max_label + 1, dtype=np.float64)
    lut_valid = np.zeros(max_label + 1, dtype=bool)
    for cid, val in zip(cell_ids, bg_values):
        if 0 <= cid <= max_label:
            lut_bg[cid] = val
            lut_valid[cid] = True

    # Value image: background pixels get their cell's bg value; cells get 0
    value_img = np.where(bg_mask, lut_bg[voronoi_labels], 0.0)

    # ── Colormap (only over background pixels) ────────────────────
    valid_vals = bg_values[lut_valid[cell_ids]]
    if len(valid_vals) == 0:
        raise HTTPException(404, "No valid background values")

    vmin, vmax = float(valid_vals.min()), float(valid_vals.max())
    if vmax == vmin:
        vmax = vmin + 1.0

    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap("RdYlBu_r")

    rgba = cmap(norm(value_img))          # (H, W, 4)

    # ── Cell interiors → dark grey ────────────────────────────────
    cell_interior = masks > 0
    rgba[cell_interior] = [0.15, 0.15, 0.15, 1.0]

    # ── Cell outlines → white ─────────────────────────────────────
    # A cell pixel is an outline if any of its 4 neighbours has a different label
    padded = np.pad(masks, 1, constant_values=0)
    outline = (
        (masks != padded[:-2, 1:-1]) |
        (masks != padded[2:,  1:-1]) |
        (masks != padded[1:-1, :-2]) |
        (masks != padded[1:-1, 2: ])
    ) & cell_interior
    rgba[outline] = [1.0, 1.0, 1.0, 1.0]

    img_uint8 = (rgba[:, :, :3] * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)

    return Response(
        content=buf.read(),
        media_type="image/png",
        headers={
            "Cache-Control": "no-cache",
            "X-Vmin": str(round(vmin, 4)),
            "X-Vmax": str(round(vmax, 4)),
        },
    )
