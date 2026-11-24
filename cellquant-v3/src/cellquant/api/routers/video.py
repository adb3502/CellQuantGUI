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


@router.get("/overlay/{session_id}/{condition}/{base_name}")
async def get_video_overlay(
    session_id: str, condition: str, base_name: str,
    style: str = "outline",
):
    """The projection with its ROIs drawn on top.

    This is the only way to check that the ROIs are real cells rather than
    debris or noise - a summary table cannot show a missed cell.
    """
    import io as _io

    import numpy as np
    from fastapi.responses import StreamingResponse
    from PIL import Image

    from cellquant.core.segmentation.cellpose_engine import (
        _create_outline_overlay,
        _create_simple_overlay,
    )
    from cellquant.tiles.thumbnail import imagej_auto_contrast

    session = get_session(session_id)

    masks = session.video_masks.get(condition, {}).get(base_name)
    proj = session.video_projections.get(condition, {}).get(base_name)

    # Fall back to disk: a session restored after a restart has no arrays in
    # memory, but the files from the run are still there.
    if masks is None:
        p = session.get_video_mask_path(condition, base_name)
        if p.exists():
            masks = np.load(p)
    if proj is None:
        p = session.get_video_projection_path(condition, base_name)
        if p.exists():
            proj = np.load(p)

    if masks is None or proj is None:
        raise HTTPException(404, "No video ROIs for this image set - run video analysis first")

    base = imagej_auto_contrast(np.asarray(proj))
    rgb = (_create_outline_overlay(base, masks) if style == "outline"
           else _create_simple_overlay(base, masks))

    buf = _io.BytesIO()
    Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


@router.get("/roi-summary/{session_id}")
async def get_roi_summary(session_id: str):
    """Which image sets have ROIs, and how many cells each found.

    Lets the UI list what can be viewed without downloading any image.
    """
    import numpy as np

    session = get_session(session_id)
    out = []
    for cond, sets in session.video_masks.items():
        for base_name, m in sets.items():
            out.append({
                "condition": cond,
                "image_set": base_name,
                "n_cells": int(np.asarray(m).max()),
            })
    if not out:
        raise HTTPException(404, "No video ROIs for this session")
    return {"rois": sorted(out, key=lambda r: (r["condition"], r["image_set"]))}


def _video_path(session, condition: str, base_name: str) -> str:
    cond = session.conditions.get(condition)
    if not cond:
        raise HTTPException(404, f"No condition {condition}")
    channels = cond.get("image_sets", {}).get(base_name)
    if not channels:
        raise HTTPException(404, f"No image set {base_name}")
    return str(next(iter(channels.values())))


def _load_video_masks(session, condition: str, base_name: str):
    import numpy as np

    masks = session.video_masks.get(condition, {}).get(base_name)
    if masks is None:
        p = session.get_video_mask_path(condition, base_name)
        if p.exists():
            masks = np.load(p)
    return masks


@router.get("/info/{session_id}/{condition}/{base_name}")
async def get_video_info(session_id: str, condition: str, base_name: str):
    """Frame count for the scrubber, and whether ROIs exist yet."""
    from cellquant.core.io.timelapse_view import frame_count

    session = get_session(session_id)
    path = _video_path(session, condition, base_name)
    masks = _load_video_masks(session, condition, base_name)
    return {
        "n_frames": frame_count(path),
        "has_rois": masks is not None,
        "n_cells": int(masks.max()) if masks is not None else 0,
    }


_HEAT_MODES = ("none", "cell", "pixel")


def _require_traces(session, condition: str, base_name: str):
    df = session.traces_df
    if df is None or df.empty:
        raise HTTPException(404, "No traces yet - run video analysis to colour cells by activity")
    sub = df[(df["Condition"] == condition) & (df["ImageSet"] == base_name)]
    if sub.empty:
        raise HTTPException(404, f"No traces for {condition}/{base_name}")
    return df


def _require_masks(session, condition: str, base_name: str):
    masks = _load_video_masks(session, condition, base_name)
    if masks is None:
        raise HTTPException(404, "No ROIs yet - run video analysis first")
    return masks


def _render_frame_png(session, condition: str, base_name: str, frame: int,
                      overlay: bool, heat: str) -> bytes:
    """Synchronous render. 40-110 ms of numpy/scipy/PNG work per frame, so it
    must not run on the event loop: playback preloads every frame, and doing
    that inline would stall every other request - logins, status polls -
    for seconds."""
    import io as _io

    import numpy as np
    from PIL import Image

    from cellquant.core.io import timelapse_view as tv

    path = _video_path(session, condition, base_name)
    try:
        raw = tv.read_frame(path, frame)
    except IndexError as exc:
        raise HTTPException(400, str(exc))

    lo, hi = tv.display_limits(path)
    rgb = tv.to_rgb(tv.to_uint8(raw, lo, hi))

    if heat == "cell":
        df = _require_traces(session, condition, base_name)
        masks = _require_masks(session, condition, base_name)
        lut = tv.cell_values_for_frame(df, condition, base_name, frame, int(masks.max()))
        vlo, vhi = tv.cell_activity_limits(df, condition, base_name)
        rgb = tv.blend_colour(rgb, lut[np.asarray(masks, dtype=np.int64)], vlo, vhi)
    elif heat == "pixel":
        masks = _require_masks(session, condition, base_name)
        n0 = tv.baseline_frame_count(tv.frame_count(path), session.video_params)
        values = tv.pixel_dff_frame(path, frame, masks, n0)
        vlo, vhi = tv.pixel_activity_limits(path, masks, n0)
        rgb = tv.blend_colour(rgb, values, vlo, vhi)

    if overlay:
        masks = _load_video_masks(session, condition, base_name)
        if masks is not None:
            # Red outlines vanish against red (high-response) cells.
            colour = (255, 255, 255) if heat != "none" else (255, 40, 40)
            rgb = tv.draw_outlines(rgb, masks, colour)

    buf = _io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


@router.get("/frame/{session_id}/{condition}/{base_name}/{frame}")
async def get_video_frame(
    session_id: str, condition: str, base_name: str, frame: int,
    overlay: bool = True, heat: str = "none",
):
    """One frame at contrast fixed for the whole recording.

    overlay: draw ROI outlines.
    heat:    "cell"  - fill each ROI with its dF/F0 at this frame (the exact
                       values in the traces table);
             "pixel" - colour each pixel inside the ROIs by its own dF/F0;
             "none"  - plain frame.
    The colour scale is fixed per recording (see /activity-range), so a
    colour means the same dF/F0 in every frame.
    """
    from fastapi.responses import Response
    from starlette.concurrency import run_in_threadpool

    if heat not in _HEAT_MODES:
        raise HTTPException(400, f"heat must be one of {', '.join(_HEAT_MODES)}")
    session = get_session(session_id)
    png = await run_in_threadpool(
        _render_frame_png, session, condition, base_name, frame, overlay, heat)
    # Playback preloads every frame. The URL carries a per-run stamp, so a
    # cached frame is never stale across runs.
    return Response(png, media_type="image/png",
                    headers={"Cache-Control": "private, max-age=3600"})


@router.get("/activity-range/{session_id}/{condition}/{base_name}")
async def get_activity_range(session_id: str, condition: str, base_name: str,
                             mode: str = "cell"):
    """The fixed colour scale of the activity overlay, for the colour bar."""
    from cellquant.core.io import timelapse_view as tv

    session = get_session(session_id)
    if mode == "cell":
        df = _require_traces(session, condition, base_name)
        lo, hi = tv.cell_activity_limits(df, condition, base_name)
    elif mode == "pixel":
        path = _video_path(session, condition, base_name)
        masks = _require_masks(session, condition, base_name)
        n0 = tv.baseline_frame_count(tv.frame_count(path), session.video_params)
        lo, hi = tv.pixel_activity_limits(path, masks, n0)
    else:
        raise HTTPException(400, "mode must be 'cell' or 'pixel'")
    return {"lo": lo, "hi": hi, "mode": mode}


@router.get("/trace-matrix/{session_id}/{condition}/{base_name}")
async def get_trace_matrix(session_id: str, condition: str, base_name: str):
    """dF/F0 for every cell of ONE video, as compact arrays.

    The /traces endpoint returns every row of a whole condition (about 200k
    rows for seven 238-cell, 120-frame videos) - far too much to plot.
    """
    session = get_session(session_id)
    df = session.traces_df
    if df is None or df.empty:
        raise HTTPException(404, "No traces for this session - run video analysis first")
    sub = df[(df["Condition"] == condition) & (df["ImageSet"] == base_name)]
    if sub.empty:
        raise HTTPException(404, f"No traces for {condition}/{base_name}")

    wide = sub.pivot_table(index="Frame", columns="CellID", values="dF_F0")
    wide = wide.sort_index()
    frames = [int(f) for f in wide.index]

    time_s = None
    if "Time_s" in sub.columns and sub["Time_s"].notna().any():
        t = sub.groupby("Frame")["Time_s"].first().reindex(wide.index)
        time_s = [None if v != v else float(v) for v in t]

    def _clean(vals):
        return [None if v != v else float(v) for v in vals]

    return {
        "frames": frames,
        "time_s": time_s,
        "cells": [{"cell_id": int(c), "dff": _clean(wide[c].tolist())}
                  for c in wide.columns],
        "mean": _clean(wide.mean(axis=1).tolist()),
    }


@router.get("/per-video/{session_id}")
async def get_per_video_summary(session_id: str):
    """One row per recording: how many cells, how many responded, how strongly.

    Backs the Results page, where comparing recordings across a dose series
    matters more than any single cell.
    """
    import math

    session = get_session(session_id)
    df = session.trace_summary_df
    if df is None or df.empty:
        raise HTTPException(404, "No video results for this session")

    def _num(v):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
        return None if math.isnan(v) else v

    rows = []
    for (cond, image_set), g in df.groupby(["Condition", "ImageSet"], sort=True):
        n = int(len(g))
        responders = int((g["n_events"].fillna(0) > 0).sum()) if "n_events" in g else 0
        rows.append({
            "condition": cond,
            "image_set": image_set,
            "n_cells": n,
            "n_responders": responders,
            "pct_responders": round(100.0 * responders / n, 2) if n else 0.0,
            # Median, ignoring NaN: one cell with no detectable peak must not
            # drag a recording's value to zero.
            "median_peak_dF_F0": _num(g["peak_dF_F0"].median()) if "peak_dF_F0" in g else None,
            "median_peak_frame": _num(g["peak_frame"].median()) if "peak_frame" in g else None,
            "median_auc": _num(g["auc"].median()) if "auc" in g else None,
        })
    return {"videos": rows}


@router.get("/export/{session_id}")
async def export_video_csv(session_id: str, kind: str = "summary"):
    """Download per-cell summaries or the full per-frame traces as CSV."""
    import io as _io

    from fastapi.responses import StreamingResponse

    session = get_session(session_id)
    if kind == "summary":
        df, name = session.trace_summary_df, "video_summary.csv"
    elif kind == "traces":
        df, name = session.traces_df, "video_traces.csv"
    else:
        raise HTTPException(400, "kind must be 'summary' or 'traces'")
    if df is None or df.empty:
        raise HTTPException(404, "No video results for this session")

    buf = _io.StringIO()
    df.to_csv(buf, index=False)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )
