import numpy as np
import pandas as pd
import pytest


def test_video_schemas_importable():
    from cellquant.api.schemas.video import VideoRunRequest, VideoRunResponse
    req = VideoRunRequest(session_id="abc", mode="whole_field")
    assert req.session_id == "abc"
    assert req.mode == "whole_field"
    assert req.baseline_fraction == 0.10


def test_video_router_registered():
    from cellquant.api.app import create_app
    app = create_app()
    paths = {r.path for r in app.routes}
    assert "/api/v1/video/run" in paths
    assert "/api/v1/video/summary/{session_id}" in paths


def test_session_has_trace_fields():
    from cellquant.sessions.models import Session
    from pathlib import Path
    s = Session(id="t", directory=Path("."))
    assert s.traces_df is None
    assert s.trace_summary_df is None


def test_build_summary_dataframe_shape():
    """The worker's table builder must emit the documented columns."""
    from cellquant.tasks.worker import _build_trace_tables
    from cellquant.core.quantification.temporal import TemporalConfig

    stack = np.zeros((20, 4, 4), dtype=np.float64)
    stack[:] = 100.0
    stack[5:10, 0, 0] += 50.0
    masks = np.zeros((4, 4), dtype=np.int32)
    masks[0, 0] = 1

    traces_df, summary_df = _build_trace_tables(
        stack, masks, condition="Ach", image_set="v1",
        config=TemporalConfig(baseline_fraction=0.10),
    )
    for col in ("Condition", "ImageSet", "CellID", "Frame", "F_raw", "dF_F0"):
        assert col in traces_df.columns
    for col in ("Condition", "ImageSet", "CellID", "peak_dF_F0", "peak_frame"):
        assert col in summary_df.columns
    assert len(traces_df) == 20
    assert summary_df.iloc[0]["peak_dF_F0"] == pytest.approx(0.5)
