"""Endpoints behind the video viewer: overlay, frame scrubber, per-video traces.

Two properties matter beyond "returns a PNG":
  * a frame read must not decode the whole 120-frame file each time, or
    scrubbing a 63 MB recording reads gigabytes;
  * contrast must be FIXED across frames. Auto-contrasting each frame on its
    own rescales every frame to look equally bright, which hides the very
    thing the scrubber exists to show - cells getting brighter.
"""
import asyncio
import io

import numpy as np
import pandas as pd
import pytest
import tifffile
from PIL import Image

from cellquant.sessions.models import Session


def _run(coro):
    return asyncio.run(coro)


async def _body(resp):
    # Frames come back as a plain Response; overlays and CSVs stream.
    if not hasattr(resp, "body_iterator"):
        return resp.body
    return b"".join([c if isinstance(c, bytes) else c.encode()
                     async for c in resp.body_iterator])


@pytest.fixture
def video_session(tmp_path, monkeypatch):
    # Frames 0-9 dim, 10-19 bright: a response the scrubber must show.
    # The cell is faintly visible before it responds (300 over a 100
    # background). Per-frame auto-contrast stretches that faint cell to full
    # white, making frame 0 look identical to the responding frame - so the
    # contrast test below fails against that bug, not just against nothing.
    stack = np.full((20, 32, 32), 100, np.uint16)
    stack[:10, 8:16, 8:16] = 300
    stack[10:, 8:16, 8:16] = 1000
    path = tmp_path / "vid.tif"
    tifffile.imwrite(path, stack)

    s = Session(id="sid", directory=tmp_path / "sess")
    s.conditions = {"Ach": {"image_sets": {"vid": {"C0": str(path)}}}}
    masks = np.zeros((32, 32), np.int32)
    masks[8:16, 8:16] = 1
    masks[20:26, 20:26] = 2

    import cellquant.api.routers.video as video_router
    monkeypatch.setattr(video_router, "get_session", lambda sid: s)
    return s, masks, stack


class TestReadFrame:
    def test_reads_the_requested_frame(self, video_session):
        from cellquant.core.io.timelapse_view import read_frame

        s, _, stack = video_session
        path = s.conditions["Ach"]["image_sets"]["vid"]["C0"]
        assert np.array_equal(read_frame(path, 15), stack[15])
        assert np.array_equal(read_frame(path, 0), stack[0])

    def test_out_of_range_raises(self, video_session):
        from cellquant.core.io.timelapse_view import read_frame

        s, _, _ = video_session
        path = s.conditions["Ach"]["image_sets"]["vid"]["C0"]
        with pytest.raises(IndexError):
            read_frame(path, 20)


class TestFixedContrast:
    def test_limits_are_shared_by_every_frame(self, video_session):
        from cellquant.core.io.timelapse_view import display_limits

        s, _, _ = video_session
        path = s.conditions["Ach"]["image_sets"]["vid"]["C0"]
        lo, hi = display_limits(path)
        assert hi > lo
        assert display_limits(path) == (lo, hi)  # cached, stable

    def test_bright_frame_renders_brighter_than_dim_frame(self, video_session):
        """The regression: per-frame auto-contrast made these look identical."""
        from cellquant.api.routers.video import get_video_frame

        _run_frame = lambda f: Image.open(io.BytesIO(_run(_body(_run(
            get_video_frame("sid", "Ach", "vid", f, overlay=False))))))
        dim = np.asarray(_run_frame(0).convert("L"), float)[8:16, 8:16].mean()
        bright = np.asarray(_run_frame(15).convert("L"), float)[8:16, 8:16].mean()
        assert bright > dim + 50


class TestFrameEndpoint:
    def test_returns_png(self, video_session):
        from cellquant.api.routers.video import get_video_frame

        resp = _run(get_video_frame("sid", "Ach", "vid", 3, overlay=False))
        assert resp.media_type == "image/png"
        img = Image.open(io.BytesIO(_run(_body(resp))))
        assert img.size == (32, 32)

    def test_out_of_range_is_400(self, video_session):
        from fastapi import HTTPException
        from cellquant.api.routers.video import get_video_frame

        with pytest.raises(HTTPException) as e:
            _run(get_video_frame("sid", "Ach", "vid", 99, overlay=False))
        assert e.value.status_code == 400

    def test_frame_count_endpoint(self, video_session):
        from cellquant.api.routers.video import get_video_info

        info = _run(get_video_info("sid", "Ach", "vid"))
        assert info["n_frames"] == 20


class TestOverlayEndpoint:
    def test_overlay_from_disk_after_restart(self, video_session):
        """A restored session has no arrays in memory; files must suffice."""
        from cellquant.api.routers.video import get_video_overlay
        from cellquant.tasks.worker import _persist_video_artifacts

        s, masks, stack = video_session
        _persist_video_artifacts(s, "Ach", "vid", masks, stack.max(axis=0))
        s.video_masks.clear()
        s.video_projections.clear()

        resp = _run(get_video_overlay("sid", "Ach", "vid"))
        assert resp.media_type == "image/png"

    def test_overlay_404_before_any_run(self, video_session):
        from fastapi import HTTPException
        from cellquant.api.routers.video import get_video_overlay

        with pytest.raises(HTTPException) as e:
            _run(get_video_overlay("sid", "Ach", "vid"))
        assert e.value.status_code == 404


class TestTraceMatrix:
    def test_compact_per_video_traces(self, video_session):
        from cellquant.api.routers.video import get_trace_matrix

        s, _, _ = video_session
        rows = []
        for cell in (1, 2):
            for f in range(5):
                rows.append({"Condition": "Ach", "ImageSet": "vid",
                             "CellID": cell, "Frame": f,
                             "Time_s": None, "dF_F0": cell * 0.1 * f})
        # a different video's rows must not leak in
        rows.append({"Condition": "Ach", "ImageSet": "other", "CellID": 1,
                     "Frame": 0, "Time_s": None, "dF_F0": 9.9})
        s.traces_df = pd.DataFrame(rows)

        out = _run(get_trace_matrix("sid", "Ach", "vid"))
        assert out["frames"] == [0, 1, 2, 3, 4]
        assert [c["cell_id"] for c in out["cells"]] == [1, 2]
        assert out["cells"][1]["dff"] == pytest.approx([0, .2, .4, .6, .8])
        assert out["mean"] == pytest.approx([0, .15, .3, .45, .6])
        assert out["time_s"] is None

    def test_trace_matrix_404_without_results(self, video_session):
        from fastapi import HTTPException
        from cellquant.api.routers.video import get_trace_matrix

        with pytest.raises(HTTPException):
            _run(get_trace_matrix("sid", "Ach", "vid"))


class TestPerVideoSummary:
    """One row per recording for the Results page."""

    def _summary(self):
        return pd.DataFrame([
            {"Condition": "Ach", "ImageSet": "v1", "CellID": 1, "peak_dF_F0": 0.2, "peak_frame": 14, "n_events": 1, "auc": 2.0},
            {"Condition": "Ach", "ImageSet": "v1", "CellID": 2, "peak_dF_F0": 0.4, "peak_frame": 16, "n_events": 0, "auc": 4.0},
            {"Condition": "Ach", "ImageSet": "v1", "CellID": 3, "peak_dF_F0": None, "peak_frame": None, "n_events": 2, "auc": None},
            {"Condition": "Epi", "ImageSet": "v2", "CellID": 1, "peak_dF_F0": 1.0, "peak_frame": 30, "n_events": 0, "auc": 9.0},
        ])

    def test_one_row_per_video(self, video_session):
        from cellquant.api.routers.video import get_per_video_summary

        s, _, _ = video_session
        s.trace_summary_df = self._summary()
        rows = _run(get_per_video_summary("sid"))["videos"]
        assert [(r["condition"], r["image_set"]) for r in rows] == [("Ach", "v1"), ("Epi", "v2")]

    def test_aggregates(self, video_session):
        from cellquant.api.routers.video import get_per_video_summary

        s, _, _ = video_session
        s.trace_summary_df = self._summary()
        v1 = _run(get_per_video_summary("sid"))["videos"][0]
        assert v1["n_cells"] == 3
        assert v1["n_responders"] == 2
        assert v1["pct_responders"] == pytest.approx(66.67, abs=0.01)
        # NaN peaks are ignored, not treated as zero
        assert v1["median_peak_dF_F0"] == pytest.approx(0.3)
        assert v1["median_peak_frame"] == pytest.approx(15)

    def test_json_safe_when_all_nan(self, video_session):
        """A video whose every peak is NaN must not emit NaN into JSON."""
        import json
        from cellquant.api.routers.video import get_per_video_summary

        s, _, _ = video_session
        s.trace_summary_df = pd.DataFrame([
            {"Condition": "A", "ImageSet": "x", "CellID": 1, "peak_dF_F0": None,
             "peak_frame": None, "n_events": 0, "auc": None}])
        out = _run(get_per_video_summary("sid"))
        json.dumps(out, allow_nan=False)
        assert out["videos"][0]["median_peak_dF_F0"] is None

    def test_404_without_results(self, video_session):
        from fastapi import HTTPException
        from cellquant.api.routers.video import get_per_video_summary

        with pytest.raises(HTTPException):
            _run(get_per_video_summary("sid"))


class TestVideoCsvExport:
    def test_summary_csv(self, video_session):
        from cellquant.api.routers.video import export_video_csv

        s, _, _ = video_session
        s.trace_summary_df = pd.DataFrame([{"Condition": "Ach", "ImageSet": "v", "CellID": 1, "peak_dF_F0": 0.5}])
        resp = _run(export_video_csv("sid", kind="summary"))
        body = _run(_body(resp)).decode()
        assert body.splitlines()[0] == "Condition,ImageSet,CellID,peak_dF_F0"
        assert "attachment" in resp.headers["content-disposition"]
        assert "video_summary" in resp.headers["content-disposition"]

    def test_traces_csv(self, video_session):
        from cellquant.api.routers.video import export_video_csv

        s, _, _ = video_session
        s.traces_df = pd.DataFrame([{"Condition": "Ach", "ImageSet": "v", "CellID": 1, "Frame": 0, "dF_F0": 0.0}])
        resp = _run(export_video_csv("sid", kind="traces"))
        assert "video_traces" in resp.headers["content-disposition"]

    def test_bad_kind_is_400(self, video_session):
        from fastapi import HTTPException
        from cellquant.api.routers.video import export_video_csv

        with pytest.raises(HTTPException) as e:
            _run(export_video_csv("sid", kind="nonsense"))
        assert e.value.status_code == 400
