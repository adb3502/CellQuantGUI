"""Pseudocolour activity overlay on the video, like SlideBook's ratio view.

Each frame is drawn in grey with cells coloured by dF/F0 at that frame,
either per cell (the exact values in the traces table) or per pixel (shows
structure inside a cell). The colour scale is fixed for the whole recording
so a colour means the same dF/F0 in every frame.
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


def _png(resp):
    return np.asarray(Image.open(io.BytesIO(_run(_body(resp)))).convert("RGB"), dtype=float)


@pytest.fixture
def activity_session(tmp_path, monkeypatch):
    # Background 100. Cell 1 (rows 4-10) flat at 400. Cell 2 (rows 20-26)
    # at 400 then jumps to 1600 from frame 10 on: dF/F0 goes 0 -> 4.
    stack = np.full((20, 32, 32), 100, np.uint16)
    stack[:, 4:10, 4:10] = 400
    stack[:, 20:26, 20:26] = 400
    stack[10:, 20:26, 20:26] = 1600
    path = tmp_path / "vid.tif"
    tifffile.imwrite(path, stack)

    masks = np.zeros((32, 32), np.int32)
    masks[4:10, 4:10] = 1
    masks[20:26, 20:26] = 2

    s = Session(id="sid", directory=tmp_path / "sess")
    s.conditions = {"A": {"image_sets": {"v": {"C0": str(path)}}}}
    s.video_params = {"baseline_mode": "fraction", "baseline_fraction": 0.25}

    from cellquant.tasks.worker import _persist_video_artifacts
    _persist_video_artifacts(s, "A", "v", masks, stack.max(axis=0))

    rows = []
    for cell in (1, 2):
        for f in range(20):
            d = 4.0 if (cell == 2 and f >= 10) else 0.0
            rows.append({"Condition": "A", "ImageSet": "v", "CellID": cell,
                         "Frame": f, "Time_s": None, "dF_F0": d})
    s.traces_df = pd.DataFrame(rows)

    import cellquant.api.routers.video as vr
    monkeypatch.setattr(vr, "get_session", lambda sid: s)
    return s


class TestTurbo:
    def test_shape_and_dtype(self):
        from cellquant.core.io.timelapse_view import turbo

        out = turbo(np.linspace(0, 1, 7))
        assert out.shape == (7, 3) and out.dtype == np.uint8

    def test_low_is_blue_high_is_red(self):
        """Turbo's very bottom (x=0) is near-black purple, so the overlay
        starts at TURBO_START: 'no change' must read as blue, not vanish."""
        from cellquant.core.io.timelapse_view import TURBO_START, turbo

        lo, hi = turbo(np.array([TURBO_START, 1.0]))
        assert lo[2] > lo[0] and lo[2] > 150      # clearly blue
        assert hi[0] > hi[2]                      # red end

    def test_no_change_is_tinted_blue_not_black(self):
        from cellquant.core.io.timelapse_view import blend_colour

        grey = np.full((1, 1, 3), 120, np.uint8)
        out = blend_colour(grey, np.array([[0.0]]), lo=0.0, hi=4.0)[0, 0]
        assert out[2] > out[0] and out.mean() > 60

    def test_clips_out_of_range(self):
        from cellquant.core.io.timelapse_view import turbo

        assert np.array_equal(turbo(np.array([-5.0])), turbo(np.array([0.0])))
        assert np.array_equal(turbo(np.array([9.0])), turbo(np.array([1.0])))


class TestCellOverlay:
    def test_responding_cell_changes_colour_and_flat_cell_does_not(self, activity_session):
        from cellquant.api.routers.video import get_video_frame

        f0 = _png(_run(get_video_frame("sid", "A", "v", 0, overlay=False, heat="cell")))
        f15 = _png(_run(get_video_frame("sid", "A", "v", 15, overlay=False, heat="cell")))
        c2_before, c2_after = f0[22, 22], f15[22, 22]
        c1_before, c1_after = f0[6, 6], f15[6, 6]
        assert c2_after[0] > c2_before[0] + 60, "responding cell should turn red"
        assert np.allclose(c1_before, c1_after, atol=2), "flat cell must keep its colour"

    def test_background_stays_grey(self, activity_session):
        from cellquant.api.routers.video import get_video_frame

        img = _png(_run(get_video_frame("sid", "A", "v", 15, overlay=False, heat="cell")))
        r, g, b = img[1, 1]
        assert r == g == b

    def test_404_without_traces(self, activity_session):
        from fastapi import HTTPException
        from cellquant.api.routers.video import get_video_frame

        activity_session.traces_df = None
        with pytest.raises(HTTPException) as e:
            _run(get_video_frame("sid", "A", "v", 0, overlay=False, heat="cell"))
        assert e.value.status_code == 404

    def test_unknown_heat_mode_is_400(self, activity_session):
        from fastapi import HTTPException
        from cellquant.api.routers.video import get_video_frame

        with pytest.raises(HTTPException) as e:
            _run(get_video_frame("sid", "A", "v", 0, overlay=False, heat="rainbow"))
        assert e.value.status_code == 400


class TestPixelOverlay:
    def test_responding_pixels_turn_red(self, activity_session):
        from cellquant.api.routers.video import get_video_frame

        f0 = _png(_run(get_video_frame("sid", "A", "v", 0, overlay=False, heat="pixel")))
        f15 = _png(_run(get_video_frame("sid", "A", "v", 15, overlay=False, heat="pixel")))
        assert f15[22, 22][0] > f0[22, 22][0] + 60

    def test_outside_cells_is_not_coloured(self, activity_session):
        from cellquant.api.routers.video import get_video_frame

        img = _png(_run(get_video_frame("sid", "A", "v", 15, overlay=False, heat="pixel")))
        r, g, b = img[14, 14]
        assert r == g == b

    def test_pixel_dff_matches_analysis_formula(self, activity_session):
        """Background-subtracted, baseline from the run's baseline window:
        cell 2 goes from (400-100) to (1600-100) -> dF/F0 = 4."""
        from cellquant.core.io.timelapse_view import pixel_dff_frame

        s = activity_session
        path = s.conditions["A"]["image_sets"]["v"]["C0"]
        masks = s.video_masks["A"]["v"]
        d = pixel_dff_frame(path, 15, masks, n_baseline=5)
        assert d[22, 22] == pytest.approx(4.0, abs=0.05)
        assert d[6, 6] == pytest.approx(0.0, abs=0.05)
        assert np.isnan(d[14, 14])


class TestActivityRange:
    def test_cell_range(self, activity_session):
        from cellquant.api.routers.video import get_activity_range

        r = _run(get_activity_range("sid", "A", "v", mode="cell"))
        assert r["lo"] <= 0.0 < r["hi"]
        assert r["hi"] == pytest.approx(4.0, abs=0.01)

    def test_pixel_range(self, activity_session):
        from cellquant.api.routers.video import get_activity_range

        r = _run(get_activity_range("sid", "A", "v", mode="pixel"))
        assert r["lo"] <= 0.0 < r["hi"]

    def test_frames_share_one_scale(self, activity_session):
        """A colour must mean the same dF/F0 in every frame."""
        from cellquant.core.io.timelapse_view import cell_activity_limits

        s = activity_session
        lims = cell_activity_limits(s.traces_df, "A", "v")
        assert lims == cell_activity_limits(s.traces_df, "A", "v")


class TestFrameCaching:
    def test_frames_are_cacheable(self, activity_session):
        """Playback preloads every frame; the browser must be allowed to keep them."""
        from cellquant.api.routers.video import get_video_frame

        resp = _run(get_video_frame("sid", "A", "v", 0, overlay=False))
        assert "max-age" in resp.headers.get("cache-control", "")


class TestRunRecordsParams:
    def test_video_params_saved_on_session(self, tmp_path):
        """The pixel overlay must use the run's own baseline window."""
        from cellquant.tasks.queue import TaskInfo
        from cellquant.tasks.worker import run_video_task

        p = tmp_path / "v.tif"
        tifffile.imwrite(p, np.full((6, 8, 8), 100, np.uint16))
        s = Session(id="t", directory=tmp_path / "s")
        s.conditions = {"A": {"image_sets": {"v": {"C0": str(p)}}}}

        class Q:
            def update_progress(self, *a, **k):
                pass

        run_video_task(TaskInfo(id="x", session_id="t", task_type="video"), Q(), s,
                       {"mode": "whole_field", "baseline_fraction": 0.3})
        assert s.video_params["baseline_fraction"] == 0.3


class TestOutlinesKeepFixedContrast:
    def test_rise_still_visible_with_outlines_on(self, activity_session):
        """The old outline helper re-stretched each frame to its own min/max,
        undoing the fixed contrast whenever ROIs were shown (the default)."""
        from cellquant.api.routers.video import get_video_frame

        f0 = _png(_run(get_video_frame("sid", "A", "v", 0, overlay=True)))
        f15 = _png(_run(get_video_frame("sid", "A", "v", 15, overlay=True)))
        # centre of cell 2, away from its outline
        assert f15[22, 22].mean() > f0[22, 22].mean() + 50

    def test_outline_pixels_are_drawn(self, activity_session):
        from cellquant.api.routers.video import get_video_frame

        img = _png(_run(get_video_frame("sid", "A", "v", 0, overlay=True)))
        r, g, b = img[4, 6]          # top edge of cell 1
        assert r > 200 and g < 80


class TestPixelNoise:
    def test_flat_noisy_cell_reads_as_flat(self, tmp_path):
        """Per-pixel dF/F0 divides single noisy pixels by a single noisy
        baseline. On a cell that does not respond the map must still read as
        'no change', not speckle."""
        from cellquant.core.io.timelapse_view import pixel_dff_frame

        rng = np.random.default_rng(0)
        stack = rng.poisson(100, size=(20, 40, 40)).astype(np.float64)
        stack[:, 10:30, 10:30] = rng.poisson(400, size=(20, 20, 20))
        p = tmp_path / "noisy.tif"
        tifffile.imwrite(p, stack.astype(np.uint16))
        masks = np.zeros((40, 40), np.int32)
        masks[10:30, 10:30] = 1

        d = pixel_dff_frame(p, 15, masks, n_baseline=5)
        inner = d[13:27, 13:27]
        assert np.nanstd(inner) < 0.04, f"speckle std {np.nanstd(inner):.3f}"

    def test_dim_baseline_pixels_are_left_out(self, tmp_path):
        """A pixel whose baseline is near background cannot be divided by
        reliably; it is left uncoloured instead of showing an extreme value."""
        from cellquant.core.io.timelapse_view import pixel_dff_frame

        stack = np.full((20, 40, 40), 100.0)
        stack[:, 10:30, 10:30] = 400.0          # bright cell
        stack[:, 10:30, 10:13] = 104.0          # dim strip inside the ROI
        p = tmp_path / "dim.tif"
        tifffile.imwrite(p, stack.astype(np.uint16))
        masks = np.zeros((40, 40), np.int32)
        masks[10:30, 10:30] = 1

        d = pixel_dff_frame(p, 15, masks, n_baseline=5)
        assert np.isnan(d[20, 10])
        assert np.isfinite(d[20, 22])
