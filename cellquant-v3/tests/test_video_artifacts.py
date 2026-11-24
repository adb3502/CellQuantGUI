"""Video analysis must keep what it computes.

run_video_task segmented a mean projection, used the masks to build traces,
then discarded them - so there was no way to see which ROIs produced the
numbers. Every other analysis in this app persists its masks; this one did
not, which is why the video page could not show anything.
"""
import numpy as np
import pytest


class TestVideoArtifactPaths:
    def test_session_exposes_video_mask_path(self, tmp_path):
        from cellquant.sessions.models import Session

        s = Session(id="t", directory=tmp_path)
        p = s.get_video_mask_path("Ach", "vid1")
        assert p.name == "vid1_video_masks.npy"
        assert p.parent.exists(), "parent dir must be created"

    def test_video_mask_path_distinct_from_segmentation(self, tmp_path):
        """A video ROI set must not overwrite a segmentation mask."""
        from cellquant.sessions.models import Session

        s = Session(id="t", directory=tmp_path)
        assert s.get_video_mask_path("A", "x") != s.get_mask_path("A", "x")

    def test_projection_path(self, tmp_path):
        from cellquant.sessions.models import Session

        s = Session(id="t", directory=tmp_path)
        p = s.get_video_projection_path("Ach", "vid1")
        assert p.name == "vid1_projection.npy"


class TestPersistVideoArtifacts:
    def test_saves_masks_and_projection(self, tmp_path):
        from cellquant.sessions.models import Session
        from cellquant.tasks.worker import _persist_video_artifacts

        s = Session(id="t", directory=tmp_path)
        masks = np.zeros((16, 16), np.int32)
        masks[2:6, 2:6] = 1
        masks[8:12, 8:12] = 2
        proj = np.random.rand(16, 16).astype(np.float32)

        _persist_video_artifacts(s, "Ach", "vid1", masks, proj)

        assert s.get_video_mask_path("Ach", "vid1").exists()
        assert s.get_video_projection_path("Ach", "vid1").exists()
        loaded = np.load(s.get_video_mask_path("Ach", "vid1"))
        assert loaded.max() == 2

    def test_registers_in_session_for_api(self, tmp_path):
        from cellquant.sessions.models import Session
        from cellquant.tasks.worker import _persist_video_artifacts

        s = Session(id="t", directory=tmp_path)
        masks = np.zeros((8, 8), np.int32)
        masks[1:3, 1:3] = 1
        _persist_video_artifacts(s, "Ach", "vid1", masks,
                                 np.zeros((8, 8), np.float32))
        assert "vid1" in s.video_masks.get("Ach", {})
        assert s.video_masks["Ach"]["vid1"].max() == 1

    def test_survives_unwritable_dir(self, tmp_path):
        """Persisting is best-effort: it must never fail the analysis."""
        from cellquant.sessions.models import Session
        from cellquant.tasks.worker import _persist_video_artifacts

        s = Session(id="t", directory=tmp_path / "nonexistent_ro")
        # Should not raise even if something goes wrong on disk
        _persist_video_artifacts(s, "A", "b", np.zeros((4, 4), np.int32),
                                 np.zeros((4, 4), np.float32))


class TestBuildProjection:
    """A cell that lights up briefly must survive the projection."""

    def _transient_stack(self):
        # 120 frames, dark background; one cell bright in only 8 frames.
        s = np.full((120, 32, 32), 100.0)
        s[40:48, 10:14, 10:14] = 2000.0
        return s

    def test_max_preserves_a_brief_transient(self):
        from cellquant.tasks.worker import build_projection

        p = build_projection(self._transient_stack(), "max")
        assert p[12, 12] == 2000.0

    def test_mean_dilutes_a_brief_transient(self):
        """Why max is the better default: mean keeps ~8% of the peak."""
        from cellquant.tasks.worker import build_projection

        p = build_projection(self._transient_stack(), "mean")
        assert p[12, 12] < 300.0

    def test_std_highlights_change(self):
        from cellquant.tasks.worker import build_projection

        p = build_projection(self._transient_stack(), "std")
        assert p[12, 12] > p[0, 0]

    def test_percentile_rejects_a_single_hot_frame(self):
        """A cosmic ray in one frame must not become a cell."""
        from cellquant.tasks.worker import build_projection

        s = np.full((120, 16, 16), 100.0)
        s[60, 5, 5] = 50000.0          # single-frame artefact
        assert build_projection(s, "max")[5, 5] == 50000.0
        assert build_projection(s, "percentile", percentile=99.0)[5, 5] < 1000.0

    def test_default_is_max(self):
        from cellquant.tasks.worker import build_projection

        s = self._transient_stack()
        assert np.array_equal(build_projection(s), build_projection(s, "max"))

    def test_unknown_mode_raises(self):
        from cellquant.tasks.worker import build_projection

        with pytest.raises(ValueError, match="Unknown projection mode"):
            build_projection(np.zeros((4, 4, 4)), "median")

    def test_shape_is_2d(self):
        from cellquant.tasks.worker import build_projection

        for m in ("max", "mean", "std", "percentile"):
            assert build_projection(np.zeros((10, 8, 6)), m).shape == (8, 6)


class TestEngineReuse:
    """Loading Cellpose is the slow part of a run; load it once, not per video."""

    def _session(self, tmp_path, n_videos):
        import tifffile
        from cellquant.sessions.models import Session

        sets = {}
        for i in range(n_videos):
            p = tmp_path / f"v{i}.tif"
            stack = np.full((6, 16, 16), 100, np.uint16)
            stack[:, 4:8, 4:8] = 400
            tifffile.imwrite(p, stack)
            sets[f"v{i}"] = {"C0": str(p)}
        s = Session(id="t", directory=tmp_path / "sess")
        s.conditions = {"A": {"image_sets": sets}}
        return s

    def test_one_engine_for_many_videos(self, tmp_path, monkeypatch):
        import cellquant.core.segmentation.cellpose_engine as ce
        from cellquant.tasks.queue import TaskInfo
        from cellquant.tasks.worker import run_video_task

        created = []

        class FakeEngine:
            def __init__(self, *a, **kw):
                created.append(kw)

            def segment_single(self, img, **kw):
                m = np.zeros(img.shape, np.int32)
                m[4:8, 4:8] = 1

                class R:
                    masks = m
                    n_cells = 1
                return R()

        monkeypatch.setattr(ce, "CellposeEngine", FakeEngine)

        class Q:
            def update_progress(self, *a, **k):
                pass

        s = self._session(tmp_path, 3)
        out = run_video_task(TaskInfo(id="x", session_id="t", task_type="video"),
                             Q(), s, {"mode": "static_roi"})
        assert out["videos_processed"] == 3
        assert len(created) == 1, f"engine built {len(created)} times for 3 videos"

    def test_whole_field_builds_no_engine(self, tmp_path, monkeypatch):
        import cellquant.core.segmentation.cellpose_engine as ce
        from cellquant.tasks.queue import TaskInfo
        from cellquant.tasks.worker import run_video_task

        def boom(*a, **k):
            raise AssertionError("whole_field must not load Cellpose")

        monkeypatch.setattr(ce, "CellposeEngine", boom)

        class Q:
            def update_progress(self, *a, **k):
                pass

        s = self._session(tmp_path, 2)
        out = run_video_task(TaskInfo(id="x", session_id="t", task_type="video"),
                             Q(), s, {"mode": "whole_field"})
        assert out["videos_processed"] == 2
