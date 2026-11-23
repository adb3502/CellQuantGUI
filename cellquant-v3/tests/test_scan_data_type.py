"""The scan response must tell the user what kind of data it found.

Without this the UI cannot route the user, and a 120-frame recording looks
exactly like a single image right up until the numbers come out wrong.
"""
import numpy as np
import tifffile

from cellquant.api.schemas.experiments import DetectionResult


class TestDetectionResultSchema:
    def test_has_data_type_fields(self):
        d = DetectionResult(channel_suffixes=["C0"], n_channels=1,
                            n_image_sets=1, n_complete=1, n_incomplete=0)
        assert d.data_type == "image"
        assert d.n_frames_typical == 1
        assert d.mixed_data_types is False

    def test_accepts_timelapse(self):
        d = DetectionResult(channel_suffixes=["C0"], n_channels=1,
                            n_image_sets=13, n_complete=13, n_incomplete=0,
                            data_type="timelapse", n_frames_typical=120)
        assert d.data_type == "timelapse"
        assert d.n_frames_typical == 120


class TestScanClassifiesFolder:
    def _make(self, d, n, frames):
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            arr = (np.zeros((frames, 32, 32), np.uint16) if frames > 1
                   else np.zeros((32, 32), np.uint16))
            tifffile.imwrite(d / f"cond_{i}_XY100{i}_Z0_T000_C0.tif", arr)

    def test_folder_of_videos_reports_timelapse(self, tmp_path):
        from cellquant.api.routers.experiments import _summarise_folder_data_type

        self._make(tmp_path / "Ach", 3, 120)
        s = _summarise_folder_data_type(tmp_path / "Ach")
        assert s.data_type.value == "timelapse"
        assert s.n_frames_typical == 120

    def test_folder_of_images_reports_image(self, tmp_path):
        from cellquant.api.routers.experiments import _summarise_folder_data_type

        self._make(tmp_path / "Still", 3, 1)
        s = _summarise_folder_data_type(tmp_path / "Still")
        assert s.data_type.value == "image"

    def test_mixed_folder_flagged(self, tmp_path):
        from cellquant.api.routers.experiments import _summarise_folder_data_type

        d = tmp_path / "Mix"
        self._make(d, 2, 120)
        self._make(d, 1, 1)
        s = _summarise_folder_data_type(d)
        assert s.mixed is True

    def test_empty_folder_safe(self, tmp_path):
        from cellquant.api.routers.experiments import _summarise_folder_data_type

        d = tmp_path / "Empty"
        d.mkdir()
        s = _summarise_folder_data_type(d)
        assert s.data_type.value == "image"
