"""Classify an image file as single image, time-lapse or z-stack.

Grounded in the real acquisitions this app must handle:
  (120, 512, 512)  axes IYX  - SlideBook time-lapse (Vasu IX83 dye recordings)
  (18, 1024, 1024) axes ZYX  - confocal z-stack (Nellie mito data)
  (1024, 1024)     axes YX   - single image
  (1024, 1024, 3)  axes YXS  - RGB export (SA-beta-gal)

SlideBook does NOT tag the time axis: it writes 'I' (generic sequence), not
'T'. A rule keyed on 'T' classifies every real video here as a single image.
A rule keyed on "3-D" instead breaks z-stacks, where taking the middle slice
is correct. Only the axis letters separate them.
"""
import pytest

from cellquant.core.io.data_type import (
    DataType,
    classify_axes,
)


class TestClassifyAxes:
    def test_slidebook_timelapse_iyx_is_timelapse(self):
        """The real case: 120-frame SlideBook video, axes IYX."""
        assert classify_axes("IYX", 120) == DataType.TIMELAPSE

    def test_explicit_tyx_is_timelapse(self):
        """Instruments that DO tag time."""
        assert classify_axes("TYX", 60) == DataType.TIMELAPSE

    def test_zyx_is_zstack_not_timelapse(self):
        """Nellie mito data - middle slice is legitimate, must not be a video."""
        assert classify_axes("ZYX", 18) == DataType.ZSTACK

    def test_plain_2d_is_image(self):
        assert classify_axes("YX", 1) == DataType.IMAGE

    def test_rgb_sample_axis_is_image(self):
        """(1024,1024,3) YXS - an RGB export, not 3 frames."""
        assert classify_axes("YXS", 1) == DataType.IMAGE

    def test_single_page_iyx_is_image(self):
        """One 'frame' is not a time series."""
        assert classify_axes("IYX", 1) == DataType.IMAGE

    def test_single_page_zyx_is_image(self):
        assert classify_axes("ZYX", 1) == DataType.IMAGE

    def test_channel_axis_is_image(self):
        """(3, Y, X) CYX is a multichannel still."""
        assert classify_axes("CYX", 3) == DataType.IMAGE

    def test_unknown_axes_multipage_is_timelapse(self):
        """Unrecognised axes with many pages: assume frames and WARN rather
        than silently collapsing. Wrong-but-loud beats wrong-and-silent."""
        assert classify_axes("QYX", 90) == DataType.TIMELAPSE

    def test_lowercase_axes_handled(self):
        assert classify_axes("iyx", 120) == DataType.TIMELAPSE

    def test_none_axes_falls_back_to_page_count(self):
        assert classify_axes(None, 120) == DataType.TIMELAPSE
        assert classify_axes(None, 1) == DataType.IMAGE


class TestProbeFile:
    """Header-only probe. Must not read pixel data: these files are 63 MB."""

    def test_probe_real_slidebook_timelapse(self, tmp_path):
        import numpy as np
        import tifffile
        from cellquant.core.io.data_type import probe_file

        p = tmp_path / "vid.tif"
        tifffile.imwrite(p, np.zeros((120, 64, 64), dtype=np.uint16))
        info = probe_file(p)
        assert info.data_type == DataType.TIMELAPSE
        assert info.n_frames == 120

    def test_probe_zstack_not_called_timelapse(self, tmp_path):
        import numpy as np
        import tifffile
        from cellquant.core.io.data_type import probe_file

        p = tmp_path / "z.tif"
        tifffile.imwrite(p, np.zeros((18, 64, 64), dtype=np.uint16),
                         metadata={"axes": "ZYX"})
        info = probe_file(p)
        assert info.data_type == DataType.ZSTACK

    def test_probe_single_image(self, tmp_path):
        import numpy as np
        import tifffile
        from cellquant.core.io.data_type import probe_file

        p = tmp_path / "img.tif"
        tifffile.imwrite(p, np.zeros((64, 64), dtype=np.uint16))
        info = probe_file(p)
        assert info.data_type == DataType.IMAGE
        assert info.n_frames == 1

    def test_probe_missing_file_is_image_not_crash(self, tmp_path):
        """A scan must never die on one unreadable file."""
        from cellquant.core.io.data_type import probe_file

        info = probe_file(tmp_path / "nope.tif")
        assert info.data_type == DataType.IMAGE
        assert info.error is not None


class TestSummarise:
    def test_all_timelapse(self):
        from cellquant.core.io.data_type import DataTypeInfo, summarise_data_types

        infos = [DataTypeInfo(DataType.TIMELAPSE, 120) for _ in range(13)]
        s = summarise_data_types(infos)
        assert s.data_type == DataType.TIMELAPSE
        assert s.n_frames_typical == 120
        assert not s.mixed

    def test_mixed_is_flagged(self):
        from cellquant.core.io.data_type import DataTypeInfo, summarise_data_types

        s = summarise_data_types([
            DataTypeInfo(DataType.TIMELAPSE, 120),
            DataTypeInfo(DataType.IMAGE, 1),
        ])
        assert s.mixed
        assert s.data_type == DataType.TIMELAPSE  # the majority/riskier one

    def test_empty(self):
        from cellquant.core.io.data_type import summarise_data_types

        s = summarise_data_types([])
        assert s.data_type == DataType.IMAGE
        assert s.n_frames_typical == 1
