"""load_image() must refuse a time-lapse instead of silently taking frame N/2.

A 120-frame SlideBook video loaded through load_image() returned ONE frame -
99.2% of the data discarded with no warning - and segmentation reported the
result as though it were a whole experiment. Wrong numbers that look right are
worse than an error.

Z-stacks keep the middle-slice behaviour: it is a legitimate projection there.
"""
import numpy as np
import pytest
import tifffile

from cellquant.core.io.image_loader import TimelapseInputError, load_image


def _write(tmp_path, name, arr, axes=None):
    p = tmp_path / name
    tifffile.imwrite(p, arr, **({"metadata": {"axes": axes}} if axes else {}))
    return p


class TestTimelapseRefused:
    def test_timelapse_raises_instead_of_collapsing(self, tmp_path):
        p = _write(tmp_path, "vid.tif", np.zeros((120, 64, 64), np.uint16))
        with pytest.raises(TimelapseInputError):
            load_image(p)

    def test_error_names_the_file_and_the_remedy(self, tmp_path):
        p = _write(tmp_path, "vid.tif", np.zeros((120, 64, 64), np.uint16))
        with pytest.raises(TimelapseInputError) as e:
            load_image(p)
        msg = str(e.value)
        assert "vid.tif" in msg
        assert "120" in msg
        assert "Video Analytics" in msg

    def test_explicit_frame_still_allowed(self, tmp_path):
        """Callers that KNOW it is a video may ask for one frame on purpose."""
        p = _write(tmp_path, "vid.tif", np.zeros((120, 64, 64), np.uint16))
        img = load_image(p, frame=0)
        assert img.shape == (64, 64)

    def test_allow_timelapse_escape_hatch(self, tmp_path):
        p = _write(tmp_path, "vid.tif", np.zeros((120, 64, 64), np.uint16))
        img = load_image(p, allow_timelapse=True)
        assert img.shape == (64, 64)


class TestZStackUnaffected:
    def test_zstack_still_takes_middle_slice(self, tmp_path):
        """The Nellie path depends on this. Must not regress."""
        arr = np.zeros((18, 64, 64), np.uint16)
        arr[9] = 7  # middle slice marked
        p = _write(tmp_path, "z.tif", arr, axes="ZYX")
        img = load_image(p)
        assert img.shape == (64, 64)
        assert img.max() == 7, "should have taken the middle slice"

    def test_plain_image_unaffected(self, tmp_path):
        p = _write(tmp_path, "img.tif", np.zeros((64, 64), np.uint16))
        assert load_image(p).shape == (64, 64)

    def test_multichannel_still_unaffected(self, tmp_path):
        p = _write(tmp_path, "c.tif", np.zeros((3, 64, 64), np.uint16))
        assert load_image(p).shape == (64, 64)
