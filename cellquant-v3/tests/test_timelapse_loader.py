import numpy as np
import pytest
import tifffile

from cellquant.core.io.image_loader import TimelapseMeta, load_timelapse


@pytest.fixture
def stack_path(tmp_path):
    """A 12-frame, 8x8 uint16 stack with a known per-frame mean."""
    data = np.zeros((12, 8, 8), dtype=np.uint16)
    for t in range(12):
        data[t] = 100 + t
    p = tmp_path / "stack.tif"
    tifffile.imwrite(p, data)
    return p


def test_load_timelapse_preserves_all_frames(stack_path):
    arr, meta = load_timelapse(stack_path)
    assert arr.shape == (12, 8, 8)
    assert meta.n_frames == 12
    assert meta.shape == (8, 8)


def test_load_timelapse_does_not_collapse_to_middle_slice(stack_path):
    """Regression: load_image returns only the middle slice; we must not."""
    arr, _ = load_timelapse(stack_path)
    means = arr.reshape(12, -1).mean(axis=1)
    assert means[0] == pytest.approx(100.0)
    assert means[-1] == pytest.approx(111.0)


def test_single_frame_stack_returns_3d(tmp_path):
    """Review Focus 1: a 1-frame file must still produce (1, Y, X)."""
    p = tmp_path / "one.tif"
    tifffile.imwrite(p, np.full((8, 8), 50, dtype=np.uint16))
    arr, meta = load_timelapse(p)
    assert arr.ndim == 3
    assert arr.shape == (1, 8, 8)
    assert meta.n_frames == 1


def test_interval_is_never_inferred(stack_path):
    """Frame interval is user-supplied; the loader must leave it unset."""
    _, meta = load_timelapse(stack_path)
    assert meta.interval_s is None
