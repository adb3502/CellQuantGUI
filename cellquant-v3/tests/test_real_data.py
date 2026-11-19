"""Integration tests against the Vasu IX83 acquisitions.

These are skipped when the data is not mounted, so CI stays green.
"""
from pathlib import Path

import numpy as np
import pytest

from cellquant.core.io.image_loader import load_timelapse

REAL_DATA = Path(r"D:\Users\Saini Lab\Vasu IX83")
ACH = REAL_DATA / "Ach" / "Ach 1 uM_XY1222807246_Z0_T000_C0.tif"

requires_real_data = pytest.mark.skipif(
    not ACH.exists(), reason="Vasu IX83 data not available"
)


@requires_real_data
def test_ach_stack_shape_and_dtype():
    arr, meta = load_timelapse(ACH)
    assert arr.shape == (120, 512, 512)
    assert meta.n_frames == 120
    assert str(arr.dtype) == "uint16"


@requires_real_data
def test_ach_channel_metadata_is_read():
    _, meta = load_timelapse(ACH)
    assert meta.channel_name is not None
    assert "GFP" in meta.channel_name


@requires_real_data
def test_ach_raw_trace_matches_measured_values():
    """Spec section 2.1: F0 ~684.2, peak ~758.7 at frame 18."""
    arr, _ = load_timelapse(ACH)
    means = np.asarray(arr, dtype=np.float64).reshape(120, -1).mean(axis=1)
    assert means[:10].mean() == pytest.approx(684.2, abs=1.0)
    assert int(means.argmax()) == 18
    assert means.max() == pytest.approx(758.7, abs=1.0)
