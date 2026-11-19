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


from cellquant.core.quantification.temporal import TemporalConfig, summarize_trace

EPI = REAL_DATA / "Epi" / "epi 100uM_XY1223228405_Z0_T000_C0.tif"

requires_epi = pytest.mark.skipif(not EPI.exists(), reason="Epi data not available")


@requires_real_data
def test_whole_field_ach_matches_spec_targets():
    """Spec section 8: Ach 1 uM -> peak dF/F0 ~0.109 at frame 18."""
    arr, _ = load_timelapse(ACH)
    trace = np.asarray(arr, dtype=np.float64).reshape(arr.shape[0], -1).mean(axis=1)
    s = summarize_trace(trace, TemporalConfig(baseline_fraction=0.10))
    assert s.peak_frame == 18
    assert s.peak_dff == pytest.approx(0.109, abs=0.005)


@requires_epi
def test_whole_field_epi_matches_spec_targets():
    """Spec section 8: epi 100uM -> peak dF/F0 ~0.051 at frame 35."""
    arr, _ = load_timelapse(EPI)
    trace = np.asarray(arr, dtype=np.float64).reshape(arr.shape[0], -1).mean(axis=1)
    s = summarize_trace(trace, TemporalConfig(baseline_fraction=0.10))
    assert s.peak_frame == 35
    assert s.peak_dff == pytest.approx(0.051, abs=0.005)
