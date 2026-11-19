"""Regression tests for findings from the whole-branch review."""
import numpy as np
import pytest

from cellquant.core.quantification.dose_response import parse_dose
from cellquant.core.quantification.sabetagal import (
    SABetaGalConfig, otsu_threshold, quantify_sabetagal)
from cellquant.core.quantification.temporal import (
    TemporalConfig, extract_traces, summarize_trace)


# --- C1: NaN traces must propagate, not become a fabricated zero response ---

def test_late_entering_cell_gets_a_flagged_fallback_baseline():
    """A tracked cell absent during the baseline window must not have its f0
    poisoned to NaN (which previously collapsed the whole trace to zeros).
    It gets a baseline from its own first observed frames, flagged as such."""
    trace = np.array([np.nan] * 3 + [150.0] * 7)
    s = summarize_trace(trace, TemporalConfig(baseline_fraction=0.25))
    assert np.isfinite(s.baseline_f0)
    assert s.baseline_is_fallback is True, "fallback baseline must be flagged"


def test_late_entering_cell_with_a_real_rise_is_measured():
    """The C1 regression: absent early, then a genuine rise above its own
    baseline. Previously reported peak_dff = 0.0."""
    trace = np.array([np.nan] * 3 + [100.0] * 4 + [150.0] * 5)
    s = summarize_trace(trace, TemporalConfig(baseline_fraction=0.25))
    assert s.peak_dff == pytest.approx(0.5, abs=0.01),         "a real 50% rise must not report as zero"
    assert s.baseline_is_fallback is True


def test_all_nan_trace_reports_nan_everywhere():
    s = summarize_trace(np.full(10, np.nan), TemporalConfig(baseline_fraction=0.2))
    assert np.isnan(s.peak_dff)
    assert s.peak_frame == -1
    assert s.n_events == 0


# --- C2: noise must not fire events or fit a tau at the SHIPPING default ---

@pytest.mark.parametrize("seed", range(8))
def test_pure_noise_fires_no_events_at_default_sigma(seed):
    rng = np.random.default_rng(seed)
    trace = 100.0 + rng.normal(0, 1.0, 200)
    s = summarize_trace(trace, TemporalConfig())  # defaults, sigma=3.0
    assert s.n_events == 0, f"seed {seed}: {s.n_events} spurious events on pure noise"
    assert np.isnan(s.decay_tau_frames), f"seed {seed}: fabricated tau={s.decay_tau_frames}"


def test_real_response_is_still_detected_at_default_sigma():
    trace = np.full(200, 100.0)
    trace[50:70] += 50.0
    s = summarize_trace(trace, TemporalConfig())
    assert s.n_events >= 1
    assert s.peak_dff == pytest.approx(0.5, abs=0.01)


# --- C3: unitless tokens must not become concentrations ---

@pytest.mark.parametrize("name", ["plate_5m_rep1", "well_30m_timepoint",
                                  "scan_10m", "field_2m"])
def test_bare_lowercase_m_does_not_parse_as_molar(name):
    """'5m' is minutes/a well id far more often than 5 molar."""
    d = parse_dose(name)
    assert d.parsed is False, f"{name!r} wrongly parsed as {d.molar} M"


@pytest.mark.parametrize("name", ["Img_3mm_field", "scan_200nm_step"])
def test_length_like_units_are_flagged_ambiguous(name):
    """'3mm' could be millimolar or millimetres. Parse it, but force the
    confirmation gate rather than guessing silently."""
    d = parse_dose(name)
    assert d.ambiguous is True, f"{name!r} should be flagged for confirmation"


@pytest.mark.parametrize("name,molar", [
    ("Ach 1 uM_XY1222807246_Z0_T000_C0", 1e-6),
    ("Ach 10mM_XY1222803278_Z0_T000_C0", 1e-2),
    ("epi 100uM_XY1223228405_Z0_T000_C0", 1e-4),
])
def test_real_doses_still_parse(name, molar):
    d = parse_dose(name)
    assert d.parsed is True
    assert d.molar == pytest.approx(molar, rel=1e-9)


# --- I1: Otsu must refuse a unimodal (negative-control) distribution ---

def test_otsu_refuses_unimodal_negative_control():
    rng = np.random.default_rng(0)
    values = rng.normal(0.01, 0.002, 200)
    t = otsu_threshold(values)
    assert np.isnan(t), "unimodal data must not get a confident threshold"


def test_quantify_marks_all_negative_when_threshold_undecidable():
    rng = np.random.default_rng(0)
    img = np.full((40, 40, 3), 235, dtype=np.uint8)
    masks = np.zeros((40, 40), dtype=np.int32)
    for i in range(4):
        masks[i * 10:i * 10 + 8, :8] = i + 1
    df = quantify_sabetagal(img, masks, SABetaGalConfig(), "Neg", "p1")
    assert not df["is_positive"].any(), "a blank field must not report positives"


# --- I4: the chromogen channel must actually drive the measurement ---

def test_chromogen_index_changes_the_result():
    """If deconvolution is wired in, selecting a different stain axis must move
    the numbers. Dead code would make these identical."""
    img = np.full((16, 16, 3), 235, dtype=np.uint8)
    img[:8, :8, 0] = 60
    img[:8, :8, 1] = 80
    img[:8, :8, 2] = 200
    masks = np.zeros((16, 16), dtype=np.int32)
    masks[:8, :8] = 1

    a = quantify_sabetagal(img, masks, SABetaGalConfig(chromogen_index=0,
                                                      threshold=0.0), "c", "i")
    b = quantify_sabetagal(img, masks, SABetaGalConfig(chromogen_index=1,
                                                      threshold=0.0), "c", "i")
    assert a["mean_OD"].iloc[0] != b["mean_OD"].iloc[0], \
        "chromogen_index has no effect - deconvolution is not wired in"


# --- I7: decay tau must not silently switch units ---

def test_decay_tau_is_none_in_seconds_without_interval():
    trace = np.full(120, 100.0)
    trace[20:40] += 50.0
    s = summarize_trace(trace, TemporalConfig(interval_s=None))
    assert s.decay_tau_s is None, "seconds must not be reported without an interval"
    assert s.decay_tau_frames is not None


# --- I5: tracked mode must not silently run static-ROI ---

def test_tracked_mode_is_rejected_until_implemented():
    """Selecting Tracked for migrating cells and silently getting static ROI
    produces wrong traces with no warning. Until tracking is wired in, the
    request must be refused."""
    from pydantic import ValidationError
    from cellquant.api.schemas.video import VideoRunRequest

    VideoRunRequest(session_id="s", mode="static_roi")
    VideoRunRequest(session_id="s", mode="whole_field")
    with pytest.raises(ValidationError):
        VideoRunRequest(session_id="s", mode="tracked")
