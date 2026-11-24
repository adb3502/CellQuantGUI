import numpy as np
import pytest

from cellquant.core.quantification.temporal import (
    TemporalConfig,
    compute_baseline,
    compute_dff,
    detect_events,
    extract_traces,
    fit_decay_tau,
    summarize_trace,
)


def make_pulse(n=120, f0=100.0, amp=50.0, start=20, width=10):
    t = np.full(n, f0, dtype=np.float64)
    t[start : start + width] += amp
    return t


def test_baseline_fraction_uses_median_of_leading_frames():
    trace = make_pulse()
    cfg = TemporalConfig(baseline_mode="fraction", baseline_fraction=0.10)
    assert compute_baseline(trace, cfg) == pytest.approx(100.0)


def test_baseline_stim_frame_mode():
    trace = make_pulse(start=30)
    cfg = TemporalConfig(baseline_mode="stim_frame", stim_frame=30)
    assert compute_baseline(trace, cfg) == pytest.approx(100.0)


def test_baseline_fraction_clamps_to_at_least_one_frame():
    """Review Focus 3: 0.01 * 10 frames rounds to 0 -> must clamp to 1."""
    trace = np.arange(10, dtype=np.float64) + 100.0
    cfg = TemporalConfig(baseline_mode="fraction", baseline_fraction=0.01)
    assert compute_baseline(trace, cfg) == pytest.approx(100.0)


def test_dff_computes_relative_change():
    dff = compute_dff(make_pulse(), 100.0)
    assert dff.max() == pytest.approx(0.5)
    assert dff[0] == pytest.approx(0.0)


def test_peak_dff_and_time_to_peak():
    cfg = TemporalConfig(baseline_fraction=0.10)
    s = summarize_trace(make_pulse(start=20, amp=50.0), cfg)
    assert s.peak_dff == pytest.approx(0.5)
    assert s.peak_frame == 20


def test_flat_trace_reports_no_events_and_nan_tau():
    """Review Focus 2: pure noise must not be fitted as a response."""
    rng = np.random.default_rng(0)
    trace = 100.0 + rng.normal(0, 0.5, 200)
    cfg = TemporalConfig(baseline_fraction=0.10, event_threshold_sigma=5.0)
    s = summarize_trace(trace, cfg)
    assert s.n_events == 0
    assert np.isnan(s.decay_tau_frames)


def test_detect_events_counts_separated_pulses():
    trace = np.full(200, 100.0)
    for start in (20, 80, 140):
        trace[start : start + 5] += 50.0
    cfg = TemporalConfig(baseline_fraction=0.05, event_threshold_sigma=3.0,
                         event_refractory_frames=10)
    f0 = compute_baseline(trace, cfg)
    events = detect_events(compute_dff(trace, f0), cfg, baseline_sigma=0.01)
    assert len(events) == 3


def test_fit_decay_tau_recovers_known_constant():
    n, tau_true = 120, 10.0
    t = np.arange(n, dtype=np.float64)
    dff = np.zeros(n)
    dff[20:] = 0.5 * np.exp(-(t[20:] - 20) / tau_true)
    tau = fit_decay_tau(dff, peak_idx=20, interval_s=1.0)
    assert tau == pytest.approx(tau_true, rel=0.15)


def test_time_metrics_are_none_without_interval():
    """Global constraint: never fabricate seconds when interval is unknown."""
    cfg = TemporalConfig(baseline_fraction=0.10, interval_s=None)
    s = summarize_trace(make_pulse(), cfg)
    assert s.time_to_peak_s is None
    assert s.event_freq_hz is None


def test_extract_traces_averages_within_each_label():
    stack = np.zeros((5, 4, 4), dtype=np.float64)
    stack[:, 0, 0] = 10.0
    stack[:, 3, 3] = 20.0
    masks = np.zeros((4, 4), dtype=np.int32)
    masks[0, 0] = 1
    masks[3, 3] = 2
    traces = extract_traces(stack, masks)
    assert set(traces) == {1, 2}
    assert traces[1] == pytest.approx(np.full(5, 10.0))
    assert traces[2] == pytest.approx(np.full(5, 20.0))
