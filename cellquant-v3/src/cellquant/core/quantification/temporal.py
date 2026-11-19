"""Temporal quantification for time-lapse reporter/dye recordings.

Deliberately independent of ctcf.py: fluorescence CTCF is a per-image
integrated measure, while this module works on per-cell traces over time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

import numpy as np

BaselineMode = Literal["fraction", "stim_frame", "auto"]


@dataclass
class TemporalConfig:
    baseline_mode: BaselineMode = "fraction"
    baseline_fraction: float = 0.10
    stim_frame: Optional[int] = None
    interval_s: Optional[float] = None
    detrend: bool = False
    event_threshold_sigma: float = 3.0
    event_refractory_frames: int = 5
    event_min_frames: int = 3  # sustained frames required to count an event
    decay_fit: bool = True


@dataclass
class TraceSummary:
    cell_id: int = 0
    baseline_f0: float = float("nan")
    peak_dff: float = float("nan")
    peak_frame: int = -1
    time_to_peak_s: Optional[float] = None
    auc: float = float("nan")
    rise_time_10_90: Optional[float] = None
    decay_tau_frames: float = float("nan")
    decay_tau_s: Optional[float] = None
    snr: float = float("nan")
    n_events: int = 0
    event_freq_hz: Optional[float] = None
    event_frames: List[int] = field(default_factory=list)
    baseline_is_fallback: bool = False  # f0 taken from first observed
                                        # frames, not a true pre-stimulus baseline


def _baseline_window(trace: np.ndarray, config: TemporalConfig) -> np.ndarray:
    n = len(trace)
    if n == 0:
        raise ValueError("Empty trace")
    if config.baseline_mode == "stim_frame" and config.stim_frame is not None:
        end = max(1, min(int(config.stim_frame), n))
    elif config.baseline_mode == "auto":
        end = _auto_onset(trace, config)
    else:
        end = max(1, int(round(n * float(config.baseline_fraction))))
    return trace[:end]


def _auto_onset(trace: np.ndarray, config: TemporalConfig) -> int:
    """First sustained departure from the leading-edge distribution."""
    n = len(trace)
    seed = max(3, int(round(n * 0.05)))
    ref = trace[:seed]
    mu, sigma = float(np.median(ref)), float(np.std(ref))
    if sigma <= 0:
        return max(1, seed)
    thresh = mu + config.event_threshold_sigma * sigma
    for i in range(seed, n - 1):
        if trace[i] > thresh and trace[i + 1] > thresh:
            return max(1, i)
    return max(1, seed)


def compute_baseline(trace: np.ndarray, config: TemporalConfig) -> float:
    """Median of the baseline window.

    Uses nanmedian: a tracked cell absent from early frames carries NaN there,
    and a NaN baseline would otherwise poison every downstream metric.
    Returns NaN only when the window holds no finite value at all.
    """
    arr = np.asarray(trace, dtype=np.float64)
    window = _baseline_window(arr, config)
    if np.any(np.isfinite(window)):
        return float(np.nanmedian(window))
    # The cell was absent for the whole baseline window (it entered the field
    # later). Fall back to its own first observed frames so a real response is
    # not reported as zero. This is NOT a pre-stimulus baseline: summarize_trace
    # sets baseline_is_fallback so the caller can tell the two apart.
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    n = max(1, int(round(len(finite) * float(config.baseline_fraction))))
    return float(np.median(finite[:n]))


def baseline_sigma(trace: np.ndarray, config: TemporalConfig) -> float:
    """Robust noise estimate over the WHOLE trace, not just the baseline window.

    A 12-frame window badly underestimates sigma for a 120-frame recording, and
    the threshold is applied to every frame, so an underestimate makes pure
    noise cross it. MAD is used because a real response would inflate a plain
    standard deviation and mask itself.
    """
    arr = np.asarray(trace, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size < 2:
        return 0.0
    mad = float(np.median(np.abs(finite - np.median(finite))))
    sigma = mad * 1.4826  # MAD -> Gaussian-equivalent sigma
    if sigma <= 0:
        sigma = float(np.std(finite))
    return sigma


def compute_dff(trace: np.ndarray, f0: float) -> np.ndarray:
    trace = np.asarray(trace, dtype=np.float64)
    if not np.isfinite(f0):
        # An unknown baseline must propagate, never read as "no response".
        return np.full_like(trace, np.nan)
    if f0 == 0:
        # After background subtraction a zero baseline means the cell sat at
        # background: dF/F0 is undefined, not zero.
        return np.full_like(trace, np.nan)
    return (trace - f0) / f0


def detrend_trace(trace: np.ndarray, config: TemporalConfig) -> np.ndarray:
    """Divide out a monoexponential photobleach fitted to the baseline."""
    from scipy.optimize import curve_fit

    trace = np.asarray(trace, dtype=np.float64)
    window = _baseline_window(trace, config)
    if len(window) < 4:
        return trace
    x = np.arange(len(window), dtype=np.float64)

    def _exp(t, a, k, c):
        return a * np.exp(-k * t) + c

    try:
        popt, _ = curve_fit(_exp, x, window, p0=(window[0] - window[-1], 0.01, window[-1]),
                            maxfev=5000)
    except Exception:
        return trace
    full = _exp(np.arange(len(trace), dtype=np.float64), *popt)
    full = np.where(np.abs(full) < 1e-9, 1e-9, full)
    return trace / full * float(np.median(window))


def detect_events(dff: np.ndarray, config: TemporalConfig,
                  baseline_sigma: float) -> List[int]:
    """Threshold crossings at k*sigma with a refractory period."""
    dff = np.asarray(dff, dtype=np.float64)
    sigma = max(float(baseline_sigma), 1e-9)
    thresh = config.event_threshold_sigma * sigma
    min_run = max(1, int(config.event_min_frames))
    events: List[int] = []
    last = -10**9
    run_start = None
    counted = False
    for i, v in enumerate(dff):
        if np.isfinite(v) and v > thresh:
            if run_start is None:
                run_start, counted = i, False
            # Only count once the crossing has been sustained: a single noisy
            # frame above threshold is not an event.
            if (not counted and (i - run_start + 1) >= min_run
                    and (run_start - last) >= config.event_refractory_frames):
                events.append(run_start)
                last = run_start
                counted = True
        else:
            run_start, counted = None, False
    return events


def fit_decay_tau(dff: np.ndarray, peak_idx: int,
                  interval_s: Optional[float] = None) -> float:
    """Returns tau in SECONDS when interval_s is given, else in FRAMES.

    Callers must keep the two apart: summarize_trace writes decay_tau_frames
    and decay_tau_s separately so a bare number can never be read as the
    wrong unit.
    """
    """Single-exponential fit to the post-peak decay.

    Returns NaN when the fit fails or there is no decay to fit, rather
    than inventing a number.
    """
    from scipy.optimize import curve_fit

    dff = np.asarray(dff, dtype=np.float64)
    if peak_idx < 0:
        return float("nan")
    seg = dff[peak_idx:]
    seg = seg[np.isfinite(seg)]
    if len(seg) < 5:
        return float("nan")
    peak = seg[0]
    if peak <= 0:
        return float("nan")
    x = np.arange(len(seg), dtype=np.float64)

    def _exp(t, a, k, c):
        return a * np.exp(-k * t) + c

    try:
        popt, _ = curve_fit(_exp, x, seg, p0=(peak, 0.1, 0.0), maxfev=5000)
    except Exception:
        return float("nan")
    k = float(popt[1])
    if not np.isfinite(k) or k <= 0:
        return float("nan")
    tau_frames = 1.0 / k
    return tau_frames * float(interval_s) if interval_s else tau_frames


def _rise_time_10_90(dff: np.ndarray, peak_idx: int) -> Optional[float]:
    peak = dff[peak_idx]
    if peak <= 0 or peak_idx == 0:
        return None
    lo, hi = 0.1 * peak, 0.9 * peak
    i_lo = i_hi = None
    for i in range(peak_idx, -1, -1):
        if i_hi is None and dff[i] <= hi:
            i_hi = i
        if dff[i] <= lo:
            i_lo = i
            break
    if i_lo is None or i_hi is None:
        return None
    rise = float(i_hi - i_lo)
    # A zero-frame rise is not a measurement, it means the signal had nowhere
    # to rise from (e.g. a step on the first frame).
    return rise if rise > 0 else None


def summarize_trace(trace: np.ndarray, config: TemporalConfig,
                    cell_id: int = 0) -> TraceSummary:
    trace = np.asarray(trace, dtype=np.float64)
    if config.detrend:
        trace = detrend_trace(trace, config)

    f0 = compute_baseline(trace, config)
    fallback = not np.any(np.isfinite(_baseline_window(trace, config)))
    sigma_raw = baseline_sigma(trace, config)
    dff = compute_dff(trace, f0)
    sigma_dff = sigma_raw / f0 if f0 else 0.0

    # An all-NaN trace has no measurable response: say so rather than
    # returning a confident zero.
    if not np.any(np.isfinite(dff)):
        return TraceSummary(cell_id=cell_id, baseline_f0=f0,
                            baseline_is_fallback=fallback)

    peak_idx = int(np.nanargmax(dff))
    peak = float(dff[peak_idx])
    events = detect_events(dff, config, sigma_dff)

    tau_frames = (fit_decay_tau(dff, peak_idx, None)
                  if config.decay_fit and events else float("nan"))

    interval = config.interval_s if config.interval_s else None
    n = len(trace)
    auc = float(np.trapezoid(dff[np.isfinite(dff)])) if hasattr(np, "trapezoid")         else float(np.trapz(dff[np.isfinite(dff)]))
    return TraceSummary(
        cell_id=cell_id,
        baseline_f0=f0,
        peak_dff=peak,
        peak_frame=peak_idx,
        time_to_peak_s=peak_idx * interval if interval else None,
        auc=auc,
        rise_time_10_90=_rise_time_10_90(dff, peak_idx),
        decay_tau_frames=tau_frames,
        decay_tau_s=(tau_frames * interval
                     if interval and np.isfinite(tau_frames) else None),
        snr=peak / sigma_dff if sigma_dff > 0 else float("nan"),
        n_events=len(events),
        event_freq_hz=(len(events) / (n * interval)) if interval else None,
        event_frames=events,
        baseline_is_fallback=fallback,
    )


def extract_traces(stack: np.ndarray, masks: np.ndarray) -> Dict[int, np.ndarray]:
    """Mean intensity per label per frame.

    stack: (T, Y, X); masks: (Y, X) static labels, or (T, Y, X) tracked labels.
    """
    from scipy import ndimage

    stack = np.asarray(stack)
    masks = np.asarray(masks)
    n_frames = stack.shape[0]

    if masks.ndim == 2:
        labels = np.unique(masks)
        labels = labels[labels > 0]
        if len(labels) == 0:
            return {}
        out = {int(l): np.zeros(n_frames, dtype=np.float64) for l in labels}
        for t in range(n_frames):
            means = ndimage.mean(stack[t], labels=masks, index=labels)
            for l, m in zip(labels, np.atleast_1d(means)):
                out[int(l)][t] = float(m)
        return out

    all_labels = np.unique(masks)
    all_labels = all_labels[all_labels > 0]
    out = {int(l): np.full(n_frames, np.nan, dtype=np.float64) for l in all_labels}
    for t in range(n_frames):
        frame_labels = np.unique(masks[t])
        frame_labels = frame_labels[frame_labels > 0]
        if len(frame_labels) == 0:
            continue
        means = ndimage.mean(stack[t], labels=masks[t], index=frame_labels)
        for l, m in zip(frame_labels, np.atleast_1d(means)):
            out[int(l)][t] = float(m)
    return out
