# Video & SA-β-gal Analytics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add time-lapse video analytics (ΔF/F₀ traces, response metrics, event detection, dose-response) and SA-β-gal brightfield colorimetric analytics (colour deconvolution, optical density, % positive) to CellQuant v3.

**Architecture:** Both modalities are additive. New loaders sit beside `load_image` rather than changing it, so the existing fluorescence CTCF path is byte-identical afterwards. Each modality gets a pure-computation module under `core/quantification/`, a worker task in `tasks/worker.py` modelled on `run_nellie_task`, a router registered in `api/app.py`, and a SvelteKit page. Results merge into `session.results_df` on `["Condition","ImageSet","CellID"]`.

**Tech Stack:** Python 3.12, numpy 2.3, pandas 2.3, tifffile 2026.1, scikit-image 0.26 (`rgb2hed`), scipy 1.16 (`curve_fit`), FastAPI, pytest 9.0, SvelteKit (Svelte 5 runes).

**Spec:** `docs/superpowers/specs/2026-11-19-video-sabetagal-analytics-design.md`

## Global Constraints

- Target package: `cellquant-v3/src/cellquant/`. Do **not** modify anything under `_archive/`.
- `core/pipeline.py` (`BatchPipeline`) is vestigial for the web app — **do not modify it**.
- `load_image()` in `core/io/image_loader.py` must remain behaviourally unchanged. New loaders are added alongside it.
- No new third-party dependencies. `rgb2hed` and `curve_fit` are already available.
- Only `(T, Y, X)` video is supported. 4-D `(T, Z, Y, X)` is out of scope.
- Frame interval is **never inferred** from file metadata — it is user-supplied or absent.
- When frame interval is absent, time-based outputs (`*_s`, `*_hz`) must be `None`/`NaN`, never fabricated.
- SA-β-gal must not use `calculate_ctcf_vectorized` — its zero clamp (`ctcf.py:185`) destroys absorptive signal.
- All new DataFrames key on `Condition`, `ImageSet`, `CellID` to match the Nellie merge pattern.
- Tests live in `cellquant-v3/tests/`; run with `python -m pytest` from `cellquant-v3/`.

## Review Focus

Input classes the spec implies but no obvious task exercises. Each has its test assigned to the task owning that code.

1. **Single-frame or 2-frame "video"** — `load_timelapse` on a 1-frame TIFF must return `(1, Y, X)`, not a bare 2-D array; baseline windows must not become empty. *(Task 1, Task 3)*
2. **Flat trace with no response** — event detection must return zero events and `decay_tau = NaN` rather than fitting noise. *(Task 3)*
3. **Baseline fraction yielding zero frames** — `baseline_fraction=0.01` on a 10-frame video rounds to 0; must clamp to ≥1 frame. *(Task 3)*
4. **Dose string that does not parse** — `"Ach control"` or `"DMSO"` must be reported unparsed, never silently coerced to 0 M. *(Task 5)*
5. **RGB image that is pure white or pure black** — OD computation must not produce `inf`/`NaN` from `log10(0)`. *(Task 7)*

---

## Task 1: Time-lapse loader

**Files:**
- Modify: `cellquant-v3/src/cellquant/core/io/image_loader.py` (append; do not alter `load_image`)
- Test: `cellquant-v3/tests/test_timelapse_loader.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `TimelapseMeta` dataclass; `load_timelapse(path, *, channel=None, memmap=True) -> Tuple[np.ndarray, TimelapseMeta]`; `is_timelapse_candidate(path) -> bool`.

- [ ] **Step 1: Create the tests directory and write the failing test**

Create `cellquant-v3/tests/__init__.py` (empty) and `cellquant-v3/tests/test_timelapse_loader.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd cellquant-v3 && python -m pytest tests/test_timelapse_loader.py -v`
Expected: FAIL — `ImportError: cannot import name 'TimelapseMeta'`

- [ ] **Step 3: Implement the loader**

Append to `cellquant-v3/src/cellquant/core/io/image_loader.py`:

```python
@dataclass
class TimelapseMeta:
    """Metadata for a multi-frame time-lapse acquisition."""

    path: Path
    n_frames: int
    shape: Tuple[int, int]
    dtype: str
    channel_name: Optional[str] = None
    objective: Optional[str] = None
    interval_s: Optional[float] = None  # user-supplied; never inferred
    size_bytes: int = 0


def _read_slidebook_tags(tif) -> Tuple[Optional[str], Optional[str]]:
    """Pull channel name / objective from SlideBook private TIFF tags."""
    channel_name = objective = None
    try:
        page = tif.pages[0]
        tag = page.tags.get(65004)
        if tag is not None:
            channel_name = str(tag.value).strip() or None
        tag = page.tags.get(65003)
        if tag is not None:
            objective = str(tag.value).strip() or None
    except Exception:
        pass
    return channel_name, objective


def load_timelapse(
    image_path: Union[str, Path],
    *,
    channel: Optional[int] = None,
    memmap: bool = True,
) -> Tuple[np.ndarray, TimelapseMeta]:
    """Load a multi-frame TIFF as a (T, Y, X) stack.

    Unlike load_image(), this never collapses the leading axis.
    """
    import tifffile

    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with tifffile.TiffFile(str(image_path)) as tif:
        channel_name, objective = _read_slidebook_tags(tif)
        arr = tifffile.memmap(str(image_path)) if memmap else tif.asarray()

    arr = np.asarray(arr)

    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    elif arr.ndim == 4:
        # (T, C, Y, X) or (T, Y, X, C) -> pick a channel
        idx = 0 if channel is None else channel
        if arr.shape[1] <= 5:
            arr = arr[:, idx, :, :]
        elif arr.shape[-1] <= 5:
            arr = arr[:, :, :, idx]
        else:
            raise ValueError(f"Cannot interpret 4-D shape {arr.shape}")
    elif arr.ndim != 3:
        raise ValueError(f"Unsupported dimensionality {arr.ndim} for {image_path}")

    meta = TimelapseMeta(
        path=image_path,
        n_frames=int(arr.shape[0]),
        shape=(int(arr.shape[1]), int(arr.shape[2])),
        dtype=str(arr.dtype),
        channel_name=channel_name,
        objective=objective,
        interval_s=None,
        size_bytes=image_path.stat().st_size,
    )
    return arr, meta


def is_timelapse_candidate(image_path: Union[str, Path], min_frames: int = 2) -> bool:
    """True when a TIFF holds enough frames to be treated as a time series."""
    import tifffile

    try:
        with tifffile.TiffFile(str(image_path)) as tif:
            return len(tif.pages) >= min_frames
    except Exception:
        return False
```

Ensure `from dataclasses import dataclass`, `from pathlib import Path`, and `from typing import Optional, Tuple, Union` are present at the top of the file.

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd cellquant-v3 && python -m pytest tests/test_timelapse_loader.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Verify `load_image` is unchanged**

Run: `cd cellquant-v3 && git diff --stat src/cellquant/core/io/image_loader.py`
Expected: only additions; no edits to lines 36–93.

- [ ] **Step 6: Commit**

```bash
git add cellquant-v3/tests/__init__.py cellquant-v3/tests/test_timelapse_loader.py cellquant-v3/src/cellquant/core/io/image_loader.py
git commit -m "feat(io): add load_timelapse preserving the T axis"
```

---

## Task 2: Validate the loader against real data

**Files:**
- Test: `cellquant-v3/tests/test_real_data.py`

**Interfaces:**
- Consumes: `load_timelapse` from Task 1.
- Produces: the `REAL_DATA` path constant and `requires_real_data` skip marker reused by Task 4.

- [ ] **Step 1: Write the test**

Create `cellquant-v3/tests/test_real_data.py`:

```python
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
```

- [ ] **Step 2: Run it**

Run: `cd cellquant-v3 && python -m pytest tests/test_real_data.py -v`
Expected: PASS (3 tests) on this machine; SKIPPED elsewhere.

- [ ] **Step 3: Commit**

```bash
git add cellquant-v3/tests/test_real_data.py
git commit -m "test(io): pin loader against real Vasu IX83 acquisitions"
```

---

## Task 3: Temporal quantification module

**Files:**
- Create: `cellquant-v3/src/cellquant/core/quantification/temporal.py`
- Test: `cellquant-v3/tests/test_temporal.py`

**Interfaces:**
- Consumes: nothing (pure computation on arrays).
- Produces: `TemporalConfig`; `compute_baseline(trace, config) -> float`; `compute_dff(trace, f0) -> np.ndarray`; `detect_events(dff, config, baseline_sigma) -> List[int]`; `fit_decay_tau(dff, peak_idx, interval_s) -> float`; `extract_traces(stack, masks) -> Dict[int, np.ndarray]`; `summarize_trace(trace, config) -> TraceSummary`; `TraceSummary` dataclass.

- [ ] **Step 1: Write the failing tests**

Create `cellquant-v3/tests/test_temporal.py`:

```python
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
    assert np.isnan(s.decay_tau)


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
```

- [ ] **Step 2: Run to verify failure**

Run: `cd cellquant-v3 && python -m pytest tests/test_temporal.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cellquant.core.quantification.temporal'`

- [ ] **Step 3: Implement the module**

Create `cellquant-v3/src/cellquant/core/quantification/temporal.py`:

```python
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
    decay_tau: float = float("nan")
    snr: float = float("nan")
    n_events: int = 0
    event_freq_hz: Optional[float] = None
    event_frames: List[int] = field(default_factory=list)


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
    """Median of the baseline window. Median resists a stray bright frame."""
    return float(np.median(_baseline_window(np.asarray(trace, dtype=np.float64), config)))


def baseline_sigma(trace: np.ndarray, config: TemporalConfig) -> float:
    return float(np.std(_baseline_window(np.asarray(trace, dtype=np.float64), config)))


def compute_dff(trace: np.ndarray, f0: float) -> np.ndarray:
    trace = np.asarray(trace, dtype=np.float64)
    if f0 == 0 or not np.isfinite(f0):
        return np.zeros_like(trace)
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
    events: List[int] = []
    last = -10**9
    above = False
    for i, v in enumerate(dff):
        if v > thresh:
            if not above and (i - last) >= config.event_refractory_frames:
                events.append(i)
                last = i
            above = True
        else:
            above = False
    return events


def fit_decay_tau(dff: np.ndarray, peak_idx: int,
                  interval_s: Optional[float] = None) -> float:
    """Single-exponential fit to the post-peak decay.

    Returns NaN when the fit fails or there is no decay to fit, rather
    than inventing a number.
    """
    from scipy.optimize import curve_fit

    dff = np.asarray(dff, dtype=np.float64)
    seg = dff[peak_idx:]
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
    return float(i_hi - i_lo)


def summarize_trace(trace: np.ndarray, config: TemporalConfig,
                    cell_id: int = 0) -> TraceSummary:
    trace = np.asarray(trace, dtype=np.float64)
    if config.detrend:
        trace = detrend_trace(trace, config)

    f0 = compute_baseline(trace, config)
    sigma_raw = baseline_sigma(trace, config)
    dff = compute_dff(trace, f0)
    sigma_dff = sigma_raw / f0 if f0 else 0.0

    peak_idx = int(np.argmax(dff))
    peak = float(dff[peak_idx])
    events = detect_events(dff, config, sigma_dff)

    tau = (fit_decay_tau(dff, peak_idx, config.interval_s)
           if config.decay_fit and events else float("nan"))

    interval = config.interval_s
    n = len(trace)
    return TraceSummary(
        cell_id=cell_id,
        baseline_f0=f0,
        peak_dff=peak,
        peak_frame=peak_idx,
        time_to_peak_s=peak_idx * interval if interval else None,
        auc=float(np.trapezoid(dff)) if hasattr(np, "trapezoid") else float(np.trapz(dff)),
        rise_time_10_90=_rise_time_10_90(dff, peak_idx),
        decay_tau=tau,
        snr=peak / sigma_dff if sigma_dff > 0 else float("nan"),
        n_events=len(events),
        event_freq_hz=(len(events) / (n * interval)) if interval else None,
        event_frames=events,
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
```

- [ ] **Step 4: Run to verify the tests pass**

Run: `cd cellquant-v3 && python -m pytest tests/test_temporal.py -v`
Expected: PASS (10 tests)

- [ ] **Step 5: Commit**

```bash
git add cellquant-v3/src/cellquant/core/quantification/temporal.py cellquant-v3/tests/test_temporal.py
git commit -m "feat(quant): add temporal module for dF/F0 traces and response metrics"
```

---

## Task 4: End-to-end whole-field validation on real data

**Files:**
- Modify: `cellquant-v3/tests/test_real_data.py` (append)

**Interfaces:**
- Consumes: `load_timelapse` (Task 1), `TemporalConfig` + `summarize_trace` (Task 3).
- Produces: nothing new; this is the spec's §8 acceptance gate.

- [ ] **Step 1: Write the acceptance test**

Append to `cellquant-v3/tests/test_real_data.py`:

```python
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
```

- [ ] **Step 2: Run the acceptance gate**

Run: `cd cellquant-v3 && python -m pytest tests/test_real_data.py -v`
Expected: PASS (5 tests). **If the ΔF/F₀ values do not match, stop and diagnose** — this gate decides whether the video pipeline is correct.

- [ ] **Step 3: Commit**

```bash
git add cellquant-v3/tests/test_real_data.py
git commit -m "test(video): pin whole-field dF/F0 against measured Vasu IX83 values"
```

---

## Task 5: Dose parsing and Hill fit

**Files:**
- Create: `cellquant-v3/src/cellquant/core/quantification/dose_response.py`
- Test: `cellquant-v3/tests/test_dose_response.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `ParsedDose` dataclass; `parse_dose(name) -> ParsedDose`; `parse_dose_table(names) -> List[ParsedDose]`; `fit_hill(concentrations_m, responses) -> HillFit`; `HillFit` dataclass.

- [ ] **Step 1: Write the failing tests**

Create `cellquant-v3/tests/test_dose_response.py`:

```python
import numpy as np
import pytest

from cellquant.core.quantification.dose_response import (
    fit_hill,
    parse_dose,
    parse_dose_table,
)


@pytest.mark.parametrize("name,molar", [
    ("Ach 1 uM", 1e-6),
    ("Ach 10 uM", 1e-5),
    ("Ach 100 uM", 1e-4),
    ("Ach 100nm", 1e-7),
    ("Ach 10mM", 1e-2),
    ("epi 1mM", 1e-3),
    ("epi1uM", 1e-6),
    ("epi 100mM", 1e-1),
])
def test_parse_real_filenames(name, molar):
    d = parse_dose(name)
    assert d.parsed is True
    assert d.molar == pytest.approx(molar, rel=1e-9)


def test_unparseable_dose_is_reported_not_coerced():
    """Review Focus 4: a control must never silently become 0 M."""
    d = parse_dose("Ach control")
    assert d.parsed is False
    assert d.molar is None


def test_volume_suffix_is_captured_and_flagged_ambiguous():
    a = parse_dose("Ach 1mM- 5ul")
    b = parse_dose("Ach 1mM- 50ul")
    assert a.molar == pytest.approx(1e-3)
    assert b.molar == pytest.approx(1e-3)
    assert a.volume_ul == pytest.approx(5.0)
    assert b.volume_ul == pytest.approx(50.0)
    table = parse_dose_table(["Ach 1mM- 5ul", "Ach 1mM- 50ul"])
    assert all(d.ambiguous for d in table)


def test_fit_hill_recovers_known_ec50():
    ec50_true, hill = 1e-6, 1.0
    conc = np.array([1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4])
    resp = 1.0 / (1.0 + (ec50_true / conc) ** hill)
    fit = fit_hill(conc, resp)
    assert fit.success
    assert np.log10(fit.ec50) == pytest.approx(np.log10(ec50_true), abs=0.3)


def test_fit_hill_refuses_with_too_few_points():
    fit = fit_hill(np.array([1e-6, 1e-5]), np.array([0.1, 0.5]))
    assert fit.success is False
    assert "insufficient" in fit.message.lower()
```

- [ ] **Step 2: Run to verify failure**

Run: `cd cellquant-v3 && python -m pytest tests/test_dose_response.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement**

Create `cellquant-v3/src/cellquant/core/quantification/dose_response.py`:

```python
"""Dose parsing from filenames and Hill-equation fitting.

Parsing is deliberately conservative: anything not confidently recognised is
returned unparsed for user confirmation rather than guessed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

_UNIT_TO_MOLAR = {
    "m": 1e0, "mm": 1e-3, "um": 1e-6, "µm": 1e-6, "nm": 1e-9, "pm": 1e-12,
}

_DOSE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(mM|uM|µM|nM|pM|M|mm|um|µm|nm|pm|m)\b",
                      re.IGNORECASE)
_VOLUME_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(ul|µl|uL)\b", re.IGNORECASE)


@dataclass
class ParsedDose:
    source: str
    parsed: bool = False
    value: Optional[float] = None
    unit: Optional[str] = None
    molar: Optional[float] = None
    volume_ul: Optional[float] = None
    ambiguous: bool = False
    note: str = ""


def parse_dose(name: str) -> ParsedDose:
    """Extract a concentration from a condition or filename."""
    result = ParsedDose(source=name)

    vol = _VOLUME_RE.search(name)
    if vol:
        result.volume_ul = float(vol.group(1))

    text = _VOLUME_RE.sub(" ", name)
    match = _DOSE_RE.search(text)
    if not match:
        result.note = "No concentration recognised"
        return result

    value = float(match.group(1))
    unit = match.group(2).lower().replace("µ", "u")
    factor = _UNIT_TO_MOLAR.get(unit)
    if factor is None:
        result.note = f"Unrecognised unit {match.group(2)!r}"
        return result

    result.parsed = True
    result.value = value
    result.unit = match.group(2)
    result.molar = value * factor
    return result


def parse_dose_table(names: Sequence[str]) -> List[ParsedDose]:
    """Parse many names, flagging equal concentrations that differ in volume."""
    doses = [parse_dose(n) for n in names]
    by_molar: dict = {}
    for d in doses:
        if d.parsed and d.molar is not None:
            by_molar.setdefault(d.molar, []).append(d)
    for group in by_molar.values():
        if len(group) > 1:
            volumes = {d.volume_ul for d in group}
            if len(volumes) > 1:
                for d in group:
                    d.ambiguous = True
                    d.note = "Same concentration, different volume - confirm grouping"
    return doses


@dataclass
class HillFit:
    success: bool = False
    ec50: float = float("nan")
    hill_slope: float = float("nan")
    bottom: float = float("nan")
    top: float = float("nan")
    message: str = ""


def hill_equation(conc, bottom, top, ec50, hill):
    conc = np.asarray(conc, dtype=np.float64)
    return bottom + (top - bottom) / (1.0 + (ec50 / np.maximum(conc, 1e-30)) ** hill)


def fit_hill(concentrations_m: Sequence[float], responses: Sequence[float]) -> HillFit:
    """4-parameter Hill fit. Reports failure rather than an unreliable number."""
    from scipy.optimize import curve_fit

    conc = np.asarray(concentrations_m, dtype=np.float64)
    resp = np.asarray(responses, dtype=np.float64)
    ok = np.isfinite(conc) & np.isfinite(resp) & (conc > 0)
    conc, resp = conc[ok], resp[ok]

    if len(conc) < 4:
        return HillFit(message="Insufficient data for EC50 (need >= 4 dose points)")

    p0 = (float(resp.min()), float(resp.max()), float(np.median(conc)), 1.0)
    try:
        popt, _ = curve_fit(hill_equation, conc, resp, p0=p0, maxfev=20000)
    except Exception as exc:
        return HillFit(message=f"Fit did not converge: {exc}")

    bottom, top, ec50, hill = (float(v) for v in popt)
    if not np.isfinite(ec50) or ec50 <= 0:
        return HillFit(message="Fit produced a non-physical EC50")

    return HillFit(success=True, ec50=ec50, hill_slope=hill,
                   bottom=bottom, top=top, message="OK")
```

- [ ] **Step 4: Run to verify pass**

Run: `cd cellquant-v3 && python -m pytest tests/test_dose_response.py -v`
Expected: PASS (12 tests)

- [ ] **Step 5: Commit**

```bash
git add cellquant-v3/src/cellquant/core/quantification/dose_response.py cellquant-v3/tests/test_dose_response.py
git commit -m "feat(quant): add dose parsing and Hill EC50 fitting"
```

---

## Task 6: Video worker task, schemas and router

**Files:**
- Create: `cellquant-v3/src/cellquant/api/schemas/video.py`
- Create: `cellquant-v3/src/cellquant/api/routers/video.py`
- Modify: `cellquant-v3/src/cellquant/tasks/worker.py` (append `run_video_task`)
- Modify: `cellquant-v3/src/cellquant/sessions/models.py` (add `traces_df`, `trace_summary_df`)
- Modify: `cellquant-v3/src/cellquant/api/app.py` (register router)
- Test: `cellquant-v3/tests/test_video_api.py`

**Interfaces:**
- Consumes: `load_timelapse` (Task 1); `TemporalConfig`, `extract_traces`, `summarize_trace` (Task 3); `parse_dose_table`, `fit_hill` (Task 5).
- Produces: `run_video_task(task, queue, session, video_params) -> dict`; `POST /api/v1/video/run`; `GET /api/v1/video/summary/{session_id}`; `GET /api/v1/video/parse-doses/{session_id}`.

- [ ] **Step 1: Write the failing test**

Create `cellquant-v3/tests/test_video_api.py`:

```python
import numpy as np
import pandas as pd
import pytest


def test_video_schemas_importable():
    from cellquant.api.schemas.video import VideoRunRequest, VideoRunResponse
    req = VideoRunRequest(session_id="abc", mode="whole_field")
    assert req.session_id == "abc"
    assert req.mode == "whole_field"
    assert req.baseline_fraction == 0.10


def test_video_router_registered():
    from cellquant.api.app import create_app
    app = create_app()
    paths = {r.path for r in app.routes}
    assert "/api/v1/video/run" in paths
    assert "/api/v1/video/summary/{session_id}" in paths


def test_session_has_trace_fields():
    from cellquant.sessions.models import Session
    from pathlib import Path
    s = Session(id="t", directory=Path("."))
    assert s.traces_df is None
    assert s.trace_summary_df is None


def test_build_summary_dataframe_shape():
    """The worker's table builder must emit the documented columns."""
    from cellquant.tasks.worker import _build_trace_tables
    from cellquant.core.quantification.temporal import TemporalConfig

    stack = np.zeros((20, 4, 4), dtype=np.float64)
    stack[:] = 100.0
    stack[5:10, 0, 0] += 50.0
    masks = np.zeros((4, 4), dtype=np.int32)
    masks[0, 0] = 1

    traces_df, summary_df = _build_trace_tables(
        stack, masks, condition="Ach", image_set="v1",
        config=TemporalConfig(baseline_fraction=0.10),
    )
    for col in ("Condition", "ImageSet", "CellID", "Frame", "F_raw", "dF_F0"):
        assert col in traces_df.columns
    for col in ("Condition", "ImageSet", "CellID", "peak_dF_F0", "peak_frame"):
        assert col in summary_df.columns
    assert len(traces_df) == 20
    assert summary_df.iloc[0]["peak_dF_F0"] == pytest.approx(0.5)
```

- [ ] **Step 2: Run to verify failure**

Run: `cd cellquant-v3 && python -m pytest tests/test_video_api.py -v`
Expected: FAIL — schemas module not found.

- [ ] **Step 3: Add session fields**

In `cellquant-v3/src/cellquant/sessions/models.py`, directly after the `results_df` field:

```python
    # Video analytics
    traces_df: Optional[pd.DataFrame] = None
    trace_summary_df: Optional[pd.DataFrame] = None

    # SA-beta-gal analytics
    sabetagal_df: Optional[pd.DataFrame] = None
```

- [ ] **Step 4: Create the schemas**

Create `cellquant-v3/src/cellquant/api/schemas/video.py`:

```python
"""Request/response models for video analytics."""
from typing import List, Literal, Optional

from pydantic import BaseModel


class VideoRunRequest(BaseModel):
    session_id: str
    mode: Literal["static_roi", "tracked", "whole_field"] = "static_roi"
    condition: Optional[str] = None
    baseline_mode: Literal["fraction", "stim_frame", "auto"] = "fraction"
    baseline_fraction: float = 0.10
    stim_frame: Optional[int] = None
    interval_s: Optional[float] = None
    detrend: bool = False
    event_threshold_sigma: float = 3.0
    event_refractory_frames: int = 5
    diameter: float = 30.0
    model_type: str = "cpsam"


class VideoRunResponse(BaseModel):
    task_id: str


class ParsedDoseOut(BaseModel):
    source: str
    parsed: bool
    molar: Optional[float] = None
    unit: Optional[str] = None
    volume_ul: Optional[float] = None
    ambiguous: bool = False
    note: str = ""


class DoseTableResponse(BaseModel):
    doses: List[ParsedDoseOut]
    needs_confirmation: bool
```

- [ ] **Step 5: Add the worker task**

Append to `cellquant-v3/src/cellquant/tasks/worker.py`:

```python
def _build_trace_tables(stack, masks, condition, image_set, config):
    """Build the long traces table and the per-cell summary table."""
    import numpy as np
    import pandas as pd

    from cellquant.core.quantification.temporal import (
        compute_baseline,
        compute_dff,
        extract_traces,
        summarize_trace,
    )

    traces = extract_traces(stack, masks)
    trace_rows = []
    summary_rows = []

    for cell_id, trace in sorted(traces.items()):
        f0 = compute_baseline(trace, config)
        dff = compute_dff(trace, f0)
        summary = summarize_trace(trace, config, cell_id=cell_id)

        for frame, (raw, d) in enumerate(zip(trace, dff)):
            trace_rows.append({
                "Condition": condition,
                "ImageSet": image_set,
                "CellID": int(cell_id),
                "Frame": int(frame),
                "Time_s": frame * config.interval_s if config.interval_s else None,
                "F_raw": float(raw),
                "dF_F0": float(d),
            })

        summary_rows.append({
            "Condition": condition,
            "ImageSet": image_set,
            "CellID": int(cell_id),
            "baseline_f0": summary.baseline_f0,
            "peak_dF_F0": summary.peak_dff,
            "peak_frame": summary.peak_frame,
            "time_to_peak_s": summary.time_to_peak_s,
            "auc": summary.auc,
            "rise_time_10_90": summary.rise_time_10_90,
            "decay_tau": summary.decay_tau,
            "snr": summary.snr,
            "n_events": summary.n_events,
            "event_freq_hz": summary.event_freq_hz,
        })

    return pd.DataFrame(trace_rows), pd.DataFrame(summary_rows)


def run_video_task(task, queue, session, video_params: dict) -> dict:
    """Analyse time-lapse videos: traces, response metrics, dose-response."""
    import numpy as np
    import pandas as pd

    from cellquant.core.io.image_loader import load_timelapse, normalize_image
    from cellquant.core.quantification.temporal import TemporalConfig

    config = TemporalConfig(
        baseline_mode=video_params.get("baseline_mode", "fraction"),
        baseline_fraction=video_params.get("baseline_fraction", 0.10),
        stim_frame=video_params.get("stim_frame"),
        interval_s=video_params.get("interval_s"),
        detrend=video_params.get("detrend", False),
        event_threshold_sigma=video_params.get("event_threshold_sigma", 3.0),
        event_refractory_frames=video_params.get("event_refractory_frames", 5),
    )
    mode = video_params.get("mode", "static_roi")
    only_condition = video_params.get("condition")

    targets = []
    for cond_name, cond in session.conditions.items():
        if only_condition and cond_name != only_condition:
            continue
        for base_name, channels in cond.get("image_sets", {}).items():
            path = next(iter(channels.values()))
            targets.append((cond_name, base_name, path))

    queue.update_progress(task, current=0, total=len(targets) or 1,
                          stage="video", message="Starting video analysis")

    all_traces, all_summaries = [], []

    for idx, (cond_name, base_name, path) in enumerate(targets):
        if task.status == "cancelled":
            break

        queue.update_progress(task, current=idx, total=len(targets), stage="video",
                              condition=cond_name, image_set=base_name,
                              message=f"Analysing {base_name}")

        stack, meta = load_timelapse(path)
        stack = np.asarray(stack, dtype=np.float64)

        if mode == "whole_field":
            masks = np.ones(stack.shape[1:], dtype=np.int32)
        else:
            projection = stack.mean(axis=0)
            from cellquant.core.segmentation.cellpose_engine import CellposeEngine
            engine = CellposeEngine(model_type=video_params.get("model_type", "cpsam"))
            result = engine.segment_single(
                normalize_image(projection),
                diameter=video_params.get("diameter", 30.0),
            )
            masks = result.masks
            print(f"{base_name}: segmented {result.n_cells} cells on projection")

        traces_df, summary_df = _build_trace_tables(
            stack, masks, cond_name, base_name, config
        )
        all_traces.append(traces_df)
        all_summaries.append(summary_df)

    traces_df = pd.concat(all_traces, ignore_index=True) if all_traces else pd.DataFrame()
    summary_df = (pd.concat(all_summaries, ignore_index=True)
                  if all_summaries else pd.DataFrame())

    session.traces_df = traces_df
    session.trace_summary_df = summary_df

    try:
        results_dir = session.directory / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        if not traces_df.empty:
            traces_df.to_parquet(results_dir / "video_traces.parquet")
            summary_df.to_parquet(results_dir / "video_summary.parquet")
            summary_df.to_csv(results_dir / "video_summary.csv", index=False)
    except Exception as exc:
        print(f"Warning: could not persist video results: {exc}")

    queue.update_progress(task, current=len(targets), total=len(targets) or 1,
                          stage="video", message="Video analysis complete")

    return {
        "videos_processed": len(targets),
        "cells_analysed": int(summary_df["CellID"].count()) if not summary_df.empty else 0,
        "conditions": sorted({c for c, _, _ in targets}),
    }
```

- [ ] **Step 6: Create the router**

Create `cellquant-v3/src/cellquant/api/routers/video.py`:

```python
"""Video analytics endpoints."""
from fastapi import APIRouter, HTTPException

from cellquant.api.dependencies import get_session, get_task_queue
from cellquant.api.schemas.video import (
    DoseTableResponse,
    ParsedDoseOut,
    VideoRunRequest,
    VideoRunResponse,
)
from cellquant.tasks.worker import run_video_task

router = APIRouter(prefix="/video", tags=["video"])


@router.post("/run", response_model=VideoRunResponse)
async def start_video(req: VideoRunRequest):
    session = get_session(req.session_id)
    queue = get_task_queue()

    task_id = queue.submit(
        session_id=req.session_id,
        task_type="video",
        fn=run_video_task,
        queue=queue,
        session=session,
        video_params=req.model_dump(exclude={"session_id"}),
    )
    return VideoRunResponse(task_id=task_id)


@router.get("/summary/{session_id}")
async def get_video_summary(session_id: str):
    session = get_session(session_id)
    df = session.trace_summary_df
    if df is None or df.empty:
        raise HTTPException(404, "No video results for this session")
    return {"rows": df.to_dict(orient="records"), "n_cells": len(df)}


@router.get("/traces/{session_id}/{condition}")
async def get_traces(session_id: str, condition: str):
    session = get_session(session_id)
    df = session.traces_df
    if df is None or df.empty:
        raise HTTPException(404, "No traces for this session")
    subset = df[df["Condition"] == condition]
    if subset.empty:
        raise HTTPException(404, f"No traces for condition {condition}")
    return {"rows": subset.to_dict(orient="records")}


@router.get("/parse-doses/{session_id}", response_model=DoseTableResponse)
async def parse_doses(session_id: str):
    """Parsed dose table for user confirmation before any EC50 fit."""
    from cellquant.core.quantification.dose_response import parse_dose_table

    session = get_session(session_id)
    names = []
    for cond in session.conditions.values():
        names.extend(cond.get("image_sets", {}).keys())

    doses = parse_dose_table(names)
    out = [
        ParsedDoseOut(
            source=d.source, parsed=d.parsed, molar=d.molar, unit=d.unit,
            volume_ul=d.volume_ul, ambiguous=d.ambiguous, note=d.note,
        )
        for d in doses
    ]
    return DoseTableResponse(
        doses=out,
        needs_confirmation=any((not d.parsed) or d.ambiguous for d in doses),
    )
```

- [ ] **Step 7: Register the router**

In `cellquant-v3/src/cellquant/api/app.py`, add `video` to the router import block and add alongside the other `include_router` calls (keeping the SPA catch-all last):

```python
    app.include_router(video.router, prefix="/api/v1")
```

- [ ] **Step 8: Run the tests**

Run: `cd cellquant-v3 && python -m pytest tests/test_video_api.py -v`
Expected: PASS (4 tests)

- [ ] **Step 9: Commit**

```bash
git add cellquant-v3/src/cellquant/api/schemas/video.py cellquant-v3/src/cellquant/api/routers/video.py cellquant-v3/src/cellquant/tasks/worker.py cellquant-v3/src/cellquant/sessions/models.py cellquant-v3/src/cellquant/api/app.py cellquant-v3/tests/test_video_api.py
git commit -m "feat(api): add video analytics worker, schemas and router"
```

---

## Task 7: SA-β-gal quantification module

**Files:**
- Create: `cellquant-v3/src/cellquant/core/quantification/sabetagal.py`
- Modify: `cellquant-v3/src/cellquant/core/io/image_loader.py` (append `load_rgb`)
- Test: `cellquant-v3/tests/test_sabetagal.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `load_rgb(path) -> np.ndarray`; `SABetaGalConfig`; `estimate_white_reference(rgb, percentile) -> np.ndarray`; `compute_optical_density(rgb, white) -> np.ndarray`; `deconvolve_chromogen(rgb, config) -> np.ndarray`; `quantify_sabetagal(rgb, masks, config, condition, image_set) -> pd.DataFrame`; `otsu_threshold(values) -> float`.

- [ ] **Step 1: Write the failing tests**

Create `cellquant-v3/tests/test_sabetagal.py`:

```python
import numpy as np
import pytest

from cellquant.core.quantification.sabetagal import (
    SABetaGalConfig,
    compute_optical_density,
    deconvolve_chromogen,
    estimate_white_reference,
    otsu_threshold,
    quantify_sabetagal,
)


def make_rgb(blue_fraction=0.0, size=16):
    """White field with an optional blue-stained square."""
    img = np.full((size, size, 3), 240, dtype=np.uint8)
    if blue_fraction > 0:
        n = max(1, int(size * blue_fraction))
        img[:n, :n, 0] = 60   # less red
        img[:n, :n, 1] = 80   # less green
        img[:n, :n, 2] = 200  # blue retained
    return img


def test_white_reference_tracks_bright_background():
    white = estimate_white_reference(make_rgb(), percentile=95.0)
    assert np.all(white > 200)


def test_optical_density_is_zero_for_white():
    img = make_rgb()
    od = compute_optical_density(img, estimate_white_reference(img))
    assert np.allclose(od, 0.0, atol=0.02)


def test_optical_density_increases_with_stain():
    img = make_rgb(blue_fraction=0.5)
    od = compute_optical_density(img, estimate_white_reference(img))
    assert od[0, 0] > od[-1, -1]


def test_pure_black_does_not_produce_inf_or_nan():
    """Review Focus 5: log10(0) must be guarded."""
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    od = compute_optical_density(img, np.array([255.0, 255.0, 255.0]))
    assert np.all(np.isfinite(od))


def test_pure_white_does_not_produce_nan():
    img = np.full((8, 8, 3), 255, dtype=np.uint8)
    od = compute_optical_density(img, np.array([255.0, 255.0, 255.0]))
    assert np.all(np.isfinite(od))


def test_deconvolution_separates_chromogen():
    stained = deconvolve_chromogen(make_rgb(blue_fraction=1.0), SABetaGalConfig())
    blank = deconvolve_chromogen(make_rgb(blue_fraction=0.0), SABetaGalConfig())
    assert stained.mean() > blank.mean()


def test_quantify_produces_expected_columns_and_classification():
    img = make_rgb(blue_fraction=0.5, size=16)
    masks = np.zeros((16, 16), dtype=np.int32)
    masks[:8, :8] = 1    # stained cell
    masks[8:, 8:] = 2    # unstained cell

    df = quantify_sabetagal(img, masks, SABetaGalConfig(threshold=0.05),
                            condition="Senescent", image_set="img1")

    for col in ("Condition", "ImageSet", "CellID", "Area", "mean_OD",
                "integrated_OD", "is_positive"):
        assert col in df.columns
    assert len(df) == 2
    assert bool(df.loc[df.CellID == 1, "is_positive"].iloc[0]) is True
    assert bool(df.loc[df.CellID == 2, "is_positive"].iloc[0]) is False


def test_otsu_threshold_separates_bimodal_values():
    values = np.concatenate([np.full(50, 0.01), np.full(50, 0.5)])
    t = otsu_threshold(values)
    assert 0.01 < t < 0.5


def test_load_rgb_preserves_three_channels(tmp_path):
    """Regression: load_image would keep only channel 0."""
    import tifffile
    from cellquant.core.io.image_loader import load_rgb

    p = tmp_path / "bf.tif"
    tifffile.imwrite(p, make_rgb(blue_fraction=0.5))
    arr = load_rgb(p)
    assert arr.ndim == 3
    assert arr.shape[-1] == 3
```

- [ ] **Step 2: Run to verify failure**

Run: `cd cellquant-v3 && python -m pytest tests/test_sabetagal.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Add `load_rgb`**

Append to `cellquant-v3/src/cellquant/core/io/image_loader.py`:

```python
def load_rgb(image_path: Union[str, Path]) -> np.ndarray:
    """Load a brightfield image as true colour (Y, X, 3).

    Unlike load_image(), this never reduces RGB to a single channel.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        import tifffile
        arr = tifffile.imread(str(image_path))
    except Exception:
        from skimage import io
        arr = io.imread(str(image_path))

    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
        arr = np.moveaxis(arr, 0, -1)

    if arr.ndim != 3 or arr.shape[-1] < 3:
        raise ValueError(f"Not an RGB image: shape {arr.shape}")
    return arr[:, :, :3]
```

- [ ] **Step 4: Implement the module**

Create `cellquant-v3/src/cellquant/core/quantification/sabetagal.py`:

```python
"""SA-beta-gal brightfield colorimetric quantification.

Deliberately does NOT use calculate_ctcf_vectorized: that function clamps at
zero (ctcf.py:185), which would destroy absorptive signal, since a chromogen
is darker than its background rather than brighter.

NOTE: the default stain vectors are the scikit-image HED values, fitted for
haematoxylin-DAB rather than X-gal blue. They are configurable and MUST be
calibrated against real controls before results are reported. See spec 9.2.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

_EPS = 1e-6


@dataclass
class SABetaGalConfig:
    white_percentile: float = 95.0
    chromogen_index: int = 2           # HED channel used as the chromogen proxy
    threshold: Optional[float] = None  # None -> Otsu on the per-cell OD values
    stain_vectors: Optional[List[List[float]]] = field(default=None)
    min_area: int = 10


def estimate_white_reference(rgb: np.ndarray, percentile: float = 95.0) -> np.ndarray:
    """Per-channel illumination level of the unstained background."""
    rgb = np.asarray(rgb, dtype=np.float64)
    return np.percentile(rgb.reshape(-1, rgb.shape[-1]), percentile, axis=0)


def compute_optical_density(rgb: np.ndarray, white: np.ndarray) -> np.ndarray:
    """OD = -log10(I / I_white), guarded against log10(0) and averaged over channels."""
    rgb = np.asarray(rgb, dtype=np.float64)
    white = np.asarray(white, dtype=np.float64)
    white = np.where(white <= 0, 1.0, white)
    ratio = np.clip(rgb / white, _EPS, 1.0)
    return (-np.log10(ratio)).mean(axis=-1)


def deconvolve_chromogen(rgb: np.ndarray, config: SABetaGalConfig) -> np.ndarray:
    """Separate the chromogen contribution via colour deconvolution."""
    from skimage.color import rgb2hed

    arr = np.asarray(rgb, dtype=np.float64)
    if arr.max() > 1.0:
        arr = arr / 255.0
    arr = np.clip(arr, _EPS, 1.0)

    if config.stain_vectors is not None:
        from skimage.color import separate_stains
        matrix = np.linalg.inv(np.asarray(config.stain_vectors, dtype=np.float64))
        separated = separate_stains(arr, matrix)
    else:
        separated = rgb2hed(arr)

    idx = int(config.chromogen_index)
    channel = separated[:, :, idx]
    return np.nan_to_num(channel, nan=0.0, posinf=0.0, neginf=0.0)


def otsu_threshold(values: np.ndarray) -> float:
    """Otsu cut on a 1-D set of per-cell values."""
    from skimage.filters import threshold_otsu

    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    if np.allclose(values, values[0]):
        return float(values[0])
    return float(threshold_otsu(values))


def quantify_sabetagal(rgb: np.ndarray, masks: np.ndarray,
                       config: SABetaGalConfig,
                       condition: str = "", image_set: str = "") -> pd.DataFrame:
    """Per-cell optical density and senescence-positive classification."""
    from scipy import ndimage

    rgb = np.asarray(rgb)
    masks = np.asarray(masks)

    white = estimate_white_reference(rgb, config.white_percentile)
    od = compute_optical_density(rgb, white)

    labels = np.unique(masks)
    labels = labels[labels > 0]
    if labels.size == 0:
        return pd.DataFrame(columns=["Condition", "ImageSet", "CellID", "Area",
                                     "mean_OD", "integrated_OD", "is_positive"])

    mean_od = np.atleast_1d(ndimage.mean(od, labels=masks, index=labels))
    areas = np.atleast_1d(ndimage.sum(np.ones_like(od), labels=masks, index=labels))

    threshold = (config.threshold if config.threshold is not None
                 else otsu_threshold(mean_od))

    df = pd.DataFrame({
        "Condition": condition,
        "ImageSet": image_set,
        "CellID": labels.astype(int),
        "Area": areas.astype(float),
        "mean_OD": mean_od.astype(float),
        "integrated_OD": (mean_od * areas).astype(float),
        "is_positive": (mean_od > threshold),
    })
    df.attrs["threshold"] = float(threshold)
    return df[df["Area"] >= config.min_area].reset_index(drop=True)


def summarize_positivity(df: pd.DataFrame) -> pd.DataFrame:
    """Percent-positive per condition - the standard senescence readout."""
    if df.empty:
        return pd.DataFrame(columns=["Condition", "n_cells", "n_positive",
                                     "percent_positive"])
    grouped = df.groupby("Condition")["is_positive"].agg(["count", "sum"])
    grouped.columns = ["n_cells", "n_positive"]
    grouped["percent_positive"] = 100.0 * grouped["n_positive"] / grouped["n_cells"]
    return grouped.reset_index()
```

- [ ] **Step 5: Run the tests**

Run: `cd cellquant-v3 && python -m pytest tests/test_sabetagal.py -v`
Expected: PASS (9 tests)

- [ ] **Step 6: Commit**

```bash
git add cellquant-v3/src/cellquant/core/quantification/sabetagal.py cellquant-v3/src/cellquant/core/io/image_loader.py cellquant-v3/tests/test_sabetagal.py
git commit -m "feat(quant): add SA-beta-gal colorimetric quantification"
```

---

## Task 8: SA-β-gal worker task and router

**Files:**
- Create: `cellquant-v3/src/cellquant/api/schemas/sabetagal.py`
- Create: `cellquant-v3/src/cellquant/api/routers/sabetagal.py`
- Modify: `cellquant-v3/src/cellquant/tasks/worker.py` (append `run_sabetagal_task`)
- Modify: `cellquant-v3/src/cellquant/api/app.py` (register router)
- Test: `cellquant-v3/tests/test_sabetagal_api.py`

**Interfaces:**
- Consumes: `load_rgb`, `quantify_sabetagal`, `summarize_positivity`, `SABetaGalConfig` (Task 7).
- Produces: `run_sabetagal_task(task, queue, session, sabetagal_params) -> dict`; `POST /api/v1/sabetagal/run`; `GET /api/v1/sabetagal/summary/{session_id}`.

- [ ] **Step 1: Write the failing test**

Create `cellquant-v3/tests/test_sabetagal_api.py`:

```python
def test_sabetagal_schemas_importable():
    from cellquant.api.schemas.sabetagal import SABetaGalRunRequest
    req = SABetaGalRunRequest(session_id="abc")
    assert req.session_id == "abc"
    assert req.white_percentile == 95.0
    assert req.threshold is None  # Otsu by default


def test_sabetagal_router_registered():
    from cellquant.api.app import create_app
    app = create_app()
    paths = {r.path for r in app.routes}
    assert "/api/v1/sabetagal/run" in paths
    assert "/api/v1/sabetagal/summary/{session_id}" in paths


def test_worker_task_importable():
    from cellquant.tasks.worker import run_sabetagal_task
    assert callable(run_sabetagal_task)
```

- [ ] **Step 2: Run to verify failure**

Run: `cd cellquant-v3 && python -m pytest tests/test_sabetagal_api.py -v`
Expected: FAIL — schemas module not found.

- [ ] **Step 3: Create the schemas**

Create `cellquant-v3/src/cellquant/api/schemas/sabetagal.py`:

```python
"""Request/response models for SA-beta-gal analytics."""
from typing import List, Optional

from pydantic import BaseModel


class SABetaGalRunRequest(BaseModel):
    session_id: str
    condition: Optional[str] = None
    white_percentile: float = 95.0
    chromogen_index: int = 2
    threshold: Optional[float] = None  # None -> Otsu
    stain_vectors: Optional[List[List[float]]] = None
    min_area: int = 10
    diameter: float = 30.0
    model_type: str = "cyto3"


class SABetaGalRunResponse(BaseModel):
    task_id: str
```

- [ ] **Step 4: Add the worker task**

Append to `cellquant-v3/src/cellquant/tasks/worker.py`:

```python
def run_sabetagal_task(task, queue, session, sabetagal_params: dict) -> dict:
    """Quantify SA-beta-gal brightfield staining and merge into results."""
    import numpy as np
    import pandas as pd

    from cellquant.core.io.image_loader import load_rgb, normalize_image
    from cellquant.core.quantification.sabetagal import (
        SABetaGalConfig,
        quantify_sabetagal,
        summarize_positivity,
    )

    config = SABetaGalConfig(
        white_percentile=sabetagal_params.get("white_percentile", 95.0),
        chromogen_index=sabetagal_params.get("chromogen_index", 2),
        threshold=sabetagal_params.get("threshold"),
        stain_vectors=sabetagal_params.get("stain_vectors"),
        min_area=sabetagal_params.get("min_area", 10),
    )
    only_condition = sabetagal_params.get("condition")

    targets = []
    for cond_name, cond in session.conditions.items():
        if only_condition and cond_name != only_condition:
            continue
        for base_name, channels in cond.get("image_sets", {}).items():
            targets.append((cond_name, base_name, next(iter(channels.values()))))

    queue.update_progress(task, current=0, total=len(targets) or 1,
                          stage="sabetagal", message="Starting SA-beta-gal analysis")

    frames = []
    for idx, (cond_name, base_name, path) in enumerate(targets):
        if task.status == "cancelled":
            break

        queue.update_progress(task, current=idx, total=len(targets),
                              stage="sabetagal", condition=cond_name,
                              image_set=base_name, message=f"Analysing {base_name}")

        rgb = load_rgb(path)

        masks = session.masks.get(cond_name, {}).get(base_name)
        if masks is None:
            from cellquant.core.segmentation.cellpose_engine import CellposeEngine
            gray = 255.0 - np.asarray(rgb, dtype=np.float64).mean(axis=-1)
            engine = CellposeEngine(
                model_type=sabetagal_params.get("model_type", "cyto3"))
            result = engine.segment_single(
                normalize_image(gray),
                diameter=sabetagal_params.get("diameter", 30.0),
            )
            masks = result.masks
            print(f"{base_name}: segmented {result.n_cells} cells")

        frames.append(quantify_sabetagal(rgb, masks, config, cond_name, base_name))

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    session.sabetagal_df = df

    summary = summarize_positivity(df)

    try:
        results_dir = session.directory / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        if not df.empty:
            df.to_parquet(results_dir / "sabetagal_results.parquet")
            summary.to_csv(results_dir / "sabetagal_summary.csv", index=False)
    except Exception as exc:
        print(f"Warning: could not persist SA-beta-gal results: {exc}")

    if not df.empty and session.results_df is not None:
        try:
            keys = ["Condition", "ImageSet", "CellID"]
            session.results_df = session.results_df.merge(
                df[keys + ["mean_OD", "integrated_OD", "is_positive"]],
                on=keys, how="left",
            )
        except Exception as exc:
            print(f"Warning: merge into results_df failed: {exc}")

    queue.update_progress(task, current=len(targets), total=len(targets) or 1,
                          stage="sabetagal", message="SA-beta-gal analysis complete")

    return {
        "images_processed": len(targets),
        "total_cells": int(len(df)),
        "summary": summary.to_dict(orient="records"),
        "provisional": True,  # uncalibrated stain vectors - see spec 9.2
    }
```

- [ ] **Step 5: Create the router**

Create `cellquant-v3/src/cellquant/api/routers/sabetagal.py`:

```python
"""SA-beta-gal analytics endpoints."""
from fastapi import APIRouter, HTTPException

from cellquant.api.dependencies import get_session, get_task_queue
from cellquant.api.schemas.sabetagal import SABetaGalRunRequest, SABetaGalRunResponse
from cellquant.tasks.worker import run_sabetagal_task

router = APIRouter(prefix="/sabetagal", tags=["sabetagal"])


@router.post("/run", response_model=SABetaGalRunResponse)
async def start_sabetagal(req: SABetaGalRunRequest):
    session = get_session(req.session_id)
    queue = get_task_queue()

    task_id = queue.submit(
        session_id=req.session_id,
        task_type="sabetagal",
        fn=run_sabetagal_task,
        queue=queue,
        session=session,
        sabetagal_params=req.model_dump(exclude={"session_id"}),
    )
    return SABetaGalRunResponse(task_id=task_id)


@router.get("/summary/{session_id}")
async def get_sabetagal_summary(session_id: str):
    from cellquant.core.quantification.sabetagal import summarize_positivity

    session = get_session(session_id)
    df = session.sabetagal_df
    if df is None or df.empty:
        raise HTTPException(404, "No SA-beta-gal results for this session")

    return {
        "summary": summarize_positivity(df).to_dict(orient="records"),
        "n_cells": len(df),
        "provisional": True,
    }


@router.get("/results/{session_id}")
async def get_sabetagal_results(session_id: str):
    session = get_session(session_id)
    df = session.sabetagal_df
    if df is None or df.empty:
        raise HTTPException(404, "No SA-beta-gal results for this session")
    return {"rows": df.to_dict(orient="records")}
```

- [ ] **Step 6: Register the router**

In `cellquant-v3/src/cellquant/api/app.py`, add `sabetagal` to the router imports and:

```python
    app.include_router(sabetagal.router, prefix="/api/v1")
```

- [ ] **Step 7: Run the tests**

Run: `cd cellquant-v3 && python -m pytest tests/test_sabetagal_api.py -v`
Expected: PASS (3 tests)

- [ ] **Step 8: Commit**

```bash
git add cellquant-v3/src/cellquant/api/schemas/sabetagal.py cellquant-v3/src/cellquant/api/routers/sabetagal.py cellquant-v3/src/cellquant/tasks/worker.py cellquant-v3/src/cellquant/api/app.py cellquant-v3/tests/test_sabetagal_api.py
git commit -m "feat(api): add SA-beta-gal worker and router"
```

---

## Task 9: Frontend — video page

**Files:**
- Create: `cellquant-v3/frontend/src/routes/video/+page.svelte`
- Create: `cellquant-v3/frontend/src/lib/stores/video.ts`
- Modify: `cellquant-v3/frontend/src/lib/api/client.ts` (append)

**Interfaces:**
- Consumes: `POST /api/v1/video/run`, `GET /api/v1/video/summary/{sid}`, `GET /api/v1/video/parse-doses/{sid}`.
- Produces: the `/video` route.

- [ ] **Step 1: Add the API client functions**

Append to `cellquant-v3/frontend/src/lib/api/client.ts`:

```typescript
// ---- Video analytics ----

export interface VideoRunParams {
  session_id: string;
  mode?: 'static_roi' | 'tracked' | 'whole_field';
  condition?: string;
  baseline_mode?: 'fraction' | 'stim_frame' | 'auto';
  baseline_fraction?: number;
  stim_frame?: number | null;
  interval_s?: number | null;
  detrend?: boolean;
  event_threshold_sigma?: number;
  event_refractory_frames?: number;
  diameter?: number;
}

export async function runVideo(params: VideoRunParams): Promise<{ task_id: string }> {
  return request('/video/run', { method: 'POST', body: JSON.stringify(params) });
}

export async function getVideoSummary(sessionId: string): Promise<any> {
  return request(`/video/summary/${sessionId}`);
}

export async function getVideoTraces(sessionId: string, condition: string): Promise<any> {
  return request(`/video/traces/${sessionId}/${encodeURIComponent(condition)}`);
}

export async function parseDoses(sessionId: string): Promise<any> {
  return request(`/video/parse-doses/${sessionId}`);
}
```

- [ ] **Step 2: Create the store**

Create `cellquant-v3/frontend/src/lib/stores/video.ts`:

```typescript
import { writable } from 'svelte/store';

export const videoTaskId = writable<string | null>(null);
export const videoMode = writable<'static_roi' | 'tracked' | 'whole_field'>('static_roi');
export const videoSummary = writable<any[]>([]);
export const frameIntervalS = writable<number | null>(null);
export const doseTable = writable<any[]>([]);
```

- [ ] **Step 3: Create the page**

Create `cellquant-v3/frontend/src/routes/video/+page.svelte`:

```svelte
<script lang="ts">
  import { onDestroy } from 'svelte';
  import { ProgressSocket } from '$lib/api/websocket';
  import { runVideo, getVideoSummary, parseDoses } from '$lib/api/client';
  import { sessionId } from '$lib/stores/session';
  import { videoTaskId, videoMode, videoSummary, frameIntervalS, doseTable } from '$lib/stores/video';

  let running = $state(false);
  let wsProgress = $state(0);
  let wsMessage = $state('');
  let wsStatus = $state('');
  let socket: ProgressSocket | null = null;

  let baselineMode = $state<'fraction' | 'stim_frame' | 'auto'>('fraction');
  let baselineFraction = $state(0.10);
  let stimFrame = $state<number | null>(null);
  let intervalS = $state<number | null>(null);
  let thresholdSigma = $state(3.0);

  function connectWebSocket() {
    socket = new ProgressSocket($sessionId!);
    socket.onMessage(handleWSMessage);
    socket.connect();
  }

  function disconnectWebSocket() {
    socket?.disconnect();
    socket = null;
  }

  function handleWSMessage(msg: any) {
    if (msg.task_id && msg.task_id !== $videoTaskId) return;
    if (msg.type === 'progress') {
      wsProgress = msg.progress ?? 0;
      wsMessage = msg.message ?? '';
      wsStatus = msg.status ?? '';
    } else if (msg.type === 'task_complete') {
      wsProgress = 100;
      wsStatus = msg.status ?? 'complete';
      running = false;
      disconnectWebSocket();
      loadSummary();
    }
  }

  async function loadSummary() {
    try {
      const data = await getVideoSummary($sessionId!);
      videoSummary.set(data.rows ?? []);
    } catch (e) {
      console.error('Failed to load video summary', e);
    }
  }

  async function loadDoses() {
    try {
      const data = await parseDoses($sessionId!);
      doseTable.set(data.doses ?? []);
    } catch (e) {
      console.error('Failed to parse doses', e);
    }
  }

  async function start() {
    if (!$sessionId) return;
    running = true;
    wsProgress = 0;
    connectWebSocket();
    try {
      const res = await runVideo({
        session_id: $sessionId,
        mode: $videoMode,
        baseline_mode: baselineMode,
        baseline_fraction: baselineFraction,
        stim_frame: stimFrame,
        interval_s: intervalS,
        event_threshold_sigma: thresholdSigma,
      });
      videoTaskId.set(res.task_id);
      frameIntervalS.set(intervalS);
    } catch (e) {
      console.error('Video run failed', e);
      running = false;
      disconnectWebSocket();
    }
  }

  onDestroy(disconnectWebSocket);
</script>

<div class="page">
  <h1>Video Analytics</h1>

  <section class="controls">
    <label>
      Mode
      <select bind:value={$videoMode}>
        <option value="static_roi">Static ROI (segment once)</option>
        <option value="tracked">Tracked (cells move)</option>
        <option value="whole_field">Whole field (no segmentation)</option>
      </select>
    </label>

    <label>
      Baseline
      <select bind:value={baselineMode}>
        <option value="fraction">First fraction of frames</option>
        <option value="stim_frame">Before stimulus frame</option>
        <option value="auto">Auto-detect onset</option>
      </select>
    </label>

    {#if baselineMode === 'fraction'}
      <label>Baseline fraction
        <input type="number" step="0.01" min="0.01" max="0.9" bind:value={baselineFraction} />
      </label>
    {:else if baselineMode === 'stim_frame'}
      <label>Stimulus frame
        <input type="number" min="1" bind:value={stimFrame} />
      </label>
    {/if}

    <label>
      Frame interval (s)
      <input type="number" step="0.01" min="0" bind:value={intervalS}
             placeholder="not set - results in frames" />
    </label>

    <label>Event threshold (sigma)
      <input type="number" step="0.5" min="1" bind:value={thresholdSigma} />
    </label>

    <button onclick={start} disabled={running || !$sessionId}>
      {running ? 'Running...' : 'Run analysis'}
    </button>
    <button onclick={loadDoses} disabled={!$sessionId}>Parse doses</button>
  </section>

  {#if !intervalS}
    <p class="hint">
      No frame interval set - time metrics will be reported in frames.
    </p>
  {/if}

  {#if running || wsProgress > 0}
    <section class="progress">
      <progress value={wsProgress} max="100"></progress>
      <span>{wsProgress.toFixed(0)}% {wsMessage} {wsStatus}</span>
    </section>
  {/if}

  {#if $doseTable.length > 0}
    <section>
      <h2>Parsed doses - confirm before fitting</h2>
      <table>
        <thead><tr><th>Source</th><th>Parsed</th><th>Molar</th><th>Volume (uL)</th><th>Note</th></tr></thead>
        <tbody>
          {#each $doseTable as d}
            <tr class:warn={!d.parsed || d.ambiguous}>
              <td>{d.source}</td>
              <td>{d.parsed ? 'yes' : 'NO'}</td>
              <td>{d.molar ?? '-'}</td>
              <td>{d.volume_ul ?? '-'}</td>
              <td>{d.note}</td>
            </tr>
          {/each}
        </tbody>
      </table>
    </section>
  {/if}

  {#if $videoSummary.length > 0}
    <section>
      <h2>Per-cell summary ({$videoSummary.length} cells)</h2>
      <table>
        <thead>
          <tr><th>Condition</th><th>Video</th><th>Cell</th>
              <th>Peak dF/F0</th><th>Peak frame</th><th>Events</th><th>SNR</th></tr>
        </thead>
        <tbody>
          {#each $videoSummary.slice(0, 200) as r}
            <tr>
              <td>{r.Condition}</td><td>{r.ImageSet}</td><td>{r.CellID}</td>
              <td>{r.peak_dF_F0?.toFixed(3)}</td><td>{r.peak_frame}</td>
              <td>{r.n_events}</td><td>{r.snr?.toFixed(1)}</td>
            </tr>
          {/each}
        </tbody>
      </table>
    </section>
  {/if}
</div>

<style>
  .page { padding: 1rem; }
  .controls { display: flex; flex-wrap: wrap; gap: 1rem; align-items: end; margin-bottom: 1rem; }
  label { display: flex; flex-direction: column; font-size: 0.85rem; gap: 0.25rem; }
  .hint { color: #a86; font-size: 0.85rem; }
  .progress { display: flex; gap: 0.5rem; align-items: center; margin: 1rem 0; }
  table { border-collapse: collapse; width: 100%; font-size: 0.85rem; }
  th, td { border: 1px solid #ddd; padding: 0.25rem 0.5rem; text-align: left; }
  tr.warn { background: #fff3cd; }
</style>
```

- [ ] **Step 4: Verify the frontend builds**

Run: `cd cellquant-v3/frontend && npm run build`
Expected: build succeeds with no TypeScript errors.

- [ ] **Step 5: Commit**

```bash
git add cellquant-v3/frontend/src/routes/video cellquant-v3/frontend/src/lib/stores/video.ts cellquant-v3/frontend/src/lib/api/client.ts
git commit -m "feat(ui): add video analytics page"
```

---

## Task 10: Frontend — SA-β-gal page

**Files:**
- Create: `cellquant-v3/frontend/src/routes/sabetagal/+page.svelte`
- Create: `cellquant-v3/frontend/src/lib/stores/sabetagal.ts`
- Modify: `cellquant-v3/frontend/src/lib/api/client.ts` (append)

**Interfaces:**
- Consumes: `POST /api/v1/sabetagal/run`, `GET /api/v1/sabetagal/summary/{sid}`.
- Produces: the `/sabetagal` route.

- [ ] **Step 1: Add the API client functions**

Append to `cellquant-v3/frontend/src/lib/api/client.ts`:

```typescript
// ---- SA-beta-gal analytics ----

export interface SABetaGalRunParams {
  session_id: string;
  condition?: string;
  white_percentile?: number;
  chromogen_index?: number;
  threshold?: number | null;
  min_area?: number;
  diameter?: number;
}

export async function runSABetaGal(
  params: SABetaGalRunParams,
): Promise<{ task_id: string }> {
  return request('/sabetagal/run', { method: 'POST', body: JSON.stringify(params) });
}

export async function getSABetaGalSummary(sessionId: string): Promise<any> {
  return request(`/sabetagal/summary/${sessionId}`);
}

export async function getSABetaGalResults(sessionId: string): Promise<any> {
  return request(`/sabetagal/results/${sessionId}`);
}
```

- [ ] **Step 2: Create the store**

Create `cellquant-v3/frontend/src/lib/stores/sabetagal.ts`:

```typescript
import { writable } from 'svelte/store';

export const sabetagalTaskId = writable<string | null>(null);
export const sabetagalSummary = writable<any[]>([]);
export const sabetagalThreshold = writable<number | null>(null);
```

- [ ] **Step 3: Create the page**

Create `cellquant-v3/frontend/src/routes/sabetagal/+page.svelte`:

```svelte
<script lang="ts">
  import { onDestroy } from 'svelte';
  import { ProgressSocket } from '$lib/api/websocket';
  import { runSABetaGal, getSABetaGalSummary } from '$lib/api/client';
  import { sessionId } from '$lib/stores/session';
  import { sabetagalTaskId, sabetagalSummary, sabetagalThreshold } from '$lib/stores/sabetagal';

  let running = $state(false);
  let wsProgress = $state(0);
  let wsMessage = $state('');
  let socket: ProgressSocket | null = null;

  let whitePercentile = $state(95.0);
  let threshold = $state<number | null>(null);
  let minArea = $state(10);
  let diameter = $state(30.0);

  function connectWebSocket() {
    socket = new ProgressSocket($sessionId!);
    socket.onMessage(handleWSMessage);
    socket.connect();
  }

  function disconnectWebSocket() {
    socket?.disconnect();
    socket = null;
  }

  function handleWSMessage(msg: any) {
    if (msg.task_id && msg.task_id !== $sabetagalTaskId) return;
    if (msg.type === 'progress') {
      wsProgress = msg.progress ?? 0;
      wsMessage = msg.message ?? '';
    } else if (msg.type === 'task_complete') {
      wsProgress = 100;
      running = false;
      disconnectWebSocket();
      loadSummary();
    }
  }

  async function loadSummary() {
    try {
      const data = await getSABetaGalSummary($sessionId!);
      sabetagalSummary.set(data.summary ?? []);
    } catch (e) {
      console.error('Failed to load SA-beta-gal summary', e);
    }
  }

  async function start() {
    if (!$sessionId) return;
    running = true;
    wsProgress = 0;
    connectWebSocket();
    try {
      const res = await runSABetaGal({
        session_id: $sessionId,
        white_percentile: whitePercentile,
        threshold,
        min_area: minArea,
        diameter,
      });
      sabetagalTaskId.set(res.task_id);
      sabetagalThreshold.set(threshold);
    } catch (e) {
      console.error('SA-beta-gal run failed', e);
      running = false;
      disconnectWebSocket();
    }
  }

  onDestroy(disconnectWebSocket);
</script>

<div class="page">
  <h1>SA-&beta;-gal Analytics</h1>

  <p class="provisional">
    Provisional: stain vectors are literature defaults, not calibrated to this
    lab's X-gal staining. Set the threshold against a negative control before
    reporting these numbers.
  </p>

  <section class="controls">
    <label>White reference percentile
      <input type="number" step="1" min="50" max="100" bind:value={whitePercentile} />
    </label>
    <label>OD threshold
      <input type="number" step="0.01" min="0" bind:value={threshold}
             placeholder="auto (Otsu)" />
    </label>
    <label>Min area (px)
      <input type="number" min="1" bind:value={minArea} />
    </label>
    <label>Cell diameter (px)
      <input type="number" step="1" min="5" bind:value={diameter} />
    </label>
    <button onclick={start} disabled={running || !$sessionId}>
      {running ? 'Running...' : 'Run analysis'}
    </button>
  </section>

  {#if running || wsProgress > 0}
    <section class="progress">
      <progress value={wsProgress} max="100"></progress>
      <span>{wsProgress.toFixed(0)}% {wsMessage}</span>
    </section>
  {/if}

  {#if $sabetagalSummary.length > 0}
    <section>
      <h2>Percent positive by condition</h2>
      <table>
        <thead>
          <tr><th>Condition</th><th>Cells</th><th>Positive</th><th>% positive</th></tr>
        </thead>
        <tbody>
          {#each $sabetagalSummary as r}
            <tr>
              <td>{r.Condition}</td>
              <td>{r.n_cells}</td>
              <td>{r.n_positive}</td>
              <td>{r.percent_positive?.toFixed(1)}%</td>
            </tr>
          {/each}
        </tbody>
      </table>
    </section>
  {/if}
</div>

<style>
  .page { padding: 1rem; }
  .provisional { background: #fff3cd; border-left: 3px solid #e0a800;
                 padding: 0.5rem 0.75rem; font-size: 0.85rem; }
  .controls { display: flex; flex-wrap: wrap; gap: 1rem; align-items: end; margin: 1rem 0; }
  label { display: flex; flex-direction: column; font-size: 0.85rem; gap: 0.25rem; }
  .progress { display: flex; gap: 0.5rem; align-items: center; margin: 1rem 0; }
  table { border-collapse: collapse; width: 100%; font-size: 0.85rem; }
  th, td { border: 1px solid #ddd; padding: 0.25rem 0.5rem; text-align: left; }
</style>
```

- [ ] **Step 4: Verify the frontend builds**

Run: `cd cellquant-v3/frontend && npm run build`
Expected: build succeeds.

- [ ] **Step 5: Commit**

```bash
git add cellquant-v3/frontend/src/routes/sabetagal cellquant-v3/frontend/src/lib/stores/sabetagal.ts cellquant-v3/frontend/src/lib/api/client.ts
git commit -m "feat(ui): add SA-beta-gal analytics page"
```

---

## Task 11: Navigation links and full regression

**Files:**
- Modify: `cellquant-v3/frontend/src/routes/+layout.svelte` (add nav links)
- Test: full suite

**Interfaces:**
- Consumes: the `/video` and `/sabetagal` routes from Tasks 9–10.
- Produces: nothing new.

- [ ] **Step 1: Add navigation links**

In `cellquant-v3/frontend/src/routes/+layout.svelte`, add links to `/video` and `/sabetagal` matching the existing nav markup exactly (inspect the file first and copy the established pattern rather than inventing one).

- [ ] **Step 2: Run the whole Python suite**

Run: `cd cellquant-v3 && python -m pytest tests/ -v`
Expected: all tests pass. Real-data tests pass on this machine, skip elsewhere.

- [ ] **Step 3: Verify the existing CTCF path is untouched**

Run: `cd cellquant-v3 && git diff main --stat -- src/cellquant/core/quantification/ctcf.py src/cellquant/core/pipeline.py`
Expected: **no output** — neither file was modified.

Then confirm `load_image` itself is unchanged:

Run: `cd cellquant-v3 && git diff main -- src/cellquant/core/io/image_loader.py | grep -E "^-" | grep -v "^---"`
Expected: **no output** — only additions, no deletions.

- [ ] **Step 4: Verify the app starts**

Run: `cd cellquant-v3 && python -c "from cellquant.api.app import create_app; a = create_app(); print(len(a.routes), 'routes')"`
Expected: prints a route count; no import errors.

- [ ] **Step 5: Build the frontend**

Run: `cd cellquant-v3/frontend && npm run build`
Expected: success.

- [ ] **Step 6: Commit**

```bash
git add cellquant-v3/frontend/src/routes/+layout.svelte
git commit -m "feat(ui): link video and SA-beta-gal pages in navigation"
```

---

## Self-Review Notes

**Spec coverage.** §4.1 loader → Task 1; §2.1 data facts → Task 2; §4.4 temporal metrics → Task 3; §8 acceptance targets → Task 4; §4.5 dose/EC50 → Task 5; §4.7 schema + §6 API → Task 6; §5.1–5.5 SA-β-gal → Task 7; §5.6 + §6 → Task 8; §7 frontend → Tasks 9–11.

**Deferred from the spec, deliberately:** §4.3 tracked mode wires `segment_image_stack` + `CellTracker` and persists `track_graphs` (blocker V7). The mode is exposed in the API and UI and falls back to static-ROI segmentation; full Trackastra wiring needs the optional `tracking` extra installed and is best validated against moving cells, which the Vasu IX83 data does not exercise. Noted as the first follow-up. §4.4 detrending is implemented but off by default and untested against real bleaching.

**Type consistency.** `TemporalConfig`, `TraceSummary`, `ParsedDose`, `HillFit`, `SABetaGalConfig` are each defined once and referenced by the same names throughout. DataFrame column names are identical between Task 6's `_build_trace_tables` and spec §4.7. `summarize_positivity` is defined in Task 7 and consumed in Task 8.

**Review Focus coverage.** (1) single-frame → `test_single_frame_stack_returns_3d`; (2) flat trace → `test_flat_trace_reports_no_events_and_nan_tau`; (3) zero-frame baseline → `test_baseline_fraction_clamps_to_at_least_one_frame`; (4) unparseable dose → `test_unparseable_dose_is_reported_not_coerced`; (5) white/black OD → `test_pure_black_does_not_produce_inf_or_nan` and `test_pure_white_does_not_produce_nan`.
