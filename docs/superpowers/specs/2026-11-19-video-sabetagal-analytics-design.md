# Video Analytics & SA-β-gal Analytics — Design Spec

**Date:** 2026-11-19
**Status:** Draft for review
**Target:** `cellquant-v3`
**Branch context:** authored while on `feat/web-folder-picker`

---

## 1. Purpose

Add two analysis modalities to CellQuant v3:

1. **Video analytics** — quantify reporter/dye time-lapse recordings, producing per-cell
   fluorescence traces and response metrics.
2. **SA-β-gal analytics** — quantify brightfield RGB X-gal colorimetric staining,
   producing a percentage of senescence-positive cells per condition.

Both are additive. Neither changes the behaviour of the existing fluorescence CTCF
pipeline.

### 1.1 Scope decision

Both modalities are specified together at the user's direction. Implementation is
**sequenced**: video first (validated against real data available today), SA-β-gal
second (calibrated when sample plates exist). See §9.

---

## 2. Source data

### 2.1 Video — verified against real files

Inspected `D:\Users\Saini Lab\Vasu IX83` on 2026-11-19:

| Property | Value |
|---|---|
| Layout | Two condition folders: `Ach/` (7 files), `Epi/` (6 files) |
| Per file | 120 frames, 512×512, `uint16`, ~61 MB |
| Series axes | `IYX` (generic index, **not** labelled `T`) |
| Acquisition software | SlideBook (Olympus IX83) |
| Channel | `GFP Imaging; 507 nm` — single channel |
| Objective | 20× Long WD, 0.45 NA, air |
| Naming | `<agonist> <dose>_XY<id>_Z0_T000_C0.tif` |

Measured signal, confirming these are genuine transients:

| File | F₀ (frames 0–9) | Peak | Peak frame | ΔF/F₀ |
|---|---|---|---|---|
| `Ach 1 uM` | 684.2 | 758.7 | 18 | **0.109** |
| `epi 100uM` | 678.7 | 713.1 | 35 | **0.051** |

**Critical finding — frame interval is not recoverable.** TIFF tag `65002` holds a
constant `3443.909912109375` across frames 0, 1, 2, 3, 50 and 119, so it is a static
field (likely Z position), not a per-frame timestamp. There is no ImageJ, OME or
MicroManager metadata block. Frame interval **must** be supplied by the user.

**Dose series.** Filenames encode a dose-response ladder (`100nm`, `1 uM`, `10 uM`,
`100 uM`, `1mM- 5ul`, `1mM- 50ul`, `10mM` for Ach), with inconsistent spacing and
casing.

### 2.2 SA-β-gal — not yet available

No sample images exist on disk (searched `D:\Users\Saini Lab`). Per the user:
**brightfield RGB with blue X-gal precipitate**. All SA-β-gal parameters in this spec
are literature defaults and are explicitly marked for calibration (§9.2).

---

## 3. Current-state constraints

Established by reading the codebase. Each is a blocker that the design must address.

### 3.1 Video blockers

| # | Blocker | Location |
|---|---|---|
| V1 | `load_image` collapses any 3-D array to its **middle slice** — a 120-frame video returns frame 60 only | `core/io/image_loader.py:87` |
| V2 | `load_multichannel_tiff` raises `ValueError` for `ndim > 3` | `core/io/image_loader.py:170` |
| V3 | `_prepare_image` raises `ValueError` for `ndim > 3`, truncates to 3 channels | `core/segmentation/cellpose_engine.py:293-297` |
| V4 | `Session.masks` is `{condition: {base_name: 2-D array}}` — no T axis | `sessions/models.py:26` |
| V5 | Results schema has no `Frame` / `Time` / `TrackID` column | `core/quantification/ctcf.py:350-356` |
| V6 | `run_tracking_task` fakes the T axis by alphabetically sorting `base_name`s | `tasks/worker.py:862` |
| V7 | `track_graphs` is declared but **never written** — lineage is discarded after CTC export | `sessions/models.py:28-30` |
| V8 | `channel_detect` has no time-axis concept; a `_t0/_t1` token is suppressed as channel-like and each frame becomes its own image set | `core/io/channel_detect.py:116`, `:197-225` |

**Evidence V1 is already causing silent data loss:** the existing
`Vasu IX83_output/session.json` records Ach and Epi as 5 images each with rendered
thumbnails — each 120-frame stack was ingested as a single flat image. 119 of 120
frames were discarded without warning.

### 3.2 SA-β-gal blockers

| # | Blocker | Location |
|---|---|---|
| S1 | `load_image` destroys RGB — the `shape[-1] <= 5` branch keeps only channel 0 | `core/io/image_loader.py:83-84` |
| S2 | **`calculate_ctcf_vectorized` clamps at zero** (`np.maximum(0.0, ctcf_raw)`). Absorptive signal sits *below* background, so every measurement would floor to 0 | `core/quantification/ctcf.py:185` |
| S3 | All seven background methods assume fluorescence semantics (dark background, bright signal). Brightfield inverts this | `core/quantification/background.py:26-35` |
| S4 | `create_composite` defaults to fluorescence false-colours; nothing handles true-colour RGB | `core/io/image_loader.py:235-283` |

S2 is the decisive one: CTCF cannot be reused for a colorimetric assay under any
configuration.

### 3.3 Existing assets that reduce the work

| Asset | Location | Relevance |
|---|---|---|
| `segment_image_stack(engine, stack, params, cb) -> (T,Y,X)` | `cellpose_engine.py:421-459` | Written, **currently unused by the API**. Direct entry point for video segmentation. |
| `CellTracker.track(images (T,Y,X), masks (T,Y,X), mode)` | `tracking/tracker.py:31-49` | Trackastra wrapper; frame-to-frame linking already solved. |
| `load_image_memmap` | `image_loader.py:96-140` | Preserves full ND shape via `tifffile.memmap`. Natural base for stack loading. |
| `TimelapsePlayer`, `TrackOverlay`, `LineageTree`, `TrackingForm` | `frontend/src/lib/components/tracking/` | Play/pause/frame-step UI already built. |
| `run_nellie_task` | `tasks/worker.py:643-817` | Template for a bolt-on modality: standalone per-cell DataFrame merged into `results_df` on `["Condition","ImageSet","CellID"]`, with standalone-primary fallback. |
| `labeled_comprehension` per-cell machinery | `ctcf.py:150-167` | Reusable for any per-label statistic, including OD. |
| `scikit-image` ≥ 0.22 | `pyproject.toml:26` | Ships `skimage.color.rgb2hed` — colour deconvolution needs **no new dependency**. |
| `livecell` Cellpose model | `cellpose_engine.py:73-80` | Suited to time-lapse segmentation. |

**Architectural note.** `BatchPipeline` (`core/pipeline.py`) is vestigial for the web
app — only tests import it. Live orchestration is `tasks/worker.py`. All new work
targets `worker.py`; `pipeline.py` is left untouched.

---

## 4. Design — Video analytics

### 4.1 Loading

New function in `core/io/image_loader.py`, leaving `load_image` unchanged so no
existing behaviour shifts:

```python
@dataclass
class TimelapseMeta:
    path: Path
    n_frames: int
    shape: Tuple[int, int]        # (Y, X) per frame
    dtype: str
    channel_name: Optional[str]   # from TIFF tag 65004
    objective: Optional[str]      # from TIFF tag 65003
    interval_s: Optional[float]   # user-supplied; never inferred
    size_bytes: int

def load_timelapse(path: Path, *, channel: Optional[int] = None,
                   memmap: bool = True) -> Tuple[np.ndarray, TimelapseMeta]:
    """Load a multi-frame TIFF as (T, Y, X). Never collapses the T axis."""
```

Built on `tifffile`, defaulting to memmap so a 61 MB stack is not fully resident until
sliced.

**Z-stack vs time-series disambiguation.** Shape alone cannot distinguish them. Resolution
order:
1. Explicit user choice of acquisition mode at experiment setup (authoritative).
2. Filename `_T` token present → time-series.
3. Frame count > 10 with no `Z` token → time-series.
4. Otherwise Z-stack, preserving today's behaviour.

The chosen interpretation is surfaced in the UI so a wrong guess is visible, never silent.

### 4.2 Memory policy

A 120×512×512 `uint16` stack is 63 MB; 13 files ≈ 800 MB if co-resident. **One video is
processed at a time**, and only extracted traces (a few KB each) are retained. Peak
additional footprint stays ~2 frames + one mask stack. This fits the existing container
without a memory-limit change.

### 4.3 Segmentation — two modes

Motion varies by experiment, so this is an explicit per-run choice, not an inference.

**Static-ROI mode (default).**
Segment once on a mean or max projection, apply those labels to all frames.
Fast, robust, no tracking. Correct for adherent cells. Label identity is trivially stable.

**Tracked mode.**
Segment every frame via `segment_image_stack` (already written), feed images + masks to
`CellTracker.track`, use tracked labels for traces. Handles drift and migration at higher
compute cost.

Blocker V7 must be fixed for tracked mode: `run_tracking_task` currently discards the
Trackastra graph. The graph is persisted to `session.track_graphs[condition]` so lineage
survives and traces can be keyed by `TrackID`.

### 4.4 Temporal quantification

New module `core/quantification/temporal.py`, independent of `ctcf.py` and unit-testable
in isolation.

```python
@dataclass
class TemporalConfig:
    baseline_mode: Literal["fraction", "stim_frame", "auto"] = "fraction"
    baseline_fraction: float = 0.10      # first 10% of frames
    stim_frame: Optional[int] = None     # used when baseline_mode == "stim_frame"
    interval_s: Optional[float] = None   # per experiment (§4.6)
    detrend: bool = False                # photobleach correction
    event_threshold_sigma: float = 3.0
    event_refractory_frames: int = 5
    decay_fit: bool = True
```

**Baseline (F₀).** Per the user's decision, all three modes ship, defaulting to
`fraction` at 0.10:
- `fraction` — median of the first `baseline_fraction` of frames. For 120 frames that is
  frames 0–11; the Ach data is flat until ~frame 15, so this is sound there.
- `stim_frame` — user gives the stimulus frame; F₀ = median of all frames before it.
- `auto` — detect onset as the first sustained departure from the baseline distribution.

Median, not mean, for robustness to a stray bright frame.

**Per-frame trace extraction.** For each cell label and frame: background-subtracted mean
intensity. Background uses the existing `estimate_background()` per frame (fluorescence
semantics hold here, unlike SA-β-gal).

**Per-cell metrics.**

| Metric | Definition |
|---|---|
| `peak_dF_F0` | max((F − F₀)/F₀) |
| `time_to_peak` | frames (and seconds when interval known) from stimulus to peak |
| `auc` | trapezoidal integral of ΔF/F₀ over the response window |
| `rise_time_10_90` | 10%→90% of peak |
| `decay_tau` | single-exponential fit to the post-peak decay; `NaN` when the fit fails |
| `snr` | peak ΔF/F₀ ÷ baseline standard deviation |
| `baseline_f0` | the F₀ used |

**Event detection.** Threshold crossing at `event_threshold_sigma` × baseline σ, with a
refractory period of `event_refractory_frames`. Yields per-cell event count, frequency
(Hz when interval known, else per-frame), and amplitude distribution.

**Whole-field mode.** Identical metrics with a single ROI covering the frame. Skips
segmentation entirely — fastest path, no per-cell heterogeneity.

**Photobleach detrending.** Off by default. When on, fit a monoexponential to the
baseline-only segment and divide it out. Kept optional because it can distort slow
genuine responses.

### 4.5 Dose-response

Aggregate peak ΔF/F₀ per condition, fit a 4-parameter Hill equation → EC50 with CI.

**Dose parsing is a correctness risk and is handled explicitly.** Filenames are
inconsistent (`100nm`, `1 uM`, `1mM- 5ul`). A regex extracts value + unit and normalises
to molar. The parsed table is **presented for confirmation before fitting** — never
silently assumed. `1mM- 5ul` vs `1mM- 50ul` share a concentration but differ in volume;
these are flagged as ambiguous for the user to resolve rather than being merged or
guessed.

Fitting uses `scipy.optimize.curve_fit` (SciPy is an existing dependency). Fewer than 4
dose points, or a failed fit, reports "insufficient data for EC50" rather than an
unreliable number.

### 4.6 Frame interval

Per the user's decision: **entered once per experiment** at setup, applied to all videos
in the run, stored on the session. When absent, all outputs are reported in frame units
and time-based metrics (`decay_tau` in seconds, frequency in Hz) are withheld rather than
fabricated.

### 4.7 Output schema

Long-format traces table, written beside the existing results rather than replacing them:

```
Condition, ImageSet, CellID, TrackID, Frame, Time_s, F_raw, F_bg_corrected, dF_F0
```

Per-cell summary table:

```
Condition, ImageSet, CellID, TrackID, baseline_f0, peak_dF_F0, peak_frame,
time_to_peak_s, auc, rise_time_10_90, decay_tau_s, snr, n_events, event_freq_hz
```

New `Session` fields: `traces_df`, `trace_summary_df`, persisted as Parquet alongside
`results_df` (`get_results_path()` pattern). The summary table merges into `results_df`
on `["Condition","ImageSet","CellID"]` following the Nellie precedent, so existing
results pages keep working.

**Schema addition.** `results_to_dataframe` gains optional `frame` and `track_id`
parameters, defaulting to `None` so every existing caller is unaffected.

---

## 5. Design — SA-β-gal analytics

> All numeric defaults here are literature values, not calibrated to Saini Lab plates.
> See §9.2.

### 5.1 Loading

```python
def load_rgb(path: Path) -> np.ndarray:
    """Load a brightfield image as true-colour (Y, X, 3). Never collapses to 2-D."""
```

Addresses S1. Separate from `load_image`, which stays unchanged.

### 5.2 Colour deconvolution

`skimage.color.rgb2hed` (Ruifrok–Johnston) separates stain contributions. No new
dependency.

**Honest limitation:** the stock HED vectors are fitted for haematoxylin–DAB, not X-gal
blue. Stain vectors are therefore **configurable**, with the HED default as a starting
point only. Calibration against real negative/positive controls is required before the
output is publishable.

### 5.3 Optical density metric

Replaces CTCF entirely (S2):

```
OD = −log10(clip(I / I_white, eps, 1.0))
```

`I_white` is the white reference — the illumination level of unstained background,
estimated from a high percentile of the image, or from a user-supplied blank field when
available. This addresses S3: rather than reusing any of the seven fluorescence
background methods, SA-β-gal gets its own white-reference normalisation. Illumination
gradients are handled by fitting a smooth surface to the background, reusing the
`polynomial_surface` machinery's approach but against a bright, not dark, reference.

Per-cell aggregation reuses the existing `labeled_comprehension` pattern from
`ctcf.py:150-167`.

### 5.4 Segmentation

Cellpose `cyto`/`cyto3` on the inverted grayscale, or on a fluorescent nuclear
counterstain when present. `channel_detect.KNOWN_CHANNEL_NAMES` already recognises
`bf`/`brightfield`/`dic`/`phase`, so detection needs no change.

### 5.5 Positivity classification

A cell is senescence-positive when its mean chromogen OD exceeds a threshold.

**Threshold is the scientifically contentious parameter and is treated as such.** Default
is Otsu on the per-cell OD distribution, exposed with a live preview and histogram so it
can be set against a negative control. A hard-coded cutoff would make results
unpublishable, so no fixed default is baked in.

### 5.6 Output schema

```
Condition, ImageSet, CellID, Area, mean_OD, integrated_OD, is_positive
```

Plus a per-condition summary: `n_cells`, `n_positive`, `percent_positive` — the standard
senescence readout. Stored in `Session.sabetagal_df`, merged into `results_df` on the
Nellie keys.

---

## 6. API design

Following the uniform existing pattern exactly (`tracking.py:12-34`,
`quantification.py:22-50`).

**New routers,** registered in `api/app.py:142-155`:

| Router | Prefix | Endpoints |
|---|---|---|
| `video.py` | `/video` | `POST /run`, `POST /cancel`, `GET /traces/{sid}/{cond}`, `GET /summary/{sid}`, `GET /dose-response/{sid}`, `GET /parse-doses/{sid}` |
| `sabetagal.py` | `/sabetagal` | `POST /run`, `POST /cancel`, `GET /results/{sid}`, `GET /summary/{sid}`, `GET /threshold-preview/{sid}` |

**New worker tasks** in `tasks/worker.py`: `run_video_task`, `run_sabetagal_task`, both
`(task, queue, session, params) -> dict`, modelled on `run_nellie_task`. Progress via
`queue.update_progress`; cancellation by polling `task.status == "cancelled"` as existing
workers do. `print()` inside a worker streams to the UI through `LogCapture`.

**New schemas:** `api/schemas/video.py`, `api/schemas/sabetagal.py`.

---

## 7. Frontend design

Two new SvelteKit pages following the canonical page↔backend pattern
(`routes/tracking/+page.svelte:28-84`): connect `ProgressSocket` **before** submitting,
filter messages by `task_id`, branch on `progress` vs `task_complete`, disconnect and
re-fetch over REST on completion.

**`routes/video/`** — reuses `TimelapsePlayer`, `TrackOverlay`, `LineageTree` from
`components/tracking/`. Adds:
- `TraceChart.svelte` — ΔF/F₀ vs time, overlaying selected cells
- `DoseResponseChart.svelte` — Hill fit with EC50 annotation
- `DoseConfirmTable.svelte` — the parsed-dose confirmation gate (§4.5)
- Run form: mode (static/tracked/whole-field), baseline mode + parameters, interval,
  event-detection settings

**`routes/sabetagal/`** — adds:
- `ThresholdPicker.svelte` — OD histogram with a draggable threshold and live
  positive/negative overlay
- `StainVectorConfig.svelte` — editable deconvolution vectors
- Per-condition `% positive` bar chart

Client functions added to `lib/api/client.ts`; stores `lib/stores/video.ts`,
`lib/stores/sabetagal.ts`, matching existing store conventions.

---

## 8. Testing

**Unit — video.** Synthetic traces with known ground truth: a square pulse, a decaying
exponential of known τ, a flat trace (must yield zero events), a noisy trace at defined
SNR. Assert recovered `peak_dF_F0`, `time_to_peak`, `decay_tau`, event counts.
Dose parsing gets its own table-driven test including the awkward real cases
(`100nm`, `1 uM`, `1mM- 5ul`, `1mM- 50ul`).

**Unit — SA-β-gal.** Synthetic RGB with known blue fractions; assert monotonic OD
response and correct positive/negative classification at a given threshold.

**Integration — video.** The real `Vasu IX83` data is the acceptance test. Known-good
expectations established in §2.1:

- `Ach 1 uM` → peak ΔF/F₀ ≈ **0.109** at frame **18**
- `epi 100uM` → peak ΔF/F₀ ≈ **0.051** at frame **35**

Whole-field mode must reproduce these within tolerance. This is the single strongest
validation available and gates the video work.

**Regression.** The existing fluorescence CTCF path must produce byte-identical results
before and after. `load_image` is untouched; new loaders are additive.

**Integration — SA-β-gal.** Deferred until sample plates exist (§9.2).

---

## 9. Implementation sequencing

### 9.1 Phase 1 — Video (validated)

1. `load_timelapse` + `TimelapseMeta` (fixes V1, V2)
2. Acquisition-mode plumbing + interval entry (§4.1, §4.6)
3. `core/quantification/temporal.py` + unit tests
4. Static-ROI segmentation path
5. `run_video_task` + `/video` router + schemas
6. Tracked mode: wire `segment_image_stack` + `CellTracker`, persist `track_graphs`
   (fixes V3, V6, V7)
7. Dose parsing + confirmation gate + Hill fit
8. Frontend `routes/video/`
9. **Validate against `Vasu IX83` expectations (§8)**

### 9.2 Phase 2 — SA-β-gal (calibrate on arrival)

10. `load_rgb` (fixes S1)
11. `core/quantification/sabetagal.py` — deconvolution, OD, classification (fixes S2, S3)
12. `run_sabetagal_task` + `/sabetagal` router + schemas
13. Frontend `routes/sabetagal/` with threshold preview
14. **Calibration gate:** on first real plates, fit stain vectors to actual X-gal blue,
    set the threshold against a negative control, and verify `% positive` against manual
    scoring. Until this gate passes, SA-β-gal output is marked provisional in the UI.

---

## 10. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| SA-β-gal stain vectors uncalibrated | Wrong positivity calls | Configurable vectors; explicit calibration gate (§9.2); provisional labelling in UI |
| Dose parsing mis-assigns a concentration | Invalid EC50 | Mandatory confirmation table before fitting; ambiguous cases flagged, never guessed |
| Frame interval entered wrong | Time metrics wrong by a scale factor | Interval shown alongside every time-based result; frame units always available |
| Z-stack misread as time-series | Nonsense traces | Explicit acquisition mode; detected interpretation surfaced in UI |
| Memory pressure on large videos | OOM in container | One video at a time; memmap; only traces retained (§4.2) |
| Tracked mode fails on dense/dividing cells | Broken traces | Static-ROI is the default; tracking is opt-in |
| Decay fit fails on noisy traces | `NaN` τ | Report `NaN` explicitly rather than a fabricated fit |

---

## 11. Out of scope

- 3-D / Z-stack time-lapse (4-D). Only `(T, Y, X)` is supported.
- Multi-channel ratiometric video (e.g. Fura-2 340/380). Single channel only, matching
  the available data.
- Real-time analysis during acquisition.
- Changes to the existing fluorescence CTCF pipeline beyond additive, optional
  parameters.
- Retro-fitting `BatchPipeline` (`core/pipeline.py`), which is vestigial for the web app.

---

## 12. Open questions

1. **SA-β-gal sample images** — needed to close §9.2. None currently on disk.
2. **`1mM- 5ul` vs `1mM- 50ul`** — same concentration, different volume. Are these
   distinct experimental conditions, or replicates? Affects dose-response grouping.
3. **Stimulus timing** — is agonist added at a consistent frame across the Vasu IX83
   series? The two measured files peak at frames 18 and 35, which may reflect genuine
   kinetic differences or different addition times.
