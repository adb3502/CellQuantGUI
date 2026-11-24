<script lang="ts">
	import { onDestroy, onMount, untrack } from 'svelte';
	import { Play, Pause, ChevronLeft, ChevronRight, Film, Activity, FlaskConical } from 'lucide-svelte';
	import { ProgressSocket } from '$lib/api/websocket';
	import {
		runVideo, getVideoSummary, parseDoses, getSegmentationStatus,
		getVideoTraceMatrix, getVideoInfo, videoOverlayUrl, videoFrameUrl, getActivityRange,
		type VideoTraceMatrix, type VideoHeat
	} from '$lib/api/client';
	import { sessionId } from '$lib/stores/session';
	import { conditions } from '$lib/stores/experiment';
	import { videoTaskId, videoMode, videoSummary, frameIntervalS, doseTable, videoFocus } from '$lib/stores/video';

	// ── Parameters ─────────────────────────────────────────────
	let projection = $state<'max' | 'percentile' | 'std' | 'mean'>('max');
	let projectionPercentile = $state(99);
	let diameter = $state(30);
	let baselineMode = $state<'fraction' | 'stim_frame' | 'auto'>('fraction');
	let baselineFraction = $state(0.1);
	let stimFrame = $state<number | null>(null);
	let intervalS = $state<number | null>($frameIntervalS);
	let thresholdSigma = $state(3.0);

	const projectionHelp: Record<string, string> = {
		max: 'Brightest value of each pixel over all frames. Finds any cell that ever lit up.',
		percentile: 'Like max, but ignores single-frame hot pixels and cosmic rays.',
		std: 'Pixels that changed over time. Drops cells that never respond, which biases a dose series.',
		mean: 'Time average. Lowest noise, but dims cells that respond only briefly.'
	};

	// ── Run state ──────────────────────────────────────────────
	let running = $state(false);
	let wsProgress = $state(0);
	let wsMessage = $state('');
	let wsStatus = $state('');
	let errorText = $state('');
	let socket: ProgressSocket | null = null;
	let pollTimer: ReturnType<typeof setInterval> | null = null;
	// Changes after every run so the browser refetches overlays and frames.
	let runStamp = $state(String(Date.now()));

	// ── Viewer state ───────────────────────────────────────────
	type VideoRef = { condition: string; baseName: string };
	let videos = $derived<VideoRef[]>(
		$conditions.flatMap((c) => c.image_sets.map((s) => ({ condition: c.name, baseName: s.base_name })))
	);
	let conditionNames = $derived([...new Set(videos.map((v) => v.condition))]);
	let activeCondition = $state('');
	let filtered = $derived(videos.filter((v) => v.condition === activeCondition));
	let videoIndex = $state(0);
	let current = $derived<VideoRef | null>(filtered[videoIndex] ?? null);

	// The video is the main view; the projection with its ROIs is secondary.
	let viewMode = $state<'rois' | 'frames'>('frames');
	let overlayStyle = $state<'outline' | 'filled'>('outline');
	let showRoisOnFrames = $state(true);
	// Pseudocolour on the video: each cell (or pixel) tinted by dF/F0.
	let heat = $state<VideoHeat>('none');
	let heatRange = $state<{ lo: number; hi: number } | null>(null);

	let nFrames = $state(0);
	let hasRois = $state(false);
	let nCells = $state(0);
	let frame = $state(0);
	let playing = $state(false);

	// Playback speed. 1x is real time when the frame interval is known,
	// otherwise 10 frames per second.
	const SPEEDS = [0.25, 0.5, 1, 2, 4];
	let speed = $state(1);
	let baseFps = $derived(intervalS && intervalS > 0 ? Math.min(30, 1 / intervalS) : 10);
	let fps = $derived(Math.min(60, baseFps * speed));

	// Every frame is fetched once into memory so playback never waits on the
	// server. Keyed by frame index; a new video or overlay setting refills it.
	let frameCache = $state<Record<number, string>>({});
	let buffered = $state(0);
	let frameError = $state('');
	let loadToken = 0;

	let matrix = $state<VideoTraceMatrix | null>(null);
	let chartEl: HTMLDivElement | undefined = $state();
	let plotly: any = null;

	// Open on the recording the user clicked on the Results page, if any.
	$effect(() => {
		const f = $videoFocus;
		if (!f || !videos.length) return;
		const inCond = videos.filter((v) => v.condition === f.condition);
		const idx = inCond.findIndex((v) => v.baseName === f.baseName);
		if (idx >= 0) {
			activeCondition = f.condition;
			videoIndex = idx;
			viewMode = 'frames';
			heat = 'cell';
		}
		videoFocus.set(null);
	});

	$effect(() => {
		if (!activeCondition && conditionNames.length) activeCondition = conditionNames[0];
	});
	$effect(() => {
		if (videoIndex >= filtered.length && filtered.length > 0) videoIndex = filtered.length - 1;
	});

	let imgSrc = $derived.by(() => {
		if (!current || !$sessionId) return null;
		if (viewMode === 'rois') {
			return hasRois
				? videoOverlayUrl($sessionId, current.condition, current.baseName, overlayStyle, runStamp)
				: null;
		}
		return frameCache[frame] ?? frameUrl(frame);
	});

	function frameUrl(f: number): string {
		if (!current || !$sessionId) return '';
		return videoFrameUrl(
			$sessionId, current.condition, current.baseName, f,
			showRoisOnFrames && hasRois, runStamp, effectiveHeat
		);
	}

	// A heat mode needs ROIs (and, per cell, traces); fall back to plain.
	let effectiveHeat = $derived<VideoHeat>(
		!hasRois ? 'none' : heat === 'cell' && !matrix ? 'none' : heat
	);

	function clearFrameCache() {
		for (const u of Object.values(frameCache)) URL.revokeObjectURL(u);
		frameCache = {};
		buffered = 0;
	}

	// Refill the frame cache whenever what a frame looks like changes.
	$effect(() => {
		const v = current;
		const sid = $sessionId;
		const n = nFrames;
		void showRoisOnFrames; void hasRois; void effectiveHeat; void runStamp;
		if (!v || !sid || n < 1 || viewMode !== 'frames') return;
		// untrack: preloading reads `frame` and writes `frameCache`; tracked,
		// either would re-run this effect and restart the load endlessly.
		untrack(() => preloadFrames(n));
	});

	let cacheKey = '';

	async function preloadFrames(n: number) {
		const key = `${current?.condition}|${current?.baseName}|${showRoisOnFrames && hasRois}|${effectiveHeat}|${runStamp}|${n}`;
		if (key === cacheKey && !frameError && buffered > 0) return;
		cacheKey = key;
		const token = ++loadToken;
		clearFrameCache();
		frameError = '';
		// Start from the frame on screen so it and its neighbours come first.
		const start = frame;
		const order = Array.from({ length: n }, (_, i) => (start + i) % n);
		const urls = order.map((f) => [f, frameUrl(f)] as const);
		let next = 0;
		async function worker() {
			while (next < urls.length) {
				const [f, url] = urls[next++];
				try {
					const res = await fetch(url);
					if (!res.ok) throw new Error(await res.text());
					const blobUrl = URL.createObjectURL(await res.blob());
					if (token !== loadToken) { URL.revokeObjectURL(blobUrl); return; }
					frameCache[f] = blobUrl;
					buffered += 1;
				} catch (e: any) {
					if (token === loadToken) frameError = `Frame ${f} failed to load: ${e?.message ?? e}`;
					return;
				}
			}
		}
		await Promise.all([worker(), worker(), worker(), worker()]);
	}

	// The colour bar's numbers: one fixed scale for the whole recording.
	$effect(() => {
		const v = current;
		const sid = $sessionId;
		const h = effectiveHeat;
		heatRange = null;
		if (!v || !sid || h === 'none') return;
		getActivityRange(sid, v.condition, v.baseName, h)
			.then((r) => { if (effectiveHeat === h) heatRange = { lo: r.lo, hi: r.hi }; })
			.catch(() => { heatRange = null; });
	});

	// Advance at the chosen speed, but never past a frame that has not
	// arrived yet: playback holds instead of stuttering through blanks.
	$effect(() => {
		if (!playing || viewMode !== 'frames' || nFrames < 2) return;
		const id = setInterval(() => {
			const nxt = frame + 1 >= nFrames ? 0 : frame + 1;
			if (frameCache[nxt]) frame = nxt;
		}, 1000 / fps);
		return () => clearInterval(id);
	});

	// Load frame count, ROI status and traces whenever the video changes.
	$effect(() => {
		const v = current;
		const sid = $sessionId;
		void runStamp;
		if (!v || !sid) return;
		playing = false;
		loadVideo(sid, v);
	});

	async function loadVideo(sid: string, v: VideoRef) {
		try {
			const info = await getVideoInfo(sid, v.condition, v.baseName);
			nFrames = info.n_frames;
			hasRois = info.has_rois;
			nCells = info.n_cells;
			if (frame >= nFrames) frame = 0;
			if (!hasRois && viewMode === 'rois') viewMode = 'frames';
		} catch {
			nFrames = 0;
			hasRois = false;
			nCells = 0;
		}
		try {
			matrix = await getVideoTraceMatrix(sid, v.condition, v.baseName);
		} catch {
			matrix = null;
		}
	}

	// ── Playback ───────────────────────────────────────────────
	function togglePlay() {
		if (viewMode !== 'frames') viewMode = 'frames';
		playing = !playing;
	}

	function step(delta: number) {
		playing = false;
		if (nFrames > 0) frame = (frame + delta + nFrames) % nFrames;
	}

	function onKey(e: KeyboardEvent) {
		const t = e.target as HTMLElement | null;
		if (t && (t.tagName === 'INPUT' || t.tagName === 'SELECT' || t.tagName === 'TEXTAREA')) return;
		if (viewMode !== 'frames' || nFrames < 2) return;
		if (e.key === ' ') { e.preventDefault(); togglePlay(); }
		else if (e.key === 'ArrowRight') { e.preventDefault(); step(1); }
		else if (e.key === 'ArrowLeft') { e.preventDefault(); step(-1); }
	}

	// ── Colour map (same as the backend overlay) ───────────────
	// Turbo, started at 0.1: its bottom end is near-black and would make
	// "no change" look like "no cell". Must match timelapse_view.TURBO_START.
	const TURBO_START = 0.1;
	function turbo(x: number): [number, number, number] {
		x = Math.min(1, Math.max(0, x));
		const r = 0.13572138 + x * (4.6153926 + x * (-42.66032258 + x * (132.13108234 + x * (-152.94239396 + x * 59.28637943))));
		const g = 0.09140261 + x * (2.19418839 + x * (4.84296658 + x * (-14.18503333 + x * (4.27729857 + x * 2.82956604))));
		const b = 0.1066733 + x * (12.64194608 + x * (-60.58204836 + x * (110.36276771 + x * (-89.90310912 + x * 27.34824973))));
		const c = (v: number) => Math.round(Math.min(1, Math.max(0, v)) * 255);
		return [c(r), c(g), c(b)];
	}
	const turboScale: [number, string][] = Array.from({ length: 11 }, (_, i) => {
		const [r, g, b] = turbo(TURBO_START + (1 - TURBO_START) * (i / 10));
		return [i / 10, `rgb(${r},${g},${b})`];
	});
	const turboGradient = `linear-gradient(to right, ${turboScale.map(([p, c]) => `${c} ${p * 100}%`).join(', ')})`;

	function prevVideo() {
		if (videoIndex > 0) videoIndex -= 1;
	}
	function nextVideo() {
		if (videoIndex < filtered.length - 1) videoIndex += 1;
	}

	// ── Trace chart ────────────────────────────────────────────
	function cssVar(name: string, fallback: string) {
		if (typeof document === 'undefined') return fallback;
		return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;
	}

	function hasTime(m: VideoTraceMatrix) {
		return !!m.time_s && m.time_s.every((t) => t !== null);
	}

	function xValues(m: VideoTraceMatrix): number[] {
		return hasTime(m) ? (m.time_s as number[]) : m.frames;
	}

	function frameMarker(m: VideoTraceMatrix) {
		const xs = xValues(m);
		const x = xs[Math.min(frame, xs.length - 1)] ?? 0;
		return [{
			type: 'line', x0: x, x1: x, yref: 'paper', y0: 0, y1: 1,
			line: { color: cssVar('--accent', '#6B5B95'), width: 1.5, dash: 'dot' }
		}];
	}

	// Which way to look at the cells-by-time matrix.
	let chartView = $state<'lines' | 'heatmap' | 'surface'>('lines');
	// Row order for heatmap and surface. Sorting by peak time is what shows
	// whether a response travels through the population or hits all at once.
	let heatSort = $state<'peak_time' | 'peak' | 'cell'>('peak_time');
	// A cell picked from the table, drawn on top of the others.
	let selectedCell = $state<number | null>(null);

	function finite(v: number | null): v is number {
		return v !== null && Number.isFinite(v);
	}

	function cellOrder(m: VideoTraceMatrix) {
		const stats = m.cells.map((c) => {
			let peak = -Infinity;
			let at = Infinity;
			c.dff.forEach((v, i) => {
				if (finite(v) && v > peak) { peak = v; at = i; }
			});
			return { c, peak, at };
		});
		if (heatSort === 'peak_time') stats.sort((a, b) => a.at - b.at || b.peak - a.peak);
		else if (heatSort === 'peak') stats.sort((a, b) => b.peak - a.peak);
		else stats.sort((a, b) => a.c.cell_id - b.c.cell_id);
		return stats.map((s) => s.c);
	}

	// Colour range clipped to the 2nd-98th percentile: a handful of very bright
	// cells would otherwise compress everyone else into one colour.
	function colourLimits(m: VideoTraceMatrix): [number, number] {
		const all = m.cells.flatMap((c) => c.dff.filter(finite)).sort((a, b) => a - b);
		if (!all.length) return [0, 1];
		const q = (p: number) => all[Math.min(all.length - 1, Math.max(0, Math.round(p * (all.length - 1))))];
		const lo = Math.min(0, q(0.02));
		const hi = Math.max(q(0.98), lo + 1e-6);
		return [lo, hi];
	}

	async function drawChart() {
		if (!chartEl || !matrix) return;
		if (!plotly) plotly = await import('plotly.js-dist-min');
		const m = matrix;
		const xs = xValues(m);
		const muted = cssVar('--text-faint', '#A8A29E');
		const text = cssVar('--text', '#333');
		const accent = cssVar('--accent', '#6B5B95');
		const xTitle = hasTime(m) ? 'Time (s)' : 'Frame';
		const base = {
			paper_bgcolor: 'transparent',
			plot_bgcolor: 'transparent',
			font: { color: text, size: 11 },
			margin: { t: 10, r: 10, b: 45, l: 55 }
		};
		const config = { displaylogo: false, responsive: true };
		const el = chartEl as any;

		if (chartView === 'lines') {
			// All cells in ONE trace separated by nulls: hundreds of separate
			// traces make Plotly sluggish, one trace with gaps draws instantly.
			const cx: (number | null)[] = [];
			const cy: (number | null)[] = [];
			for (const c of m.cells) {
				cx.push(...xs, null);
				cy.push(...c.dff, null);
			}
			const traces: any[] = [
				{
					x: cx, y: cy, type: 'scattergl', mode: 'lines', name: `cells (${m.cells.length})`,
					line: { color: muted, width: 0.6 }, opacity: 0.45, hoverinfo: 'skip', connectgaps: false
				},
				{
					x: xs, y: m.mean, type: 'scatter', mode: 'lines', name: 'mean',
					line: { color: accent, width: 2.5 }
				}
			];
			const sel = m.cells.find((c) => c.cell_id === selectedCell);
			if (sel) {
				traces.push({
					x: xs, y: sel.dff, type: 'scatter', mode: 'lines', name: `cell ${sel.cell_id}`,
					line: { color: '#e0563e', width: 2 }
				});
			}
			await plotly.react(chartEl, traces, {
				...base,
				xaxis: { title: xTitle, color: text, zeroline: false },
				yaxis: { title: 'ΔF/F₀', color: text, zeroline: true, zerolinecolor: muted },
				shapes: frameMarker(m),
				legend: { orientation: 'h', y: 1.08, x: 0 },
				hovermode: 'x'
			}, config);
		} else {
			const ordered = cellOrder(m);
			const z = ordered.map((c) => c.dff);
			const rows = ordered.map((_, i) => i + 1);
			const [zmin, zmax] = colourLimits(m);
			const hover = ordered.map((c) => xs.map(() => `cell ${c.cell_id}`));
			const sortLabel = heatSort === 'peak_time' ? 'cells, by time of peak'
				: heatSort === 'peak' ? 'cells, by peak ΔF/F₀' : 'cells, by ID';

			if (chartView === 'heatmap') {
				await plotly.react(chartEl, [{
					x: xs, y: rows, z, type: 'heatmap', colorscale: turboScale, zmin, zmax,
					text: hover, hovertemplate: '%{text}<br>' + xTitle + ' %{x}<br>ΔF/F₀ %{z:.3f}<extra></extra>',
					colorbar: { title: 'ΔF/F₀', thickness: 12 }
				}], {
					...base,
					xaxis: { title: xTitle, color: text },
					yaxis: { title: sortLabel, color: text, autorange: 'reversed', showticklabels: false },
					shapes: frameMarker(m)
				}, config);
			} else {
				await plotly.react(chartEl, [{
					x: xs, y: rows, z, type: 'surface', colorscale: turboScale, cmin: zmin, cmax: zmax,
					text: hover, hovertemplate: '%{text}<br>' + xTitle + ' %{x}<br>ΔF/F₀ %{z:.3f}<extra></extra>',
					contours: { z: { show: true, usecolormap: true, project: { z: true } } },
					colorbar: { title: 'ΔF/F₀', thickness: 12 }
				}], {
					...base,
					margin: { t: 0, r: 0, b: 0, l: 0 },
					scene: {
						xaxis: { title: xTitle, color: text },
						yaxis: { title: sortLabel, color: text, showticklabels: false },
						zaxis: { title: 'ΔF/F₀', color: text },
						camera: { eye: { x: 1.6, y: -1.6, z: 0.9 } },
						aspectratio: { x: 1.6, y: 1, z: 0.6 }
					}
				}, config);
			}
		}

		// Clicking the chart jumps the viewer to that frame (2-D views only:
		// in 3D a click is a rotation).
		el.removeAllListeners?.('plotly_click');
		if (chartView !== 'surface') {
			el.on?.('plotly_click', (ev: any) => {
				const x = ev?.points?.[0]?.x;
				if (x === undefined || x === null) return;
				const idx = xs.findIndex((v) => v >= x);
				frame = Math.max(0, idx === -1 ? xs.length - 1 : idx);
				playing = false;
				viewMode = 'frames';
			});
		}
	}

	$effect(() => {
		void matrix;
		void chartEl;
		void chartView;
		void heatSort;
		void selectedCell;
		drawChart();
	});

	$effect(() => {
		const f = frame;
		if (plotly && chartEl && matrix && chartView !== 'surface') {
			void f;
			plotly.relayout(chartEl, { shapes: frameMarker(matrix) });
		}
	});

	// ── Running ────────────────────────────────────────────────
	function connectWebSocket() {
		if (!$sessionId) return;
		socket = new ProgressSocket($sessionId);
		socket.onMessage(handleWSMessage);
		socket.connect();
	}

	function disconnectWebSocket() {
		socket?.disconnect();
		socket = null;
	}

	function stopPoll() {
		if (pollTimer) clearInterval(pollTimer);
		pollTimer = null;
	}

	function finish(status: string, message = '') {
		running = false;
		wsStatus = status;
		stopPoll();
		disconnectWebSocket();
		if (status === 'complete') {
			wsProgress = 100;
			wsMessage = 'Video analysis complete';
			errorText = '';
			runStamp = String(Date.now());
			// Show the result as a video, cells coloured by response, playing.
			viewMode = 'frames';
			heat = $videoMode === 'static_roi' ? 'cell' : 'none';
			frame = 0;
			playing = true;
			loadSummary();
		} else {
			errorText = message || 'Video analysis failed';
		}
	}

	// The websocket owns live progress, but a task that fails before the
	// socket subscribes would leave this page stuck at 0% forever. Poll for
	// terminal states as a backstop, exactly as the segmentation page does.
	function startPoll(taskId: string) {
		stopPoll();
		pollTimer = setInterval(async () => {
			try {
				const st = await getSegmentationStatus(taskId);
				if (st.status === 'complete') finish('complete');
				else if (st.status === 'error' || st.status === 'cancelled')
					finish(st.status, st.error || st.message);
			} catch {
				finish('error', 'Task is no longer known to the server');
			}
		}, 3000);
	}

	function handleWSMessage(msg: any) {
		if (msg.task_id && msg.task_id !== $videoTaskId) return;
		if (msg.type === 'progress') {
			wsProgress = msg.progress ?? 0;
			wsMessage = msg.message ?? '';
			wsStatus = msg.status ?? '';
		} else if (msg.type === 'task_complete') {
			finish(msg.status ?? 'complete', msg.error ?? msg.message);
		}
	}

	async function loadSummary() {
		if (!$sessionId) return;
		try {
			const data = await getVideoSummary($sessionId);
			videoSummary.set(data.rows ?? []);
		} catch {
			// No results yet for this session.
		}
	}

	async function loadDoses() {
		if (!$sessionId) return;
		try {
			const data = await parseDoses($sessionId);
			doseTable.set(data.doses ?? []);
		} catch (e) {
			console.error('Failed to parse doses', e);
		}
	}

	async function start() {
		if (!$sessionId) return;
		running = true;
		wsProgress = 0;
		wsStatus = '';
		wsMessage = 'Submitting…';
		errorText = '';
		connectWebSocket();
		try {
			const res = await runVideo($sessionId, {
				mode: $videoMode,
				projection,
				projection_percentile: projectionPercentile,
				diameter,
				baseline_mode: baselineMode,
				baseline_fraction: baselineFraction,
				stim_frame: stimFrame,
				interval_s: intervalS,
				event_threshold_sigma: thresholdSigma
			});
			videoTaskId.set(res.task_id);
			frameIntervalS.set(intervalS);
			startPoll(res.task_id);
		} catch (e: any) {
			finish('error', e?.message ?? String(e));
		}
	}

	onMount(loadSummary);
	onDestroy(() => {
		playing = false;
		loadToken += 1;
		clearFrameCache();
		stopPoll();
		disconnectWebSocket();
		if (plotly && chartEl) plotly.purge(chartEl);
	});

	// ── Per-cell table for the video on screen ─────────────────
	let currentSummary = $derived(
		current
			? $videoSummary.filter((r) => r.Condition === current!.condition && r.ImageSet === current!.baseName)
			: []
	);
	let summarySort = $state<'peak' | 'cell'>('peak');
	let sortedSummary = $derived(
		[...currentSummary].sort((a, b) =>
			summarySort === 'peak'
				? (b.peak_dF_F0 ?? -Infinity) - (a.peak_dF_F0 ?? -Infinity)
				: a.CellID - b.CellID
		)
	);
	let responders = $derived(currentSummary.filter((r) => (r.n_events ?? 0) > 0).length);
	let timeLabel = $derived(intervalS ? `${(frame * intervalS).toFixed(2)} s` : '');
</script>

<svelte:window onkeydown={onKey} />

<div class="page-video">
	{#if $conditions.length === 0}
		<section class="panel">
			<h2 class="section-header">Video Analytics</h2>
			<p class="font-ui muted">
				No experiment loaded. Scan a folder on the <a href="/experiment">Experiment</a> page first.
			</p>
		</section>
	{:else}
		<div class="two-col">
			<!-- ── Parameters ─────────────────────────────────── -->
			<section class="panel">
				<h2 class="section-header">Video Parameters</h2>

				<div class="form-grid">
					<div class="form-field">
						<label class="field-label font-ui" for="v-mode">Mode</label>
						<select id="v-mode" class="field-input font-ui" bind:value={$videoMode}>
							<option value="static_roi">Per-cell (segment once, fixed ROIs)</option>
							<option value="whole_field">Whole field (no segmentation)</option>
						</select>
						<span class="field-hint font-ui">Fixed ROIs assume cells do not move during the recording.</span>
					</div>

					{#if $videoMode === 'static_roi'}
						<div class="form-field">
							<label class="field-label font-ui" for="v-proj">Segment on</label>
							<select id="v-proj" class="field-input font-ui" bind:value={projection}>
								<option value="max">Max projection</option>
								<option value="percentile">Percentile projection</option>
								<option value="std">Std-dev projection</option>
								<option value="mean">Mean projection</option>
							</select>
							<span class="field-hint font-ui">{projectionHelp[projection]}</span>
						</div>

						{#if projection === 'percentile'}
							<div class="form-field">
								<label class="field-label font-ui" for="v-pct">Percentile</label>
								<div class="range-row">
									<input id="v-pct" type="range" class="field-range" bind:value={projectionPercentile}
										min="90" max="99.9" step="0.1" />
									<span class="field-value font-mono">{projectionPercentile.toFixed(1)}</span>
								</div>
							</div>
						{/if}

						<div class="form-field">
							<label class="field-label font-ui" for="v-diam">Cell diameter (px)</label>
							<input id="v-diam" type="number" class="field-input font-mono" bind:value={diameter} min="5" max="300" />
						</div>
					{/if}

					<div class="form-field">
						<label class="field-label font-ui" for="v-base">Baseline (F₀)</label>
						<select id="v-base" class="field-input font-ui" bind:value={baselineMode}>
							<option value="fraction">First fraction of frames</option>
							<option value="stim_frame">Frames before stimulus</option>
							<option value="auto">Auto-detect onset</option>
						</select>
					</div>

					{#if baselineMode === 'fraction'}
						<div class="form-field">
							<label class="field-label font-ui" for="v-bf">Baseline fraction</label>
							<div class="range-row">
								<input id="v-bf" type="range" class="field-range" bind:value={baselineFraction}
									min="0.02" max="0.5" step="0.01" />
								<span class="field-value font-mono">{(baselineFraction * 100).toFixed(0)}%</span>
							</div>
							{#if nFrames}
								<span class="field-hint font-ui">
									= first {Math.max(1, Math.round(baselineFraction * nFrames))} of {nFrames} frames
								</span>
							{/if}
						</div>
					{:else if baselineMode === 'stim_frame'}
						<div class="form-field">
							<label class="field-label font-ui" for="v-stim">Stimulus frame</label>
							<input id="v-stim" type="number" class="field-input font-mono" bind:value={stimFrame} min="1" />
							<button class="btn-link font-ui" type="button" onclick={() => (stimFrame = frame)}>
								Use frame shown in preview ({frame})
							</button>
						</div>
					{/if}

					<div class="form-field">
						<label class="field-label font-ui" for="v-int">Frame interval (s)</label>
						<input id="v-int" type="number" class="field-input font-mono" step="0.01" min="0"
							bind:value={intervalS} placeholder="Not set" />
						{#if !intervalS}
							<span class="field-hint warn font-ui">Not set: time metrics will be in frames, not seconds.</span>
						{/if}
					</div>

					<div class="form-field">
						<label class="field-label font-ui" for="v-sig">Event threshold</label>
						<div class="range-row">
							<input id="v-sig" type="range" class="field-range" bind:value={thresholdSigma} min="1" max="8" step="0.5" />
							<span class="field-value font-mono">{thresholdSigma.toFixed(1)}σ</span>
						</div>
					</div>
				</div>

				<div class="action-row">
					<button class="btn btn-primary font-ui" onclick={start} disabled={running || !$sessionId}>
						<Play size={16} />
						{running ? 'Running…' : 'Run Video Analysis'}
					</button>
					<button class="btn btn-secondary font-ui" onclick={loadDoses} disabled={!$sessionId}>
						<FlaskConical size={16} />
						Parse doses
					</button>
				</div>

				{#if running || wsStatus === 'complete'}
					<div class="progress-block">
						<div class="progress-track"><div class="progress-fill" style="width: {wsProgress}%"></div></div>
						<span class="progress-text font-mono">{wsProgress.toFixed(0)}% {wsMessage}</span>
					</div>
				{/if}
				{#if errorText}
					<pre class="error-box font-mono">{errorText}</pre>
				{/if}
			</section>

			<!-- ── Viewer ─────────────────────────────────────── -->
			<section class="panel preview-panel">
				<h2 class="section-header">Video Preview</h2>

				<div class="seg-preview">
					<h3 class="seg-preview-header font-ui">
						{#if current}
							<span class="font-mono seg-preview-count">
								{nFrames} frames{hasRois ? ` · ${nCells} cells` : ''}
							</span>
						{/if}
						<span class="style-toggle">
							<button class="style-btn font-ui" class:active={viewMode === 'frames'} onclick={() => (viewMode = 'frames')}>
								<Film size={12} /> Video
							</button>
							<button class="style-btn font-ui" class:active={viewMode === 'rois'} disabled={!hasRois}
								title={hasRois ? 'The projection the cells were segmented on' : 'Run video analysis to segment cells'}
								onclick={() => { viewMode = 'rois'; playing = false; }}>
								ROIs (projection)
							</button>
						</span>
						{#if viewMode === 'rois'}
							<span class="style-toggle">
								<button class="style-btn font-ui" class:active={overlayStyle === 'filled'}
									onclick={() => (overlayStyle = 'filled')}>Filled</button>
								<button class="style-btn font-ui" class:active={overlayStyle === 'outline'}
									onclick={() => (overlayStyle = 'outline')}>Outline</button>
							</span>
						{/if}
					</h3>

					{#if viewMode === 'frames' && hasRois}
						<div class="overlay-row font-ui">
							<span class="overlay-label">Colour by ΔF/F₀</span>
							<span class="style-toggle">
								<button class="style-btn font-ui" class:active={heat === 'none'} onclick={() => (heat = 'none')}>Off</button>
								<button class="style-btn font-ui" class:active={heat === 'cell'} disabled={!matrix}
									title="Each cell filled with its measured ΔF/F₀ at this frame - the numbers in the table"
									onclick={() => (heat = 'cell')}>Per cell</button>
								<button class="style-btn font-ui" class:active={heat === 'pixel'}
									title="Each pixel inside the cells coloured by its own ΔF/F₀ - shows where in a cell the signal rises"
									onclick={() => (heat = 'pixel')}>Per pixel</button>
							</span>
							<span class="style-toggle">
								<button class="style-btn font-ui" class:active={showRoisOnFrames}
									onclick={() => (showRoisOnFrames = !showRoisOnFrames)}>
									Outlines {showRoisOnFrames ? 'on' : 'off'}
								</button>
							</span>
						</div>
					{/if}

					{#if conditionNames.length > 1}
						<div class="preview-tabs">
							{#each conditionNames as cName}
								<button class="preview-tab font-ui" class:active={activeCondition === cName}
									onclick={() => { activeCondition = cName; videoIndex = 0; }}>
									{cName}
								</button>
							{/each}
						</div>
					{/if}

					<div class="preview-frame">
						{#if imgSrc}
							<img src={imgSrc} alt="{current?.condition}/{current?.baseName}" class="preview-img" />
						{:else if viewMode === 'rois'}
							<p class="muted font-ui">No ROIs yet. Run video analysis to segment this recording.</p>
						{/if}
					</div>

					{#if viewMode === 'frames' && effectiveHeat !== 'none'}
						<div class="colourbar font-mono">
							<span>{heatRange ? heatRange.lo.toFixed(2) : '…'}</span>
							<div class="colourbar-ramp" style="background: {turboGradient}"></div>
							<span>{heatRange ? heatRange.hi.toFixed(2) : '…'}</span>
							<span class="colourbar-label font-ui">
								ΔF/F₀ {effectiveHeat === 'cell' ? 'per cell' : 'per pixel'} · same scale for every frame
							</span>
						</div>
					{/if}

					{#if viewMode === 'rois' && hasRois}
						<p class="caption font-ui">
							Cells were found once on the projection, and these same outlines are measured in every frame.
						</p>
					{/if}

					{#if viewMode === 'frames' && nFrames > 1}
						<div class="scrubber">
							<button class="icon-btn" onclick={() => step(-1)} title="Previous frame (←)">
								<ChevronLeft size={16} />
							</button>
							<button class="icon-btn" onclick={togglePlay} title={playing ? 'Pause (space)' : 'Play (space)'}>
								{#if playing}<Pause size={16} />{:else}<Play size={16} />{/if}
							</button>
							<button class="icon-btn" onclick={() => step(1)} title="Next frame (→)">
								<ChevronRight size={16} />
							</button>
							<div class="scrub-track">
								<div class="buffer-bar" style="width: {nFrames ? (buffered / nFrames) * 100 : 0}%"></div>
								<input type="range" class="field-range" min="0" max={nFrames - 1} step="1"
									bind:value={frame} oninput={() => (playing = false)} />
							</div>
							<span class="font-mono frame-label">
								{frame + 1}/{nFrames}{timeLabel ? ` · ${timeLabel}` : ''}
							</span>
						</div>
						<div class="speed-row font-ui">
							<span class="overlay-label">Speed</span>
							<span class="style-toggle">
								{#each SPEEDS as sp}
									<button class="style-btn font-mono" class:active={speed === sp} onclick={() => (speed = sp)}>
										{sp}×
									</button>
								{/each}
							</span>
							<span class="font-mono muted speed-fps">
								{fps.toFixed(fps < 10 ? 1 : 0)} fps{intervalS ? '' : ' (set a frame interval for real time)'}
							</span>
							{#if buffered < nFrames}
								<span class="font-mono muted">loading {buffered}/{nFrames}</span>
							{/if}
						</div>
						{#if frameError}
							<p class="field-hint warn font-ui">{frameError}</p>
						{/if}
					{/if}

					<div class="preview-nav">
						<button class="icon-btn" onclick={prevVideo} disabled={videoIndex === 0}>
							<ChevronLeft size={18} />
						</button>
						<span class="preview-label font-ui">
							{current?.condition} / {current?.baseName}
							<span class="preview-count font-mono">({videoIndex + 1}/{filtered.length})</span>
						</span>
						<button class="icon-btn" onclick={nextVideo} disabled={videoIndex >= filtered.length - 1}>
							<ChevronRight size={18} />
						</button>
					</div>
				</div>
			</section>
		</div>

		<!-- ── Traces ─────────────────────────────────────────── -->
		<section class="panel traces-panel">
			<h2 class="section-header">
				<Activity size={14} /> Traces
				{#if current}<span class="muted font-ui header-sub">{current.condition} / {current.baseName}</span>{/if}
			</h2>
			{#if matrix}
				<div class="chart-controls">
					<span class="style-toggle">
						<button class="style-btn font-ui" class:active={chartView === 'lines'} onclick={() => (chartView = 'lines')}>Lines</button>
						<button class="style-btn font-ui" class:active={chartView === 'heatmap'} onclick={() => (chartView = 'heatmap')}>Heatmap</button>
						<button class="style-btn font-ui" class:active={chartView === 'surface'} onclick={() => (chartView = 'surface')}>3D surface</button>
					</span>
					{#if chartView !== 'lines'}
						<span class="style-toggle">
							<button class="style-btn font-ui" class:active={heatSort === 'peak_time'} onclick={() => (heatSort = 'peak_time')}>Sort: peak time</button>
							<button class="style-btn font-ui" class:active={heatSort === 'peak'} onclick={() => (heatSort = 'peak')}>Sort: peak size</button>
							<button class="style-btn font-ui" class:active={heatSort === 'cell'} onclick={() => (heatSort = 'cell')}>Sort: cell ID</button>
						</span>
					{:else if selectedCell !== null}
						<span class="style-toggle">
							<button class="style-btn font-ui active" onclick={() => (selectedCell = null)}>Cell {selectedCell} ✕</button>
						</span>
					{/if}
				</div>
				<p class="caption font-ui">
					{#if chartView === 'lines'}
						Grey: each cell. Bold: mean of {matrix.cells.length} cells. Dotted line: the frame in the
						preview. Click the chart to jump to a frame; click a cell in the table below to highlight it.
					{:else if chartView === 'heatmap'}
						One row per cell ({matrix.cells.length}), one column per frame, colour = ΔF/F₀. Colour range
						clipped to the 2nd–98th percentile so a few very bright cells don't wash out the rest.
					{:else}
						Same data as the heatmap, drawn as a surface. Drag to rotate, scroll to zoom.
					{/if}
				</p>
				<div class="chart" class:chart-tall={chartView !== 'lines'} bind:this={chartEl}></div>
			{:else}
				<p class="muted font-ui">No traces for this recording yet. Run video analysis.</p>
			{/if}
		</section>

		<!-- ── Per-cell results ───────────────────────────────── -->
		{#if currentSummary.length > 0}
			<section class="panel">
				<h2 class="section-header">
					Per-cell results
					<span class="muted font-ui header-sub">
						{currentSummary.length} cells · {responders} with events ·
						<a href="/results">all videos in Results</a>
					</span>
				</h2>
				<div class="style-toggle sort-toggle">
					<button class="style-btn font-ui" class:active={summarySort === 'peak'}
						onclick={() => (summarySort = 'peak')}>Sort by peak</button>
					<button class="style-btn font-ui" class:active={summarySort === 'cell'}
						onclick={() => (summarySort = 'cell')}>Sort by cell</button>
				</div>
				<div class="table-wrap">
					<table class="data-table font-ui">
						<thead>
							<tr>
								<th>Cell</th><th class="r">Peak ΔF/F₀</th><th class="r">Peak frame</th>
								<th class="r">AUC</th><th class="r">Events</th><th class="r">SNR</th>
							</tr>
						</thead>
						<tbody>
							{#each sortedSummary.slice(0, 300) as r}
								<tr>
									<td class="font-mono">
									<button class="btn-link font-mono" title="Highlight this cell's trace"
										onclick={() => { selectedCell = r.CellID; chartView = 'lines'; }}>
										{r.CellID}
									</button>
								</td>
									<td class="r font-mono">{r.peak_dF_F0?.toFixed(3) ?? '–'}</td>
									<td class="r font-mono">
										<button class="btn-link font-mono" title="Show this frame"
											onclick={() => { frame = r.peak_frame ?? 0; playing = false; viewMode = 'frames'; }}>
											{r.peak_frame ?? '–'}
										</button>
									</td>
									<td class="r font-mono">{r.auc?.toFixed(2) ?? '–'}</td>
									<td class="r font-mono">{r.n_events ?? 0}</td>
									<td class="r font-mono">{r.snr?.toFixed(1) ?? '–'}</td>
								</tr>
							{/each}
						</tbody>
					</table>
				</div>
			</section>
		{/if}

		{#if $doseTable.length > 0}
			<section class="panel">
				<h2 class="section-header">
					Parsed doses <span class="muted font-ui header-sub">confirm before fitting</span>
				</h2>
				<div class="table-wrap">
					<table class="data-table font-ui">
						<thead>
							<tr>
								<th>Source</th><th>Parsed</th><th class="r">Molar</th>
								<th class="r">Volume (µL)</th><th>Group</th><th>Note</th>
							</tr>
						</thead>
						<tbody>
							{#each $doseTable as d}
								<tr class:warn-row={!d.parsed || d.ambiguous}>
									<td>{d.source}</td>
									<td>{d.parsed ? 'yes' : 'NO'}</td>
									<td class="r font-mono">{d.molar ?? '–'}</td>
									<td class="r font-mono">{d.volume_ul ?? '–'}</td>
									<td class="font-mono">{d.group_key}</td>
									<td>{d.note}</td>
								</tr>
							{/each}
						</tbody>
					</table>
				</div>
			</section>
		{/if}
	{/if}
</div>

<style>
	.page-video {
		max-width: 1200px;
		display: flex;
		flex-direction: column;
		gap: 24px;
	}

	/* Shared with the Segmentation page so both pages look the same. */
	.two-col {
		display: grid;
		grid-template-columns: 380px 1fr;
		gap: 24px;
		align-items: start;
	}

.form-grid {
		display: flex;
		flex-direction: column;
		gap: 18px;
	}

.form-field {
		display: flex;
		flex-direction: column;
		gap: 6px;
	}

.field-label {
		font-size: 12px;
		font-weight: 500;
		color: var(--text);
		display: flex;
		align-items: center;
		gap: 6px;
	}

:global(.dark) .field-label {
		color: var(--text-muted);
		text-transform: uppercase;
		letter-spacing: 0.04em;
		font-size: 11px;
	}

.field-input {
		padding: 10px 12px;
		background: var(--bg);
		border: 1px solid var(--border);
		border-radius: var(--radius-md);
		color: var(--text);
		font-size: 13px;
	}

.field-input:focus {
		border-color: var(--accent);
		outline: none;
		box-shadow: 0 0 0 3px var(--accent-soft);
	}

.range-row {
		display: flex;
		align-items: center;
		gap: 10px;
	}

.field-range {
		flex: 1;
		accent-color: var(--accent);
	}

.field-value {
		font-size: 12px;
		color: var(--accent);
		min-width: 36px;
		text-align: right;
	}

.field-hint {
		font-size: 11px;
		color: var(--text-faint);
	}

.action-row {
		display: flex;
		gap: 10px;
		margin-top: 24px;
	}

.btn {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		padding: 10px 20px;
		border-radius: var(--radius-md);
		font-size: 13px;
		font-weight: 600;
		cursor: pointer;
		transition: all var(--transition-fast);
		border: none;
	}

.btn:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}

.btn-primary {
		background: var(--accent);
		color: white;
	}

:global(.dark) .btn-primary {
		color: #000;
	}

.btn-primary:hover:not(:disabled) {
		filter: brightness(1.1);
		transform: translateY(-1px);
	}

.btn-secondary {
		background: var(--bg);
		color: var(--text);
		border: 1px solid var(--border);
	}

.btn-secondary:hover {
		border-color: var(--accent);
		color: var(--accent);
	}

.preview-panel {
		min-height: 400px;
		display: flex;
		flex-direction: column;
		gap: 16px;
	}

/* ── Segmentation preview ──────────────────────────── */

	.seg-preview {
		background: var(--bg);
		border: 1px solid var(--border);
		border-radius: var(--radius-md);
		padding: 12px;
	}

.seg-preview-header {
		font-size: 13px;
		font-weight: 600;
		color: var(--text);
		margin: 0 0 10px;
		display: flex;
		align-items: center;
		gap: 8px;
	}

.seg-preview-count {
		font-weight: 400;
		font-size: 11px;
		color: var(--text-faint);
	}

.style-toggle {
		margin-left: auto;
		display: flex;
		gap: 1px;
		background: var(--border);
		border-radius: var(--radius-sm);
		overflow: hidden;
	}

.style-btn {
		background: var(--bg-elevated);
		border: none;
		padding: 3px 10px;
		font-size: 11px;
		font-weight: 500;
		color: var(--text-muted);
		cursor: pointer;
		transition: all var(--transition-fast);
	}

.style-btn:hover {
		color: var(--text);
	}

.style-btn.active {
		background: var(--accent);
		color: white;
	}

:global(.dark) .style-btn.active {
		color: #000;
	}

.preview-tabs {
		display: flex;
		gap: 1px;
		margin-bottom: 8px;
		border-bottom: 1px solid var(--border);
	}

.preview-tab {
		background: none;
		border: none;
		padding: 5px 12px;
		font-size: 12px;
		font-weight: 600;
		color: var(--text-muted);
		cursor: pointer;
		border-bottom: 2px solid transparent;
		transition: all var(--transition-fast);
	}

.preview-tab:hover {
		color: var(--text);
	}

.preview-tab.active {
		color: var(--accent);
		border-bottom-color: var(--accent);
	}

.preview-frame {
		background: #000;
		border-radius: var(--radius-md);
		overflow: hidden;
		position: relative;
		aspect-ratio: 4 / 3;
	}

.preview-img {
		position: absolute;
		inset: 0;
		width: 100%;
		height: 100%;
		object-fit: contain;
	}

.preview-nav {
		display: flex;
		align-items: center;
		justify-content: center;
		gap: 10px;
		margin-top: 8px;
	}

.preview-label {
		font-size: 12px;
		color: var(--text);
		text-align: center;
	}

.preview-count {
		color: var(--text-muted);
		font-size: 11px;
	}

.icon-btn {
		background: none;
		border: none;
		color: var(--text-muted);
		cursor: pointer;
		padding: 4px;
		display: inline-flex;
		align-items: center;
		border-radius: var(--radius-sm);
	}

.icon-btn:hover:not(:disabled) {
		color: var(--accent);
		background: var(--accent-soft);
	}

.icon-btn:disabled {
		opacity: 0.3;
		cursor: not-allowed;
	}

/* ── Log panel ──────────────────────────────────────── */

	.log-panel {
		margin-top: 16px;
		background: var(--bg);
		border: 1px solid var(--border);
		border-radius: var(--radius-md);
		overflow: hidden;
	}

.log-summary {
		padding: 10px 14px;
		cursor: pointer;
		font-size: 13px;
		font-weight: 600;
		color: var(--text);
		user-select: none;
		display: flex;
		align-items: center;
		gap: 8px;
	}

.log-summary:hover {
		color: var(--accent);
	}

.log-count {
		font-weight: 400;
		font-size: 11px;
		color: var(--text-faint);
	}

.log-output {
		max-height: 320px;
		overflow-y: auto;
		padding: 12px 14px;
		margin: 0;
		font-size: 12px;
		line-height: 1.5;
		color: #d4d4d4;
		background: #1e1e1e;
		border-top: 1px solid var(--border);
		white-space: pre-wrap;
		word-break: break-all;
	}

.per-condition-channels .field-label {
		margin-bottom: 4px;
	}

	.muted {
		color: var(--text-muted);
	}
	.header-sub {
		font-size: 12px;
		font-weight: 400;
		margin-left: 8px;
		text-transform: none;
		letter-spacing: 0;
	}
	.header-sub a,
	.muted a {
		color: var(--accent);
	}
	.field-hint.warn {
		color: #c98a2b;
	}
	.caption {
		font-size: 12px;
		color: var(--text-muted);
		margin: 0;
	}

	.btn-link {
		background: none;
		border: none;
		padding: 0;
		cursor: pointer;
		color: var(--accent);
		font-size: 12px;
		text-align: left;
	}
	.btn-link:hover {
		text-decoration: underline;
	}

	.progress-block {
		margin-top: 16px;
		display: flex;
		flex-direction: column;
		gap: 6px;
	}
	.progress-track {
		height: 6px;
		background: var(--bg-sunken);
		border-radius: 3px;
		overflow: hidden;
	}
	.progress-fill {
		height: 100%;
		background: var(--accent);
		transition: width 0.3s ease;
	}
	.progress-text {
		font-size: 11px;
		color: var(--text-muted);
	}

	.error-box {
		margin-top: 12px;
		padding: 10px 12px;
		max-height: 220px;
		overflow: auto;
		font-size: 11px;
		white-space: pre-wrap;
		background: rgba(200, 60, 60, 0.08);
		border: 1px solid rgba(200, 60, 60, 0.4);
		border-radius: var(--radius-md);
		color: var(--text);
	}

	.scrubber {
		display: flex;
		align-items: center;
		gap: 6px;
	}
	.scrub-track {
		position: relative;
		flex: 1;
		display: flex;
		align-items: center;
	}
	.scrub-track .field-range {
		width: 100%;
		position: relative;
	}
	/* How many frames are in memory, drawn under the slider like a video
	   player's buffer bar. */
	.buffer-bar {
		position: absolute;
		left: 0;
		top: 50%;
		height: 4px;
		transform: translateY(-50%);
		background: var(--accent-soft);
		border-radius: 2px;
		pointer-events: none;
		transition: width 0.2s ease;
	}
	.speed-row,
	.overlay-row {
		display: flex;
		align-items: center;
		gap: 10px;
		flex-wrap: wrap;
		font-size: 12px;
	}
	.overlay-label {
		color: var(--text-muted);
		font-size: 12px;
	}
	.speed-fps {
		font-size: 11px;
	}
	.colourbar {
		display: flex;
		align-items: center;
		gap: 8px;
		font-size: 11px;
		color: var(--text-muted);
	}
	.colourbar-ramp {
		flex: 0 0 200px;
		height: 10px;
		border-radius: 2px;
		border: 1px solid var(--border);
	}
	.colourbar-label {
		margin-left: 6px;
	}
	.frame-label {
		font-size: 12px;
		color: var(--text-muted);
		min-width: 92px;
		text-align: right;
	}

	.traces-panel .chart {
		width: 100%;
		min-height: 320px;
	}
	.traces-panel .chart.chart-tall {
		min-height: 480px;
	}
	.chart-controls {
		display: flex;
		flex-wrap: wrap;
		gap: 10px;
		margin-bottom: 8px;
	}

	.sort-toggle {
		margin-bottom: 10px;
		display: inline-flex;
	}
	.table-wrap {
		max-height: 420px;
		overflow: auto;
		border: 1px solid var(--border);
		border-radius: var(--radius-md);
	}
	.data-table {
		width: 100%;
		border-collapse: collapse;
		font-size: 12px;
	}
	.data-table th {
		position: sticky;
		top: 0;
		background: var(--bg-elevated);
		text-align: left;
		font-weight: 600;
		padding: 8px 10px;
		border-bottom: 1px solid var(--border);
		color: var(--text-muted);
	}
	.data-table td {
		padding: 6px 10px;
		border-bottom: 1px solid var(--border);
	}
	.data-table .r {
		text-align: right;
	}
	.data-table tr:hover td {
		background: var(--bg-hover);
	}
	.warn-row td {
		background: rgba(224, 163, 62, 0.12);
	}

	@media (max-width: 900px) {
		.two-col {
			grid-template-columns: 1fr;
		}
	}
</style>
