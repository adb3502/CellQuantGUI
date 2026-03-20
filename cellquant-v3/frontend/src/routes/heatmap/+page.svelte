<script lang="ts">
	import { untrack } from 'svelte';
	import { sessionId } from '$stores/session';
	import { conditions as conditionsStore } from '$stores/experiment';
	import { Download, RefreshCw, Info, ChevronLeft, ChevronRight } from 'lucide-svelte';

	const BASE = '/api/v1';

	// ── Selectors ─────────────────────────────────────────────────
	let conditionNames = $derived($conditionsStore.map(c => c.name));
	let selectedCondition = $state('');
	let selectedImageSet = $state('');
	let selectedChannel = $state('');

	let imageSets = $derived(
		($conditionsStore.find(c => c.name === selectedCondition)?.image_sets ?? [])
			.map(s => s.base_name)
	);

	let channels = $derived(() => {
		const cond = $conditionsStore.find(c => c.name === selectedCondition);
		const imgSet = cond?.image_sets.find(s => s.base_name === selectedImageSet);
		return Object.keys(imgSet?.channels ?? {});
	});

	let imageSetIndex = $derived(imageSets.indexOf(selectedImageSet));

	function prevImageSet() {
		if (imageSetIndex > 0) selectedImageSet = imageSets[imageSetIndex - 1];
	}
	function nextImageSet() {
		if (imageSetIndex < imageSets.length - 1) selectedImageSet = imageSets[imageSetIndex + 1];
	}

	// Auto-select first values when options change
	$effect(() => {
		const names = conditionNames;
		if (names.length > 0 && !names.includes(selectedCondition))
			selectedCondition = names[0];
	});
	$effect(() => {
		const sets = imageSets;
		if (sets.length > 0 && !sets.includes(selectedImageSet))
			selectedImageSet = sets[0];
	});
	$effect(() => {
		const chs = channels();
		if (chs.length > 0 && !chs.includes(selectedChannel))
			selectedChannel = chs[0];
	});

	// ── Version counter for cache-busting ─────────────────────────
	let version = $state(0);

	function heatmapUrl(metric: string) {
		if (!$sessionId || !selectedCondition || !selectedImageSet || !selectedChannel) return '';
		if (metric === 'background') {
			// Use background-regions endpoint: shows sampled bg pixels, not cells colored by bg
			const p = new URLSearchParams({
				condition: selectedCondition, image_set: selectedImageSet,
				channel: selectedChannel, v: String(version),
			});
			return `${BASE}/quantification/background-regions/${$sessionId}?${p}`;
		}
		const p = new URLSearchParams({
			condition: selectedCondition, image_set: selectedImageSet,
			channel: selectedChannel, metric, v: String(version),
		});
		return `${BASE}/quantification/heatmap-image/${$sessionId}?${p}`;
	}

	function idMapUrl() {
		if (!$sessionId || !selectedCondition || !selectedImageSet) return '';
		const p = new URLSearchParams({
			condition: selectedCondition, image_set: selectedImageSet, v: String(version),
		});
		return `${BASE}/quantification/mask-id-map/${$sessionId}?${p}`;
	}

	// ── Cell data (fetched once per image-set) ────────────────────
	type CellRecord = Record<string, number | boolean | string | null>;
	let cellData = $state<Record<string, CellRecord>>({});
	let cellDataError = $state('');
	// Channels that actually have quantification results for this image set
	let quantifiedChannels = $state<string[]>([]);

	async function loadCellData() {
		if (!$sessionId || !selectedCondition || !selectedImageSet) return;
		cellData = {};
		cellDataError = '';
		quantifiedChannels = [];
		try {
			const p = new URLSearchParams({ condition: selectedCondition, image_set: selectedImageSet });
			const res = await fetch(`${BASE}/quantification/cell-data/${$sessionId}?${p}`);
			if (res.ok) {
				const json = await res.json();
				cellData = json.cells ?? {};
				// Extract channel names from column keys like "{channel}_MeanIntensity"
				const first: Record<string, unknown> = Object.values(cellData)[0] as any ?? {};
				const chSet = new Set<string>();
				for (const col of Object.keys(first)) {
					const m = col.match(/^(.+)_(?:CTCF|MeanIntensity|Background)$/);
					if (m) chSet.add(m[1]);
				}
				quantifiedChannels = [...chSet].sort();
				// Auto-select first quantified channel if current selection isn't quantified
				if (quantifiedChannels.length > 0 && !quantifiedChannels.includes(selectedChannel)) {
					selectedChannel = quantifiedChannels[0];
				}
			} else {
				cellDataError = (await res.json().catch(() => ({}))).detail ?? 'No results';
			}
		} catch (e: any) {
			cellDataError = e.message;
		}
	}

	// ── ID map (offscreen canvas for hit-testing) ─────────────────
	let idMapCanvas: HTMLCanvasElement | null = null;
	let idMapCtx: CanvasRenderingContext2D | null = null;
	let idMapWidth = 0;
	let idMapHeight = 0;

	async function loadIdMap() {
		const url = idMapUrl();
		if (!url) return;
		idMapCanvas = null;
		idMapCtx = null;
		try {
			const img = new Image();
			img.src = url;
			await new Promise<void>((resolve, reject) => {
				img.onload = () => resolve();
				img.onerror = () => reject(new Error('ID map load failed'));
			});
			const cvs = document.createElement('canvas');
			cvs.width = img.naturalWidth;
			cvs.height = img.naturalHeight;
			idMapWidth = img.naturalWidth;
			idMapHeight = img.naturalHeight;
			const ctx = cvs.getContext('2d', { willReadFrequently: true });
			if (!ctx) return;
			ctx.drawImage(img, 0, 0);
			idMapCanvas = cvs;
			idMapCtx = ctx;
		} catch {}
	}

	function getCellIdAt(canvasX: number, canvasY: number): number {
		if (!idMapCtx || canvasX < 0 || canvasY < 0 || canvasX >= idMapWidth || canvasY >= idMapHeight) return 0;
		const px = idMapCtx.getImageData(canvasX, canvasY, 1, 1).data;
		return (px[0] << 8) | px[1];
	}

	// ── Hover / selection state ───────────────────────────────────
	let hoveredCellId = $state(0);
	let selectedCellId = $state(0);
	let tooltipX = $state(0);
	let tooltipY = $state(0);
	let showTooltip = $state(false);

	function handleMouseMove(e: MouseEvent) {
		if (!idMapCtx) return;
		const img = (e.currentTarget as HTMLElement).querySelector('img.heatmap-img') as HTMLImageElement | null;
		if (!img) return;
		const rect = img.getBoundingClientRect();
		const scaleX = idMapWidth / rect.width;
		const scaleY = idMapHeight / rect.height;
		const cx = Math.floor((e.clientX - rect.left) * scaleX);
		const cy = Math.floor((e.clientY - rect.top) * scaleY);
		hoveredCellId = getCellIdAt(cx, cy);
		tooltipX = e.clientX;
		tooltipY = e.clientY;
		showTooltip = hoveredCellId > 0;
	}

	function handleMouseLeave() {
		hoveredCellId = 0;
		showTooltip = false;
	}

	function handleClick(e: MouseEvent) {
		if (hoveredCellId > 0) {
			selectedCellId = hoveredCellId === selectedCellId ? 0 : hoveredCellId;
		} else {
			selectedCellId = 0;
		}
	}

	// ── Selected cell data ─────────────────────────────────────────
	let selectedCell = $derived(selectedCellId > 0 ? (cellData[String(selectedCellId)] ?? null) : null);

	// ── Range tracking ────────────────────────────────────────────
	let meanRange = $state('');
	let bgRange = $state('');
	let ctcfRange = $state('');
	let loadingMean = $state(false);
	let loadingBg = $state(false);
	let loadingCtcf = $state(false);

	async function fetchWithRange(metric: string, url: string) {
		if (!url) return;
		if (metric === 'mean') loadingMean = true;
		if (metric === 'background') loadingBg = true;
		if (metric === 'ctcf') loadingCtcf = true;
		try {
			const res = await fetch(url);
			if (res.ok) {
				const vmin = res.headers.get('X-Vmin');
				const vmax = res.headers.get('X-Vmax');
				const range = vmin && vmax ? `${Number(vmin).toFixed(1)} – ${Number(vmax).toFixed(1)}` : '';
				if (metric === 'mean') meanRange = range;
				if (metric === 'background') bgRange = range;
				if (metric === 'ctcf') ctcfRange = range;
			}
		} finally {
			if (metric === 'mean') loadingMean = false;
			if (metric === 'background') loadingBg = false;
			if (metric === 'ctcf') loadingCtcf = false;
		}
	}

	function reload() {
		version++;
		meanRange = ''; bgRange = ''; ctcfRange = '';
		hoveredCellId = 0; selectedCellId = 0;
		// Snapshot URLs now (after version bump) so async fetches don't read reactive state
		const urlMean = heatmapUrl('mean');
		const urlBg   = heatmapUrl('background');
		const urlCtcf = heatmapUrl('ctcf');
		fetchWithRange('mean', urlMean);
		fetchWithRange('background', urlBg);
		fetchWithRange('ctcf', urlCtcf);
		loadIdMap();
		loadCellData();
	}

	$effect(() => {
		const _c = selectedCondition, _i = selectedImageSet, _ch = selectedChannel, _sid = $sessionId;
		if (_c && _i && _ch && _sid) untrack(() => reload());
	});

	function downloadImage(metric: string) {
		const url = heatmapUrl(metric);
		if (!url) return;
		const a = document.createElement('a');
		a.href = url;
		a.download = `${selectedCondition}_${selectedImageSet}_${selectedChannel}_${metric}.png`;
		a.click();
	}

	// ── Format helpers ────────────────────────────────────────────
	function fmt(v: number | boolean | string | null | undefined): string {
		if (v == null) return '—';
		if (typeof v === 'boolean') return v ? 'Yes' : 'No';
		if (typeof v === 'number') return Number.isInteger(v) ? v.toLocaleString() : v.toFixed(2);
		return String(v);
	}

	// Key metrics to show prominently
	const KEY_METRICS = ['Area', 'MeanIntensity', 'CTCF', 'Background', 'MCF'];
	function metricLabel(key: string): string {
		return key.replace(/^[^_]+_/, '').replace(/([A-Z])/g, ' $1').trim();
	}
</script>

<!-- Tooltip -->
{#if showTooltip && hoveredCellId > 0}
	<div
		class="cell-tooltip font-mono"
		style="left: {tooltipX + 14}px; top: {tooltipY - 10}px"
	>
		Cell {hoveredCellId}
		{#if cellData[String(hoveredCellId)]}
			{@const cell = cellData[String(hoveredCellId)]}
			{#each KEY_METRICS as m}
				{@const col = Object.keys(cell).find(k => k.endsWith('_' + m))}
				{#if col}
					<span class="tt-val">{metricLabel(col)}: {fmt(cell[col])}</span>
				{/if}
			{/each}
		{:else if cellDataError}
			<span class="tt-val tt-err">{cellDataError}</span>
		{/if}
	</div>
{/if}

<div class="page-heatmap">
	<div class="page-header">
		<div>
			<h1 class="page-title font-display">Cell Heatmaps</h1>
			<p class="page-subtitle font-ui">Hover to inspect cells · click to select · values match Results table Cell IDs</p>
		</div>
	</div>

	<!-- Selectors + navigation -->
	<div class="selector-bar">
		<div class="selector-group">
			<label class="selector-label font-ui">Condition</label>
			<select class="selector font-ui" bind:value={selectedCondition}>
				{#each conditionNames as c}<option value={c}>{c}</option>{/each}
			</select>
		</div>
		<div class="selector-group">
			<label class="selector-label font-ui">Image Set</label>
			<div class="img-set-nav">
				<button class="nav-btn" onclick={prevImageSet} disabled={imageSetIndex <= 0}>
					<ChevronLeft size={16} />
				</button>
				<select class="selector font-ui" bind:value={selectedImageSet}>
					{#each imageSets as s}<option value={s}>{s}</option>{/each}
				</select>
				<button class="nav-btn" onclick={nextImageSet} disabled={imageSetIndex >= imageSets.length - 1}>
					<ChevronRight size={16} />
				</button>
				<span class="img-set-counter font-mono">
					{imageSetIndex + 1} / {imageSets.length}
				</span>
			</div>
		</div>
		<div class="selector-group">
			<label class="selector-label font-ui">Channel</label>
			<select class="selector font-ui" bind:value={selectedChannel}>
				{#each (quantifiedChannels.length > 0 ? quantifiedChannels : channels()) as ch}
					<option value={ch}>{ch}</option>
				{/each}
			</select>
		</div>
		<button class="reload-btn font-ui" onclick={reload}>
			<RefreshCw size={14} /> Reload
		</button>
	</div>

	<div class="main-layout">
		<!-- Three panels -->
		<div class="heatmap-grid">
			{#each [
				{ metric: 'mean',       label: 'Mean Intensity',  range: meanRange,  loading: loadingMean,  cmap: 'cb-viridis' },
				{ metric: 'ctcf',       label: 'CTCF',            range: ctcfRange,  loading: loadingCtcf,  cmap: 'cb-viridis' },
				{ metric: 'background', label: 'Background Regions', range: bgRange,  loading: loadingBg,    cmap: 'cb-rdylbu'  },
			] as panel}
				{@const panelUrl = heatmapUrl(panel.metric)}
				<div class="heatmap-panel">
					<div class="panel-header">
						<div class="panel-title font-ui">
							{panel.label}
							{#if panel.range}
								<span class="range-badge font-mono">{panel.range}</span>
							{/if}
						</div>
						<div class="panel-actions">
							<div class="colorbar-row font-ui">
								<span class="cb-low">low</span>
								<div class="colorbar {panel.cmap}"></div>
								<span class="cb-high">high</span>
							</div>
							<button class="dl-btn" onclick={() => downloadImage(panel.metric)} title="Download PNG">
								<Download size={13} />
							</button>
						</div>
					</div>

					<div
						class="panel-body"
						onmousemove={handleMouseMove}
						onmouseleave={handleMouseLeave}
						onclick={handleClick}
						role="img"
						aria-label="{panel.label} heatmap"
					>
						{#if panelUrl}
							<div class="img-wrap" class:loading={panel.loading}>
								<img
									src={panelUrl}
									alt="{panel.label} heatmap"
									class="heatmap-img"
									draggable="false"
								/>
								{#if panel.loading}
									<div class="img-spinner"></div>
								{/if}
								<!-- Cell highlight overlay (CSS outline trick via data attr) -->
								{#if hoveredCellId > 0}
									<div class="hover-badge font-mono">Cell {hoveredCellId}</div>
								{/if}
							</div>
						{:else}
							<div class="panel-empty font-ui">Select condition, image set and channel</div>
						{/if}
					</div>
				</div>
			{/each}
		</div>

		<!-- Selected cell sidebar -->
		{#if selectedCellId > 0 && selectedCell}
			<div class="cell-sidebar">
				<div class="sidebar-header font-ui">
					Cell <span class="cell-id-badge font-mono">#{selectedCellId}</span>
					<button class="close-btn" onclick={() => selectedCellId = 0}>×</button>
				</div>
				<div class="sidebar-body">
					{#each Object.entries(selectedCell) as [key, val]}
						{@const isFlag = typeof val === 'boolean'}
						<div class="metric-row" class:flag-row={isFlag && val === true}>
							<span class="metric-key font-ui">{key}</span>
							<span class="metric-val font-mono" class:flag-true={isFlag && val === true}>{fmt(val)}</span>
						</div>
					{/each}
				</div>
			</div>
		{:else if selectedCellId > 0}
			<div class="cell-sidebar">
				<div class="sidebar-header font-ui">
					Cell <span class="cell-id-badge font-mono">#{selectedCellId}</span>
					<button class="close-btn" onclick={() => selectedCellId = 0}>×</button>
				</div>
				<div class="sidebar-empty font-ui">
					{cellDataError || 'No data — run quantification first'}
				</div>
			</div>
		{/if}
	</div>

	<div class="legend-note font-ui">
		<Info size={13} />
		Cell IDs match the CellID column in the Results table.
		Viridis: dark purple = low, yellow = high.
		RdYlBu: blue = low background, red = high.
	</div>
</div>

<style>
	/* ── Layout ───────────────────────────────────── */
	.page-heatmap { display: flex; flex-direction: column; gap: 16px; }

	.page-header { margin-bottom: 4px; }
	.page-title { font-size: 22px; font-weight: 600; color: var(--text); margin: 0 0 4px 0; }
	:global(.dark) .page-title { font-weight: 400; }
	.page-subtitle { font-size: 12px; color: var(--text-muted); margin: 0; }

	/* ── Selectors ────────────────────────────────── */
	.selector-bar {
		display: flex; align-items: flex-end; gap: 12px; flex-wrap: wrap;
		padding: 14px 16px; background: var(--bg-elevated);
		border: 1px solid var(--border); border-radius: var(--radius-lg);
	}
	.selector-group { display: flex; flex-direction: column; gap: 4px; }
	.selector-label { font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; color: var(--text-muted); }
	.selector {
		padding: 6px 10px; background: var(--bg); border: 1px solid var(--border);
		border-radius: var(--radius-sm); color: var(--text); font-size: 13px; min-width: 140px; cursor: pointer;
	}
	.selector:focus { border-color: var(--accent); outline: none; }

	.img-set-nav { display: flex; align-items: center; gap: 4px; }
	.nav-btn {
		display: flex; align-items: center; justify-content: center;
		padding: 5px; background: var(--bg); border: 1px solid var(--border);
		border-radius: var(--radius-sm); color: var(--text-muted); cursor: pointer;
		transition: all 0.15s; flex-shrink: 0;
	}
	.nav-btn:hover:not(:disabled) { border-color: var(--accent); color: var(--accent); }
	.nav-btn:disabled { opacity: 0.3; cursor: not-allowed; }
	.img-set-counter { font-size: 11px; color: var(--text-faint); white-space: nowrap; padding: 0 4px; }

	.reload-btn {
		display: inline-flex; align-items: center; gap: 6px;
		padding: 7px 14px; background: var(--bg); border: 1px solid var(--border);
		border-radius: var(--radius-sm); color: var(--text-muted); font-size: 12px;
		font-weight: 500; cursor: pointer; transition: all 0.15s; align-self: flex-end;
	}
	.reload-btn:hover { border-color: var(--accent); color: var(--accent); }

	/* ── Main layout ──────────────────────────────── */
	.main-layout { display: flex; gap: 14px; align-items: flex-start; }
	.heatmap-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; flex: 1; min-width: 0; }
	@media (max-width: 1200px) { .heatmap-grid { grid-template-columns: repeat(2, 1fr); } }
	@media (max-width: 700px)  { .heatmap-grid { grid-template-columns: 1fr; } }

	/* ── Panel ────────────────────────────────────── */
	.heatmap-panel {
		background: var(--bg-elevated); border: 1px solid var(--border);
		border-radius: var(--radius-lg); overflow: hidden;
		display: flex; flex-direction: column;
	}
	.panel-header {
		display: flex; align-items: center; justify-content: space-between;
		padding: 10px 14px; border-bottom: 1px solid var(--border); gap: 8px;
	}
	.panel-title {
		font-size: 13px; font-weight: 600; color: var(--text);
		display: flex; align-items: center; gap: 8px;
	}
	:global(.dark) .panel-title { font-weight: 500; }
	.range-badge {
		font-size: 10px; font-weight: 400; color: var(--text-muted);
		background: var(--bg); border: 1px solid var(--border);
		border-radius: var(--radius-sm); padding: 1px 6px;
	}
	.panel-actions { display: flex; align-items: center; gap: 8px; }
	.colorbar-row { display: flex; align-items: center; gap: 4px; font-size: 10px; color: var(--text-faint); }
	.colorbar { width: 48px; height: 10px; border-radius: 2px; }
	.cb-viridis { background: linear-gradient(to right, #440154, #482878, #3e4989, #31688e, #26828e, #1f9e89, #35b779, #6dcd59, #b4de2c, #fde725); }
	.cb-rdylbu  { background: linear-gradient(to right, #4575b4, #74add1, #abd9e9, #e0f3f8, #ffffbf, #fee090, #fdae61, #f46d43, #d73027); }
	.cb-low, .cb-high { font-size: 9px; color: var(--text-faint); }
	.dl-btn {
		display: flex; align-items: center; padding: 4px; background: none;
		border: 1px solid transparent; border-radius: var(--radius-sm);
		color: var(--text-faint); cursor: pointer; transition: all 0.15s;
	}
	.dl-btn:hover { border-color: var(--border); color: var(--text-muted); }

	/* ── Panel body / image ───────────────────────── */
	.panel-body {
		flex: 1; display: flex; align-items: center; justify-content: center;
		min-height: 180px; background: #111; cursor: crosshair; position: relative;
	}
	.img-wrap {
		width: 100%; position: relative;
		display: flex; align-items: center; justify-content: center;
	}
	.img-wrap.loading { opacity: 0.4; }
	.heatmap-img { width: 100%; height: auto; display: block; image-rendering: pixelated; user-select: none; }
	.img-spinner {
		position: absolute; width: 28px; height: 28px;
		border: 3px solid rgba(255,255,255,0.15); border-top-color: rgba(255,255,255,0.7);
		border-radius: 50%; animation: spin 0.8s linear infinite;
	}
	@keyframes spin { to { transform: rotate(360deg); } }

	.hover-badge {
		position: absolute; top: 6px; left: 6px;
		background: rgba(0,0,0,0.72); color: #fff;
		font-size: 11px; padding: 2px 8px; border-radius: 4px;
		pointer-events: none; letter-spacing: 0.02em;
	}

	.panel-empty {
		font-size: 12px; color: var(--text-faint); text-align: center; padding: 40px 20px;
	}

	/* ── Tooltip ──────────────────────────────────── */
	.cell-tooltip {
		position: fixed; z-index: 9999;
		background: rgba(12, 12, 18, 0.92);
		color: #e8e8e8;
		border: 1px solid rgba(255,255,255,0.12);
		border-radius: 6px;
		padding: 6px 10px;
		font-size: 11px;
		pointer-events: none;
		display: flex; flex-direction: column; gap: 2px;
		max-width: 200px;
		box-shadow: 0 4px 16px rgba(0,0,0,0.5);
		line-height: 1.4;
	}
	.tt-val { color: rgba(255,255,255,0.65); font-size: 10px; }
	.tt-err { color: #f88; }

	/* ── Cell sidebar ─────────────────────────────── */
	.cell-sidebar {
		width: 220px; flex-shrink: 0;
		background: var(--bg-elevated); border: 1px solid var(--border);
		border-radius: var(--radius-lg); overflow: hidden;
		display: flex; flex-direction: column;
		max-height: 600px;
	}
	.sidebar-header {
		display: flex; align-items: center; gap: 8px;
		padding: 10px 14px; border-bottom: 1px solid var(--border);
		font-size: 13px; font-weight: 600; color: var(--text);
	}
	:global(.dark) .sidebar-header { font-weight: 500; }
	.cell-id-badge {
		background: var(--accent-soft); color: var(--accent);
		border-radius: 4px; padding: 1px 7px; font-size: 12px;
	}
	.close-btn {
		margin-left: auto; background: none; border: none;
		color: var(--text-faint); font-size: 18px; cursor: pointer;
		padding: 0 4px; line-height: 1;
	}
	.close-btn:hover { color: var(--text); }
	.sidebar-body { overflow-y: auto; flex: 1; }
	.sidebar-empty { padding: 20px 14px; font-size: 12px; color: var(--text-faint); }

	.metric-row {
		display: flex; justify-content: space-between; align-items: center;
		padding: 5px 14px; border-bottom: 1px solid var(--border); gap: 8px;
	}
	.metric-row:last-child { border-bottom: none; }
	.flag-row { background: rgba(255, 60, 60, 0.07); }
	.metric-key { font-size: 11px; color: var(--text-muted); flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
	.metric-val { font-size: 11px; color: var(--text); flex-shrink: 0; }
	.flag-true { color: #e44; font-weight: 600; }

	/* ── Legend ───────────────────────────────────── */
	.legend-note {
		display: flex; align-items: center; gap: 6px;
		font-size: 11px; color: var(--text-faint);
		padding: 8px 12px; background: var(--bg-elevated);
		border: 1px solid var(--border); border-radius: var(--radius-md);
	}
</style>
