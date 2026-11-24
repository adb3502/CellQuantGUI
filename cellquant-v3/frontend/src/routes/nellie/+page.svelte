<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { get } from 'svelte/store';
	import { Activity, Play, RotateCcw, ChevronDown, ChevronRight, ChevronLeft, Info, FolderOpen } from 'lucide-svelte';
	import { runNellie, getNellieSummary, getNellieResultsPage, maskRenderUrl, openResultFolder } from '$api/client';
	import { ProgressSocket } from '$api/websocket';
	import type { ProgressMessage } from '$api/types';
	import { sessionId } from '$stores/session';
	import { detection } from '$stores/experiment';
	import {
		nellieTaskId, nellieLogs, nellieCompletedImages, nellieRunning,
		nellieProgress, nellieMessage, nellieWsStatus, nellieElapsed,
		nellieResult, nellieSummary, resetNellieState,
		type NellieCompletedImage
	} from '$stores/nellie';
	import TaskStatus from '$components/progress/TaskStatus.svelte';

	// Channel selection
	let channelSuffix = $state('');

	// Nellie params
	let device = $state('auto');
	let removeEdges = $state(false);
	let otsuThresh = $state(false);
	let lowMemory = $state(false);
	let includeOrganelle = $state(true);
	let includeBranch = $state(true);

	// Pixel size overrides
	let pixelSizeXY = $state('');
	let pixelSizeZ = $state('');
	let timeInterval = $state('');

	// Collapsible panels
	let advancedExpanded = $state(false);
	let pixelExpanded = $state(false);

	// WS / socket
	let socket: ProgressSocket | null = null;
	let logPre: HTMLPreElement | undefined = $state();

	// Image preview state
	let previewIndex = $state(0);
	let activeCondition = $state('');
	let overlayStyle: 'filled' | 'outline' = $state('outline');

	// Results state
	let summaryData = $state<Record<string, unknown>[] | null>(null);
	let resultColumns = $state<string[]>([]);
	let resultRows = $state<Record<string, unknown>[]>([]);
	let totalCells = $state(0);
	let loadingResults = $state(false);

	// Track progress across WS messages
	let prevCurrent = -1;
	let prevCondition = '';
	let prevImageSet = '';

	// Available channels from session
	let availableChannels = $derived($detection?.suggested_markers ?? []);

	let previewConditions = $derived(
		[...new Set($nellieCompletedImages.map((img: NellieCompletedImage) => img.condition))]
	);

	let filteredImages = $derived(
		activeCondition
			? $nellieCompletedImages.filter((img: NellieCompletedImage) => img.condition === activeCondition)
			: $nellieCompletedImages
	);

	let currentPreview = $derived(filteredImages[previewIndex] ?? null);

	let previewSrc = $derived(
		currentPreview && $sessionId
			? maskRenderUrl($sessionId, currentPreview.condition, currentPreview.baseName, 800, overlayStyle)
			: null
	);

	// Clamp index when filter changes
	$effect(() => {
		if (previewIndex >= filteredImages.length && filteredImages.length > 0) {
			previewIndex = filteredImages.length - 1;
		}
	});

	// Auto-select first condition tab
	$effect(() => {
		if (!activeCondition && previewConditions.length > 0) {
			activeCondition = previewConditions[0];
		}
	});

	function connectWebSocket() {
		if (!$sessionId || socket) return;
		socket = new ProgressSocket($sessionId);
		socket.onMessage(handleWSMessage);
		socket.connect();
	}

	function disconnectWebSocket() {
		socket?.disconnect();
		socket = null;
	}

	function handleWSMessage(msg: ProgressMessage) {
		if (msg.task_id && msg.task_id !== $nellieTaskId) return;

		// Capture logs
		if (msg.logs && msg.logs.length > 0) {
			$nellieLogs = msg.logs;
			if (logPre) {
				requestAnimationFrame(() => {
					if (logPre) logPre.scrollTop = logPre.scrollHeight;
				});
			}
		}

		if (msg.type === 'progress') {
			$nellieProgress = msg.progress ?? 0;
			$nellieMessage = msg.message ?? '';
			$nellieWsStatus = 'running';
			$nellieElapsed = msg.elapsed_seconds ?? 0;

			// Detect newly completed image
			const cur = msg.current ?? 0;
			if (cur > prevCurrent && prevCurrent >= 0 && prevCondition && prevImageSet) {
				const images = get(nellieCompletedImages);
				const already = images.some(
					(img: NellieCompletedImage) => img.condition === prevCondition && img.baseName === prevImageSet
				);
				if (!already) {
					$nellieCompletedImages = [...images, { condition: prevCondition, baseName: prevImageSet }];
					previewIndex = filteredImages.length;
				}
			}
			prevCurrent = cur;
			prevCondition = msg.condition ?? '';
			prevImageSet = msg.image_set ?? '';

		} else if (msg.type === 'task_complete') {
			// Capture last image
			if (prevCondition && prevImageSet) {
				const images = get(nellieCompletedImages);
				const already = images.some(
					(img: NellieCompletedImage) => img.condition === prevCondition && img.baseName === prevImageSet
				);
				if (!already) {
					$nellieCompletedImages = [...images, { condition: prevCondition, baseName: prevImageSet }];
				}
			}
			$nellieProgress = 100;
			$nellieWsStatus = msg.status ?? 'complete';
			$nellieMessage = msg.message ?? '';
			$nellieElapsed = msg.elapsed_seconds ?? 0;
			$nellieResult = (msg.data as Record<string, unknown>) ?? null;
			$nellieRunning = false;
			disconnectWebSocket();
			if ($nellieWsStatus === 'complete') loadResults();
		}
	}

	async function handleRun() {
		if (!$sessionId) return;
		$nellieRunning = true;
		$nellieProgress = 0;
		$nellieMessage = 'Submitting...';
		$nellieWsStatus = 'pending';
		$nellieResult = null;
		summaryData = null;
		prevCurrent = -1;
		prevCondition = '';
		prevImageSet = '';

		connectWebSocket();

		const includeLevels: string[] = [];
		if (includeOrganelle) includeLevels.push('organelle');
		if (includeBranch) includeLevels.push('branch');

		try {
			const { task_id } = await runNellie($sessionId, {
				channel_suffix: channelSuffix || undefined,
				device,
				remove_edges: removeEdges,
				otsu_thresh_intensity: otsuThresh,
				low_memory: lowMemory,
				pixel_size_xy: pixelSizeXY ? parseFloat(pixelSizeXY) : null,
				pixel_size_z: pixelSizeZ ? parseFloat(pixelSizeZ) : null,
				time_interval: timeInterval ? parseFloat(timeInterval) : null,
				include_levels: includeLevels,
			});
			$nellieTaskId = task_id;
			$nellieWsStatus = 'running';
		} catch (e) {
			$nellieRunning = false;
			$nellieWsStatus = 'error';
			$nellieMessage = e instanceof Error ? e.message : 'Failed to start nellie analysis';
			disconnectWebSocket();
		}
	}

	async function loadResults() {
		if (!$sessionId) return;
		loadingResults = true;
		try {
			const [summary, page] = await Promise.all([
				getNellieSummary($sessionId),
				getNellieResultsPage($sessionId, 0, 500),
			]);
			summaryData = summary.per_condition;
			$nellieSummary = summary.per_condition;
			totalCells = summary.total_cells;
			resultColumns = page.columns;
			resultRows = page.data;
		} catch {
			// No results yet
		} finally {
			loadingResults = false;
		}
	}

	function handleReset() {
		resetNellieState();
		activeCondition = '';
		previewIndex = 0;
		summaryData = null;
		resultRows = [];
		prevCurrent = -1;
		prevCondition = '';
		prevImageSet = '';
	}

	function prevPreview() { if (previewIndex > 0) previewIndex--; }
	function nextPreview() { if (previewIndex < filteredImages.length - 1) previewIndex++; }

	function prettyCol(col: string): string {
		return col
			.replace(/^nellie_org_/, '')
			.replace(/^nellie_branch_/, 'branch/')
			.replace(/^nellie_img_/, 'img/')
			.replace(/_mean$/, ' (mean)')
			.replace(/_std$/, ' (std)')
			.replace(/_sum$/, ' (sum)')
			.replace(/_/g, ' ');
	}

	let summaryNellieKeys = $derived(
		summaryData && summaryData.length > 0
			? Object.keys(summaryData[0]).filter(
					(k) => k !== 'condition' && k !== 'n_cells' && !k.endsWith('_std')
				)
			: []
	);

	onMount(() => {
		if ($nellieRunning && $nellieTaskId && $sessionId) {
			connectWebSocket();
		}
	});

	onDestroy(() => {
		disconnectWebSocket();
	});
</script>

<div class="page-nellie">
	<div class="two-col">
		<!-- Settings Panel -->
		<section class="panel">
			<h2 class="section-header">Nellie Organelle Analysis</h2>

			<div class="form-grid">
				<!-- Channel selection -->
				<div class="form-field">
					<label class="field-label font-ui" for="channel-suffix">Organelle Channel</label>
					{#if availableChannels.length > 0}
						<select id="channel-suffix" class="field-input font-ui" bind:value={channelSuffix}>
							<option value="">Auto (first available)</option>
							{#each availableChannels as ch}
								<option value={ch}>{ch}</option>
							{/each}
						</select>
					{:else}
						<input
							id="channel-suffix"
							type="text"
							class="field-input font-mono"
							bind:value={channelSuffix}
							placeholder="e.g. C1, mito, GFP"
						/>
					{/if}
					<span class="field-hint font-ui">Channel containing the organelle signal</span>
				</div>

				<!-- Hierarchy levels -->
				<div class="form-field">
					<span class="field-label font-ui">Feature Levels</span>
					<div class="checkbox-group">
						<label class="toggle-row font-ui">
							<input type="checkbox" bind:checked={includeOrganelle} />
							<span>Organelle-level (area, shape, solidity per organelle)</span>
						</label>
						<label class="toggle-row font-ui">
							<input type="checkbox" bind:checked={includeBranch} />
							<span>Branch-level (network connectivity, skeleton metrics)</span>
						</label>
					</div>
				</div>

				<!-- Device -->
				<div class="form-field">
					<label class="field-label font-ui" for="device">Compute Device</label>
					<select id="device" class="field-input font-ui" bind:value={device}>
						<option value="auto">Auto (GPU if available, else CPU)</option>
						<option value="cpu">CPU</option>
						<option value="gpu">GPU (CUDA)</option>
					</select>
				</div>

				<!-- Advanced options -->
				<div class="collapsible">
					<button
						class="collapsible-header font-ui"
						onclick={() => (advancedExpanded = !advancedExpanded)}
					>
						<Info size={14} />
						<span>Advanced Options</span>
						{#if advancedExpanded}<ChevronDown size={14} />{:else}<ChevronRight size={14} />{/if}
					</button>
					{#if advancedExpanded}
						<div class="collapsible-body">
							<label class="toggle-row font-ui">
								<input type="checkbox" bind:checked={removeEdges} />
								<span>Remove edge organelles</span>
							</label>
							<label class="toggle-row font-ui">
								<input type="checkbox" bind:checked={otsuThresh} />
								<span>Use Otsu thresholding (instead of adaptive)</span>
							</label>
							<label class="toggle-row font-ui">
								<input type="checkbox" bind:checked={lowMemory} />
								<span>Low-memory mode (slower, use for large images)</span>
							</label>
						</div>
					{/if}
				</div>

				<!-- Pixel size overrides -->
				<div class="collapsible">
					<button
						class="collapsible-header font-ui"
						onclick={() => (pixelExpanded = !pixelExpanded)}
					>
						<Info size={14} />
						<span>Pixel Size Override</span>
						{#if pixelSizeXY || pixelSizeZ || timeInterval}
							<span class="badge font-mono">Set</span>
						{/if}
						{#if pixelExpanded}<ChevronDown size={14} />{:else}<ChevronRight size={14} />{/if}
					</button>
					{#if pixelExpanded}
						<div class="collapsible-body">
							<p class="pp-hint font-ui">
								Leave blank to auto-detect from OME-TIFF / ND2 metadata. Required for plain TIFFs.
							</p>
							<div class="form-field">
								<label class="field-label font-ui" for="pxy">XY pixel size (µm)</label>
								<input id="pxy" type="number" class="field-input font-mono"
									bind:value={pixelSizeXY} placeholder="e.g. 0.108" step="0.001" min="0" />
							</div>
							<div class="form-field">
								<label class="field-label font-ui" for="pz">Z step size (µm, 3D only)</label>
								<input id="pz" type="number" class="field-input font-mono"
									bind:value={pixelSizeZ} placeholder="e.g. 0.3" step="0.01" min="0" />
							</div>
							<div class="form-field">
								<label class="field-label font-ui" for="ti">Time interval (s, timelapse only)</label>
								<input id="ti" type="number" class="field-input font-mono"
									bind:value={timeInterval} placeholder="e.g. 30" step="1" min="0" />
							</div>
						</div>
					{/if}
				</div>

				<!-- Info card -->
				<div class="info-card">
					<h4 class="info-title font-ui">Nellie Pipeline</h4>
					<p class="info-desc font-ui">
						Full nellie pipeline (Nature Methods 2025): adaptive filtering → instance labelling →
						network skeleton → motion capture → Hu-moment tracking → hierarchical feature extraction.
						Features are aggregated per cell and merged into the results table alongside CTCF data.
					</p>
					<p class="info-desc font-ui" style="margin-top:6px; color: var(--accent);">
						Supports 2D, 3D, timelapse — mitochondria, ER, and other network organelles.
					</p>
				</div>
			</div>

			<div class="action-row">
				<button
					class="btn btn-primary font-ui"
					onclick={handleRun}
					disabled={$nellieRunning || !$sessionId}
				>
					<Play size={16} />
					Run Nellie
				</button>
				{#if !$nellieRunning && $nellieWsStatus !== 'pending'}
					<button class="btn btn-secondary font-ui" onclick={handleReset}>
						<RotateCcw size={16} />
						Reset
					</button>
				{/if}
				{#if !$nellieRunning && $nellieWsStatus === 'pending' && !summaryData}
					<button class="btn btn-secondary font-ui" onclick={loadResults} disabled={loadingResults}>
						Load Previous
					</button>
				{/if}
			</div>
		</section>

		<!-- Status + Preview Panel -->
		<section class="panel preview-panel">
			<h2 class="section-header">Analysis Status</h2>

			{#if $nellieTaskId}
				<TaskStatus
					taskId={$nellieTaskId}
					status={$nellieWsStatus}
					progress={$nellieProgress}
					message={$nellieMessage}
					elapsed={$nellieElapsed}
					result={$nellieResult}
				/>

				<!-- Image preview carousel -->
				{#if $nellieCompletedImages.length > 0}
					<div class="seg-preview">
						<h3 class="seg-preview-header font-ui">
							Cell Preview
							<span class="seg-preview-count font-mono">{$nellieCompletedImages.length} processed</span>
							<span class="style-toggle">
								<button class="style-btn font-ui" class:active={overlayStyle === 'filled'}
									onclick={() => { overlayStyle = 'filled'; }}>Filled</button>
								<button class="style-btn font-ui" class:active={overlayStyle === 'outline'}
									onclick={() => { overlayStyle = 'outline'; }}>Outline</button>
							</span>
						</h3>

						<!-- Condition tabs -->
						{#if previewConditions.length > 1}
							<div class="preview-tabs">
								{#each previewConditions as cName}
									<button class="preview-tab font-ui" class:active={activeCondition === cName}
										onclick={() => { activeCondition = cName; previewIndex = 0; }}>
										{cName}
									</button>
								{/each}
							</div>
						{/if}

						<!-- Preview frame -->
						<div class="preview-frame">
							{#if previewSrc}
								{#key previewSrc}
									<img src={previewSrc} alt="{currentPreview?.condition}/{currentPreview?.baseName}" class="preview-img" />
								{/key}
							{/if}
						</div>

						<!-- Navigation -->
						<div class="preview-nav">
							<button class="icon-btn" onclick={prevPreview} disabled={previewIndex === 0}>
								<ChevronLeft size={18} />
							</button>
							<span class="preview-label font-ui">
								{currentPreview?.condition} / {currentPreview?.baseName}
								<span class="preview-count font-mono">({previewIndex + 1}/{filteredImages.length})</span>
							</span>
							<button class="icon-btn" onclick={nextPreview} disabled={previewIndex >= filteredImages.length - 1}>
								<ChevronRight size={18} />
							</button>
							<button
								class="icon-btn folder-btn"
								title="Open result folder"
								onclick={() => {
									if (currentPreview && $sessionId) {
										openResultFolder($sessionId, currentPreview.condition, currentPreview.baseName);
									}
								}}
								disabled={!currentPreview}
							>
								<FolderOpen size={16} />
							</button>
						</div>
					</div>
				{/if}

				<!-- Terminal log -->
				<details class="log-panel">
					<summary class="log-summary font-ui">
						Terminal Output
						{#if $nellieLogs.length > 0}
							<span class="log-count">{$nellieLogs.length} lines</span>
						{/if}
					</summary>
					<pre class="log-output font-mono" bind:this={logPre}>{$nellieLogs.length > 0 ? $nellieLogs.join('\n') : 'Waiting for output...'}</pre>
				</details>

			{:else}
				<div class="placeholder font-ui">
					<Activity size={48} strokeWidth={1} />
					<p>Configure and run Nellie organelle analysis</p>
					<p class="hint">Results merge into the Results page alongside CTCF data</p>
				</div>
			{/if}
		</section>
	</div>

	<!-- Summary table -->
	{#if summaryData && summaryData.length > 0}
		<section class="panel summary-panel">
			<h2 class="section-header">Per-Condition Summary ({totalCells} cells)</h2>
			<div class="table-wrap">
				<table class="summary-table font-mono">
					<thead>
						<tr>
							<th>Condition</th>
							<th>Cells</th>
							<th>Organelles / cell</th>
							{#each summaryNellieKeys.slice(2) as key}
								<th>{prettyCol(key)}</th>
							{/each}
						</tr>
					</thead>
					<tbody>
						{#each summaryData as row}
							<tr>
								<td class="font-ui">{row.condition}</td>
								<td>{row.n_cells}</td>
								<td>{typeof row.mean_n_organelles === 'number' ? (row.mean_n_organelles as number).toFixed(1) : '—'}</td>
								{#each summaryNellieKeys.slice(2) as key}
									<td>{typeof row[key] === 'number' ? (row[key] as number).toFixed(3) : '—'}</td>
								{/each}
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
		</section>
	{/if}

	<!-- Per-cell results preview -->
	{#if resultRows.length > 0}
		<section class="panel">
			<h2 class="section-header">Per-Cell Features (first 500 rows)</h2>
			<div class="table-wrap">
				<table class="results-table font-mono">
					<thead>
						<tr>
							{#each resultColumns.slice(0, 12) as col}
								<th title={col}>{prettyCol(col)}</th>
							{/each}
							{#if resultColumns.length > 12}
								<th>+{resultColumns.length - 12} more</th>
							{/if}
						</tr>
					</thead>
					<tbody>
						{#each resultRows.slice(0, 50) as row}
							<tr>
								{#each resultColumns.slice(0, 12) as col}
									<td>
										{#if typeof row[col] === 'number'}
											{(row[col] as number).toFixed(3)}
										{:else}
											{row[col] ?? '—'}
										{/if}
									</td>
								{/each}
								{#if resultColumns.length > 12}
									<td class="text-muted">…</td>
								{/if}
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
			{#if resultRows.length > 50}
				<p class="table-hint font-ui">Showing 50 of {resultRows.length} rows. Export from Results page for full data.</p>
			{/if}
		</section>
	{/if}
</div>

<style>
	.page-nellie {
		max-width: 1200px;
		display: flex;
		flex-direction: column;
		gap: 24px;
	}

	.two-col {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 24px;
		align-items: start;
	}

	.panel {
		background: var(--bg-elevated);
		border: 1px solid var(--border);
		border-radius: var(--radius-lg);
		padding: 24px;
		box-shadow: var(--shadow-card);
		transition: var(--transition-theme);
	}

	:global(.dark) .panel {
		box-shadow: none;
	}

	.preview-panel {
		display: flex;
		flex-direction: column;
		gap: 16px;
	}

	.summary-panel { overflow: auto; }

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

	.field-hint { font-size: 11px; color: var(--text-faint); }

	.checkbox-group {
		display: flex;
		flex-direction: column;
		gap: 8px;
	}

	.toggle-row {
		display: flex;
		align-items: center;
		gap: 8px;
		font-size: 12px;
		color: var(--text);
		cursor: pointer;
	}

	.toggle-row input[type='checkbox'] { accent-color: var(--accent); }

	.collapsible {
		border: 1px solid var(--border);
		border-radius: var(--radius-md);
		overflow: hidden;
	}

	.collapsible-header {
		display: flex;
		align-items: center;
		gap: 8px;
		width: 100%;
		padding: 10px 14px;
		background: var(--bg);
		border: none;
		color: var(--text);
		font-size: 12px;
		font-weight: 600;
		cursor: pointer;
		transition: background 0.15s;
	}

	.collapsible-header:hover { background: var(--accent-soft); }

	.collapsible-header .badge {
		margin-left: auto;
		margin-right: 4px;
		padding: 2px 8px;
		background: var(--accent);
		color: white;
		border-radius: var(--radius-pill);
		font-size: 10px;
		font-weight: 500;
	}

	:global(.dark) .collapsible-header .badge { color: #000; }

	.collapsible-body {
		padding: 14px;
		display: flex;
		flex-direction: column;
		gap: 12px;
		border-top: 1px solid var(--border);
	}

	.pp-hint {
		font-size: 11px;
		color: var(--text-muted);
		margin: 0;
		line-height: 1.5;
	}

	.info-card {
		background: var(--bg);
		border: 1px solid var(--border);
		border-radius: var(--radius-md);
		padding: 14px;
	}

	.info-title {
		font-size: 11px;
		font-weight: 600;
		color: var(--accent);
		margin: 0 0 8px 0;
		text-transform: uppercase;
		letter-spacing: 0.04em;
	}

	.info-desc {
		font-size: 12px;
		color: var(--text-muted);
		margin: 0;
		line-height: 1.5;
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

	.btn:disabled { opacity: 0.5; cursor: not-allowed; }

	.btn-primary { background: var(--accent); color: white; }
	:global(.dark) .btn-primary { color: #000; }
	.btn-primary:hover:not(:disabled) { filter: brightness(1.1); transform: translateY(-1px); }

	.btn-secondary {
		background: var(--bg);
		color: var(--text);
		border: 1px solid var(--border);
	}

	.btn-secondary:hover { border-color: var(--accent); color: var(--accent); }

	.placeholder {
		text-align: center;
		color: var(--text-faint);
		padding: 40px 0;
	}

	.placeholder p { margin-top: 12px; font-size: 13px; }
	.placeholder .hint { font-size: 11px; margin-top: 4px; }

	/* ── Image preview (same as segmentation tab) ── */
	.seg-preview {
		display: flex;
		flex-direction: column;
		gap: 10px;
		border: 1px solid var(--border);
		border-radius: var(--radius-md);
		padding: 14px;
		background: var(--bg);
	}

	.seg-preview-header {
		display: flex;
		align-items: center;
		gap: 10px;
		font-size: 12px;
		font-weight: 600;
		color: var(--text);
		margin: 0;
		flex-wrap: wrap;
	}

	.seg-preview-count {
		font-size: 11px;
		color: var(--text-faint);
		font-weight: 400;
	}

	.style-toggle {
		display: inline-flex;
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		overflow: hidden;
		margin-left: auto;
	}

	.style-btn {
		padding: 3px 10px;
		font-size: 11px;
		background: none;
		border: none;
		color: var(--text-muted);
		cursor: pointer;
		transition: background 0.1s, color 0.1s;
	}

	.style-btn.active {
		background: var(--accent);
		color: white;
	}

	:global(.dark) .style-btn.active { color: #000; }

	.preview-tabs {
		display: flex;
		gap: 4px;
		flex-wrap: wrap;
	}

	.preview-tab {
		padding: 4px 12px;
		font-size: 11px;
		background: var(--bg-elevated);
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		color: var(--text-muted);
		cursor: pointer;
		transition: all 0.15s;
	}

	.preview-tab.active {
		background: var(--accent-soft);
		color: var(--accent);
		border-color: var(--accent);
	}

	.preview-frame {
		width: 100%;
		aspect-ratio: 1;
		background: #111;
		border-radius: var(--radius-sm);
		overflow: hidden;
		display: flex;
		align-items: center;
		justify-content: center;
	}

	.preview-img {
		width: 100%;
		height: 100%;
		object-fit: contain;
	}

	.preview-nav {
		display: flex;
		align-items: center;
		gap: 8px;
	}

	.icon-btn {
		display: flex;
		align-items: center;
		justify-content: center;
		width: 30px;
		height: 30px;
		border-radius: var(--radius-sm);
		background: var(--bg-elevated);
		border: 1px solid var(--border);
		color: var(--text-muted);
		cursor: pointer;
		transition: all 0.15s;
		flex-shrink: 0;
	}

	.icon-btn:hover:not(:disabled) { border-color: var(--accent); color: var(--accent); }
	.icon-btn:disabled { opacity: 0.35; cursor: not-allowed; }

	.folder-btn { margin-left: auto; }

	.preview-label {
		flex: 1;
		font-size: 11px;
		color: var(--text-muted);
		text-align: center;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	.preview-count { color: var(--text-faint); }

	/* ── Terminal log (same as segmentation tab) ── */
	.log-panel {
		border: 1px solid var(--border);
		border-radius: var(--radius-md);
		overflow: hidden;
	}

	.log-summary {
		padding: 8px 14px;
		font-size: 12px;
		font-weight: 600;
		color: var(--text-muted);
		cursor: pointer;
		background: var(--bg);
		display: flex;
		align-items: center;
		gap: 8px;
		list-style: none;
		transition: background 0.15s;
	}

	.log-summary:hover { background: var(--accent-soft); color: var(--text); }

	.log-count {
		font-size: 10px;
		padding: 1px 6px;
		background: var(--border);
		border-radius: var(--radius-pill);
		color: var(--text-faint);
	}

	.log-output {
		margin: 0;
		padding: 12px 14px;
		font-size: 11px;
		line-height: 1.6;
		color: var(--text-muted);
		background: var(--bg);
		border-top: 1px solid var(--border);
		max-height: 260px;
		overflow-y: auto;
		white-space: pre-wrap;
		word-break: break-all;
	}

	/* ── Tables ── */
	.table-wrap { overflow-x: auto; }

	.summary-table, .results-table {
		width: 100%;
		border-collapse: collapse;
		font-size: 12px;
	}

	.summary-table th, .results-table th {
		text-align: left;
		padding: 8px 10px;
		font-size: 10px;
		font-weight: 600;
		color: var(--text-muted);
		text-transform: uppercase;
		letter-spacing: 0.04em;
		border-bottom: 1px solid var(--border);
		white-space: nowrap;
	}

	.summary-table td, .results-table td {
		padding: 7px 10px;
		border-bottom: 1px solid var(--border);
		color: var(--text);
		font-size: 12px;
		white-space: nowrap;
	}

	.summary-table tbody tr:hover,
	.results-table tbody tr:hover { background: var(--bg-hover); }

	.text-muted { color: var(--text-faint); }

	.table-hint {
		font-size: 11px;
		color: var(--text-faint);
		margin-top: 10px;
	}

	@media (max-width: 800px) {
		.two-col { grid-template-columns: 1fr; }
	}
</style>
