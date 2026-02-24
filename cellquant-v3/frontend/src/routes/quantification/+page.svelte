<script lang="ts">
	import { onDestroy } from 'svelte';
	import { Calculator, Play, RotateCcw, ChevronDown, ChevronRight, Filter, Beaker } from 'lucide-svelte';
	import { runQuantification, configurePreprocessing, browseFolder } from '$api/client';
	import { ProgressSocket } from '$api/websocket';
	import type { ProgressMessage, QCFilterParams } from '$api/types';
	import { sessionId } from '$stores/session';
	import { detection } from '$stores/experiment';
	import { quantTaskId } from '$stores/quantification';
	import TaskStatus from '$components/progress/TaskStatus.svelte';

	let bgMethod = $state('auto');
	let running = $state(false);
	let wsProgress = $state(0);
	let wsMessage = $state('');
	let wsStatus = $state('pending');
	let wsElapsed = $state(0);
	let wsResult = $state<Record<string, unknown> | null>(null);
	let socket: ProgressSocket | null = null;

	// QC filter state
	let qcExpanded = $state(false);
	let qcEnabled = $state(true);
	let removeBorder = $state(true);
	// Morphological filters are opt-in (off by default)
	let filterSolidity = $state(false);
	let minSolidity = $state(0.80);
	let filterEccentricity = $state(false);
	let maxEccentricity = $state(0.90);
	let filterCircularity = $state(false);
	let minCircularity = $state(0.40);
	let filterAspectRatio = $state(false);
	let maxAspectRatio = $state(3.0);
	let filterArea = $state(false);
	let areaIqrFactor = $state(1.5);
	let outlierThreshold = $state(3.5);

	// Preprocessing state
	let ppExpanded = $state(false);
	let hasDark = $state(false);
	let hasFlat = $state(false);
	let ppWarnings = $state<string[]>([]);

	// Negative control / manual BG
	let negControlPath = $state('');
	let manualBgValue = $state<string>('');

	let markerSuffixes = $derived($detection?.suggested_markers ?? []);
	let markerNames = $derived(markerSuffixes.map((s: string) => s));

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
		if (msg.task_id && msg.task_id !== $quantTaskId) return;

		if (msg.type === 'progress') {
			wsProgress = msg.progress ?? 0;
			wsMessage = msg.message ?? '';
			wsStatus = 'running';
			wsElapsed = msg.elapsed_seconds ?? 0;
		} else if (msg.type === 'task_complete') {
			wsProgress = 100;
			wsStatus = msg.status ?? 'complete';
			wsMessage = msg.message ?? '';
			wsElapsed = msg.elapsed_seconds ?? 0;
			wsResult = (msg.data as Record<string, unknown>) ?? null;
			running = false;
			disconnectWebSocket();
		}
	}

	async function handleRun() {
		if (!$sessionId) return;
		running = true;
		wsProgress = 0;
		wsMessage = 'Submitting...';
		wsStatus = 'pending';
		wsResult = null;

		connectWebSocket();

		const qcFilters: QCFilterParams = {
			enabled: qcEnabled,
			remove_border_objects: removeBorder,
			min_area: null,
			max_area: null,
			area_iqr_factor: 0,
			min_solidity: null,
			max_eccentricity: null,
			min_circularity: null,
			max_aspect_ratio: null,
		};
		if (filterArea) qcFilters.area_iqr_factor = areaIqrFactor;
		if (filterSolidity) qcFilters.min_solidity = minSolidity;
		if (filterEccentricity) qcFilters.max_eccentricity = maxEccentricity;
		if (filterCircularity) qcFilters.min_circularity = minCircularity;
		if (filterAspectRatio) qcFilters.max_aspect_ratio = maxAspectRatio;
		console.log('[CellQuant] FRONTEND qcFilters:', JSON.stringify(qcFilters));

		try {
			const { task_id } = await runQuantification($sessionId, {
				background_method: bgMethod,
				marker_suffixes: markerSuffixes,
				marker_names: markerNames,
				mitochondrial_markers: [],
				qc_filters: qcFilters,
				negative_control_path: negControlPath || null,
				manual_background_value: manualBgValue ? parseFloat(manualBgValue) : null,
				outlier_threshold: outlierThreshold,
			});
			$quantTaskId = task_id;
			wsStatus = 'running';
		} catch (e) {
			running = false;
			wsStatus = 'error';
			wsMessage = e instanceof Error ? e.message : 'Failed to start quantification';
			disconnectWebSocket();
		}
	}

	function handleReset() {
		$quantTaskId = null;
		wsProgress = 0;
		wsMessage = '';
		wsStatus = 'pending';
		wsResult = null;
	}

	async function browseDark() {
		const path = await browseFolder();
		if (path && $sessionId) {
			const res = await configurePreprocessing($sessionId, [path], []);
			hasDark = res.has_dark;
			ppWarnings = res.warnings;
		}
	}

	async function browseFlat() {
		const path = await browseFolder();
		if (path && $sessionId) {
			const res = await configurePreprocessing($sessionId, [], [path]);
			hasFlat = res.has_flat;
			ppWarnings = res.warnings;
		}
	}

	onDestroy(() => {
		disconnectWebSocket();
	});
</script>

<div class="page-quantification">
	<div class="two-col">
		<!-- Settings Panel -->
		<section class="panel">
			<h2 class="section-header">CTCF Quantification</h2>

			<div class="form-grid">
				<!-- Background Method -->
				<div class="form-field">
					<label class="field-label font-ui" for="bg-method">Background Method</label>
					<select id="bg-method" class="field-input font-ui" bind:value={bgMethod}>
						<optgroup label="Automatic">
							<option value="auto">Auto-Detect (Recommended)</option>
						</optgroup>
						<optgroup label="Per-Cell (Local)">
							<option value="annular_ring">Annular Ring</option>
							<option value="masked_annular_ring">Masked Annular Ring</option>
							<option value="voronoi">Voronoi Partition</option>
						</optgroup>
						<optgroup label="Surface Estimation">
							<option value="rolling_ball">Rolling Ball</option>
							<option value="white_tophat">White Top-Hat</option>
							<option value="polynomial_surface">Polynomial Surface</option>
						</optgroup>
						<optgroup label="Global">
							<option value="global_roi">Global ROI (Median)</option>
						</optgroup>
					</select>
				</div>

				{#if markerSuffixes.length > 0}
					<div class="form-field">
						<span class="field-label font-ui">Marker Channels</span>
						<div class="marker-list font-mono">
							{#each markerSuffixes as suffix}
								<span class="marker-tag">{suffix}</span>
							{/each}
						</div>
					</div>
				{/if}

				<!-- Outlier Threshold -->
				<div class="form-field">
					<label class="field-label font-ui" for="outlier-thresh">
						Outlier Threshold (MAD)
					</label>
					<input
						id="outlier-thresh"
						type="number"
						class="field-input font-mono"
						bind:value={outlierThreshold}
						min={1}
						max={10}
						step={0.5}
					/>
					<span class="field-hint font-ui">Modified Z-score cutoff (3.5 recommended)</span>
				</div>

				<!-- Negative Control / Manual BG -->
				<div class="form-field">
					<label class="field-label font-ui" for="neg-control">
						Negative Control Path (optional)
					</label>
					<input
						id="neg-control"
						type="text"
						class="field-input font-mono"
						bind:value={negControlPath}
						placeholder="Path to negative control image..."
					/>
				</div>

				<div class="form-field">
					<label class="field-label font-ui" for="manual-bg">
						Manual Background Value (optional)
					</label>
					<input
						id="manual-bg"
						type="text"
						class="field-input font-mono"
						bind:value={manualBgValue}
						placeholder="e.g. 120.5"
					/>
				</div>

				<!-- QC Filters Collapsible -->
				<div class="collapsible">
					<button
						class="collapsible-header font-ui"
						onclick={() => qcExpanded = !qcExpanded}
					>
						<Filter size={14} />
						<span>QC Filters</span>
						{#if qcExpanded}
							<ChevronDown size={14} />
						{:else}
							<ChevronRight size={14} />
						{/if}
					</button>

					{#if qcExpanded}
						<div class="collapsible-body">
							<label class="toggle-row font-ui">
								<input type="checkbox" bind:checked={qcEnabled} />
								<span>Enable QC Filtering</span>
							</label>

							{#if qcEnabled}
								<label class="toggle-row font-ui">
									<input type="checkbox" bind:checked={removeBorder} />
									<span>Remove Border Objects</span>
								</label>

								<div class="filter-group">
									<label class="toggle-row font-ui">
										<input type="checkbox" bind:checked={filterArea} />
										<span>Area Filter (IQR outlier fences)</span>
									</label>
									{#if filterArea}
										<div class="slider-field nested">
											<label class="slider-label font-ui">
												IQR Factor
												<span class="slider-value font-mono">{areaIqrFactor.toFixed(1)}</span>
											</label>
											<input type="range" min={0.5} max={5} step={0.5} bind:value={areaIqrFactor} />
											<span class="field-hint font-ui">Higher = more permissive (1.5 = standard, 3.0 = lenient)</span>
										</div>
									{/if}
								</div>

								<div class="filter-group">
									<label class="toggle-row font-ui">
										<input type="checkbox" bind:checked={filterSolidity} />
										<span>Solidity Filter</span>
									</label>
									{#if filterSolidity}
										<div class="slider-field nested">
											<label class="slider-label font-ui">
												Min Solidity
												<span class="slider-value font-mono">{minSolidity.toFixed(2)}</span>
											</label>
											<input type="range" min={0} max={1} step={0.05} bind:value={minSolidity} />
											<span class="field-hint font-ui">Rejects cells with concavities (low = permissive)</span>
										</div>
									{/if}
								</div>

								<div class="filter-group">
									<label class="toggle-row font-ui">
										<input type="checkbox" bind:checked={filterEccentricity} />
										<span>Eccentricity Filter</span>
									</label>
									{#if filterEccentricity}
										<div class="slider-field nested">
											<label class="slider-label font-ui">
												Max Eccentricity
												<span class="slider-value font-mono">{maxEccentricity.toFixed(2)}</span>
											</label>
											<input type="range" min={0} max={1} step={0.05} bind:value={maxEccentricity} />
											<span class="field-hint font-ui">Rejects elongated cells (high = permissive)</span>
										</div>
									{/if}
								</div>

								<div class="filter-group">
									<label class="toggle-row font-ui">
										<input type="checkbox" bind:checked={filterCircularity} />
										<span>Circularity Filter</span>
									</label>
									{#if filterCircularity}
										<div class="slider-field nested">
											<label class="slider-label font-ui">
												Min Circularity
												<span class="slider-value font-mono">{minCircularity.toFixed(2)}</span>
											</label>
											<input type="range" min={0} max={1} step={0.05} bind:value={minCircularity} />
											<span class="field-hint font-ui">Rejects irregular shapes (low = permissive)</span>
										</div>
									{/if}
								</div>

								<div class="filter-group">
									<label class="toggle-row font-ui">
										<input type="checkbox" bind:checked={filterAspectRatio} />
										<span>Aspect Ratio Filter</span>
									</label>
									{#if filterAspectRatio}
										<div class="slider-field nested">
											<label class="slider-label font-ui">
												Max Aspect Ratio
												<span class="slider-value font-mono">{maxAspectRatio.toFixed(1)}</span>
											</label>
											<input type="range" min={1} max={10} step={0.5} bind:value={maxAspectRatio} />
											<span class="field-hint font-ui">Rejects very elongated cells (high = permissive)</span>
										</div>
									{/if}
								</div>
							{/if}
						</div>
					{/if}
				</div>

				<!-- Preprocessing Collapsible -->
				<div class="collapsible">
					<button
						class="collapsible-header font-ui"
						onclick={() => ppExpanded = !ppExpanded}
					>
						<Beaker size={14} />
						<span>Preprocessing</span>
						{#if hasDark || hasFlat}
							<span class="badge font-mono">Active</span>
						{/if}
						{#if ppExpanded}
							<ChevronDown size={14} />
						{:else}
							<ChevronRight size={14} />
						{/if}
					</button>

					{#if ppExpanded}
						<div class="collapsible-body">
							<p class="pp-hint font-ui">
								Optional dark-frame subtraction and flat-field correction applied before quantification.
							</p>

							<div class="pp-row">
								<span class="pp-label font-ui">Dark Frame:</span>
								<button class="btn btn-secondary btn-sm font-ui" onclick={browseDark}>
									Browse...
								</button>
								{#if hasDark}
									<span class="pp-status font-mono">Loaded</span>
								{/if}
							</div>

							<div class="pp-row">
								<span class="pp-label font-ui">Flat Field:</span>
								<button class="btn btn-secondary btn-sm font-ui" onclick={browseFlat}>
									Browse...
								</button>
								{#if hasFlat}
									<span class="pp-status font-mono">Loaded</span>
								{/if}
							</div>

							{#if ppWarnings.length > 0}
								<div class="pp-warnings font-ui">
									{#each ppWarnings as w}
										<p class="warning-text">{w}</p>
									{/each}
								</div>
							{/if}
						</div>
					{/if}
				</div>

				<!-- Info Card -->
				<div class="info-card">
					<h4 class="info-title font-ui">Enhanced CTCF Pipeline</h4>
					<p class="info-formula font-mono">
						CTCF = IntDen - (Area x BG)
					</p>
					<p class="info-desc font-ui">
						Full pipeline: preprocessing, 7 background methods with auto-detection,
						morphological QC filtering, error propagation, and MAD outlier detection.
					</p>
				</div>
			</div>

			<div class="action-row">
				<button
					class="btn btn-primary font-ui"
					onclick={handleRun}
					disabled={running || !$sessionId}
				>
					<Play size={16} />
					Run Quantification
				</button>
				{#if !running && wsStatus !== 'pending'}
					<button class="btn btn-secondary font-ui" onclick={handleReset}>
						<RotateCcw size={16} />
						Reset
					</button>
				{/if}
			</div>
		</section>

		<!-- Status Panel -->
		<section class="panel">
			<h2 class="section-header">Status</h2>

			{#if $quantTaskId}
				<TaskStatus
					taskId={$quantTaskId}
					status={wsStatus}
					progress={wsProgress}
					message={wsMessage}
					elapsed={wsElapsed}
					result={wsResult}
				/>
			{:else}
				<div class="placeholder font-ui">
					<Calculator size={48} strokeWidth={1} />
					<p>Configure settings and run quantification</p>
					<p class="hint">Results will appear on the Results page</p>
				</div>
			{/if}
		</section>
	</div>
</div>

<style>
	.page-quantification {
		max-width: 1000px;
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

	.field-hint {
		font-size: 11px;
		color: var(--text-faint);
	}

	.marker-list {
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
	}

	.marker-tag {
		font-size: 11px;
		padding: 4px 10px;
		background: var(--accent-soft);
		color: var(--accent);
		border-radius: var(--radius-pill);
	}

	/* Collapsible sections */
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

	.collapsible-header:hover {
		background: var(--accent-soft);
	}

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

	:global(.dark) .collapsible-header .badge {
		color: #000;
	}

	.collapsible-body {
		padding: 14px;
		display: flex;
		flex-direction: column;
		gap: 12px;
		border-top: 1px solid var(--border);
	}

	.toggle-row {
		display: flex;
		align-items: center;
		gap: 8px;
		font-size: 12px;
		color: var(--text);
		cursor: pointer;
	}

	.toggle-row input[type="checkbox"] {
		accent-color: var(--accent);
	}

	.filter-group {
		display: flex;
		flex-direction: column;
		gap: 6px;
	}

	.slider-field.nested {
		padding-left: 26px;
	}

	.slider-field {
		display: flex;
		flex-direction: column;
		gap: 4px;
	}

	.slider-label {
		display: flex;
		justify-content: space-between;
		font-size: 11px;
		color: var(--text-muted);
	}

	.slider-value {
		color: var(--accent);
		font-size: 11px;
	}

	.slider-field input[type="range"] {
		width: 100%;
		accent-color: var(--accent);
	}

	/* Preprocessing */
	.pp-hint {
		font-size: 11px;
		color: var(--text-muted);
		margin: 0;
		line-height: 1.5;
	}

	.pp-row {
		display: flex;
		align-items: center;
		gap: 10px;
	}

	.pp-label {
		font-size: 12px;
		color: var(--text);
		min-width: 80px;
	}

	.pp-status {
		font-size: 11px;
		color: var(--accent);
	}

	.pp-warnings {
		padding: 8px;
		background: rgba(255, 100, 0, 0.1);
		border-radius: var(--radius-sm);
	}

	.warning-text {
		font-size: 11px;
		color: #ff6400;
		margin: 0;
	}

	.btn-sm {
		padding: 5px 12px;
		font-size: 11px;
	}

	/* Info card */
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

	.info-formula {
		font-size: 14px;
		color: var(--text);
		margin: 0 0 8px 0;
		padding: 6px 0;
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

	.placeholder {
		text-align: center;
		color: var(--text-faint);
		padding: 40px 0;
	}

	.placeholder p {
		margin-top: 12px;
		font-size: 13px;
	}

	.placeholder .hint {
		font-size: 11px;
		margin-top: 4px;
	}

	@media (max-width: 800px) {
		.two-col {
			grid-template-columns: 1fr;
		}
	}
</style>
