<script lang="ts">
	import { onMount } from 'svelte';
	import { BarChart3, FileSpreadsheet, FileText, AlertTriangle } from 'lucide-svelte';
	import { getResultsPage, getResultsSummary, getQCSummary, exportCsv, exportExcel } from '$api/client';
	import { sessionId } from '$stores/session';
	import { resultsPage, quantSummary, qcSummary } from '$stores/quantification';

	type TabId = 'table' | 'boxplot' | 'scatter' | 'histogram' | 'spatial' | 'qcsummary' | 'perfov';
	let activeTab = $state<TabId>('table');
	let loading = $state(false);
	let showFlagged = $state(true);

	let columns = $derived($resultsPage?.columns ?? []);
	let rows = $derived($resultsPage?.data ?? []);
	let currentPage = $derived($resultsPage?.page ?? 0);
	let totalPages = $derived($resultsPage?.total_pages ?? 0);
	let totalRows = $derived($resultsPage?.total_rows ?? 0);

	// Detect quality flag columns
	let flagColumns = $derived(columns.filter(c =>
		c.startsWith('is_outlier_') || c === 'is_saturated' || c === 'is_dim' || c === 'low_confidence_background'
	));
	let hasFlaggedRows = $derived(rows.some(r =>
		flagColumns.some(c => r[c] === true)
	));

	const tabs: { id: TabId; label: string }[] = [
		{ id: 'table', label: 'Data Table' },
		{ id: 'boxplot', label: 'Box Plot' },
		{ id: 'scatter', label: 'Scatter' },
		{ id: 'histogram', label: 'Histogram' },
		{ id: 'spatial', label: 'Spatial Map' },
		{ id: 'qcsummary', label: 'QC Summary' },
		{ id: 'perfov', label: 'Per-FOV' }
	];

	// QC Summary data
	let summaryRows = $derived($qcSummary?.summary ?? []);
	let fovRows = $derived($qcSummary?.fov_data ?? []);

	onMount(() => {
		if ($sessionId && !$resultsPage) {
			loadPage(0);
			loadSummary();
			loadQCSummary();
		}
	});

	async function loadPage(page: number) {
		if (!$sessionId) return;
		loading = true;
		try {
			$resultsPage = await getResultsPage($sessionId, page);
		} catch {
			// No results yet
		}
		loading = false;
	}

	async function loadSummary() {
		if (!$sessionId) return;
		try {
			$quantSummary = await getResultsSummary($sessionId);
		} catch {
			// No results yet
		}
	}

	async function loadQCSummary() {
		if (!$sessionId) return;
		try {
			$qcSummary = await getQCSummary($sessionId);
		} catch {
			// No results yet
		}
	}

	function formatValue(val: unknown): string {
		if (val == null) return '--';
		if (typeof val === 'boolean') return val ? 'Yes' : 'No';
		if (typeof val === 'number') {
			if (Number.isInteger(val)) return val.toLocaleString();
			return val.toFixed(2);
		}
		return String(val);
	}

	function isFlaggedRow(row: Record<string, unknown>): boolean {
		return flagColumns.some(c => row[c] === true);
	}

	function isFlagColumn(col: string): boolean {
		return col.startsWith('is_outlier_') || col === 'is_saturated' || col === 'is_dim';
	}

	async function renderChart(tab: string) {
		if (!$resultsPage || rows.length === 0) return;

		const Plotly = await import('plotly.js-dist-min');
		const themeColors = getComputedStyle(document.documentElement);
		const accent = themeColors.getPropertyValue('--accent').trim();
		const bg = themeColors.getPropertyValue('--bg-elevated').trim();
		const text = themeColors.getPropertyValue('--text').trim();
		const border = themeColors.getPropertyValue('--border').trim();

		const layout: Partial<Plotly.Layout> = {
			paper_bgcolor: bg,
			plot_bgcolor: bg,
			font: { color: text, size: 11 },
			margin: { t: 40, r: 30, b: 50, l: 60 },
			xaxis: { gridcolor: border },
			yaxis: { gridcolor: border }
		};

		const ctcfCols = columns.filter((c) => c.endsWith('_CTCF'));
		const ctcfCol = ctcfCols[0] || 'CTCF';

		if (tab === 'boxplot') {
			const condCol = columns.includes('Condition') ? 'Condition' : columns[0];
			const conditions = [...new Set(rows.map((r) => String(r[condCol])))];
			const traces = conditions.map((cond) => ({
				y: rows.filter((r) => String(r[condCol]) === cond).map((r) => Number(r[ctcfCol]) || 0),
				type: 'box' as const,
				name: cond,
				marker: { color: accent }
			}));
			Plotly.newPlot('plotly-boxplot', traces, { ...layout, title: 'CTCF by Condition' });
		} else if (tab === 'scatter') {
			const trace = {
				x: rows.map((r) => Number(r['Area']) || 0),
				y: rows.map((r) => Number(r[ctcfCol]) || 0),
				mode: 'markers' as const,
				type: 'scatter' as const,
				marker: { color: accent, size: 4, opacity: 0.6 }
			};
			Plotly.newPlot('plotly-scatter', [trace], {
				...layout,
				title: 'Area vs CTCF',
				xaxis: { ...layout.xaxis, title: 'Area (px)' },
				yaxis: { ...layout.yaxis, title: ctcfCol }
			});
		} else if (tab === 'histogram') {
			const trace = {
				x: rows.map((r) => Number(r[ctcfCol]) || 0),
				type: 'histogram' as const,
				marker: { color: accent }
			};
			Plotly.newPlot('plotly-histogram', [trace], {
				...layout,
				title: `${ctcfCol} Distribution`,
				xaxis: { ...layout.xaxis, title: ctcfCol },
				yaxis: { ...layout.yaxis, title: 'Count' }
			});
		} else if (tab === 'spatial') {
			const hasXY = columns.includes('x_centroid') && columns.includes('y_centroid');
			if (!hasXY) return;

			const ctcfValues = rows.map(r => Number(r[ctcfCol]) || 0);
			const trace = {
				x: rows.map((r) => Number(r['x_centroid']) || 0),
				y: rows.map((r) => Number(r['y_centroid']) || 0),
				mode: 'markers' as const,
				type: 'scatter' as const,
				marker: {
					color: ctcfValues,
					colorscale: 'Viridis',
					size: 5,
					opacity: 0.8,
					colorbar: { title: ctcfCol, titlefont: { size: 11 } }
				},
				text: rows.map((r, i) => `Cell ${r['CellID']}<br>${ctcfCol}: ${ctcfValues[i].toFixed(1)}`),
				hoverinfo: 'text' as const,
			};
			Plotly.newPlot('plotly-spatial', [trace], {
				...layout,
				title: 'Spatial CTCF Map',
				xaxis: { ...layout.xaxis, title: 'X (px)', scaleanchor: 'y' },
				yaxis: { ...layout.yaxis, title: 'Y (px)', autorange: 'reversed' as const }
			});
		} else if (tab === 'perfov') {
			if (fovRows.length === 0) return;

			// Find median CTCF columns in FOV data
			const medCols = Object.keys(fovRows[0] || {}).filter(k => k.endsWith('_median_CTCF'));
			const medCol = medCols[0];
			if (!medCol) return;

			const conditions = [...new Set(fovRows.map(r => String(r['Condition'])))];
			const traces = conditions.map(cond => ({
				y: fovRows.filter(r => String(r['Condition']) === cond).map(r => Number(r[medCol]) || 0),
				type: 'box' as const,
				name: cond,
				marker: { color: accent },
				boxpoints: 'all' as const,
				jitter: 0.3,
				pointpos: -1.5,
			}));
			Plotly.newPlot('plotly-perfov', traces, {
				...layout,
				title: 'Per-FOV Median CTCF',
				yaxis: { ...layout.yaxis, title: medCol }
			});
		}
	}

	$effect(() => {
		if (activeTab !== 'table' && activeTab !== 'qcsummary' && rows.length > 0) {
			setTimeout(() => renderChart(activeTab), 50);
		}
	});

	// Compute outlier count from summary
	let outlierCount = $derived.by(() => {
		if (!$quantSummary) return 0;
		// Check if wsResult has outliers_flagged from the task
		return 0;
	});
</script>

<div class="page-results">
	<!-- Summary Cards -->
	{#if $quantSummary}
		<div class="summary-row">
			<div class="stat-card">
				<div class="stat-value font-mono">{$quantSummary.total_cells.toLocaleString()}</div>
				<div class="stat-label font-ui">Total Cells</div>
			</div>
			<div class="stat-card">
				<div class="stat-value font-mono">{$quantSummary.n_conditions}</div>
				<div class="stat-label font-ui">Conditions</div>
			</div>
			<div class="stat-card">
				<div class="stat-value font-mono">{$quantSummary.n_image_sets}</div>
				<div class="stat-label font-ui">Image Sets</div>
			</div>
			<div class="stat-card">
				<div class="stat-value font-mono">{totalRows.toLocaleString()}</div>
				<div class="stat-label font-ui">Data Rows</div>
			</div>
		</div>
	{/if}

	<!-- Export Bar -->
	<div class="export-bar">
		<span class="export-label font-ui">Export:</span>
		<button class="export-btn font-ui" onclick={() => $sessionId && exportCsv($sessionId)}>
			<FileText size={14} />
			CSV
		</button>
		<button class="export-btn font-ui" onclick={() => $sessionId && exportExcel($sessionId)}>
			<FileSpreadsheet size={14} />
			Excel
		</button>

		{#if hasFlaggedRows}
			<label class="toggle-flagged font-ui">
				<input type="checkbox" bind:checked={showFlagged} />
				Show flagged rows
			</label>
		{/if}

		{#if totalRows > 0}
			<button class="export-btn refresh-btn font-ui" onclick={() => { loadPage(0); loadSummary(); loadQCSummary(); }}>
				Refresh
			</button>
		{/if}
	</div>

	<!-- Tab Navigation -->
	<div class="tab-bar">
		{#each tabs as tab}
			<button
				class="tab-btn font-ui"
				class:active={activeTab === tab.id}
				onclick={() => activeTab = tab.id}
			>
				{tab.label}
			</button>
		{/each}
	</div>

	<!-- Tab Content -->
	<section class="results-content">
		{#if activeTab === 'table'}
			<div class="table-container">
				{#if rows.length > 0}
					<table class="results-table">
						<thead>
							<tr>
								{#each columns as col}
									<th class="font-ui" class:flag-col={isFlagColumn(col)}>{col}</th>
								{/each}
							</tr>
						</thead>
						<tbody>
							{#each rows as row}
								{#if showFlagged || !isFlaggedRow(row)}
									<tr class:flagged-row={isFlaggedRow(row)}>
										{#each columns as col}
											<td
												class="font-mono"
												class:flag-cell={isFlagColumn(col) && row[col] === true}
											>
												{formatValue(row[col])}
											</td>
										{/each}
									</tr>
								{/if}
							{/each}
						</tbody>
					</table>

					<!-- Pagination -->
					<div class="pagination">
						<button
							class="page-btn font-ui"
							disabled={currentPage <= 0 || loading}
							onclick={() => loadPage(currentPage - 1)}
						>
							Previous
						</button>
						<span class="page-info font-mono">
							Page {currentPage + 1} of {totalPages}
						</span>
						<button
							class="page-btn font-ui"
							disabled={currentPage >= totalPages - 1 || loading}
							onclick={() => loadPage(currentPage + 1)}
						>
							Next
						</button>
					</div>
				{:else}
					<div class="placeholder font-ui">
						<BarChart3 size={48} strokeWidth={1} />
						<p>Run quantification to see results</p>
					</div>
				{/if}
			</div>
		{:else if activeTab === 'boxplot'}
			<div class="chart-container" id="plotly-boxplot">
				{#if rows.length === 0}
					<div class="placeholder font-ui">
						<BarChart3 size={48} strokeWidth={1} />
						<p>CTCF distribution by condition</p>
					</div>
				{/if}
			</div>
		{:else if activeTab === 'scatter'}
			<div class="chart-container" id="plotly-scatter">
				{#if rows.length === 0}
					<div class="placeholder font-ui">
						<BarChart3 size={48} strokeWidth={1} />
						<p>Area vs CTCF scatter plot</p>
					</div>
				{/if}
			</div>
		{:else if activeTab === 'histogram'}
			<div class="chart-container" id="plotly-histogram">
				{#if rows.length === 0}
					<div class="placeholder font-ui">
						<BarChart3 size={48} strokeWidth={1} />
						<p>CTCF value distribution</p>
					</div>
				{/if}
			</div>
		{:else if activeTab === 'spatial'}
			<div class="chart-container" id="plotly-spatial">
				{#if rows.length === 0}
					<div class="placeholder font-ui">
						<BarChart3 size={48} strokeWidth={1} />
						<p>Spatial map of cell centroids colored by CTCF</p>
					</div>
				{:else if !columns.includes('x_centroid')}
					<div class="placeholder font-ui">
						<AlertTriangle size={48} strokeWidth={1} />
						<p>Spatial data not available (QC filters may need to be enabled)</p>
					</div>
				{/if}
			</div>
		{:else if activeTab === 'qcsummary'}
			<div class="table-container">
				{#if summaryRows.length > 0}
					<table class="results-table">
						<thead>
							<tr>
								<th class="font-ui">Condition</th>
								<th class="font-ui">Marker</th>
								<th class="font-ui">N Cells</th>
								<th class="font-ui">N FOVs</th>
								<th class="font-ui">Mean (FOV Medians)</th>
								<th class="font-ui">SD</th>
								<th class="font-ui">CV%</th>
								<th class="font-ui">SEM</th>
							</tr>
						</thead>
						<tbody>
							{#each summaryRows as row}
								<tr>
									<td class="font-mono">{row['Condition']}</td>
									<td class="font-mono">{row['Marker']}</td>
									<td class="font-mono">{formatValue(row['N_cells'])}</td>
									<td class="font-mono">{formatValue(row['N_FOVs'])}</td>
									<td class="font-mono">{formatValue(row['mean_of_fov_medians'])}</td>
									<td class="font-mono">{formatValue(row['sd_of_fov_medians'])}</td>
									<td class="font-mono">{formatValue(row['cv_percent'])}</td>
									<td class="font-mono">{formatValue(row['sem'])}</td>
								</tr>
							{/each}
						</tbody>
					</table>
				{:else}
					<div class="placeholder font-ui">
						<BarChart3 size={48} strokeWidth={1} />
						<p>Hierarchical summary (Cells → FOVs → Conditions)</p>
						<p class="hint">Run quantification to generate QC summary</p>
					</div>
				{/if}
			</div>
		{:else if activeTab === 'perfov'}
			<div class="chart-container" id="plotly-perfov">
				{#if fovRows.length === 0}
					<div class="placeholder font-ui">
						<BarChart3 size={48} strokeWidth={1} />
						<p>Per-FOV median CTCF distribution</p>
					</div>
				{/if}
			</div>
		{/if}
	</section>
</div>

<style>
	.page-results {
		display: flex;
		flex-direction: column;
		gap: 16px;
	}

	.summary-row {
		display: grid;
		grid-template-columns: repeat(4, 1fr);
		gap: 16px;
	}

	:global(.dark) .summary-row {
		gap: 1px;
		background: var(--border);
		border-radius: var(--radius-lg);
		overflow: hidden;
	}

	:global(.dark) .summary-row .stat-card {
		border: none;
		border-radius: 0;
	}

	.stat-value {
		font-size: 28px;
		font-weight: 500;
		color: var(--accent);
		line-height: 1.2;
	}

	:global(.dark) .stat-value {
		color: var(--text);
		font-size: 26px;
	}

	.stat-label {
		font-size: 11px;
		color: var(--text-muted);
		text-transform: uppercase;
		letter-spacing: 0.04em;
		margin-top: 4px;
	}

	.export-bar {
		display: flex;
		align-items: center;
		gap: 8px;
	}

	.export-label {
		font-size: 12px;
		color: var(--text-muted);
		font-weight: 500;
	}

	.export-btn {
		display: inline-flex;
		align-items: center;
		gap: 4px;
		padding: 5px 12px;
		background: var(--bg-elevated);
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		color: var(--text-muted);
		font-size: 12px;
		font-weight: 500;
		cursor: pointer;
		transition: all var(--transition-fast);
	}

	.export-btn:hover {
		border-color: var(--accent);
		color: var(--accent);
	}

	.refresh-btn {
		margin-left: auto;
	}

	.toggle-flagged {
		display: flex;
		align-items: center;
		gap: 6px;
		font-size: 12px;
		color: var(--text-muted);
		cursor: pointer;
		margin-left: 12px;
	}

	.toggle-flagged input {
		accent-color: var(--accent);
	}

	.tab-bar {
		display: flex;
		gap: 0;
		border-bottom: 1px solid var(--border);
		overflow-x: auto;
	}

	.tab-btn {
		padding: 12px 20px;
		background: transparent;
		border: none;
		border-bottom: 3px solid transparent;
		color: var(--text-muted);
		font-size: 13px;
		font-weight: 400;
		cursor: pointer;
		transition: all 0.2s ease;
		white-space: nowrap;
	}

	.tab-btn:hover {
		color: var(--accent);
	}

	.tab-btn.active {
		color: var(--accent);
		font-weight: 700;
		border-bottom-color: var(--accent);
	}

	:global(.dark) .tab-btn.active {
		font-weight: 500;
	}

	.results-content {
		background: var(--bg-elevated);
		border: 1px solid var(--border);
		border-radius: var(--radius-lg);
		box-shadow: var(--shadow-card);
		transition: var(--transition-theme);
		min-height: 400px;
	}

	:global(.dark) .results-content {
		box-shadow: none;
	}

	.table-container {
		overflow-x: auto;
	}

	.results-table {
		width: 100%;
		border-collapse: collapse;
	}

	.results-table thead th {
		background: linear-gradient(180deg, var(--accent), #5a4a84);
		color: white;
		font-size: 11px;
		font-weight: 700;
		text-transform: uppercase;
		letter-spacing: 0.04em;
		padding: 12px 14px;
		text-align: left;
		position: sticky;
		top: 0;
		white-space: nowrap;
	}

	.results-table thead th.flag-col {
		background: linear-gradient(180deg, #b44, #844);
	}

	:global(.dark) .results-table thead th {
		background: var(--bg);
		color: var(--text-muted);
		font-weight: 500;
		border-bottom: 1px solid var(--border);
	}

	:global(.dark) .results-table thead th.flag-col {
		background: var(--bg);
		color: #f88;
	}

	.results-table tbody td {
		padding: 10px 14px;
		font-size: 12px;
		border-bottom: 1px solid var(--border);
		color: var(--text);
		white-space: nowrap;
	}

	.results-table tbody td.flag-cell {
		background: rgba(255, 60, 60, 0.15);
		color: #e44;
		font-weight: 600;
	}

	.results-table tbody tr.flagged-row td:first-child {
		border-left: 3px solid #e44;
	}

	.results-table tbody tr:nth-child(even) td {
		background: rgba(212, 165, 165, 0.06);
	}

	.results-table tbody tr.flagged-row:nth-child(even) td.flag-cell {
		background: rgba(255, 60, 60, 0.15);
	}

	:global(.dark) .results-table tbody tr:nth-child(even) td {
		background: transparent;
	}

	.results-table tbody tr:hover td {
		background: var(--accent-soft);
	}

	.pagination {
		display: flex;
		align-items: center;
		justify-content: center;
		gap: 16px;
		padding: 16px;
		border-top: 1px solid var(--border);
	}

	.page-btn {
		padding: 6px 16px;
		background: var(--bg);
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		color: var(--text);
		font-size: 12px;
		font-weight: 500;
		cursor: pointer;
		transition: all var(--transition-fast);
	}

	.page-btn:hover:not(:disabled) {
		border-color: var(--accent);
		color: var(--accent);
	}

	.page-btn:disabled {
		opacity: 0.4;
		cursor: not-allowed;
	}

	.page-info {
		font-size: 12px;
		color: var(--text-muted);
	}

	.chart-container {
		min-height: 400px;
		padding: 16px;
	}

	.placeholder {
		text-align: center;
		color: var(--text-faint);
		padding: 80px 0;
	}

	.placeholder p {
		margin-top: 12px;
		font-size: 13px;
	}

	.placeholder .hint {
		font-size: 11px;
		margin-top: 4px;
	}
</style>
