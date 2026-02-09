<script lang="ts">
	import { onMount } from 'svelte';
	import { BarChart3, FileSpreadsheet, FileText } from 'lucide-svelte';
	import { getResultsPage, getResultsSummary, exportCsv, exportExcel } from '$api/client';
	import { sessionId } from '$stores/session';
	import { resultsPage, quantSummary } from '$stores/quantification';

	let activeTab = $state<'table' | 'boxplot' | 'scatter' | 'histogram'>('table');
	let loading = $state(false);

	let columns = $derived($resultsPage?.columns ?? []);
	let rows = $derived($resultsPage?.data ?? []);
	let currentPage = $derived($resultsPage?.page ?? 0);
	let totalPages = $derived($resultsPage?.total_pages ?? 0);
	let totalRows = $derived($resultsPage?.total_rows ?? 0);

	const tabs = [
		{ id: 'table' as const, label: 'Data Table' },
		{ id: 'boxplot' as const, label: 'Box Plot' },
		{ id: 'scatter' as const, label: 'Scatter' },
		{ id: 'histogram' as const, label: 'Histogram' }
	];

	onMount(() => {
		if ($sessionId && !$resultsPage) {
			loadPage(0);
			loadSummary();
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

	function formatValue(val: unknown): string {
		if (val == null) return '--';
		if (typeof val === 'number') {
			if (Number.isInteger(val)) return val.toLocaleString();
			return val.toFixed(2);
		}
		return String(val);
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
		}
	}

	$effect(() => {
		if (activeTab !== 'table' && rows.length > 0) {
			// Render chart after DOM update
			setTimeout(() => renderChart(activeTab), 50);
		}
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
		{#if totalRows > 0}
			<button class="export-btn refresh-btn font-ui" onclick={() => { loadPage(0); loadSummary(); }}>
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
									<th class="font-ui">{col}</th>
								{/each}
							</tr>
						</thead>
						<tbody>
							{#each rows as row}
								<tr>
									{#each columns as col}
										<td class="font-mono">{formatValue(row[col])}</td>
									{/each}
								</tr>
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

	.tab-bar {
		display: flex;
		gap: 0;
		border-bottom: 1px solid var(--border);
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

	:global(.dark) .results-table thead th {
		background: var(--bg);
		color: var(--text-muted);
		font-weight: 500;
		border-bottom: 1px solid var(--border);
	}

	.results-table tbody td {
		padding: 10px 14px;
		font-size: 12px;
		border-bottom: 1px solid var(--border);
		color: var(--text);
		white-space: nowrap;
	}

	.results-table tbody tr:nth-child(even) td {
		background: rgba(212, 165, 165, 0.06);
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
</style>
