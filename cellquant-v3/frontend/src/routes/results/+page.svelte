<script lang="ts">
	import { BarChart3, Download, FileSpreadsheet, FileText, Image } from 'lucide-svelte';
	import { getResultsPage, getResultsSummary, exportCsv, exportExcel, exportRois } from '$api/client';
	import { sessionId } from '$stores/session';
	import {
		quantResults,
		quantSummary,
		currentPage,
		totalPages,
		totalRows
	} from '$stores/quantification';

	let activeTab = $state<'table' | 'boxplot' | 'scatter' | 'histogram'>('table');

	async function loadPage(page: number) {
		if (!$sessionId) return;
		const result = await getResultsPage($sessionId, page);
		$quantResults = result.rows;
		$currentPage = result.page;
		$totalPages = result.total_pages;
		$totalRows = result.total_rows;
	}

	async function loadSummary() {
		if (!$sessionId) return;
		$quantSummary = await getResultsSummary($sessionId);
	}

	const tabs = [
		{ id: 'table' as const, label: 'Data Table' },
		{ id: 'boxplot' as const, label: 'Box Plot' },
		{ id: 'scatter' as const, label: 'Scatter' },
		{ id: 'histogram' as const, label: 'Histogram' }
	];
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
				<div class="stat-value font-mono">{$quantSummary.conditions.length}</div>
				<div class="stat-label font-ui">Conditions</div>
			</div>
			<div class="stat-card">
				<div class="stat-value font-mono">{$totalRows.toLocaleString()}</div>
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
		<button class="export-btn font-ui" onclick={() => $sessionId && exportRois($sessionId)}>
			<Image size={14} />
			ROIs
		</button>
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
				{#if $quantResults.length > 0}
					<table class="results-table">
						<thead>
							<tr>
								<th class="font-ui">Cell ID</th>
								<th class="font-ui">Condition</th>
								<th class="font-ui">Image</th>
								<th class="font-ui">Area</th>
								<th class="font-ui">Mean Int.</th>
								<th class="font-ui">Int. Density</th>
								<th class="font-ui">CTCF</th>
							</tr>
						</thead>
						<tbody>
							{#each $quantResults as row}
								<tr>
									<td class="font-mono">{row.cell_id}</td>
									<td class="font-ui">{row.condition}</td>
									<td class="font-mono cell-image">{row.image}</td>
									<td class="font-mono">{row.area}</td>
									<td class="font-mono">{row.mean_intensity.toFixed(2)}</td>
									<td class="font-mono">{row.integrated_density.toFixed(1)}</td>
									<td class="font-mono ctcf-value">{row.ctcf.toFixed(1)}</td>
								</tr>
							{/each}
						</tbody>
					</table>

					<!-- Pagination -->
					<div class="pagination">
						<button
							class="page-btn font-ui"
							disabled={$currentPage <= 0}
							onclick={() => loadPage($currentPage - 1)}
						>
							Previous
						</button>
						<span class="page-info font-mono">
							Page {$currentPage + 1} of {$totalPages}
						</span>
						<button
							class="page-btn font-ui"
							disabled={$currentPage >= $totalPages - 1}
							onclick={() => loadPage($currentPage + 1)}
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
				<div class="placeholder font-ui">
					<BarChart3 size={48} strokeWidth={1} />
					<p>CTCF distribution by condition</p>
				</div>
			</div>
		{:else if activeTab === 'scatter'}
			<div class="chart-container" id="plotly-scatter">
				<div class="placeholder font-ui">
					<BarChart3 size={48} strokeWidth={1} />
					<p>Area vs CTCF scatter plot</p>
				</div>
			</div>
		{:else if activeTab === 'histogram'}
			<div class="chart-container" id="plotly-histogram">
				<div class="placeholder font-ui">
					<BarChart3 size={48} strokeWidth={1} />
					<p>CTCF value distribution</p>
				</div>
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
		grid-template-columns: repeat(3, 1fr);
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

	.cell-image {
		max-width: 150px;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
		font-size: 11px;
		color: var(--text-muted);
	}

	.ctcf-value {
		font-weight: 600;
		color: var(--accent);
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
		display: flex;
		align-items: center;
		justify-content: center;
		padding: 24px;
	}

	.placeholder {
		text-align: center;
		color: var(--text-faint);
	}

	.placeholder p {
		margin-top: 12px;
		font-size: 13px;
	}
</style>
