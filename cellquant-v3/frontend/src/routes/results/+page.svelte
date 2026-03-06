<script lang="ts">
	import { onMount } from 'svelte';
	import { BarChart3, FileSpreadsheet, FileText, AlertTriangle, Grid3x3 } from 'lucide-svelte';
	import { getResultsPage, getResultsSummary, getQCSummary, getChartData, exportCsv, exportExcel } from '$api/client';
	import { sessionId } from '$stores/session';
	import { resultsPage, quantSummary, qcSummary } from '$stores/quantification';
	import ChartCard from '$components/charts/ChartCard.svelte';
	import {
		COLOR_PALETTES, type PaletteId, getColor,
		getChartTheme, baseLayout, computeKDE, removeOutliersIQR,
		CLEAN_CONFIG, INTERACTIVE_CONFIG, VIRIDIS,
	} from '$components/charts/chart-theme';

	type TabId = 'table' | 'distribution' | 'scatter' | 'spatial' | 'qcsummary' | 'perfov';
	let activeTab = $state<TabId>('table');
	let loading = $state(false);
	let showFlagged = $state(true);
	let selectedCtcfCol = $state('');

	// Chart options
	type DistType = 'violin' | 'box' | 'density';
	let distType = $state<DistType>('violin');
	let palette = $state<PaletteId>('default');
	let showOutliers = $state(false);
	let showPoints = $state(false);
	let showGrid = $state(false);
	let logScale = $state(false);
	let excludeZeros = $state(false);

	// Table (paginated)
	let tableColumns = $derived($resultsPage?.columns ?? []);
	let tableRows = $derived($resultsPage?.data ?? []);
	let currentPage = $derived($resultsPage?.page ?? 0);
	let totalPages = $derived($resultsPage?.total_pages ?? 0);
	let totalRows = $derived($resultsPage?.total_rows ?? 0);

	// Chart data (all rows, lightweight columns)
	let chartData = $state<{ columns: string[]; data: Record<string, unknown>[]; total_rows: number } | null>(null);
	let chartColumns = $derived(chartData?.columns ?? []);
	let chartRows = $derived(chartData?.data ?? []);

	// CTCF column options (from chart data which has all rows)
	let ctcfColumns = $derived(chartColumns.filter(c => c.endsWith('_CTCF')));
	let activeCtcfCol = $derived(selectedCtcfCol && ctcfColumns.includes(selectedCtcfCol)
		? selectedCtcfCol
		: ctcfColumns[0] || 'CTCF');

	// Fallback: use table columns if chart data not loaded yet
	let ctcfColumnsFromTable = $derived(tableColumns.filter(c => c.endsWith('_CTCF')));
	let effectiveCtcfColumns = $derived(ctcfColumns.length > 0 ? ctcfColumns : ctcfColumnsFromTable);

	// Detect quality flag columns (table view)
	let flagColumns = $derived(tableColumns.filter(c =>
		c.startsWith('is_outlier_') || c === 'is_saturated' || c === 'is_dim' || c === 'low_confidence_background'
	));
	let hasFlaggedRows = $derived(tableRows.some(r =>
		flagColumns.some(c => r[c] === true)
	));

	const tabs: { id: TabId; label: string }[] = [
		{ id: 'table', label: 'Data Table' },
		{ id: 'distribution', label: 'Distribution' },
		{ id: 'scatter', label: 'Scatter' },
		{ id: 'spatial', label: 'Spatial Map' },
		{ id: 'qcsummary', label: 'QC Summary' },
		{ id: 'perfov', label: 'Per-FOV' },
	];

	// QC Summary data
	let summaryRows = $derived($qcSummary?.summary ?? []);
	let fovRows = $derived($qcSummary?.fov_data ?? []);

	// Top scrollbar sync
	let topScrollEl: HTMLDivElement | undefined = $state();
	let topScrollInner: HTMLDivElement | undefined = $state();
	let tableContainerEl: HTMLDivElement | undefined = $state();
	let tableEl: HTMLTableElement | undefined = $state();
	let syncing = false;

	$effect(() => {
		if (tableEl && topScrollInner && tableRows.length > 0) {
			requestAnimationFrame(() => {
				if (tableEl && topScrollInner) {
					topScrollInner.style.width = tableEl.scrollWidth + 'px';
				}
			});
		}
	});

	function syncScrollFromTop() {
		if (syncing || !topScrollEl || !tableContainerEl) return;
		syncing = true;
		tableContainerEl.scrollLeft = topScrollEl.scrollLeft;
		syncing = false;
	}

	function syncScrollFromTable() {
		if (syncing || !topScrollEl || !tableContainerEl) return;
		syncing = true;
		topScrollEl.scrollLeft = tableContainerEl.scrollLeft;
		syncing = false;
	}

	onMount(() => {
		if ($sessionId) {
			if (!$resultsPage) loadPage(0);
			if (!chartData) loadChartData();
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

	let chartLoading = $state(false);

	async function loadChartData() {
		if (!$sessionId) return;
		chartLoading = true;
		try { chartData = await getChartData($sessionId); } catch {}
		chartLoading = false;
	}

	async function loadSummary() {
		if (!$sessionId) return;
		try { $quantSummary = await getResultsSummary($sessionId); } catch {}
	}

	async function loadQCSummary() {
		if (!$sessionId) return;
		try { $qcSummary = await getQCSummary($sessionId); } catch {}
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

	// ── Chart rendering ──────────────────────────────────

	async function renderChart(tab: string) {
		if (chartRows.length === 0) return;

		const Plotly = await import('plotly.js-dist-min');
		const theme = getChartTheme();
		const layout = baseLayout(theme);
		layout.xaxis = { ...(layout.xaxis as object), showgrid: showGrid };
		layout.yaxis = { ...(layout.yaxis as object), showgrid: showGrid };

		const allRows = chartRows;
		const allCols = chartColumns;
		const ctcfCol = activeCtcfCol;
		const condCol = allCols.includes('Condition') ? 'Condition' : allCols[0];
		const conditions = [...new Set(allRows.map((r) => String(r[condCol])))];

		if (tab === 'distribution') {
			renderDistribution(Plotly, layout, theme, ctcfCol, condCol, conditions);
		} else if (tab === 'scatter') {
			renderScatter(Plotly, layout, theme, ctcfCol, condCol, conditions);
		} else if (tab === 'spatial') {
			renderSpatial(Plotly, layout, theme, ctcfCol);
		} else if (tab === 'perfov') {
			renderPerFov(Plotly, layout, theme);
		}
	}

	function renderDistribution(Plotly: any, layout: any, theme: any, ctcfCol: string, condCol: string, conditions: string[]) {
		const traces: any[] = [];
		const yLabel = logScale
			? (distType === 'density' ? 'Density' : `log₁₀(${ctcfCol} + 1)`)
			: (distType === 'density' ? 'Density' : ctcfCol);

		conditions.forEach((cond, i) => {
			let vals = chartRows.filter((r) => String(r[condCol]) === cond).map((r) => Number(r[ctcfCol]) || 0);
			if (!showOutliers) {
				vals = removeOutliersIQR(vals).filtered;
			}
			if (excludeZeros) {
				vals = vals.filter(v => v > 0);
			}
			// Log transform: log10(x + 1) — safe for zero values
			if (logScale && distType !== 'density') {
				vals = vals.map(v => Math.log10(Math.max(0, v) + 1));
			}
			const color = getColor(i, palette);

			if (distType === 'violin') {
				traces.push({
					type: 'violin',
					name: cond,
					x: vals.map(() => cond),
					y: vals,
					width: 0.6,
					scalegroup: cond,
					spanmode: 'soft',
					box: { visible: true },
					meanline: { visible: true },
					points: showPoints ? 'all' : false,
					marker: { color, size: 3, opacity: 0.6 },
					line: { color, width: 1.5 },
					fillcolor: color + '33',
					hovertemplate: `<b>${cond}</b><br>%{y:.1f}<extra></extra>`,
				});
			} else if (distType === 'box') {
				traces.push({
					type: 'box',
					name: cond,
					x: vals.map(() => cond),
					y: vals,
					width: 0.5,
					boxpoints: showPoints ? 'all' : 'outliers',
					jitter: 0.4,
					pointpos: 0,
					marker: { color, size: 3, opacity: 0.6 },
					line: { color, width: 1.5 },
					fillcolor: color + '33',
					hovertemplate: `<b>${cond}</b><br>%{y:.1f}<extra></extra>`,
				});
			} else if (distType === 'density') {
				// For density, apply log to raw values before KDE
				const kdeVals = logScale ? vals.map(v => Math.log10(Math.max(0, v) + 1)) : vals;
				const kde = computeKDE(kdeVals);
				if (kde.x.length > 0) {
					traces.push({
						type: 'scatter',
						mode: 'lines',
						name: cond,
						x: kde.x,
						y: kde.y,
						fill: 'tozeroy',
						line: { color, width: 2 },
						fillcolor: color + '22',
						hovertemplate: `<b>${cond}</b><br>${logScale ? 'log₁₀(' + ctcfCol + '+1)' : ctcfCol}: %{x:.1f}<br>Density: %{y:.4f}<extra></extra>`,
					});
				}
			}
		});

		const distLayout = {
			...layout,
			margin: { l: 60, r: 20, t: 20, b: 80 },
			yaxis: {
				...(layout.yaxis as object),
				title: { text: yLabel, font: { size: 12, color: theme.textMuted } },
				rangemode: (distType !== 'density' && !logScale) ? 'nonnegative' as const : undefined,
			},
			xaxis: {
				...(layout.xaxis as object),
				title: distType === 'density'
					? { text: logScale ? `log₁₀(${ctcfCol} + 1)` : ctcfCol, font: { size: 12, color: theme.textMuted } }
					: '',
			},
			violingap: 0.35,
			violingroupgap: 0.15,
			boxgap: 0.3,
			boxgroupgap: 0.15,
		};

		Plotly.newPlot('plotly-distribution', traces, distLayout, CLEAN_CONFIG);
	}

	function renderScatter(Plotly: any, layout: any, theme: any, ctcfCol: string, condCol: string, conditions: string[]) {
		const traces = conditions.map((cond, i) => {
			const condRows = chartRows.filter((r) => String(r[condCol]) === cond);
			const color = getColor(i, palette);
			return {
				x: condRows.map((r) => Number(r['Area']) || 0),
				y: condRows.map((r) => Number(r[ctcfCol]) || 0),
				mode: 'markers',
				type: 'scattergl',
				name: cond,
				marker: {
					color: color + '80',
					size: 4,
					line: { color, width: 0.5 },
				},
				hovertemplate: `<b>${cond}</b><br>Area: %{x:.0f} px<br>${ctcfCol}: %{y:.1f}<extra></extra>`,
			};
		});

		Plotly.newPlot('plotly-scatter', traces, {
			...layout,
			margin: { l: 60, r: 20, t: 20, b: 80 },
			xaxis: { ...(layout.xaxis as object), title: { text: 'Area (px)', font: { size: 12, color: theme.textMuted } } },
			yaxis: { ...(layout.yaxis as object), title: { text: ctcfCol, font: { size: 12, color: theme.textMuted } }, rangemode: 'nonnegative' },
		}, INTERACTIVE_CONFIG);
	}

	function renderSpatial(Plotly: any, layout: any, theme: any, ctcfCol: string) {
		const hasXY = chartColumns.includes('x_centroid') && chartColumns.includes('y_centroid');
		if (!hasXY) return;

		const ctcfValues = chartRows.map(r => Number(r[ctcfCol]) || 0);
		const trace = {
			x: chartRows.map((r) => Number(r['x_centroid']) || 0),
			y: chartRows.map((r) => Number(r['y_centroid']) || 0),
			mode: 'markers',
			type: 'scattergl',
			marker: {
				color: ctcfValues,
				colorscale: VIRIDIS,
				size: 4,
				opacity: 0.85,
				colorbar: {
					title: { text: ctcfCol, font: { size: 11, color: theme.textMuted } },
					thickness: 14,
					len: 0.6,
					tickfont: { size: 10, color: theme.textMuted },
					outlinewidth: 0,
				},
			},
			text: chartRows.map((r) => `Cell ${r['CellID']}`),
			hovertemplate: `<b>%{text}</b><br>x: %{x:.0f}, y: %{y:.0f}<br>${ctcfCol}: %{marker.color:.1f}<extra></extra>`,
			showlegend: false,
		};

		Plotly.newPlot('plotly-spatial', [trace], {
			...layout,
			margin: { l: 60, r: 20, t: 20, b: 60 },
			xaxis: { ...(layout.xaxis as object), title: { text: 'X (px)', font: { size: 12, color: theme.textMuted } }, scaleanchor: 'y' },
			yaxis: { ...(layout.yaxis as object), title: { text: 'Y (px)', font: { size: 12, color: theme.textMuted } }, autorange: 'reversed' },
			showlegend: false,
		}, INTERACTIVE_CONFIG);
	}

	function renderPerFov(Plotly: any, layout: any, theme: any) {
		if (fovRows.length === 0) return;

		const medCols = Object.keys(fovRows[0] || {}).filter(k => k.endsWith('_median_CTCF'));
		const medCol = medCols[0];
		if (!medCol) return;

		const fovConditions = [...new Set(fovRows.map(r => String(r['Condition'])))];
		const traces = fovConditions.map((cond, i) => {
			const vals = fovRows.filter(r => String(r['Condition']) === cond).map(r => Number(r[medCol]) || 0);
			const color = getColor(i, palette);
			return {
				type: 'box',
				name: cond,
				x: vals.map(() => cond),
				y: vals,
				width: 0.5,
				marker: { color, size: 6, opacity: 0.7 },
				line: { color, width: 1.5 },
				fillcolor: color + '30',
				boxpoints: 'all',
				jitter: 0.4,
				pointpos: 0,
				hovertemplate: `<b>${cond}</b><br>Median CTCF: %{y:.1f}<extra></extra>`,
			};
		});

		Plotly.newPlot('plotly-perfov', traces, {
			...layout,
			margin: { l: 60, r: 20, t: 20, b: 80 },
			yaxis: { ...(layout.yaxis as object), title: { text: medCol.replace('_median_CTCF', ' Median CTCF'), font: { size: 12, color: theme.textMuted } } },
			boxgap: 0.3,
			boxgroupgap: 0.15,
		}, CLEAN_CONFIG);
	}

	// Re-render when tab, data, options, or marker changes
	$effect(() => {
		const _tab = activeTab;
		const _col = activeCtcfCol;
		const _len = chartRows.length;
		const _dist = distType;
		const _pal = palette;
		const _out = showOutliers;
		const _pts = showPoints;
		const _grid = showGrid;
		const _log = logScale;
		const _zeros = excludeZeros;
		if (_tab !== 'table' && _tab !== 'qcsummary' && _len > 0) {
			setTimeout(() => renderChart(_tab), 50);
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

	<!-- Toolbar -->
	<div class="toolbar">
		<div class="toolbar-left">
			<span class="toolbar-label font-ui">Export:</span>
			<button class="toolbar-btn font-ui" onclick={() => $sessionId && exportCsv($sessionId)}>
				<FileText size={14} /> CSV
			</button>
			<button class="toolbar-btn font-ui" onclick={() => $sessionId && exportExcel($sessionId)}>
				<FileSpreadsheet size={14} /> Excel
			</button>

			{#if effectiveCtcfColumns.length > 1}
				<span class="toolbar-sep"></span>
				<select class="toolbar-select font-ui" bind:value={selectedCtcfCol}>
					{#each effectiveCtcfColumns as col}
						<option value={col}>{col.replace('_CTCF', '')}</option>
					{/each}
				</select>
			{/if}

			{#if hasFlaggedRows && activeTab === 'table'}
				<span class="toolbar-sep"></span>
				<label class="toolbar-check font-ui">
					<input type="checkbox" bind:checked={showFlagged} />
					Show flagged
				</label>
			{/if}
		</div>

		<div class="toolbar-right">
			{#if activeTab !== 'table' && activeTab !== 'qcsummary'}
				<select class="toolbar-select palette-select font-ui" bind:value={palette}>
					{#each Object.entries(COLOR_PALETTES) as [id, p]}
						<option value={id}>
							{p.label}
						</option>
					{/each}
				</select>

				<label class="toolbar-check font-ui" title="Show grid lines">
					<input type="checkbox" bind:checked={showGrid} />
					<Grid3x3 size={13} />
				</label>
			{/if}

			{#if activeTab === 'distribution'}
				<span class="toolbar-sep"></span>
				<div class="chart-type-toggle">
					<button class="toggle-btn font-ui" class:active={distType === 'violin'} onclick={() => distType = 'violin'}>Violin</button>
					<button class="toggle-btn font-ui" class:active={distType === 'box'} onclick={() => distType = 'box'}>Box</button>
					<button class="toggle-btn font-ui" class:active={distType === 'density'} onclick={() => distType = 'density'}>Density</button>
				</div>
				<span class="toolbar-sep"></span>
				<label class="toolbar-check font-ui">
					<input type="checkbox" bind:checked={showOutliers} />
					Outliers
				</label>
				{#if distType !== 'density'}
					<label class="toolbar-check font-ui">
						<input type="checkbox" bind:checked={showPoints} />
						Points
					</label>
				{/if}
				<label class="toolbar-check font-ui" title="Exclude cells with CTCF = 0">
					<input type="checkbox" bind:checked={excludeZeros} />
					No zeros
				</label>
				<label class="toolbar-check font-ui" title="Log₁₀(x+1) transform">
					<input type="checkbox" bind:checked={logScale} />
					Log
				</label>
			{/if}

			{#if totalRows > 0}
				<button class="toolbar-btn font-ui" onclick={() => { loadPage(0); loadChartData(); loadSummary(); loadQCSummary(); }}>
					Refresh
				</button>
			{/if}
		</div>
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
			<div class="table-scroll-top" bind:this={topScrollEl} onscroll={syncScrollFromTop}>
				<div class="table-scroll-top-inner" bind:this={topScrollInner}></div>
			</div>
			<div class="table-container" bind:this={tableContainerEl} onscroll={syncScrollFromTable}>
				{#if tableRows.length > 0}
					<table class="results-table" bind:this={tableEl}>
						<thead>
							<tr>
								{#each tableColumns as col}
									<th class="font-ui" class:flag-col={isFlagColumn(col)}>{col}</th>
								{/each}
							</tr>
						</thead>
						<tbody>
							{#each tableRows as row}
								{#if showFlagged || !isFlaggedRow(row)}
									<tr class:flagged-row={isFlaggedRow(row)}>
										{#each tableColumns as col}
											<td class="font-mono" class:flag-cell={isFlagColumn(col) && row[col] === true}>
												{formatValue(row[col])}
											</td>
										{/each}
									</tr>
								{/if}
							{/each}
						</tbody>
					</table>
					<div class="pagination">
						<button class="page-btn font-ui" disabled={currentPage <= 0 || loading} onclick={() => loadPage(currentPage - 1)}>Previous</button>
						<span class="page-info font-mono">Page {currentPage + 1} of {totalPages}</span>
						<button class="page-btn font-ui" disabled={currentPage >= totalPages - 1 || loading} onclick={() => loadPage(currentPage + 1)}>Next</button>
					</div>
				{:else}
					<div class="placeholder font-ui">
						<BarChart3 size={48} strokeWidth={1} />
						<p>Run quantification to see results</p>
					</div>
				{/if}
			</div>

		{:else if activeTab === 'distribution'}
			<ChartCard
				title="{activeCtcfCol} by Condition"
				subtitle="{distType === 'violin' ? 'Violin' : distType === 'box' ? 'Box' : 'Density'} plot{logScale ? ' · log₁₀(x+1)' : ''}{excludeZeros ? ' · zeros excluded' : ''}{!showOutliers ? ' · outliers removed (IQR ×1.5)' : ''}"
				loading={chartLoading}
			empty={chartRows.length === 0 && !chartLoading}
				emptyMessage="Run quantification to see distributions"
			>
				{#snippet children()}
					<div class="chart-plot" id="plotly-distribution"></div>
				{/snippet}
			</ChartCard>

		{:else if activeTab === 'scatter'}
			<ChartCard
				title="Area vs {activeCtcfCol}"
				subtitle="Cell area vs corrected fluorescence"
				loading={chartLoading}
			empty={chartRows.length === 0 && !chartLoading}
				emptyMessage="Run quantification to see scatter plot"
			>
				{#snippet children()}
					<div class="chart-plot" id="plotly-scatter"></div>
				{/snippet}
			</ChartCard>

		{:else if activeTab === 'spatial'}
			<ChartCard
				title="Spatial {activeCtcfCol} Map"
				subtitle="Cell centroids colored by fluorescence intensity"
				loading={chartLoading}
			empty={(chartRows.length === 0 && !chartLoading) || !chartColumns.includes('x_centroid')}
				emptyMessage={!chartColumns.includes('x_centroid') ? 'Spatial data not available (centroid columns missing)' : 'Run quantification to see spatial map'}
			>
				{#snippet children()}
					<div class="chart-plot" id="plotly-spatial"></div>
				{/snippet}
			</ChartCard>

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
						<p>Hierarchical summary (Cells &rarr; FOVs &rarr; Conditions)</p>
						<p class="hint">Run quantification to generate QC summary</p>
					</div>
				{/if}
			</div>

		{:else if activeTab === 'perfov'}
			<ChartCard
				title="Per-FOV Median CTCF"
				subtitle="Each point is one field of view"
				empty={fovRows.length === 0}
				emptyMessage="Run quantification to see per-FOV data"
			>
				{#snippet children()}
					<div class="chart-plot" id="plotly-perfov"></div>
				{/snippet}
			</ChartCard>
		{/if}
	</section>
</div>

<style>
	.page-results {
		display: flex;
		flex-direction: column;
		gap: 12px;
	}

	/* ── Summary cards ──────────────────────────────── */

	.summary-row {
		display: grid;
		grid-template-columns: repeat(4, 1fr);
		gap: 12px;
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

	/* ── Toolbar ────────────────────────────────────── */

	.toolbar {
		display: flex;
		align-items: center;
		justify-content: space-between;
		flex-wrap: wrap;
		gap: 8px;
	}

	.toolbar-left,
	.toolbar-right {
		display: flex;
		align-items: center;
		gap: 6px;
		flex-wrap: wrap;
	}

	.toolbar-label {
		font-size: 12px;
		color: var(--text-muted);
		font-weight: 500;
	}

	.toolbar-btn {
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
		transition: all 0.15s ease;
	}

	.toolbar-btn:hover {
		border-color: var(--accent);
		color: var(--accent);
	}

	.toolbar-select {
		padding: 4px 8px;
		background: var(--bg);
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		color: var(--text);
		font-size: 12px;
		cursor: pointer;
	}

	.toolbar-select:focus {
		border-color: var(--accent);
		outline: none;
	}

	.palette-select {
		color: var(--accent);
		font-weight: 600;
	}

	.toolbar-check {
		display: flex;
		align-items: center;
		gap: 4px;
		font-size: 11px;
		color: var(--text-muted);
		cursor: pointer;
		white-space: nowrap;
	}

	.toolbar-check input {
		accent-color: var(--accent);
		width: 13px;
		height: 13px;
	}

	.toolbar-sep {
		width: 1px;
		height: 18px;
		background: var(--border);
		margin: 0 2px;
	}

	/* ── Chart type toggle ──────────────────────────── */

	.chart-type-toggle {
		display: flex;
		gap: 1px;
		background: var(--border);
		border-radius: var(--radius-sm);
		overflow: hidden;
	}

	.toggle-btn {
		padding: 4px 12px;
		background: var(--bg-elevated);
		border: none;
		font-size: 11px;
		font-weight: 500;
		color: var(--text-muted);
		cursor: pointer;
		transition: all 0.15s ease;
	}

	.toggle-btn:hover {
		color: var(--text);
	}

	.toggle-btn.active {
		background: var(--accent);
		color: white;
	}

	:global(.dark) .toggle-btn.active {
		color: #000;
	}

	/* ── Tabs ───────────────────────────────────────── */

	.tab-bar {
		display: flex;
		gap: 0;
		border-bottom: 1px solid var(--border);
		overflow-x: auto;
	}

	.tab-btn {
		padding: 10px 18px;
		background: transparent;
		border: none;
		border-bottom: 2px solid transparent;
		color: var(--text-muted);
		font-size: 13px;
		font-weight: 400;
		cursor: pointer;
		transition: all 0.15s ease;
		white-space: nowrap;
	}

	.tab-btn:hover { color: var(--text); }

	.tab-btn.active {
		color: var(--accent);
		font-weight: 600;
		border-bottom-color: var(--accent);
	}

	/* ── Content area ───────────────────────────────── */

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

	/* ── Chart plot area ────────────────────────────── */

	.chart-plot {
		width: 100%;
		min-height: 440px;
	}

	/* Hide plotly modebar by default, show on hover */
	.results-content :global(.modebar) {
		opacity: 0;
		transition: opacity 0.2s ease;
	}

	.results-content:hover :global(.modebar) {
		opacity: 1;
	}

	.results-content :global(.modebar-btn) {
		font-size: 14px !important;
	}

	/* ── Table ──────────────────────────────────────── */

	.table-scroll-top {
		overflow-x: auto;
		overflow-y: hidden;
	}

	.table-scroll-top-inner {
		height: 1px;
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
		padding: 10px 12px;
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
		padding: 8px 12px;
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
		background: rgba(0, 0, 0, 0.02);
	}

	:global(.dark) .results-table tbody tr:nth-child(even) td {
		background: rgba(255, 255, 255, 0.02);
	}

	.results-table tbody tr:hover td {
		background: var(--accent-soft);
	}

	/* ── Pagination ─────────────────────────────────── */

	.pagination {
		display: flex;
		align-items: center;
		justify-content: center;
		gap: 16px;
		padding: 14px;
		border-top: 1px solid var(--border);
	}

	.page-btn {
		padding: 5px 14px;
		background: var(--bg);
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		color: var(--text);
		font-size: 12px;
		font-weight: 500;
		cursor: pointer;
		transition: all 0.15s ease;
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

	/* ── Placeholder ────────────────────────────────── */

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
